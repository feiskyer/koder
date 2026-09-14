"""OAuth 2.0 authentication flow for remote MCP servers.

Implements RFC 6749 Authorization Code Grant with PKCE (RFC 7636),
dynamic client registration (RFC 7591), and OAuth discovery via
``/.well-known/oauth-protected-resource`` and
``/.well-known/oauth-authorization-server``.

Token state is persisted under an endpoint-bound, hashed cache identity below
``~/.koder/mcp-auth/v4/``.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import hashlib
import ipaddress
import json
import logging
import os
import secrets
import socket
import stat
import time
import webbrowser
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, AsyncGenerator, Callable, Dict, Generator
from urllib.parse import urlencode, urlsplit

import httpx

from koder_agent.utils.async_tasks import run_sync_owned

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_AUTH_DIR_NAME = "mcp-auth"
_CACHE_VERSION = 4
_SERVER_BINDING_KEY = "_oauth_server_binding"
_TOKEN_FILE_NAME = "tokens.json"
_TOKEN_LOCK_NAME = "tokens.lock"
_REFRESH_LOCK_NAME = "refresh.lock"
_TEMP_FILE_PREFIX = f".{_TOKEN_FILE_NAME}."
_TEMP_FILE_SUFFIX = ".tmp"
_PROTECTED_PAYLOAD_MAGIC = b"KODER-MCP-OAUTH-DPAPI-v1\0"
_LOCK_OPEN_ATTEMPTS = 32
_REFRESH_LOCK_POLL_INTERVAL = 0.05
_CALLBACK_REQUEST_TIMEOUT_SECONDS = 5.0

try:  # pragma: no branch - exactly one platform branch is active
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # pragma: no branch - exactly one platform branch is active
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

_HAS_POSIX_DIR_FD = (
    bool(getattr(os, "O_DIRECTORY", 0))
    and os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
)


@dataclass(frozen=True)
class _DataProtector:
    """User-scoped at-rest protection for non-POSIX token payloads."""

    protect: Callable[[bytes, bytes], bytes]
    unprotect: Callable[[bytes, bytes], bytes]


def _windows_dpapi_transform(
    data: bytes,
    entropy: bytes,
    *,
    decrypt: bool,
) -> bytes:
    """Protect or unprotect bytes with the current Windows user's DPAPI key."""
    if os.name != "nt":  # pragma: no cover - guarded by capability selection
        raise RuntimeError("Windows DPAPI is unavailable on this platform")

    import ctypes
    from ctypes import wintypes

    class _DataBlob(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    def make_blob(value: bytes) -> tuple[_DataBlob, Any]:
        buffer = ctypes.create_string_buffer(value)
        blob = _DataBlob(
            len(value),
            ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)),
        )
        return blob, buffer

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    blob_pointer = ctypes.POINTER(_DataBlob)
    crypt32.CryptProtectData.argtypes = [
        blob_pointer,
        wintypes.LPCWSTR,
        blob_pointer,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        blob_pointer,
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        blob_pointer,
        ctypes.POINTER(wintypes.LPWSTR),
        blob_pointer,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        blob_pointer,
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
    kernel32.LocalFree.restype = wintypes.HLOCAL
    input_blob, input_buffer = make_blob(data)
    entropy_blob, entropy_buffer = make_blob(entropy)
    output_blob = _DataBlob()
    flags = 0x1  # CRYPTPROTECT_UI_FORBIDDEN
    if decrypt:
        succeeded = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            ctypes.byref(entropy_blob),
            None,
            None,
            flags,
            ctypes.byref(output_blob),
        )
    else:
        succeeded = crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            "Koder MCP OAuth credentials",
            ctypes.byref(entropy_blob),
            None,
            None,
            flags,
            ctypes.byref(output_blob),
        )
    # Keep the input buffers alive through the native call.
    del input_buffer, entropy_buffer
    if not succeeded:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(output_blob.pbData, wintypes.HLOCAL))


def _windows_dpapi_protector() -> _DataProtector:
    return _DataProtector(
        protect=lambda data, entropy: _windows_dpapi_transform(
            data,
            entropy,
            decrypt=False,
        ),
        unprotect=lambda data, entropy: _windows_dpapi_transform(
            data,
            entropy,
            decrypt=True,
        ),
    )


@dataclass(frozen=True)
class _SecureStorageCapabilities:
    """Filesystem controls available to the OAuth token cache.

    POSIX mode enforcement is a security boundary, not a best-effort option:
    POSIX hosts must provide descriptor chmod and ``O_NOFOLLOW``. Windows token
    payloads are encrypted with user-scoped DPAPI rather than trusting inherited
    profile ACLs. Other non-POSIX hosts fail closed unless an equivalent
    protector is explicitly supplied.
    """

    enforce_posix_modes: bool
    nofollow_flag: int
    has_fchmod: bool
    data_protector: _DataProtector | None = None


def _secure_storage_capabilities() -> _SecureStorageCapabilities:
    """Return platform capabilities separately from storage policy."""
    is_posix = os.name == "posix"
    return _SecureStorageCapabilities(
        enforce_posix_modes=is_posix,
        nofollow_flag=getattr(os, "O_NOFOLLOW", 0) if is_posix else 0,
        has_fchmod=hasattr(os, "fchmod") if is_posix else False,
        data_protector=_windows_dpapi_protector() if os.name == "nt" else None,
    )


def _validate_secure_storage_capabilities(capabilities: _SecureStorageCapabilities) -> None:
    """Refuse to silently weaken either platform's secure-storage contract."""
    if capabilities.enforce_posix_modes:
        if (
            not capabilities.nofollow_flag
            or not capabilities.has_fchmod
            or fcntl is None
            or not _HAS_POSIX_DIR_FD
        ):
            raise RuntimeError("POSIX MCP OAuth storage security controls are unavailable")
    elif capabilities.data_protector is None:
        raise RuntimeError("Non-POSIX MCP OAuth at-rest protection is unavailable")


@dataclass
class MCPOAuthConfig:
    """OAuth configuration for an MCP server."""

    client_id: str | None = None
    client_secret: str | None = None
    callback_port: int | None = None
    auth_server_metadata_url: str | None = None
    scopes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> MCPOAuthConfig | None:
        """Build from the ``oauth`` dict stored in ``MCPServerConfig``.

        Returns ``None`` when *data* is ``None`` or empty.
        """
        if not data:
            return None
        return cls(
            client_id=data.get("clientId") or data.get("client_id"),
            client_secret=data.get("clientSecret") or data.get("client_secret"),
            callback_port=data.get("callbackPort") or data.get("callback_port"),
            auth_server_metadata_url=(
                data.get("authServerMetadataUrl") or data.get("auth_server_metadata_url")
            ),
            scopes=data.get("scopes") or [],
        )


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def _secret_fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _NormalizedCacheURL:
    """Digest of HTTPX's fragment-free URL representation."""

    transport_fingerprint: str


def _normalize_cache_url(url: str) -> _NormalizedCacheURL:
    """Return a secret-safe identity using the actual HTTP transport parser.

    HTTPX owns the URL semantics used by MCP requests, including IDNA, default
    ports, literal dot segments, and encoded paths. Reusing its serialized URL
    avoids security-sensitive normalization drift. Fragments are removed
    because they are not sent in HTTP requests; all remaining components,
    including userinfo and query text, are identity-sensitive but digest-only.
    """
    try:
        transport_url = httpx.URL(url)
    except Exception:
        raise ValueError("MCP OAuth cache identity requires a valid absolute URL") from None
    if not transport_url.is_absolute_url or transport_url.raw_host == b"":
        raise ValueError("MCP OAuth cache identity requires a valid absolute URL")
    fragment_free = transport_url.copy_with(fragment=None)
    return _NormalizedCacheURL(
        transport_fingerprint=hashlib.sha256(str(fragment_free).encode("utf-8")).hexdigest()
    )


def _is_loopback_host(host: str) -> bool:
    normalized = host.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _validate_oauth_url(value: Any, label: str) -> httpx.URL:
    """Validate an OAuth issuer/endpoint before it can carry credentials."""
    error = f"OAuth metadata {label} must be an absolute HTTP(S) URL without a fragment"
    if not isinstance(value, str) or "#" in value:
        raise RuntimeError(error)
    try:
        url = httpx.URL(value)
    except Exception:
        raise RuntimeError(error) from None
    if not url.is_absolute_url or url.raw_host == b"" or url.raw_scheme not in {b"http", b"https"}:
        raise RuntimeError(error)
    if url.raw_scheme == b"http" and not _is_loopback_host(url.host):
        raise RuntimeError(f"OAuth metadata {label} must use HTTPS unless it is loopback")
    return url


@dataclass(frozen=True)
class OAuthCacheIdentity:
    """Stable, endpoint-bound identity for MCP OAuth credentials.

    The serialized identity is hashed before it becomes a filesystem path, so
    server URLs, client identifiers, query parameters, and secrets are not
    exposed in cache filenames.
    """

    server_name: str = field(repr=False)
    endpoint: _NormalizedCacheURL
    client_id: str | None
    scopes: tuple[str, ...]
    auth_server_metadata: _NormalizedCacheURL | None
    client_secret_fingerprint: str | None
    callback_port: int | None

    @classmethod
    def create(
        cls,
        server_name: str,
        server_url: str,
        config: MCPOAuthConfig,
        *,
        client_secret: str | None,
        callback_port: int | None,
    ) -> OAuthCacheIdentity:
        validated_metadata_url = (
            _validate_oauth_url(config.auth_server_metadata_url, "auth_server_metadata_url")
            if config.auth_server_metadata_url
            else None
        )
        metadata_url = (
            _normalize_cache_url(str(validated_metadata_url)) if validated_metadata_url else None
        )
        return cls(
            server_name=server_name,
            endpoint=_normalize_cache_url(server_url),
            client_id=config.client_id,
            scopes=tuple(sorted(set(config.scopes))),
            auth_server_metadata=metadata_url,
            client_secret_fingerprint=_secret_fingerprint(client_secret),
            callback_port=callback_port,
        )

    def cache_key(self) -> str:
        payload = {
            "version": _CACHE_VERSION,
            "server_name": self.server_name,
            "endpoint": self.endpoint.__dict__,
            "client_id": self.client_id,
            "scopes": self.scopes,
            "auth_server_metadata": (
                self.auth_server_metadata.__dict__ if self.auth_server_metadata else None
            ),
            "client_secret_fingerprint": self.client_secret_fingerprint,
            "callback_port": self.callback_port,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _OAuthServerBinding:
    """Validated authorization-server identity attached to cached secrets."""

    authorization_server: str
    authorization_endpoint: str
    token_endpoint: str
    registration_endpoint: str | None

    def as_dict(self) -> Dict[str, str | None]:
        return {
            "authorization_server": self.authorization_server,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "registration_endpoint": self.registration_endpoint,
        }


def _bound_tokens(tokens: Dict[str, Any], binding: _OAuthServerBinding) -> Dict[str, Any]:
    bound = dict(tokens)
    bound[_SERVER_BINDING_KEY] = binding.as_dict()
    return bound


def _binding_matches(tokens: Dict[str, Any], binding: _OAuthServerBinding) -> bool:
    return tokens.get(_SERVER_BINDING_KEY) == binding.as_dict()


def _token_record_digest(tokens: Dict[str, Any]) -> str:
    """Return a stable, secret-preserving generation digest for a token record."""
    serialized = json.dumps(tokens, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _RefreshRecordSnapshot:
    """Exact token-record generation used to issue one refresh request."""

    binding: _OAuthServerBinding
    refresh_token: str
    record_digest: str


class _LockUnavailableError(RuntimeError):
    """Raised when a verified nonblocking cache-lock attempt would block."""


class _OAuthBindingMismatchError(RuntimeError):
    """Raised when cached credentials belong to another authorization server."""


class _OAuthStaleStateError(RuntimeError):
    """Raised when persisted OAuth state changes during an in-flight operation."""


def _token_dir(identity: OAuthCacheIdentity) -> Path:
    return Path.home() / ".koder" / _AUTH_DIR_NAME / f"v{_CACHE_VERSION}" / identity.cache_key()


def _token_file(identity: OAuthCacheIdentity) -> Path:
    return _token_dir(identity) / _TOKEN_FILE_NAME


def _is_link_or_reparse_point(file_stat: os.stat_result) -> bool:
    """Return whether *file_stat* identifies a symlink or Windows reparse point."""
    if stat.S_ISLNK(file_stat.st_mode):
        return True
    attributes = getattr(file_stat, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(reparse_flag and attributes & reparse_flag)


def _same_file(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _protection_entropy(identity: OAuthCacheIdentity) -> bytes:
    return hashlib.sha256(
        f"koder-mcp-oauth-v{_CACHE_VERSION}:{identity.cache_key()}".encode("ascii")
    ).digest()


def _encode_token_payload(
    identity: OAuthCacheIdentity,
    capabilities: _SecureStorageCapabilities,
    tokens: Dict[str, Any],
) -> bytes:
    plaintext = json.dumps(tokens, indent=2).encode("utf-8")
    if capabilities.enforce_posix_modes:
        return plaintext
    protector = capabilities.data_protector
    if protector is None:
        raise RuntimeError("Non-POSIX MCP OAuth at-rest protection is unavailable")
    protected = protector.protect(plaintext, _protection_entropy(identity))
    if not protected:
        raise RuntimeError("Non-POSIX MCP OAuth at-rest protection failed")
    return _PROTECTED_PAYLOAD_MAGIC + protected


def _decode_token_payload(
    identity: OAuthCacheIdentity,
    capabilities: _SecureStorageCapabilities,
    payload: bytes,
) -> Dict[str, Any]:
    if capabilities.enforce_posix_modes:
        plaintext = payload
    else:
        protector = capabilities.data_protector
        if protector is None or not payload.startswith(_PROTECTED_PAYLOAD_MAGIC):
            raise PermissionError("MCP OAuth token payload is not protected at rest")
        plaintext = protector.unprotect(
            payload[len(_PROTECTED_PAYLOAD_MAGIC) :],
            _protection_entropy(identity),
        )
    decoded = json.loads(plaintext.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("MCP OAuth token payload must be a JSON object")
    return decoded


def _validate_regular_file_stat(
    file_stat: os.stat_result,
    *,
    label: str,
    enforce_mode: bool,
    allow_unlinked: bool = False,
) -> None:
    """Reject links, reparse points, foreign owners, and weak POSIX modes."""
    link_count = getattr(file_stat, "st_nlink", 0)
    if (
        not stat.S_ISREG(file_stat.st_mode)
        or _is_link_or_reparse_point(file_stat)
        or (link_count != 1 and not (allow_unlinked and link_count == 0))
    ):
        raise OSError(f"MCP OAuth {label} is not a private single-link regular file")
    if enforce_mode and file_stat.st_mode & 0o777 != 0o600:
        raise PermissionError(f"MCP OAuth {label} is not owner-only")
    if enforce_mode and hasattr(os, "geteuid"):
        owner = getattr(file_stat, "st_uid", None)
        if owner is None or owner != os.geteuid():
            raise PermissionError(f"MCP OAuth {label} is not owned by the current user")


def _validate_opened_named_file(
    opened_stat: os.stat_result,
    path_stat: os.stat_result,
    *,
    label: str,
    enforce_mode: bool,
) -> None:
    _validate_regular_file_stat(opened_stat, label=label, enforce_mode=enforce_mode)
    _validate_regular_file_stat(path_stat, label=label, enforce_mode=enforce_mode)
    if not _same_file(opened_stat, path_stat):
        raise OSError(f"MCP OAuth {label} changed during operation")


def _validate_private_directory_fd(fd: int, *, enforce_mode: bool) -> None:
    directory_stat = os.fstat(fd)
    if not stat.S_ISDIR(directory_stat.st_mode):
        raise OSError("MCP OAuth cache path is not a private directory")
    if enforce_mode and directory_stat.st_mode & 0o777 != 0o700:
        raise PermissionError("MCP OAuth cache directory is not owner-only")


def _open_posix_directory_path(path: Path, nofollow_flag: int) -> int:
    """Open an absolute directory by walking every component without links."""
    if not path.is_absolute():
        raise OSError("MCP OAuth cache root must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    current_fd = os.open(path.anchor, flags | nofollow_flag)
    try:
        for component in path.parts[1:]:
            if component in {"", ".", ".."}:
                raise OSError("MCP OAuth cache root contains an unsafe component")
            next_fd = os.open(component, flags | nofollow_flag, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        _validate_private_directory_fd(current_fd, enforce_mode=False)
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _open_posix_token_directory(
    identity: OAuthCacheIdentity,
    capabilities: _SecureStorageCapabilities,
    *,
    create: bool,
) -> int | None:
    """Return the cache directory fd, anchored below a no-follow home fd."""
    current_fd = _open_posix_directory_path(Path.home(), capabilities.nofollow_flag)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | capabilities.nofollow_flag
    )
    components = (".koder", _AUTH_DIR_NAME, f"v{_CACHE_VERSION}", identity.cache_key())
    try:
        for component in components:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    os.close(current_fd)
                    return None
                raise
            os.close(current_fd)
            current_fd = next_fd
            if create:
                os.fchmod(current_fd, 0o700)
            _validate_private_directory_fd(current_fd, enforce_mode=True)
        return current_fd
    except Exception:
        os.close(current_fd)
        raise


def _non_posix_cache_chain(identity: OAuthCacheIdentity) -> tuple[Path, ...]:
    home = Path.home()
    return (
        home,
        home / ".koder",
        home / ".koder" / _AUTH_DIR_NAME,
        home / ".koder" / _AUTH_DIR_NAME / f"v{_CACHE_VERSION}",
        _token_dir(identity),
    )


def _validate_non_posix_chain(
    identity: OAuthCacheIdentity,
    *,
    expected_parent: os.stat_result | None = None,
) -> os.stat_result:
    """Reject reparse points and changed parents on platforms without openat."""
    chain = _non_posix_cache_chain(identity)
    for path in chain:
        path_stat = path.lstat()
        if _is_link_or_reparse_point(path_stat) or not stat.S_ISDIR(path_stat.st_mode):
            raise OSError("MCP OAuth cache path is not a private directory")
    parent_stat = chain[-1].lstat()
    if expected_parent is not None and not _same_file(parent_stat, expected_parent):
        raise OSError("MCP OAuth cache directory changed during operation")
    return parent_stat


def _open_non_posix_token_directory(
    identity: OAuthCacheIdentity,
    *,
    create: bool,
) -> tuple[Path, os.stat_result] | None:
    chain = _non_posix_cache_chain(identity)
    if create:
        parent_stat = chain[0].lstat()
        if _is_link_or_reparse_point(parent_stat) or not stat.S_ISDIR(parent_stat.st_mode):
            raise OSError("MCP OAuth cache path is not a private directory")
        for path in chain[1:]:
            current_parent = path.parent.lstat()
            if not _same_file(current_parent, parent_stat):
                raise OSError("MCP OAuth cache directory changed during operation")
            try:
                os.mkdir(path, 0o700)
            except FileExistsError:
                pass
            if not _same_file(path.parent.lstat(), parent_stat):
                raise OSError("MCP OAuth cache directory changed during operation")
            path_stat = path.lstat()
            if _is_link_or_reparse_point(path_stat) or not stat.S_ISDIR(path_stat.st_mode):
                raise OSError("MCP OAuth cache path is not a private directory")
            parent_stat = path_stat
    elif not chain[-1].exists():
        return None
    return chain[-1], _validate_non_posix_chain(identity)


def _lock_file(fd: int, *, blocking: bool = True) -> bool:
    if fcntl is not None:
        operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(fd, operation)
        except BlockingIOError:
            return False
        return True
    if msvcrt is None:  # pragma: no cover - unsupported interpreter
        raise RuntimeError("Cross-process MCP OAuth cache locking is unavailable")
    if os.fstat(fd).st_size == 0:  # pragma: no cover - Windows
        os.write(fd, b"\0")
        os.fsync(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        msvcrt.locking(fd, msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
    except OSError as exc:  # pragma: no cover - Windows
        if not blocking and exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise
    return True


def _unlock_file(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return
    if msvcrt is not None:  # pragma: no cover - Windows
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _open_posix_lock_file(
    parent_fd: int,
    lock_name: str,
    capabilities: _SecureStorageCapabilities,
) -> int:
    """Create or open a lock with a verified winner inode and bounded retries."""
    base_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | capabilities.nofollow_flag
    for _ in range(_LOCK_OPEN_ATTEMPTS):
        try:
            fd = os.open(
                lock_name,
                base_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            try:
                fd = os.open(lock_name, base_flags, dir_fd=parent_fd)
            except FileNotFoundError:
                continue
        except FileNotFoundError:
            continue
        try:
            os.fchmod(fd, 0o600)
            opened_stat = os.fstat(fd)
            try:
                path_stat = os.stat(
                    lock_name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                os.close(fd)
                continue
            _validate_opened_named_file(
                opened_stat,
                path_stat,
                label="cache lock",
                enforce_mode=True,
            )
            return fd
        except Exception:
            os.close(fd)
            raise
    raise OSError(errno.ENOENT, "Could not stably open the MCP OAuth cache lock")


def _open_non_posix_lock_file(
    identity: OAuthCacheIdentity,
    parent_path: Path,
    parent_stat: os.stat_result,
    lock_name: str,
) -> int:
    """Non-POSIX equivalent of verified create-or-open lock acquisition."""
    lock_path = parent_path / lock_name
    base_flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
    for _ in range(_LOCK_OPEN_ATTEMPTS):
        _validate_non_posix_chain(identity, expected_parent=parent_stat)
        try:
            fd = os.open(lock_path, base_flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                fd = os.open(lock_path, base_flags)
            except FileNotFoundError:
                continue
        except FileNotFoundError:
            continue
        try:
            opened_stat = os.fstat(fd)
            try:
                path_stat = lock_path.lstat()
            except FileNotFoundError:
                os.close(fd)
                continue
            _validate_non_posix_chain(identity, expected_parent=parent_stat)
            _validate_opened_named_file(
                opened_stat,
                path_stat,
                label="cache lock",
                enforce_mode=False,
            )
            return fd
        except Exception:
            os.close(fd)
            raise
    raise OSError(errno.ENOENT, "Could not stably open the MCP OAuth cache lock")


@dataclass
class _TokenStore:
    identity: OAuthCacheIdentity
    capabilities: _SecureStorageCapabilities
    parent_fd: int | None = None
    parent_path: Path | None = None
    parent_stat: os.stat_result | None = None

    def _validate_non_posix_parent(self) -> None:
        if self.parent_stat is not None:
            _validate_non_posix_chain(self.identity, expected_parent=self.parent_stat)

    @staticmethod
    def _is_owned_temp_name(name: str) -> bool:
        if not name.startswith(_TEMP_FILE_PREFIX) or not name.endswith(_TEMP_FILE_SUFFIX):
            return False
        nonce = name[len(_TEMP_FILE_PREFIX) : -len(_TEMP_FILE_SUFFIX)]
        return len(nonce) == 32 and all(character in "0123456789abcdef" for character in nonce)

    def _path_stat(self, name: str) -> os.stat_result:
        if self.parent_fd is not None:
            return os.stat(name, dir_fd=self.parent_fd, follow_symlinks=False)
        assert self.parent_path is not None
        self._validate_non_posix_parent()
        return (self.parent_path / name).lstat()

    def _open_named_readonly(self, name: str) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if self.parent_fd is not None:
            return os.open(
                name,
                flags | self.capabilities.nofollow_flag,
                dir_fd=self.parent_fd,
            )
        assert self.parent_path is not None
        self._validate_non_posix_parent()
        return os.open(self.parent_path / name, flags)

    def _unlink_name(self, name: str) -> None:
        if self.parent_fd is not None:
            os.unlink(name, dir_fd=self.parent_fd)
        else:
            assert self.parent_path is not None
            self._validate_non_posix_parent()
            (self.parent_path / name).unlink()
            self._validate_non_posix_parent()

    def _validate_named_file(self, name: str, *, label: str) -> os.stat_result:
        fd = self._open_named_readonly(name)
        try:
            opened_stat = os.fstat(fd)
            path_stat = self._path_stat(name)
            _validate_opened_named_file(
                opened_stat,
                path_stat,
                label=label,
                enforce_mode=self.capabilities.enforce_posix_modes,
            )
            return opened_stat
        finally:
            os.close(fd)

    def _validate_created_temp(
        self,
        name: str,
        expected_stat: os.stat_result,
        *,
        enforce_mode: bool,
    ) -> os.stat_result:
        """Verify that *name* still resolves to this save's created inode."""
        fd = self._open_named_readonly(name)
        try:
            opened_stat = os.fstat(fd)
            path_stat = self._path_stat(name)
            _validate_opened_named_file(
                opened_stat,
                path_stat,
                label="temporary file",
                enforce_mode=enforce_mode,
            )
            if not _same_file(opened_stat, expected_stat):
                raise OSError("MCP OAuth temporary file changed during operation")
            return opened_stat
        finally:
            os.close(fd)

    def sweep_stale_temps(self) -> None:
        """Remove only exact, private, single-link cache temp files."""
        if self.parent_fd is not None:
            names = os.listdir(self.parent_fd)
        else:
            assert self.parent_path is not None
            self._validate_non_posix_parent()
            names = [path.name for path in self.parent_path.iterdir()]
        for name in names:
            if not self._is_owned_temp_name(name):
                continue
            try:
                self._validate_named_file(name, label="temporary file")
            except FileNotFoundError:
                continue
            self._unlink_name(name)
        if self.parent_fd is not None:
            os.fsync(self.parent_fd)

    def load(self) -> Dict[str, Any] | None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        if self.parent_fd is not None:
            flags |= self.capabilities.nofollow_flag
            try:
                fd = os.open(_TOKEN_FILE_NAME, flags, dir_fd=self.parent_fd)
            except FileNotFoundError:
                return None
        else:
            assert self.parent_path is not None
            self._validate_non_posix_parent()
            path = self.parent_path / _TOKEN_FILE_NAME
            try:
                fd = os.open(path, flags)
            except FileNotFoundError:
                return None
            opened_stat = os.fstat(fd)
            try:
                path_stat = path.lstat()
                self._validate_non_posix_parent()
                _validate_opened_named_file(
                    opened_stat,
                    path_stat,
                    label="token file",
                    enforce_mode=False,
                )
            except Exception:
                os.close(fd)
                raise

        file_stat = os.fstat(fd)
        try:
            _validate_regular_file_stat(
                file_stat,
                label="token file",
                enforce_mode=self.capabilities.enforce_posix_modes,
                # A POSIX path swapped after the no-follow open leaves this
                # already-open trusted inode at nlink=0. Reading that exact fd
                # preserves the prior anti-swap guarantee; nlink>1 is still
                # rejected as a hardlink substitution.
                allow_unlinked=self.parent_fd is not None,
            )
        except Exception:
            os.close(fd)
            raise
        with os.fdopen(fd, "rb") as handle:
            return _decode_token_payload(
                self.identity,
                self.capabilities,
                handle.read(),
            )

    def save(self, tokens: Dict[str, Any]) -> None:
        temp_name = f".{_TOKEN_FILE_NAME}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if self.parent_fd is not None:
            flags |= self.capabilities.nofollow_flag
            fd = os.open(temp_name, flags, 0o600, dir_fd=self.parent_fd)
        else:
            assert self.parent_path is not None
            self._validate_non_posix_parent()
            temp_path = self.parent_path / temp_name
            fd = os.open(temp_path, flags, 0o600)
            opened_stat = os.fstat(fd)
            try:
                path_stat = temp_path.lstat()
                self._validate_non_posix_parent()
                _validate_opened_named_file(
                    opened_stat,
                    path_stat,
                    label="temporary file",
                    enforce_mode=False,
                )
            except Exception:
                os.close(fd)
                raise

        created_temp_stat = os.fstat(fd)
        try:
            if self.capabilities.enforce_posix_modes:
                os.fchmod(fd, 0o600)
            file_stat = os.fstat(fd)
            _validate_regular_file_stat(
                file_stat,
                label="temporary file",
                enforce_mode=self.capabilities.enforce_posix_modes,
            )
            payload = _encode_token_payload(self.identity, self.capabilities, tokens)
            with os.fdopen(fd, "wb") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            self._validate_created_temp(
                temp_name,
                created_temp_stat,
                enforce_mode=self.capabilities.enforce_posix_modes,
            )

            if self.parent_fd is not None:
                os.replace(
                    temp_name,
                    _TOKEN_FILE_NAME,
                    src_dir_fd=self.parent_fd,
                    dst_dir_fd=self.parent_fd,
                )
                os.fsync(self.parent_fd)
            else:
                assert self.parent_path is not None
                self._validate_non_posix_parent()
                os.replace(
                    self.parent_path / temp_name,
                    self.parent_path / _TOKEN_FILE_NAME,
                )
                self._validate_non_posix_parent()
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                self._validate_created_temp(
                    temp_name,
                    created_temp_stat,
                    # Cleanup may be needed specifically because securing the
                    # mode failed. The inode, type, link count, and path match
                    # still must be exact before unlinking it.
                    enforce_mode=False,
                )
                self._unlink_name(temp_name)
            except FileNotFoundError:
                pass

    def clear(self) -> None:
        try:
            self._validate_named_file(_TOKEN_FILE_NAME, label="token file")
            self._unlink_name(_TOKEN_FILE_NAME)
            if self.parent_fd is not None:
                os.fsync(self.parent_fd)
        except FileNotFoundError:
            pass


@contextmanager
def _locked_token_store(
    identity: OAuthCacheIdentity,
    *,
    create: bool,
    lock_name: str = _TOKEN_LOCK_NAME,
    sweep_temps: bool = True,
    lock_blocking: bool = True,
) -> Generator[_TokenStore | None, None, None]:
    capabilities = _secure_storage_capabilities()
    _validate_secure_storage_capabilities(capabilities)
    parent_fd: int | None = None
    lock_fd: int | None = None
    lock_acquired = False
    store: _TokenStore | None = None
    try:
        if capabilities.enforce_posix_modes:
            parent_fd = _open_posix_token_directory(identity, capabilities, create=create)
            if parent_fd is None:
                yield None
                return
            lock_fd = _open_posix_lock_file(parent_fd, lock_name, capabilities)
            store = _TokenStore(identity, capabilities, parent_fd=parent_fd)
        else:
            opened = _open_non_posix_token_directory(identity, create=create)
            if opened is None:
                yield None
                return
            parent_path, parent_stat = opened
            lock_fd = _open_non_posix_lock_file(
                identity,
                parent_path,
                parent_stat,
                lock_name,
            )
            store = _TokenStore(
                identity,
                capabilities,
                parent_path=parent_path,
                parent_stat=parent_stat,
            )

        if not _lock_file(lock_fd, blocking=lock_blocking):
            raise _LockUnavailableError(f"MCP OAuth cache lock is busy: {lock_name}")
        lock_acquired = True
        if sweep_temps:
            store.sweep_stale_temps()
        yield store
    finally:
        if lock_fd is not None:
            try:
                if lock_acquired:
                    _unlock_file(lock_fd)
            finally:
                os.close(lock_fd)
        if parent_fd is not None:
            os.close(parent_fd)


async def _await_thread_cleanup(task: asyncio.Task[Any]) -> Any:
    """Finish a thread-backed lock operation even if the caller is cancelled."""
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


@asynccontextmanager
async def _locked_refresh_grant(
    identity: OAuthCacheIdentity,
) -> AsyncGenerator[None, None]:
    """Hold the refresh lock using promptly cancellable nonblocking polling."""
    while True:
        manager = _locked_token_store(
            identity,
            create=True,
            lock_name=_REFRESH_LOCK_NAME,
            sweep_temps=False,
            lock_blocking=False,
        )
        enter_task = asyncio.create_task(asyncio.to_thread(manager.__enter__))
        entered = False
        try:
            await asyncio.shield(enter_task)
            entered = True
        except asyncio.CancelledError:
            try:
                await asyncio.shield(enter_task)
                entered = True
            except _LockUnavailableError:
                pass
            if entered:
                exit_task = asyncio.create_task(
                    asyncio.to_thread(manager.__exit__, None, None, None)
                )
                await _await_thread_cleanup(exit_task)
            raise
        except _LockUnavailableError:
            await asyncio.sleep(_REFRESH_LOCK_POLL_INTERVAL)
            continue
        break

    try:
        yield
    finally:
        exit_task = asyncio.create_task(asyncio.to_thread(manager.__exit__, None, None, None))
        await _await_thread_cleanup(exit_task)


def load_tokens(identity: OAuthCacheIdentity) -> Dict[str, Any] | None:
    """Load tokens from the exact regular file opened below a trusted parent."""
    try:
        with _locked_token_store(identity, create=False) as store:
            return store.load() if store is not None else None
    except Exception:
        logger.debug("Failed to read MCP OAuth token file")
        return None


def save_tokens(identity: OAuthCacheIdentity, tokens: Dict[str, Any]) -> None:
    """Atomically persist *tokens* while holding the cross-process cache lock."""
    with _locked_token_store(identity, create=True) as store:
        assert store is not None
        store.save(tokens)


def clear_tokens(identity: OAuthCacheIdentity) -> None:
    """Remove persisted tokens for an exact cache *identity*."""
    with _locked_token_store(identity, create=False) as store:
        if store is not None:
            store.clear()


def _update_tokens(
    identity: OAuthCacheIdentity,
    update: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    """Serialize a token-record read-modify-write across processes."""
    with _locked_token_store(identity, create=True) as store:
        assert store is not None
        try:
            latest = store.load() or {}
        except (OSError, ValueError, json.JSONDecodeError):
            latest = {}
        updated = update(latest)
        store.save(updated)
        return updated


def _load_bound_tokens(
    identity: OAuthCacheIdentity,
    binding: _OAuthServerBinding,
    *,
    reject_mismatch: bool = False,
    clear_mismatch: bool = True,
) -> Dict[str, Any] | None:
    """Load and validate a bound record under one cache lock."""
    with _locked_token_store(identity, create=False) as store:
        if store is None:
            return None
        try:
            cached = store.load()
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        if cached and not _binding_matches(cached, binding):
            if clear_mismatch:
                store.clear()
            if reject_mismatch:
                raise _OAuthBindingMismatchError(
                    "OAuth authorization server changed; reauthorization required"
                )
            return None
        return cached


def _merge_bound_tokens(
    identity: OAuthCacheIdentity,
    tokens: Dict[str, Any],
    binding: _OAuthServerBinding,
) -> Dict[str, Any]:
    """Merge fields into the latest same-server record under one cache lock."""

    def merge(latest: Dict[str, Any]) -> Dict[str, Any]:
        if latest and not _binding_matches(latest, binding):
            latest = {}
        return _bound_tokens({**latest, **tokens}, binding)

    return _update_tokens(identity, merge)


def _commit_refreshed_tokens(
    identity: OAuthCacheIdentity,
    tokens: Dict[str, Any],
    snapshot: _RefreshRecordSnapshot,
) -> Dict[str, Any]:
    """Commit a refresh response only if its exact input generation remains current."""
    stale_error = "OAuth token refresh response is stale because cached credentials changed"
    with _locked_token_store(identity, create=False) as store:
        if store is None:
            raise _OAuthStaleStateError(stale_error)
        try:
            latest = store.load()
        except (OSError, ValueError, json.JSONDecodeError):
            raise _OAuthStaleStateError(stale_error) from None
        if (
            not latest
            or not _binding_matches(latest, snapshot.binding)
            or latest.get("refresh_token") != snapshot.refresh_token
            or _token_record_digest(latest) != snapshot.record_digest
        ):
            raise _OAuthStaleStateError(stale_error)
        merged = _bound_tokens({**latest, **tokens}, snapshot.binding)
        store.save(merged)
        return merged


# ---------------------------------------------------------------------------
# Local callback HTTP server
# ---------------------------------------------------------------------------


@dataclass
class _CallbackResult:
    """Per-flow container for the loopback callback outcome.

    Instances are bound to a single :class:`HTTPServer` so concurrent OAuth
    flows never share mutable state (previously stored as class attributes on
    the handler, which raced across flows).
    """

    auth_code: str | None = None
    error: str | None = None


class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Handler that captures the ``code`` query parameter for one flow.

    Expected ``state`` (CSRF token), the callback path, and the result
    container are read from the owning :class:`HTTPServer` instance so that
    each flow validates against its own generated state and writes into its
    own container. A request whose ``state`` does not match the value this
    flow generated is rejected without recording an auth code.
    """

    timeout = _CALLBACK_REQUEST_TIMEOUT_SECONDS

    def do_GET(self) -> None:  # noqa: N802
        from urllib.parse import parse_qs, urlparse

        result: _CallbackResult = getattr(self.server, "oauth_result", None) or _CallbackResult()
        expected_state: str | None = getattr(self.server, "oauth_expected_state", None)
        expected_path: str | None = getattr(self.server, "oauth_callback_path", None)

        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        code = query.get("code", [None])[0]
        returned_state = query.get("state", [None])[0]

        # Reject callbacks on an unexpected path (ignore favicon/probes, etc.).
        if expected_path is not None and parsed.path != expected_path:
            self.send_response(404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Not found.</h2></body></html>")
            return

        # CSRF protection: the returned state MUST match the value this flow
        # generated. A missing or mismatched state is rejected outright so a
        # forged callback carrying an attacker-chosen ``code`` cannot be
        # accepted (RFC 6749 section 10.12).
        if expected_state is not None and returned_state != expected_state:
            result.error = "state_mismatch"
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization failed: state mismatch.</h2></body></html>"
            )
            return

        if code:
            result.auth_code = code
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h2>Authorization successful.</h2>"
                b"<p>You can close this window and return to the terminal.</p>"
                b"</body></html>"
            )
        else:
            result.error = "authorization_error"
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h2>Authorization failed.</h2></body></html>")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # BaseHTTPRequestHandler's default format includes the full request
        # target. OAuth callbacks carry authorization codes and CSRF state in
        # that target, so never forward it to diagnostics.
        logger.debug("OAuth callback HTTP request handled")


class _OAuthCallbackServer(HTTPServer):
    """Own accepted connections so a stalled request cannot prevent shutdown."""

    def __init__(self, server_address, handler):
        self._request_lock = Lock()
        self._active_requests: set[socket.socket] = set()
        self._callback_thread: Thread | None = None
        self._closing = False
        super().__init__(server_address, handler)

    def get_request(self):
        request, address = super().get_request()
        with self._request_lock:
            if not self._closing:
                self._active_requests.add(request)
                return request, address
        request.close()
        raise OSError("OAuth callback server is stopping")

    def close_request(self, request):
        try:
            super().close_request(request)
        finally:
            with self._request_lock:
                self._active_requests.discard(request)

    def begin_shutdown(self) -> None:
        with self._request_lock:
            self._closing = True
            requests = tuple(self._active_requests)
        for request in requests:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                # The handler may have closed the connection concurrently.
                pass


def _stop_callback_server(server: HTTPServer) -> None:
    """Stop admission, release active reads and close the listening socket."""
    try:
        if isinstance(server, _OAuthCallbackServer):
            server.begin_shutdown()
    finally:
        try:
            server.shutdown()
        finally:
            try:
                server.server_close()
            finally:
                if isinstance(server, _OAuthCallbackServer) and server._callback_thread is not None:
                    server._callback_thread.join()


def _start_callback_server(port: int | None) -> tuple[HTTPServer, int]:
    """Start a local HTTP server and return ``(server, actual_port)``.

    The returned server carries a fresh :class:`_CallbackResult` on
    ``server.oauth_result``; the caller binds the expected ``state`` and
    callback path before opening the browser.
    """
    bind_port = port or 0  # 0 = OS picks a free port
    server = _OAuthCallbackServer(("127.0.0.1", bind_port), _OAuthCallbackHandler)
    thread = None
    try:
        # Per-flow state lives on the server instance, not on the handler class,
        # so concurrent flows do not clobber each other.
        server.oauth_result = _CallbackResult()  # type: ignore[attr-defined]
        server.oauth_expected_state = None  # type: ignore[attr-defined]
        server.oauth_callback_path = None  # type: ignore[attr-defined]
        actual_port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        server._callback_thread = thread
        thread.start()
    except BaseException:
        if thread is not None and thread.is_alive():
            _stop_callback_server(server)
        else:
            server.server_close()
        raise
    return server, actual_port


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def _generate_pkce() -> tuple[str, str]:
    """Return ``(code_verifier, code_challenge)`` for S256."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


# ---------------------------------------------------------------------------
# OAuth flow
# ---------------------------------------------------------------------------


class MCPOAuthFlow:
    """Performs a full OAuth 2.0 Authorization Code flow for an MCP server."""

    def __init__(
        self,
        server_name: str,
        server_url: str,
        config: MCPOAuthConfig,
    ) -> None:
        self.server_name = server_name
        self.server_url = server_url
        self.config = config
        self._client_secret = config.client_secret or os.environ.get("MCP_CLIENT_SECRET")
        self._callback_port = (
            config.callback_port or int(os.environ.get("MCP_OAUTH_CALLBACK_PORT", "0")) or None
        )
        self.cache_identity = OAuthCacheIdentity.create(
            server_name,
            server_url,
            config,
            client_secret=self._client_secret,
            callback_port=self._callback_port,
        )
        self._metadata: Dict[str, Any] | None = None
        self._discovered_authorization_server: str | None = None

    # -- public api ---------------------------------------------------------

    async def authenticate(self) -> Dict[str, str]:
        """Run the OAuth flow and return authorization headers.

        If valid cached tokens exist they are reused.  Expired tokens are
        refreshed automatically when a refresh token is available.
        """
        # Discovery must precede every credential reuse. The protected
        # resource may delegate to a different authorization server over time,
        # and cached access/refresh/DCR credentials must never follow it.
        metadata = await self._discover_fresh_metadata()
        binding = self._server_binding(metadata)

        cached = _load_bound_tokens(self.cache_identity, binding)
        if cached:
            if cached.get("access_token") and not self._is_expired(cached):
                return {"Authorization": f"Bearer {cached['access_token']}"}
            if cached.get("refresh_token"):
                try:
                    return await self.refresh_token(
                        metadata=metadata,
                        cached=cached,
                        binding=binding,
                    )
                except _OAuthStaleStateError:
                    logger.debug("OAuth state changed during refresh; aborting stale flow")
                    raise
                except Exception:
                    logger.debug("Token refresh failed; starting new flow")

        # Start the local callback server first so dynamic client
        #    registration can use the real redirect URI. Servers that
        #    enforce exact redirect_uri matching (RFC 8252) reject
        #    authorization requests whose URI differs from registration.
        callback_port = self._resolve_callback_port()
        server, actual_port = _start_callback_server(callback_port)
        redirect_uri = f"http://127.0.0.1:{actual_port}/callback"

        primary_error: BaseException | None = None
        try:
            client_id, client_secret = await self._ensure_client(metadata, redirect_uri)

            tokens = await self._authorization_code_flow(
                metadata, client_id, client_secret, server=server, redirect_uri=redirect_uri
            )
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await run_sync_owned(_stop_callback_server, server)
            except BaseException as exc:
                if primary_error is None:
                    raise
                logger.debug("OAuth callback cleanup failed (%s)", type(exc).__name__)
        _merge_bound_tokens(self.cache_identity, tokens, binding)
        return {"Authorization": f"Bearer {tokens['access_token']}"}

    async def refresh_token(
        self,
        *,
        metadata: Dict[str, Any] | None = None,
        cached: Dict[str, Any] | None = None,
        binding: _OAuthServerBinding | None = None,
    ) -> Dict[str, str]:
        """Refresh an expired access token and return new headers."""
        if metadata is None:
            metadata = await self._discover_fresh_metadata()
        if binding is None:
            binding = self._server_binding(metadata)
        observed_refresh_record = bool(
            cached and cached.get("refresh_token") and _binding_matches(cached, binding)
        )
        stale_before_request = (
            "OAuth token refresh state is stale because cached credentials "
            "changed before the refresh request started"
        )
        # A separate per-identity refresh lock spans the complete rotating
        # grant. Token-file operations still take their short-lived data lock,
        # so other cache updates cannot deadlock behind a network await.
        async with _locked_refresh_grant(self.cache_identity):
            try:
                cached = _load_bound_tokens(
                    self.cache_identity,
                    binding,
                    reject_mismatch=True,
                    clear_mismatch=not observed_refresh_record,
                )
            except _OAuthBindingMismatchError:
                if observed_refresh_record:
                    raise _OAuthStaleStateError(stale_before_request) from None
                raise
            if not cached or not cached.get("refresh_token"):
                if observed_refresh_record:
                    raise _OAuthStaleStateError(stale_before_request)
                raise RuntimeError("No refresh token available")
            if not _binding_matches(cached, binding):
                raise RuntimeError("OAuth authorization server changed; reauthorization required")

            refresh_snapshot = _RefreshRecordSnapshot(
                binding=binding,
                refresh_token=cached["refresh_token"],
                record_digest=_token_record_digest(cached),
            )

            token_endpoint = str(
                self._validated_oauth_url(metadata.get("token_endpoint"), "token_endpoint")
            )

            client_id = cached.get("client_id") or self.config.client_id
            client_secret = cached.get("client_secret") or self._client_secret

            payload: Dict[str, str] = {
                "grant_type": "refresh_token",
                "refresh_token": cached["refresh_token"],
                "client_id": client_id,
            }
            if client_secret:
                payload["client_secret"] = client_secret

            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(token_endpoint, data=payload)
                    resp.raise_for_status()
                    token_data = resp.json()
            except httpx.HTTPError:
                raise RuntimeError("OAuth token refresh request failed") from None

            # Merge new tokens with old (server may not return a new refresh_token).
            # No other refresh grant can have used the cached rotating token
            # while this lock was held.
            token_data["obtained_at"] = time.time()
            merged = _commit_refreshed_tokens(
                self.cache_identity,
                token_data,
                refresh_snapshot,
            )
            return {"Authorization": f"Bearer {merged['access_token']}"}

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _is_expired(tokens: Dict[str, Any]) -> bool:
        expires_in = tokens.get("expires_in")
        obtained_at = tokens.get("obtained_at")
        if expires_in is None or obtained_at is None:
            # No usable expiry metadata: do NOT assume infinite validity.
            # Treat the token as expired so the caller re-validates it —
            # refreshing when a refresh_token exists, otherwise re-running the
            # full authorization flow. This is conservative on purpose; a token
            # that carries expiry still uses the precise check below.
            return True
        return time.time() > obtained_at + expires_in - 30  # 30s grace

    async def _discover_metadata(self) -> Dict[str, Any]:
        if self._metadata is not None:
            return self._metadata

        parsed = urlsplit(self.server_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        urls_to_try = [
            f"{base}/.well-known/oauth-protected-resource",
            f"{base}/.well-known/oauth-authorization-server",
        ]
        if self.config.auth_server_metadata_url:
            urls_to_try.append(self.config.auth_server_metadata_url)

        async with httpx.AsyncClient(follow_redirects=True) as client:
            for url in urls_to_try:
                try:
                    resp = await client.get(url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        discovered_authorization_server: str | None = None
                        # If this is a protected resource document, it may
                        # point to the authorization server via
                        # ``authorization_servers``.
                        if "authorization_servers" in data and "token_endpoint" not in data:
                            authorization_servers = data.get("authorization_servers")
                            if (
                                not isinstance(authorization_servers, list)
                                or not authorization_servers
                            ):
                                continue
                            auth_server_url = authorization_servers[0]
                            if not isinstance(auth_server_url, str):
                                continue
                            self._validated_oauth_url(auth_server_url, "authorization server")
                            meta_resp = await client.get(
                                f"{auth_server_url.rstrip('/')}/.well-known/oauth-authorization-server",
                                timeout=10,
                            )
                            if meta_resp.status_code != 200:
                                continue
                            data = meta_resp.json()
                            discovered_authorization_server = auth_server_url
                        self._server_binding(data, discovered_authorization_server)
                        if discovered_authorization_server is not None:
                            self._discovered_authorization_server = discovered_authorization_server
                        self._metadata = data
                        return data
                except Exception as exc:
                    logger.debug(
                        "OAuth metadata discovery attempt failed (%s)",
                        type(exc).__name__,
                    )
                    continue

            # Fallback: try override URL directly
            if self.config.auth_server_metadata_url:
                try:
                    resp = await client.get(self.config.auth_server_metadata_url, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        self._server_binding(data)
                        self._metadata = data
                        return self._metadata
                except Exception as exc:
                    logger.debug(
                        "OAuth metadata fallback URL fetch failed (%s)",
                        type(exc).__name__,
                    )

        raise RuntimeError(
            "Could not discover valid OAuth metadata for the configured MCP endpoint"
        )

    async def _discover_fresh_metadata(self) -> Dict[str, Any]:
        """Perform a network discovery check before credential reuse."""
        self._metadata = None
        self._discovered_authorization_server = None
        return await self._discover_metadata()

    @staticmethod
    def _validated_oauth_url(value: Any, label: str) -> httpx.URL:
        return _validate_oauth_url(value, label)

    @staticmethod
    def _origin_url(url: httpx.URL) -> str:
        host = url.raw_host.decode("ascii")
        if b":" in url.raw_host:
            host = f"[{host}]"
        port = f":{url.port}" if url.port is not None else ""
        return f"{url.raw_scheme.decode('ascii')}://{host}{port}/"

    def _server_binding(
        self,
        metadata: Dict[str, Any],
        discovered_authorization_server: str | None = None,
    ) -> _OAuthServerBinding:
        authorization_url = self._validated_oauth_url(
            metadata.get("authorization_endpoint"), "authorization_endpoint"
        )
        token_url = self._validated_oauth_url(metadata.get("token_endpoint"), "token_endpoint")
        registration_value = metadata.get("registration_endpoint")
        registration_url = (
            self._validated_oauth_url(registration_value, "registration_endpoint")
            if registration_value is not None
            else None
        )
        issuer_value = metadata.get("issuer")
        issuer_url = self._validated_oauth_url(issuer_value, "issuer") if issuer_value else None
        discovered_url = (
            self._validated_oauth_url(
                discovered_authorization_server,
                "discovered authorization server",
            )
            if discovered_authorization_server
            else None
        )
        if (
            issuer_url is not None
            and discovered_url is not None
            and _normalize_cache_url(str(issuer_url)) != _normalize_cache_url(str(discovered_url))
        ):
            raise RuntimeError(
                "OAuth metadata issuer does not match the discovered authorization server"
            )
        if issuer_url is not None:
            authorization_server_value = str(issuer_url)
        elif discovered_url is not None:
            authorization_server_value = str(discovered_url)
        elif self._discovered_authorization_server:
            authorization_server_value = str(
                self._validated_oauth_url(
                    self._discovered_authorization_server,
                    "discovered authorization server",
                )
            )
        else:
            authorization_server_value = self._origin_url(authorization_url)
        self._validated_oauth_url(authorization_server_value, "issuer")
        return _OAuthServerBinding(
            authorization_server=_normalize_cache_url(
                authorization_server_value
            ).transport_fingerprint,
            authorization_endpoint=_normalize_cache_url(
                str(authorization_url)
            ).transport_fingerprint,
            token_endpoint=_normalize_cache_url(str(token_url)).transport_fingerprint,
            registration_endpoint=(
                _normalize_cache_url(str(registration_url)).transport_fingerprint
                if registration_url is not None
                else None
            ),
        )

    def _resolve_callback_port(self) -> int | None:
        return self._callback_port

    async def _ensure_client(
        self, metadata: Dict[str, Any], redirect_uri: str
    ) -> tuple[str, str | None]:
        """Return ``(client_id, client_secret)`` valid for *redirect_uri*.

        Uses pre-configured values when available. Otherwise reuses a cached
        dynamic registration when it was made for the same redirect URI, and
        performs dynamic client registration (RFC 7591) when not.
        """
        client_secret = self._client_secret
        if self.config.client_id:
            return self.config.client_id, client_secret

        binding = self._server_binding(metadata)

        # Reuse a cached registration only when its redirect URI still
        # matches; a registration made for another port would be rejected by
        # servers that enforce exact redirect_uri matching.
        cached = _load_bound_tokens(self.cache_identity, binding) or {}
        if (
            cached.get("client_id")
            and cached.get("redirect_uri") == redirect_uri
            and _binding_matches(cached, binding)
        ):
            return cached["client_id"], cached.get("client_secret")

        registration_value = metadata.get("registration_endpoint")
        if not registration_value:
            raise RuntimeError(
                "No client_id configured and server does not advertise a "
                "registration_endpoint for dynamic client registration."
            )
        registration_endpoint = str(
            self._validated_oauth_url(registration_value, "registration_endpoint")
        )

        reg_payload: Dict[str, Any] = {
            "client_name": f"koder-mcp-{self.server_name}",
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        if self.config.scopes:
            reg_payload["scope"] = " ".join(self.config.scopes)

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    registration_endpoint,
                    json=reg_payload,
                    timeout=15,
                )
                resp.raise_for_status()
                reg_data = resp.json()
        except httpx.HTTPError:
            raise RuntimeError("OAuth dynamic client registration request failed") from None

        registered_id = reg_data["client_id"]
        registered_secret = reg_data.get("client_secret")

        # Reacquire the cross-process lock after the network request and merge
        # registration fields into the latest record. A concurrent successful
        # authorization/refresh may have written tokens while registration was
        # in flight; those fields must survive. If another process completed an
        # equivalent registration first, prefer that client instead.
        selected: Dict[str, str | None] = {}

        def merge_registration(latest: Dict[str, Any]) -> Dict[str, Any]:
            if latest and not _binding_matches(latest, binding):
                latest = {}
            if latest.get("client_id") and latest.get("redirect_uri") == redirect_uri:
                selected["client_id"] = latest["client_id"]
                selected["client_secret"] = latest.get("client_secret")
                return _bound_tokens(latest, binding)
            merged = dict(latest)
            merged["client_id"] = registered_id
            merged["redirect_uri"] = redirect_uri
            if registered_secret:
                merged["client_secret"] = registered_secret
            selected["client_id"] = registered_id
            selected["client_secret"] = registered_secret
            return _bound_tokens(merged, binding)

        _update_tokens(self.cache_identity, merge_registration)
        return str(selected["client_id"]), selected.get("client_secret")

    async def _authorization_code_flow(
        self,
        metadata: Dict[str, Any],
        client_id: str,
        client_secret: str | None,
        *,
        server: HTTPServer,
        redirect_uri: str,
    ) -> Dict[str, Any]:
        """Run the authorization code grant with PKCE.

        The callback *server* is already listening on the port embedded in
        *redirect_uri*; the caller owns its shutdown.
        """
        authorization_value = metadata.get("authorization_endpoint")
        token_value = metadata.get("token_endpoint")
        if not authorization_value or not token_value:
            raise RuntimeError("OAuth metadata missing authorization_endpoint or token_endpoint")
        authorization_endpoint = str(
            self._validated_oauth_url(authorization_value, "authorization_endpoint")
        )
        token_endpoint = str(self._validated_oauth_url(token_value, "token_endpoint"))

        code_verifier, code_challenge = _generate_pkce()
        state = secrets.token_urlsafe(32)

        # Bind the expected CSRF state and callback path to THIS server so the
        # handler validates the returned state and only this flow's result
        # container is written. State lives on the server instance (the fresh
        # ``oauth_result`` created by ``_start_callback_server``), not on the
        # handler class, so concurrent flows stay isolated.
        server.oauth_expected_state = state  # type: ignore[attr-defined]
        server.oauth_callback_path = urlsplit(redirect_uri).path  # type: ignore[attr-defined]

        auth_params: Dict[str, str] = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        scopes = self.config.scopes or metadata.get("scopes_supported", [])
        if scopes:
            auth_params["scope"] = " ".join(scopes)

        auth_url = f"{authorization_endpoint}?{urlencode(auth_params)}"

        logger.info("Opening browser for OAuth authorization...")
        webbrowser.open(auth_url)

        code = await self._wait_for_code(server, timeout=300)

        token_payload: Dict[str, str] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
        }
        if client_secret:
            token_payload["client_secret"] = client_secret

        try:
            async with httpx.AsyncClient() as http_client:
                resp = await http_client.post(token_endpoint, data=token_payload, timeout=15)
                resp.raise_for_status()
                token_data = resp.json()
        except httpx.HTTPError:
            raise RuntimeError("OAuth authorization code exchange failed") from None

        token_data["obtained_at"] = time.time()
        token_data["client_id"] = client_id
        token_data["redirect_uri"] = redirect_uri
        if client_secret:
            token_data["client_secret"] = client_secret
        return token_data

    @staticmethod
    async def _wait_for_code(server: HTTPServer, timeout: float = 300) -> str:
        """Poll until this flow's callback handler has received the auth code.

        The result is read from the per-flow container bound to *server*, so a
        callback that failed state (CSRF) validation surfaces as an error and
        never yields a code.
        """
        result: _CallbackResult = getattr(server, "oauth_result", None) or _CallbackResult()
        deadline = time.time() + timeout
        while time.time() < deadline:
            if result.auth_code:
                return result.auth_code
            if result.error:
                raise RuntimeError(f"OAuth authorization failed: {result.error}")
            await asyncio.sleep(0.25)
        raise TimeoutError("Timed out waiting for OAuth authorization callback")


# ---------------------------------------------------------------------------
# Convenience: resolve OAuth headers for a server config
# ---------------------------------------------------------------------------


async def resolve_oauth_headers(
    server_name: str,
    server_url: str,
    oauth_dict: Dict[str, Any] | None,
) -> Dict[str, str]:
    """Return OAuth ``Authorization`` headers for a server, or ``{}``."""
    config = MCPOAuthConfig.from_dict(oauth_dict)
    if config is None:
        return {}
    flow = MCPOAuthFlow(server_name, server_url, config)
    return await flow.authenticate()
