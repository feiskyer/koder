"""Tests for MCP OAuth authentication flow (koder_agent.mcp.oauth)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import multiprocessing
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs

import httpx
import pytest

from koder_agent.mcp import oauth as oauth_module
from koder_agent.mcp.oauth import (
    MCPOAuthConfig,
    MCPOAuthFlow,
    _bound_tokens,
    _DataProtector,
    _normalize_cache_url,
    _secure_storage_capabilities,
    _SecureStorageCapabilities,
    _start_callback_server,
    _stop_callback_server,
    _token_file,
    _validate_secure_storage_capabilities,
    clear_tokens,
    load_tokens,
    resolve_oauth_headers,
    save_tokens,
)

AUTH_METADATA = {
    "issuer": "https://auth.example/",
    "authorization_endpoint": "https://auth.example/authorize",
    "token_endpoint": "https://auth.example/token",
    "registration_endpoint": "https://auth.example/register",
}


def _save_bound(
    flow: MCPOAuthFlow,
    tokens: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> None:
    active_metadata = metadata or AUTH_METADATA
    save_tokens(
        flow.cache_identity,
        _bound_tokens(tokens, flow._server_binding(active_metadata)),
    )


def _first_write_worker(home, trial, field, start_event, result_queue) -> None:
    os.environ["HOME"] = home
    flow = MCPOAuthFlow(
        f"first-write-{trial}",
        "https://example.com/mcp",
        MCPOAuthConfig(client_id="client-id"),
    )
    try:
        from koder_agent.mcp.oauth import _update_tokens

        if not start_event.wait(20):
            raise RuntimeError("start timeout")
        _update_tokens(
            flow.cache_identity,
            lambda latest: {**latest, field: field},
        )
        result_queue.put((field, None))
    except BaseException as exc:
        result_queue.put((field, f"{type(exc).__name__}: {exc}"))


def _dynamic_registration_worker(home, metadata, start_event, result_queue) -> None:
    os.environ["HOME"] = home
    flow = MCPOAuthFlow(
        "two-process-registration",
        "https://example.com/mcp",
        MCPOAuthConfig(scopes=["read"]),
    )
    try:
        if not start_event.wait(20):
            raise RuntimeError("start timeout")
        result = asyncio.run(flow._ensure_client(metadata, "http://127.0.0.1:54321/callback"))
        result_queue.put((result, None))
    except BaseException as exc:
        result_queue.put((None, f"{type(exc).__name__}: {exc}"))


def _refresh_worker(home, metadata, start_event, result_queue) -> None:
    os.environ["HOME"] = home
    flow = MCPOAuthFlow(
        "rotating-refresh",
        "https://example.com/mcp",
        MCPOAuthConfig(client_id="client-id"),
    )
    try:
        if not start_event.wait(20):
            raise RuntimeError("start timeout")
        binding = flow._server_binding(metadata)
        headers = asyncio.run(flow.refresh_token(metadata=metadata, binding=binding))
        result_queue.put((headers, None))
    except BaseException as exc:
        result_queue.put((None, f"{type(exc).__name__}: {exc}"))


def _crash_before_replace_worker(home) -> None:
    os.environ["HOME"] = home
    flow = MCPOAuthFlow(
        "crash-before-replace",
        "https://example.com/mcp",
        MCPOAuthConfig(client_id="client-id"),
    )

    def crash(*args, **kwargs):
        os._exit(74)

    with patch("koder_agent.mcp.oauth.os.replace", side_effect=crash):
        save_tokens(
            flow.cache_identity,
            {"access_token": "crash-access", "refresh_token": "crash-refresh"},
        )


# ---------------------------------------------------------------------------
# MCPOAuthConfig
# ---------------------------------------------------------------------------


class TestMCPOAuthConfig:
    def test_from_dict_none_returns_none(self):
        assert MCPOAuthConfig.from_dict(None) is None

    def test_from_dict_empty_returns_none(self):
        assert MCPOAuthConfig.from_dict({}) is None

    def test_from_dict_camel_case_keys(self):
        cfg = MCPOAuthConfig.from_dict(
            {
                "clientId": "my-id",
                "clientSecret": "my-secret",
                "callbackPort": 9999,
                "authServerMetadataUrl": "https://auth.example/.well-known/openid",
                "scopes": ["read", "write"],
            }
        )
        assert cfg is not None
        assert cfg.client_id == "my-id"
        assert cfg.client_secret == "my-secret"
        assert cfg.callback_port == 9999
        assert cfg.auth_server_metadata_url == "https://auth.example/.well-known/openid"
        assert cfg.scopes == ["read", "write"]

    def test_from_dict_snake_case_keys(self):
        cfg = MCPOAuthConfig.from_dict(
            {
                "client_id": "snake-id",
                "client_secret": "snake-secret",
                "callback_port": 8888,
                "auth_server_metadata_url": "https://auth2.example/meta",
            }
        )
        assert cfg is not None
        assert cfg.client_id == "snake-id"
        assert cfg.client_secret == "snake-secret"
        assert cfg.callback_port == 8888
        assert cfg.auth_server_metadata_url == "https://auth2.example/meta"

    def test_from_dict_defaults(self):
        cfg = MCPOAuthConfig.from_dict({"clientId": "only-id"})
        assert cfg is not None
        assert cfg.client_id == "only-id"
        assert cfg.client_secret is None
        assert cfg.callback_port is None
        assert cfg.auth_server_metadata_url is None
        assert cfg.scopes == []


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


class TestTokenPersistence:
    @staticmethod
    def _identity(
        name: str,
        url: str,
        *,
        client_id: str | None = "client-id",
        scopes: list[str] | None = None,
    ):
        config = MCPOAuthConfig(client_id=client_id, scopes=scopes or [])
        return MCPOAuthFlow(name, url, config).cache_identity

    def test_same_name_different_endpoint_does_not_share_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        first = self._identity("shared-name", "https://one.example/mcp")
        second = self._identity("shared-name", "https://two.example/mcp")

        save_tokens(first, {"access_token": "first-endpoint-token"})

        assert load_tokens(first)["access_token"] == "first-endpoint-token"
        assert load_tokens(second) is None

    def test_equivalent_endpoint_spellings_share_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        first = self._identity(
            "equivalent",
            "HTTPS://Example.COM:443/api/../mcp#first-fragment",
            scopes=["write", "read", "read"],
        )
        second = self._identity(
            "equivalent",
            "https://example.com/mcp#second-fragment",
            scopes=["read", "write"],
        )

        save_tokens(first, {"access_token": "shared-token"})

        assert load_tokens(second)["access_token"] == "shared-token"
        assert _token_file(first) == _token_file(second)

    def test_httpx_distinct_idna_origins_never_share_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        unicode_host = self._identity("idna", "https://faß.de/")
        ascii_host = self._identity("idna", "https://fass.de/")

        save_tokens(unicode_host, {"access_token": "unicode-origin"})

        assert load_tokens(ascii_host) is None
        assert _token_file(unicode_host) != _token_file(ascii_host)

    def test_encoded_dot_segments_never_collide_with_normalized_route(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        encoded = self._identity("encoded-dot", "https://example.com/a/%2e%2e/mcp")
        normalized = self._identity("encoded-dot", "https://example.com/mcp")

        save_tokens(encoded, {"access_token": "encoded-route"})

        assert load_tokens(normalized) is None
        assert _token_file(encoded) != _token_file(normalized)

    def test_ipv6_and_port_identity_follow_httpx_conservatively(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        expanded = self._identity("ipv6", "https://[2001:0db8::1]:443/mcp")
        expanded_default_omitted = self._identity("ipv6", "https://[2001:0db8::1]/mcp")
        compressed = self._identity("ipv6", "https://[2001:db8::1]/mcp")
        nondefault_port = self._identity("ipv6", "https://[2001:0db8::1]:444/mcp")

        save_tokens(expanded, {"access_token": "expanded"})

        assert load_tokens(expanded_default_omitted)["access_token"] == "expanded"
        assert load_tokens(compressed) is None
        assert load_tokens(nondefault_port) is None

    def test_different_server_names_same_endpoint_do_not_share_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        first = self._identity("old-display-name", "https://example.com/mcp")
        renamed = self._identity("new-display-name", "https://example.com/mcp")

        save_tokens(first, {"access_token": "same-resource"})

        assert load_tokens(renamed) is None
        assert _token_file(first) != _token_file(renamed)

    def test_query_and_path_are_identity_sensitive_but_fragment_is_not(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        base = self._identity("routing", "https://example.com/mcp?tenant=one#first")
        same_without_fragment = self._identity(
            "routing", "https://example.com/mcp?tenant=one#second"
        )
        different_query = self._identity("routing", "https://example.com/mcp?tenant=two")
        different_path = self._identity("routing", "https://example.com/mcp/?tenant=one")

        save_tokens(base, {"access_token": "routed-token"})

        assert load_tokens(same_without_fragment)["access_token"] == "routed-token"
        assert load_tokens(different_query) is None
        assert load_tokens(different_path) is None

        ordered_query = self._identity("query-order", "https://example.com/mcp?a=1&b=2")
        reordered_query = self._identity("query-order", "https://example.com/mcp?b=2&a=1")
        save_tokens(ordered_query, {"access_token": "ordered"})
        # Query ordering is preserved conservatively because some endpoints
        # assign meaning to repeated or ordered parameters.
        assert load_tokens(reordered_query) is None

    def test_userinfo_and_client_dimensions_are_identity_sensitive(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        first = self._identity(
            "dimensions",
            "https://alice:first-secret@example.com/mcp",
            client_id="client-a",
            scopes=["read"],
        )
        different_userinfo = self._identity(
            "dimensions",
            "https://alice:second-secret@example.com/mcp",
            client_id="client-a",
            scopes=["read"],
        )
        different_client = self._identity(
            "dimensions",
            "https://alice:first-secret@example.com/mcp",
            client_id="client-b",
            scopes=["read"],
        )
        different_scopes = self._identity(
            "dimensions",
            "https://alice:first-secret@example.com/mcp",
            client_id="client-a",
            scopes=["write"],
        )

        save_tokens(first, {"access_token": "dimension-token"})

        assert load_tokens(different_userinfo) is None
        assert load_tokens(different_client) is None
        assert load_tokens(different_scopes) is None
        cache_path = str(_token_file(first))
        assert "alice" not in cache_path
        assert "first-secret" not in cache_path
        assert "example.com" not in cache_path

    def test_metadata_secret_and_callback_dimensions_are_identity_sensitive(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        base_config = MCPOAuthConfig(
            client_id="client-id",
            client_secret="secret-a",
            callback_port=7000,
            auth_server_metadata_url="https://auth.example/tenant-a/.well-known/oauth",
        )
        base = MCPOAuthFlow("dimensions", "https://example.com/mcp", base_config)
        different_metadata = MCPOAuthFlow(
            "dimensions",
            "https://example.com/mcp",
            MCPOAuthConfig(
                client_id="client-id",
                client_secret="secret-a",
                callback_port=7000,
                auth_server_metadata_url="https://auth.example/tenant-b/.well-known/oauth",
            ),
        )
        different_secret = MCPOAuthFlow(
            "dimensions",
            "https://example.com/mcp",
            MCPOAuthConfig(
                client_id="client-id",
                client_secret="secret-b",
                callback_port=7000,
                auth_server_metadata_url="https://auth.example/tenant-a/.well-known/oauth",
            ),
        )
        different_callback = MCPOAuthFlow(
            "dimensions",
            "https://example.com/mcp",
            MCPOAuthConfig(
                client_id="client-id",
                client_secret="secret-a",
                callback_port=7001,
                auth_server_metadata_url="https://auth.example/tenant-a/.well-known/oauth",
            ),
        )

        save_tokens(base.cache_identity, {"access_token": "dimension-token"})

        assert load_tokens(different_metadata.cache_identity) is None
        assert load_tokens(different_secret.cache_identity) is None
        assert load_tokens(different_callback.cache_identity) is None

    def test_legacy_name_only_entry_is_ignored_not_reused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        legacy_path = tmp_path / ".koder" / "mcp-auth" / "legacy" / "tokens.json"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text('{"access_token": "unbound-token"}', "utf-8")
        identity = self._identity("legacy", "https://new.example/mcp")

        assert load_tokens(identity) is None
        assert legacy_path.exists()

    def test_previous_name_less_cache_version_is_never_migrated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        first = self._identity("first-name", "https://example.com/mcp")
        second = self._identity("second-name", "https://example.com/mcp")
        old_payload = {
            "version": 3,
            "endpoint": first.endpoint.__dict__,
            "client_id": first.client_id,
            "scopes": first.scopes,
            "auth_server_metadata": None,
            "client_secret_fingerprint": first.client_secret_fingerprint,
            "callback_port": first.callback_port,
        }
        old_key = hashlib.sha256(
            json.dumps(old_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        old_path = tmp_path / ".koder" / "mcp-auth" / "v3" / old_key / "tokens.json"
        old_path.parent.mkdir(parents=True)
        old_path.write_text('{"access_token": "shared-old-token"}', "utf-8")

        assert load_tokens(first) is None
        assert load_tokens(second) is None
        assert old_path.exists()

    def test_clear_only_removes_matching_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        first = self._identity("clear-shared", "https://one.example/mcp")
        second = self._identity("clear-shared", "https://two.example/mcp")
        save_tokens(first, {"access_token": "one"})
        save_tokens(second, {"access_token": "two"})

        clear_tokens(first)

        assert load_tokens(first) is None
        assert load_tokens(second)["access_token"] == "two"

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("test-server", "https://example.com/mcp")
        tokens = {
            "access_token": "abc123",
            "refresh_token": "ref456",
            "expires_in": 3600,
            "obtained_at": time.time(),
        }
        save_tokens(identity, tokens)
        loaded = load_tokens(identity)
        assert loaded is not None
        assert loaded["access_token"] == "abc123"
        assert loaded["refresh_token"] == "ref456"

    def test_load_nonexistent_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("nonexistent", "https://example.com/mcp")
        assert load_tokens(identity) is None

    def test_clear_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("deleteme", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "x"})
        assert load_tokens(identity) is not None
        clear_tokens(identity)
        assert load_tokens(identity) is None

    def test_clear_nonexistent_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("never-saved", "https://example.com/mcp")
        clear_tokens(identity)  # should not raise

    def test_token_file_permissions(self, tmp_path, monkeypatch):
        if os.name != "posix":
            pytest.skip("POSIX mode verification requires a POSIX filesystem")
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("perm-test", "https://example.com/mcp")
        previous_umask = os.umask(0)
        try:
            save_tokens(identity, {"access_token": "secret"})
        finally:
            os.umask(previous_umask)
        token_path = _token_file(identity)
        assert token_path.exists()
        assert token_path.stat().st_mode & 0o777 == 0o600
        assert token_path.parent.stat().st_mode & 0o777 == 0o700
        assert token_path.parent.parent.stat().st_mode & 0o777 == 0o700
        assert token_path.parent.parent.parent.stat().st_mode & 0o777 == 0o700

    def test_secure_write_failure_leaves_no_secret_file(self, tmp_path, monkeypatch):
        if os.name != "posix":
            pytest.skip("descriptor chmod is part of the POSIX storage policy")
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("write-failure", "https://example.com/mcp")

        with patch("koder_agent.mcp.oauth.os.fchmod", side_effect=OSError("permission denied")):
            with pytest.raises(OSError, match="permission denied"):
                save_tokens(identity, {"access_token": "must-not-land"})

        token_dir = _token_file(identity).parent
        assert not _token_file(identity).exists()
        assert not token_dir.exists() or {path.name for path in token_dir.iterdir()} <= {
            "tokens.lock"
        }

    def test_posix_capability_path_uses_nofollow_and_descriptor_chmod(self, tmp_path, monkeypatch):
        if os.name != "posix":
            pytest.skip("POSIX mode verification requires a POSIX filesystem")
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("posix-policy", "https://example.com/mcp")
        capabilities = _SecureStorageCapabilities(
            enforce_posix_modes=True,
            nofollow_flag=os.O_NOFOLLOW,
            has_fchmod=True,
        )
        real_open = os.open
        real_fchmod = os.fchmod
        opened_fds = []

        def tracking_open(*args, **kwargs):
            fd = real_open(*args, **kwargs)
            opened_fds.append(fd)
            return fd

        with (
            patch(
                "koder_agent.mcp.oauth._secure_storage_capabilities",
                return_value=capabilities,
            ),
            patch("koder_agent.mcp.oauth.os.open", side_effect=tracking_open) as open_mock,
            patch("koder_agent.mcp.oauth.os.fchmod", wraps=real_fchmod) as fchmod_mock,
        ):
            save_tokens(identity, {"access_token": "secret"})

        assert open_mock.call_args.args[1] & os.O_NOFOLLOW
        assert opened_fds
        assert any(call.args[1] == 0o700 for call in fchmod_mock.call_args_list)
        assert any(call.args[1] == 0o600 for call in fchmod_mock.call_args_list)
        assert _token_file(identity).stat().st_mode & 0o777 == 0o600

    @pytest.mark.parametrize(
        "capabilities",
        [
            _SecureStorageCapabilities(
                enforce_posix_modes=True,
                nofollow_flag=0,
                has_fchmod=True,
            ),
            _SecureStorageCapabilities(
                enforce_posix_modes=True,
                nofollow_flag=1,
                has_fchmod=False,
            ),
        ],
    )
    def test_posix_capability_path_never_silently_weakens(self, capabilities):
        with pytest.raises(RuntimeError, match="security controls are unavailable"):
            _validate_secure_storage_capabilities(capabilities)

    def test_non_posix_capability_path_encrypts_with_injected_user_protection(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("non-posix-policy", "https://example.com/mcp")
        protection_calls = []

        def protect(data: bytes, entropy: bytes) -> bytes:
            protection_calls.append(("protect", entropy))
            return bytes(value ^ entropy[index % len(entropy)] for index, value in enumerate(data))

        def unprotect(data: bytes, entropy: bytes) -> bytes:
            protection_calls.append(("unprotect", entropy))
            return bytes(value ^ entropy[index % len(entropy)] for index, value in enumerate(data))

        capabilities = _SecureStorageCapabilities(
            enforce_posix_modes=False,
            nofollow_flag=0,
            has_fchmod=False,
            data_protector=_DataProtector(protect=protect, unprotect=unprotect),
        )
        real_open = os.open
        real_replace = os.replace

        with (
            patch(
                "koder_agent.mcp.oauth._secure_storage_capabilities",
                return_value=capabilities,
            ),
            patch("koder_agent.mcp.oauth.os.open", wraps=real_open) as open_mock,
            patch("koder_agent.mcp.oauth.os.replace", wraps=real_replace) as replace_mock,
            patch(
                "koder_agent.mcp.oauth.os.chmod",
                side_effect=AssertionError("non-POSIX policy called chmod"),
            ),
            patch(
                "koder_agent.mcp.oauth.os.fchmod",
                side_effect=AssertionError("non-POSIX policy called fchmod"),
            ),
        ):
            save_tokens(identity, {"access_token": "portable"})

        token_path = _token_file(identity)
        assert open_mock.call_args.args[1] & getattr(os, "O_NOFOLLOW", 0) == 0
        replace_mock.assert_called_once()
        stored = token_path.read_bytes()
        assert b"portable" not in stored
        assert b"access_token" not in stored
        os.chmod(token_path, 0o644)
        with patch(
            "koder_agent.mcp.oauth._secure_storage_capabilities",
            return_value=capabilities,
        ):
            assert load_tokens(identity)["access_token"] == "portable"
        assert [name for name, _ in protection_calls] == ["protect", "unprotect"]
        assert protection_calls[0][1] == protection_calls[1][1]

    def test_non_posix_capability_path_fails_closed_without_protection(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("non-posix-unprotected", "https://example.com/mcp")
        capabilities = _SecureStorageCapabilities(
            enforce_posix_modes=False,
            nofollow_flag=0,
            has_fchmod=False,
        )

        with patch(
            "koder_agent.mcp.oauth._secure_storage_capabilities",
            return_value=capabilities,
        ):
            with pytest.raises(RuntimeError, match="at-rest protection is unavailable"):
                save_tokens(identity, {"access_token": "must-not-land"})

        assert not _token_file(identity).exists()

    def test_platform_capability_helper_matches_host(self):
        capabilities = _secure_storage_capabilities()

        assert capabilities.enforce_posix_modes is (os.name == "posix")
        if os.name == "posix":
            assert capabilities.nofollow_flag == getattr(os, "O_NOFOLLOW", 0)
            assert capabilities.has_fchmod is hasattr(os, "fchmod")
        else:
            assert capabilities.nofollow_flag == 0
            assert capabilities.has_fchmod is False
            assert capabilities.data_protector is not None

    def test_repeated_eight_process_first_write_is_lossless(self, tmp_path, monkeypatch):
        if os.name != "posix":
            pytest.skip("reviewer multiprocessing repro uses POSIX advisory locks")
        monkeypatch.setenv("HOME", str(tmp_path))
        context = multiprocessing.get_context("spawn")

        for trial in range(5):
            start_event = context.Event()
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=_first_write_worker,
                    args=(str(tmp_path), trial, f"writer-{index}", start_event, result_queue),
                )
                for index in range(8)
            ]
            for process in processes:
                process.start()
            start_event.set()
            results = [result_queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(30)

            assert [process.exitcode for process in processes] == [0] * 8
            assert [error for _, error in results if error] == []
            identity = self._identity(
                f"first-write-{trial}",
                "https://example.com/mcp",
            )
            cached = load_tokens(identity)
            assert cached is not None
            assert {key for key in cached if key.startswith("writer-")} == {
                f"writer-{index}" for index in range(8)
            }

    @pytest.mark.skipif(os.name != "posix", reason="hardlink repro requires POSIX links")
    @pytest.mark.parametrize("target_name", ["tokens.json", "tokens.lock"])
    def test_hardlinked_token_and_lock_files_are_rejected(self, tmp_path, monkeypatch, target_name):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity(f"hardlink-{target_name}", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "trusted"})
        target = _token_file(identity).parent / target_name
        outside = tmp_path / f"outside-{target_name}"
        os.link(target, outside)
        assert target.stat().st_nlink == 2

        if target_name == "tokens.json":
            assert load_tokens(identity) is None
        else:
            with pytest.raises(OSError, match="single-link"):
                save_tokens(identity, {"access_token": "replacement"})

    @pytest.mark.skipif(os.name != "posix", reason="hardlink repro requires POSIX links")
    def test_hardlinked_stale_temp_is_rejected_not_swept(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("hardlink-temp", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "trusted"})
        temp_path = _token_file(identity).parent / f".tokens.json.{'a' * 32}.tmp"
        temp_path.write_text('{"access_token":"attacker"}', "utf-8")
        os.chmod(temp_path, 0o600)
        outside = tmp_path / "outside-temp"
        os.link(temp_path, outside)

        with pytest.raises(OSError, match="single-link"):
            save_tokens(identity, {"access_token": "replacement"})

        assert temp_path.exists()
        assert outside.exists()

    def test_crash_before_replace_then_clear_sweeps_only_owned_stale_temp(
        self, tmp_path, monkeypatch
    ):
        if os.name != "posix":
            pytest.skip("crash repro uses POSIX process exit and file locking")
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("crash-before-replace", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "committed"})
        unrelated = _token_file(identity).parent / "unrelated.tmp"
        unrelated.write_text("do not delete", "utf-8")

        context = multiprocessing.get_context("spawn")
        process = context.Process(target=_crash_before_replace_worker, args=(str(tmp_path),))
        process.start()
        process.join(30)

        assert process.exitcode == 74
        stale_temps = [
            path
            for path in _token_file(identity).parent.iterdir()
            if path.name.startswith(".tokens.json.") and path.name.endswith(".tmp")
        ]
        assert len(stale_temps) == 1
        assert b"crash-access" in stale_temps[0].read_bytes()
        assert b"crash-refresh" in stale_temps[0].read_bytes()

        clear_tokens(identity)

        assert not _token_file(identity).exists()
        assert not stale_temps[0].exists()
        assert unrelated.read_text("utf-8") == "do not delete"

    def test_atomic_replace_failure_preserves_previous_tokens(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("replace-failure", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "old"})

        with patch("os.replace", side_effect=OSError("replace failed")):
            with pytest.raises(OSError, match="replace failed"):
                save_tokens(identity, {"access_token": "new"})

        assert load_tokens(identity)["access_token"] == "old"
        assert sorted(path.name for path in _token_file(identity).parent.iterdir()) == [
            "tokens.json",
            "tokens.lock",
        ]

    def test_load_reads_opened_regular_file_when_final_path_is_swapped(self, tmp_path, monkeypatch):
        if os.name != "posix":
            pytest.skip("fd-anchored no-follow load is the POSIX storage path")
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("load-swap", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "trusted-token"})
        token_path = _token_file(identity)
        attacker_path = tmp_path / "attacker.json"
        attacker_path.write_text('{"access_token": "attacker-token"}', "utf-8")
        real_open = os.open
        swapped = False

        def swap_after_open(path, flags, *args, **kwargs):
            nonlocal swapped
            fd = real_open(path, flags, *args, **kwargs)
            if path == "tokens.json" and not swapped:
                swapped = True
                os.replace(attacker_path, token_path)
            return fd

        with patch("koder_agent.mcp.oauth.os.open", side_effect=swap_after_open):
            loaded = load_tokens(identity)

        assert swapped
        assert loaded["access_token"] == "trusted-token"
        assert json.loads(token_path.read_text("utf-8"))["access_token"] == "attacker-token"

    def test_parent_swap_during_write_cannot_escape_verified_directory(self, tmp_path, monkeypatch):
        if os.name != "posix":
            pytest.skip("dir-fd atomic replacement is the POSIX storage path")
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("parent-swap", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "old-token"})
        token_dir = _token_file(identity).parent
        moved_dir = token_dir.with_name(f"{token_dir.name}-pinned")
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        real_replace = os.replace
        swapped = False

        def swap_parent_before_replace(src, dst, *args, **kwargs):
            nonlocal swapped
            if kwargs.get("src_dir_fd") is not None and not swapped:
                swapped = True
                os.rename(token_dir, moved_dir)
                token_dir.symlink_to(outside_dir, target_is_directory=True)
            return real_replace(src, dst, *args, **kwargs)

        with patch(
            "koder_agent.mcp.oauth.os.replace",
            side_effect=swap_parent_before_replace,
        ):
            save_tokens(identity, {"access_token": "new-token"})

        assert swapped
        assert not (outside_dir / "tokens.json").exists()
        assert (
            json.loads((moved_dir / "tokens.json").read_text("utf-8"))["access_token"]
            == "new-token"
        )

    def test_token_dir_under_koder(self, tmp_path, monkeypatch):
        """Tokens must live under ~/.koder/mcp-auth/."""
        monkeypatch.setenv("HOME", str(tmp_path))
        identity = self._identity("dir-check", "https://example.com/mcp")
        save_tokens(identity, {"access_token": "t"})
        token_path = _token_file(identity)
        assert token_path.exists()
        assert token_path.parent.parent.name == "v4"
        assert len(token_path.parent.name) == 64
        assert set(token_path.parent.name) <= set("0123456789abcdef")
        assert sorted(path.name for path in tmp_path.iterdir()) == [".koder"]


# ---------------------------------------------------------------------------
# Callback server
# ---------------------------------------------------------------------------


class TestCallbackServer:
    def test_start_callback_server_picks_free_port(self):
        server, port = _start_callback_server(None)
        try:
            assert port > 0
        finally:
            _stop_callback_server(server)

    def test_start_callback_server_uses_specified_port(self):
        server, port = _start_callback_server(0)
        try:
            assert port > 0
        finally:
            _stop_callback_server(server)


# ---------------------------------------------------------------------------
# MCPOAuthFlow._is_expired
# ---------------------------------------------------------------------------


class TestTokenExpiry:
    def test_not_expired(self):
        tokens = {
            "access_token": "x",
            "expires_in": 3600,
            "obtained_at": time.time(),
        }
        assert MCPOAuthFlow._is_expired(tokens) is False

    def test_expired(self):
        tokens = {
            "access_token": "x",
            "expires_in": 3600,
            "obtained_at": time.time() - 4000,
        }
        assert MCPOAuthFlow._is_expired(tokens) is True

    def test_no_expiry_info_treated_as_expired(self):
        # A token that carries no usable expiry metadata must not be assumed
        # valid forever; it is treated as expired so the caller re-validates
        # (refresh when possible, else a fresh flow).
        tokens = {"access_token": "x"}
        assert MCPOAuthFlow._is_expired(tokens) is True

    def test_missing_obtained_at_treated_as_expired(self):
        tokens = {"access_token": "x", "expires_in": 3600}
        assert MCPOAuthFlow._is_expired(tokens) is True

    def test_missing_expires_in_treated_as_expired(self):
        tokens = {"access_token": "x", "obtained_at": time.time()}
        assert MCPOAuthFlow._is_expired(tokens) is True


# ---------------------------------------------------------------------------
# MCPOAuthFlow.authenticate — cached token path
# ---------------------------------------------------------------------------


class TestAuthenticateCachedTokens:
    def test_returns_cached_valid_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = MCPOAuthConfig(client_id="cid")
        flow = MCPOAuthFlow("cached-server", "https://example.com/mcp", config)
        _save_bound(
            flow,
            {
                "access_token": "cached-token",
                "expires_in": 3600,
                "obtained_at": time.time(),
            },
        )
        with patch.object(
            flow,
            "_discover_metadata",
            new_callable=AsyncMock,
            return_value=AUTH_METADATA,
        ):
            headers = asyncio.run(flow.authenticate())
        assert headers == {"Authorization": "Bearer cached-token"}

    def test_expired_token_triggers_refresh(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = MCPOAuthConfig(client_id="cid")
        flow = MCPOAuthFlow("expired-server", "https://example.com/mcp", config)
        _save_bound(
            flow,
            {
                "access_token": "old-token",
                "refresh_token": "ref-tok",
                "expires_in": 3600,
                "obtained_at": time.time() - 4000,
                "client_id": "cid",
            },
        )

        with (
            patch.object(
                flow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=AUTH_METADATA,
            ),
            patch.object(flow, "refresh_token", new_callable=AsyncMock) as mock_refresh,
        ):
            mock_refresh.return_value = {"Authorization": "Bearer refreshed-token"}
            headers = asyncio.run(flow.authenticate())
            mock_refresh.assert_awaited_once()
            assert headers == {"Authorization": "Bearer refreshed-token"}

    def test_cached_token_without_expiry_refreshes_when_possible(self, tmp_path, monkeypatch):
        # A cached token that lacks expiry metadata but has a refresh_token
        # must be refreshed rather than assumed valid forever.
        monkeypatch.setenv("HOME", str(tmp_path))
        config = MCPOAuthConfig(client_id="cid")
        flow = MCPOAuthFlow("no-expiry-refresh", "https://example.com/mcp", config)
        _save_bound(
            flow,
            {
                "access_token": "stale-token",
                "refresh_token": "ref-tok",
                "client_id": "cid",
            },
        )

        with (
            patch.object(
                flow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=AUTH_METADATA,
            ),
            patch.object(flow, "refresh_token", new_callable=AsyncMock) as mock_refresh,
        ):
            mock_refresh.return_value = {"Authorization": "Bearer fresh-token"}
            headers = asyncio.run(flow.authenticate())
            mock_refresh.assert_awaited_once()
            assert headers == {"Authorization": "Bearer fresh-token"}

    def test_cached_token_without_expiry_no_refresh_starts_new_flow(self, tmp_path, monkeypatch):
        # No expiry AND no refresh_token: the stale token must NOT be returned;
        # instead the full flow is (attempted to be) started.
        monkeypatch.setenv("HOME", str(tmp_path))
        config = MCPOAuthConfig(client_id="cid")
        flow = MCPOAuthFlow("no-expiry-noref", "https://example.com/mcp", config)
        _save_bound(
            flow,
            {"access_token": "stale-token", "client_id": "cid"},
        )

        with patch.object(flow, "_discover_metadata", new_callable=AsyncMock) as mock_discover:
            mock_discover.side_effect = RuntimeError("new flow started")
            # The stale token must not be returned; a new flow is attempted.
            try:
                asyncio.run(flow.authenticate())
            except RuntimeError as exc:
                assert "new flow started" in str(exc)
            else:  # pragma: no cover - defensive
                raise AssertionError("stale token should not have been returned")
            mock_discover.assert_awaited_once()

    def test_different_endpoint_does_not_attempt_refresh_with_cached_token(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = MCPOAuthConfig(client_id="cid")
        original = MCPOAuthFlow("refresh-shared", "https://one.example/mcp", config)
        replacement = MCPOAuthFlow("refresh-shared", "https://two.example/mcp", config)
        save_tokens(
            original.cache_identity,
            {
                "access_token": "old-token",
                "refresh_token": "old-refresh-token",
                "expires_in": 3600,
                "obtained_at": time.time() - 4000,
                "client_id": "cid",
            },
        )

        with (
            patch.object(
                replacement, "_discover_metadata", new_callable=AsyncMock
            ) as mock_discover,
            patch.object(replacement, "refresh_token", new_callable=AsyncMock) as mock_refresh,
        ):
            mock_discover.side_effect = RuntimeError("fresh authorization required")
            try:
                asyncio.run(replacement.authenticate())
            except RuntimeError as exc:
                assert "fresh authorization required" in str(exc)
            else:  # pragma: no cover - defensive
                raise AssertionError("replacement endpoint should start a new OAuth flow")

        mock_discover.assert_awaited_once()
        mock_refresh.assert_not_awaited()

    def test_authorization_server_change_never_reuses_access_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        old_metadata = {
            "issuer": "https://auth-a.example/",
            "authorization_endpoint": "https://auth-a.example/authorize",
            "token_endpoint": "https://auth-a.example/token",
        }
        new_metadata = {
            "issuer": "https://auth-b.example/",
            "authorization_endpoint": "https://auth-b.example/authorize",
            "token_endpoint": "https://auth-b.example/token",
        }
        flow = MCPOAuthFlow("delegated", "https://mcp.example/mcp", MCPOAuthConfig(client_id="cid"))
        _save_bound(
            flow,
            {
                "access_token": "old-access-token",
                "expires_in": 3600,
                "obtained_at": time.time(),
            },
            old_metadata,
        )
        callback_server = MagicMock()

        with (
            patch.object(
                flow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=new_metadata,
            ),
            patch(
                "koder_agent.mcp.oauth._start_callback_server",
                return_value=(callback_server, 54321),
            ),
            patch.object(
                flow,
                "_authorization_code_flow",
                new_callable=AsyncMock,
                return_value={
                    "access_token": "new-access-token",
                    "expires_in": 3600,
                    "obtained_at": time.time(),
                },
            ) as code_flow,
        ):
            headers = asyncio.run(flow.authenticate())

        assert headers == {"Authorization": "Bearer new-access-token"}
        code_flow.assert_awaited_once()
        callback_server.shutdown.assert_called_once()
        cached = load_tokens(flow.cache_identity)
        assert cached["access_token"] == "new-access-token"
        assert cached["_oauth_server_binding"] == flow._server_binding(new_metadata).as_dict()

    def test_unbound_cache_is_never_reused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = MCPOAuthFlow("unbound", "https://mcp.example/mcp", MCPOAuthConfig(client_id="cid"))
        save_tokens(
            flow.cache_identity,
            {
                "access_token": "legacy-unbound-token",
                "expires_in": 3600,
                "obtained_at": time.time(),
            },
        )
        callback_server = MagicMock()

        with (
            patch.object(
                flow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=AUTH_METADATA,
            ),
            patch(
                "koder_agent.mcp.oauth._start_callback_server",
                return_value=(callback_server, 54321),
            ),
            patch.object(
                flow,
                "_authorization_code_flow",
                new_callable=AsyncMock,
                return_value={
                    "access_token": "new-bound-token",
                    "expires_in": 3600,
                    "obtained_at": time.time(),
                },
            ),
        ):
            headers = asyncio.run(flow.authenticate())

        assert headers == {"Authorization": "Bearer new-bound-token"}
        assert load_tokens(flow.cache_identity)["access_token"] == "new-bound-token"


class TestRefreshTokenCacheIdentity:
    def test_refresh_reads_and_writes_only_matching_identity(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = MCPOAuthConfig(client_id="cid")
        flow = MCPOAuthFlow("refresh-save", "https://one.example/mcp", config)
        other = MCPOAuthFlow("refresh-save", "https://two.example/mcp", config)
        _save_bound(
            flow,
            {
                "access_token": "old-token",
                "refresh_token": "refresh-token",
                "expires_in": 3600,
                "obtained_at": time.time() - 4000,
                "client_id": "cid",
            },
        )

        with (
            patch.object(
                flow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=AUTH_METADATA,
            ),
            patch("httpx.AsyncClient") as mock_cls,
        ):
            response = MagicMock()
            response.json.return_value = {"access_token": "new-token", "expires_in": 7200}
            response.raise_for_status = MagicMock()
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            headers = asyncio.run(flow.refresh_token())

        assert headers == {"Authorization": "Bearer new-token"}
        refreshed = load_tokens(flow.cache_identity)
        assert refreshed["access_token"] == "new-token"
        assert refreshed["refresh_token"] == "refresh-token"
        assert load_tokens(other.cache_identity) is None

    def test_authorization_server_change_never_sends_old_refresh_credentials(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        old_metadata = {
            "issuer": "https://auth-a.example/",
            "authorization_endpoint": "https://auth-a.example/authorize",
            "token_endpoint": "https://auth-a.example/token",
        }
        new_metadata = {
            "issuer": "https://auth-b.example/",
            "authorization_endpoint": "https://auth-b.example/authorize",
            "token_endpoint": "https://auth-b.example/token",
        }
        flow = MCPOAuthFlow(
            "refresh-change",
            "https://mcp.example/mcp",
            MCPOAuthConfig(client_id="old-client", client_secret="old-secret"),
        )
        _save_bound(
            flow,
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "client_id": "old-client",
                "client_secret": "old-secret",
            },
            old_metadata,
        )

        with (
            patch.object(
                flow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=new_metadata,
            ),
            patch("httpx.AsyncClient") as mock_client,
        ):
            with pytest.raises(RuntimeError, match="authorization server changed"):
                asyncio.run(flow.refresh_token())

        mock_client.assert_not_called()
        assert load_tokens(flow.cache_identity) is None

    def test_refresh_http_error_redacts_endpoint_userinfo_and_query(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        secret_endpoint = "https://alice:token-password@auth.example/token?credential=query-secret"
        metadata = {
            "issuer": "https://auth.example/",
            "authorization_endpoint": "https://auth.example/authorize",
            "token_endpoint": secret_endpoint,
        }
        flow = MCPOAuthFlow(
            "refresh-redaction",
            "https://mcp.example/mcp",
            MCPOAuthConfig(client_id="cid"),
        )
        _save_bound(
            flow,
            {"refresh_token": "refresh-secret", "client_id": "cid"},
            metadata,
        )
        request = httpx.Request("POST", secret_endpoint)
        response = httpx.Response(400, request=request)
        http_error = httpx.HTTPStatusError(
            f"request failed for {secret_endpoint}",
            request=request,
            response=response,
        )

        with (
            patch.object(
                flow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=metadata,
            ),
            patch("httpx.AsyncClient") as mock_cls,
        ):
            mock_client = AsyncMock()
            mocked_response = MagicMock()
            mocked_response.raise_for_status.side_effect = http_error
            mock_client.post = AsyncMock(return_value=mocked_response)
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(flow.refresh_token())

        error = str(exc_info.value)
        assert error == "OAuth token refresh request failed"
        assert "alice" not in error
        assert "token-password" not in error
        assert "query-secret" not in error

    def test_clear_during_refresh_discards_response_without_recreating_tokens(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = MCPOAuthFlow(
            "clear-during-refresh",
            "https://example.com/mcp",
            MCPOAuthConfig(client_id="client-id"),
        )
        _save_bound(
            flow,
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "client_id": "client-id",
            },
        )
        post_started = asyncio.Event()
        release_response = asyncio.Event()

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, data=None):
                assert data["refresh_token"] == "old-refresh"
                post_started.set()
                await release_response.wait()
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "access_token": "stale-access",
                    "refresh_token": "stale-refresh",
                }
                return response

        async def exercise_race():
            binding = flow._server_binding(AUTH_METADATA)
            refresh = asyncio.create_task(
                flow.refresh_token(metadata=AUTH_METADATA, binding=binding)
            )
            await asyncio.wait_for(post_started.wait(), timeout=10)
            clear_tokens(flow.cache_identity)
            release_response.set()
            with pytest.raises(RuntimeError, match="refresh response is stale"):
                await refresh

        with patch("httpx.AsyncClient", FakeClient):
            asyncio.run(exercise_race())

        assert load_tokens(flow.cache_identity) is None

    def test_binding_switch_during_refresh_preserves_new_binding_and_tokens(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        old_metadata = {
            "issuer": "https://auth-old.example/",
            "authorization_endpoint": "https://auth-old.example/authorize",
            "token_endpoint": "https://auth-old.example/token",
        }
        new_metadata = {
            "issuer": "https://auth-new.example/",
            "authorization_endpoint": "https://auth-new.example/authorize",
            "token_endpoint": "https://auth-new.example/token",
        }
        flow = MCPOAuthFlow(
            "binding-switch-during-refresh",
            "https://example.com/mcp",
            MCPOAuthConfig(client_id="client-id"),
        )
        _save_bound(
            flow,
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "client_id": "client-id",
            },
            old_metadata,
        )
        post_started = asyncio.Event()
        release_response = asyncio.Event()

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, data=None):
                assert url == old_metadata["token_endpoint"]
                assert data["refresh_token"] == "old-refresh"
                post_started.set()
                await release_response.wait()
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "access_token": "stale-access",
                    "refresh_token": "stale-refresh",
                }
                return response

        async def exercise_race():
            old_binding = flow._server_binding(old_metadata)
            refresh = asyncio.create_task(
                flow.refresh_token(metadata=old_metadata, binding=old_binding)
            )
            await asyncio.wait_for(post_started.wait(), timeout=10)
            _save_bound(
                flow,
                {
                    "access_token": "new-access",
                    "refresh_token": "new-refresh",
                    "client_id": "new-client-id",
                },
                new_metadata,
            )
            release_response.set()
            with pytest.raises(RuntimeError, match="refresh response is stale"):
                await refresh

        with patch("httpx.AsyncClient", FakeClient):
            asyncio.run(exercise_race())

        cached = load_tokens(flow.cache_identity)
        assert cached["access_token"] == "new-access"
        assert cached["refresh_token"] == "new-refresh"
        assert cached["_oauth_server_binding"] == flow._server_binding(new_metadata).as_dict()

    @pytest.mark.parametrize("entrypoint", ["authenticate", "resolve_oauth_headers"])
    @pytest.mark.parametrize("mutation", ["clear", "binding-switch", "full-record"])
    def test_public_authentication_aborts_when_refresh_generation_changes(
        self,
        tmp_path,
        monkeypatch,
        entrypoint,
        mutation,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        old_metadata = {
            "issuer": "https://auth-old.example/",
            "authorization_endpoint": "https://auth-old.example/authorize",
            "token_endpoint": "https://auth-old.example/token",
        }
        new_metadata = {
            "issuer": "https://auth-new.example/",
            "authorization_endpoint": "https://auth-new.example/authorize",
            "token_endpoint": "https://auth-new.example/token",
        }
        server_name = f"public-refresh-race-{entrypoint}-{mutation}"
        server_url = "https://example.com/mcp"
        oauth_dict = {"clientId": "client-id"}
        flow = MCPOAuthFlow(
            server_name,
            server_url,
            MCPOAuthConfig(client_id="client-id"),
        )
        old_record = {
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "client_id": "client-id",
        }
        _save_bound(flow, old_record, old_metadata)
        post_started = asyncio.Event()
        release_response = asyncio.Event()
        callback_server = MagicMock()

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, data=None):
                assert url == old_metadata["token_endpoint"]
                assert data["refresh_token"] == "old-refresh"
                post_started.set()
                await release_response.wait()
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "access_token": "stale-refresh-access",
                    "refresh_token": "stale-refresh-token",
                }
                return response

        if mutation == "clear":
            expected_record = None

            def mutate_record():
                clear_tokens(flow.cache_identity)

        elif mutation == "binding-switch":
            expected_record = {
                "access_token": "new-binding-access",
                "refresh_token": "new-binding-refresh",
                "client_id": "new-binding-client",
            }

            def mutate_record():
                _save_bound(flow, expected_record, new_metadata)

        else:
            expected_record = {
                **old_record,
                "concurrent_generation_marker": "replacement",
            }

            def mutate_record():
                _save_bound(flow, expected_record, old_metadata)

        async def exercise_race():
            if entrypoint == "authenticate":
                authentication = flow.authenticate()
            else:
                authentication = resolve_oauth_headers(
                    server_name,
                    server_url,
                    oauth_dict,
                )
            task = asyncio.create_task(authentication)
            await asyncio.wait_for(post_started.wait(), timeout=10)
            mutate_record()
            release_response.set()
            with pytest.raises(
                oauth_module._OAuthStaleStateError,
                match="refresh response is stale",
            ):
                await task

        with (
            patch.object(
                MCPOAuthFlow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=old_metadata,
            ),
            patch("httpx.AsyncClient", FakeClient),
            patch(
                "koder_agent.mcp.oauth._start_callback_server",
                return_value=(callback_server, 54321),
            ) as start_callback,
            patch.object(
                MCPOAuthFlow,
                "_ensure_client",
                new_callable=AsyncMock,
                return_value=("stale-client", None),
            ) as ensure_client,
            patch.object(
                MCPOAuthFlow,
                "_authorization_code_flow",
                new_callable=AsyncMock,
                return_value={
                    "access_token": "authorization-code-fallback",
                    "refresh_token": "authorization-code-refresh",
                },
            ) as authorization_code_flow,
        ):
            asyncio.run(exercise_race())

        start_callback.assert_not_called()
        ensure_client.assert_not_awaited()
        authorization_code_flow.assert_not_awaited()
        callback_server.shutdown.assert_not_called()
        cached = load_tokens(flow.cache_identity)
        if expected_record is None:
            assert cached is None
        else:
            assert cached == _bound_tokens(
                expected_record,
                flow._server_binding(
                    new_metadata if mutation == "binding-switch" else old_metadata
                ),
            )
            assert cached["access_token"] != "stale-refresh-access"
            assert cached["access_token"] != "authorization-code-fallback"
            if mutation == "binding-switch":
                assert (
                    cached["_oauth_server_binding"] != flow._server_binding(old_metadata).as_dict()
                )

    @pytest.mark.parametrize("entrypoint", ["authenticate", "resolve_oauth_headers"])
    @pytest.mark.parametrize("mutation", ["clear", "binding-switch"])
    def test_public_authentication_aborts_when_refresh_record_changes_while_waiting_for_lock(
        self,
        tmp_path,
        monkeypatch,
        entrypoint,
        mutation,
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        old_metadata = {
            "issuer": "https://auth-old.example/",
            "authorization_endpoint": "https://auth-old.example/authorize",
            "token_endpoint": "https://auth-old.example/token",
        }
        new_metadata = {
            "issuer": "https://auth-new.example/",
            "authorization_endpoint": "https://auth-new.example/authorize",
            "token_endpoint": "https://auth-new.example/token",
        }
        server_name = f"public-refresh-lock-wait-{entrypoint}-{mutation}"
        server_url = "https://example.com/mcp"
        oauth_dict = {"clientId": "client-id"}
        flow = MCPOAuthFlow(
            server_name,
            server_url,
            MCPOAuthConfig(client_id="client-id"),
        )
        _save_bound(
            flow,
            {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "client_id": "client-id",
            },
            old_metadata,
        )
        waiter_observed_busy_lock = threading.Event()
        original_lock_file = oauth_module._lock_file

        def observed_lock_file(fd, *, blocking=True):
            acquired = original_lock_file(fd, blocking=blocking)
            if not blocking and not acquired:
                waiter_observed_busy_lock.set()
            return acquired

        monkeypatch.setattr(oauth_module, "_lock_file", observed_lock_file)

        if mutation == "clear":
            expected_record = None

            def mutate_record():
                clear_tokens(flow.cache_identity)

        else:
            replacement = {
                "access_token": "new-binding-access",
                "refresh_token": "new-binding-refresh",
                "client_id": "new-binding-client",
            }
            expected_record = _bound_tokens(
                replacement,
                flow._server_binding(new_metadata),
            )

            def mutate_record():
                _save_bound(flow, replacement, new_metadata)

        async def exercise_race():
            task = None
            try:
                async with oauth_module._locked_refresh_grant(flow.cache_identity):
                    if entrypoint == "authenticate":
                        authentication = flow.authenticate()
                    else:
                        authentication = resolve_oauth_headers(
                            server_name,
                            server_url,
                            oauth_dict,
                        )
                    task = asyncio.create_task(authentication)
                    assert await asyncio.to_thread(waiter_observed_busy_lock.wait, 10)
                    mutate_record()

                with pytest.raises(
                    oauth_module._OAuthStaleStateError,
                    match="refresh state is stale",
                ):
                    await asyncio.wait_for(task, timeout=10)
            finally:
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        callback_server = MagicMock()
        with (
            patch.object(
                MCPOAuthFlow,
                "_discover_metadata",
                new_callable=AsyncMock,
                return_value=old_metadata,
            ),
            patch("koder_agent.mcp.oauth.httpx.AsyncClient") as client_factory,
            patch(
                "koder_agent.mcp.oauth._start_callback_server",
                return_value=(callback_server, 54321),
            ) as start_callback,
            patch.object(
                MCPOAuthFlow,
                "_ensure_client",
                new_callable=AsyncMock,
                return_value=("stale-client", None),
            ) as ensure_client,
            patch.object(
                MCPOAuthFlow,
                "_authorization_code_flow",
                new_callable=AsyncMock,
                return_value={
                    "access_token": "authorization-code-fallback",
                    "refresh_token": "authorization-code-refresh",
                },
            ) as authorization_code_flow,
            patch("koder_agent.mcp.oauth.webbrowser.open") as browser_open,
        ):
            asyncio.run(exercise_race())

        client_factory.assert_not_called()
        start_callback.assert_not_called()
        ensure_client.assert_not_awaited()
        authorization_code_flow.assert_not_awaited()
        browser_open.assert_not_called()
        callback_server.shutdown.assert_not_called()
        assert load_tokens(flow.cache_identity) == expected_record

    def test_two_process_rotating_refreshes_cannot_persist_stale_response(
        self, tmp_path, monkeypatch
    ):
        if os.name != "posix":
            pytest.skip("reviewer multiprocessing repro uses POSIX advisory locks")
        monkeypatch.setenv("HOME", str(tmp_path))
        first_request_received = threading.Event()
        release_first_response = threading.Event()
        request_tokens = []
        request_guard = threading.Lock()

        class RefreshHandler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                payload = parse_qs(self.rfile.read(length).decode("utf-8"))
                refresh_token = payload["refresh_token"][0]
                with request_guard:
                    request_index = len(request_tokens)
                    request_tokens.append(refresh_token)
                if request_index == 0:
                    first_request_received.set()
                    if not release_first_response.wait(20):
                        self.send_error(500)
                        return
                    response = {
                        "access_token": "access-r1",
                        "refresh_token": "refresh-r1",
                    }
                elif refresh_token == "refresh-r0":
                    # This is the stale-ordering response that an unserialized
                    # second process can persist before request zero finishes.
                    response = {
                        "access_token": "access-r2",
                        "refresh_token": "refresh-r2",
                    }
                else:
                    assert refresh_token == "refresh-r1"
                    response = {
                        "access_token": "access-r2",
                        "refresh_token": "refresh-r2",
                    }
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), RefreshHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.server_address[1]
        metadata = {
            "issuer": f"http://127.0.0.1:{port}/",
            "authorization_endpoint": f"http://127.0.0.1:{port}/authorize",
            "token_endpoint": f"http://127.0.0.1:{port}/token",
        }
        flow = MCPOAuthFlow(
            "rotating-refresh",
            "https://example.com/mcp",
            MCPOAuthConfig(client_id="client-id"),
        )
        _save_bound(
            flow,
            {
                "access_token": "access-r0",
                "refresh_token": "refresh-r0",
                "client_id": "client-id",
            },
            metadata,
        )
        context = multiprocessing.get_context("spawn")
        start_first = context.Event()
        start_second = context.Event()
        result_queue = context.Queue()
        first = context.Process(
            target=_refresh_worker,
            args=(str(tmp_path), metadata, start_first, result_queue),
        )
        second = context.Process(
            target=_refresh_worker,
            args=(str(tmp_path), metadata, start_second, result_queue),
        )
        try:
            first.start()
            start_first.set()
            assert first_request_received.wait(20)
            second.start()
            start_second.set()
            # Give an unprotected implementation enough time to issue and
            # persist its second stale-token request before request zero ends.
            time.sleep(0.5)
            release_first_response.set()
            results = [result_queue.get(timeout=30) for _ in range(2)]
            first.join(30)
            second.join(30)
        finally:
            release_first_response.set()
            server.shutdown()
            server.server_close()
            server_thread.join(10)

        assert [first.exitcode, second.exitcode] == [0, 0]
        assert [error for _, error in results if error] == []
        assert request_tokens == ["refresh-r0", "refresh-r1"]
        cached = load_tokens(flow.cache_identity)
        assert cached["access_token"] == "access-r2"
        assert cached["refresh_token"] == "refresh-r2"

    def test_refresh_cancellation_releases_grant_lock(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = MCPOAuthFlow(
            "refresh-cancel",
            "https://example.com/mcp",
            MCPOAuthConfig(client_id="client-id"),
        )
        _save_bound(
            flow,
            {
                "access_token": "access-r0",
                "refresh_token": "refresh-r0",
                "client_id": "client-id",
            },
        )
        first_post_started = asyncio.Event()
        post_count = 0

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, data=None):
                nonlocal post_count
                post_count += 1
                if post_count == 1:
                    first_post_started.set()
                    await asyncio.Event().wait()
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "access_token": "access-r1",
                    "refresh_token": "refresh-r1",
                }
                return response

        async def exercise_cancellation():
            binding = flow._server_binding(AUTH_METADATA)
            first = asyncio.create_task(flow.refresh_token(metadata=AUTH_METADATA, binding=binding))
            await asyncio.wait_for(first_post_started.wait(), timeout=10)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            return await asyncio.wait_for(
                flow.refresh_token(metadata=AUTH_METADATA, binding=binding),
                timeout=10,
            )

        with patch("httpx.AsyncClient", FakeClient):
            headers = asyncio.run(exercise_cancellation())

        assert headers == {"Authorization": "Bearer access-r1"}
        cached = load_tokens(flow.cache_identity)
        assert cached["refresh_token"] == "refresh-r1"

    def test_cancelled_refresh_lock_waiter_finishes_promptly_and_never_acquires_later(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = MCPOAuthFlow(
            "refresh-waiter-cancel",
            "https://example.com/mcp",
            MCPOAuthConfig(client_id="client-id"),
        )
        _save_bound(
            flow,
            {
                "access_token": "access-r0",
                "refresh_token": "refresh-r0",
                "client_id": "client-id",
            },
        )
        holder_post_started = asyncio.Event()
        release_holder = asyncio.Event()
        waiter_observed_busy_lock = threading.Event()
        request_tokens = []
        original_lock_file = oauth_module._lock_file

        def observed_lock_file(fd, *, blocking=True):
            acquired = original_lock_file(fd, blocking=blocking)
            if not blocking and not acquired:
                waiter_observed_busy_lock.set()
            return acquired

        monkeypatch.setattr(oauth_module, "_lock_file", observed_lock_file)

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def post(self, url, data=None):
                refresh_token = data["refresh_token"]
                request_tokens.append(refresh_token)
                if refresh_token == "refresh-r0":
                    holder_post_started.set()
                    await release_holder.wait()
                    next_generation = 1
                else:
                    assert refresh_token == "refresh-r1"
                    next_generation = 2
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "access_token": f"access-r{next_generation}",
                    "refresh_token": f"refresh-r{next_generation}",
                }
                return response

        async def exercise_cancellation():
            binding = flow._server_binding(AUTH_METADATA)
            holder = asyncio.create_task(
                flow.refresh_token(metadata=AUTH_METADATA, binding=binding)
            )
            await asyncio.wait_for(holder_post_started.wait(), timeout=10)
            waiter = asyncio.create_task(
                flow.refresh_token(metadata=AUTH_METADATA, binding=binding)
            )
            assert await asyncio.to_thread(waiter_observed_busy_lock.wait, 10)

            started = asyncio.get_running_loop().time()
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiter, timeout=0.5)
            elapsed = asyncio.get_running_loop().time() - started
            assert elapsed < 0.5

            await asyncio.sleep(0.1)
            assert request_tokens == ["refresh-r0"]
            release_holder.set()
            assert await asyncio.wait_for(holder, timeout=10) == {
                "Authorization": "Bearer access-r1"
            }
            return await asyncio.wait_for(
                flow.refresh_token(metadata=AUTH_METADATA, binding=binding),
                timeout=10,
            )

        with patch("httpx.AsyncClient", FakeClient):
            headers = asyncio.run(exercise_cancellation())

        assert headers == {"Authorization": "Bearer access-r2"}
        assert request_tokens == ["refresh-r0", "refresh-r1"]
        cached = load_tokens(flow.cache_identity)
        assert cached["refresh_token"] == "refresh-r2"


# ---------------------------------------------------------------------------
# _OAuthCallbackHandler — CSRF/state validation & per-flow isolation
# ---------------------------------------------------------------------------


class TestCallbackStateValidation:
    @staticmethod
    def _bind(server, state, path="/callback"):
        server.oauth_expected_state = state
        server.oauth_callback_path = path

    @staticmethod
    def _get(port, path):
        import urllib.request

        with urllib.request.urlopen(  # noqa: S310 - loopback only
            f"http://127.0.0.1:{port}{path}"
        ) as resp:
            return resp.status, resp.read()

    def test_correct_state_accepted(self):
        server, port = _start_callback_server(None)
        self._bind(server, "good-state")
        try:
            status, _ = self._get(port, "/callback?code=the-code&state=good-state")
            assert status == 200
            assert server.oauth_result.auth_code == "the-code"
            assert server.oauth_result.error is None
        finally:
            _stop_callback_server(server)

    def test_wrong_state_rejected(self):
        server, port = _start_callback_server(None)
        self._bind(server, "expected-state")
        try:
            import urllib.error

            try:
                self._get(port, "/callback?code=evil-code&state=attacker-state")
            except urllib.error.HTTPError as exc:
                try:
                    assert exc.code == 400
                finally:
                    exc.close()
            else:  # pragma: no cover - defensive
                raise AssertionError("wrong state should be rejected with HTTP 400")
            # The forged code must NOT be captured.
            assert server.oauth_result.auth_code is None
            assert server.oauth_result.error == "state_mismatch"
        finally:
            _stop_callback_server(server)

    def test_missing_state_rejected(self):
        server, port = _start_callback_server(None)
        self._bind(server, "expected-state")
        try:
            import urllib.error

            try:
                self._get(port, "/callback?code=evil-code")
            except urllib.error.HTTPError as exc:
                try:
                    assert exc.code == 400
                finally:
                    exc.close()
            else:  # pragma: no cover - defensive
                raise AssertionError("missing state should be rejected with HTTP 400")
            assert server.oauth_result.auth_code is None
            assert server.oauth_result.error == "state_mismatch"
        finally:
            _stop_callback_server(server)

    def test_concurrent_flows_are_isolated(self):
        # Two live servers (two flows) must not share the code/state that the
        # other captured — previously stored as handler CLASS attributes.
        server_a, port_a = _start_callback_server(None)
        server_b, port_b = _start_callback_server(None)
        self._bind(server_a, "state-a")
        self._bind(server_b, "state-b")
        try:
            self._get(port_a, "/callback?code=code-a&state=state-a")
            self._get(port_b, "/callback?code=code-b&state=state-b")
            assert server_a.oauth_result.auth_code == "code-a"
            assert server_b.oauth_result.auth_code == "code-b"
            assert server_a.oauth_result is not server_b.oauth_result
        finally:
            try:
                _stop_callback_server(server_a)
            finally:
                _stop_callback_server(server_b)

    def test_wait_for_code_reads_per_flow_result(self):
        server, _port = _start_callback_server(None)
        server.oauth_result.auth_code = "captured"
        try:
            code = asyncio.run(MCPOAuthFlow._wait_for_code(server, timeout=1))
            assert code == "captured"
        finally:
            _stop_callback_server(server)

    def test_wait_for_code_raises_on_state_mismatch_error(self):
        server, _port = _start_callback_server(None)
        server.oauth_result.error = "state_mismatch"
        try:
            try:
                asyncio.run(MCPOAuthFlow._wait_for_code(server, timeout=1))
            except RuntimeError as exc:
                assert "state_mismatch" in str(exc)
            else:  # pragma: no cover - defensive
                raise AssertionError("expected RuntimeError on state mismatch")
        finally:
            _stop_callback_server(server)

    def test_callback_diagnostics_redact_code_and_state(self, caplog):
        caplog.set_level(logging.DEBUG, logger="koder_agent.mcp.oauth")
        server, port = _start_callback_server(None)
        self._bind(server, "super-secret-state")
        try:
            status, _ = self._get(
                port,
                "/callback?code=super-secret-code&state=super-secret-state",
            )
            assert status == 200
        finally:
            _stop_callback_server(server)

        diagnostics = caplog.text
        assert "super-secret-code" not in diagnostics
        assert "super-secret-state" not in diagnostics
        assert "/callback?" not in diagnostics


# ---------------------------------------------------------------------------
# MCPOAuthFlow._discover_metadata
# ---------------------------------------------------------------------------


class TestOAuthEndpointValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("issuer", "http://auth.example/"),
            ("authorization_endpoint", "http://auth.example/authorize"),
            ("token_endpoint", "http://auth.example/token"),
            ("registration_endpoint", "http://auth.example/register"),
        ],
    )
    def test_rejects_non_loopback_cleartext_authorization_server_urls(self, field, value):
        flow = MCPOAuthFlow("cleartext", "https://mcp.example/mcp", MCPOAuthConfig())
        metadata = dict(AUTH_METADATA)
        metadata[field] = value

        with pytest.raises(RuntimeError, match="must use HTTPS unless it is loopback"):
            flow._server_binding(metadata)

    @pytest.mark.parametrize("host", ["127.0.0.1", "127.42.0.9", "localhost", "[::1]"])
    def test_allows_cleartext_only_for_validated_loopback_development_endpoints(self, host):
        base = f"http://{host}:8765"
        metadata = {
            "issuer": f"{base}/",
            "authorization_endpoint": f"{base}/authorize",
            "token_endpoint": f"{base}/token",
            "registration_endpoint": f"{base}/register",
        }
        flow = MCPOAuthFlow("loopback", "https://mcp.example/mcp", MCPOAuthConfig())

        binding = flow._server_binding(metadata)

        assert binding.registration_endpoint is not None

    @pytest.mark.parametrize(
        "field",
        ["issuer", "authorization_endpoint", "token_endpoint", "registration_endpoint"],
    )
    @pytest.mark.parametrize("fragment", ["#tenant-a", "#"])
    def test_rejects_fragments_on_authorization_server_identity_and_endpoints(
        self, field, fragment
    ):
        flow = MCPOAuthFlow("fragment", "https://mcp.example/mcp", MCPOAuthConfig())
        metadata = dict(AUTH_METADATA)
        metadata[field] = f"{metadata[field]}{fragment}"

        with pytest.raises(RuntimeError, match="without a fragment"):
            flow._server_binding(metadata)

    def test_distinct_fragment_bearing_issuers_are_both_rejected(self):
        flow = MCPOAuthFlow("fragment", "https://mcp.example/mcp", MCPOAuthConfig())
        for tenant in ("tenant-a", "tenant-b"):
            metadata = dict(AUTH_METADATA)
            metadata["issuer"] = f"https://auth.example/#{tenant}"
            with pytest.raises(RuntimeError, match="without a fragment"):
                flow._server_binding(metadata)

    def test_rejects_cleartext_configured_metadata_url_before_discovery(self):
        with pytest.raises(RuntimeError, match="must use HTTPS unless it is loopback"):
            MCPOAuthFlow(
                "metadata-cleartext",
                "https://mcp.example/mcp",
                MCPOAuthConfig(auth_server_metadata_url="http://auth.example/.well-known/oauth"),
            )


class TestDiscoverMetadata:
    def test_discovers_from_well_known(self, monkeypatch):
        metadata = {
            "authorization_endpoint": "https://auth.example/authorize",
            "token_endpoint": "https://auth.example/token",
        }

        async def mock_get(self, url, **kwargs):
            resp = MagicMock()
            if "oauth-protected-resource" in url:
                resp.status_code = 200
                resp.json.return_value = metadata
            else:
                resp.status_code = 404
            return resp

        config = MCPOAuthConfig()
        flow = MCPOAuthFlow("disc-server", "https://example.com/mcp", config)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=lambda url, **kw: _make_resp(url, metadata))
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = asyncio.run(flow._discover_metadata())
            assert result["authorization_endpoint"] == "https://auth.example/authorize"

    def test_uses_override_url(self):
        metadata = {
            "authorization_endpoint": "https://custom.auth/authorize",
            "token_endpoint": "https://custom.auth/token",
        }
        config = MCPOAuthConfig(auth_server_metadata_url="https://custom.auth/.well-known/openid")
        flow = MCPOAuthFlow("override-server", "https://example.com/mcp", config)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            call_count = 0

            async def smart_get(url, **kw):
                nonlocal call_count
                call_count += 1
                resp = MagicMock()
                if "custom.auth" in url:
                    resp.status_code = 200
                    resp.json.return_value = metadata
                else:
                    resp.status_code = 404
                return resp

            mock_client.get = smart_get
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            result = asyncio.run(flow._discover_metadata())
            assert result["token_endpoint"] == "https://custom.auth/token"

    def test_rejects_issuer_that_disagrees_with_delegated_authorization_server(self):
        protected_resource = {"authorization_servers": ["https://auth-a.example"]}
        mismatched_metadata = {
            "issuer": "https://auth-b.example/",
            "authorization_endpoint": "https://auth-b.example/authorize",
            "token_endpoint": "https://auth-b.example/token",
        }
        flow = MCPOAuthFlow("issuer-mismatch", "https://mcp.example/mcp", MCPOAuthConfig())

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def smart_get(url, **kwargs):
                response = MagicMock()
                if "oauth-protected-resource" in url:
                    response.status_code = 200
                    response.json.return_value = protected_resource
                elif url.startswith("https://auth-a.example/"):
                    response.status_code = 200
                    response.json.return_value = mismatched_metadata
                else:
                    response.status_code = 404
                return response

            mock_client.get = smart_get
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError, match="Could not discover valid OAuth metadata"):
                asyncio.run(flow._discover_metadata())

    def test_discovery_errors_redact_userinfo_and_query_secrets(self, caplog):
        caplog.set_level(logging.DEBUG, logger="koder_agent.mcp.oauth")
        flow = MCPOAuthFlow(
            "redaction",
            "https://alice:server-password@example.com/mcp?resource_token=resource-secret",
            MCPOAuthConfig(
                auth_server_metadata_url=(
                    "https://bob:metadata-password@auth.example/metadata?token=metadata-secret"
                )
            ),
        )

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=RuntimeError("query=exception-secret"))
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(RuntimeError) as exc_info:
                asyncio.run(flow._discover_metadata())

        diagnostics = f"{exc_info.value}\n{caplog.text}"
        for secret in (
            "alice",
            "server-password",
            "resource-secret",
            "bob",
            "metadata-password",
            "metadata-secret",
            "exception-secret",
        ):
            assert secret not in diagnostics

    def test_invalid_url_error_does_not_echo_userinfo_or_query(self):
        secret_url = "https://alice:password@example.com:bad/mcp?token=query-secret"

        with pytest.raises(ValueError) as exc_info:
            _normalize_cache_url(secret_url)

        message = str(exc_info.value)
        assert "alice" not in message
        assert "password" not in message
        assert "query-secret" not in message


# ---------------------------------------------------------------------------
# resolve_oauth_headers convenience
# ---------------------------------------------------------------------------


class TestResolveOauthHeaders:
    def test_returns_empty_when_no_oauth(self):
        result = asyncio.run(resolve_oauth_headers("server", "https://example.com", None))
        assert result == {}

    def test_returns_empty_for_empty_dict(self):
        result = asyncio.run(resolve_oauth_headers("server", "https://example.com", {}))
        assert result == {}

    def test_uses_cached_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = MCPOAuthConfig(client_id="cid")
        flow = MCPOAuthFlow("resolve-server", "https://example.com/mcp", config)
        _save_bound(
            flow,
            {
                "access_token": "resolve-tok",
                "expires_in": 3600,
                "obtained_at": time.time(),
            },
        )
        with patch.object(
            MCPOAuthFlow,
            "_discover_metadata",
            new_callable=AsyncMock,
            return_value=AUTH_METADATA,
        ):
            result = asyncio.run(
                resolve_oauth_headers(
                    "resolve-server",
                    "https://example.com/mcp",
                    {"clientId": "cid"},
                )
            )
        assert result == {"Authorization": "Bearer resolve-tok"}


# ---------------------------------------------------------------------------
# Integration with _build_effective_headers
# ---------------------------------------------------------------------------


class TestBuildEffectiveHeadersIntegration:
    def test_oauth_headers_merged(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        from koder_agent.mcp.server_config import MCPServerConfig, MCPServerType
        from koder_agent.mcp.server_factory import _build_effective_headers

        config = MCPServerConfig(
            name="factory-test",
            transport_type=MCPServerType.HTTP,
            url="https://example.com/mcp",
            headers={"X-Custom": "val"},
            oauth={"clientId": "cid"},
        )
        oauth_config = MCPOAuthConfig.from_dict(config.oauth)
        assert oauth_config is not None
        flow = MCPOAuthFlow(config.name, config.url, oauth_config)
        _save_bound(
            flow,
            {
                "access_token": "factory-tok",
                "expires_in": 3600,
                "obtained_at": time.time(),
            },
        )
        with patch.object(
            MCPOAuthFlow,
            "_discover_metadata",
            new_callable=AsyncMock,
            return_value=AUTH_METADATA,
        ):
            headers = asyncio.run(_build_effective_headers(config))
        assert headers["X-Custom"] == "val"
        assert headers["Authorization"] == "Bearer factory-tok"

    def test_no_oauth_no_change(self):
        from koder_agent.mcp.server_config import MCPServerConfig, MCPServerType
        from koder_agent.mcp.server_factory import _build_effective_headers

        config = MCPServerConfig(
            name="no-oauth",
            transport_type=MCPServerType.HTTP,
            url="https://example.com/mcp",
            headers={"X-Only": "static"},
        )
        headers = asyncio.run(_build_effective_headers(config))
        assert headers == {"X-Only": "static"}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_resp(url: str, metadata: Dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    if "oauth-protected-resource" in url:
        resp.status_code = 200
        resp.json.return_value = metadata
    else:
        resp.status_code = 404
    return resp


# ---------------------------------------------------------------------------
# Dynamic client registration redirect URI handling
# ---------------------------------------------------------------------------


class TestDynamicRegistrationRedirectUri:
    METADATA = {
        "issuer": "https://auth.example/",
        "authorization_endpoint": "https://auth.example/authorize",
        "token_endpoint": "https://auth.example/token",
        "registration_endpoint": "https://auth.example/register",
    }

    @staticmethod
    def _flow(name: str) -> MCPOAuthFlow:
        # No client_id configured -> dynamic registration path
        return MCPOAuthFlow(name, "https://example.com/mcp", MCPOAuthConfig(scopes=["read"]))

    def test_registers_with_real_redirect_uri(self, tmp_path, monkeypatch):
        """The registered redirect_uri must match the actual callback URI."""
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = self._flow("dyn-reg-server")
        captured: Dict[str, Any] = {}

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def fake_post(url, json=None, **kw):
                captured["url"] = url
                captured["payload"] = json
                resp = MagicMock()
                resp.status_code = 201
                resp.json.return_value = {"client_id": "dyn-client-id"}
                resp.raise_for_status = MagicMock()
                return resp

            mock_client.post = fake_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            client_id, secret = asyncio.run(
                flow._ensure_client(self.METADATA, "http://127.0.0.1:54321/callback")
            )

        assert client_id == "dyn-client-id"
        assert secret is None
        assert captured["payload"]["redirect_uris"] == ["http://127.0.0.1:54321/callback"]
        # No placeholder port-0 URI may ever be registered.
        assert "127.0.0.1:0" not in captured["payload"]["redirect_uris"][0]
        cached = load_tokens(flow.cache_identity)
        assert cached["client_id"] == "dyn-client-id"
        assert cached["redirect_uri"] == "http://127.0.0.1:54321/callback"

    def test_registration_merge_preserves_tokens_saved_while_request_is_in_flight(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = self._flow("dyn-interleaved")
        redirect_uri = "http://127.0.0.1:54321/callback"

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def fake_post(url, json=None, **kwargs):
                _save_bound(
                    flow,
                    {
                        "access_token": "concurrent-access",
                        "refresh_token": "concurrent-refresh",
                        "expires_in": 3600,
                        "obtained_at": time.time(),
                    },
                    self.METADATA,
                )
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "client_id": "registered-client",
                    "client_secret": "registered-secret",
                }
                return response

            mock_client.post = fake_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            client_id, client_secret = asyncio.run(flow._ensure_client(self.METADATA, redirect_uri))

        assert (client_id, client_secret) == ("registered-client", "registered-secret")
        cached = load_tokens(flow.cache_identity)
        assert cached["access_token"] == "concurrent-access"
        assert cached["refresh_token"] == "concurrent-refresh"
        assert cached["client_id"] == "registered-client"
        assert cached["client_secret"] == "registered-secret"
        assert cached["redirect_uri"] == redirect_uri

    def test_two_process_first_registration_persists_one_verified_winner(
        self, tmp_path, monkeypatch
    ):
        if os.name != "posix":
            pytest.skip("reviewer multiprocessing repro uses POSIX advisory locks")
        monkeypatch.setenv("HOME", str(tmp_path))
        registration_count = 0
        registration_guard = threading.Lock()

        class RegistrationHandler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                nonlocal registration_count
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                assert payload["redirect_uris"] == ["http://127.0.0.1:54321/callback"]
                with registration_guard:
                    registration_count += 1
                    client_number = registration_count
                body = json.dumps(
                    {
                        "client_id": f"dynamic-client-{client_number}",
                        "client_secret": f"dynamic-secret-{client_number}",
                    }
                ).encode("utf-8")
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), RegistrationHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        port = server.server_address[1]
        metadata = {
            "issuer": f"http://127.0.0.1:{port}/",
            "authorization_endpoint": f"http://127.0.0.1:{port}/authorize",
            "token_endpoint": f"http://127.0.0.1:{port}/token",
            "registration_endpoint": f"http://127.0.0.1:{port}/register",
        }
        context = multiprocessing.get_context("spawn")
        start_event = context.Event()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_dynamic_registration_worker,
                args=(str(tmp_path), metadata, start_event, result_queue),
            )
            for _ in range(2)
        ]
        try:
            for process in processes:
                process.start()
            start_event.set()
            results = [result_queue.get(timeout=30) for _ in processes]
            for process in processes:
                process.join(30)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(10)

        assert [process.exitcode for process in processes] == [0, 0]
        assert [error for _, error in results if error] == []
        selected_clients = {result[0] for result, _ in results}
        selected_secrets = {result[1] for result, _ in results}
        assert len(selected_clients) == 1
        assert len(selected_secrets) == 1
        flow = self._flow("two-process-registration")
        cached = load_tokens(flow.cache_identity)
        assert cached["client_id"] in selected_clients
        assert cached["client_secret"] in selected_secrets
        assert cached["redirect_uri"] == "http://127.0.0.1:54321/callback"
        assert registration_count in {1, 2}

    def test_reuses_cached_registration_for_same_redirect_uri(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = self._flow("dyn-cache-server")
        _save_bound(
            flow,
            {"client_id": "cached-id", "redirect_uri": "http://127.0.0.1:7777/callback"},
            self.METADATA,
        )

        with patch("httpx.AsyncClient") as mock_cls:
            client_id, _ = asyncio.run(
                flow._ensure_client(self.METADATA, "http://127.0.0.1:7777/callback")
            )
            mock_cls.assert_not_called()

        assert client_id == "cached-id"

    def test_does_not_reuse_registration_from_same_name_at_different_endpoint(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        original = MCPOAuthFlow(
            "dyn-shared", "https://one.example/mcp", MCPOAuthConfig(scopes=["read"])
        )
        replacement = MCPOAuthFlow(
            "dyn-shared", "https://two.example/mcp", MCPOAuthConfig(scopes=["read"])
        )
        redirect_uri = "http://127.0.0.1:7777/callback"
        save_tokens(
            original.cache_identity,
            {"client_id": "original-client", "redirect_uri": redirect_uri},
        )

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def fake_post(url, json=None, **kw):
                resp = MagicMock()
                resp.json.return_value = {"client_id": "replacement-client"}
                resp.raise_for_status = MagicMock()
                return resp

            mock_client.post = fake_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            client_id, _ = asyncio.run(replacement._ensure_client(self.METADATA, redirect_uri))

        assert client_id == "replacement-client"
        assert load_tokens(original.cache_identity)["client_id"] == "original-client"
        assert load_tokens(replacement.cache_identity)["client_id"] == "replacement-client"

    def test_re_registers_when_redirect_uri_changes(self, tmp_path, monkeypatch):
        """A stale registration for another port must not be reused."""
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = self._flow("dyn-stale-server")
        _save_bound(
            flow,
            {"client_id": "stale-id", "redirect_uri": "http://127.0.0.1:1111/callback"},
            self.METADATA,
        )

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def fake_post(url, json=None, **kw):
                resp = MagicMock()
                resp.json.return_value = {"client_id": "fresh-id"}
                resp.raise_for_status = MagicMock()
                return resp

            mock_client.post = fake_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            client_id, _ = asyncio.run(
                flow._ensure_client(self.METADATA, "http://127.0.0.1:2222/callback")
            )

        assert client_id == "fresh-id"
        cached = load_tokens(flow.cache_identity)
        assert cached["redirect_uri"] == "http://127.0.0.1:2222/callback"

    def test_authorization_server_change_never_reuses_old_dynamic_client(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        redirect_uri = "http://127.0.0.1:7777/callback"
        old_metadata = {
            "issuer": "https://auth-a.example/",
            "authorization_endpoint": "https://auth-a.example/authorize",
            "token_endpoint": "https://auth-a.example/token",
            "registration_endpoint": "https://auth-a.example/register",
        }
        new_metadata = {
            "issuer": "https://auth-b.example/",
            "authorization_endpoint": "https://auth-b.example/authorize",
            "token_endpoint": "https://auth-b.example/token",
            "registration_endpoint": "https://auth-b.example/register",
        }
        flow = self._flow("dyn-as-change")
        _save_bound(
            flow,
            {
                "client_id": "old-client",
                "client_secret": "old-client-secret",
                "redirect_uri": redirect_uri,
            },
            old_metadata,
        )
        captured: Dict[str, Any] = {}

        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()

            async def fake_post(url, json=None, **kwargs):
                captured["url"] = url
                captured["payload"] = json
                response = MagicMock()
                response.raise_for_status = MagicMock()
                response.json.return_value = {
                    "client_id": "new-client",
                    "client_secret": "new-client-secret",
                }
                return response

            mock_client.post = fake_post
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            client_id, client_secret = asyncio.run(flow._ensure_client(new_metadata, redirect_uri))

        assert client_id == "new-client"
        assert client_secret == "new-client-secret"
        assert captured["url"] == "https://auth-b.example/register"
        assert "old-client" not in repr(captured["payload"])
        cached = load_tokens(flow.cache_identity)
        assert cached["client_id"] == "new-client"
        assert cached["_oauth_server_binding"] == flow._server_binding(new_metadata).as_dict()

    def test_missing_registration_endpoint_is_explicit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = self._flow("dyn-no-endpoint")
        metadata = {
            "authorization_endpoint": "https://auth.example/authorize",
            "token_endpoint": "https://auth.example/token",
        }
        try:
            asyncio.run(flow._ensure_client(metadata, "http://127.0.0.1:3333/callback"))
        except RuntimeError as exc:
            assert "registration_endpoint" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for missing registration_endpoint")


class TestAuthenticateUsesActualCallbackPort:
    def test_full_flow_registers_actual_port(self, tmp_path, monkeypatch):
        """authenticate() must start the callback server before registration
        and use the same redirect URI for registration and authorization."""
        monkeypatch.setenv("HOME", str(tmp_path))
        flow = MCPOAuthFlow(
            "full-flow-server", "https://example.com/mcp", MCPOAuthConfig(scopes=["read"])
        )

        metadata = {
            "authorization_endpoint": "https://auth.example/authorize",
            "token_endpoint": "https://auth.example/token",
            "registration_endpoint": "https://auth.example/register",
        }
        seen: Dict[str, Any] = {}

        async def fake_discover():
            return metadata

        async def fake_ensure(meta, redirect_uri):
            seen["registration_redirect"] = redirect_uri
            return "cid", None

        async def fake_code_flow(meta, client_id, client_secret, *, server, redirect_uri):
            seen["authorization_redirect"] = redirect_uri
            assert server is not None
            return {"access_token": "tok", "obtained_at": time.time()}

        with (
            patch.object(flow, "_discover_metadata", side_effect=fake_discover),
            patch.object(flow, "_ensure_client", side_effect=fake_ensure),
            patch.object(flow, "_authorization_code_flow", side_effect=fake_code_flow),
        ):
            headers = asyncio.run(flow.authenticate())

        assert headers == {"Authorization": "Bearer tok"}
        assert seen["registration_redirect"] == seen["authorization_redirect"]
        assert "127.0.0.1:0" not in seen["registration_redirect"]
        assert load_tokens(flow.cache_identity)["access_token"] == "tok"
