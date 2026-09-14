"""Enter and exit isolated Git worktrees without losing local work."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import subprocess
import threading
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Literal, Optional
from weakref import WeakKeyDictionary

from koder_agent.harness.execution_context import get_execution_cwd

from .compat import function_tool
from .todo import TodoStore, get_todo_store_or_none

_SEGMENT_RE = re.compile(r"^[a-zA-Z0-9._-]+$")
_REPOSITORY_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_ID_FILE = ".koder-repository-id"
_WORKTREE_ID_FILE = ".koder-worktree-id"
_UNMERGED_STATUSES = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
_OPERATION_MARKERS = {
    "MERGE_HEAD": "merge",
    "CHERRY_PICK_HEAD": "cherry-pick",
    "REVERT_HEAD": "revert",
    "BISECT_LOG": "bisect",
    "rebase-merge": "rebase",
    "rebase-apply": "rebase",
    "sequencer": "sequencer",
}

SessionPhase = Literal[
    "active",
    "worktree_cleanup_pending",
    "branch_cleanup_pending",
]


def _validate_slug(slug: str) -> str | None:
    if len(slug) > 64:
        return "Slug must be at most 64 characters"
    if slug.startswith("/") or slug.endswith("/"):
        return "Slug must not start or end with /"
    for segment in slug.split("/"):
        if segment in (".", ".."):
            return f"Invalid segment: {segment}"
        if not _SEGMENT_RE.match(segment):
            return f"Invalid characters in segment: {segment}"
    return None


@dataclass(frozen=True)
class GitIdentity:
    root: Path
    common_dir: Path
    admin_dir: Path


@dataclass
class WorktreeSession:
    original_cwd: str
    owner_root: str
    owner_common_dir: str
    approved_root: str
    worktree_path: str
    worktree_branch: str
    name: str
    phase: SessionPhase = "active"
    branch_owned: bool = False
    owner_device: int | None = None
    owner_inode: int | None = None
    owner_fingerprint: str | None = None
    owner_structure: str | None = None
    owner_admin_dir: str | None = None
    approved_device: int | None = None
    approved_inode: int | None = None
    worktree_device: int | None = None
    worktree_inode: int | None = None
    worktree_admin_dir: str | None = None
    worktree_admin_relative: str | None = None
    worktree_admin_device: int | None = None
    worktree_admin_inode: int | None = None
    worktree_fingerprint: str | None = None
    branch_cleanup_head: str | None = None
    branch_cleanup_identity: str | None = None


@dataclass
class WorktreeState:
    path_exists: bool | None = None
    registered: bool | None = None
    registered_branch: str | None = None
    registered_head: str | None = None
    path_owner_matches: bool | None = None
    worktree_instance_matches: bool | None = None
    branch_exists: bool | None = None
    branch_head: str | None = None
    inspection_errors: list[str] = field(default_factory=list)

    @property
    def worktree_absent(self) -> bool:
        return self.path_exists is False and self.registered is False

    @property
    def worktree_owned(self) -> bool:
        return (
            self.path_exists is True
            and self.registered is True
            and self.path_owner_matches is True
            and self.worktree_instance_matches is True
        )


@dataclass
class WorktreePreflight:
    local_state: list[dict[str, str]] = field(default_factory=list)
    inspection_errors: list[str] = field(default_factory=list)


class RepositoryIdentityError(RuntimeError):
    """The recorded owner can no longer be proven to be the same repository."""


@dataclass
class _WorktreeOwner:
    session: WorktreeSession | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


# Reuse the runtime's existing lifetime/capability object, not its display labels.
# Main session switches and background resumes retain their TodoStore; unrelated
# schedulers (even with identical labels) cannot acquire each other's state.
# Weak keys release bookkeeping with the runtime, never delete its Git work.
_owners: WeakKeyDictionary[TodoStore, _WorktreeOwner] = WeakKeyDictionary()
_owners_lock = threading.RLock()
# Legacy direct callers / the single-client stdio MCP server have no runtime
# scope. Keep their standalone workflow, separate from every scoped agent.
_direct_owner = _WorktreeOwner()


def _get_worktree_owner() -> _WorktreeOwner:
    runtime = get_todo_store_or_none()
    if runtime is None:
        return _direct_owner
    with _owners_lock:
        owner = _owners.get(runtime)
        if owner is None:
            owner = _WorktreeOwner()
            _owners[runtime] = owner
        return owner


def _serialized_worktree_operation(function):
    """Serialize same-owner enter/exit, including calls from copied thread contexts."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with _get_worktree_owner().lock:
            return function(*args, **kwargs)

    return wrapped


def _get_worktree_session() -> WorktreeSession | None:
    owner = _get_worktree_owner()
    with owner.lock:
        return owner.session


def _set_worktree_session(session: WorktreeSession | None) -> None:
    owner = _get_worktree_owner()
    with owner.lock:
        owner.session = session


def _git_failure_detail(error: BaseException) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        stderr = error.stderr.strip() if isinstance(error.stderr, str) else ""
        if stderr:
            return stderr
        return f"git exited with status {error.returncode}"
    return str(error) or error.__class__.__name__


def _resolve_git_path(cwd: Path | None, *arguments: str) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve(strict=True)


def _git_identity(cwd: Path | None = None) -> GitIdentity | None:
    try:
        return GitIdentity(
            root=_resolve_git_path(cwd, "--show-toplevel"),
            common_dir=_resolve_git_path(cwd, "--path-format=absolute", "--git-common-dir"),
            admin_dir=_resolve_git_path(cwd, "--path-format=absolute", "--absolute-git-dir"),
        )
    except subprocess.CalledProcessError:
        return None


def _require_descriptor_platform() -> None:
    required = (os.open, os.stat, os.unlink, os.rmdir)
    if (
        os.name != "posix"
        or not hasattr(os, "fchdir")
        or any(function not in os.supports_dir_fd for function in required)
    ):
        raise OSError("This platform cannot safely bind cleanup to open directory descriptors.")


def _identity_from_stat(path_stat: os.stat_result, description: str) -> tuple[int, int]:
    if not stat.S_ISDIR(path_stat.st_mode):
        raise OSError(f"{description} is not a directory.")
    device = getattr(path_stat, "st_dev", 0)
    inode = getattr(path_stat, "st_ino", 0)
    if not isinstance(device, int) or not isinstance(inode, int) or device == 0 or inode == 0:
        raise OSError(f"{description} does not expose a stable filesystem identity.")
    return device, inode


def _open_directory(path: Path) -> int:
    _require_descriptor_platform()
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        _identity_from_stat(os.fstat(descriptor), str(path))
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_directory_at(parent_descriptor: int, name: str) -> int:
    if name in {"", ".", ".."} or "/" in name or (os.altsep and os.altsep in name):
        raise OSError(f"Unsafe directory component: {name!r}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        _identity_from_stat(os.fstat(descriptor), name)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _open_relative_directory(parent_descriptor: int, relative: str) -> int:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise OSError(f"Unsafe relative directory: {relative!r}")
    current = os.dup(parent_descriptor)
    try:
        for part in parts:
            next_descriptor = _open_directory_at(current, part)
            os.close(current)
            current = next_descriptor
        return current
    except BaseException:
        os.close(current)
        raise


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _validate_marker_stat(marker_stat: os.stat_result, description: str) -> None:
    if not stat.S_ISREG(marker_stat.st_mode):
        raise OSError(f"{description} is not a regular file.")
    if marker_stat.st_nlink != 1:
        raise OSError(f"{description} must have exactly one hard link.")
    if hasattr(os, "geteuid") and marker_stat.st_uid != os.geteuid():
        raise OSError(f"{description} is not owned by the current user.")
    permissions = stat.S_IMODE(marker_stat.st_mode)
    if permissions & 0o077 or not permissions & stat.S_IRUSR:
        raise OSError(f"{description} does not have private owner-only permissions.")


def _read_fingerprint_at(parent_descriptor: int, name: str, description: str) -> str:
    entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        marker_stat = os.fstat(descriptor)
        _validate_marker_stat(marker_stat, description)
        if not _same_stat(entry_stat, marker_stat):
            raise OSError(f"{description} changed while it was opened.")
        data = os.read(descriptor, 66)
        if os.read(descriptor, 1):
            raise OSError(f"{description} is too large.")
    finally:
        os.close(descriptor)
    try:
        fingerprint = data.decode("ascii").removesuffix("\n")
    except UnicodeDecodeError as error:
        raise OSError(f"{description} is invalid.") from error
    if not _REPOSITORY_FINGERPRINT_RE.fullmatch(fingerprint):
        raise OSError(f"{description} is invalid.")
    return fingerprint


def _fingerprint_at(parent_descriptor: int, name: str, *, create: bool, description: str) -> str:
    parent_identity = _identity_from_stat(os.fstat(parent_descriptor), "Marker parent directory")
    try:
        return _read_fingerprint_at(parent_descriptor, name, description)
    except FileNotFoundError:
        if not create:
            raise OSError(f"{description} is missing.") from None

    fingerprint = secrets.token_hex(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=parent_descriptor)
    except FileExistsError:
        return _read_fingerprint_at(parent_descriptor, name, description)

    created_stat = os.fstat(descriptor)
    try:
        _validate_marker_stat(created_stat, description)
        data = f"{fingerprint}\n".encode("ascii")
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written == 0:
                raise OSError(f"Failed to write complete {description}.")
            offset += written
        os.fsync(descriptor)
        if (
            _identity_from_stat(os.fstat(parent_descriptor), "Marker parent directory")
            != parent_identity
        ):
            raise OSError("Marker parent directory changed during creation.")
    except BaseException:
        try:
            entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
            if _same_stat(entry_stat, created_stat):
                os.unlink(name, dir_fd=parent_descriptor)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    return fingerprint


def _repository_fingerprint(common_dir: Path, *, create: bool) -> str:
    descriptor = _open_directory(common_dir)
    try:
        return _fingerprint_at(
            descriptor,
            _REPOSITORY_ID_FILE,
            create=create,
            description="Repository identity marker",
        )
    finally:
        os.close(descriptor)


def _hash_regular_file_at(
    parent_descriptor: int, name: str, *, include_identity: bool = True
) -> dict[str, int | str]:
    entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    descriptor = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_descriptor)
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or not _same_stat(entry_stat, file_stat):
            raise OSError(f"Git structural file changed while opening: {name}")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 65536):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    result: dict[str, int | str] = {"sha256": digest.hexdigest()}
    if include_identity:
        result.update({"device": file_stat.st_dev, "inode": file_stat.st_ino})
    return result


def _identity_record(path_stat: os.stat_result, description: str) -> dict[str, int]:
    device, inode = _identity_from_stat(path_stat, description)
    return {"device": device, "inode": inode}


def _directory_entry_identity(parent_descriptor: int, name: str) -> dict[str, int]:
    entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    return _identity_record(entry_stat, f"Git structural directory {name}")


def _repository_structure(root_fd: int, common_fd: int, admin_fd: int) -> str:
    structure = {
        "root": _identity_record(os.fstat(root_fd), "Git root"),
        "common": _identity_record(os.fstat(common_fd), "Git common directory"),
        "admin": _identity_record(os.fstat(admin_fd), "Git admin directory"),
        "common_head": _hash_regular_file_at(common_fd, "HEAD", include_identity=False),
        "config": _hash_regular_file_at(common_fd, "config", include_identity=False),
        "objects": _directory_entry_identity(common_fd, "objects"),
        "refs": _directory_entry_identity(common_fd, "refs"),
        "admin_head": _hash_regular_file_at(admin_fd, "HEAD", include_identity=False),
    }
    return json.dumps(structure, sort_keys=True, separators=(",", ":"))


def _branch_structural_identity(common_fd: int, branch: str) -> str:
    refs_fd = _open_relative_directory(common_fd, "refs/heads")
    logs_fd = _open_relative_directory(common_fd, "logs/refs/heads")
    try:
        identity = {
            "ref": _hash_regular_file_at(refs_fd, branch),
            "reflog": _hash_regular_file_at(logs_fd, branch),
        }
    finally:
        os.close(logs_fd)
        os.close(refs_fd)
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _run_pinned(descriptor: int, command: list[str], **kwargs) -> subprocess.CompletedProcess:
    _require_descriptor_platform()
    return subprocess.run(
        command,
        pass_fds=(descriptor,),
        preexec_fn=lambda: os.fchdir(descriptor),
        **kwargs,
    )


def _paths_match(left: str | Path, right: str | Path) -> bool:
    try:
        left_path = Path(left).resolve(strict=False)
        right_path = Path(right).resolve(strict=False)
    except OSError:
        return False
    return os.path.normcase(str(left_path)) == os.path.normcase(str(right_path))


def _path_exists(path: str | Path) -> bool | None:
    try:
        return os.path.lexists(path)
    except OSError:
        return None


def _first_symlink(root: Path, path: Path) -> Path | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path

    current = root
    if current.is_symlink():
        return current
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return current
    return None


def _prepare_worktree_path(identity: GitIdentity, name: str) -> tuple[Path, Path, str | None]:
    approved_path = identity.root / ".koder" / "worktrees"
    target_path = approved_path / name

    symlink = _first_symlink(identity.root, target_path)
    if symlink is not None:
        return approved_path, target_path, f"Symlink ancestor is not allowed: {symlink}"

    try:
        approved_path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return approved_path, target_path, f"Failed to create worktree directory: {error}"

    symlink = _first_symlink(identity.root, target_path)
    if symlink is not None:
        return approved_path, target_path, f"Symlink ancestor is not allowed: {symlink}"

    try:
        approved_root = approved_path.resolve(strict=True)
        worktree_path = target_path.resolve(strict=False)
        worktree_path.relative_to(approved_root)
    except (OSError, ValueError) as error:
        return (
            approved_path,
            target_path,
            f"Worktree path is outside the canonical approved root: {error}",
        )
    return approved_root, worktree_path, None


def _validate_recorded_path(session: WorktreeSession) -> str | None:
    owner_root = Path(session.owner_root)
    approved_root = Path(session.approved_root)
    worktree_path = Path(session.worktree_path)

    symlink = _first_symlink(owner_root, worktree_path)
    if symlink is not None:
        return f"Recorded worktree path now contains a symlink ancestor: {symlink}"

    try:
        canonical_approved = approved_root.resolve(strict=False)
        canonical_path = worktree_path.resolve(strict=False)
        canonical_approved.relative_to(owner_root)
        canonical_path.relative_to(canonical_approved)
    except (OSError, ValueError) as error:
        return f"Recorded worktree path is outside its canonical approved root: {error}"

    if not _paths_match(canonical_approved, approved_root):
        return "The canonical approved worktree root has changed."
    if not _paths_match(canonical_path, worktree_path):
        return "The canonical worktree path has changed."
    return None


def _parse_worktree_list(output: str) -> list[dict[str, str | None]]:
    registrations: list[dict[str, str | None]] = []
    for block in output.strip().split("\n\n"):
        if not block:
            continue
        registration: dict[str, str | None] = {
            "path": None,
            "head": None,
            "branch": None,
        }
        for line in block.splitlines():
            if line.startswith("worktree "):
                registration["path"] = line.removeprefix("worktree ")
            elif line.startswith("HEAD "):
                registration["head"] = line.removeprefix("HEAD ")
            elif line.startswith("branch refs/heads/"):
                registration["branch"] = line.removeprefix("branch refs/heads/")
        registrations.append(registration)
    return registrations


def _branch_state(owner_descriptor: int, branch: str) -> tuple[bool | None, str | None, str | None]:
    try:
        result = _run_pinned(
            owner_descriptor,
            ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return None, None, _git_failure_detail(error)
    if result.returncode == 0:
        return True, result.stdout.strip(), None
    if result.returncode == 1:
        return False, None, None
    return None, None, f"git exited with status {result.returncode}"


class WorktreeLifecycle:
    """Own all inspection and transitions for one recorded worktree session."""

    def __init__(
        self,
        session: WorktreeSession,
        identity: GitIdentity,
        root_fd: int,
        common_fd: int,
        admin_fd: int,
        approved_fd: int,
    ):
        self.session = session
        self.owner = _get_worktree_owner()
        self.identity = identity
        self.root_fd = root_fd
        self.common_fd = common_fd
        self.admin_fd = admin_fd
        self.approved_fd = approved_fd

    def close(self) -> None:
        for descriptor in (self.approved_fd, self.admin_fd, self.common_fd, self.root_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass

    @classmethod
    def create(
        cls, session: WorktreeSession, identity: GitIdentity
    ) -> tuple[WorktreeLifecycle | None, str | None]:
        descriptors: list[int] = []
        try:
            root_fd = _open_directory(identity.root)
            descriptors.append(root_fd)
            common_fd = _open_directory(identity.common_dir)
            descriptors.append(common_fd)
            admin_fd = _open_directory(identity.admin_dir)
            descriptors.append(admin_fd)
            approved_fd = _open_directory(Path(session.approved_root))
            descriptors.append(approved_fd)
            session.owner_fingerprint = _fingerprint_at(
                common_fd,
                _REPOSITORY_ID_FILE,
                create=True,
                description="Repository identity marker",
            )
            session.owner_structure = _repository_structure(root_fd, common_fd, admin_fd)
            session.owner_device, session.owner_inode = _identity_from_stat(
                os.fstat(common_fd), "Git common directory"
            )
            session.approved_device, session.approved_inode = _identity_from_stat(
                os.fstat(approved_fd), "Approved worktree directory"
            )
            session.owner_admin_dir = str(identity.admin_dir)
            return cls(session, identity, root_fd, common_fd, admin_fd, approved_fd), None
        except OSError as error:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
            return None, str(error)

    @classmethod
    def reopen(cls, session: WorktreeSession) -> tuple[WorktreeLifecycle | None, str | None]:
        required = (
            session.owner_fingerprint,
            session.owner_structure,
            session.owner_admin_dir,
            session.owner_device,
            session.owner_inode,
            session.approved_device,
            session.approved_inode,
            session.worktree_admin_dir,
            session.worktree_admin_relative,
            session.worktree_admin_device,
            session.worktree_admin_inode,
            session.worktree_fingerprint,
            session.worktree_device,
            session.worktree_inode,
        )
        if any(value is None for value in required):
            return (
                None,
                "The recorded session lacks required repository/worktree ownership identity.",
            )
        try:
            identity = _git_identity(Path(session.owner_root))
        except OSError as error:
            return None, f"Failed to inspect owning Git repository: {_git_failure_detail(error)}"
        if identity is None:
            return None, "The recorded owning Git repository is no longer available."
        if not _paths_match(identity.root, session.owner_root):
            return None, "The recorded owning Git root no longer has the same canonical identity."
        if not _paths_match(identity.common_dir, session.owner_common_dir):
            return (
                None,
                "The recorded Git common directory no longer has the same canonical identity.",
            )
        if not _paths_match(identity.admin_dir, session.owner_admin_dir):
            return (
                None,
                "The recorded Git admin directory no longer has the same canonical identity.",
            )
        descriptors: list[int] = []
        try:
            root_fd = _open_directory(identity.root)
            descriptors.append(root_fd)
            common_fd = _open_directory(identity.common_dir)
            descriptors.append(common_fd)
            admin_fd = _open_directory(identity.admin_dir)
            descriptors.append(admin_fd)
            approved_fd = _open_directory(Path(session.approved_root))
            descriptors.append(approved_fd)
            lifecycle = cls(session, identity, root_fd, common_fd, admin_fd, approved_fd)
            identity_error = lifecycle.repository_identity_error()
            if identity_error is not None:
                lifecycle.close()
                return None, identity_error
            return lifecycle, None
        except OSError as error:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            return None, f"Failed to open recorded ownership handles: {error}"

    def repository_identity_error(self) -> str | None:
        try:
            fingerprint = _fingerprint_at(
                self.common_fd,
                _REPOSITORY_ID_FILE,
                create=False,
                description="Repository identity marker",
            )
            if fingerprint != self.session.owner_fingerprint:
                return "The recorded Git common directory has a different repository fingerprint."
            if (
                _repository_structure(self.root_fd, self.common_fd, self.admin_fd)
                != self.session.owner_structure
            ):
                return "The recorded repository's independent Git structural identity changed."
            if _identity_from_stat(os.fstat(self.common_fd), "Git common directory") != (
                self.session.owner_device,
                self.session.owner_inode,
            ):
                return "The recorded Git common directory has a different filesystem identity."
            if _identity_from_stat(os.fstat(self.approved_fd), "Approved worktree directory") != (
                self.session.approved_device,
                self.session.approved_inode,
            ):
                return "The approved worktree directory has a different filesystem identity."
        except OSError as error:
            return f"The recorded repository identity could not be revalidated: {error}"
        return None

    def repository_path_error(self) -> str | None:
        try:
            root_stat = os.stat(self.session.owner_root, follow_symlinks=False)
            if not _same_stat(root_stat, os.fstat(self.root_fd)):
                return "The owning repository path no longer names the descriptor-bound repository."
            common_stat = os.stat(self.session.owner_common_dir, follow_symlinks=False)
            if not _same_stat(common_stat, os.fstat(self.common_fd)):
                return (
                    "The Git common-directory path no longer names the descriptor-bound repository."
                )
        except OSError as error:
            return f"The recorded repository paths could not be revalidated: {error}"
        return None

    def run_owner_git(self, arguments: list[str], **kwargs) -> subprocess.CompletedProcess:
        return _run_pinned(self.root_fd, ["git", *arguments], **kwargs)

    def _open_worktree(self) -> int:
        descriptor = _open_directory_at(self.approved_fd, self.session.name)
        expected = (self.session.worktree_device, self.session.worktree_inode)
        if _identity_from_stat(os.fstat(descriptor), "Recorded worktree") != expected:
            os.close(descriptor)
            raise OSError("The worktree path names a different worktree instance.")
        return descriptor

    def _open_worktree_admin(self) -> int:
        descriptor = _open_relative_directory(
            self.common_fd, self.session.worktree_admin_relative or ""
        )
        expected = (self.session.worktree_admin_device, self.session.worktree_admin_inode)
        if _identity_from_stat(os.fstat(descriptor), "Linked-worktree admin directory") != expected:
            os.close(descriptor)
            raise OSError("The linked-worktree admin path names a different instance.")
        fingerprint = _fingerprint_at(
            descriptor,
            _WORKTREE_ID_FILE,
            create=False,
            description="Linked-worktree identity marker",
        )
        if fingerprint != self.session.worktree_fingerprint:
            os.close(descriptor)
            raise OSError("The linked-worktree identity marker changed.")
        return descriptor

    def run_worktree_git(self, arguments: list[str], **kwargs) -> subprocess.CompletedProcess:
        descriptor = self._open_worktree()
        try:
            return _run_pinned(descriptor, ["git", *arguments], **kwargs)
        finally:
            os.close(descriptor)

    def capture_worktree_identity(self, *, create: bool) -> str | None:
        try:
            worktree_fd = _open_directory_at(self.approved_fd, self.session.name)
        except OSError as error:
            return str(error)
        try:
            admin_result = _run_pinned(
                worktree_fd,
                ["git", "rev-parse", "--path-format=absolute", "--absolute-git-dir"],
                check=True,
                capture_output=True,
                text=True,
            )
            admin_dir = Path(admin_result.stdout.strip()).resolve(strict=True)
            admin_relative = str(admin_dir.relative_to(self.identity.common_dir))
            admin_fd = _open_relative_directory(self.common_fd, admin_relative)
            try:
                fingerprint = _fingerprint_at(
                    admin_fd,
                    _WORKTREE_ID_FILE,
                    create=create,
                    description="Linked-worktree identity marker",
                )
                worktree_identity = _identity_from_stat(os.fstat(worktree_fd), "Linked worktree")
                admin_identity = _identity_from_stat(
                    os.fstat(admin_fd), "Linked-worktree admin directory"
                )
            finally:
                os.close(admin_fd)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            return _git_failure_detail(error)
        finally:
            os.close(worktree_fd)
        self.session.worktree_device, self.session.worktree_inode = worktree_identity
        self.session.worktree_admin_dir = str(admin_dir)
        self.session.worktree_admin_relative = admin_relative
        self.session.worktree_admin_device, self.session.worktree_admin_inode = admin_identity
        self.session.worktree_fingerprint = fingerprint
        return None

    def inspect(self) -> WorktreeState:
        try:
            path_stat = os.stat(self.session.name, dir_fd=self.approved_fd, follow_symlinks=False)
            state = WorktreeState(path_exists=True)
            if not stat.S_ISDIR(path_stat.st_mode):
                state.path_owner_matches = False
        except FileNotFoundError:
            state = WorktreeState(path_exists=False)
        except OSError as error:
            state = WorktreeState(path_exists=None)
            state.inspection_errors.append(f"worktree path: {error}")

        identity_error = self.repository_identity_error()
        if identity_error is not None:
            state.inspection_errors.append(f"repository identity: {identity_error}")
            return state

        try:
            result = self.run_owner_git(
                ["worktree", "list", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            )
            state.registered = False
            for registration in _parse_worktree_list(result.stdout):
                path = registration["path"]
                if path is not None and _paths_match(path, self.session.worktree_path):
                    state.registered = True
                    state.registered_branch = registration["branch"]
                    state.registered_head = registration["head"]
                    break
        except (subprocess.CalledProcessError, OSError) as error:
            state.inspection_errors.append(f"worktree registration: {_git_failure_detail(error)}")

        branch_exists, branch_head, branch_error = _branch_state(
            self.root_fd, self.session.worktree_branch
        )
        state.branch_exists = branch_exists
        state.branch_head = branch_head
        if branch_error:
            state.inspection_errors.append(f"branch: {branch_error}")

        if state.path_exists is True and state.registered is True:
            try:
                worktree_fd = self._open_worktree()
                admin_fd = self._open_worktree_admin()
            except OSError:
                state.path_owner_matches = False
                state.worktree_instance_matches = False
            else:
                os.close(admin_fd)
                os.close(worktree_fd)
                state.path_owner_matches = True
                state.worktree_instance_matches = True
        return state

    def reconcile(self, state: WorktreeState, *, creation_incomplete: bool = False) -> None:
        """Store exactly the retryable phase represented by current Git state."""

        if state.worktree_absent:
            if state.branch_exists is False:
                self.owner.session = None
                return
            self.session.phase = "branch_cleanup_pending"
        elif creation_incomplete or not state.worktree_owned:
            self.session.phase = "worktree_cleanup_pending"
        else:
            self.session.phase = "active"
        self.owner.session = self.session

    def result(self, action: str) -> dict:
        return {
            "action": action,
            "original_cwd": self.session.original_cwd,
            "owner_root": self.session.owner_root,
            "owner_common_dir": self.session.owner_common_dir,
            "worktree_path": self.session.worktree_path,
            "worktree_branch": self.session.worktree_branch,
            "session_state": self.session.phase,
        }

    @staticmethod
    def add_state(result: dict, state: WorktreeState) -> None:
        result.update(
            {
                "worktree_path_exists": state.path_exists,
                "worktree_registered": state.registered,
                "registered_branch": state.registered_branch,
                "registered_head": state.registered_head,
                "worktree_owner_matches": state.path_owner_matches,
                "worktree_instance_matches": state.worktree_instance_matches,
                "branch_exists": state.branch_exists,
                "branch_head": state.branch_head,
            }
        )
        if state.inspection_errors:
            result.setdefault("inspection_errors", []).extend(state.inspection_errors)

    def remove_owned_worktree(self, state: WorktreeState) -> BaseException | None:
        if not state.worktree_owned:
            return OSError("The exact linked-worktree instance is not verified.")
        if state.registered_head is None:
            return OSError("The verified worktree registration has no HEAD.")
        identity_error = self.repository_identity_error()
        if identity_error is not None:
            return RepositoryIdentityError(identity_error)

        try:
            head = self.run_worktree_git(
                ["rev-parse", "HEAD"], check=True, capture_output=True, text=True
            ).stdout.strip()
            if head != state.registered_head:
                raise OSError("The worktree HEAD changed before removal.")
            owned_branch = (
                state.registered_branch == self.session.worktree_branch
                and state.branch_head == head
            )
            branch_identity = (
                _branch_structural_identity(self.common_fd, self.session.worktree_branch)
                if owned_branch
                else None
            )
            worktree_fd = self._open_worktree()
            admin_fd = self._open_worktree_admin()
            try:
                _empty_directory(worktree_fd)
                _rmdir_verified(self.approved_fd, self.session.name, worktree_fd)
                _empty_directory(admin_fd)
                admin_parent, admin_name = _open_parent_relative(
                    self.common_fd, self.session.worktree_admin_relative or ""
                )
                try:
                    _rmdir_verified(admin_parent, admin_name, admin_fd)
                finally:
                    os.close(admin_parent)
            finally:
                os.close(admin_fd)
                os.close(worktree_fd)
            if owned_branch:
                self.session.branch_cleanup_head = head
                self.session.branch_cleanup_identity = branch_identity
                self.session.branch_owned = True
            return None
        except BaseException as error:
            return error

    def delete_owned_branch(self, discard_changes: bool) -> BaseException | None:
        expected_head = self.session.branch_cleanup_head
        if expected_head is None:
            return OSError(
                "No branch head was captured from Koder's verified successful worktree removal."
            )
        if self.session.branch_cleanup_identity is None:
            return OSError("No stable branch-instance identity was captured at worktree removal.")
        exists, current_head, state_error = _branch_state(
            self.root_fd, self.session.worktree_branch
        )
        if state_error:
            return OSError(state_error)
        if exists is not True or current_head != expected_head:
            return OSError("The branch was deleted, recreated, or moved after worktree removal.")
        try:
            current_identity = _branch_structural_identity(
                self.common_fd, self.session.worktree_branch
            )
        except OSError as error:
            return error
        if current_identity != self.session.branch_cleanup_identity:
            return OSError("The branch was deleted, recreated, or moved after worktree removal.")
        if not discard_changes:
            merge_check_arguments = ["merge-base", "--is-ancestor", expected_head, "HEAD"]
            merge_check = self.run_owner_git(
                merge_check_arguments,
                check=False,
                capture_output=True,
                text=True,
            )
            if merge_check.returncode != 0:
                return subprocess.CalledProcessError(
                    merge_check.returncode,
                    ["git", *merge_check_arguments],
                    stderr="branch is not fully merged into the owning worktree HEAD",
                )
        try:
            self.run_owner_git(
                [
                    "update-ref",
                    "-d",
                    f"refs/heads/{self.session.worktree_branch}",
                    expected_head,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except BaseException as error:
            return error
        return None


def _open_parent_relative(parent_descriptor: int, relative: str) -> tuple[int, str]:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise OSError(f"Unsafe relative directory: {relative!r}")
    if len(parts) == 1:
        return os.dup(parent_descriptor), parts[0]
    return _open_relative_directory(parent_descriptor, str(Path(*parts[:-1]))), parts[-1]


def _unlink_verified(parent_descriptor: int, name: str) -> None:
    entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(entry_stat.st_mode):
        os.unlink(name, dir_fd=parent_descriptor)
        return
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened_stat = os.fstat(descriptor)
        if not _same_stat(entry_stat, opened_stat):
            raise OSError(f"Entry changed before unlink: {name}")
        os.unlink(name, dir_fd=parent_descriptor)
    finally:
        os.close(descriptor)


def _rmdir_verified(parent_descriptor: int, name: str, descriptor: int) -> None:
    entry_stat = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    opened_stat = os.fstat(descriptor)
    if not _same_stat(entry_stat, opened_stat):
        raise OSError(f"Directory changed before removal: {name}")
    os.rmdir(name, dir_fd=parent_descriptor)


def _empty_directory(descriptor: int) -> None:
    for name in os.listdir(descriptor):
        entry_stat = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(entry_stat.st_mode):
            child = _open_directory_at(descriptor, name)
            try:
                if not _same_stat(entry_stat, os.fstat(child)):
                    raise OSError(f"Directory changed while opening: {name}")
                _empty_directory(child)
                _rmdir_verified(descriptor, name, child)
            finally:
                os.close(child)
        else:
            _unlink_verified(descriptor, name)


def _classify_porcelain_status(status: str) -> str:
    if status == "??":
        return "untracked"
    if status == "!!":
        return "ignored"
    if status in _UNMERGED_STATUSES:
        return "unmerged"
    return "tracked_or_submodule"


def _parse_porcelain_status(output: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in output.splitlines():
        if len(line) < 3 or line[2] != " ":
            raise ValueError("unexpected Git status porcelain record")
        status = line[:2]
        entries.append(
            {
                "kind": _classify_porcelain_status(status),
                "status": status,
                "path": line[3:],
            }
        )
    return entries


def _checkout_preflight(lifecycle: WorktreeLifecycle, state: WorktreeState) -> WorktreePreflight:
    session = lifecycle.session
    preflight = WorktreePreflight()

    if state.registered_branch != session.worktree_branch:
        status = "detached" if state.registered_branch is None else "switched_branch"
        preflight.local_state.append(
            {
                "kind": "checkout_mismatch",
                "status": status,
                "path": state.registered_branch or "HEAD",
            }
        )

    try:
        symbolic = lifecycle.run_worktree_git(
            ["symbolic-ref", "-q", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        if symbolic.returncode == 0:
            actual_branch = symbolic.stdout.strip()
            if actual_branch != session.worktree_branch and not preflight.local_state:
                preflight.local_state.append(
                    {
                        "kind": "checkout_mismatch",
                        "status": "switched_branch",
                        "path": actual_branch,
                    }
                )
        elif symbolic.returncode == 1:
            if not preflight.local_state:
                preflight.local_state.append(
                    {"kind": "checkout_mismatch", "status": "detached", "path": "HEAD"}
                )
        else:
            preflight.inspection_errors.append(
                f"HEAD attachment inspection failed with status {symbolic.returncode}"
            )

        head = lifecycle.run_worktree_git(
            ["rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        expected_heads = [state.registered_head, state.branch_head]
        if any(expected is not None and expected != head for expected in expected_heads):
            preflight.local_state.append(
                {
                    "kind": "checkout_mismatch",
                    "status": "head_mismatch",
                    "path": "HEAD",
                }
            )
    except (subprocess.CalledProcessError, OSError) as error:
        preflight.inspection_errors.append(
            f"HEAD/branch inspection failed: {_git_failure_detail(error)}"
        )
    return preflight


def _hidden_index_preflight(lifecycle: WorktreeLifecycle) -> WorktreePreflight:
    try:
        result = lifecycle.run_worktree_git(
            ["ls-files", "-v", "-z"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, OSError) as error:
        return WorktreePreflight(
            inspection_errors=[f"index flag inspection failed: {_git_failure_detail(error)}"]
        )

    preflight = WorktreePreflight()
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        tag_bytes, separator, path_bytes = record.partition(b" ")
        if not separator or len(tag_bytes) != 1:
            return WorktreePreflight(inspection_errors=["unexpected git ls-files -v record"])
        tag = tag_bytes.decode("ascii", errors="replace")
        path = path_bytes.decode("utf-8", errors="surrogateescape")
        if tag.islower():
            preflight.local_state.append({"kind": "assume-unchanged", "status": tag, "path": path})
        if tag.upper() == "S":
            preflight.local_state.append({"kind": "skip-worktree", "status": tag, "path": path})
    return preflight


def _removal_preflight(lifecycle: WorktreeLifecycle, state: WorktreeState) -> WorktreePreflight:
    preflight = _checkout_preflight(lifecycle, state)

    try:
        status = lifecycle.run_worktree_git(
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
                "--ignore-submodules=none",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).stdout
        preflight.local_state.extend(_parse_porcelain_status(status))
    except (subprocess.CalledProcessError, OSError, ValueError) as error:
        preflight.inspection_errors.append(
            f"git status inspection failed: {_git_failure_detail(error)}"
        )

    hidden = _hidden_index_preflight(lifecycle)
    preflight.local_state.extend(hidden.local_state)
    preflight.inspection_errors.extend(hidden.inspection_errors)

    try:
        admin_fd = lifecycle._open_worktree_admin()
        try:
            for marker, operation in _OPERATION_MARKERS.items():
                try:
                    os.stat(marker, dir_fd=admin_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                preflight.local_state.append(
                    {"kind": "operation", "status": operation, "path": marker}
                )
        finally:
            os.close(admin_fd)
    except OSError as error:
        preflight.inspection_errors.append(
            f"Git operation-state inspection failed: {_git_failure_detail(error)}"
        )
    return preflight


def _preflight_error_code(preflight: WorktreePreflight) -> str:
    if preflight.inspection_errors:
        return "worktree_inspection_failed"
    if any(item["kind"] == "checkout_mismatch" for item in preflight.local_state):
        return "worktree_checkout_mismatch"
    return "worktree_not_clean"


def _partial_creation_result(
    lifecycle: WorktreeLifecycle,
    error: BaseException,
) -> dict:
    if lifecycle.session.worktree_fingerprint is None:
        lifecycle.capture_worktree_identity(create=True)
    state = lifecycle.inspect()
    uncertain = bool(state.inspection_errors) or None in (
        state.path_exists,
        state.registered,
        state.branch_exists,
    )
    anything_created = uncertain or any(
        value is True for value in (state.path_exists, state.registered, state.branch_exists)
    )
    if anything_created:
        lifecycle.reconcile(state, creation_incomplete=True)

    result = lifecycle.result("enter")
    result.update(
        {
            "error": f"Failed to create worktree: {_git_failure_detail(error)}",
            "worktree_created": state.path_exists is True or state.registered is True,
            "partial_state": anything_created,
            "session_preserved": anything_created,
            "cleanup_required": anything_created,
            "session_state": lifecycle.session.phase if anything_created else None,
            "message": (
                "Creation failed after Git changed or may have changed repository state. "
                "A recoverable cleanup session was retained."
                if anything_created
                else "Creation failed before any worktree state was created."
            ),
        }
    )
    lifecycle.add_state(result, state)
    return result


@_serialized_worktree_operation
def enter_worktree(name: Optional[str] = None) -> str:
    """Create an isolated Git worktree for parallel development."""

    current = _get_worktree_session()
    if current is not None:
        return json.dumps(
            {
                "message": f"Already in a worktree session at {current.worktree_path}. "
                "Exit the current worktree first.",
                "session_state": current.phase,
            }
        )

    if name is None:
        import uuid

        name = f"wt-{uuid.uuid4().hex[:8]}"

    validation_error = _validate_slug(name)
    if validation_error:
        return json.dumps({"message": f"Invalid worktree name: {validation_error}"})

    try:
        identity = _git_identity(get_execution_cwd())
    except OSError as error:
        return json.dumps(
            {
                "error": f"Failed to locate Git repository: {_git_failure_detail(error)}",
                "message": "No worktree was created.",
            }
        )
    if identity is None:
        return json.dumps({"message": "Not in a git repository. Cannot create worktree."})

    branch_name = f"worktree-{name.replace('/', '+')}"
    approved_root, worktree_path, path_error = _prepare_worktree_path(identity, name)
    if path_error:
        return json.dumps({"error": path_error, "message": "No worktree was created."})
    if _path_exists(worktree_path):
        return json.dumps(
            {
                "error": f"Worktree path already exists: {worktree_path}",
                "message": "Existing paths are never adopted by enter_worktree.",
            }
        )

    session = WorktreeSession(
        original_cwd=str(get_execution_cwd()),
        owner_root=str(identity.root),
        owner_common_dir=str(identity.common_dir),
        approved_root=str(approved_root),
        worktree_path=str(worktree_path),
        worktree_branch=branch_name,
        name=name,
        phase="worktree_cleanup_pending",
    )
    lifecycle, lifecycle_error = WorktreeLifecycle.create(session, identity)
    if lifecycle is None:
        return json.dumps(
            {
                "error": f"Failed to bind owning Git repository identity: {lifecycle_error}",
                "message": "No worktree was created.",
            }
        )

    try:
        branch_exists, _, branch_error = _branch_state(lifecycle.root_fd, branch_name)
        if branch_error:
            return json.dumps(
                {
                    "error": f"Failed to inspect target branch: {branch_error}",
                    "message": "No worktree was created.",
                }
            )
        if branch_exists:
            return json.dumps(
                {
                    "error": f"Branch already exists: {branch_name}",
                    "message": "Existing branches are never reset or adopted by enter_worktree.",
                }
            )

        try:
            lifecycle.run_owner_git(
                [
                    "worktree",
                    "add",
                    "-b",
                    branch_name,
                    str(Path(".koder") / "worktrees" / name),
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except BaseException as error:
            result = _partial_creation_result(lifecycle, error)
            if not isinstance(error, Exception):
                raise
            return json.dumps(result)

        instance_error = lifecycle.capture_worktree_identity(create=True)
        state = lifecycle.inspect()
        complete = (
            instance_error is None
            and _validate_recorded_path(session) is None
            and not state.inspection_errors
            and state.worktree_owned
            and state.registered_branch == branch_name
            and state.branch_exists is True
        )
        session.branch_owned = complete
        lifecycle.reconcile(state, creation_incomplete=not complete)

        result = lifecycle.result("enter")
        lifecycle.add_state(result, state)
        result.update(
            {
                "worktree_created": state.path_exists is True or state.registered is True,
                "partial_state": not complete,
                "session_preserved": True,
                "cleanup_required": not complete,
            }
        )
        if complete:
            result["message"] = f"Created and entered worktree at {worktree_path}"
        else:
            result.update(
                {
                    "error": instance_error
                    or "Git did not establish the exact expected worktree ownership.",
                    "message": "Creation produced partial state. A recoverable cleanup session was retained.",
                }
            )
        return json.dumps(result)
    finally:
        lifecycle.close()


def _cleanup_branch(
    lifecycle: WorktreeLifecycle,
    state: WorktreeState,
    discard_changes: bool,
    result: dict,
) -> str:

    lifecycle.reconcile(state)
    lifecycle.add_state(result, state)
    result["worktree_removed"] = state.worktree_absent
    result["session_state"] = (
        lifecycle.session.phase if lifecycle.owner.session is not None else None
    )

    if not state.worktree_absent:
        result.update(
            {
                "error": "Branch cleanup is blocked because worktree removal is not complete.",
                "branch_deleted": False,
                "session_preserved": True,
                "message": "The retryable cleanup session was preserved.",
            }
        )
        return json.dumps(result)
    if state.branch_exists is False:
        result.update(
            {
                "branch_deleted": True,
                "session_preserved": False,
                "session_state": None,
                "message": "Exited worktree. Worktree and branch are absent.",
            }
        )
        return json.dumps(result)
    if state.branch_exists is not True:
        result.update(
            {
                "error": "Branch existence could not be verified.",
                "branch_deleted": None,
                "session_preserved": True,
                "message": "The retryable branch-cleanup session was preserved.",
            }
        )
        return json.dumps(result)
    if lifecycle.session.branch_cleanup_head is None:
        result.update(
            {
                "error_code": "branch_ownership_unproven",
                "error": (
                    "The branch was not captured from Koder's verified successful removal of "
                    "this exact worktree instance."
                ),
                "branch_deleted": False,
                "session_preserved": True,
                "message": "The branch was preserved for manual inspection.",
            }
        )
        return json.dumps(result)

    try:
        failure = lifecycle.delete_owned_branch(discard_changes)
    except BaseException as error:
        failure = error
    final_state = lifecycle.inspect()
    lifecycle.reconcile(final_state)
    lifecycle.add_state(result, final_state)
    branch_deleted = final_state.branch_exists is False
    result.update(
        {
            "worktree_removed": final_state.worktree_absent,
            "branch_deleted": branch_deleted,
            "session_preserved": lifecycle.owner.session is not None,
            "session_state": (
                lifecycle.session.phase if lifecycle.owner.session is not None else None
            ),
        }
    )

    if failure is not None:
        if branch_deleted:
            result.update(
                {
                    "warning": f"Git reported a branch deletion failure: {_git_failure_detail(failure)}",
                    "message": "The worktree was removed and the branch is absent.",
                }
            )
        else:
            result.update(
                {
                    "error": (
                        f"Worktree removed, but failed to delete branch "
                        f"{lifecycle.session.worktree_branch!r}: {_git_failure_detail(failure)}"
                    ),
                    "message": (
                        "The worktree is gone and a retryable branch-cleanup session was retained. "
                        "Use discard_changes=true only to force deletion of recoverable branch work."
                    ),
                }
            )
        if not isinstance(failure, Exception):
            raise failure
        return json.dumps(result)

    path_error = lifecycle.repository_path_error()
    if path_error is not None:
        result.update(
            {
                "error_code": "repository_identity_mismatch",
                "error": path_error,
                "message": (
                    "The descriptor-bound branch mutation completed, but the recorded repository "
                    "path changed during cleanup. Replacement state was not mutated."
                ),
            }
        )
        return json.dumps(result)
    result["message"] = "Exited worktree. Worktree and branch removed."
    return json.dumps(result)


@_serialized_worktree_operation
def exit_worktree(
    action: Literal["keep", "remove"],
    discard_changes: Optional[bool] = None,
) -> str:
    """Exit the recorded worktree session, optionally removing owned Git state."""

    if action not in ("keep", "remove"):
        return json.dumps({"error": f"Invalid action {action!r}. Expected 'keep' or 'remove'."})
    session = _get_worktree_session()
    if session is None:
        return json.dumps(
            {
                "message": "No active worktree session. This tool only operates on "
                "worktrees created by enter_worktree in the current session."
            }
        )

    if action == "keep":
        _set_worktree_session(None)
        return json.dumps(
            {
                "action": action,
                "original_cwd": session.original_cwd,
                "owner_root": session.owner_root,
                "owner_common_dir": session.owner_common_dir,
                "worktree_path": session.worktree_path,
                "worktree_branch": session.worktree_branch,
                "session_preserved": False,
                "session_state": None,
                "message": f"Exited worktree. Worktree kept at {session.worktree_path}",
            }
        )

    lifecycle, owner_error = WorktreeLifecycle.reopen(session)
    if lifecycle is None:
        return json.dumps(
            {
                "action": action,
                "original_cwd": session.original_cwd,
                "owner_root": session.owner_root,
                "owner_common_dir": session.owner_common_dir,
                "worktree_path": session.worktree_path,
                "worktree_branch": session.worktree_branch,
                "session_state": session.phase,
                "error_code": "repository_identity_mismatch",
                "error": owner_error,
                "message": "No cleanup was attempted. The recorded owner could not be revalidated.",
                "worktree_path_exists": _path_exists(session.worktree_path),
                "worktree_registered": None,
                "branch_exists": None,
                "worktree_removed": False,
                "branch_deleted": False,
                "session_preserved": True,
            }
        )

    try:
        return _exit_with_lifecycle(lifecycle, discard_changes)
    finally:
        lifecycle.close()


def _exit_with_lifecycle(lifecycle: WorktreeLifecycle, discard_changes: Optional[bool]) -> str:

    session = lifecycle.session
    result = lifecycle.result("remove")
    path_error = _validate_recorded_path(session)
    if path_error:
        result.update(
            {
                "error_code": "worktree_ownership_mismatch",
                "error": path_error,
                "message": "No cleanup was attempted. The retryable session was preserved.",
                "worktree_path_exists": _path_exists(session.worktree_path),
                "worktree_registered": None,
                "branch_exists": None,
                "worktree_removed": False,
                "branch_deleted": False,
                "session_preserved": True,
            }
        )
        return json.dumps(result)

    state = lifecycle.inspect()
    lifecycle.add_state(result, state)
    if state.inspection_errors:
        result.update(
            {
                "error_code": "worktree_inspection_failed",
                "error": "Cleanup state could not be safely inspected.",
                "message": "No cleanup was attempted. The retryable session was preserved.",
                "worktree_removed": False,
                "branch_deleted": False,
                "session_preserved": True,
            }
        )
        return json.dumps(result)

    if state.worktree_absent:
        return _cleanup_branch(lifecycle, state, discard_changes is True, result)

    if not state.worktree_owned:
        lifecycle.reconcile(state, creation_incomplete=True)
        result.update(
            {
                "error_code": "worktree_ownership_mismatch",
                "error": "The exact recorded linked-worktree instance could not be proven.",
                "message": "No cleanup was attempted. The retryable session was preserved.",
                "worktree_removed": False,
                "branch_deleted": False,
                "session_preserved": True,
                "session_state": lifecycle.session.phase,
            }
        )
        return json.dumps(result)

    if discard_changes is not True:
        preflight = _removal_preflight(lifecycle, state)
        if preflight.local_state or preflight.inspection_errors:
            result.update(
                {
                    "error_code": _preflight_error_code(preflight),
                    "error": "Worktree removal blocked because default cleanup could lose work.",
                    "message": (
                        "No removal was attempted. Commit, stash, or clear the listed state, choose "
                        "action='keep', or explicitly retry with discard_changes=true if it may be destroyed."
                    ),
                    "local_state": preflight.local_state,
                    "worktree_removed": False,
                    "branch_deleted": False,
                    "session_preserved": True,
                }
            )
            if preflight.inspection_errors:
                result.setdefault("inspection_errors", []).extend(preflight.inspection_errors)
            return json.dumps(result)

    try:
        failure = lifecycle.remove_owned_worktree(state)
    except BaseException as error:
        failure = error
    final_state = lifecycle.inspect()
    lifecycle.reconcile(final_state, creation_incomplete=not final_state.worktree_absent)
    lifecycle.add_state(result, final_state)
    result.update(
        {
            "worktree_removed": final_state.worktree_absent,
            "branch_deleted": final_state.branch_exists is False,
            "session_preserved": lifecycle.owner.session is not None,
            "session_state": (
                lifecycle.session.phase if lifecycle.owner.session is not None else None
            ),
        }
    )
    if failure is not None:
        result.update(
            {
                "error": f"Failed to remove worktree: {_git_failure_detail(failure)}",
                "message": (
                    "Forced removal was already requested. Cleanup may have partially mutated the "
                    "verified owned worktree; state was reconciled and replacement paths were not adopted."
                    if discard_changes is True
                    else "Cleanup may have partially mutated the verified owned worktree. Repository "
                    "state was reconciled and replacement paths were not adopted."
                ),
            }
        )
        if isinstance(failure, RepositoryIdentityError):
            result["error_code"] = "repository_identity_mismatch"
        if not isinstance(failure, Exception):
            raise failure
        return json.dumps(result)

    return _cleanup_branch(lifecycle, final_state, discard_changes is True, result)


enter_worktree_tool = function_tool(enter_worktree)
exit_worktree_tool = function_tool(exit_worktree)
