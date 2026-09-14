"""MCP server configuration management across user, local, and project scopes."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional

import aiosqlite

from ..config import MCPLocalProjectConfigYaml, MCPServerConfigYaml, get_config_manager
from ..utils.sqlite_connections import sqlite_connection
from .server_config import MCPServerConfig, MCPServerScope, MCPServerType

logger = logging.getLogger(__name__)

# File-based locking: use fcntl on Unix, fall back to no-op on Windows/other platforms.
try:
    import fcntl

    _HAS_FCNTL = True
except ImportError:  # pragma: no cover
    _HAS_FCNTL = False

_PROJECT_MCP_FILENAME = ".mcp.json"
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
_PROJECT_HELPER_SHELL_SYNTAX = re.compile(r"[\n\r;&|<>`$()]")


class MCPServerManager:
    """Manages MCP server configurations in runtime-owned config files."""

    def __init__(self):
        self.config_manager = get_config_manager()
        self._migration_lock = asyncio.Lock()
        self._migration_completed = False

    @contextmanager
    def _config_lock(self) -> Generator[None, None, None]:
        """Acquire an exclusive file lock around config write operations.

        Uses fcntl.flock on Unix to prevent concurrent koder instances from
        clobbering each other's config writes. Falls back to a no-op on
        platforms where fcntl is unavailable (e.g. Windows).
        """
        if not _HAS_FCNTL:
            yield
            return

        lock_path = Path(str(self.config_manager.config_path) + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = lock_path.open("w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()

    @staticmethod
    def _legacy_db_path() -> Path:
        return Path.home() / ".koder" / "koder.db"

    @staticmethod
    def _cwd_path(cwd: str | Path | None = None) -> Path:
        return Path(cwd or os.getcwd()).resolve()

    @staticmethod
    def _project_config_path(cwd: str | Path | None = None) -> Path:
        return MCPServerManager._cwd_path(cwd) / _PROJECT_MCP_FILENAME

    @staticmethod
    def project_boundary(cwd: str | Path | None = None) -> Path:
        """Return the repository/workspace boundary for project MCP discovery."""
        current = MCPServerManager._cwd_path(cwd)
        for candidate in (current, *current.parents):
            if (candidate / ".git").exists():
                return candidate
        return current

    @staticmethod
    def _normalize_scope(scope: MCPServerScope | str | None) -> MCPServerScope:
        if scope is None:
            return MCPServerScope.LOCAL
        if isinstance(scope, MCPServerScope):
            return scope
        return MCPServerScope(scope)

    @staticmethod
    def _scope_source_path(scope: MCPServerScope, cwd: str | Path | None = None) -> str:
        if scope == MCPServerScope.PROJECT:
            return str(MCPServerManager._project_config_path(cwd))
        return str(get_config_manager().config_path)

    @staticmethod
    def _project_candidate_paths(cwd: str | Path | None = None) -> list[Path]:
        current = MCPServerManager._cwd_path(cwd)
        boundary = MCPServerManager.project_boundary(current)
        candidates: list[Path] = []
        candidate = current
        while True:
            candidates.append(candidate)
            if candidate == boundary:
                break
            candidate = candidate.parent
        candidates.reverse()
        return [candidate / _PROJECT_MCP_FILENAME for candidate in candidates]

    @staticmethod
    def _path_contains(parent: Path, child: Path) -> bool:
        try:
            child.relative_to(parent)
            return True
        except ValueError:
            return False

    @classmethod
    def validate_project_source_path(
        cls,
        source_path: str | Path,
        project_root: str | Path,
    ) -> Path:
        """Return a canonical in-project source path, rejecting source symlinks.

        Project MCP approval is intentionally bound to a regular file reached
        without traversing symlinks. This prevents a repository path from being
        retargeted outside the reviewed workspace after approval.
        """
        root = Path(project_root).expanduser().resolve(strict=True)
        source = Path(source_path).expanduser()
        if not source.is_absolute():
            source = root / source
        source = source.absolute()

        try:
            relative = source.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Project MCP source is outside project root: {source}") from exc

        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"Project MCP source symlinks are not allowed: {current}")

        resolved = source.resolve(strict=True)
        if not cls._path_contains(root, resolved):
            raise ValueError(f"Project MCP source resolves outside project root: {source}")
        if not resolved.is_file():
            raise ValueError(f"Project MCP source is not a regular file: {resolved}")
        return resolved

    @staticmethod
    def _source_content_digest(source_path: str | Path) -> str:
        return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()

    @staticmethod
    def _reviewed_path(env_vars: dict[str, str], execution_cwd: Path) -> str:
        """Return an explicit absolute PATH whose relative entries use the reviewed cwd."""
        configured = env_vars.get("PATH", os.environ.get("PATH", os.defpath))
        entries: list[str] = []
        for raw_entry in configured.split(os.pathsep):
            entry = Path(raw_entry or execution_cwd).expanduser()
            if not entry.is_absolute():
                entry = execution_cwd / entry
            entries.append(str(entry.absolute()))
        return os.pathsep.join(entries)

    @staticmethod
    def _executable_fingerprint(path: Path) -> dict[str, object]:
        digest = hashlib.sha256()
        with path.open("rb") as executable:
            for chunk in iter(lambda: executable.read(1024 * 1024), b""):
                digest.update(chunk)
            stat = os.fstat(executable.fileno())
        return {"sha256": digest.hexdigest(), "size": stat.st_size}

    @classmethod
    def _resolve_reviewed_executable(
        cls,
        command: str,
        *,
        reviewed_path: str,
        execution_cwd: Path,
        label: str,
    ) -> tuple[str, dict[str, object]]:
        """Resolve and fingerprint one executable without accepting a symlink target."""
        has_separator = os.sep in command or (os.altsep is not None and os.altsep in command)
        command_path = Path(command).expanduser()
        if command_path.is_absolute() or has_separator:
            candidate = command_path if command_path.is_absolute() else execution_cwd / command_path
            candidate = candidate.absolute()
        else:
            located = shutil.which(command, path=reviewed_path)
            if located is None:
                raise ValueError(
                    f"{label} executable could not be resolved on reviewed PATH: {command}"
                )
            candidate = Path(located).absolute()

        if candidate.is_symlink():
            raise ValueError(f"{label} executable symlinks are not allowed: {candidate}")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{label} executable could not be resolved: {candidate}") from exc
        if not resolved.is_file():
            raise ValueError(f"{label} executable is not a regular file: {resolved}")
        if not os.access(resolved, os.X_OK):
            raise ValueError(f"{label} executable is not executable: {resolved}")
        return str(resolved), cls._executable_fingerprint(resolved)

    @classmethod
    def _project_headers_helper_argv(
        cls,
        helper: str,
        *,
        reviewed_path: str,
        execution_cwd: Path,
    ) -> tuple[list[str], dict[str, object]]:
        if _PROJECT_HELPER_SHELL_SYNTAX.search(helper):
            raise ValueError(
                "Project headersHelper must be a shell-free command; "
                "shell metacharacters and substitutions are not allowed"
            )
        try:
            argv = shlex.split(helper, posix=os.name != "nt")
        except ValueError as exc:
            raise ValueError(f"Invalid project headersHelper argv: {exc}") from exc
        if not argv:
            raise ValueError("Project headersHelper must contain an executable")
        executable, fingerprint = cls._resolve_reviewed_executable(
            argv[0],
            reviewed_path=reviewed_path,
            execution_cwd=execution_cwd,
            label="Project headersHelper",
        )
        reviewed_argv = [executable, *argv[1:]]
        return reviewed_argv, {
            "executable": executable,
            "argv": reviewed_argv,
            "fingerprint": fingerprint,
        }

    async def _ensure_legacy_migration(self) -> None:
        if self._migration_completed:
            return

        async with self._migration_lock:
            if self._migration_completed:
                return

            db_path = self._legacy_db_path()
            if not db_path.exists():
                self._migration_completed = True
                return

            try:
                async with sqlite_connection(db_path) as conn:
                    cursor = await conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='mcp_servers'"
                    )
                    if not await cursor.fetchone():
                        self._migration_completed = True
                        return

                    conn.row_factory = aiosqlite.Row
                    data_cursor = await conn.execute("SELECT * FROM mcp_servers")
                    rows = await data_cursor.fetchall()
            except Exception:
                self._migration_completed = True
                return

            if not rows:
                self._migration_completed = True
                return

            koder_config = self.config_manager.load()
            existing_names = {server.name for server in koder_config.mcp_servers}
            migrated = 0

            for row in rows:
                try:
                    config = MCPServerConfig.from_db_dict(dict(row))
                except Exception:
                    continue
                if config.name in existing_names:
                    continue
                koder_config.mcp_servers.append(self._mcp_config_to_yaml(config))
                existing_names.add(config.name)
                migrated += 1

            if migrated:
                self.config_manager.save(koder_config)

            self._migration_completed = True

    def _yaml_to_mcp_config(
        self,
        yaml_config: MCPServerConfigYaml,
        *,
        scope: MCPServerScope | None = None,
        source_path: str | None = None,
    ) -> MCPServerConfig:
        return MCPServerConfig(
            name=yaml_config.name,
            transport_type=MCPServerType(yaml_config.transport_type),
            command=yaml_config.command,
            args=yaml_config.args or [],
            env_vars=yaml_config.env_vars or {},
            url=yaml_config.url,
            headers=yaml_config.headers or {},
            cache_tools_list=yaml_config.cache_tools_list,
            allowed_tools=yaml_config.allowed_tools,
            blocked_tools=yaml_config.blocked_tools,
            scope=scope,
            source_path=source_path,
        )

    def _mcp_config_to_yaml(self, config: MCPServerConfig) -> MCPServerConfigYaml:
        return MCPServerConfigYaml(
            name=config.name,
            transport_type=config.transport_type.value,
            command=config.command,
            args=config.args or [],
            env_vars=config.env_vars or {},
            url=config.url,
            headers=config.headers or {},
            cache_tools_list=config.cache_tools_list,
            allowed_tools=config.allowed_tools,
            blocked_tools=config.blocked_tools,
        )

    def _yaml_servers_to_unique_map(
        self,
        entries: list[tuple[MCPServerConfigYaml, str]],
        *,
        scope: MCPServerScope,
    ) -> dict[str, MCPServerConfig]:
        """Convert list-backed configs without silently dropping duplicates."""
        servers: dict[str, MCPServerConfig] = {}
        origins: dict[str, str] = {}
        source_path = str(self.config_manager.config_path)

        for server, origin in entries:
            previous_origin = origins.get(server.name)
            if previous_origin is not None:
                raise ValueError(
                    f"Duplicate MCP server name '{server.name}' in {scope.value}-scoped "
                    f"configured definitions at {source_path}: {previous_origin} conflicts "
                    f"with {origin}. Exact names must be unique within that scope."
                )
            origins[server.name] = origin
            servers[server.name] = self._yaml_to_mcp_config(
                server,
                scope=scope,
                source_path=source_path,
            )

        return servers

    def _local_scope_entry(
        self,
        cwd: str | Path | None,
        *,
        create: bool = False,
    ) -> MCPLocalProjectConfigYaml | None:
        koder_config = self.config_manager.load()
        project_root = str(self._cwd_path(cwd))
        for entry in koder_config.mcp_local_projects:
            if Path(entry.project_root).resolve() == Path(project_root):
                return entry
        if not create:
            return None
        entry = MCPLocalProjectConfigYaml(project_root=project_root, servers=[])
        koder_config.mcp_local_projects.append(entry)
        return entry

    def _load_user_servers(self) -> dict[str, MCPServerConfig]:
        koder_config = self.config_manager.load()
        entries = [
            (server, f"user list entry {index}")
            for index, server in enumerate(koder_config.mcp_servers, start=1)
        ]
        return self._yaml_servers_to_unique_map(entries, scope=MCPServerScope.USER)

    def _load_local_servers(self, cwd: str | Path | None = None) -> dict[str, MCPServerConfig]:
        koder_config = self.config_manager.load()
        cwd_path = self._cwd_path(cwd)
        matching_entries = [
            entry
            for entry in koder_config.mcp_local_projects
            if self._path_contains(Path(entry.project_root).resolve(), cwd_path)
        ]
        matching_entries.sort(key=lambda entry: len(Path(entry.project_root).parts))

        entries = [
            (
                server,
                f"local list entry {index} for project root '{Path(entry.project_root).resolve()}'",
            )
            for entry in matching_entries
            for index, server in enumerate(entry.servers, start=1)
        ]
        return self._yaml_servers_to_unique_map(entries, scope=MCPServerScope.LOCAL)

    @staticmethod
    def _expand_env_string(value: str) -> str:
        def _replace(match: re.Match[str]) -> str:
            name = match.group(1)
            default = match.group(2)
            resolved = os.environ.get(name)
            if resolved is not None:
                return resolved
            if default is not None:
                return default
            raise ValueError(f"Missing required environment variable: {name}")

        return _ENV_VAR_PATTERN.sub(_replace, value)

    @classmethod
    def _expand_env_data(cls, value):
        if isinstance(value, str):
            return cls._expand_env_string(value)
        if isinstance(value, list):
            return [cls._expand_env_data(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._expand_env_data(item) for key, item in value.items()}
        return value

    def _config_from_mapping(
        self,
        name: str,
        mapping: dict,
        *,
        scope: MCPServerScope,
        source_path: str,
        expand_env: bool,
    ) -> MCPServerConfig:
        raw = self._expand_env_data(mapping) if expand_env else mapping
        transport = raw.get("type")
        if transport is None:
            if raw.get("command"):
                transport = MCPServerType.STDIO.value
            elif raw.get("url"):
                transport = MCPServerType.HTTP.value
        if transport is None:
            raise ValueError(f"MCP server '{name}' is missing a transport type")

        return MCPServerConfig(
            name=name,
            transport_type=MCPServerType(transport),
            command=raw.get("command"),
            args=raw.get("args") or [],
            env_vars=raw.get("env") or raw.get("env_vars") or {},
            url=raw.get("url"),
            headers=raw.get("headers") or {},
            headers_helper=raw.get("headersHelper") or raw.get("headers_helper"),
            oauth=raw.get("oauth"),
            cache_tools_list=bool(
                raw.get("cacheToolsList") or raw.get("cache_tools_list") or False
            ),
            allowed_tools=raw.get("allowedTools") or raw.get("allowed_tools"),
            blocked_tools=raw.get("blockedTools") or raw.get("blocked_tools"),
            scope=scope,
            source_path=source_path,
        )

    def _read_project_config(self, path: Path) -> dict:
        if not path.exists():
            return {"mcpServers": {}}
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid MCP config at {path}: expected an object")
        raw_servers = raw.get("mcpServers")
        if raw_servers is None:
            return {"mcpServers": {}}
        if not isinstance(raw_servers, dict):
            raise ValueError(f"Invalid MCP config at {path}: mcpServers must be an object")
        return {"mcpServers": raw_servers}

    def _write_project_config(self, path: Path, servers: dict[str, dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mcpServers": servers}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def project_source_config_groups(
        self, cwd: str | Path | None = None
    ) -> list[list[MCPServerConfig]]:
        """Load each in-repository ``.mcp.json`` as a provenance-bound group."""
        project_root = self.project_boundary(cwd)
        groups: list[list[MCPServerConfig]] = []
        for candidate in self._project_candidate_paths(cwd):
            if not candidate.exists():
                continue
            try:
                source_path = self.validate_project_source_path(candidate, project_root)
                raw = self._read_project_config(source_path)
                configs = self.build_project_source_configs(
                    raw["mcpServers"],
                    source_path=source_path,
                    project_root=project_root,
                    execution_cwd=project_root,
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping invalid project MCP source '%s': %s", candidate, exc)
                continue
            if configs:
                groups.append(configs)
        return groups

    def _load_project_servers(self, cwd: str | Path | None = None) -> dict[str, MCPServerConfig]:
        servers: dict[str, MCPServerConfig] = {}
        for configs in self.project_source_config_groups(cwd):
            for config in configs:
                servers[config.name] = config
        return servers

    def build_project_source_configs(
        self,
        servers: dict[str, dict],
        *,
        source_path: str | Path,
        project_root: str | Path,
        execution_cwd: str | Path,
    ) -> list[MCPServerConfig]:
        """Build configs whose approval covers expanded values, root, and cwd."""
        root = Path(project_root).expanduser().resolve(strict=True)
        source = self.validate_project_source_path(source_path, root)
        reviewed_cwd = Path(execution_cwd).expanduser().resolve(strict=True)
        if not self._path_contains(root, reviewed_cwd) or not reviewed_cwd.is_dir():
            raise ValueError(
                f"Project MCP execution directory must be inside project root: {reviewed_cwd}"
            )

        configs: list[MCPServerConfig] = []
        canonical_servers: dict[str, object] = {}
        source_template = copy.deepcopy(servers)
        content_digest = self._source_content_digest(source)
        for name in sorted(servers):
            mapping = servers[name]
            if not isinstance(mapping, dict):
                canonical_servers[name] = {"invalid": True, "source": mapping}
                logger.warning(
                    "Skipping invalid project MCP server '%s' from %s: expected an object",
                    name,
                    source,
                )
                continue
            try:
                config = self._config_from_mapping(
                    name,
                    mapping,
                    scope=MCPServerScope.PROJECT,
                    source_path=str(source),
                    expand_env=True,
                )
                env_vars = dict(config.env_vars or {})
                reviewed_path = self._reviewed_path(env_vars, reviewed_cwd)
                env_vars["PATH"] = reviewed_path
                config.env_vars = env_vars

                descriptor: dict[str, object] = {
                    "cwd": str(reviewed_cwd),
                    "path": reviewed_path,
                }
                if config.transport_type == MCPServerType.STDIO:
                    if not config.command:
                        raise ValueError(f"Project MCP server '{name}' is missing a command")
                    executable, fingerprint = self._resolve_reviewed_executable(
                        config.command,
                        reviewed_path=reviewed_path,
                        execution_cwd=reviewed_cwd,
                        label=f"Project MCP server '{name}'",
                    )
                    config.command = executable
                    descriptor["stdio"] = {
                        "executable": executable,
                        "argv": [executable, *(config.args or [])],
                        "fingerprint": fingerprint,
                    }
                else:
                    descriptor["url"] = config.url

                if config.headers_helper:
                    helper_argv, helper_descriptor = self._project_headers_helper_argv(
                        config.headers_helper,
                        reviewed_path=reviewed_path,
                        execution_cwd=reviewed_cwd,
                    )
                    config.headers_helper_argv = helper_argv
                    descriptor["headersHelper"] = helper_descriptor

                config.execution_descriptor = descriptor
                canonical_servers[name] = {
                    **self._serialize_project_server(config),
                    "executionDescriptor": descriptor,
                }
                configs.append(config)
            except (OSError, ValueError) as exc:
                canonical_servers[name] = {
                    "invalid": True,
                    "source": mapping,
                    "error": str(exc),
                }
                logger.warning(
                    "Skipping invalid project MCP server '%s' from %s: %s",
                    name,
                    source,
                    exc,
                )
                continue

        if not configs:
            return []

        canonical = json.dumps(
            {
                "projectRoot": str(root),
                "sourcePath": str(source),
                "executionCwd": str(reviewed_cwd),
                "mcpServers": canonical_servers,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        source_digest = hashlib.sha256(canonical).hexdigest()
        for config in configs:
            config.source_digest = source_digest
            config.project_root = str(root)
            config.execution_cwd = str(reviewed_cwd)
            config.source_content_digest = content_digest
            config.source_template = copy.deepcopy(source_template)
        return configs

    def revalidate_project_config(
        self,
        config: MCPServerConfig,
        *,
        approval_lock_held: bool = False,
    ) -> bool:
        """Recheck current source, environment expansion, root, cwd, and approval.

        ``approval_lock_held`` is reserved for the runtime authorization
        admission path. It lets that path keep the cross-process approval lock
        held until the operation has been recorded as in flight.
        """
        if config.scope != MCPServerScope.PROJECT:
            return True
        if not all(
            [
                config.source_path,
                config.source_digest,
                config.project_root,
                config.execution_cwd,
                config.execution_descriptor is not None,
                config.source_content_digest,
                config.source_template is not None,
            ]
        ):
            return False

        try:
            root = Path(config.project_root).resolve(strict=True)
            reviewed_cwd = Path(config.execution_cwd).resolve(strict=True)
            source = self.validate_project_source_path(config.source_path, root)
            if not self._path_contains(root, reviewed_cwd) or not reviewed_cwd.is_dir():
                return False
            if self._source_content_digest(source) != config.source_content_digest:
                return False
            refreshed = self.build_project_source_configs(
                config.source_template or {},
                source_path=source,
                project_root=root,
                execution_cwd=reviewed_cwd,
            )
        except (OSError, ValueError):
            return False

        refreshed_config = next((item for item in refreshed if item.name == config.name), None)
        if refreshed_config is None or refreshed_config.source_digest != config.source_digest:
            return False
        if refreshed_config.execution_descriptor != config.execution_descriptor:
            return False

        if approval_lock_held:
            from .project_approvals import _is_project_connect_allowed_unlocked

            return _is_project_connect_allowed_unlocked(
                project_root=root,
                source_path=source,
                source_digest=config.source_digest,
            )

        from .project_approvals import is_project_connect_allowed

        return is_project_connect_allowed(
            project_root=root,
            source_path=source,
            source_digest=config.source_digest,
        )

    def _serialize_project_server(self, config: MCPServerConfig) -> dict:
        payload: dict[str, object] = {"type": config.transport_type.value}
        if config.command:
            payload["command"] = config.command
        if config.args:
            payload["args"] = list(config.args)
        if config.env_vars:
            payload["env"] = dict(config.env_vars)
        if config.url:
            payload["url"] = config.url
        if config.headers:
            payload["headers"] = dict(config.headers)
        if config.headers_helper:
            payload["headersHelper"] = config.headers_helper
        if config.oauth:
            payload["oauth"] = dict(config.oauth)
        if config.cache_tools_list:
            payload["cacheToolsList"] = True
        if config.allowed_tools:
            payload["allowedTools"] = list(config.allowed_tools)
        if config.blocked_tools:
            payload["blockedTools"] = list(config.blocked_tools)
        return payload

    def _effective_servers(self, cwd: str | Path | None = None) -> dict[str, MCPServerConfig]:
        user = self._load_user_servers()
        project = self._load_project_servers(cwd)
        local = self._load_local_servers(cwd)

        merged = dict(user)

        # Project (.mcp.json, lower trust) must NOT silently override a
        # user/global server of the same name — that would let a checked-in repo
        # file shadow the user's own trusted config. Keep the user entry and log
        # a visible warning on collision.
        for name, config in project.items():
            if name in user:
                logger.warning(
                    "MCP name collision: project server '%s' (from %s) is ignored "
                    "because a higher-trust user-scoped server with the same name exists.",
                    name,
                    config.source_path,
                )
                continue
            merged[name] = config

        # Local scope lives in the user's own config.yaml (mcp_local_projects),
        # so it is trusted and may intentionally override user/project entries.
        for name, config in local.items():
            if name in project and name not in user:
                logger.debug(
                    "MCP name collision: local server '%s' overrides the "
                    "project server of the same name.",
                    name,
                )
            merged[name] = config

        return merged

    async def add_server(
        self,
        config: MCPServerConfig,
        *,
        scope: MCPServerScope | str = MCPServerScope.LOCAL,
        cwd: str | Path | None = None,
    ) -> None:
        await self._ensure_legacy_migration()
        normalized_scope = self._normalize_scope(scope)
        if normalized_scope == MCPServerScope.USER:
            with self._config_lock():
                koder_config = self.config_manager.load()
                if any(server.name == config.name for server in koder_config.mcp_servers):
                    raise ValueError(f"MCP server already exists in user scope: {config.name}")
                koder_config.mcp_servers.append(self._mcp_config_to_yaml(config))
                self.config_manager.save(koder_config)
            return

        if normalized_scope == MCPServerScope.LOCAL:
            with self._config_lock():
                koder_config = self.config_manager.load()
                entry = self._local_scope_entry(cwd, create=True)
                assert entry is not None
                if any(server.name == config.name for server in entry.servers):
                    raise ValueError(f"MCP server already exists in local scope: {config.name}")
                entry.servers.append(self._mcp_config_to_yaml(config))
                self.config_manager.save(koder_config)
            return

        project_path = self._project_config_path(cwd)
        with self._config_lock():
            raw = self._read_project_config(project_path)
            if config.name in raw["mcpServers"]:
                raise ValueError(f"MCP server already exists in project scope: {config.name}")
            raw["mcpServers"][config.name] = self._serialize_project_server(config)
            self._write_project_config(project_path, raw["mcpServers"])

    async def import_json_server(
        self,
        name: str,
        json_payload: str,
        *,
        scope: MCPServerScope | str = MCPServerScope.LOCAL,
        cwd: str | Path | None = None,
    ) -> None:
        mapping = json.loads(json_payload)
        normalized_scope = self._normalize_scope(scope)
        config = self._config_from_mapping(
            name,
            mapping,
            scope=normalized_scope,
            source_path=self._scope_source_path(normalized_scope, cwd),
            expand_env=False,
        )
        await self.add_server(config, scope=normalized_scope, cwd=cwd)

    async def get_server(
        self,
        name: str,
        *,
        cwd: str | Path | None = None,
        scope: MCPServerScope | str | None = None,
    ) -> Optional[MCPServerConfig]:
        await self._ensure_legacy_migration()
        if scope is None:
            return self._effective_servers(cwd).get(name)

        normalized_scope = self._normalize_scope(scope)
        if normalized_scope == MCPServerScope.USER:
            return self._load_user_servers().get(name)
        if normalized_scope == MCPServerScope.LOCAL:
            return self._load_local_servers(cwd).get(name)
        return self._load_project_servers(cwd).get(name)

    async def list_servers(
        self,
        *,
        cwd: str | Path | None = None,
        scope: MCPServerScope | str | None = None,
    ) -> List[MCPServerConfig]:
        await self._ensure_legacy_migration()
        if scope is None:
            servers = self._effective_servers(cwd)
        else:
            normalized_scope = self._normalize_scope(scope)
            if normalized_scope == MCPServerScope.USER:
                servers = self._load_user_servers()
            elif normalized_scope == MCPServerScope.LOCAL:
                servers = self._load_local_servers(cwd)
            else:
                servers = self._load_project_servers(cwd)
        return [servers[name] for name in sorted(servers)]

    async def update_server(
        self,
        config: MCPServerConfig,
        *,
        scope: MCPServerScope | str | None = None,
        cwd: str | Path | None = None,
    ) -> bool:
        # Note: update_server delegates to remove_server + add_server which each
        # acquire their own _config_lock. This is safe because fcntl locks are
        # re-entrant within the same process (same fd not reused, each call opens
        # a fresh fd). For cross-process safety the remove+add is NOT atomic, but
        # this matches the pre-existing semantics.
        target_scope = scope or config.scope
        if target_scope is None:
            existing = await self.get_server(config.name, cwd=cwd)
            if existing is None:
                return False
            target_scope = existing.scope
        if not await self.remove_server(config.name, scope=target_scope, cwd=cwd):
            return False
        await self.add_server(config, scope=target_scope, cwd=cwd)
        return True

    async def remove_server(
        self,
        name: str,
        *,
        scope: MCPServerScope | str | None = None,
        cwd: str | Path | None = None,
    ) -> bool:
        await self._ensure_legacy_migration()
        target_scope = scope
        if target_scope is None:
            existing = await self.get_server(name, cwd=cwd)
            if existing is None:
                return False
            target_scope = existing.scope
        normalized_scope = self._normalize_scope(target_scope)

        if normalized_scope == MCPServerScope.USER:
            with self._config_lock():
                koder_config = self.config_manager.load()
                initial_count = len(koder_config.mcp_servers)
                koder_config.mcp_servers = [s for s in koder_config.mcp_servers if s.name != name]
                if len(koder_config.mcp_servers) < initial_count:
                    self.config_manager.save(koder_config)
                    return True
                return False

        if normalized_scope == MCPServerScope.LOCAL:
            with self._config_lock():
                koder_config = self.config_manager.load()
                entry = self._local_scope_entry(cwd, create=False)
                if entry is None:
                    return False
                initial_count = len(entry.servers)
                entry.servers = [server for server in entry.servers if server.name != name]
                if len(entry.servers) == initial_count:
                    return False
                self.config_manager.save(koder_config)
                return True

        project_path = self._project_config_path(cwd)
        with self._config_lock():
            raw = self._read_project_config(project_path)
            if name not in raw["mcpServers"]:
                return False
            del raw["mcpServers"][name]
            self._write_project_config(project_path, raw["mcpServers"])
            return True

    async def server_exists(
        self,
        name: str,
        *,
        cwd: str | Path | None = None,
        scope: MCPServerScope | str | None = None,
    ) -> bool:
        return await self.get_server(name, cwd=cwd, scope=scope) is not None

    async def get_servers_by_type(
        self,
        transport_type: str,
        *,
        cwd: str | Path | None = None,
        scope: MCPServerScope | str | None = None,
    ) -> List[MCPServerConfig]:
        normalized = MCPServerType(transport_type)
        servers = await self.list_servers(cwd=cwd, scope=scope)
        return [server for server in servers if server.transport_type == normalized]
