"""Plugin marketplace registry with local and GitHub support."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from koder_agent.utils.atomic_file import write_text_atomic

from .manifest import find_manifest, parse_manifest
from .name_validation import canonical_marketplace_name
from .state import PluginInstallOrigin

_GITHUB_SHORTHAND = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


class MarketplaceStoreError(ValueError):
    """Persisted marketplace data cannot be safely canonicalized."""


@dataclass(frozen=True)
class MarketplaceSource:
    """A registered marketplace source."""

    name: str
    source_type: str  # "directory", "github", "git", "file"
    path: str  # local path (after clone) or original source


@dataclass(frozen=True)
class MarketplacePlugin:
    """A plugin available from a marketplace source."""

    name: str
    version: str
    description: str
    source: str  # marketplace name
    path: str  # local path to plugin directory
    origin: PluginInstallOrigin | None = None


def _marketplace_cache_dir() -> Path:
    from koder_agent.harness.paths import harness_home_dir

    return harness_home_dir() / "plugins" / "marketplace-cache"


def _repository_cache_path(name: str, source: str) -> Path:
    fingerprint = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return _marketplace_cache_dir() / f"{name[:180]}-{fingerprint}"


def _clone_repository(url: str, target: Path) -> bool:
    """Reuse only a checkout of the requested origin, and surface pull failure."""
    try:
        if target.is_symlink():
            return False
        if target.exists():
            remote = subprocess.run(
                ["git", "-C", str(target), "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if remote.returncode != 0 or remote.stdout.strip() != url:
                return False
            result = subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            return result.returncode == 0
        target.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", url, str(target)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _clone_github_repo(repo: str, target: Path) -> bool:
    return _clone_repository(f"https://github.com/{repo}.git", target)


def _parse_marketplace_input(source: str) -> tuple[str, str, str]:
    """Parse a marketplace source string.

    Returns (source_type, name, resolved_path_or_source).
    """
    # GitHub shorthand: owner/repo
    if _GITHUB_SHORTHAND.match(source):
        parts = source.split("/")
        name = parts[-1]
        return "github", name, source

    # Git URL
    if source.startswith("https://") or source.startswith("git@"):
        # Derive name from URL
        name = source.rstrip("/").rsplit("/", 1)[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return "git", name, source

    # Local path
    path = Path(source).resolve()
    name = path.name
    return "directory", name, str(path)


class MarketplaceStore:
    """Reads/writes marketplace sources to a JSON file.

    Supports local directories and GitHub repositories.
    GitHub repos are cloned to ~/.koder/plugins/marketplace-cache/.
    """

    def __init__(self, store_path: Path):
        self._path = store_path

    @classmethod
    def default(cls) -> "MarketplaceStore":
        from koder_agent.harness.paths import harness_home_dir

        return cls(harness_home_dir() / "plugins" / "marketplaces.json")

    @classmethod
    def for_test(cls, root: Path) -> "MarketplaceStore":
        return cls(root / "marketplaces.json")

    def _load(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            raw_data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(raw_data, dict):
            return {}

        migrated: dict[str, dict] = {}
        changed = False
        original_names: dict[str, str] = {}
        for legacy_name, entry in raw_data.items():
            canonical_name, reason = canonical_marketplace_name(legacy_name)
            if canonical_name is None or not isinstance(entry, dict):
                raise MarketplaceStoreError(
                    f"Invalid persisted marketplace '{legacy_name}': {reason or 'invalid entry'}"
                )
            changed = changed or canonical_name != legacy_name
            existing = migrated.get(canonical_name)
            if existing is None:
                migrated[canonical_name] = dict(entry)
                original_names[canonical_name] = legacy_name
                continue
            if self._source_identity(existing) != self._source_identity(entry):
                first_name = original_names[canonical_name]
                raise MarketplaceStoreError(
                    "Persisted marketplace names "
                    f"'{first_name}' and '{legacy_name}' both canonicalize to "
                    f"'{canonical_name}' but reference different sources"
                )
            for key, value in entry.items():
                existing.setdefault(key, value)
            changed = True

        if changed:
            self._save(migrated)
        return migrated

    @staticmethod
    def _source_identity(entry: dict) -> tuple[str, object]:
        source_type = entry.get("source_type", "directory")
        raw_source = entry.get("raw_source")
        if raw_source is None and source_type == "directory":
            raw_source = entry.get("path")
        return str(source_type), raw_source

    def _save(self, data: dict[str, dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(self._path, json.dumps(data, indent=2))

    def add(self, source_input: str) -> tuple[MarketplaceSource | None, str]:
        """Register a marketplace source.

        Accepts: owner/repo (GitHub), git URL, or local path.
        Returns (source, message). source is None on failure.
        """
        source_type, source_name, raw_source = _parse_marketplace_input(source_input)

        name, error_reason = canonical_marketplace_name(source_name)
        if name is None:
            return None, f"Invalid marketplace name: {error_reason}"

        try:
            data = self._load()
        except MarketplaceStoreError as exc:
            return None, str(exc)
        existing = data.get(name)
        if existing is not None:
            existing_type = existing.get("source_type", "directory")
            existing_source = existing.get("raw_source")
            if existing_source is None and existing_type == "directory":
                existing_source = existing.get("path")
            if existing_type == source_type and existing_source == raw_source:
                source = MarketplaceSource(
                    name=name,
                    source_type=existing_type,
                    path=existing.get("path", raw_source),
                )
                return source, f"Marketplace already added: {name}"
            return (
                None,
                f"Marketplace name '{name}' is already registered to a different source; "
                "remove it before adding a replacement",
            )

        local_path = raw_source
        if source_type == "github":
            cache = _repository_cache_path(name, raw_source)
            if not _clone_github_repo(raw_source, cache):
                return None, f"Failed to clone https://github.com/{raw_source}.git"
            local_path = str(cache)
        elif source_type == "git":
            cache = _repository_cache_path(name, raw_source)
            if not _clone_repository(raw_source, cache):
                return None, f"Failed to prepare repository cache for marketplace '{name}'"
            local_path = str(cache)
        elif source_type == "directory":
            if not Path(local_path).is_dir():
                return None, f"Directory not found: {local_path}"

        data[name] = {
            "source_type": source_type,
            "path": local_path,
            "raw_source": raw_source,
        }
        self._save(data)
        return (
            MarketplaceSource(name=name, source_type=source_type, path=local_path),
            f"Added marketplace: {name}",
        )

    def remove(self, name: str) -> bool:
        name, _reason = canonical_marketplace_name(name)
        if name is None:
            return False
        data = self._load()
        if name not in data:
            return False
        del data[name]
        self._save(data)
        return True

    def list_all(self) -> list[MarketplaceSource]:
        data = self._load()
        return [
            MarketplaceSource(
                name=name,
                source_type=entry.get("source_type", "directory"),
                path=entry.get("path", ""),
            )
            for name, entry in data.items()
        ]

    def get(self, name: str) -> MarketplaceSource | None:
        name, _reason = canonical_marketplace_name(name)
        if name is None:
            return None
        data = self._load()
        entry = data.get(name)
        if entry is None:
            return None
        return MarketplaceSource(
            name=name,
            source_type=entry.get("source_type", "directory"),
            path=entry.get("path", ""),
        )

    @staticmethod
    def _entry_origin(name: str, entry: dict) -> PluginInstallOrigin | None:
        source_type = entry.get("source_type", "directory")
        path = entry.get("path")
        raw_source = entry.get("raw_source")
        if raw_source is None and source_type == "directory":
            raw_source = path
        if (
            not isinstance(source_type, str)
            or source_type not in {"directory", "github", "git"}
            or not isinstance(path, str)
            or not path
            or not isinstance(raw_source, str)
            or not raw_source
        ):
            return None
        # Bind both the registered source and its effective local checkout.
        # Store only a digest, not potentially credential-bearing source URLs.
        identity = json.dumps(
            [source_type, raw_source, str(Path(path).expanduser().resolve())],
            separators=(",", ":"),
        )
        return PluginInstallOrigin(name, hashlib.sha256(identity.encode("utf-8")).hexdigest())

    def matches_origin(self, origin: PluginInstallOrigin) -> bool:
        """Check an installation receipt against the current registered source."""
        try:
            entry = self._load().get(origin.marketplace)
            return entry is not None and self._entry_origin(origin.marketplace, entry) == origin
        except (OSError, RuntimeError, ValueError):
            return False

    def discover_plugins(self, marketplace_name: str) -> list[MarketplacePlugin]:
        """List all plugins available from a registered marketplace.

        Scans immediate children of the marketplace root, and also common
        subdirectories like ``plugins/`` and ``external_plugins/`` where
        GitHub-hosted marketplaces typically nest their plugin directories.
        """
        marketplace_name, _reason = canonical_marketplace_name(marketplace_name)
        if marketplace_name is None:
            return []
        entry = self._load().get(marketplace_name)
        if entry is None:
            return []
        source_path = Path(entry.get("path", ""))
        if not source_path.is_dir():
            return []
        origin = self._entry_origin(marketplace_name, entry)

        plugins: list[MarketplacePlugin] = []
        seen_names: set[str] = set()

        # Directories to scan for plugin subdirs
        scan_roots = [source_path]
        for subname in ("plugins", "external_plugins"):
            candidate = source_path / subname
            if candidate.is_dir():
                scan_roots.append(candidate)

        for root in scan_roots:
            for subdir in sorted(root.iterdir()):
                if not subdir.is_dir() or subdir.name.startswith("."):
                    continue
                manifest_path = find_manifest(subdir)
                if manifest_path is None:
                    continue
                manifest, errors, _ = parse_manifest(subdir)
                if manifest is None or errors:
                    continue
                if manifest.name in seen_names:
                    continue
                seen_names.add(manifest.name)
                plugins.append(
                    MarketplacePlugin(
                        name=manifest.name,
                        version=manifest.version,
                        description=manifest.description,
                        source=marketplace_name,
                        path=str(subdir),
                        origin=origin,
                    )
                )
        return plugins

    def find_plugin(self, plugin_id: str) -> MarketplacePlugin | None:
        """Find a plugin by name@marketplace identifier.

        If no @marketplace suffix, searches all marketplaces.
        """
        if "@" in plugin_id:
            name, marketplace = plugin_id.rsplit("@", 1)
            for plugin in self.discover_plugins(marketplace):
                if plugin.name == name:
                    return plugin
            return None

        # Search all marketplaces
        for source in self.list_all():
            for plugin in self.discover_plugins(source.name):
                if plugin.name == plugin_id:
                    return plugin
        return None
