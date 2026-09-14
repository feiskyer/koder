"""Harness CLI handlers for plugin commands."""

from __future__ import annotations

import json
from pathlib import Path

from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
from koder_agent.harness.plugins.manifest import find_manifest, parse_manifest
from koder_agent.harness.plugins.marketplace import MarketplaceStore
from koder_agent.harness.plugins.path_safety import (
    PluginPathError,
    open_plugin_component,
)
from koder_agent.harness.plugins.registry import PluginRegistry


async def handle_plugin_subcommand(args) -> int:
    """Dispatch plugin CLI subcommands."""
    plugin_root = Path.home() / ".koder" / "plugins"
    lifecycle = PluginLifecycleService(plugin_root)

    action = getattr(args, "plugin_action", None) or "list"

    if action == "list":
        return _handle_list(lifecycle, json_output=getattr(args, "json", False))

    if action == "install":
        plugin_ref = getattr(args, "plugin_ref", "")
        scope = getattr(args, "scope", "user") or "user"
        return _handle_install(lifecycle, plugin_ref, scope)

    if action == "uninstall":
        name = getattr(args, "name", "")
        return _handle_uninstall(lifecycle, name)

    if action == "enable":
        name = getattr(args, "name", "")
        return _handle_enable(lifecycle, name)

    if action == "disable":
        name = getattr(args, "name", "")
        return _handle_disable(lifecycle, name)

    if action == "validate":
        path = Path(getattr(args, "path", ".")).resolve()
        return _handle_validate(path)

    if action == "marketplace":
        return _handle_marketplace(args)

    print(f"Unknown plugin action: {action}")
    return 1


def _handle_list(lifecycle: PluginLifecycleService, *, json_output: bool = False) -> int:
    registry = PluginRegistry.from_lifecycle(lifecycle, include_disabled=True)
    plugins = registry.list_plugins()
    if not plugins:
        print("No installed plugins.")
        return 0

    if json_output:
        data = [
            {
                "name": p.name,
                "version": p.version,
                "scope": p.scope,
                "enabled": p.enabled,
                "description": p.description,
                "components": list(p.components),
            }
            for p in plugins
        ]
        print(json.dumps(data, indent=2))
        return 0

    header = f"{'NAME':<24} {'VERSION':<10} {'SCOPE':<8} {'STATUS':<10} COMPONENTS"
    print(header)
    for p in plugins:
        status = "enabled" if p.enabled else "disabled"
        comps = ", ".join(p.components) if p.components else "-"
        print(f"{p.name:<24} {p.version:<10} {p.scope:<8} {status:<10} {comps}")
    return 0


def _handle_install(lifecycle: PluginLifecycleService, plugin_ref: str, scope: str) -> int:
    """Install a plugin from a local path or name@marketplace."""
    origin = None
    expected_name = None
    if Path(plugin_ref).is_dir():
        plugin_dir = Path(plugin_ref).resolve()
    else:
        store = MarketplaceStore.default()
        plugin = store.find_plugin(plugin_ref)
        if plugin is not None:
            plugin_dir = Path(plugin.path)
            origin = plugin.origin
            expected_name = plugin.name
        else:
            print(f"Error: '{plugin_ref}' is not a directory and not found in any marketplace")
            return 1

    result = lifecycle.install_from_dir(
        plugin_dir, scope=scope, origin=origin, expected_name=expected_name
    )
    if result.success:
        print(f"Installed {result.plugin_name}")
    else:
        print(f"Error: {result.message}")
    return 0 if result.success else 1


def _handle_uninstall(lifecycle: PluginLifecycleService, name: str) -> int:
    result = lifecycle.uninstall(name)
    if result.success:
        print(f"Uninstalled {name}")
    else:
        print(f"Error: {result.message}")
    return 0 if result.success else 1


def _handle_enable(lifecycle: PluginLifecycleService, name: str) -> int:
    result = lifecycle.enable(name)
    if result.success:
        print(f"Enabled {name}")
    else:
        print(f"Error: {result.message}")
    return 0 if result.success else 1


def _handle_disable(lifecycle: PluginLifecycleService, name: str) -> int:
    result = lifecycle.disable(name)
    if result.success:
        print(f"Disabled {name}")
    else:
        print(f"Error: {result.message}")
    return 0 if result.success else 1


def _handle_validate(plugin_dir: Path) -> int:
    """Validate a plugin manifest and content."""
    manifest_path = find_manifest(plugin_dir)
    if manifest_path is None:
        print(f"Error: No plugin.json found in {plugin_dir}")
        return 1

    manifest, errors, warnings = parse_manifest(plugin_dir)

    for warning in warnings:
        print(f"  Warning: {warning}")
    for error in errors:
        print(f"  Error: {error}")

    if manifest is None or errors:
        print("Validation failed.")
        return 1

    component_checks = [
        ("skills", manifest.skills, "directory"),
        ("agents", manifest.agents, "directory"),
        ("hooks", manifest.hooks, "file"),
        ("mcpServers", manifest.mcp_servers, "file"),
    ]
    has_component_warning = False
    for field_name, path_val, expect in component_checks:
        if path_val is None:
            continue
        try:
            with open_plugin_component(
                plugin_dir,
                path_val,
                default=path_val,
                field_name=field_name,
                expect=expect,
            ) as full_path:
                missing = full_path is None
        except PluginPathError as exc:
            print(f"  Error: unsafe declared {field_name} path '{path_val}': {exc}")
            return 1
        if missing:
            print(f"  Warning: declared {field_name} path '{path_val}' does not exist")
            has_component_warning = True

    if not has_component_warning and not warnings:
        print(f"Plugin '{manifest.name}' v{manifest.version} is valid.")
    else:
        print(f"Plugin '{manifest.name}' v{manifest.version} is valid (with warnings).")
    return 0


def _handle_marketplace(args) -> int:
    """Handle marketplace subcommands."""
    marketplace_action = getattr(args, "marketplace_action", None) or "list"
    store = MarketplaceStore.default()

    if marketplace_action == "list":
        sources = store.list_all()
        if not sources:
            print("No configured marketplaces.")
            return 0
        for source in sources:
            print(f"  {source.name} ({source.source_type}): {source.path}")
        return 0

    if marketplace_action == "add":
        source_input = getattr(args, "source_path", "")
        if not source_input:
            print("Error: marketplace source required")
            return 1
        result, message = store.add(source_input)
        print(message)
        return 0 if result is not None else 1

    if marketplace_action in ("remove", "rm"):
        name = getattr(args, "marketplace_name", "")
        if not name:
            print("Error: marketplace name required")
            return 1
        if store.remove(name):
            print(f"Removed marketplace: {name}")
        else:
            print(f"Marketplace '{name}' not found")
            return 1
        return 0

    print(f"Unknown marketplace action: {marketplace_action}")
    return 1
