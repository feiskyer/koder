"""Session plugin overlays must reach the real runtime consumers, not only commands."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import koder_agent.mcp as mcp_pkg
import koder_agent.tools.skill as skill_module
from koder_agent.core.scheduler import _GoalTurnLifecycle
from koder_agent.harness.agents.definitions import get_agent_definitions
from koder_agent.harness.hooks.runtime import load_hook_scopes
from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
from koder_agent.harness.session_flow import _SchedulerBuilder
from koder_agent.tools.todo import TodoRuntimeIdentity, TodoStore


def _install(root, source, label):
    source.mkdir(parents=True)
    (source / "plugin.json").write_text('{"name":"scoped","version":"1.0.0"}')
    (source / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"scoped-server": {"command": label}}})
    )
    skill = source / "skills" / label
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {label}\ndescription: synthetic skill\n---\n{label}"
    )
    agents = source / "agents"
    agents.mkdir()
    (agents / "reviewer.md").write_text(
        f"---\nname: {label}\ndescription: synthetic agent\n---\n{label}"
    )
    hooks = source / "hooks"
    hooks.mkdir()
    (hooks / "hooks.json").write_text(
        json.dumps({"hooks": {"Stop": [{"hooks": [{"type": "command", "command": label}]}]}})
    )
    assert PluginLifecycleService(root).install_from_dir(source).success


@pytest.fixture
def roots(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    project = tmp_path / "project"
    project.mkdir()
    (project / ".git").mkdir()
    monkeypatch.chdir(project)
    base = home / ".koder" / "plugins"
    overlay = tmp_path / "overlay"
    _install(base, tmp_path / "base-source", "base-plugin")
    _install(overlay, tmp_path / "overlay-source", "overlay-plugin")
    monkeypatch.setattr(
        skill_module,
        "get_config",
        lambda: SimpleNamespace(
            skills=SimpleNamespace(
                user_skills_dir=str(home / ".koder" / "skills"),
                project_skills_dir=str(project / ".koder" / "skills"),
            )
        ),
    )
    monkeypatch.setattr(skill_module, "_merged_skills", None)
    monkeypatch.setattr(skill_module, "_merged_skills_key", None)
    return project, base, overlay


def _scheduler(root, label):
    return SimpleNamespace(
        session=None,
        plugin_root=root,
        permission_service=None,
        approver=None,
        goal_runtime=SimpleNamespace(on_turn_start=AsyncMock()),
        todo_store=TodoStore(TodoRuntimeIdentity(session_id=label, agent_id="main", run_id=label)),
        _goal_cumulative_tokens=lambda: 0,
        _finish_goal_turn=AsyncMock(),
    )


def _observed_consumers(project):
    with mcp_pkg._load_plugin_mcp_configs() as configs:
        command = configs[0].command
    skills = skill_module._get_merged_skills()
    agents = get_agent_definitions(cwd=project)
    with load_hook_scopes(project) as scopes:
        hook_commands = [
            hook["command"]
            for scope in scopes
            for rule in scope.hooks.get("Stop", [])
            for hook in rule.get("hooks", [])
        ]
    return (
        command,
        set(skills),
        {agent.agent_type for agent in agents.active_agents if agent.source == "plugin"},
        hook_commands,
    )


@pytest.mark.asyncio
async def test_turn_overlay_reaches_consumers_and_restores_previous_root(roots):
    project, _base, overlay = roots
    async with _GoalTurnLifecycle(_scheduler(overlay, "overlay-owner")):
        command, skills, agents, hooks = await asyncio.to_thread(_observed_consumers, project)
        assert command == "overlay-plugin"
        assert "scoped:overlay-plugin" in skills and "scoped:base-plugin" not in skills
        assert any("overlay-plugin" in name for name in agents)
        assert hooks == ["overlay-plugin"]
    command, skills, agents, hooks = _observed_consumers(project)
    assert command == "base-plugin"
    assert "scoped:base-plugin" in skills and "scoped:overlay-plugin" not in skills
    assert any("base-plugin" in name for name in agents)
    assert hooks == ["base-plugin"]


@pytest.mark.asyncio
async def test_concurrent_turns_do_not_share_plugin_root(roots):
    project, base, overlay = roots
    first, second = asyncio.Event(), asyncio.Event()

    async def run(root, label, ready, peer):
        async with _GoalTurnLifecycle(_scheduler(root, label)):
            ready.set()
            await asyncio.wait_for(peer.wait(), timeout=2)
            with mcp_pkg._load_plugin_mcp_configs() as configs:
                return configs[0].command

    assert await asyncio.gather(
        run(base, "base", first, second), run(overlay, "overlay", second, first)
    ) == ["base-plugin", "overlay-plugin"]


def test_scheduler_builder_forwards_effective_plugin_root(roots):
    _project, _base, overlay = roots

    class Scheduler:
        def __init__(self, *, session_id, plugin_root=None):
            self.plugin_root = plugin_root

    builder = _SchedulerBuilder(
        scheduler_type=Scheduler,
        streaming=False,
        agent_definition=None,
        instructions_override=None,
        instructions_append=None,
        permission_service=None,
        approver=None,
        agent_definitions=None,
        plugin_root=overlay,
    )
    assert builder.build("scoped-session").plugin_root == overlay


def test_threaded_skill_cache_does_not_return_another_root(roots, monkeypatch):
    from koder_agent.harness.plugins.context import get_plugin_root, plugin_root_scope

    _project, base, overlay = roots
    comparing, release, discovered_other = threading.Event(), threading.Event(), threading.Event()

    class PausedKey(str):
        def __ne__(self, other):
            if other == "base":
                comparing.set()
                assert release.wait(3)
            return str.__ne__(self, other)

    monkeypatch.setattr(skill_module, "_merged_skills", {"root": "base"})
    monkeypatch.setattr(skill_module, "_merged_skills_key", PausedKey("base"))
    monkeypatch.setattr(
        skill_module,
        "_compute_merged_skills_cache_key",
        lambda **_kwargs: "base" if get_plugin_root() == base else "overlay",
    )

    def discover(**_kwargs):
        discovered_other.set()
        return {"root": "overlay"}

    monkeypatch.setattr(skill_module, "discover_merged_skills", discover)

    def lookup(root):
        with plugin_root_scope(root):
            return skill_module._get_merged_skills()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lookup, base)
        try:
            assert comparing.wait(3)
            second = executor.submit(lookup, overlay)
            discovered_other.wait(0.3)
        finally:
            release.set()
        assert first.result(timeout=3) == {"root": "base"}
        assert second.result(timeout=3) == {"root": "overlay"}


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_agent_initialization_outside_turn_uses_overlay_and_restores_root(
    roots, monkeypatch, fail
):
    from koder_agent.core.scheduler import AgentScheduler
    from koder_agent.harness.plugins.context import get_plugin_root

    _project, base, overlay = roots
    scheduler = AgentScheduler.__new__(AgentScheduler)
    scheduler._migration_done = True
    scheduler._agent_initialized = False
    scheduler.agent_definition = None
    scheduler.instructions_override = None
    scheduler.instructions_append = None
    scheduler.tools = []
    scheduler.plugin_root = overlay

    async def create(*_args, **_kwargs):
        assert get_plugin_root() == overlay
        if fail:
            raise RuntimeError("synthetic initialization failure")
        return SimpleNamespace(
            model=SimpleNamespace(context_window=128000),
            model_settings=SimpleNamespace(max_tokens=1024),
            _koder_mcp_servers=[],
        )

    monkeypatch.setattr("koder_agent.core.scheduler.create_dev_agent", create)
    monkeypatch.setattr("koder_agent.core.scheduler.get_model_name", lambda: "gpt-4.1")
    if fail:
        with pytest.raises(RuntimeError, match="synthetic initialization"):
            await scheduler._ensure_agent_initialized()
    else:
        await scheduler._ensure_agent_initialized()
        assert scheduler._agent_initialized
    assert get_plugin_root() == base


def test_active_root_symlink_is_not_normalized_past_plugin_guards(roots, tmp_path):
    from koder_agent.harness.plugins.context import plugin_root_scope

    project, _base, overlay = roots
    link = tmp_path / "root-link"
    link.symlink_to(overlay, target_is_directory=True)
    with plugin_root_scope(link):
        with mcp_pkg._load_plugin_mcp_configs() as configs:
            assert configs == []
        assert "scoped:overlay-plugin" not in skill_module._get_merged_skills()
        agents = get_agent_definitions(cwd=project)
        assert not any(agent.source == "plugin" for agent in agents.active_agents)
        with load_hook_scopes(project) as scopes:
            assert not any(scope.source == "plugin" for scope in scopes)
