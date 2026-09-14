import asyncio
import json
from pathlib import Path

import pytest

from koder_agent.agentic.skill_guardrail import skill_tool_restriction_guardrail
from koder_agent.config.models import SkillsConfig
from koder_agent.harness.commands.interactive import HarnessInteractiveCommandHandler
from koder_agent.harness.memory.candidates import CandidateRecord, CandidateStore
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.tools.skill import Skill, _apply_skill_restrictions
from koder_agent.tools.skill_context import (
    begin_skill_restriction_scope,
    clear_restrictions,
    get_active_restrictions,
    reset_skill_restriction_scope,
)


class _DummyConfig:
    def __init__(self, user_dir: Path, project_dir: Path):
        self.skills = SkillsConfig(
            enabled=True,
            user_skills_dir=str(user_dir),
            project_skills_dir=str(project_dir),
        )
        self.model = type("_M", (), {"name": "test-model", "provider": "test"})()
        self.cli = type("_C", (), {"stream": False, "session": None})()


async def _run_skills(handler: HarnessInteractiveCommandHandler) -> str:
    return await handler.handle_slash_input("/skills", scheduler=None)


class _ToolGuardrailData:
    def __init__(self, tool_name: str):
        self.context = type(
            "ToolContext",
            (),
            {"tool_name": tool_name, "tool_arguments": None},
        )()


def _guardrail_behavior(tool_name: str) -> str:
    result = skill_tool_restriction_guardrail(_ToolGuardrailData(tool_name))
    behavior = result.behavior
    if isinstance(behavior, dict):
        return str(behavior["type"])
    return str(behavior.type)


def test_skills_command_lists_available_skills(tmp_path, monkeypatch):
    user_dir = tmp_path / "user-skills" / "demo-skill"
    project_dir = tmp_path / "project-skills" / "proj-skill"
    user_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)
    (user_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo user skill\n---\ncontent",
        encoding="utf-8",
    )
    (project_dir / "SKILL.md").write_text(
        "---\nname: proj-skill\ndescription: demo project skill\n---\ncontent",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(user_dir.parent, project_dir.parent),
    )
    handler = HarnessInteractiveCommandHandler()
    import asyncio

    result = asyncio.run(_run_skills(handler))
    assert "batch" in result
    assert "debug" in result
    assert "loop" in result
    assert "simplify" in result
    assert "demo-skill" in result
    assert "proj-skill" in result


def test_skills_command_hides_non_user_invocable_skills(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "hidden-skill"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: hidden-skill\ndescription: hidden\nuser-invocable: false\n---\ncontent",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )
    handler = HarnessInteractiveCommandHandler()
    import asyncio

    result = asyncio.run(_run_skills(handler))
    assert "hidden-skill" not in result


def test_direct_skill_invocation_executes_inline_skill(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "explain-code"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: explain-code\ndescription: explain code\nargument-hint: [path]\n---\nExplain $ARGUMENTS in ${KODER_SESSION_ID}",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )

    class _Session:
        session_id = "skill-session"

    class _Scheduler:
        session = _Session()

        async def handle(
            self, prompt: str, render_output: bool = True, multimodal_input=None
        ) -> str:
            return prompt

    handler = HarnessInteractiveCommandHandler()
    import asyncio

    result = asyncio.run(
        handler.handle_slash_input("/explain-code src/auth.py", scheduler=_Scheduler())
    )
    assert "Explain src/auth.py in skill-session" in result


def test_direct_restricted_skill_enforces_policy_across_scheduler_tasks(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "read-only"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\n"
        "name: read-only\n"
        "description: read only\n"
        "allowed-tools:\n"
        "  - read_file\n"
        "---\n"
        "Inspect $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )

    class _Scheduler:
        session = type("Session", (), {"session_id": "skill-session"})()

        async def handle(self, prompt: str, render_output: bool = True) -> str:
            token = begin_skill_restriction_scope()
            try:

                async def check_tool(tool_name: str) -> str:
                    return _guardrail_behavior(tool_name)

                allowed = await asyncio.create_task(check_tool("read_file"))
                blocked = await asyncio.create_task(check_tool("write_file"))

                async def load_nested_skill() -> None:
                    _apply_skill_restrictions(
                        Skill(
                            name="nested-search",
                            description="nested",
                            content="nested",
                            allowed_tools=["glob_search"],
                        )
                    )

                await asyncio.create_task(load_nested_skill())
                nested_allowed = await asyncio.create_task(check_tool("glob_search"))
                restrictions = get_active_restrictions()
                assert restrictions is not None
                assert restrictions.loaded_skills == ["read-only", "nested-search"]
                return f"{prompt}|{allowed}|{blocked}|{nested_allowed}"
            finally:
                reset_skill_restriction_scope(token)

    clear_restrictions()
    handler = HarnessInteractiveCommandHandler()
    result = asyncio.run(handler.handle_slash_input("/read-only src", scheduler=_Scheduler()))

    assert result == "Inspect src|allow|reject_content|allow"
    assert get_active_restrictions() is None


def test_direct_unrestricted_skill_keeps_scheduler_tools_unrestricted(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "inspect"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: inspect\ndescription: inspect\n---\nInspect $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )

    class _Scheduler:
        session = type("Session", (), {"session_id": "skill-session"})()

        async def handle(self, prompt: str, render_output: bool = True) -> str:
            token = begin_skill_restriction_scope()
            try:
                return _guardrail_behavior("write_file")
            finally:
                reset_skill_restriction_scope(token)

    clear_restrictions()
    handler = HarnessInteractiveCommandHandler()
    result = asyncio.run(handler.handle_slash_input("/inspect src", scheduler=_Scheduler()))

    assert result == "allow"
    assert get_active_restrictions() is None


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_direct_restricted_skill_cleans_up_after_scheduler_exit(tmp_path, monkeypatch, outcome):
    project_dir = tmp_path / "project-skills" / "read-only"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: read-only\ndescription: read only\nallowed-tools: [read_file]\n---\nInspect",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )

    class _Scheduler:
        session = type("Session", (), {"session_id": "skill-session"})()

        async def handle(self, prompt: str, render_output: bool = True) -> str:
            token = begin_skill_restriction_scope()
            try:
                assert _guardrail_behavior("write_file") == "reject_content"
                if outcome == "error":
                    raise RuntimeError("boom")
                if outcome == "cancel":
                    raise asyncio.CancelledError
                return prompt
            finally:
                reset_skill_restriction_scope(token)

    clear_restrictions()
    handler = HarnessInteractiveCommandHandler()
    invocation = handler.handle_slash_input("/read-only", scheduler=_Scheduler())
    if outcome == "error":
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(invocation)
    elif outcome == "cancel":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(invocation)
    else:
        assert asyncio.run(invocation) == "Inspect"

    assert get_active_restrictions() is None


def test_direct_manual_skill_invocation_renders_without_model(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "manual-check"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: manual-check\ndescription: manual check\ndisable-model-invocation: true\n---\nManual $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )

    class _Scheduler:
        session = type("Session", (), {"session_id": "skill-session"})()

        async def handle(
            self, prompt: str, render_output: bool = True, multimodal_input=None
        ) -> str:
            raise AssertionError("manual skill should not call the model scheduler")

    handler = HarnessInteractiveCommandHandler()
    result = asyncio.run(
        handler.handle_slash_input("/manual-check fixture", scheduler=_Scheduler())
    )

    assert result == "Manual fixture"


def test_direct_remember_skill_persists_project_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handler = HarnessInteractiveCommandHandler()

    result = asyncio.run(
        handler.handle_slash_input("/remember durable fixture memory", scheduler=None)
    )

    assert "remember: saved" in result
    assert "type: project" in result
    memory_files = list((tmp_path / ".koder" / "memory").glob("*.md"))
    assert any(path.name != "MEMORY.md" for path in memory_files)
    assert "durable fixture memory" in (tmp_path / ".koder" / "memory" / "MEMORY.md").read_text(
        encoding="utf-8"
    )


def test_memory_command_lists_shows_approves_and_rejects_candidates(tmp_path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "origin-project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    memory_store = CandidateStore(home / ".koder" / "memory-candidates", kind="memory")
    approved = memory_store.stage(
        {"type": "project", "content": "Approve me", "description": "Approval fixture"},
        storage_scope="project",
        origin_project_root=project,
        origin_session_id="command-test-session",
    )
    rejected = memory_store.stage(
        {"type": "reference", "content": "Reject me", "description": "Reject fixture"},
        storage_scope="project",
        origin_project_root=project,
        origin_session_id="command-test-session",
    )
    handler = HarnessInteractiveCommandHandler()

    listed = asyncio.run(handler.handle_slash_input("/memory candidates", scheduler=None))
    shown = asyncio.run(handler.handle_slash_input(f"/memory show {approved.id}", scheduler=None))
    approved_output = asyncio.run(
        handler.handle_slash_input(f"/memory approve {approved.id}", scheduler=None)
    )
    rejected_output = asyncio.run(
        handler.handle_slash_input(f"/memory reject {rejected.id}", scheduler=None)
    )

    assert approved.id in listed and rejected.id in listed
    assert "scope=project" in listed
    assert f"origin_project={project}" in listed
    assert "origin_session=command-test-session" in listed
    assert "Approve me" in shown
    assert "storage_scope: project" in shown
    assert f"origin_project_root: {project}" in shown
    assert "memory candidate approved" in approved_output
    assert "candidate rejected" in rejected_output
    assert len(list((project / ".koder" / "memory").glob("*.md"))) == 1
    assert not (home / ".koder" / "memory").exists()


def test_candidate_listing_sanitizes_control_characters(monkeypatch):
    candidate = CandidateRecord(
        id="a" * 64,
        kind="memory",
        created_at="2026-01-01T00:00:00+00:00",
        storage_scope="project",
        origin_project_root="/tmp/candidate-origin",
        origin_session_id="candidate-session",
        payload={
            "type": "project",
            "content": "safe",
            "description": "line one\n\x1b[31mline two",
        },
    )

    class _Store:
        def __init__(self, records):
            self.records = records

        def list(self):
            return self.records

    monkeypatch.setattr(
        "koder_agent.harness.memory.candidates.default_memory_candidate_store",
        lambda: _Store([candidate]),
    )
    monkeypatch.setattr(
        "koder_agent.harness.memory.candidates.default_skill_candidate_store",
        lambda: _Store([]),
    )

    output = asyncio.run(
        HarnessInteractiveCommandHandler().handle_slash_input("/memory candidates", scheduler=None)
    )

    assert "\x1b" not in output
    assert "line one line two" in output


def test_direct_skill_invocation_runs_forked_skill_via_agent_service(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "deploy"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: deploy app\ncontext: fork\nagent: reviewer\n---\nDeploy $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_agent_definitions",
        lambda **_kwargs: type(
            "Defs", (), {"active_agents": [type("A", (), {"agent_type": "reviewer"})()]}
        )(),
    )

    from koder_agent.harness.agents.service import AgentService

    class _AgentService(AgentService):
        async def run_sync(self, *, agent_definition, prompt, seed_items=None, cwd=None):
            return f"{agent_definition.agent_type}: {prompt}"

    service = _AgentService(output_root=tmp_path / "agents")
    handler = HarnessInteractiveCommandHandler(agent_service=service)
    import asyncio

    try:
        result = asyncio.run(handler.handle_slash_input("/deploy production", scheduler=None))
        assert result == "reviewer: Deploy production"
    finally:
        service.close()


def test_direct_skill_invocation_activates_skill_scoped_hooks(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "explain-code"
    marker = tmp_path / "skill-hook.json"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: explain-code\ndescription: explain code\nhooks:\n  PostToolUse:\n    - hooks:\n        - type: command\n          command: >-\n            python -c \"import sys, pathlib; pathlib.Path(r'"
        + str(marker)
        + "').write_text(sys.stdin.read())\"\n---\nExplain $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )

    class _Scheduler:
        async def handle(
            self, prompt: str, render_output: bool = True, multimodal_input=None
        ) -> str:
            from koder_agent.harness.hooks.runtime import dispatch_command_hooks

            dispatch_command_hooks(
                cwd=tmp_path,
                event_name="PostToolUse",
                match_value="read_file",
                payload={
                    "event": "PostToolUse",
                    "tool_name": "read_file",
                    "tool_input": {"file_path": "demo.txt"},
                    "result": "ok",
                },
            )
            return prompt

        session = type("Session", (), {"session_id": "skill-session"})()

    handler = HarnessInteractiveCommandHandler()
    result = asyncio.run(handler.handle_slash_input("/explain-code demo", scheduler=_Scheduler()))

    assert "Explain demo" in result
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["event"] == "PostToolUse"


def test_direct_skill_invocation_respects_skill_permission_rules(tmp_path, monkeypatch):
    project_dir = tmp_path / "project-skills" / "deploy"
    project_dir.mkdir(parents=True)
    (project_dir / "SKILL.md").write_text(
        "---\nname: deploy\ndescription: deploy app\n---\nDeploy $ARGUMENTS",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.get_config",
        lambda: _DummyConfig(tmp_path / "user-skills", project_dir.parent),
    )
    permissions = PermissionService.default()
    permissions.add_rule("Skill", "deny", "Skill(deploy *)")

    handler = HarnessInteractiveCommandHandler(permission_service=permissions)
    result = asyncio.run(handler.handle_slash_input("/deploy production", scheduler=None))

    assert "skills: blocked" in result
    assert "Denied by rule" in result


def test_direct_bundled_skill_invocation_works():
    handler = HarnessInteractiveCommandHandler()
    result = asyncio.run(handler.handle_slash_input("/simplify", scheduler=None))
    assert "improving the **quality** of the changed code" in result
