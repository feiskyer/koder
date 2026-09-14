"""Fix 3: subagents must enforce PermissionService argument-level checks.

The main agent enforces deny/approval rules because the scheduler publishes the
PermissionService into the tool layer via set_tool_permission_context before
Runner.run. Subagents (Agent tool, background/team) previously ran their own
Runner.run OUTSIDE any such scope and built SubagentLifecycleHooks without a
permission_service, so project/user deny + approval rules did not apply to them.

These tests pin the fix: _execute_agent_run publishes the permission context
around the subagent Runner.run (with a non-interactive approver so approval-gated
calls fail closed, since a subagent cannot prompt) and threads the service into
SubagentLifecycleHooks.
"""

from __future__ import annotations

import sys
import types

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

import asyncio
import json

from koder_agent.harness.agents import service as service_mod
from koder_agent.harness.agents.definitions import AgentDefinition
from koder_agent.tools.permission_context import get_tool_permission_context
from koder_agent.tools.todo import (
    TodoRuntimeIdentity,
    TodoStore,
    get_todo_store,
    reset_todo_state_for_tests,
    todo_write,
)


def _agent_def() -> AgentDefinition:
    return AgentDefinition(
        agent_type="worker",
        when_to_use="does work",
        system_prompt="You are a worker.",
        source="built-in",
    )


def _todo_store(session_id: str) -> TodoStore:
    return TodoStore(TodoRuntimeIdentity(session_id, "worker", f"test:{session_id}"))


def test_execute_agent_run_publishes_permission_context(monkeypatch):
    """Inside the subagent Runner.run, the permission context must be active and
    carry the permission_service we passed in."""
    sentinel_service = object()
    observed = {}

    async def fake_create_dev_agent(*args, **kwargs):
        return object()

    class _Result:
        final_output = "done"

    async def fake_run(*args, **kwargs):
        # This runs in the same task/context the subagent tools will copy from.
        ctx = get_tool_permission_context()
        observed["ctx_present"] = ctx is not None
        observed["service"] = getattr(ctx, "permission_service", None)
        observed["hooks"] = kwargs.get("hooks")
        return _Result()

    monkeypatch.setattr(service_mod, "create_dev_agent", fake_create_dev_agent)
    monkeypatch.setattr(service_mod, "get_all_tools", lambda: [])
    monkeypatch.setattr(service_mod.Runner, "run", staticmethod(fake_run))

    class _FakeSession:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, items):
            return None

    monkeypatch.setattr(service_mod, "EnhancedSQLiteSession", _FakeSession)

    asyncio.run(
        service_mod._execute_agent_run(
            agent_definition=_agent_def(),
            prompt="do it",
            session_id="s1",
            seed_items=None,
            cwd=None,
            permission_service=sentinel_service,
            todo_store=_todo_store("s1"),
        )
    )

    assert observed["ctx_present"] is True
    assert observed["service"] is sentinel_service
    # Hooks must also carry the service for the name-level check on non-guarded tools.
    assert getattr(observed["hooks"], "_permission_service", None) is sentinel_service


def test_execute_agent_run_context_resets_after_run(monkeypatch):
    """The published context must not leak into the caller's scope after the run."""
    sentinel_service = object()

    async def fake_create_dev_agent(*args, **kwargs):
        return object()

    class _Result:
        final_output = "done"

    async def fake_run(*args, **kwargs):
        return _Result()

    monkeypatch.setattr(service_mod, "create_dev_agent", fake_create_dev_agent)
    monkeypatch.setattr(service_mod, "get_all_tools", lambda: [])
    monkeypatch.setattr(service_mod.Runner, "run", staticmethod(fake_run))

    class _FakeSession:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, items):
            return None

    monkeypatch.setattr(service_mod, "EnhancedSQLiteSession", _FakeSession)

    async def scenario():
        await service_mod._execute_agent_run(
            agent_definition=_agent_def(),
            prompt="do it",
            session_id="s1",
            seed_items=None,
            cwd=None,
            permission_service=sentinel_service,
            todo_store=_todo_store("s1"),
        )
        # After the subagent run, the caller's context must be unchanged (None here).
        return get_tool_permission_context()

    assert asyncio.run(scenario()) is None


def test_parallel_subagents_receive_isolated_todo_runtime_scopes(monkeypatch):
    observed = {}

    async def fake_create_dev_agent(*args, **kwargs):
        return object()

    class _Result:
        final_output = "done"

    async def fake_run(*args, **kwargs):
        session_id = kwargs["session"].session_id
        store = get_todo_store()
        await todo_write.on_invoke_tool(
            None,
            json.dumps(
                {
                    "todos": [
                        {
                            "content": f"todo-{session_id}",
                            "status": "in_progress",
                            "priority": "high",
                            "id": session_id,
                        }
                    ]
                }
            ),
        )
        await asyncio.sleep(0)
        observed[session_id] = (store.identity, store.todos)
        return _Result()

    monkeypatch.setattr(service_mod, "create_dev_agent", fake_create_dev_agent)
    monkeypatch.setattr(service_mod, "get_all_tools", lambda: [])
    monkeypatch.setattr(service_mod.Runner, "run", staticmethod(fake_run))

    class _FakeSession:
        def __init__(self, session_id, *args, **kwargs):
            self.session_id = session_id

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, items):
            return None

    monkeypatch.setattr(service_mod, "EnhancedSQLiteSession", _FakeSession)
    reset_todo_state_for_tests()

    async def scenario():
        await asyncio.gather(
            service_mod._execute_agent_run(
                agent_definition=_agent_def(),
                prompt="first",
                session_id="subagent-a",
                seed_items=None,
                cwd=None,
                todo_store=_todo_store("subagent-a"),
            ),
            service_mod._execute_agent_run(
                agent_definition=_agent_def(),
                prompt="second",
                session_id="subagent-b",
                seed_items=None,
                cwd=None,
                todo_store=_todo_store("subagent-b"),
            ),
        )

    asyncio.run(scenario())

    identity_a, todos_a = observed["subagent-a"]
    identity_b, todos_b = observed["subagent-b"]
    assert identity_a.session_id == "subagent-a"
    assert identity_b.session_id == "subagent-b"
    assert todos_a[0]["content"] == "todo-subagent-a"
    assert todos_b[0]["content"] == "todo-subagent-b"


def test_interactive_handler_wires_permission_service_into_agent_service():
    """Fix 3 production wiring: the interactive command handler must construct its
    AgentService WITH the permission service, or subagent enforcement is a no-op."""
    from koder_agent.harness.commands.interactive import HarnessInteractiveCommandHandler
    from koder_agent.harness.permissions.service import PermissionService

    svc = PermissionService.default()
    handler = HarnessInteractiveCommandHandler(permission_service=svc)
    assert handler.agent_service._permission_service is svc


def test_agent_tool_threads_active_permission_service(monkeypatch):
    """Fix 3 production wiring: agent_tool reads the active tool-permission context
    and hands the service to the AgentService it spawns."""
    import koder_agent.tools.agent as agent_tool_mod
    from koder_agent.tools.permission_context import (
        reset_tool_permission_context,
        set_tool_permission_context,
    )

    captured = {}

    class _FakeService:
        def __init__(self, *, permission_service=None, **kw):
            captured["permission_service"] = permission_service

        def resolve_agent_id(self, *a, **k):
            return None

        async def run_sync(self, **kwargs):
            return "done"

    # agent.py imports AgentService lazily from the service module, so patch it there.
    monkeypatch.setattr(service_mod, "AgentService", _FakeService)

    sentinel = object()
    tok = set_tool_permission_context(sentinel, approver=None)
    try:
        # Drive just enough of _agent_tool_impl to reach AgentService construction.
        import asyncio

        try:
            asyncio.run(
                agent_tool_mod._agent_tool_impl(
                    description="d", prompt="p", subagent_type="Explore"
                )
            )
        except Exception:
            pass  # We only care that AgentService got the service before any later error.
    finally:
        reset_tool_permission_context(tok)

    assert captured.get("permission_service") is sentinel


def test_subagent_uses_deny_approver_not_fail_open_none(monkeypatch):
    """Fix 3 (review finding 3): a subagent must be given an explicit always-DENY
    approver, not approver=None. With None, enforce_tool_permission's TTY-aware
    fallback fails OPEN in an interactive session; an approval-gated call in a
    subagent must instead be denied (a subagent cannot prompt)."""
    captured = {}

    async def fake_create_dev_agent(*a, **k):
        return object()

    class _Result:
        final_output = "done"

    async def fake_run(*a, **k):
        from koder_agent.tools.permission_context import get_tool_permission_context

        ctx = get_tool_permission_context()
        captured["approver"] = getattr(ctx, "approver", None)
        return _Result()

    monkeypatch.setattr(service_mod, "create_dev_agent", fake_create_dev_agent)
    monkeypatch.setattr(service_mod, "get_all_tools", lambda: [])
    monkeypatch.setattr(service_mod.Runner, "run", staticmethod(fake_run))

    class _FakeSession:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, items):
            return None

    monkeypatch.setattr(service_mod, "EnhancedSQLiteSession", _FakeSession)

    asyncio.run(
        service_mod._execute_agent_run(
            agent_definition=_agent_def(),
            prompt="p",
            session_id="s",
            seed_items=None,
            cwd=None,
            permission_service=object(),
            todo_store=_todo_store("s"),
        )
    )
    approver = captured["approver"]
    assert approver is not None, "subagent approver must not be None (fails open on TTY)"
    # The approver must resolve to a denial for any approval-gated call.
    verdict = asyncio.run(approver("run_shell", {"command": "rm -rf /"}, object()))
    assert verdict in ("deny", False), f"subagent approver must deny, got {verdict!r}"


def test_subagent_falls_back_to_inherited_service_when_none_passed(monkeypatch):
    """Fix 3 (review finding 2): when no explicit permission_service is passed,
    _execute_agent_run must NOT clear the context inherited from the parent run
    (which would downgrade enforcement); it should reuse the inherited service."""
    inherited = object()
    observed = {}

    async def fake_create_dev_agent(*a, **k):
        return object()

    class _Result:
        final_output = "done"

    async def fake_run(*a, **k):
        from koder_agent.tools.permission_context import get_tool_permission_context

        ctx = get_tool_permission_context()
        observed["service"] = getattr(ctx, "permission_service", None)
        return _Result()

    monkeypatch.setattr(service_mod, "create_dev_agent", fake_create_dev_agent)
    monkeypatch.setattr(service_mod, "get_all_tools", lambda: [])
    monkeypatch.setattr(service_mod.Runner, "run", staticmethod(fake_run))

    class _FakeSession:
        def __init__(self, *a, **k):
            pass

        def close(self):
            pass

        async def get_items(self):
            return []

        async def add_items(self, items):
            return None

    monkeypatch.setattr(service_mod, "EnhancedSQLiteSession", _FakeSession)

    from koder_agent.tools.permission_context import (
        reset_tool_permission_context,
        set_tool_permission_context,
    )

    async def scenario():
        # Simulate running inside a parent scope that already published a service.
        tok = set_tool_permission_context(inherited, approver=None)
        try:
            await service_mod._execute_agent_run(
                agent_definition=_agent_def(),
                prompt="p",
                session_id="s",
                seed_items=None,
                cwd=None,
                permission_service=None,  # none passed explicitly
                todo_store=_todo_store("s"),
            )
        finally:
            reset_tool_permission_context(tok)

    asyncio.run(scenario())
    assert observed["service"] is inherited, "must reuse inherited service, not clear it"
