import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from koder_agent.harness import session_flow
from koder_agent.harness.onboarding import OnboardingState
from koder_agent.tools.todo import TodoRuntimeIdentity, TodoStore


@pytest.fixture(autouse=True)
def _isolate_default_runtime_home(tmp_path, monkeypatch):
    original_home = Path.home
    user_home = original_home()

    def test_home(_cls):
        current = original_home()
        # Keep explicit per-test home overrides, but never use the real profile.
        return tmp_path if current == user_home else current

    monkeypatch.setattr(Path, "home", classmethod(test_home))


class _FakeStdin:
    def __init__(self, text: str, *, is_tty: bool):
        self._text = text
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty

    def read(self) -> str:
        return self._text


class _FakeSchedulerSession:
    def __init__(self, session_id: str):
        self.session_id = session_id

    async def set_title(self, _name: str) -> None:
        return None

    async def get_display_name(self) -> str:
        return self.session_id

    async def get_items(self):
        return []

    async def get_most_recent_session_for_cwd(self, _cwd: str):
        return None


class _FakeScheduler:
    prompts: list[tuple[str, bool]] = []
    instances = []

    def __init__(
        self,
        session_id: str,
        streaming: bool,
        agent_definition=None,
        instructions_override=None,
        instructions_append=None,
        permission_service=None,
        approver=None,
        todo_store=None,
        project_root=None,
    ):
        self.session = _FakeSchedulerSession(session_id)
        agent_id = agent_definition.agent_type if agent_definition is not None else "main"
        self.todo_store = todo_store or TodoStore(
            TodoRuntimeIdentity(
                session_id=session_id,
                agent_id=agent_id,
                run_id=f"fake-{len(self.instances)}",
            )
        )
        self.streaming = streaming
        self.agent_definition = agent_definition
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.instructions_override = instructions_override
        self.instructions_append = instructions_append
        self.permission_service = permission_service
        self.approver = approver
        self.usage_tracker = object()
        self._title_generation_task = None
        self._mcp_servers = []
        self.agent_definitions = None
        self.cleanup_calls = 0
        self.initialized = False
        self.instances.append(self)

    async def handle(self, prompt: str, render_output: bool = True, multimodal_input=None) -> str:
        self.prompts.append((prompt, render_output))
        return prompt

    async def cleanup(self) -> None:
        self.cleanup_calls += 1

    async def _ensure_agent_initialized(self) -> None:
        self.initialized = True


class _FakeInteractivePrompt:
    status_line = None

    def __init__(self, *_args, **_kwargs):
        pass

    async def get_input(self) -> str:
        raise EOFError

    def update_session(self, _session_id: str) -> None:
        return None


class _FakeCommandHandler:
    config_service = None

    def __init__(self, **_kwargs):
        pass

    def get_command_list(self):
        return []

    def is_slash_command(self, _prompt: str) -> bool:
        return False


class _FakeEnhancedSQLiteSession:
    def __init__(self, session_id: str, *_args, **_kwargs):
        self.session_id = session_id

    async def get_cwd(self):
        return None

    async def get_agent(self):
        return None

    def close(self) -> None:
        return None

    @staticmethod
    async def record_session_cwd(_session_id: str, _cwd: str) -> None:
        return None

    @staticmethod
    async def get_session_agent(_session_id: str):
        return None

    @staticmethod
    async def record_session_agent(_session_id: str, _agent_name: str) -> None:
        return None

    @staticmethod
    async def get_most_recent_session_for_cwd(_cwd: str):
        return None


def _patch_session_flow(monkeypatch, stdin_text: str, *, stdin_is_tty: bool = False) -> None:
    fake_config = SimpleNamespace(cli=SimpleNamespace(stream=False, session=None))
    fake_manager = SimpleNamespace(get_effective_value=lambda _value, _default: None)

    monkeypatch.setattr(session_flow, "get_config", lambda: fake_config)
    monkeypatch.setattr(session_flow, "get_config_manager", lambda: fake_manager)
    monkeypatch.setattr(session_flow, "load_context", _async_value(""))
    monkeypatch.setattr(session_flow, "HarnessInteractiveCommandHandler", _FakeCommandHandler)
    monkeypatch.setattr(session_flow.sys, "stdin", _FakeStdin(stdin_text, is_tty=stdin_is_tty))

    monkeypatch.setattr("koder_agent.utils.setup_openai_client", lambda: None)
    monkeypatch.setattr("koder_agent.utils.default_session_local_ms", lambda: "test-session")
    monkeypatch.setattr(
        "koder_agent.core.session.EnhancedSQLiteSession", _FakeEnhancedSQLiteSession
    )
    monkeypatch.setattr("koder_agent.core.scheduler.AgentScheduler", _FakeScheduler)
    monkeypatch.setattr("koder_agent.core.interactive.InteractivePrompt", _FakeInteractivePrompt)

    _FakeScheduler.prompts = []
    _FakeScheduler.instances = []


def _async_value(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


def test_run_harness_session_flow_uses_piped_stdin_without_args(monkeypatch):
    _patch_session_flow(monkeypatch, "hello from stdin")

    exit_code = asyncio.run(session_flow.run_harness_session_flow(first_arg=None, argv=[]))

    assert exit_code == 0
    assert _FakeScheduler.prompts == [("hello from stdin", True)]


def test_run_harness_session_flow_combines_prompt_with_piped_stdin(monkeypatch):
    _patch_session_flow(monkeypatch, "hello from stdin")

    exit_code = asyncio.run(
        session_flow.run_harness_session_flow(first_arg=None, argv=["--print", "echo stdin only"])
    )

    assert exit_code == 0
    assert _FakeScheduler.prompts == [
        (
            "echo stdin only\n\nStdin content:\nhello from stdin",
            True,
        )
    ]


def test_startup_onboarding_uses_persisted_session_env_without_mutating_process(monkeypatch):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unrelated-openai-key")
    monkeypatch.delenv("KODER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    captured_envs = []
    events = []

    monkeypatch.setattr(
        session_flow,
        "load_session_env",
        lambda session_id: (
            {
                "KODER_MODEL": "openrouter/anthropic/claude-3-opus",
                "OPENROUTER_API_KEY": "synthetic-session-openrouter-key",
            }
            if session_id == "test-session"
            else {}
        ),
        raising=False,
    )

    def fake_check_onboarding_state(_project_dir, env=None):
        captured_envs.append(None if env is None else dict(env))
        return OnboardingState(
            completed=True,
            api_key_configured=True,
            model_selected=True,
            workspace_trusted=True,
        )

    def fake_dispatch_command_hooks(*, event_name, **_kwargs):
        events.append(event_name)
        return SimpleNamespace(blocked=False, block_reason=None, watch_paths=[])

    monkeypatch.setattr(
        "koder_agent.harness.onboarding.check_onboarding_state",
        fake_check_onboarding_state,
    )
    monkeypatch.setattr(session_flow, "dispatch_command_hooks", fake_dispatch_command_hooks)
    process_env_before = dict(session_flow.os.environ)

    exit_code = asyncio.run(session_flow.run_harness_session_flow(first_arg=None, argv=[]))

    assert exit_code == 0
    assert len(captured_envs) == 1
    effective_env = captured_envs[0]
    assert effective_env["KODER_MODEL"] == "openrouter/anthropic/claude-3-opus"
    assert effective_env["OPENROUTER_API_KEY"] == "synthetic-session-openrouter-key"
    assert effective_env["OPENAI_API_KEY"] == "synthetic-unrelated-openai-key"
    assert dict(session_flow.os.environ) == process_env_before
    assert "Setup" not in events


def test_startup_missing_credentials_renders_panel_and_dispatches_setup_hook(monkeypatch):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    events = []
    printed = []

    monkeypatch.setattr(
        session_flow,
        "load_session_env",
        lambda _session_id: {"KODER_MODEL": "openrouter/anthropic/claude-3-opus"},
        raising=False,
    )
    monkeypatch.setattr(
        "koder_agent.harness.onboarding.check_onboarding_state",
        lambda _project_dir, env=None: OnboardingState(
            completed=False,
            api_key_configured=False,
            model_selected=True,
            workspace_trusted=True,
        ),
    )

    def fake_dispatch_command_hooks(*, event_name, payload=None, **_kwargs):
        events.append((event_name, payload))
        return SimpleNamespace(blocked=False, block_reason=None, watch_paths=[])

    monkeypatch.setattr(session_flow, "dispatch_command_hooks", fake_dispatch_command_hooks)
    monkeypatch.setattr(session_flow.console, "print", printed.append)

    exit_code = asyncio.run(session_flow.run_harness_session_flow(first_arg=None, argv=[]))

    assert exit_code == 0
    setup_events = [payload for event, payload in events if event == "Setup"]
    assert setup_events == [
        {
            "event": "Setup",
            "missing_steps": [
                "Configure API key: Set KODER_API_KEY, OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, or another provider's API key"
            ],
        }
    ]
    setup_panels = [item for item in printed if "Setup Recommended" in str(item.title)]
    assert len(setup_panels) == 1
    assert "synthetic" not in str(setup_panels[0].renderable)


def test_bare_mode_skips_startup_onboarding_session_env(monkeypatch):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("bare mode must skip startup onboarding")

    monkeypatch.setattr(session_flow, "load_session_env", fail_if_called, raising=False)
    monkeypatch.setattr(
        "koder_agent.harness.onboarding.check_onboarding_state",
        fail_if_called,
    )

    exit_code = asyncio.run(session_flow.run_harness_session_flow(first_arg=None, argv=["--bare"]))

    assert exit_code == 0


def test_run_harness_session_flow_dispatches_session_end_when_cron_stop_fails(monkeypatch):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    events = []

    class _FailingCronPromptRunner:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self) -> None:
            return None

        async def stop(self) -> None:
            raise RuntimeError("cron stop failed")

    def _fake_dispatch_command_hooks(*, event_name, **_kwargs):
        events.append(event_name)
        return SimpleNamespace(blocked=False, block_reason=None, watch_paths=[])

    from koder_agent.harness.cron import runtime as cron_runtime

    monkeypatch.setattr(cron_runtime, "CronPromptRunner", _FailingCronPromptRunner)
    monkeypatch.setattr(session_flow, "dispatch_command_hooks", _fake_dispatch_command_hooks)

    exit_code = asyncio.run(session_flow.run_harness_session_flow(first_arg=None, argv=[]))

    assert exit_code == 0
    assert "SessionEnd" in events


def test_keyboard_interrupt_exit_skips_auto_dream_consolidation(monkeypatch):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    calls = []

    class _InterruptPrompt(_FakeInteractivePrompt):
        async def get_input(self) -> str:
            raise KeyboardInterrupt

    class _DreamManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def record_session(self) -> None:
            calls.append("record_session")

        def should_dream(self) -> bool:
            return True

        def save(self) -> None:
            calls.append("save")

    async def _run_auto_dream_from_messages(*_args, **_kwargs):
        calls.append("run_auto_dream")
        return SimpleNamespace(memories_written=0, saved_path=None, errors=[])

    from koder_agent.harness.memory import auto_dream

    monkeypatch.setattr("koder_agent.core.interactive.InteractivePrompt", _InterruptPrompt)
    monkeypatch.setattr(auto_dream, "AutoDreamManager", _DreamManager)
    monkeypatch.setattr(auto_dream, "run_auto_dream_from_messages", _run_auto_dream_from_messages)
    monkeypatch.setattr(auto_dream, "default_auto_dream_task_storage", lambda: object())

    exit_code = asyncio.run(session_flow.run_harness_session_flow(first_arg=None, argv=[]))

    assert exit_code == 0
    assert calls == ["record_session", "save"]


def test_eof_exit_allows_auto_dream_consolidation(monkeypatch):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    calls = []
    provenance = {}

    class _EofPrompt(_FakeInteractivePrompt):
        async def get_input(self) -> str:
            raise EOFError

    class _DreamManager:
        def __init__(self, *_args, **_kwargs):
            pass

        def record_session(self) -> None:
            calls.append("record_session")

        def should_dream(self) -> bool:
            return True

        def save(self) -> None:
            calls.append("save")

    async def _run_auto_dream_from_messages(*_args, **_kwargs):
        calls.append("run_auto_dream")
        provenance.update(_kwargs)
        return SimpleNamespace(memories_written=0, saved_path=None, errors=[])

    from koder_agent.harness.memory import auto_dream

    monkeypatch.setattr("koder_agent.core.interactive.InteractivePrompt", _EofPrompt)
    monkeypatch.setattr(auto_dream, "AutoDreamManager", _DreamManager)
    monkeypatch.setattr(auto_dream, "run_auto_dream_from_messages", _run_auto_dream_from_messages)
    monkeypatch.setattr(auto_dream, "default_auto_dream_task_storage", lambda: object())

    exit_code = asyncio.run(session_flow.run_harness_session_flow(first_arg=None, argv=[]))

    assert exit_code == 0
    assert calls == ["record_session", "run_auto_dream"]
    assert provenance["origin_project_root"] == Path.cwd()
    assert provenance["origin_session_id"] == "test-session"


def test_blocked_channel_session_start_uses_unified_resource_cleanup(monkeypatch, tmp_path):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    events = []
    from koder_agent.harness.channels.notification import ChannelNotificationRouter
    from koder_agent.mcp.notifications import get_notification_handler

    router = ChannelNotificationRouter()
    get_notification_handler().set_channel_router(router)

    def fake_dispatch_command_hooks(*, event_name, payload=None, **_kwargs):
        events.append((event_name, payload))
        return SimpleNamespace(
            blocked=event_name == "SessionStart",
            block_reason="blocked for test" if event_name == "SessionStart" else None,
            watch_paths=[],
        )

    monkeypatch.setattr(session_flow, "dispatch_command_hooks", fake_dispatch_command_hooks)

    for _ in range(2):
        exit_code = asyncio.run(
            session_flow.run_harness_session_flow(
                first_arg=None,
                argv=["--channels", "server:test-channel"],
            )
        )
        assert exit_code == 1

    assert len(_FakeScheduler.instances) == 2
    for scheduler in _FakeScheduler.instances:
        assert scheduler.initialized is True
        assert scheduler.cleanup_calls == 1
    assert [name for name, _payload in events].count("SessionEnd") == 2
    assert router._message_callbacks == {}


def test_multiple_session_switches_preserve_scheduler_options_and_final_session_id(
    monkeypatch, tmp_path
):
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    session_end_payloads = []
    observed_mcp_owners = []

    class _SwitchingPrompt(_FakeInteractivePrompt):
        def __init__(self, *_args, **_kwargs):
            self.inputs = iter(["/switch-one", "/switch-two", "exit"])
            self.updated_sessions = []

        async def get_input(self) -> str:
            return next(self.inputs)

        def update_session(self, session_id: str) -> None:
            self.updated_sessions.append(session_id)

        def reset_history(self) -> None:
            return None

    class _SwitchingCommandHandler(_FakeCommandHandler):
        def __init__(self, **kwargs):
            self.mcp_owner_provider = kwargs["mcp_owner_provider"]

        def is_slash_command(self, prompt: str) -> bool:
            return prompt.startswith("/")

        async def handle_slash_input(self, prompt: str, _scheduler):
            observed_mcp_owners.append(self.mcp_owner_provider())
            if prompt == "/switch-one":
                return "session_switch:session-one"
            if prompt == "/switch-two":
                return "session_switch:session-two"
            return None

    class _PrStatusPoller:
        def start(self) -> None:
            return None

        def touch(self) -> None:
            return None

        def stop(self) -> None:
            return None

    class _CronPromptRunner:
        def __init__(self, scheduler_getter):
            self.scheduler_getter = scheduler_getter

        def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    def fake_dispatch_command_hooks(*, event_name, payload=None, **_kwargs):
        if event_name == "SessionEnd":
            session_end_payloads.append(payload)
        return SimpleNamespace(blocked=False, block_reason=None, watch_paths=[])

    monkeypatch.setattr(
        "koder_agent.core.interactive.InteractivePrompt",
        _SwitchingPrompt,
    )
    monkeypatch.setattr(
        session_flow,
        "HarnessInteractiveCommandHandler",
        _SwitchingCommandHandler,
    )
    monkeypatch.setattr("koder_agent.harness.pr_status.PrStatusPoller", _PrStatusPoller)
    monkeypatch.setattr("koder_agent.harness.cron.runtime.CronPromptRunner", _CronPromptRunner)
    monkeypatch.setattr(session_flow, "dispatch_command_hooks", fake_dispatch_command_hooks)

    exit_code = asyncio.run(
        session_flow.run_harness_session_flow(
            first_arg=None,
            argv=[
                "--session",
                "original-session",
                "--agents",
                '{"reviewer":{"description":"Reviews code","prompt":"Review carefully."}}',
                "--agent",
                "reviewer",
                "--system-prompt",
                "override prompt",
                "--append-system-prompt",
                "append prompt",
            ],
        )
    )

    assert exit_code == 0
    assert [item.session.session_id for item in _FakeScheduler.instances] == [
        "original-session",
        "session-one",
        "session-two",
    ]
    first = _FakeScheduler.instances[0]
    assert first.agent_definition.agent_type == "reviewer"
    assert first.todo_store.identity.agent_id == "reviewer"
    for scheduler in _FakeScheduler.instances:
        assert scheduler.streaming is first.streaming
        assert scheduler.instructions_override == "override prompt"
        assert scheduler.instructions_append == "append prompt"
        assert scheduler.permission_service is first.permission_service
        assert scheduler.approver is not None
        assert scheduler.cleanup_calls == 1
    for scheduler in _FakeScheduler.instances[1:]:
        assert scheduler.agent_definition is None
        assert scheduler.todo_store.identity.agent_id == "main"
        assert scheduler.approver is not first.approver
    assert observed_mcp_owners[0] is _FakeScheduler.instances[0]._mcp_servers
    assert observed_mcp_owners[1] is _FakeScheduler.instances[1]._mcp_servers
    assert [scheduler.todo_store.identity.session_id for scheduler in _FakeScheduler.instances] == [
        "original-session",
        "session-one",
        "session-two",
    ]
    assert session_end_payloads == [
        {
            "event": "SessionEnd",
            "reason": "other",
            "session_id": "session-two",
        }
    ]


class _LifecycleScheduler(_FakeScheduler):
    def __init__(self, session_id: str):
        super().__init__(session_id=session_id, streaming=True)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.handled: list[str] = []
        self.cleaned = False
        self.cleanup_error: Exception | None = None
        self.cleanup_finished = asyncio.Event()

    async def handle(self, prompt: str, **_kwargs) -> str:
        if self.cleaned:
            raise AssertionError("turn dispatched after scheduler cleanup")
        self.handled.append(prompt)
        if prompt == "interactive":
            self.started.set()
            await self.release.wait()
        if self.cleaned:
            raise AssertionError("scheduler cleaned before turn completed")
        return prompt

    async def cleanup(self) -> None:
        self.cleanup_calls += 1
        self.cleaned = True
        self.cleanup_finished.set()
        if self.cleanup_error is not None:
            raise self.cleanup_error


class _LifecycleBuilder:
    def __init__(self):
        self.instances: list[_LifecycleScheduler] = []
        self.fail_session: str | None = None

    def build(self, session_id: str) -> _LifecycleScheduler:
        if session_id == self.fail_session:
            raise RuntimeError("replacement construction failed")
        scheduler = _LifecycleScheduler(session_id)
        self.instances.append(scheduler)
        return scheduler


def test_scheduler_state_serializes_queued_channel_and_cron_turns_with_switch():
    async def scenario():
        builder = _LifecycleBuilder()
        state = session_flow._SchedulerState.create(builder, "old")
        old = state.scheduler

        interactive = asyncio.create_task(state.dispatch_handle("interactive"))
        await old.started.wait()
        channel = asyncio.create_task(state.dispatch_handle("channel"))
        await asyncio.sleep(0)
        switch = asyncio.create_task(state.switch("new"))
        await asyncio.sleep(0)
        cron = asyncio.create_task(state.dispatch_handle("cron"))

        old.release.set()
        await asyncio.gather(interactive, channel, switch, cron)
        await old.cleanup_finished.wait()
        new = state.scheduler

        assert old.handled == ["interactive", "channel"]
        assert old.cleanup_calls == 1
        assert new.handled == ["cron"]
        assert new.cleanup_calls == 0

        await state.cleanup()
        assert old.cleanup_calls == 1
        assert new.cleanup_calls == 1

    asyncio.run(scenario())


def test_switch_blocks_dispatch_until_cwd_and_ui_commit(monkeypatch):
    async def scenario():
        builder = _LifecycleBuilder()
        state = session_flow._SchedulerState.create(builder, "old")
        args = SimpleNamespace(session="old")
        restore_started = asyncio.Event()
        allow_restore = asyncio.Event()

        class Prompt:
            status_line = None

            def __init__(self):
                self.session_id = "old"
                self.history_reset = False

            def update_session(self, session_id):
                self.session_id = session_id

            def reset_history(self):
                self.history_reset = True

        prompt = Prompt()

        async def blocked_prepare(_builder, session_id, **_kwargs):
            assert session_id == "new"
            restore_started.set()
            await allow_restore.wait()
            return session_flow._SessionSwitchTarget(
                cwd=None,
                agent_name=None,
                agent_definition=None,
            )

        monkeypatch.setattr(session_flow, "_prepare_session_switch_target", blocked_prepare)

        switch_task = asyncio.create_task(
            session_flow._switch_active_session(
                state,
                args,
                "new",
                interactive_prompt=prompt,
            )
        )
        await restore_started.wait()
        dispatch_task = asyncio.create_task(state.dispatch_handle("channel-during-switch"))
        await asyncio.sleep(0)

        assert not dispatch_task.done()
        assert state.session_id == "old"
        assert args.session == "old"
        assert prompt.session_id == "old"
        assert prompt.history_reset is False

        allow_restore.set()
        replacement = await switch_task
        await dispatch_task

        assert state.scheduler is replacement
        assert replacement.handled == ["channel-during-switch"]
        assert args.session == "new"
        assert prompt.session_id == "new"
        assert prompt.history_reset is True

        await state.cleanup()

    asyncio.run(scenario())


def test_switch_cancellation_during_ui_commit_restores_old_state(monkeypatch, tmp_path):
    async def scenario():
        display_started = asyncio.Event()

        class Builder(_LifecycleBuilder):
            def build(self, session_id):
                scheduler = super().build(session_id)
                if session_id == "new":

                    async def blocked_display_name():
                        display_started.set()
                        await asyncio.Event().wait()

                    scheduler.session.get_display_name = blocked_display_name
                return scheduler

        builder = Builder()
        state = session_flow._SchedulerState.create(builder, "old")
        old = state.scheduler
        args = SimpleNamespace(session="old", agent="old-agent")
        old_history = object()
        old_usage = object()

        class StatusLine:
            def __init__(self):
                self.session_id = "old"
                self._display_name = "old display"
                self.usage_tracker = old_usage

            def update_session(self, session_id):
                self.session_id = session_id
                self._display_name = None

            def update_display_name(self, display_name):
                self._display_name = display_name

        class Prompt:
            def __init__(self):
                self.status_line = StatusLine()
                self.history = old_history

            def update_session(self, session_id):
                self.status_line.update_session(session_id)

            def reset_history(self):
                self.history = object()

        prompt = Prompt()
        original_cwd = tmp_path / "old-cwd"
        target_cwd = tmp_path / "new-cwd"
        original_cwd.mkdir()
        target_cwd.mkdir()
        monkeypatch.chdir(original_cwd)

        async def prepared_target(_builder, session_id, **_kwargs):
            assert session_id == "new"
            return session_flow._SessionSwitchTarget(
                cwd=target_cwd,
                agent_name="new-agent",
                agent_definition=None,
            )

        monkeypatch.setattr(
            session_flow,
            "_prepare_session_switch_target",
            prepared_target,
        )
        cwd_hook_payloads = []

        def dispatch_cwd_hook(*, event_name, payload, **_kwargs):
            assert event_name == "CwdChanged"
            cwd_hook_payloads.append(payload)
            return SimpleNamespace(watch_paths=[])

        monkeypatch.setattr(session_flow, "dispatch_command_hooks", dispatch_cwd_hook)

        switch_task = asyncio.create_task(
            session_flow._switch_active_session(
                state,
                args,
                "new",
                interactive_prompt=prompt,
            )
        )
        await display_started.wait()
        replacement = builder.instances[-1]
        switch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await switch_task
        await replacement.cleanup_finished.wait()

        assert state.scheduler is old
        assert state.session_id == "old"
        assert args.session == "old"
        assert args.agent == "old-agent"
        assert prompt.status_line.session_id == "old"
        assert prompt.status_line._display_name == "old display"
        assert prompt.status_line.usage_tracker is old_usage
        assert prompt.history is old_history
        assert session_flow.Path.cwd() == original_cwd
        assert cwd_hook_payloads == []
        assert old.cleanup_calls == 0
        assert replacement.cleanup_calls == 1

        await state.cleanup()
        assert old.cleanup_calls == 1
        assert replacement.cleanup_calls == 1

    asyncio.run(scenario())


def test_real_target_preparation_cancellation_does_not_fire_or_consume_cwd_hook(
    monkeypatch,
    tmp_path,
):
    async def scenario():
        from koder_agent.harness.hooks import runtime as hooks_runtime

        display_started = asyncio.Event()
        hook_payloads = []

        class Builder(_LifecycleBuilder):
            def __init__(self):
                super().__init__()
                self.block_next_display = True

            def build(self, session_id):
                scheduler = super().build(session_id)
                if session_id == "new" and self.block_next_display:
                    self.block_next_display = False

                    async def blocked_display_name():
                        display_started.set()
                        await asyncio.Event().wait()

                    scheduler.session.get_display_name = blocked_display_name
                return scheduler

        class Probe:
            def __init__(self, session_id, *_args, **_kwargs):
                self.session_id = session_id

            async def get_cwd(self):
                return str(target_cwd)

            async def get_agent(self):
                return None

            def close(self):
                return None

        class StatusLine:
            def __init__(self):
                self.session_id = "old"
                self._display_name = "old display"
                self.usage_tracker = object()

            def update_session(self, session_id):
                self.session_id = session_id
                self._display_name = None

            def update_display_name(self, display_name):
                self._display_name = display_name

        class Prompt:
            def __init__(self):
                self.status_line = StatusLine()
                self.history = object()

            def update_session(self, session_id):
                self.status_line.update_session(session_id)

            def reset_history(self):
                self.history = []

        home = tmp_path / "home"
        settings_dir = home / ".koder"
        original_cwd = tmp_path / "old-cwd"
        target_cwd = tmp_path / "new-cwd"
        settings_dir.mkdir(parents=True)
        original_cwd.mkdir()
        target_cwd.mkdir()
        (settings_dir / "settings.json").write_text(
            json.dumps(
                {
                    "hooks": {
                        "CwdChanged": [
                            {
                                "hooks": [
                                    {
                                        "type": "command",
                                        "command": "record-cwd-change",
                                        "once": True,
                                    }
                                ]
                            }
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(original_cwd)
        monkeypatch.setattr("koder_agent.core.session.EnhancedSQLiteSession", Probe)

        def record_hook(*, payload_text, **_kwargs):
            hook_payloads.append(json.loads(payload_text))
            return (
                0,
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "watchPaths": ["relative-watch-path"],
                        }
                    }
                ),
                "",
            )

        monkeypatch.setattr(hooks_runtime, "_run_command_hook", record_hook)
        once_fired = set()
        watched_paths = {}
        monkeypatch.setattr(hooks_runtime, "_once_fired", once_fired)
        monkeypatch.setattr(hooks_runtime, "_watched_paths", watched_paths)

        builder = Builder()
        state = session_flow._SchedulerState.create(builder, "old")
        old = state.scheduler
        old_agent_definitions = object()
        old_selected_agent = object()
        state.agent_definitions = old_agent_definitions
        state.selected_agent = old_selected_agent
        old_todo_map = dict(state.todo_stores_by_identity)
        args = SimpleNamespace(session="old", agent="old-agent")
        prompt = Prompt()

        switch_task = asyncio.create_task(
            session_flow._switch_active_session(
                state,
                args,
                "new",
                interactive_prompt=prompt,
            )
        )
        await display_started.wait()
        aborted_replacement = builder.instances[-1]
        switch_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await switch_task
        await aborted_replacement.cleanup_finished.wait()

        assert hook_payloads == []
        assert once_fired == set()
        assert watched_paths == {}
        assert state.scheduler is old
        assert state.session_id == "old"
        assert args.session == "old"
        assert session_flow.Path.cwd() == original_cwd
        assert state.agent_definitions is old_agent_definitions
        assert state.selected_agent is old_selected_agent
        assert state.todo_stores_by_identity == old_todo_map

        replacement = await session_flow._switch_active_session(
            state,
            args,
            "new",
            interactive_prompt=prompt,
        )
        await old.cleanup_finished.wait()

        assert state.scheduler is replacement
        assert hook_payloads == [
            {
                "event": "CwdChanged",
                "old_cwd": str(original_cwd),
                "cwd": str(target_cwd),
            }
        ]
        assert len(once_fired) == 1
        assert watched_paths == {str(original_cwd / "relative-watch-path"): None}

        await state.cleanup()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "cancel_stage",
    ["lifecycle_lock", "target_cwd", "target_agent", "probe_close"],
)
def test_switch_cancellation_at_each_target_preparation_await_preserves_old_state(
    monkeypatch,
    tmp_path,
    cancel_stage,
):
    async def scenario():
        stage_started = asyncio.Event()
        close_release = threading.Event()
        loop = asyncio.get_running_loop()

        class Probe:
            instances = []

            def __init__(self, session_id, *_args, **_kwargs):
                self.session_id = session_id
                self.close_calls = 0
                self.instances.append(self)

            async def get_cwd(self):
                if cancel_stage == "target_cwd":
                    stage_started.set()
                    await asyncio.Event().wait()
                return str(target_cwd)

            async def get_agent(self):
                if cancel_stage == "target_agent":
                    stage_started.set()
                    await asyncio.Event().wait()
                return "new-agent"

            def close(self):
                self.close_calls += 1
                if cancel_stage == "probe_close":
                    loop.call_soon_threadsafe(stage_started.set)
                    assert close_release.wait(timeout=5)

        builder = _LifecycleBuilder()
        state = session_flow._SchedulerState.create(builder, "old")
        old = state.scheduler
        args = SimpleNamespace(session="old", agent="old-agent")
        old_history = object()
        old_usage = object()
        old_auto_history = ["old suggestion"]
        old_speculative = object()

        class StatusLine:
            def __init__(self):
                self.session_id = "old"
                self._display_name = "old display"
                self.usage_tracker = old_usage

            def update_session(self, session_id):
                self.session_id = session_id
                self._display_name = None

            def update_display_name(self, display_name):
                self._display_name = display_name

        class AutoSuggest:
            def __init__(self):
                self._history = list(old_auto_history)
                self._speculative_suggestion = old_speculative

        class Prompt:
            def __init__(self):
                self.status_line = StatusLine()
                self.history = old_history
                self.auto_suggest = AutoSuggest()

            def update_session(self, session_id):
                self.status_line.update_session(session_id)

            def reset_history(self):
                self.history = object()
                self.auto_suggest._history.clear()
                self.auto_suggest._speculative_suggestion = None

        prompt = Prompt()
        original_cwd = tmp_path / "old-cwd"
        target_cwd = tmp_path / "new-cwd"
        original_cwd.mkdir()
        target_cwd.mkdir()
        monkeypatch.chdir(original_cwd)
        monkeypatch.setattr("koder_agent.core.session.EnhancedSQLiteSession", Probe)
        monkeypatch.setattr(
            session_flow,
            "dispatch_command_hooks",
            lambda **_kwargs: SimpleNamespace(watch_paths=[]),
        )

        if cancel_stage == "lifecycle_lock":
            await state._lifecycle_lock.acquire()

        switch_task = asyncio.create_task(
            session_flow._switch_active_session(
                state,
                args,
                "new",
                interactive_prompt=prompt,
            )
        )
        try:
            if cancel_stage == "lifecycle_lock":
                await asyncio.sleep(0)
            else:
                await stage_started.wait()
            switch_task.cancel()
            if cancel_stage == "probe_close":
                await asyncio.sleep(0)
                assert not switch_task.done()
                close_release.set()
            with pytest.raises(asyncio.CancelledError):
                await switch_task
        finally:
            close_release.set()
            if cancel_stage == "lifecycle_lock" and state._lifecycle_lock.locked():
                state._lifecycle_lock.release()

        assert state.scheduler is old
        assert state.session_id == "old"
        assert args.session == "old"
        assert args.agent == "old-agent"
        assert prompt.status_line.session_id == "old"
        assert prompt.status_line._display_name == "old display"
        assert prompt.status_line.usage_tracker is old_usage
        assert prompt.history is old_history
        assert prompt.auto_suggest._history == old_auto_history
        assert prompt.auto_suggest._speculative_suggestion is old_speculative
        assert session_flow.Path.cwd() == original_cwd
        assert old.cleanup_calls == 0
        assert len(builder.instances) == 1
        if cancel_stage == "lifecycle_lock":
            assert Probe.instances == []
        else:
            assert len(Probe.instances) == 1
            assert Probe.instances[0].close_calls == 1

        await state.cleanup()
        assert old.cleanup_calls == 1

    asyncio.run(scenario())


def test_switch_cancellation_requested_inside_no_await_commit_is_fully_committed(
    monkeypatch,
    tmp_path,
):
    async def scenario():
        builder = _LifecycleBuilder()
        state = session_flow._SchedulerState.create(builder, "old")
        old = state.scheduler
        args = SimpleNamespace(session="old", agent="old-agent")
        old_history = object()
        old_usage = object()
        queued_dispatch = None

        class StatusLine:
            def __init__(self):
                self.session_id = "old"
                self._display_name = "old display"
                self.usage_tracker = old_usage

            def update_session(self, session_id):
                self.session_id = session_id
                self._display_name = None

            def update_display_name(self, display_name):
                self._display_name = display_name

        class Prompt:
            def __init__(self):
                self.status_line = StatusLine()
                self.history = old_history

            def update_session(self, session_id):
                nonlocal queued_dispatch
                self.status_line.update_session(session_id)
                asyncio.current_task().cancel()
                queued_dispatch = asyncio.create_task(state.dispatch_handle("queued-after-commit"))

            def reset_history(self):
                self.history = []

        prompt = Prompt()
        original_cwd = tmp_path / "old-cwd"
        target_cwd = tmp_path / "new-cwd"
        original_cwd.mkdir()
        target_cwd.mkdir()
        monkeypatch.chdir(original_cwd)

        async def prepared_target(_builder, session_id, **_kwargs):
            assert session_id == "new"
            return session_flow._SessionSwitchTarget(
                cwd=target_cwd,
                agent_name="new-agent",
                agent_definition=None,
            )

        monkeypatch.setattr(
            session_flow,
            "_prepare_session_switch_target",
            prepared_target,
        )
        cwd_hook_payloads = []

        def dispatch_cwd_hook(*, event_name, payload, **_kwargs):
            assert event_name == "CwdChanged"
            cwd_hook_payloads.append(payload)
            return SimpleNamespace(watch_paths=[])

        monkeypatch.setattr(session_flow, "dispatch_command_hooks", dispatch_cwd_hook)

        switch_task = asyncio.create_task(
            session_flow._switch_active_session(
                state,
                args,
                "new",
                interactive_prompt=prompt,
            )
        )
        with pytest.raises(asyncio.CancelledError):
            await switch_task

        replacement = builder.instances[-1]
        await old.cleanup_finished.wait()
        assert state.scheduler is replacement
        assert state.session_id == "new"
        assert args.session == "new"
        assert args.agent == "new-agent"
        assert prompt.status_line.session_id == "new"
        assert prompt.status_line._display_name == "new"
        assert prompt.status_line.usage_tracker is replacement.usage_tracker
        assert prompt.history == []
        assert session_flow.Path.cwd() == target_cwd
        assert cwd_hook_payloads == [
            {
                "event": "CwdChanged",
                "old_cwd": str(original_cwd),
                "cwd": str(target_cwd),
            }
        ]
        assert old.cleanup_calls == 1
        assert replacement.cleanup_calls == 0
        assert queued_dispatch is not None
        assert await queued_dispatch == "queued-after-commit"
        assert replacement.handled == ["queued-after-commit"]
        assert state.scheduler is replacement
        assert args.session == "new"
        assert session_flow.Path.cwd() == target_cwd

        await state.cleanup()
        assert replacement.cleanup_calls == 1

    asyncio.run(scenario())


def test_old_cleanup_failure_does_not_desynchronize_switch_state(caplog, monkeypatch):
    monkeypatch.setattr(
        "koder_agent.core.session.EnhancedSQLiteSession", _FakeEnhancedSQLiteSession
    )

    async def scenario():
        builder = _LifecycleBuilder()
        state = session_flow._SchedulerState.create(builder, "old")
        old = state.scheduler
        old.cleanup_error = RuntimeError("old cleanup failed")
        args = SimpleNamespace(session="old")

        replacement = await session_flow._switch_active_session(state, args, "new")
        await old.cleanup_finished.wait()

        assert state.scheduler is replacement
        assert state.session_id == "new"
        assert args.session == "new"
        await state.dispatch_handle("after-switch")
        assert replacement.handled == ["after-switch"]
        assert old.cleanup_calls == 1

        await state.cleanup()

    with caplog.at_level("WARNING", logger=session_flow.__name__):
        asyncio.run(scenario())

    assert "Failed to clean retired session scheduler" in caplog.text


def test_scheduler_state_failed_replacement_keeps_old_scheduler_live():
    async def scenario():
        builder = _LifecycleBuilder()
        state = session_flow._SchedulerState.create(builder, "old")
        old = state.scheduler
        builder.fail_session = "broken"

        with pytest.raises(RuntimeError, match="replacement construction failed"):
            await state.switch("broken")

        assert state.scheduler is old
        assert old.cleanup_calls == 0
        await state.dispatch_handle("still-live")
        assert old.handled == ["still-live"]

        await state.cleanup()
        assert old.cleanup_calls == 1

    asyncio.run(scenario())


class _CleanupState:
    session_id = "cleanup-session"

    def __init__(self, calls, *, started=None, release=None):
        self.calls = calls
        self.started = started
        self.release = release

    async def cleanup(self):
        self.calls.append("scheduler")
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()


def test_cleanup_owner_guards_poller_channel_and_cron_failures():
    async def scenario():
        calls = []
        state = _CleanupState(calls)
        owner = session_flow._SessionCleanupOwner(
            state,
            bare_mode=False,
            previous_simple=None,
        )

        class FailingPoller:
            def stop(self):
                calls.append("poller")
                raise RuntimeError("poller failed")

        class FailingCron:
            async def stop(self):
                calls.append("cron")
                raise RuntimeError("cron failed")

        async def channel_worker():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                calls.append("channel")
                raise RuntimeError("channel cancellation failed")

        def unregister():
            calls.append("unregister")
            raise RuntimeError("unregister failed")

        owner.pr_poller = FailingPoller()
        owner.cron_prompt_runner = FailingCron()
        owner.channel_task = asyncio.create_task(channel_worker())
        await asyncio.sleep(0)
        owner.unregister_channel_callback = unregister
        owner._dispatch_session_end = lambda: calls.append("session-end")

        async def auto_dream():
            calls.append("auto-dream")

        owner._run_auto_dream = auto_dream

        await owner.finish()

        assert calls == [
            "poller",
            "unregister",
            "channel",
            "cron",
            "session-end",
            "auto-dream",
            "scheduler",
        ]

    asyncio.run(scenario())


def test_cleanup_owner_waits_through_repeated_cancellation():
    async def scenario():
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()
        state = _CleanupState(calls, started=started, release=release)
        owner = session_flow._SessionCleanupOwner(
            state,
            bare_mode=False,
            previous_simple=None,
        )
        owner._dispatch_session_end = lambda: calls.append("session-end")

        async def auto_dream():
            calls.append("auto-dream")

        owner._run_auto_dream = auto_dream

        finish_task = asyncio.create_task(owner.finish())
        await started.wait()
        finish_task.cancel()
        await asyncio.sleep(0)
        finish_task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await finish_task

        assert calls == ["session-end", "auto-dream", "scheduler"]
        await owner.finish()
        assert calls == ["session-end", "auto-dream", "scheduler"]

    asyncio.run(scenario())
