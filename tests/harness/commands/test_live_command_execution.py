import asyncio
import json
import os
import sqlite3
import subprocess
from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agents.models.chatcmpl_converter import Converter

from koder_agent.config import reset_config_manager
from koder_agent.core.session import EnhancedSQLiteSession
from koder_agent.core.usage_tracker import UsageTracker, usage_snapshot_path
from koder_agent.harness.agents.definitions import AgentDefinition, get_agent_definitions
from koder_agent.harness.agents.service import AgentService
from koder_agent.harness.commands.interactive import HarnessInteractiveCommandHandler
from koder_agent.harness.memory.compact import CompactionResult
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
from koder_agent.harness.tasks.service import TaskService


@pytest.fixture
def session_factory():
    """Own only Sessions explicitly created by these tests, not production probes."""
    with ExitStack() as cleanup:

        def create(*args, **kwargs):
            session = EnhancedSQLiteSession(*args, **kwargs)
            cleanup.callback(session.close)
            return session

        yield create


def _run(command: str, *, handler: HarnessInteractiveCommandHandler) -> str:
    return asyncio.run(handler.handle_slash_input(command, scheduler=None))


def _run_with_scheduler(
    command: str, *, handler: HarnessInteractiveCommandHandler, scheduler
) -> str:
    return asyncio.run(handler.handle_slash_input(command, scheduler=scheduler))


class _ResettableScheduler(SimpleNamespace):
    async def reset_agent(self):
        self.reset_count += 1
        self.dev_agent = None
        self._agent_initialized = False


def test_live_color_command_sets_resets_and_persists_session_color(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_factory("color-session")
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    assert "Please provide a color" in _run_with_scheduler(
        "/color", handler=handler, scheduler=scheduler
    )
    assert "Invalid color" in _run_with_scheduler(
        "/color chartreuse", handler=handler, scheduler=scheduler
    )

    set_output = _run_with_scheduler("/color red", handler=handler, scheduler=scheduler)
    session_output = _run_with_scheduler("/session", handler=handler, scheduler=scheduler)
    style_output = _run_with_scheduler("/output-style", handler=handler, scheduler=scheduler)

    assert set_output == "Session color set to: red"
    assert asyncio.run(session.get_color()) == "red"
    assert asyncio.run(session_factory("color-session").get_color()) == "red"
    assert "color: red" in session_output
    assert "color: red" in style_output

    reset_output = _run_with_scheduler("/color default", handler=handler, scheduler=scheduler)
    style_output = _run_with_scheduler("/output-style", handler=handler, scheduler=scheduler)

    assert reset_output == "Session color reset to default"
    assert asyncio.run(session.get_color()) is None
    assert asyncio.run(session_factory("color-session").get_color()) is None
    assert "color: default" in style_output


def test_keybindings_command_persists_valid_keys_and_rejects_invalid_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    set_output = _run("/keybindings set complete c-space", handler=handler)
    invalid_action_output = _run("/keybindings set missing c-x", handler=handler)
    invalid_key_output = _run("/keybindings set submit definitely-not-a-key", handler=handler)
    list_output = _run("/keybindings", handler=handler)

    saved = json.loads((tmp_path / ".koder" / "keybindings.json").read_text())
    assert saved == {"complete": "c-space"}
    assert "keybindings: set" in set_output
    assert "action: complete" in set_output
    assert "keybindings: unknown action" in invalid_action_output
    assert "keybindings: invalid key" in invalid_key_output
    assert "definitely-not-a-key" in invalid_key_output
    assert "overrides: 1" in list_output
    assert "- complete: c-space" in list_output


def test_vim_command_persists_state_without_interactive_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    on_output = _run("/vim on", handler=handler)
    off_output = _run("/vim off", handler=handler)
    invalid_output = _run("/vim maybe", handler=handler)

    state_path = tmp_path / ".koder" / "vim_state.json"
    saved = json.loads(state_path.read_text())
    assert saved == {"vim_enabled": False}
    assert "vim: enabled" in on_output
    assert f"settings_path: {state_path}" in on_output
    assert "vim: disabled" in off_output
    assert "Usage: /vim [on|off]" == invalid_output


def test_theme_command_persists_valid_theme_and_rejects_invalid_theme(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    default_output = _run("/theme", handler=handler)
    dark_output = _run("/theme dark", handler=handler)
    invalid_output = _run("/theme ultraviolet", handler=handler)
    reread_output = _run("/theme", handler=handler)
    reset_output = _run("/theme adaptive", handler=handler)

    settings_path = tmp_path / ".koder" / "settings.json"
    saved = json.loads(settings_path.read_text())
    assert saved["outputStyle"]["theme"] == "adaptive"
    assert "theme: adaptive" in default_output
    assert "theme: dark" in dark_output
    assert f"settings_path: {settings_path}" in dark_output
    assert "theme: invalid ultraviolet" in invalid_output
    assert "valid_themes: adaptive, dark, light" in invalid_output
    assert "theme: dark" in reread_output
    assert "theme: adaptive" in reset_output


def test_output_style_resets_all_style_controls_and_persists_state(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_factory("output-style-session")
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    default_output = _run_with_scheduler("/output-style", handler=handler, scheduler=scheduler)
    theme_output = _run_with_scheduler(
        "/output-style theme dark", handler=handler, scheduler=scheduler
    )
    color_output = _run_with_scheduler(
        "/output-style color cyan", handler=handler, scheduler=scheduler
    )
    vim_output = _run_with_scheduler("/output-style vim on", handler=handler, scheduler=scheduler)

    settings_path = tmp_path / ".koder" / "settings.json"
    saved = json.loads(settings_path.read_text())
    saved["statusLine"] = {
        "type": "command",
        "command": "printf style-ready",
        "padding": 0,
    }
    settings_path.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")

    configured_output = _run_with_scheduler("/output-style", handler=handler, scheduler=scheduler)
    reset_output = _run_with_scheduler("/output-style reset", handler=handler, scheduler=scheduler)
    reset_status_output = _run_with_scheduler("/output-style", handler=handler, scheduler=scheduler)

    saved_after_reset = json.loads(settings_path.read_text())
    vim_state = json.loads((tmp_path / ".koder" / "vim_state.json").read_text())
    reloaded_output = _run_with_scheduler(
        "/output-style",
        handler=HarnessInteractiveCommandHandler(emit_console=False),
        scheduler=scheduler,
    )

    assert "theme: adaptive" in default_output
    assert "color: default" in default_output
    assert "vim_mode: false" in default_output
    assert "statusline: not configured" in default_output
    assert "theme: dark" in theme_output
    assert color_output == "Session color set to: cyan"
    assert "vim: enabled" in vim_output
    assert "theme: dark" in configured_output
    assert "color: cyan" in configured_output
    assert "vim_mode: true" in configured_output
    assert "statusline: printf style-ready" in configured_output
    assert "output-style: reset" in reset_output
    assert "theme: adaptive" in reset_output
    assert "color: default" in reset_output
    assert "vim_mode: false" in reset_output
    assert "theme_settings_path:" in reset_output
    assert "vim_settings_path:" in reset_output
    assert "statusline_settings_path:" in reset_output
    assert saved_after_reset["outputStyle"]["theme"] == "adaptive"
    assert "statusLine" not in saved_after_reset
    assert vim_state == {"vim_enabled": False}
    assert asyncio.run(session.get_color()) is None
    assert "theme: adaptive" in reset_status_output
    assert "color: default" in reset_status_output
    assert "vim_mode: false" in reset_status_output
    assert "statusline: not configured" in reset_status_output
    assert "vim_mode: false" in reloaded_output


def test_version_command_reports_runtime_source_build_and_cli_banner(monkeypatch):
    monkeypatch.setenv("KODER_BUILD_TIME", "scenario-build")
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    output = _run("/version", handler=handler)

    assert "version: " in output
    assert "package: koder" in output
    assert "source: " in output
    assert "build_time: scenario-build" in output
    assert "cli_banner: " in output
    assert "(Koder)" in output
    assert "python:" not in output


def test_live_resume_resolves_existing_sessions_and_rejects_missing_targets(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setenv("HOME", str(tmp_path))
    handler = HarnessInteractiveCommandHandler(emit_console=False)
    current = session_factory("resume-current")
    target = session_factory("resume-target")
    duplicate_one = session_factory("resume-duplicate-one")
    duplicate_two = session_factory("resume-duplicate-two")
    asyncio.run(target.set_title("resume-title"))
    asyncio.run(duplicate_one.set_title("duplicate-title"))
    asyncio.run(duplicate_two.set_title("duplicate-title"))
    scheduler = SimpleNamespace(session=current)

    by_title = _run_with_scheduler("/resume resume-title", handler=handler, scheduler=scheduler)
    by_id = _run_with_scheduler("/resume resume-target", handler=handler, scheduler=scheduler)
    missing = _run_with_scheduler("/resume missing-session", handler=handler, scheduler=scheduler)
    duplicate = _run_with_scheduler("/resume duplicate-title", handler=handler, scheduler=scheduler)

    assert by_title == "session_switch:resume-target"
    assert by_id == "session_switch:resume-target"
    assert missing == "Session missing-session was not found."
    assert "Found 2 sessions matching duplicate-title" in duplicate


def test_live_backfill_sessions_migrates_legacy_ctx_rows(tmp_path, monkeypatch, session_factory):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_factory("backfill-current")
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)
    db_path = Path(session.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("create table ctx (sid text primary key, msgs text, title text)")
        conn.execute(
            "insert into ctx values (?, ?, ?)",
            (
                "legacy-session",
                json.dumps([{"role": "user", "content": "legacy hello"}]),
                "legacy-title",
            ),
        )

    first = _run_with_scheduler("/backfill-sessions", handler=handler, scheduler=scheduler)
    second = _run_with_scheduler("/backfill-sessions", handler=handler, scheduler=scheduler)
    resume_output = _run_with_scheduler(
        "/resume legacy-title", handler=handler, scheduler=scheduler
    )

    assert "backfill_sessions:" in first
    assert "migrated: 1" in first
    assert "legacy-title" in first
    assert "migrated: 0" in second
    assert resume_output == "session_switch:legacy-session"
    assert asyncio.run(session_factory("legacy-session").get_title()) == "legacy-title"
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("select migrated_sessions from migration_status").fetchone()[0] == 1


def test_live_branch_command_reports_dirty_state_and_creates_branches(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "koder@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Koder Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    sample = repo / "sample.txt"
    sample.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "sample.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True)
    sample.write_text("initial\nchanged\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    initial_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    status_output = _run("/branch", handler=handler)
    invalid_output = _run("/branch bad..name", handler=handler)
    created_output = _run("/branch scenario-branch", handler=handler)
    switched_output = _run(f"/branch {initial_branch}", handler=handler)

    assert f"branch: {initial_branch}" in status_output
    assert "dirty: true" in status_output
    assert "sample.txt" in status_output
    assert "branch: invalid name bad..name" == invalid_output
    assert "branch: scenario-branch" in created_output
    assert "action: created" in created_output
    assert "dirty: true" in created_output
    assert "branch: " + initial_branch in switched_output
    assert "action: switched" in switched_output


def test_live_rewind_help_is_available_without_history(tmp_path, session_factory):
    session = session_factory("rewind-help-session", db_path=str(tmp_path / "koder.db"))
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    help_output = _run_with_scheduler("/rewind help", handler=handler, scheduler=scheduler)
    assert help_output.startswith("Usage: /rewind [number] [conversation|code|both]")
    assert "conversation" in help_output
    assert "code" in help_output
    assert "both" in help_output


def test_live_rewind_lists_trim_counts_and_reports_removed_items(tmp_path, session_factory):
    session = session_factory("rewind-count-session", db_path=str(tmp_path / "koder.db"))
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "first prompt"},
                {"role": "assistant", "content": "first reply"},
                {"role": "user", "content": "second prompt"},
                {"role": "assistant", "content": "second reply"},
            ]
        )
    )
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    listing = _run_with_scheduler("/rewind", handler=handler, scheduler=scheduler)

    assert "1. second prompt (removes 2 newer transcript items)" in listing
    assert "2. first prompt (removes 4 newer transcript items)" in listing
    # The unfinished-capability disclaimer must not be advertised.
    assert "not yet available" not in listing

    result = _run_with_scheduler("/rewind 1", handler=handler, scheduler=scheduler)

    assert "Rewound conversation to prompt 1." in result
    assert "Removed transcript items: 2" in result
    assert "Restored input: second prompt" in result
    remaining = asyncio.run(
        session_factory("rewind-count-session", db_path=str(tmp_path / "koder.db")).get_items()
    )
    assert remaining == [
        {"role": "user", "content": "first prompt"},
        {"role": "assistant", "content": "first reply"},
    ]


def test_live_magic_docs_command_lists_and_refreshes_marker_docs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    docs = project / "docs"
    docs.mkdir(parents=True)
    (project / "AGENTS.md").write_text("# Project\n", encoding="utf-8")
    magic_doc = docs / "runtime.md"
    magic_doc.write_text("# MAGIC DOC: Runtime\n\nBody\n", encoding="utf-8")
    monkeypatch.chdir(project)

    handler = HarnessInteractiveCommandHandler(emit_console=False)

    status_output = _run("/magic-docs", handler=handler)
    refresh_output = _run("/magic-docs refresh", handler=handler)

    assert "magic_docs:" in status_output
    assert "docs/runtime.md: Runtime" in status_output
    assert "magic_docs: refresh" in refresh_output
    assert "updated: 1" in refresh_output
    assert "## Koder Session Notes" in magic_doc.read_text(encoding="utf-8")


def test_live_harness_commands_return_runtime_backed_output(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".zshrc").write_text('export PS1="project:\\W\\$ "\n', encoding="utf-8")
    home_settings = tmp_path / ".koder" / "settings.json"
    home_settings.parent.mkdir(parents=True, exist_ok=True)
    home_settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "koder_agent.harness.commands.security_review.llm_completion",
        lambda messages, model=None, **kwargs: asyncio.sleep(
            0, result="# Security Review\n\nNo high-confidence security findings."
        ),
    )
    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.run_pr_comments",
        lambda cwd=None: asyncio.sleep(
            0,
            result=("## Comments\n\n- @alice PR conversation:\n  > Looks good overall."),
        ),
    )

    lifecycle = PluginLifecycleService(tmp_path / "plugins")
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir()
    (plugin_dir / "plugin.json").write_text(
        json.dumps({"name": "demo-plugin", "version": "1.0.0"}),
        encoding="utf-8",
    )
    lifecycle.install_from_dir(plugin_dir)

    task_service = TaskService.in_memory()
    task_service.create_task("demo task")

    permission_service = PermissionService.default()
    handler = HarnessInteractiveCommandHandler(
        plugin_root=lifecycle.root,
        task_service=task_service,
        permission_service=permission_service,
    )

    plugin_output = _run("/plugin", handler=handler)
    tasks_output = _run("/tasks", handler=handler)
    permissions_output = _run("/permissions", handler=handler)
    theme_output = _run("/theme", handler=handler)
    model_output = _run("/model", handler=handler)
    files_output = _run("/files", handler=handler)
    diff_output = _run("/diff", handler=handler)
    branch_output = _run("/branch", handler=handler)
    plan_output = _run("/plan", handler=handler)
    hooks_output = _run("/hooks", handler=handler)
    vim_output = _run("/vim", handler=handler)
    login_output = _run("/login", handler=handler)
    logout_output = _run("/logout", handler=handler)
    commit_output = _run("/commit", handler=handler)
    commit_push_pr_output = _run("/commit-push-pr", handler=handler)
    review_output = _run("/review", handler=handler)
    release_notes_output = _run("/release-notes", handler=handler)
    doctor_output = _run("/doctor", handler=handler)
    usage_output = _run("/usage", handler=handler)
    effort_output = _run("/effort", handler=handler)
    version_output = _run("/version", handler=handler)
    env_output = _run("/env", handler=handler)
    add_dir_output = _run("/add-dir", handler=handler)
    agents_output = _run("/agents", handler=handler)
    fork_output = _run("/fork", handler=handler)
    issue_output = _run("/issue", handler=handler)
    pr_comments_output = _run("/pr_comments", handler=handler)
    peers_output = _run("/peers", handler=handler)
    feedback_output = _run("/feedback", handler=handler)
    statusline_output = _run("/statusline", handler=handler)
    color_output = _run("/color", handler=handler)
    ctx_viz_output = _run("/ctx_viz", handler=handler)
    sandbox_output = _run("/sandbox", handler=handler)
    security_repo = tmp_path / "security-review-repo"
    security_repo.mkdir()
    subprocess.run(["git", "init"], cwd=security_repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=security_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=security_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    security_file = security_repo / "app.py"
    security_file.write_text("print('before')\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "app.py"], cwd=security_repo, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=security_repo,
        check=True,
        capture_output=True,
        text=True,
    )
    security_file.write_text("print('after')\n", encoding="utf-8")
    with monkeypatch.context() as scoped_monkeypatch:
        scoped_monkeypatch.chdir(security_repo)
        security_review_output = _run("/security-review", handler=handler)
    summary_output = _run("/summary", handler=handler)

    assert "demo-plugin" in plugin_output
    assert "demo task" in tasks_output
    assert "mode:" in permissions_output
    assert "theme: adaptive" in theme_output
    assert "model:" in model_output
    assert files_output
    assert "## Diff" in diff_output
    assert "### Uncommitted changes" in diff_output
    assert branch_output
    assert "plan" in plan_output.lower()
    assert "hooks:" in hooks_output
    assert "vim:" in vim_output
    assert login_output  # /login was removed; returns unknown-command message
    assert logout_output  # /logout was removed; returns unknown-command message
    assert "Branch:" in commit_output or "branch:" in commit_output
    assert "staged_diff:" in commit_push_pr_output
    assert "unstaged_diff:" in commit_push_pr_output
    assert "review" in review_output.lower() or "No changes" in review_output
    assert release_notes_output
    assert "python:" in doctor_output
    assert "installation_type:" in doctor_output
    assert "invoked_binary:" in doctor_output
    assert "ripgrep_working:" in doctor_output
    assert "ripgrep_mode:" in doctor_output
    assert "ripgrep_path:" in doctor_output
    assert "requests:" in usage_output
    assert "effort level" in effort_output.lower()
    assert version_output.startswith("version: ")
    assert "package: koder" in version_output
    assert "cli_banner:" in version_output
    assert "python:" not in version_output
    assert "cwd:" in env_output
    assert "active_project_dir:" in add_dir_output
    assert "agents:" in agents_output
    assert "fork:" in fork_output
    assert "issue" in issue_output.lower()
    assert "## Comments" in pr_comments_output
    assert "peers:" in peers_output
    assert "feedback:" in feedback_output
    assert "repo:" in feedback_output
    assert "statusline: configured from" in statusline_output
    saved_statusline = json.loads(home_settings.read_text(encoding="utf-8"))
    assert saved_statusline["statusLine"]["type"] == "command"
    assert "Please provide a color." in color_output
    assert "Working directory:" in ctx_viz_output
    assert "sandbox_enabled:" in sandbox_output
    assert "backend: unix-local" in sandbox_output
    assert "# Security Review" in security_review_output
    assert summary_output


def test_plan_command_returns_harness_owned_response():
    handler = HarnessInteractiveCommandHandler()
    output = _run("/plan", handler=handler)
    assert "plan" in output.lower()


def test_feedback_command_persists_redacted_event(tmp_path, monkeypatch):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("KODER_FEEDBACK_SECRET", "feedback-secret-token-123456")
    monkeypatch.chdir(repo)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "koder@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Koder Test"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "sample.txt").write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "sample.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    empty_output = _run("/feedback", handler=handler)
    assert "saved: false" in empty_output
    assert "usage: /feedback <message>" in empty_output
    assert "cwd:" in empty_output

    output = _run(
        "/feedback please keep api_key=feedback-secret-token-123456",
        handler=handler,
    )
    assert "feedback: saved" in output
    assert "message: please keep api_key=[REDACTED]" in output
    assert "repo: No git remote configured." in output
    assert "branch:" in output
    assert "cwd:" in output
    assert "path:" in output
    assert "feedback-secret-token-123456" not in output

    feedback_path = home / ".koder" / "feedback" / "feedback.jsonl"
    content = feedback_path.read_text(encoding="utf-8")
    record = json.loads(content.splitlines()[-1])
    assert record["message"] == "please keep api_key=[REDACTED]"
    assert record["cwd"] == str(repo)
    assert record["repo"] == "No git remote configured."
    assert record["branch"]
    assert "git_status" in record
    assert "feedback-secret-token-123456" not in content


def test_plan_command_toggles_permission_mode_and_blocks_mutations():
    handler = HarnessInteractiveCommandHandler()

    enter_output = _run("/plan", handler=handler)
    blocked_output = _run("/permissions check write_file plan-output.txt", handler=handler)
    allowed_output = _run("/permissions check read_file sample.txt", handler=handler)
    exit_output = _run("/plan", handler=handler)

    assert "permission_mode: plan" in enter_output
    assert handler.permission_service.mode == PermissionMode.DEFAULT
    assert "allowed: false" in blocked_output
    assert "requires_approval: false" in blocked_output
    assert "plan mode: mutations not allowed" in blocked_output
    assert "allowed: true" in allowed_output
    assert "read-only tool allowed in plan mode" in allowed_output
    assert "permission_mode: default" in exit_output


def test_permissions_check_passes_notebook_path_to_permission_service():
    permission_service = PermissionService.default()
    permission_service.evaluate_tool_call_async = AsyncMock(  # type: ignore[method-assign]
        wraps=permission_service.evaluate_tool_call_async
    )
    handler = HarnessInteractiveCommandHandler(permission_service=permission_service)
    notebook_path = "analysis.ipynb"

    output = _run(f"/permissions check notebook_edit {notebook_path}", handler=handler)

    permission_service.evaluate_tool_call_async.assert_awaited_once_with(
        "notebook_edit",
        {"notebook_path": notebook_path},
    )
    assert "tool: notebook_edit" in output
    assert "allowed: false" in output
    assert "requires_approval: true" in output
    assert "workspace mutation requires approval" in output


def test_help_command_uses_registry_descriptions_without_execute_placeholders():
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    output = _run("/help", handler=handler)
    command_output = _run("/help hooks", handler=handler)
    missing_output = _run("/help nope", handler=handler)

    assert "Command Catalog:" in output
    assert "/hooks" in output
    assert "List configured runtime hooks" in output
    assert "/agents" in output
    assert "List, inspect, create, and manage local agents" in output
    assert "Execute /" not in output
    for command_name in handler.registry.list_names():
        assert f"/{command_name}" in output
    assert command_output == "/hooks: List configured runtime hooks"
    assert missing_output == "help: unknown command /nope\nUse /help to list commands."


def test_local_semantic_gap_commands_are_runtime_backed(tmp_path, monkeypatch, session_factory):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_factory("semantic-gap-session", db_path=str(tmp_path / "koder.db"))
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "inspect sample failure"},
                {
                    "role": "assistant",
                    "content": "checking it",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "function": {"name": "run_shell", "arguments": "pytest -k sample"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tool-1", "content": "sample failed"},
                {"role": "user", "content": "summarize the failure"},
            ]
        )
    )
    usage = SimpleNamespace(request_count=2, input_tokens=120, output_tokens=45)
    scheduler = SimpleNamespace(session=session, usage_tracker=SimpleNamespace(session_usage=usage))
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    keybindings_output = _run("/keybindings", handler=handler)
    keybinding_set_output = _run("/keybindings set submit c-m", handler=handler)
    output_style_output = _run("/output-style status", handler=handler)
    thinkback_output = _run_with_scheduler("/thinkback", handler=handler, scheduler=scheduler)
    replay_output = _run_with_scheduler("/thinkback-play", handler=handler, scheduler=scheduler)
    tool_output = _run_with_scheduler("/debug-tool-call", handler=handler, scheduler=scheduler)
    tool_detail_output = _run_with_scheduler(
        "/debug-tool-call show 1", handler=handler, scheduler=scheduler
    )
    bughunter_output = _run("/bughunter sample", handler=handler)

    assert "keybindings:" in keybindings_output
    assert "- submit:" in keybindings_output
    assert "keybindings: set" in keybinding_set_output
    assert "action: submit" in keybinding_set_output
    assert "output-style:" in output_style_output
    assert "controls: /theme, /color, /statusline, /vim" in output_style_output
    assert "thinkback: session review" in thinkback_output
    assert "inspect sample failure" in thinkback_output
    assert "thinkback-play: replaying" in replay_output
    assert "assistant: checking it" in replay_output
    assert "debug-tool-call:" in tool_output
    assert "run_shell" in tool_output
    assert "debug-tool-call: detail" in tool_detail_output
    assert "sample failed" not in tool_detail_output
    assert "bughunter: local triage" in bughunter_output
    assert "focus: sample" in bughunter_output
    assert "working_tree:" in bughunter_output
    assert "diff_evidence:" in bughunter_output


def test_insights_command_reports_session_counts(tmp_path, monkeypatch, session_factory):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_factory("insights-session", db_path=str(tmp_path / "koder.db"))
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "inspect current session"},
                {
                    "role": "assistant",
                    "content": "reading file",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "function": {
                                "name": "read_file",
                                "arguments": json.dumps({"path": "AGENTS.md"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "tool-1", "content": "# Test project"},
                {
                    "type": "function_call",
                    "name": "write_file",
                    "arguments": json.dumps({"file_path": "docs/runtime-notes.md"}),
                },
            ]
        )
    )
    usage = SimpleNamespace(
        request_count=2,
        input_tokens=120,
        output_tokens=45,
        total_cost=0.0012,
    )
    scheduler = SimpleNamespace(
        session=session,
        usage_tracker=SimpleNamespace(
            session_usage=usage,
            get_per_model_usage=lambda: {},
        ),
    )
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    output = _run_with_scheduler("/insights", handler=handler, scheduler=scheduler)

    assert "Session Insights:" in output
    assert "Transcript items: 4" in output
    assert "Messages: 3" in output
    assert "User messages: 1" in output
    assert "Assistant messages: 1" in output
    assert "Tool results: 1" in output
    assert "Tool calls: 1" in output
    assert "Tool usage:" in output
    assert "- read_file: 1" in output
    assert "Files in context: 2" in output
    assert "- AGENTS.md" in output
    assert "- docs/runtime-notes.md" in output
    assert "Requests: 2" in output
    assert "Input tokens: 120" in output
    assert "Output tokens: 45" in output
    assert "Total cost: $0.0012" in output
    assert "Avg tokens/request: 60 in / 22 out" in output


def test_usage_and_cost_commands_load_persisted_usage_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KODER_MODEL", "gpt-4.1")
    reset_config_manager()
    try:
        session_id = "usage-session"
        usage_path = usage_snapshot_path(session_id, home=tmp_path)
        seeded = UsageTracker()
        seeded.record_usage(1000, 2000, context_tokens=4096, model="gpt-4.1")
        seeded.save(usage_path)

        live_tracker = UsageTracker()
        scheduler = SimpleNamespace(usage_tracker=live_tracker, usage_path=usage_path)
        handler = HarnessInteractiveCommandHandler(emit_console=False)

        usage_output = _run_with_scheduler("/usage", handler=handler, scheduler=scheduler)
        cost_output = _run_with_scheduler("/cost", handler=handler, scheduler=scheduler)

        assert "requests: 1" in usage_output
        assert "input_tokens: 1000" in usage_output
        assert "output_tokens: 2000" in usage_output
        assert "last_input_tokens: 1000" in usage_output
        assert "last_output_tokens: 2000" in usage_output
        assert "context_tokens: 4096" in usage_output
        assert "cost: 0.0180" in usage_output
        assert "rate_limit_status: unknown" in usage_output
        assert "requests: 1" in cost_output
        assert "context_tokens: 4096" in cost_output
        assert "cost: 0.0180" in cost_output
    finally:
        reset_config_manager()


def test_schedule_command_lists_cron_registry_and_edges(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    from koder_agent.tools import cron as cron_module

    cron_module._set_cron_storage(None)
    handler = HarnessInteractiveCommandHandler(emit_console=False)
    try:
        empty_output = _run("/schedule", handler=handler)
        assert "No scheduled tasks" in empty_output
        assert _run("/schedule delete fixture", handler=handler) == "Usage: /schedule"

        first = json.loads(cron_module.cron_create("0 9 * * *", "morning standup"))
        second = json.loads(
            cron_module.cron_create("30 14 * * 1", "monday review", recurring=False)
        )

        list_output = _run("/schedule", handler=handler)
        assert "Scheduled tasks (2):" in list_output
        assert f"id: {first['id']}" in list_output
        assert "cron: 0 9 * * *" in list_output
        assert "human_schedule: at 9:00" in list_output
        assert "recurring: true" in list_output
        assert "prompt: morning standup" in list_output
        assert f"id: {second['id']}" in list_output
        assert "cron: 30 14 * * 1" in list_output
        assert "human_schedule: on Mon at 14:30" in list_output
        assert "recurring: false" in list_output
        assert "prompt: monday review" in list_output

        cron_module.cron_delete(first["id"])
        after_delete_output = _run("/schedule", handler=handler)
        assert "Scheduled tasks (1):" in after_delete_output
        assert first["id"] not in after_delete_output
        assert f"id: {second['id']}" in after_delete_output

        registry_path = home / ".koder" / "scheduled_tasks.json"
        registry_path.write_text("{not json", encoding="utf-8")
        cron_module._set_cron_storage(None)
        malformed_output = _run("/schedule", handler=handler)
        assert "schedule: failed to read scheduled task registry" in malformed_output
        assert str(registry_path) in malformed_output
        assert "error:" in malformed_output
    finally:
        cron_module._set_cron_storage(None)


def test_loop_command_creates_lists_and_deletes_cron_jobs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    from koder_agent.tools import cron as cron_module

    cron_module._set_cron_storage(None)
    handler = HarnessInteractiveCommandHandler(emit_console=False)
    try:
        empty_output = _run("/loop", handler=handler)
        assert "No loop jobs" in empty_output
        assert "Usage: /loop" in empty_output

        created_output = _run("/loop 0 9 * * * morning standup", handler=handler)
        assert "Loop job created" in created_output
        assert "cron: 0 9 * * *" in created_output
        assert "recurring: true" in created_output
        assert "prompt: morning standup" in created_output

        every_output = _run("/loop @every 5m check build", handler=handler)
        assert "Loop job created" in every_output
        assert "cron: */5 * * * *" in every_output
        assert "prompt: check build" in every_output

        once_output = _run("/loop once 30 14 * * 1 monday review", handler=handler)
        assert "Loop job created" in once_output
        assert "cron: 30 14 * * 1" in once_output
        assert "recurring: false" in once_output

        list_output = _run("/loop list", handler=handler)
        assert "Loop jobs (3):" in list_output
        assert "cron: 0 9 * * *" in list_output
        assert "cron: */5 * * * *" in list_output
        assert "cron: 30 14 * * 1" in list_output

        jobs = json.loads(cron_module.cron_list())["jobs"]
        delete_output = _run(f"/loop delete {jobs[0]['id']}", handler=handler)
        assert delete_output == f"Loop job deleted: {jobs[0]['id']}"

        after_delete_output = _run("/loop", handler=handler)
        assert "Loop jobs (2):" in after_delete_output
        assert jobs[0]["id"] not in after_delete_output

        too_fast_output = _run("/loop @every 30s too fast", handler=handler)
        assert "loop: unsupported schedule" in too_fast_output
        assert "sub-minute" in too_fast_output

        after_turn_output = _run("/loop @after-turn follow up", handler=handler)
        assert "loop: unsupported schedule" in after_turn_output
        assert "@after-turn" in after_turn_output
    finally:
        cron_module._set_cron_storage(None)


def test_loop_command_reports_active_registry_path_on_read_error(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))

    from koder_agent.harness.cron.storage import CronStorage
    from koder_agent.tools import cron as cron_module

    custom_path = tmp_path / "custom" / "loop_tasks.json"
    custom_path.parent.mkdir(parents=True)
    custom_path.write_text("{not json", encoding="utf-8")
    cron_module._set_cron_storage(CronStorage(custom_path))
    handler = HarnessInteractiveCommandHandler(emit_console=False)
    try:
        output = _run("/loop", handler=handler)

        assert "loop: failed to read scheduled task registry" in output
        assert str(custom_path) in output
        assert str(home / ".koder" / "scheduled_tasks.json") not in output
    finally:
        cron_module._set_cron_storage(None)


def test_command_list_prefers_runtime_command_over_shadowed_skill():
    handler = HarnessInteractiveCommandHandler(emit_console=False)
    command_names = [name for name, _description in handler.get_command_list()]

    assert command_names.count("loop") == 1


def test_bughunter_reports_diff_evidence_and_clean_edge(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)

    git("init")
    git("config", "user.email", "koder@example.invalid")
    git("config", "user.name", "Koder Test")
    target = repo / "bughunter_target.py"
    target.write_text(
        "def ratio(numerator, denominator):\n    return numerator / denominator\n",
        encoding="utf-8",
    )
    git("add", "bughunter_target.py")
    git("commit", "-m", "bughunter baseline")
    target.write_text(
        "def ratio(numerator, denominator):\n    return numerator / 0\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(repo)
    handler = HarnessInteractiveCommandHandler(emit_console=False)
    dirty_output = _run("/bughunter division regression", handler=handler)

    assert "working_tree: dirty" in dirty_output
    assert "bughunter_target.py" in dirty_output
    assert "diff_evidence:" in dirty_output
    assert "-    return numerator / denominator" in dirty_output
    assert "+    return numerator / 0" in dirty_output

    git("add", "bughunter_target.py")
    git("commit", "-m", "bughunter clean edge")
    clean_output = _run("/bughunter clean edge", handler=handler)

    assert "working_tree: clean" in clean_output
    assert "diff_stat:\nnone" in clean_output
    assert "diff_evidence:\nnone" in clean_output


def test_debug_tool_call_redacts_sensitive_arguments_and_outputs(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OPENAI_API_KEY", "debug-secret-value-12345")
    session = session_factory("debug-tool-session")
    asyncio.run(
        session.add_items(
            [
                {
                    "role": "assistant",
                    "content": "checking it",
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "function": {
                                "name": "run_shell",
                                "arguments": json.dumps(
                                    {
                                        "command": "echo ready",
                                        "api_key": "debug-secret-value-12345",
                                    }
                                ),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool-1",
                    "name": "run_shell",
                    "content": "completed secret=debug-secret-value-12345",
                },
            ]
        )
    )
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    list_output = _run_with_scheduler("/debug-tool-call", handler=handler, scheduler=scheduler)
    call_detail = _run_with_scheduler(
        "/debug-tool-call show 1", handler=handler, scheduler=scheduler
    )
    output_detail = _run_with_scheduler(
        "/debug-tool-call show 2", handler=handler, scheduler=scheduler
    )

    assert "debug-tool-call: 2 recorded item(s)" in list_output
    assert "call run_shell id=tool-1" in list_output
    assert "output run_shell id=tool-1" in list_output
    assert "api_key" in list_output
    assert "[REDACTED]" in list_output
    assert "debug-secret-value-12345" not in list_output
    assert "preview:" in call_detail
    assert "api_key" in call_detail
    assert "[REDACTED]" in call_detail
    assert "debug-secret-value-12345" not in call_detail
    assert "kind: output" in output_detail
    assert "secret=[REDACTED]" in output_detail
    assert "debug-secret-value-12345" not in output_detail


def test_local_setup_and_diagnostic_gap_commands_are_runtime_backed(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "AGENTS.md").write_text("# Fixture\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    session = session_factory("setup-gap-session", db_path=str(tmp_path / "koder.db"))
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    mcp_output = _run("/mcp", handler=handler)
    mcp_invalid_output = _run("/mcp now", handler=handler)
    assistant_output = _run_with_scheduler("/assistant", handler=handler, scheduler=scheduler)
    assistant_list_output = _run("/assistant list", handler=handler)
    verifier_output = _run("/init-verifiers cli", handler=handler)
    skills_output = _run("/skills", handler=handler)
    verifier_exists_output = _run("/init-verifiers cli", handler=handler)

    assert mcp_output == "No MCP servers configured."
    assert mcp_invalid_output == "Usage: /mcp"
    assert "assistant:" in assistant_output
    assert "active_profile:" in assistant_output
    assert "related_commands: /agents, /model, /session, /skills" in assistant_output
    assert "assistant_profiles:" in assistant_list_output
    assert "general-purpose" in assistant_list_output
    assert "init-verifiers: created" in verifier_output
    assert "name: verifier-cli" in verifier_output
    assert "type: cli" in verifier_output
    assert "verifier-cli" in skills_output
    assert "Verify CLI and TUI behavior" in skills_output
    assert "init-verifiers: exists" in verifier_exists_output

    skill_file = repo / ".koder" / "skills" / "verifier-cli" / "SKILL.md"
    assert skill_file.exists()
    skill_text = skill_file.read_text(encoding="utf-8")
    assert "allowed-tools:" in skill_text
    assert "run_shell:tmux *" in skill_text


def test_init_command_generates_local_agents_md_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "runtime-notes.md").write_text(
        "# MAGIC DOC: Runtime Notes\n\nKeep this fixture current.\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    assert _run("/init extra", handler=handler) == "Usage: /init"
    output = _run("/init", handler=handler)

    assert "AGENTS.md generated." in output
    assert f"path: {repo / 'AGENTS.md'}" in output
    assert "commands_detected: 4" in output
    assert "Found 1 magic doc(s):" in output
    assert "docs/runtime-notes.md: Runtime Notes" in output
    assert "tip: run /init-explore" in output

    agents_md = repo / "AGENTS.md"
    assert agents_md.exists()
    content = agents_md.read_text(encoding="utf-8")
    assert "# AGENTS.md" in content
    assert "This file provides guidance to Koder" in content
    assert "## Commands" in content
    assert "`uv run pytest`" in content
    assert "## Working Guidelines" in content

    assert (
        _run("/init", handler=handler)
        == "AGENTS.md already exists. Run /init-explore to improve it from the codebase."
    )
    assert agents_md.read_text(encoding="utf-8") == content


def test_advisor_command_runs_local_review_with_current_session_and_repo(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setenv("HOME", str(tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = repo / "auth.py"
    tracked.write_text("def auth():\n    return 'ok'\n", encoding="utf-8")
    subprocess.run(["git", "add", "auth.py"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked.write_text("def auth(user_input):\n    return user_input\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    session = session_factory("advisor-live-session", db_path=str(tmp_path / "koder.db"))
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "Please review auth changes."},
                {"role": "assistant", "content": "I updated auth and tests."},
            ]
        )
    )
    scheduler = SimpleNamespace(session=session)
    captured: dict[str, object] = {}

    async def _fake_completion(messages, model=None, **kwargs):
        captured["messages"] = messages
        captured["model"] = model
        return "# Advisor Review\n\n## Recommended Next Steps\n- Add a regression test."

    monkeypatch.setattr("koder_agent.harness.commands.advisor.llm_completion", _fake_completion)

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run_with_scheduler(
        "/advisor focus on auth regressions", handler=handler, scheduler=scheduler
    )

    assert output.startswith("# Advisor Review")
    assert "regression test" in output
    assert str(captured["model"]).endswith("gpt-5.5")
    prompt = captured["messages"][-1]["content"]
    assert "Please review auth changes." in prompt
    assert "focus on auth regressions" in prompt.lower()
    assert "return user_input" in prompt


def test_brief_command_toggles_runtime_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    enabled = _run("/brief", handler=handler)
    disabled = _run("/brief", handler=handler)

    assert enabled == "Brief-only mode enabled"
    assert disabled == "Brief-only mode disabled"


def test_buddy_command_hatches_and_pets_companion(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USER", "live-buddy")
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    hatched = _run("/buddy", handler=handler)
    pet = _run("/buddy", handler=handler)

    assert hatched.startswith("buddy: hatched")
    assert "name:" in hatched
    assert pet.startswith("buddy: pet")


def test_ctx_viz_command_reports_session_snapshot_when_scheduler_present(
    tmp_path, monkeypatch, session_factory
):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_factory("ctx-viz-session", db_path=str(tmp_path / "koder.db"))
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "Investigate auth failures."},
                {"role": "assistant", "content": "I inspected auth.py."},
            ]
        )
    )
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    output = _run_with_scheduler("/ctx_viz", handler=handler, scheduler=scheduler)

    assert "Working directory:" in output
    assert "Session messages: 2" in output
    assert "Investigate auth failures." in output
    assert "Context usage (estimated):" in output
    assert "project context:" in output
    assert "user messages:" in output
    assert "assistant messages:" in output
    assert "free space:" in output
    assert "tokens (" in output


def test_sandbox_command_reports_and_updates_local_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    status_output = _run("/sandbox", handler=handler)
    assert "sandbox_enabled: false" in status_output
    assert "backend: unix-local" in status_output

    choices_output = _run("/sandbox enable", handler=handler)
    assert "sandbox: choose a backend" in choices_output
    assert "unix-local" in choices_output

    enable_output = _run("/sandbox enable unix-local", handler=handler)
    assert "sandbox: enabled" in enable_output
    assert "backend: unix-local" in enable_output

    exclude_output = _run('/sandbox exclude "touch *"', handler=handler)
    assert "sandbox: excluded command added" in exclude_output
    assert "pattern: touch *" in exclude_output

    settings_path = tmp_path / ".koder" / "settings.local.json"
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    assert saved["sandbox"]["enabled"] is True
    assert saved["sandbox"]["backend"] == "unix-local"
    assert saved["sandbox"]["excludedCommands"] == ["touch *"]


def test_sandbox_command_reports_managed_policy_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".koder").mkdir(parents=True)
    (tmp_path / ".koder" / "managed-settings.json").write_text(
        json.dumps({"sandbox": {"enabled": True, "backend": "unix-local"}}),
        encoding="utf-8",
    )
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    output = _run("/sandbox", handler=handler)
    toggle_output = _run("/sandbox enable docker", handler=handler)

    assert "sandbox_enabled: true" in output
    assert "backend: unix-local" in output
    assert "policy_locked: true" in output
    assert "sandbox: settings locked by managed policy" in toggle_output
    assert "sandbox_enabled: true" in toggle_output
    assert "policy_locked: true" in toggle_output


def test_env_command_persists_session_scoped_variables(tmp_path, monkeypatch, session_factory):
    monkeypatch.setenv("HOME", str(tmp_path))
    session = session_factory("env-session")
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    assert _run_with_scheduler("/env DEMO_ENV=hello", handler=handler, scheduler=scheduler) == (
        "env: set DEMO_ENV for this session."
    )
    env_output = _run_with_scheduler("/env", handler=handler, scheduler=scheduler)
    assert "session_env:" in env_output
    assert "- DEMO_ENV=hello" in env_output
    monkeypatch.setenv("DEMO_ENV", "hello")

    assert _run_with_scheduler("/env unset DEMO_ENV", handler=handler, scheduler=scheduler) == (
        "env: removed DEMO_ENV from this session."
    )
    assert "DEMO_ENV" not in os.environ
    env_output = _run_with_scheduler("/env", handler=handler, scheduler=scheduler)
    assert "session_env: none" in env_output


def test_model_and_effort_commands_update_runtime_config_and_reset_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KODER_MODEL", "gpt-4.1")
    monkeypatch.delenv("KODER_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("KODER_REASONING_DISPLAY", raising=False)
    reset_config_manager()
    try:
        scheduler = _ResettableScheduler(
            session=SimpleNamespace(session_id="model-session"),
            dev_agent=object(),
            _agent_initialized=True,
            reset_count=0,
        )
        handler = HarnessInteractiveCommandHandler(emit_console=False)

        model_output = _run_with_scheduler(
            "/model anthropic/claude-sonnet-4-6",
            handler=handler,
            scheduler=scheduler,
        )
        status_output = _run_with_scheduler("/status", handler=handler, scheduler=scheduler)
        config_output = _run_with_scheduler("/config", handler=handler, scheduler=scheduler)

        assert "model: claude-sonnet-4-6" in model_output
        assert "provider: anthropic" in model_output
        assert "effective_model: litellm/anthropic/claude-sonnet-4-6" in model_output
        assert "agent_reloaded: True" in model_output
        assert os.environ["KODER_MODEL"] == "anthropic/claude-sonnet-4-6"
        assert "Model: litellm/anthropic/claude-sonnet-4-6" in status_output
        assert "provider: anthropic" in status_output
        assert "name: claude-sonnet-4-6" in config_output
        assert "provider: anthropic" in config_output
        assert scheduler.reset_count == 1

        effort_output = _run_with_scheduler("/effort high", handler=handler, scheduler=scheduler)
        effort_status = _run_with_scheduler("/effort", handler=handler, scheduler=scheduler)
        config_output = _run_with_scheduler("/config", handler=handler, scheduler=scheduler)

        assert "Set effort level to high" in effort_output
        assert "agent_reloaded: True" in effort_output
        assert os.environ["KODER_REASONING_EFFORT"] == "high"
        assert "Current effort level: high" == effort_status
        assert "reasoning_effort: high" in config_output
        assert scheduler.reset_count == 2

        auto_output = _run_with_scheduler("/effort auto", handler=handler, scheduler=scheduler)
        effort_status = _run_with_scheduler("/effort", handler=handler, scheduler=scheduler)
        config_output = _run_with_scheduler("/config", handler=handler, scheduler=scheduler)

        assert "Effort level reset to default: medium" in auto_output
        assert "KODER_REASONING_EFFORT" not in os.environ
        assert "Current effort level: medium" == effort_status
        assert "reasoning_effort: medium" in config_output
        assert scheduler.reset_count == 3

        reasoning_output = _run_with_scheduler(
            "/reasoning summary", handler=handler, scheduler=scheduler
        )
        reasoning_status = _run_with_scheduler("/reasoning", handler=handler, scheduler=scheduler)
        config_output = _run_with_scheduler("/config", handler=handler, scheduler=scheduler)

        assert "Reasoning display set to summary" in reasoning_output
        assert "agent_reloaded: True" in reasoning_output
        assert os.environ["KODER_REASONING_DISPLAY"] == "summary"
        assert reasoning_status == "Reasoning display: summary"
        assert "reasoning_display: summary" in config_output
        assert scheduler.reset_count == 4

        invalid_output = _run_with_scheduler(
            "/reasoning impossible", handler=handler, scheduler=scheduler
        )

        assert "Invalid argument: impossible" in invalid_output
        assert scheduler.reset_count == 4
    finally:
        reset_config_manager()


def test_effort_command_supports_xhigh_max_and_resets_to_medium(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KODER_REASONING_EFFORT", raising=False)
    reset_config_manager()
    try:
        scheduler = _ResettableScheduler(
            session=SimpleNamespace(session_id="extended-effort-session"),
            dev_agent=object(),
            _agent_initialized=True,
            reset_count=0,
        )
        handler = HarnessInteractiveCommandHandler(emit_console=False)

        assert (
            _run_with_scheduler("/effort", handler=handler, scheduler=scheduler)
            == "Current effort level: medium"
        )

        for effort in ("xhigh", "max"):
            output = _run_with_scheduler(f"/effort {effort}", handler=handler, scheduler=scheduler)
            status = _run_with_scheduler("/effort", handler=handler, scheduler=scheduler)
            config_output = _run_with_scheduler("/config", handler=handler, scheduler=scheduler)

            assert f"Set effort level to {effort}" in output
            assert os.environ["KODER_REASONING_EFFORT"] == effort
            assert status == f"Current effort level: {effort}"
            assert f"reasoning_effort: {effort}" in config_output

        auto_output = _run_with_scheduler("/effort auto", handler=handler, scheduler=scheduler)
        status = _run_with_scheduler("/effort", handler=handler, scheduler=scheduler)
        config_output = _run_with_scheduler("/config", handler=handler, scheduler=scheduler)

        assert "Effort level reset to default: medium" in auto_output
        assert "KODER_REASONING_EFFORT" not in os.environ
        assert status == "Current effort level: medium"
        assert "reasoning_effort: medium" in config_output
    finally:
        reset_config_manager()


def test_statusline_command_uses_setup_agent_for_natural_language_requests(monkeypatch):
    captured: dict[str, object] = {}

    async def _fake_run_sync(self, *, agent_definition, prompt, seed_items=None, cwd=None):
        captured["agent_type"] = agent_definition.agent_type
        captured["prompt"] = prompt
        captured["cwd"] = str(cwd)
        return "Configured ~/.koder/settings.json"

    monkeypatch.setattr(AgentService, "run_sync", _fake_run_sync)

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run("/statusline show model name and context percentage", handler=handler)

    assert output == "statusline: setup complete\nConfigured ~/.koder/settings.json"
    assert captured["agent_type"] == "statusline-setup"
    assert captured["prompt"] == "show model name and context percentage"


def test_btw_command_uses_session_context_without_mutating_history(
    tmp_path, monkeypatch, session_factory
):
    session = session_factory("btw-session", db_path=str(tmp_path / "koder.db"))
    asyncio.run(
        session.add_items(
            [
                {"role": "user", "content": "Investigate auth flow"},
                {"role": "assistant", "content": "Auth flow uses OAuth refresh tokens."},
            ]
        )
    )
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    captured_messages: list[dict] = []

    async def _fake_completion(messages, model=None, **kwargs):
        captured_messages.extend(messages)
        return "Auth note: OAuth refresh tokens."

    monkeypatch.setattr("koder_agent.harness.commands.interactive.llm_completion", _fake_completion)

    before_items = asyncio.run(session.get_items())
    result = _run_with_scheduler(
        "/btw what was the auth note?", handler=handler, scheduler=scheduler
    )
    after_items = asyncio.run(session.get_items())

    assert result == "Auth note: OAuth refresh tokens."
    assert before_items == after_items
    assert any(
        "Investigate auth flow" in str(message.get("content")) for message in captured_messages
    )


def test_add_dir_command_allows_file_reads_from_added_directory(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "notes.txt"
    outside_file.write_text("hello", encoding="utf-8")

    permission_service = PermissionService.default(workspace_root=workspace)
    handler = HarnessInteractiveCommandHandler(
        permission_service=permission_service,
        emit_console=False,
    )
    before = permission_service.evaluate_tool_call("read_file", {"file_path": str(outside_file)})
    assert before.requires_approval is True
    assert before.allowed is False

    add_output = _run(f"/add-dir {outside_dir}", handler=handler)
    assert f"Added {outside_dir.resolve()} as a working directory for this session" in add_output

    after = permission_service.evaluate_tool_call("read_file", {"file_path": str(outside_file)})
    assert after.allowed is True
    assert after.requires_approval is False


def test_add_dir_command_reports_missing_paths_before_parent_suggestion(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    missing_dir = tmp_path / "missing-dir"

    permission_service = PermissionService.default(workspace_root=workspace)
    handler = HarnessInteractiveCommandHandler(
        permission_service=permission_service,
        emit_console=False,
    )

    output = _run(f"/add-dir {missing_dir}", handler=handler)

    assert f"Path {missing_dir.resolve()} was not found." in output
    assert "Did you mean to add the parent directory" not in output


def test_compact_command_persists_runtime_compaction_summary(monkeypatch):
    class _Session:
        session_id = "compact-demo"
        summarization_threshold = None

        def __init__(self):
            self.items = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "latest question"},
                {"role": "assistant", "content": "latest answer"},
            ]

        async def get_items(self):
            return list(self.items)

        async def replace_items(self, items):
            self.items = list(items)

    async def fake_compact(messages):
        return CompactionResult(
            summary="Older messages summarized.",
            kept_messages=messages[-2:],
            token_count=12,
            original_count=len(messages),
        )

    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.llm_compact_messages",
        fake_compact,
    )

    session = _Session()
    scheduler = SimpleNamespace(session=session)
    handler = HarnessInteractiveCommandHandler()
    output = asyncio.run(handler.handle_slash_input("/compact", scheduler=scheduler))
    assert output.startswith("compacted, context size ")
    assert " -> " in output
    assert session.items[0]["content"].startswith("[Conversation compacted]")
    assert "Older messages summarized." in session.items[0]["content"]
    assert [item["content"] for item in session.items[1:]] == [
        "latest question",
        "latest answer",
    ]


def test_compact_command_pins_only_the_current_scheduler_todo(monkeypatch):
    class _Session:
        session_id = "compact-todo"

        def __init__(self):
            self.items = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "latest answer"},
            ]

        async def get_items(self):
            return list(self.items)

        async def replace_items(self, items):
            self.items = list(items)

    async def fake_compact(messages):
        return CompactionResult(
            summary="Older messages summarized.",
            kept_messages=messages[-1:],
            token_count=12,
            original_count=len(messages),
        )

    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.llm_compact_messages",
        fake_compact,
    )

    session = _Session()
    scheduler = SimpleNamespace(
        session=session,
        _active_todo_preserved_message=lambda: {
            "role": "user",
            "content": "Active Plan (pinned across compaction)\n  └ □ current scheduler task",
        },
    )
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    output = asyncio.run(handler.handle_slash_input("/compact", scheduler=scheduler))

    assert output.startswith("compacted, context size ")
    assert "current scheduler task" in session.items[1]["content"]
    assert session.items[-1]["content"] == "latest answer"


def test_compact_command_summarizes_response_items_instead_of_replaying_them(monkeypatch):
    class _Session:
        session_id = "compact-sdk-items"
        summarization_threshold = None

        def __init__(self):
            self.items = [
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "unknown", "content": ""},
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "read_file",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "contents"},
            ]

        async def get_items(self):
            return list(self.items)

        async def replace_items(self, items):
            self.items = list(items)

    async def fake_completion(_messages):
        return "Summary: old messages summarized."

    monkeypatch.setattr(
        "koder_agent.harness.memory.compact.llm_completion",
        fake_completion,
    )

    session = _Session()
    scheduler = SimpleNamespace(
        session=session,
        refreshed_items=None,
    )

    async def refresh_context_usage_from_session(items):
        scheduler.refreshed_items = list(items)
        return 123

    scheduler.refresh_context_usage_from_session = refresh_context_usage_from_session
    handler = HarnessInteractiveCommandHandler()

    output = asyncio.run(handler.handle_slash_input("/compact", scheduler=scheduler))

    assert output == "compacted, context size 123 -> 123"
    # Earlier turns are summarized into a plain-text head (user/assistant only),
    # but the most recent replayable tool round-trip (function_call + matching
    # function_call_output) is now preserved verbatim so the model can continue
    # an in-flight tool sequence after compaction (llm-compact tail
    # preservation). The unknown-role placeholder is dropped.
    plain_roles = {item.get("role") for item in session.items if "type" not in item}
    assert plain_roles == {"user", "assistant"}
    tool_items = [
        item
        for item in session.items
        if item.get("type") in {"function_call", "function_call_output"}
    ]
    assert [item["type"] for item in tool_items] == ["function_call", "function_call_output"]
    # The preserved pair stays matched by call_id (never a dangling half).
    assert {item["call_id"] for item in tool_items} == {"call-1"}
    assert scheduler.refreshed_items == session.items
    Converter.items_to_messages(session.items)


def test_compact_command_rejects_extra_arguments():
    handler = HarnessInteractiveCommandHandler()

    output = asyncio.run(handler.handle_slash_input("/compact now", scheduler=None))

    assert output == "Usage: /compact"


def test_session_rename_commands_use_scheduler_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    class _UnexpectedGlobalSession:
        def __init__(self, *args, **kwargs):
            raise AssertionError("scheduler commands must not open global session storage")

    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.EnhancedSQLiteSession",
        _UnexpectedGlobalSession,
    )

    class _Session:
        def __init__(self):
            self.session_id = "runtime-demo"
            self._title = "Demo Session"
            self._tag = "demo"

        async def get_display_name(self):
            return self._title

        async def set_title(self, title: str):
            self._title = title

        async def get_title(self):
            return self._title

        async def get_tag(self):
            return self._tag

        async def get_color(self):
            return "blue"

    scheduler = SimpleNamespace(session=_Session())
    handler = HarnessInteractiveCommandHandler()

    session_output = _run_with_scheduler("/session", handler=handler, scheduler=scheduler)
    rename_output = _run_with_scheduler(
        "/rename Better Title", handler=handler, scheduler=scheduler
    )

    assert "session_id: runtime-demo" in session_output
    assert "display_name: Demo Session" in session_output
    assert "tag: demo" in session_output
    assert "Session renamed to: Better Title" == rename_output


def test_export_command_summarizes_and_writes_session_content(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    handler = HarnessInteractiveCommandHandler()

    class _Session:
        session_id = "export-demo"

        async def get_display_name(self):
            return "Export Demo"

        async def get_items(self):
            return [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
                {"role": "user", "content": "please export this"},
            ]

    scheduler = SimpleNamespace(session=_Session())

    export_output = _run_with_scheduler("/export", handler=handler, scheduler=scheduler)
    json_output = _run_with_scheduler(
        "/export json export-demo.json", handler=handler, scheduler=scheduler
    )
    markdown_output = _run_with_scheduler(
        "/export markdown export-demo.md", handler=handler, scheduler=scheduler
    )
    inferred_output = _run_with_scheduler(
        "/export inferred-demo.json", handler=handler, scheduler=scheduler
    )
    missing_parent_output = _run_with_scheduler(
        "/export json missing/export.json", handler=handler, scheduler=scheduler
    )
    directory_output = _run_with_scheduler("/export .", handler=handler, scheduler=scheduler)

    assert "export session_id: export-demo" in export_output
    assert "display_name: Export Demo" in export_output
    assert "messages: 3" in export_output
    assert "### Transcript" in export_output
    assert "user: hello" in export_output
    assert "assistant: world" in export_output
    assert "export: written" in json_output
    assert "format: json" in json_output
    assert "messages: 3" in json_output
    assert "export: written" in markdown_output
    assert "format: markdown" in markdown_output
    assert "format: json" in inferred_output
    assert "export: parent directory not found" in missing_parent_output
    assert "export: target is a directory" in directory_output

    exported_json = json.loads((tmp_path / "export-demo.json").read_text(encoding="utf-8"))
    exported_inferred = json.loads((tmp_path / "inferred-demo.json").read_text(encoding="utf-8"))
    exported_markdown = (tmp_path / "export-demo.md").read_text(encoding="utf-8")
    assert exported_json == exported_inferred
    assert exported_json["session_id"] == "export-demo"
    assert exported_json["display_name"] == "Export Demo"
    assert exported_json["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
        {"role": "user", "content": "please export this"},
    ]
    assert "# Koder Session Export: Export Demo" in exported_markdown
    assert "session_id: export-demo" in exported_markdown
    assert "messages: 3" in exported_markdown
    assert "assistant: world" in exported_markdown


def test_files_command_lists_files_already_in_session_context(tmp_path):
    handler = HarnessInteractiveCommandHandler()

    tracked_one = tmp_path / "src" / "auth.py"
    tracked_two = tmp_path / "docs" / "guide.md"
    tracked_notebook = tmp_path / "notebooks" / "analysis.ipynb"
    missing_file = tmp_path / "missing.md"
    tracked_one.parent.mkdir(parents=True)
    tracked_two.parent.mkdir(parents=True)
    tracked_notebook.parent.mkdir(parents=True)
    tracked_one.write_text("print('auth')\n", encoding="utf-8")
    tracked_two.write_text("# guide\n", encoding="utf-8")
    tracked_notebook.write_text('{"cells": []}\n', encoding="utf-8")

    class _Session:
        session_id = "files-context-demo"

        async def get_items(self):
            return [
                {
                    "type": "function_call",
                    "name": "read_file",
                    "arguments": json.dumps({"path": str(tracked_one)}),
                },
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": json.dumps({"file_path": str(tracked_two)}),
                            }
                        }
                    ]
                },
                {
                    "type": "function_call",
                    "name": "read_file",
                    "arguments": json.dumps({"path": str(tracked_one)}),
                },
                {
                    "type": "function_call",
                    "name": "edit_file",
                    "arguments": json.dumps({"path": str(missing_file)}),
                },
                {
                    "type": "function_call",
                    "name": "notebook_edit",
                    "arguments": json.dumps({"notebook_path": str(tracked_notebook)}),
                },
            ]

    scheduler = SimpleNamespace(session=_Session())

    files_output = _run_with_scheduler("/files", handler=handler, scheduler=scheduler)

    assert files_output.startswith("Files in context:\n")
    assert "src/auth.py (exists)" in files_output
    assert "docs/guide.md (exists)" in files_output
    assert "notebooks/analysis.ipynb (exists)" in files_output
    assert "missing.md (missing)" in files_output
    assert files_output.count("src/auth.py") == 1


def test_context_consumers_ignore_invalid_persisted_tool_targets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    decoy = tmp_path / "safe-decoy.ipynb"
    actual = tmp_path / "outside-target.ipynb"
    decoy.write_text('{"cells": []}\n', encoding="utf-8")
    actual.write_text('{"cells": []}\n', encoding="utf-8")

    class _Session:
        async def get_items(self):
            return [
                {
                    "type": "function_call",
                    "name": "notebook_edit",
                    "arguments": json.dumps(
                        {
                            "path": str(decoy),
                            "notebook_path": str(actual),
                            "operation": "delete",
                        }
                    ),
                },
                {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "read_file",
                                "arguments": "{malformed",
                            }
                        },
                        {
                            "name": "write_file",
                            "arguments": json.dumps([str(actual)]),
                        },
                    ]
                },
            ]

    usage = SimpleNamespace(
        current_context_tokens=0,
        request_count=0,
        input_tokens=0,
        output_tokens=0,
        total_cost=0.0,
    )
    scheduler = SimpleNamespace(
        session=_Session(),
        usage_tracker=SimpleNamespace(session_usage=usage, model="gpt-4.1"),
    )
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    files_output = _run_with_scheduler("/files", handler=handler, scheduler=scheduler)
    context_output = _run_with_scheduler("/context", handler=handler, scheduler=scheduler)
    insights_output = _run_with_scheduler("/insights", handler=handler, scheduler=scheduler)

    assert files_output == "No files in context"
    assert "### Files in context" not in context_output
    assert "Files in context: 0" in insights_output
    for output in (files_output, context_output, insights_output):
        assert decoy.name not in output
        assert actual.name not in output


def test_diff_command_reports_git_and_conversation_edits(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("old line\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked.write_text("new line\nextra line\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    handler = HarnessInteractiveCommandHandler(emit_console=False)

    class _Session:
        session_id = "diff-session-demo"

        async def get_items(self):
            return [
                {"role": "user", "content": "Update auth implementation"},
                {
                    "toolUseResult": {
                        "filePath": "src/auth.py",
                        "structuredPatch": [
                            {
                                "oldStart": 1,
                                "oldLines": 1,
                                "newStart": 1,
                                "newLines": 2,
                                "lines": ["-old", "+new", "+extra"],
                            }
                        ],
                    }
                },
            ]

    scheduler = SimpleNamespace(session=_Session())

    diff_output = _run_with_scheduler("/diff", handler=handler, scheduler=scheduler)

    assert "## Diff" in diff_output
    assert "### Uncommitted changes" in diff_output
    assert "tracked.txt" in diff_output
    assert "### Conversation edits" in diff_output
    assert "Turn 1" in diff_output
    assert "src/auth.py" in diff_output


def test_commit_command_reports_staged_unstaged_untracked_and_clean_states(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("old line\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=repo, check=True)

    tracked.write_text("new line\n", encoding="utf-8")
    staged = repo / "staged.txt"
    staged.write_text("staged\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged.txt"], cwd=repo, check=True)
    (repo / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    monkeypatch.chdir(repo)

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run("/commit", handler=handler)

    assert "Branch:" in output
    assert "Staged changes:" in output
    assert "staged.txt" in output
    assert "Unstaged changes:" in output
    assert "tracked.txt" in output
    assert "1 untracked file(s):" in output
    assert "- untracked.txt" in output
    assert "Ready to commit." in output

    subprocess.run(["git", "add", "tracked.txt", "untracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "scenario-clean"], cwd=repo, check=True)

    clean_output = _run("/commit", handler=handler)
    assert "No staged changes." in clean_output
    assert "Nothing to commit, working tree clean." in clean_output
    assert "untracked file(s)" not in clean_output


def test_security_review_command_short_circuits_when_repo_has_no_pending_changes(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("seed\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.chdir(repo)

    handler = HarnessInteractiveCommandHandler(emit_console=False)

    output = _run("/security-review", handler=handler)

    assert output == "security-review: no pending changes to review."


def test_security_review_command_uses_prompt_backed_markdown_contract(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = repo / "src" / "auth.py"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("def login(user):\n    return user\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "src/auth.py"], cwd=repo, check=True, capture_output=True, text=True
    )
    subprocess.run(
        ["git", "commit", "-m", "seed"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked.write_text(
        "def login(user):\n    return f'<div>{user}</div>'\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(repo)

    captured_messages: list[dict] = []

    async def _fake_completion(messages, model=None, **kwargs):
        captured_messages.extend(messages)
        return (
            "# Vuln 1: XSS: `src/auth.py:2`\n\n"
            "* Severity: High\n"
            "* Description: User-controlled input is interpolated into HTML.\n"
            "* Exploit Scenario: An attacker can inject script content.\n"
            "* Recommendation: Escape user input before rendering."
        )

    monkeypatch.setattr(
        "koder_agent.harness.commands.security_review.llm_completion", _fake_completion
    )

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run("/security-review", handler=handler)

    assert output.startswith("# Vuln 1: XSS:")
    prompt_text = "\n".join(str(message.get("content", "")) for message in captured_messages)
    assert "GIT STATUS:" in prompt_text
    assert "FILES MODIFIED:" in prompt_text
    assert "src/auth.py" in prompt_text
    assert "DIFF CONTENT:" in prompt_text


def test_pr_comments_command_formats_review_threads_from_gh(monkeypatch):
    async def _fake_run_gh(*args: str, timeout: float = 5.0, cwd=None):
        if args[:3] == ("pr", "view", "--json"):
            return '{"number":123,"headRepository":{"name":"demo","owner":{"login":"octo"}}}'
        if args == ("api", "/repos/octo/demo/issues/123/comments"):
            return '[{"id":1,"user":{"login":"alice"},"body":"Top-level PR comment"}]'
        if args == ("api", "/repos/octo/demo/pulls/123/comments"):
            return (
                "["
                '{"id":10,"user":{"login":"bob"},"path":"src/app.py","line":42,'
                '"body":"Escape this value","diff_hunk":"@@ -1 +1 @@\\n-old\\n+new"},'
                '{"id":11,"user":{"login":"carol"},"in_reply_to_id":10,"body":"Agreed"}'
                "]"
            )
        raise AssertionError(f"unexpected gh args: {args}")

    monkeypatch.setattr("koder_agent.harness.commands.pr_comments._run_gh", _fake_run_gh)

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run("/pr_comments", handler=handler)

    assert output.startswith("## Comments")
    assert "- @alice PR conversation:" in output
    assert "- @bob src/app.py#42:" in output
    assert "```diff" in output
    assert "Escape this value" in output
    assert "- @carol:" in output


def test_issue_command_reports_gh_failure_instead_of_empty_state(monkeypatch):
    def _fake_run(args, capture_output=True, text=True, timeout=15):
        return subprocess.CompletedProcess(args, 2, "", "gh issue list failed")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run("/issue", handler=handler)

    assert output == "Failed to fetch issues: gh issue list failed"
    assert "No open issues" not in output


def test_subscribe_pr_command_reports_gh_failure_instead_of_empty_state(monkeypatch):
    def _fake_run(args, capture_output=True, text=True, timeout=10):
        return subprocess.CompletedProcess(args, 2, "", "gh pr list failed")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run("/subscribe-pr", handler=handler)

    assert output == "Failed to fetch PRs: gh pr list failed"
    assert "No open PRs" not in output


def test_oauth_refresh_command_reports_seeded_token_states(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    tokens = tmp_path / ".koder" / "tokens"
    tokens.mkdir(parents=True)
    (tokens / "google.json").write_text(
        json.dumps(
            {
                "provider": "google",
                "access_token": "google-access",
                "refresh_token": "google-refresh",
                "expires_at": 4102444800000,
                "email": "google@example.invalid",
            }
        ),
        encoding="utf-8",
    )
    (tokens / "claude.json").write_text(
        json.dumps(
            {
                "provider": "claude",
                "access_token": "claude-access",
                "refresh_token": "claude-refresh",
                "expires_at": 1,
                "email": "claude@example.invalid",
            }
        ),
        encoding="utf-8",
    )
    (tokens / "chatgpt.json").write_text("{not json", encoding="utf-8")

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    output = _run("/oauth-refresh", handler=handler)

    assert "oauth_refresh:" in output
    assert "- claude: expired" in output
    assert "email: claude@example.invalid" in output
    assert "- google: valid" in output
    assert "email: google@example.invalid" in output
    assert "refresh_command: koder auth login <provider>" in output
    assert "chatgpt" not in output
    assert _run("/oauth-refresh now", handler=handler) == "Usage: /oauth-refresh"


def test_peers_history_command_renders_discussion_events(tmp_path):
    from koder_agent.harness.agents.teams.service import TeamService

    team_service = TeamService.for_test(root=tmp_path)
    team_id = team_service.create_team("history-team")
    team_service.add_member(team_id, "agent-critic", name="critic", cwd=tmp_path)
    team_service.route(team_id, "please review", recipient="critic", sender="integrator")
    team_service.consume_next_mailbox_entry(team_id, recipient="critic")
    team_service.record_run(
        team_id,
        agent_id="agent-critic",
        member_name="critic",
        prompt="please review",
        output="CRITIC_GOT_DIRECT_PING",
        state="completed",
        source="mailbox",
    )
    handler = HarnessInteractiveCommandHandler(team_service=team_service, emit_console=False)

    output = _run(f"/peers history {team_id}", handler=handler)
    json_output = _run(f"/peers history {team_id} --json", handler=handler)

    assert "peers: history" in output
    assert "sent integrator -> critic: please review" in output
    assert "read critic <= integrator" in output
    assert "run critic state=completed source=mailbox: CRITIC_GOT_DIRECT_PING" in output
    parsed = json.loads(json_output)
    assert [entry["event"] for entry in parsed] == [
        "message_sent",
        "message_read",
        "run_completed",
    ]


def test_peers_mailbox_and_task_lifecycle_commands(tmp_path):
    from koder_agent.harness.agents.teams.service import TeamService

    team_service = TeamService.for_test(root=tmp_path)
    handler = HarnessInteractiveCommandHandler(team_service=team_service, emit_console=False)

    create_output = _run("/peers create task-team", handler=handler)
    send_output = _run("/peers send task-team worker-a team-message", handler=handler)
    consume_output = _run("/peers inbox task-team worker-a --consume", handler=handler)
    history_output = _run("/peers history task-team", handler=handler)
    create_task_output = _run("/peers task create task-team check mailbox", handler=handler)
    rejected_claim = _run("/peers task claim task-team 1 worker-a", handler=handler)
    assert "success: False" in rejected_claim
    assert "inactive_member" in rejected_claim
    team_service.add_member("task-team", "worker-a", name="worker-a")
    claim_output = _run("/peers task claim task-team 1 worker-a", handler=handler)
    task_list_output = _run("/peers task list task-team", handler=handler)
    complete_output = _run("/peers task complete task-team 1", handler=handler)
    completed_list_output = _run("/peers task list task-team", handler=handler)

    assert "team_id: task-team" in create_output
    assert "recipient: worker-a" in send_output
    assert "sender: team-lead" in send_output
    assert "peers: inbox consumed" in consume_output
    assert "sender: team-lead" in consume_output
    assert "read: true" in consume_output
    assert "message: team-message" in consume_output
    assert "sent team-lead -> worker-a: team-message" in history_output
    assert "read worker-a <= team-lead" in history_output
    assert "task_id: 1" in create_task_output
    assert "subject: check mailbox" in create_task_output
    assert "success: True" in claim_output
    assert "owner: worker-a" in claim_output
    assert "status: in_progress" in claim_output
    assert "- 1: check mailbox status=in_progress owner=worker-a" in task_list_output
    assert "status: completed" in complete_output
    assert "owner: worker-a" in complete_output
    assert "- 1: check mailbox status=completed owner=worker-a" in completed_list_output


def test_peers_send_can_attribute_message_to_named_sender(tmp_path):
    from koder_agent.harness.agents.teams.service import TeamService

    team_service = TeamService.for_test(root=tmp_path)
    handler = HarnessInteractiveCommandHandler(team_service=team_service, emit_console=False)

    _run("/peers create discussion-team", handler=handler)
    send_output = _run(
        "/peers send discussion-team team-lead --from proposer-a PROPOSAL_A",
        handler=handler,
    )
    inbox_output = _run(
        "/peers inbox discussion-team team-lead --consume",
        handler=handler,
    )
    history_output = _run("/peers history discussion-team", handler=handler)

    assert "recipient: team-lead" in send_output
    assert "sender: proposer-a" in send_output
    assert "sender: proposer-a" in inbox_output
    assert "message: PROPOSAL_A" in inbox_output
    assert "sent proposer-a -> team-lead: PROPOSAL_A" in history_output
    assert "read team-lead <= proposer-a" in history_output


def test_peers_spawn_auto_mode_uses_in_process_runner(tmp_path, monkeypatch):
    from koder_agent.harness.agents.teams.in_process import TeammateSpawnResult
    from koder_agent.harness.agents.teams.service import TeamService

    agent_service = AgentService.for_test(tmp_path)
    team_service = TeamService.for_test(root=tmp_path)
    config_service = SimpleNamespace(
        load=lambda: SimpleNamespace(harness=SimpleNamespace(teammate_mode="auto"))
    )
    handler = HarnessInteractiveCommandHandler(
        agent_service=agent_service,
        team_service=team_service,
        config_service=config_service,
        emit_console=False,
    )
    calls = []

    async def fail_launch_background(**_kwargs):
        raise AssertionError("auto mode should not use plain AgentService.launch_background")

    async def fake_spawn_teammate(
        *,
        team_id,
        name,
        agent_definition,
        prompt,
        cwd,
        plan_mode_required=False,
        model=None,
    ):
        calls.append(
            {
                "team_id": team_id,
                "name": name,
                "agent_type": agent_definition.agent_type,
                "prompt": prompt,
                "cwd": cwd,
                "plan_mode_required": plan_mode_required,
                "model": model,
            }
        )
        agent_id = agent_service.spawn(agent_definition.agent_type)
        team_service.add_member(
            team_id,
            agent_id,
            name=name,
            agent_type=agent_definition.agent_type,
            model=model,
            prompt=prompt,
            plan_mode_required=plan_mode_required,
            cwd=cwd,
            session_id=agent_service.get(agent_id).session_id,
            mode="default",
        )
        return TeammateSpawnResult(agent_id=agent_id, name=name, team_id=team_id)

    monkeypatch.setattr(agent_service, "launch_background", fail_launch_background)
    monkeypatch.setattr(handler.in_process_teammate_runner, "spawn_teammate", fake_spawn_teammate)

    create_output = _run("/peers create auto-team", handler=handler)
    spawn_output = _run(
        "/peers spawn auto-team general-purpose planner Draft plan", handler=handler
    )

    assert "teammate_mode: auto" in create_output
    assert "effective_teammate_mode: in-process" in create_output
    assert "peers: teammate spawned" in spawn_output
    assert "name: planner" in spawn_output
    assert len(calls) == 1
    assert calls[0] | {"model": "<ignored>"} == {
        "team_id": "auto-team",
        "name": "planner",
        "agent_type": "general-purpose",
        "prompt": "Draft plan",
        "cwd": Path.cwd(),
        "plan_mode_required": False,
        "model": "<ignored>",
    }


def test_peers_in_process_teammate_executes_local_slash_mailbox_work(tmp_path):
    from koder_agent.harness.agents.teams.service import TeamService

    agent_service = AgentService.for_test(tmp_path)
    team_service = TeamService.for_test(root=tmp_path)
    handler = HarnessInteractiveCommandHandler(
        agent_service=agent_service,
        team_service=team_service,
        teammate_mode="in-process",
        emit_console=False,
    )

    async def wait_for_history(
        team_id: str,
        marker: str,
        *,
        sender: str | None = None,
        recipient: str | None = None,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + 3
        while asyncio.get_running_loop().time() < deadline:
            history = team_service.history_entries(team_id)
            if any(
                marker in (entry.content or "")
                and (sender is None or entry.sender == sender)
                and (recipient is None or entry.recipient == recipient)
                for entry in history
            ):
                return
            await asyncio.sleep(0.05)
        raise AssertionError(f"timed out waiting for {marker}")

    async def run_case():
        create_output = await handler.handle_slash_input(
            "/peers create inproc-team",
            scheduler=None,
        )
        spawn_output = await handler.handle_slash_input(
            "/peers spawn inproc-team general-purpose worker-a /version",
            scheduler=None,
        )
        member = team_service.member_records("inproc-team")[0]
        await handler.in_process_teammate_runner.wait(member.agent_id)

        send_output = await handler.handle_slash_input(
            "/peers send inproc-team worker-a /peers send inproc-team team-lead PROPOSAL_A",
            scheduler=None,
        )
        await wait_for_history(
            "inproc-team",
            "PROPOSAL_A",
            sender="worker-a",
            recipient="team-lead",
        )
        inbox_output = await handler.handle_slash_input(
            "/peers inbox inproc-team team-lead",
            scheduler=None,
        )
        history_output = await handler.handle_slash_input(
            "/peers history inproc-team",
            scheduler=None,
        )

        assert "effective_teammate_mode: in-process" in create_output
        assert "peers: teammate spawned" in spawn_output
        assert "name: worker-a" in spawn_output
        assert "sender: team-lead" in send_output
        assert "PROPOSAL_A" in inbox_output
        assert (
            "sent team-lead -> worker-a: /peers send inproc-team team-lead PROPOSAL_A"
            in history_output
        )
        assert "read worker-a <= team-lead" in history_output
        assert "sent worker-a -> team-lead: PROPOSAL_A" in history_output
        assert "run worker-a state=completed source=mailbox" in history_output

        await handler.in_process_teammate_runner.terminate(member.agent_id)

    asyncio.run(run_case())


def test_permissions_check_reports_sandbox_and_tool_decisions(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    handler = HarnessInteractiveCommandHandler(emit_console=False)

    enable_output = _run("/sandbox enable unix-local", handler=handler)
    blocked_output = _run("/permissions check run_shell touch blocked.txt", handler=handler)
    exclude_output = _run("/sandbox exclude touch *", handler=handler)
    excluded_output = _run("/permissions check run_shell touch blocked.txt", handler=handler)
    read_output = _run("/permissions check run_shell rg TODO .", handler=handler)

    assert "sandbox: enabled" in enable_output
    assert "permissions: check" in blocked_output
    assert "tool: run_shell" in blocked_output
    assert "allowed: false" in blocked_output
    assert "requires_approval: true" in blocked_output
    assert "sandboxed shell command auto-allowed" not in blocked_output
    assert "sandbox: excluded command added" in exclude_output
    assert "allowed: false" in excluded_output
    assert "requires_approval: true" in excluded_output
    assert "mutate filesystem" in excluded_output
    assert "allowed: true" in read_output
    assert "requires_approval: false" in read_output


def test_peers_spawn_tmux_mode_registers_pane_member(tmp_path, monkeypatch):
    from koder_agent.harness.agents.teams.service import TeamService

    class FakeBackend:
        def spawn_member(self, *, name, prompt, cwd, model=None, env=None):
            assert name == "pane-worker"
            assert prompt == "/model"
            assert cwd == str(Path.cwd())
            assert isinstance(model, str) and model
            assert env is not None
            return SimpleNamespace(pane_id="%42")

    monkeypatch.setattr(
        "koder_agent.harness.agents.teams.runtime.create_backend",
        lambda mode, team_id: FakeBackend(),
    )

    def fake_tmux_display(cmd, **_kwargs):
        if cmd[:4] == ["tmux", "list-panes", "-a", "-F"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="%42\n", stderr="")
        assert cmd[:4] == ["tmux", "display-message", "-p", "-t"]
        return subprocess.CompletedProcess(cmd, 0, stdout="0\n", stderr="")

    monkeypatch.setattr(
        "koder_agent.harness.commands.interactive.subprocess.run",
        fake_tmux_display,
    )
    team_service = TeamService.for_test(root=tmp_path)
    handler = HarnessInteractiveCommandHandler(
        team_service=team_service,
        teammate_mode="tmux",
        emit_console=False,
    )

    create_output = _run("/peers create tmux-team", handler=handler)
    spawn_output = _run(
        "/peers spawn tmux-team general-purpose pane-worker /model", handler=handler
    )
    show_output = _run("/peers show tmux-team", handler=handler)

    assert "teammate_mode: tmux" in create_output
    assert "effective_teammate_mode: tmux" in create_output
    assert "peers: teammate spawned in tmux" in spawn_output
    assert "agent_id: %42" in spawn_output
    assert "pane_id: %42" in spawn_output
    assert "member_count: 1" in show_output
    assert (
        "- member %42: name=pane-worker type=general-purpose mode=tmux active=True "
        "pane_state=running" in show_output
    )


def test_fork_command_launches_background_agent(tmp_path):
    service = AgentService.for_test(tmp_path)
    seen_seed_items = []

    async def fake_execute(*, agent_definition, prompt, session_id, seed_items, cwd, **_kwargs):
        seen_seed_items.append(seed_items)
        return "forked result"

    class _Session:
        async def get_items(self):
            return [{"role": "user", "content": "hello"}]

    scheduler = SimpleNamespace(session=_Session())
    handler = HarnessInteractiveCommandHandler(
        agent_service=service,
    )

    import koder_agent.harness.agents.service as agent_service_module

    original = agent_service_module._execute_agent_run
    agent_service_module._execute_agent_run = fake_execute
    try:
        output = asyncio.run(handler.handle_slash_input("/fork Investigate auth flow", scheduler))
        forked_agent_id = output.split("forked_agent_id: ", 1)[1].splitlines()[0]
        asyncio.run(service.wait(forked_agent_id))
        fork_context_output = asyncio.run(
            handler.handle_slash_input("/fork --context fork Investigate auth flow", scheduler)
        )
        fork_context_agent_id = fork_context_output.split("forked_agent_id: ", 1)[1].splitlines()[0]
        asyncio.run(service.wait(fork_context_agent_id))
    finally:
        agent_service_module._execute_agent_run = original

    assert "forked_agent_id:" in output
    assert "agent_type: general-purpose" in output
    assert "context_mode: isolated" in output
    assert "status: background" in output
    assert "output_file:" in output
    output_path = output.split("output_file: ", 1)[1].splitlines()[0]
    assert Path(output_path).exists()
    assert "context_mode: fork" in fork_context_output
    assert seen_seed_items == [None, [{"role": "user", "content": "hello"}]]


def test_fork_command_respects_main_agent_spawn_allowlist(tmp_path):
    service = AgentService.for_test(tmp_path)
    coordinator = AgentDefinition(
        agent_type="coordinator",
        when_to_use="Coordinates work",
        system_prompt="You are a coordinator.",
        source="flagSettings",
        tools=["Agent(worker)", "Read"],
        allowed_agent_types=["worker"],
    )
    worker = AgentDefinition(
        agent_type="worker",
        when_to_use="Does work",
        system_prompt="You are a worker.",
        source="flagSettings",
    )
    reviewer = AgentDefinition(
        agent_type="reviewer",
        when_to_use="Reviews work",
        system_prompt="You are a reviewer.",
        source="flagSettings",
    )

    class _Session:
        async def get_items(self):
            return [{"role": "user", "content": "hello"}]

    scheduler = SimpleNamespace(
        session=_Session(),
        agent_definition=coordinator,
        agent_definitions=SimpleNamespace(active_agents=[coordinator, worker, reviewer]),
    )
    handler = HarnessInteractiveCommandHandler(agent_service=service)

    import koder_agent.harness.agents.service as agent_service_module

    async def fake_execute(*, agent_definition, prompt, session_id, seed_items, cwd, **_kwargs):
        return "forked allowlist result"

    original = agent_service_module._execute_agent_run
    agent_service_module._execute_agent_run = fake_execute
    try:
        blocked = asyncio.run(handler.handle_slash_input("/fork reviewer Inspect auth", scheduler))
        allowed = asyncio.run(handler.handle_slash_input("/fork worker Inspect auth", scheduler))
        assert "not allowed" in blocked
        assert "forked_agent_id:" in allowed
        agent_id = allowed.split("forked_agent_id: ", 1)[1].splitlines()[0]
        asyncio.run(service.wait(agent_id))
    finally:
        agent_service_module._execute_agent_run = original


def test_fork_command_can_resume_background_agent(tmp_path):
    service = AgentService.for_test(tmp_path)
    definition = next(
        agent
        for agent in get_agent_definitions(cwd=tmp_path).active_agents
        if agent.agent_type == "general-purpose"
    )

    import koder_agent.harness.agents.service as agent_service_module

    async def fake_execute(*, agent_definition, prompt, session_id, seed_items, cwd, **_kwargs):
        return f"result for {prompt}"

    original = agent_service_module._execute_agent_run
    agent_service_module._execute_agent_run = fake_execute
    try:
        first = asyncio.run(
            service.launch_background(
                agent_definition=definition,
                prompt="first task",
                description="First task",
                cwd=tmp_path,
            )
        )
        asyncio.run(service.wait(first.id))

        scheduler = SimpleNamespace(
            session=SimpleNamespace(get_items=lambda: []),
            agent_definitions=SimpleNamespace(active_agents=[definition]),
        )
        handler = HarnessInteractiveCommandHandler(agent_service=service)
        output = asyncio.run(
            handler.handle_slash_input(f"/fork --resume {first.id} continue task", scheduler)
        )
    finally:
        agent_service_module._execute_agent_run = original

    assert "fork: resumed background subagent" in output
    assert f"forked_agent_id: {first.id}" in output


def test_fork_command_preserves_plan_permission_mode_on_resume(tmp_path):
    service = AgentService.for_test(tmp_path)
    project = tmp_path / "project"
    (project / ".koder" / "agents").mkdir(parents=True)
    (project / ".koder" / "agents" / "planner.md").write_text(
        "---\nname: planner\ndescription: Plans work\npermissionMode: plan\n---\n"
        "You are a planner.\n",
        encoding="utf-8",
    )
    definition = next(
        agent
        for agent in get_agent_definitions(cwd=project).active_agents
        if agent.agent_type == "planner"
    )

    import koder_agent.harness.agents.service as agent_service_module
    from koder_agent.tools.plan_mode import _get_plan_service

    async def fake_execute(*, agent_definition, prompt, session_id, seed_items, cwd, **_kwargs):
        return f"mode={_get_plan_service().mode}; prompt={prompt}"

    original = agent_service_module._execute_agent_run
    agent_service_module._execute_agent_run = fake_execute
    try:
        first = asyncio.run(
            service.launch_background(
                agent_definition=definition,
                prompt="draft plan",
                description="Draft plan",
                permission_mode="plan",
                cwd=project,
            )
        )
        asyncio.run(service.wait(first.id))

        scheduler = SimpleNamespace(
            session=SimpleNamespace(get_items=lambda: []),
            agent_definitions=SimpleNamespace(active_agents=[definition]),
        )
        handler = HarnessInteractiveCommandHandler(agent_service=service)
        output = asyncio.run(
            handler.handle_slash_input(f"/fork --resume {first.id} continue planning", scheduler)
        )
        asyncio.run(service.wait(first.id))
    finally:
        agent_service_module._execute_agent_run = original

    assert "permission_mode: plan" in output
    assert (
        Path(first.output_path).read_text(encoding="utf-8") == "mode=plan; prompt=continue planning"
    )


def test_fork_resume_uses_origin_project_definition_after_main_cwd_changes(tmp_path, monkeypatch):
    service = AgentService.for_test(tmp_path)
    origin = tmp_path / "origin"
    other = tmp_path / "other"
    for project, body in ((origin, "Origin worker."), (other, "Other worker.")):
        (project / ".koder" / "agents").mkdir(parents=True)
        (project / ".koder" / "agents" / "worker.md").write_text(
            "---\nname: worker\ndescription: Does work\npermissionMode: plan\n---\n" + body + "\n",
            encoding="utf-8",
        )
    origin_definition = next(
        agent
        for agent in get_agent_definitions(cwd=origin).active_agents
        if agent.agent_type == "worker"
    )
    seen: list[tuple[str, str, str]] = []

    import koder_agent.harness.agents.service as agent_service_module

    async def fake_execute(*, agent_definition, prompt, session_id, seed_items, cwd, **_kwargs):
        seen.append((agent_definition.system_prompt, prompt, cwd))
        return f"result for {prompt}"

    original_execute = agent_service_module._execute_agent_run
    agent_service_module._execute_agent_run = fake_execute
    try:
        first = asyncio.run(
            service.launch_background(
                agent_definition=origin_definition,
                prompt="first task",
                description="Origin task",
                cwd=origin,
                permission_mode="plan",
            )
        )
        asyncio.run(service.wait(first.id))
        monkeypatch.chdir(other)
        other_definition = next(
            agent
            for agent in get_agent_definitions(cwd=other).active_agents
            if agent.agent_type == "worker"
        )
        scheduler = SimpleNamespace(
            session=SimpleNamespace(get_items=lambda: []),
            agent_definitions=SimpleNamespace(active_agents=[other_definition]),
        )
        handler = HarnessInteractiveCommandHandler(agent_service=service)
        output = asyncio.run(
            handler.handle_slash_input(f"/fork --resume {first.id} continue task", scheduler)
        )
        asyncio.run(service.wait(first.id))
    finally:
        agent_service_module._execute_agent_run = original_execute

    assert "fork: resumed background subagent" in output
    assert "permission_mode: plan" in output
    assert seen[-1] == ("Origin worker.", "continue task", str(origin.resolve()))


def test_fork_resume_rejects_missing_origin_definition_even_with_same_name_elsewhere(
    tmp_path, monkeypatch
):
    service = AgentService.for_test(tmp_path)
    origin = tmp_path / "origin"
    other = tmp_path / "other"
    for project, body in ((origin, "Origin worker."), (other, "Other worker.")):
        (project / ".koder" / "agents").mkdir(parents=True)
        (project / ".koder" / "agents" / "worker.md").write_text(
            "---\nname: worker\ndescription: Does work\n---\n" + body + "\n",
            encoding="utf-8",
        )
    origin_definition = next(
        agent
        for agent in get_agent_definitions(cwd=origin).active_agents
        if agent.agent_type == "worker"
    )

    async def fake_execute(**_kwargs):
        return "done"

    import koder_agent.harness.agents.service as agent_service_module

    original_execute = agent_service_module._execute_agent_run
    agent_service_module._execute_agent_run = fake_execute
    try:
        first = asyncio.run(
            service.launch_background(
                agent_definition=origin_definition,
                prompt="first task",
                description="Origin task",
                cwd=origin,
            )
        )
        asyncio.run(service.wait(first.id))
        (origin / ".koder" / "agents" / "worker.md").unlink()
        monkeypatch.chdir(other)
        handler = HarnessInteractiveCommandHandler(agent_service=service)
        output = asyncio.run(
            handler.handle_slash_input(
                f"/fork --resume {first.id} continue task",
                SimpleNamespace(agent_definitions=get_agent_definitions(cwd=other)),
            )
        )
    finally:
        agent_service_module._execute_agent_run = original_execute

    assert "original agent definition is missing or changed" in output
    assert service.get(first.id).state == "completed"
