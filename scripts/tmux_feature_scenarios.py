#!/usr/bin/env python3
"""Validate and run scenario-based tmux coverage for Koder TUI features."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
import tempfile
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "tests" / "e2e" / "tui_feature_scenarios.json"
CLI_STARTUP_TIMEOUT_SECONDS = 60.0
VALIDATION_LEVELS = {"smoke", "workflow", "acceptance"}
RAW_HEX_PATTERN = re.compile(r"^(?:[0-9a-fA-F]{2})(?:\s+[0-9a-fA-F]{2})*$")
TURN_ASSERTION_KEYS = (
    "expect_any",
    "expect_all",
    "expect_regex",
    "expect_not",
    "expect_bottom_all",
    "expect_session_dead",
    "expect_tmux_panes_min",
    "expect_tmux_any_pane_any",
    "expect_tmux_any_pane_all",
)
POST_ASSERTION_KEYS = {
    "file_contains",
    "file_glob_contains",
    "file_glob_not_contains",
    "file_not_contains",
    "file_occurrences",
    "path_exists",
    "path_glob_exists",
    "path_not_exists",
    "sqlite_contains",
}


@dataclass(frozen=True)
class ScenarioRef:
    suite: str
    name: str
    payload: dict[str, Any]


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, check=True, timeout=30)


def _tmux(
    *args: str, check: bool = False, timeout: float = 30.0, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    tmux = shutil.which("tmux")
    if tmux is None:
        raise RuntimeError("tmux is not available")
    return subprocess.run(
        [tmux, *args], text=True, capture_output=True, check=check, timeout=timeout, env=env
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _harness_command_names() -> set[str]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from koder_agent.harness.commands.interactive import HarnessInteractiveCommandHandler

    handler = HarnessInteractiveCommandHandler(emit_console=False)
    return set(handler.commands)


def _scenario_refs(manifest: dict[str, Any]) -> list[ScenarioRef]:
    refs: list[ScenarioRef] = []
    for suite_name in ("slash_commands", "agents", "teams", "skills", "features"):
        suite = manifest.get(suite_name, {})
        if not isinstance(suite, dict):
            continue
        for name, payload in suite.items():
            if isinstance(payload, dict):
                refs.append(ScenarioRef(suite=suite_name, name=name, payload=payload))
    return refs


def _validation_level(payload: dict[str, Any]) -> str:
    level = payload.get("validation_level", "smoke")
    return level if isinstance(level, str) else ""


def _non_empty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _validate_acceptance_metadata(ref: ScenarioRef) -> list[str]:
    if _validation_level(ref.payload) != "acceptance":
        return []

    errors: list[str] = []
    criteria = ref.payload.get("acceptance_criteria")
    artifacts = ref.payload.get("acceptance_artifacts")
    turns = ref.payload.get("turns", [])
    if not _non_empty_string_list(criteria):
        errors.append(f"{ref.suite}/{ref.name}: acceptance_criteria is required")
    if not _non_empty_string_list(artifacts):
        errors.append(f"{ref.suite}/{ref.name}: acceptance_artifacts is required")
    has_exact_visible_assertion = any(
        isinstance(turn, dict) and (turn.get("expect_all") or turn.get("expect_regex"))
        for turn in turns
    )
    if not has_exact_visible_assertion:
        errors.append(f"{ref.suite}/{ref.name}: acceptance needs exact visible assertions")
    has_durable_or_external_assertion = bool(ref.payload.get("post_assertions")) or any(
        isinstance(turn, dict)
        and (
            turn.get("expect_session_dead")
            or turn.get("expect_tmux_panes_min")
            or turn.get("capture") == "visible"
        )
        for turn in turns
    )
    if not has_durable_or_external_assertion:
        errors.append(
            f"{ref.suite}/{ref.name}: acceptance needs durable, external, or pane evidence"
        )
    return errors


def _validate_post_assertions(ref: ScenarioRef) -> list[str]:
    post_assertions = ref.payload.get("post_assertions", [])
    if post_assertions is None:
        return []
    if not isinstance(post_assertions, list):
        return [f"{ref.suite}/{ref.name}: post_assertions must be a list"]

    errors: list[str] = []
    for index, assertion in enumerate(post_assertions, start=1):
        if not isinstance(assertion, dict):
            errors.append(f"{ref.suite}/{ref.name}: post_assertion {index} must be an object")
            continue
        keys = [key for key in assertion if key in POST_ASSERTION_KEYS]
        if len(keys) != 1:
            errors.append(
                f"{ref.suite}/{ref.name}: post_assertion {index} needs exactly one of "
                + ", ".join(sorted(POST_ASSERTION_KEYS))
            )
            continue
        value = assertion[keys[0]]
        if keys[0] in {"path_exists", "path_not_exists", "path_glob_exists"}:
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"{ref.suite}/{ref.name}: post_assertion {index} {keys[0]} needs a path"
                )
        elif keys[0] == "file_occurrences":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], str)
                or not value[0].strip()
                or not isinstance(value[1], dict)
                or not value[1]
                or not all(
                    isinstance(text, str)
                    and bool(text.strip())
                    and isinstance(count, int)
                    and not isinstance(count, bool)
                    and count >= 0
                    for text, count in value[1].items()
                )
            ):
                errors.append(
                    f"{ref.suite}/{ref.name}: post_assertion {index} file_occurrences "
                    "needs [path, {text: exact-nonnegative-count}]"
                )
        elif keys[0] == "sqlite_contains":
            if (
                not isinstance(value, list)
                or len(value) != 3
                or not isinstance(value[0], str)
                or not value[0].strip()
                or not isinstance(value[1], str)
                or not value[1].strip()
                or not (
                    isinstance(value[2], str)
                    or (
                        isinstance(value[2], list)
                        and value[2]
                        and all(isinstance(item, str) for item in value[2])
                    )
                )
            ):
                errors.append(
                    f"{ref.suite}/{ref.name}: post_assertion {index} sqlite_contains needs "
                    "[path, select-query, text-or-text-list]"
                )
            elif not value[1].lstrip().lower().startswith("select"):
                errors.append(
                    f"{ref.suite}/{ref.name}: post_assertion {index} sqlite_contains query "
                    "must be SELECT-only"
                )
        else:
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(value[0], str)
                or not value[0].strip()
                or not (
                    isinstance(value[1], str)
                    or (
                        isinstance(value[1], list)
                        and value[1]
                        and all(isinstance(item, str) for item in value[1])
                    )
                )
            ):
                errors.append(
                    f"{ref.suite}/{ref.name}: post_assertion {index} {keys[0]} needs "
                    "[path, text-or-text-list]"
                )
    return errors


def validate_manifest(manifest: dict[str, Any], *, strict_acceptance: bool = False) -> list[str]:
    errors: list[str] = []
    command_names = _harness_command_names()
    slash_suite = manifest.get("slash_commands", {})
    if not isinstance(slash_suite, dict):
        return ["slash_commands must be an object"]

    scenario_commands = set(slash_suite)
    missing = sorted(command_names - scenario_commands)
    extra = sorted(scenario_commands - command_names)
    if missing:
        errors.append("missing slash command scenarios: " + ", ".join(missing))
    if extra:
        errors.append("unknown slash command scenarios: " + ", ".join(extra))

    required_suites = {"agents", "teams", "skills", "features"}
    for suite in required_suites:
        if not manifest.get(suite):
            errors.append(f"missing scenario suite: {suite}")

    for ref in _scenario_refs(manifest):
        payload = ref.payload
        level = payload.get("validation_level", "smoke")
        if not isinstance(level, str) or level not in VALIDATION_LEVELS:
            errors.append(
                f"{ref.suite}/{ref.name}: validation_level must be one of "
                + ", ".join(sorted(VALIDATION_LEVELS))
            )
        purpose = payload.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            errors.append(f"{ref.suite}/{ref.name}: purpose is required")
        env = payload.get("env", {})
        if env and (
            not isinstance(env, dict)
            or not all(
                isinstance(key, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                and isinstance(value, str)
                for key, value in env.items()
            )
        ):
            errors.append(f"{ref.suite}/{ref.name}: env must map env var names to strings")
        cli_args = payload.get("cli_args", [])
        if cli_args and (
            not isinstance(cli_args, list)
            or not all(isinstance(item, str) and item.strip() for item in cli_args)
        ):
            errors.append(f"{ref.suite}/{ref.name}: cli_args must contain strings")
        prelaunch_files = payload.get("prelaunch_files", [])
        if prelaunch_files and not isinstance(prelaunch_files, list):
            errors.append(f"{ref.suite}/{ref.name}: prelaunch_files must be a list")
        for index, item in enumerate(prelaunch_files, start=1):
            if not isinstance(item, dict):
                errors.append(f"{ref.suite}/{ref.name}: prelaunch_file {index} must be an object")
                continue
            if not isinstance(item.get("path"), str) or not item["path"].strip():
                errors.append(f"{ref.suite}/{ref.name}: prelaunch_file {index} needs a path")
            if "source" in item:
                try:
                    _prelaunch_source(item["source"])
                    if "content" in item:
                        raise ValueError("choose source or content, not both")
                except (TypeError, ValueError) as exc:
                    errors.append(f"{ref.suite}/{ref.name}: prelaunch_file {index}: {exc}")
            elif not isinstance(item.get("content"), str):
                errors.append(f"{ref.suite}/{ref.name}: prelaunch_file {index} needs content")
        teammate_mode = payload.get("teammate_mode", "tmux")
        if teammate_mode not in {"auto", "in-process", "tmux"}:
            errors.append(
                f"{ref.suite}/{ref.name}: teammate_mode must be auto, in-process, or tmux"
            )
        fake_openai = payload.get("fake_openai")
        if fake_openai is not None:
            if not isinstance(fake_openai, dict):
                errors.append(f"{ref.suite}/{ref.name}: fake_openai must be an object")
            else:
                port = fake_openai.get("port")
                if not isinstance(port, int) or isinstance(port, bool):
                    errors.append(f"{ref.suite}/{ref.name}: fake_openai.port must be an integer")
                elif not 0 <= port <= 65535:
                    errors.append(
                        f"{ref.suite}/{ref.name}: fake_openai.port must be from 0 through 65535"
                    )
                if (
                    not isinstance(fake_openai.get("response"), str)
                    or not fake_openai["response"].strip()
                ):
                    errors.append(f"{ref.suite}/{ref.name}: fake_openai.response is required")
                for key in ("log_file", "ready_file"):
                    if not isinstance(fake_openai.get(key), str) or not fake_openai[key].strip():
                        errors.append(f"{ref.suite}/{ref.name}: fake_openai.{key} is required")
                scenario_name = fake_openai.get("scenario")
                if scenario_name is not None and scenario_name not in {
                    "single",
                    "git_query_mutation",
                    "sed_query_mutation",
                    "sandbox_shell_tool",
                    "streaming_subagent_tool",
                    "streaming_tool_error",
                    "streaming_tool_queue",
                }:
                    errors.append(
                        f"{ref.suite}/{ref.name}: fake_openai.scenario must be single "
                        "or a supported streaming tool scenario"
                    )
                for option in ("stream_delay", "subagent_delay"):
                    option_value = fake_openai.get(option)
                    if option_value is not None and not isinstance(option_value, (int, float)):
                        errors.append(
                            f"{ref.suite}/{ref.name}: fake_openai.{option} must be a number"
                        )
                stream_lines = fake_openai.get("stream_lines")
                if stream_lines is not None and (
                    not isinstance(stream_lines, int)
                    or isinstance(stream_lines, bool)
                    or stream_lines <= 0
                ):
                    errors.append(
                        f"{ref.suite}/{ref.name}: fake_openai.stream_lines must be a "
                        "positive integer"
                    )
        if (
            isinstance(env, dict)
            and any(
                isinstance(value, str) and "$FAKE_OPENAI_URL" in value for value in env.values()
            )
            and not isinstance(fake_openai, dict)
        ):
            errors.append(f"{ref.suite}/{ref.name}: $FAKE_OPENAI_URL requires fake_openai")
        turns = payload.get("turns")
        if not isinstance(turns, list) or len(turns) < 2:
            errors.append(f"{ref.suite}/{ref.name}: at least two interactive turns are required")
            continue
        if ref.suite == "slash_commands" and not any(
            isinstance(turn, dict)
            and isinstance(turn.get("send"), str)
            and turn["send"].split()[0] in {f"/{ref.name}", f"/{ref.name.replace('_', '-')}"}
            for turn in turns
        ):
            errors.append(f"{ref.suite}/{ref.name}: no turn sends /{ref.name}")
        for index, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict):
                errors.append(f"{ref.suite}/{ref.name}: turn {index} must be an object")
                continue
            has_send = isinstance(turn.get("send"), str) and bool(turn["send"].strip())
            has_type = isinstance(turn.get("type"), str) and bool(turn["type"].strip())
            has_keys = isinstance(turn.get("keys"), list) and bool(turn["keys"])
            raw_hex = turn.get("raw_hex")
            has_raw_hex = isinstance(raw_hex, str) and bool(
                RAW_HEX_PATTERN.fullmatch(raw_hex.strip())
            )
            has_wait = isinstance(turn.get("wait"), (int, float))
            has_resize = isinstance(turn.get("resize"), dict)
            has_kill_tmux_pane = isinstance(turn.get("kill_tmux_pane_matching"), str) and bool(
                turn["kill_tmux_pane_matching"].strip()
            )
            if not any(
                [
                    has_send,
                    has_type,
                    has_keys,
                    has_raw_hex,
                    has_wait,
                    has_resize,
                    has_kill_tmux_pane,
                    turn.get("expect_session_dead"),
                ]
            ):
                errors.append(
                    f"{ref.suite}/{ref.name}: turn {index} needs an action or session-dead check"
                )
            if "raw_hex" in turn and not has_raw_hex:
                errors.append(
                    f"{ref.suite}/{ref.name}: turn {index} raw_hex needs space-separated "
                    "two-digit hexadecimal octets"
                )
            if has_resize:
                resize = turn["resize"]
                if not isinstance(resize.get("width"), int) or not isinstance(
                    resize.get("height"), int
                ):
                    errors.append(
                        f"{ref.suite}/{ref.name}: turn {index} resize needs integer width/height"
                    )
            if "kill_tmux_pane_matching" in turn and not has_kill_tmux_pane:
                errors.append(
                    f"{ref.suite}/{ref.name}: turn {index} kill_tmux_pane_matching needs text"
                )
            expect_any = turn.get("expect_any", [])
            expect_all = turn.get("expect_all", [])
            expect_regex = turn.get("expect_regex", [])
            expect_not = turn.get("expect_not", [])
            expect_bottom_all = turn.get("expect_bottom_all", [])
            for field_name, field_value in (
                ("expect_any", expect_any),
                ("expect_all", expect_all),
                ("expect_regex", expect_regex),
                ("expect_not", expect_not),
                ("expect_bottom_all", expect_bottom_all),
            ):
                if field_value and (
                    not isinstance(field_value, list)
                    or not all(isinstance(item, str) for item in field_value)
                ):
                    errors.append(
                        f"{ref.suite}/{ref.name}: turn {index} {field_name} must contain strings"
                    )
            for pattern in expect_regex:
                if isinstance(pattern, str):
                    try:
                        re.compile(pattern)
                    except re.error as exc:
                        errors.append(
                            f"{ref.suite}/{ref.name}: turn {index} invalid expect_regex "
                            f"{pattern!r}: {exc}"
                        )
            has_tmux_pane_assertion = any(
                key in turn
                for key in (
                    "expect_tmux_panes_min",
                    "expect_tmux_any_pane_any",
                    "expect_tmux_any_pane_all",
                )
            )
            if (
                not turn.get("expect_session_dead")
                and not expect_any
                and not expect_all
                and not expect_regex
                and not expect_not
                and not expect_bottom_all
                and not has_tmux_pane_assertion
            ):
                errors.append(f"{ref.suite}/{ref.name}: turn {index} needs an assertion")
        if strict_acceptance:
            errors.extend(_validate_acceptance_metadata(ref))
        errors.extend(_validate_post_assertions(ref))
    return errors


def _expand_scenario_path(value: str, *, home: Path, repo: Path) -> Path:
    expanded = value.replace("$HOME", str(home)).replace("$REPO", str(repo))
    return Path(expanded).expanduser()


def _expand_scenario_text(value: str, *, home: Path, repo: Path) -> str:
    from koder_agent.harness.version_info import resolve_runtime_version

    return (
        value.replace("$HOME", str(home))
        .replace("$REPO", str(repo))
        .replace("$RUNTIME_VERSION", resolve_runtime_version())
        .replace(
            "$RUNTIME_PYTHON_PATTERN",
            re.escape(str(Path(sysconfig.get_path("scripts")) / "python"))
            + r"(?:3(?:\.\d+)?)?"
            + (r"\.exe" if os.name == "nt" else "")
            + r"(?=\s|$)",
        )
        .replace("$RUNTIME_PYTHON_RESOLVED", str(Path(sys.executable).resolve()))
        .replace("$RUNTIME_PYTHON", sys.executable)
        .replace(
            "$RUNTIME_CLI",
            str(
                Path(sysconfig.get_path("scripts")) / ("koder.exe" if os.name == "nt" else "koder")
            ),
        )
    )


def _expand_scenario_glob(value: str, *, home: Path, repo: Path) -> list[Path]:
    expanded = value.replace("$HOME", str(home)).replace("$REPO", str(repo))
    return [Path(path) for path in sorted(glob.glob(str(Path(expanded).expanduser())))]


def _expand_scenario_env_value(
    value: str,
    *,
    home: Path,
    repo: Path,
    fake_openai_url: str | None = None,
) -> str:
    expanded = (
        value.replace("$HOME", str(home))
        .replace("$REPO", str(repo))
        .replace("$PATH", os.environ.get("PATH", ""))
    )
    if "$FAKE_OPENAI_URL" not in expanded:
        return expanded
    if fake_openai_url is None:
        raise RuntimeError("$FAKE_OPENAI_URL used without a fake OpenAI provider")
    return expanded.replace("$FAKE_OPENAI_URL", fake_openai_url)


def _expected_strings(value: str | list[str], *, home: Path, repo: Path) -> list[str]:
    strings = [value] if isinstance(value, str) else value
    return [_expand_scenario_text(item, home=home, repo=repo) for item in strings]


def _run_post_assertions(scenario: ScenarioRef, *, home: Path, repo: Path) -> list[str]:
    failures: list[str] = []
    for index, assertion in enumerate(scenario.payload.get("post_assertions", []), start=1):
        if "path_exists" in assertion:
            path = _expand_scenario_path(assertion["path_exists"], home=home, repo=repo)
            if not path.exists():
                failures.append(f"post_assertion {index}: expected path to exist: {path}")
            continue
        if "path_not_exists" in assertion:
            path = _expand_scenario_path(assertion["path_not_exists"], home=home, repo=repo)
            if path.exists():
                failures.append(f"post_assertion {index}: expected path not to exist: {path}")
            continue
        if "file_contains" in assertion:
            raw_path, expected = assertion["file_contains"]
            path = _expand_scenario_path(raw_path, home=home, repo=repo)
            if not path.exists():
                failures.append(f"post_assertion {index}: expected file to exist: {path}")
                continue
            content = path.read_text(encoding="utf-8")
            missing = [
                item
                for item in _expected_strings(expected, home=home, repo=repo)
                if item not in content
            ]
            if missing:
                failures.append(
                    f"post_assertion {index}: {path} missing "
                    + ", ".join(repr(item) for item in missing)
                )
            continue
        if "file_occurrences" in assertion:
            raw_path, expected = assertion["file_occurrences"]
            path = _expand_scenario_path(raw_path, home=home, repo=repo)
            if not path.exists():
                failures.append(f"post_assertion {index}: expected file to exist: {path}")
                continue
            content = path.read_text(encoding="utf-8")
            mismatches = []
            for item, expected_count in expected.items():
                expanded_item = _expand_scenario_text(item, home=home, repo=repo)
                actual_count = content.count(expanded_item)
                if actual_count != expected_count:
                    mismatches.append(
                        f"{expanded_item!r}: expected {expected_count}, found {actual_count}"
                    )
            if mismatches:
                failures.append(
                    f"post_assertion {index}: {path} occurrence mismatch: " + "; ".join(mismatches)
                )
            continue
        if "path_glob_exists" in assertion:
            matches = _expand_scenario_glob(assertion["path_glob_exists"], home=home, repo=repo)
            if not matches:
                failures.append(
                    f"post_assertion {index}: expected glob to match: "
                    f"{assertion['path_glob_exists']}"
                )
            continue
        if "file_glob_contains" in assertion:
            raw_pattern, expected = assertion["file_glob_contains"]
            matches = _expand_scenario_glob(raw_pattern, home=home, repo=repo)
            if not matches:
                failures.append(f"post_assertion {index}: expected glob to match: {raw_pattern}")
                continue
            missing = []
            for item in _expected_strings(expected, home=home, repo=repo):
                if not any(item in path.read_text(encoding="utf-8") for path in matches):
                    missing.append(item)
            if missing:
                failures.append(
                    f"post_assertion {index}: {raw_pattern} missing "
                    + ", ".join(repr(item) for item in missing)
                )
            continue
        if "file_glob_not_contains" in assertion:
            raw_pattern, forbidden = assertion["file_glob_not_contains"]
            matches = _expand_scenario_glob(raw_pattern, home=home, repo=repo)
            present = []
            for item in _expected_strings(forbidden, home=home, repo=repo):
                if any(item in path.read_text(encoding="utf-8") for path in matches):
                    present.append(item)
            if present:
                failures.append(
                    f"post_assertion {index}: {raw_pattern} unexpectedly contains "
                    + ", ".join(repr(item) for item in present)
                )
            continue
        if "file_not_contains" in assertion:
            raw_path, forbidden = assertion["file_not_contains"]
            path = _expand_scenario_path(raw_path, home=home, repo=repo)
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            present = [
                item
                for item in _expected_strings(forbidden, home=home, repo=repo)
                if item in content
            ]
            if present:
                failures.append(
                    f"post_assertion {index}: {path} unexpectedly contains "
                    + ", ".join(repr(item) for item in present)
                )
            continue
        if "sqlite_contains" in assertion:
            raw_path, query, expected = assertion["sqlite_contains"]
            path = _expand_scenario_path(raw_path, home=home, repo=repo)
            if not path.exists():
                failures.append(f"post_assertion {index}: expected database to exist: {path}")
                continue
            try:
                with closing(sqlite3.connect(path)) as conn, conn:
                    rows = conn.execute(query).fetchall()
            except sqlite3.Error as exc:
                failures.append(f"post_assertion {index}: sqlite query failed for {path}: {exc}")
                continue
            content = "\n".join(
                "\t".join("" if value is None else str(value) for value in row) for row in rows
            )
            missing = [
                item
                for item in _expected_strings(expected, home=home, repo=repo)
                if item not in content
            ]
            if missing:
                failures.append(
                    f"post_assertion {index}: sqlite result from {path} missing "
                    + ", ".join(repr(item) for item in missing)
                )
    return failures


def _prepare_workspace(root: Path) -> tuple[Path, Path]:
    home = root / "home"
    repo = root / "repo"
    home.mkdir(parents=True, exist_ok=True)
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "AGENTS.md").write_text(
        "# Test project\n\nUse deterministic local outputs.\n", encoding="utf-8"
    )
    (repo / "docs").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "runtime-notes.md").write_text(
        "# MAGIC DOC: Runtime Notes\n\nKeep this fixture current.\n", encoding="utf-8"
    )
    (repo / "sample.txt").write_text("initial\n", encoding="utf-8")
    (repo / ".koder" / "skills" / "demo-skill").mkdir(parents=True, exist_ok=True)
    (repo / ".koder" / "skills" / "demo-skill" / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demo skill for tmux validation\ndisable-model-invocation: true\n---\nUse deterministic local output.\n",
        encoding="utf-8",
    )
    (repo / ".koder" / "agents").mkdir(parents=True, exist_ok=True)
    (repo / ".koder" / "agents" / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews fixture changes\ntools:\n  - Read\n  - Bash\nmodel: sonnet\npermissionMode: plan\n---\nYou review fixture changes.\n",
        encoding="utf-8",
    )
    plugin = home / ".koder" / "plugins" / "demo-plugin"
    (plugin / "skills" / "plugin-skill").mkdir(parents=True, exist_ok=True)
    (plugin / "plugin.json").write_text(
        json.dumps({"name": "demo-plugin", "version": "1.0.0"}), encoding="utf-8"
    )
    (plugin / "skills" / "plugin-skill" / "SKILL.md").write_text(
        "---\nname: plugin-skill\ndescription: Plugin skill fixture\ndisable-model-invocation: true\n---\nPlugin skill body.\n",
        encoding="utf-8",
    )
    _run(["git", "init"], cwd=repo)
    _run(["git", "config", "user.email", "koder@example.invalid"], cwd=repo)
    _run(["git", "config", "user.name", "Koder Test"], cwd=repo)
    _run(["git", "add", "sample.txt"], cwd=repo)
    _run(["git", "commit", "-m", "initial"], cwd=repo)
    (repo / "sample.txt").write_text("initial\nchanged\n", encoding="utf-8")
    return home, repo


def _prelaunch_source(source: str) -> Path:
    """Only repository fixture paths may seed the disposable workspace."""
    if not isinstance(source, str) or not source or Path(source).is_absolute():
        raise ValueError("source must be a repository-relative fixture path")
    path = (PROJECT_ROOT / source).resolve()
    if not path.is_relative_to(PROJECT_ROOT / "tests/fixtures") or not path.is_file():
        raise ValueError("source must be an existing file inside tests/fixtures")
    return path


def _write_prelaunch_files(scenario: ScenarioRef, *, home: Path, repo: Path) -> None:
    for item in scenario.payload.get("prelaunch_files", []):
        path = _expand_scenario_path(item["path"], home=home, repo=repo)
        path.parent.mkdir(parents=True, exist_ok=True)
        if "source" in item:
            shutil.copyfile(_prelaunch_source(item["source"]), path)
        else:
            path.write_text(item["content"], encoding="utf-8")


def _start_fake_openai(
    scenario: ScenarioRef, *, home: Path, repo: Path
) -> tuple[subprocess.Popen | None, str | None]:
    fake_openai = scenario.payload.get("fake_openai")
    if not isinstance(fake_openai, dict):
        return None, None

    log_file = _expand_scenario_path(fake_openai["log_file"], home=home, repo=repo)
    ready_file = _expand_scenario_path(fake_openai["ready_file"], home=home, repo=repo)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ready_file.parent.mkdir(parents=True, exist_ok=True)
    if ready_file.exists():
        ready_file.unlink()

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "fake_openai_chat_server.py"),
        "--port",
        str(fake_openai["port"]),
        "--response",
        fake_openai["response"],
        "--log-file",
        str(log_file),
        "--ready-file",
        str(ready_file),
    ]
    scenario_name = fake_openai.get("scenario")
    if scenario_name:
        command.extend(["--scenario", str(scenario_name)])
    optional_args = {
        "stream_delay": "--stream-delay",
        "subagent_delay": "--subagent-delay",
        "stream_lines": "--stream-lines",
    }
    for option, flag in optional_args.items():
        value = fake_openai.get(option)
        if value is not None:
            command.extend([flag, str(value)])

    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if ready_file.exists():
                ready_text = ready_file.read_text(encoding="utf-8").strip()
                if ready_text.startswith("ready http"):
                    return proc, ready_text.removeprefix("ready ")
            if proc.poll() is not None:
                raise RuntimeError(
                    f"fake OpenAI provider exited for {scenario.suite}/{scenario.name}"
                )
            time.sleep(0.1)
        raise RuntimeError(
            f"fake OpenAI provider did not become ready for {scenario.suite}/{scenario.name}"
        )
    except BaseException:
        # The caller cannot own this child until startup returns successfully.
        _stop_fake_openai(proc)
        raise


def _stop_fake_openai(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _copy_fake_openai_log(
    scenario: ScenarioRef, *, home: Path, repo: Path, output_dir: Path
) -> None:
    fake_openai = scenario.payload.get("fake_openai")
    if not isinstance(fake_openai, dict):
        return
    source = _expand_scenario_path(fake_openai["log_file"], home=home, repo=repo)
    if not source.exists():
        return
    target = output_dir / f"{scenario.suite}-{scenario.name}-fake-openai.log"
    shutil.copyfile(source, target)


def _launch_session(
    home: Path,
    repo: Path,
    scenario: ScenarioRef,
    *,
    fake_openai_url: str | None,
) -> str:
    session = f"koder-scenario-{scenario.suite[:3]}-{scenario.name[:18]}-{uuid.uuid4().hex[:6]}"
    import_paths = [
        *os.environ.get("PYTHONPATH", "").split(os.pathsep),
        str(PROJECT_ROOT),
    ]
    env_assignments = {
        "HOME": str(home),
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(filter(None, import_paths))),
        "KODER_SCENARIO_SOURCE_ROOT": str(PROJECT_ROOT),
        "KODER_MODEL": "gpt-4.1",
    }
    if sys.prefix != sys.base_prefix:
        # tmux may have a long-lived server with a stale environment. Pass the
        # test interpreter explicitly so isolated homes cannot select/rebuild .venv.
        env_assignments["UV_PROJECT_ENVIRONMENT"] = sys.prefix
        env_assignments["UV_PYTHON"] = sys.executable
    env_assignments.update(
        {
            key: _expand_scenario_env_value(
                value,
                home=home,
                repo=repo,
                fake_openai_url=fake_openai_url,
            )
            for key, value in scenario.payload.get("env", {}).items()
        }
    )
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not available")
    environment_args = [
        argument for key, value in env_assignments.items() for argument in ("-e", f"{key}={value}")
    ]
    teammate_mode = scenario.payload.get("teammate_mode", "tmux")
    # Multiple arguments make tmux exec uv directly. A shell command string can
    # source personal startup files before HOME is applied and reinterpret args.
    launch = [
        uv,
        "--project",
        str(PROJECT_ROOT),
        "run",
        "--no-sync",
        "--no-env-file",
        "koder",
        "--teammate-mode",
        teammate_mode,
        *scenario.payload.get("cli_args", []),
    ]
    result = _tmux(
        "new-session",
        "-d",
        "-s",
        session,
        "-x",
        "160",
        "-y",
        "48",
        "-c",
        str(repo),
        *environment_args,
        *launch,
        check=False,
        timeout=20,
        # tmux uses an unattached new-session client's PATH when spawning.
        # Match the requested pane environment rather than losing fixture bins.
        env={**os.environ, **env_assignments},
    )
    if result.returncode != 0:
        # tmux may echo environment assignments in stderr; keep only safe status.
        raise subprocess.CalledProcessError(result.returncode, ["tmux", "new-session"])
    try:
        # Cold SDK imports can be delayed on a busy test host. Readiness remains
        # mandatory; a larger finite startup budget is not a passing assertion.
        startup = _wait_for_prompt(session, timeout=CLI_STARTUP_TIMEOUT_SECONDS)
        if not _session_exists(session):
            raise RuntimeError(
                f"koder session exited before prompt for {scenario.suite}/{scenario.name}"
            )
        if "| ⚡ Koder |" not in startup or "│>" not in startup:
            raise RuntimeError(
                f"koder prompt did not appear for {scenario.suite}/{scenario.name} "
                f"within {CLI_STARTUP_TIMEOUT_SECONDS:g}s"
            )
    except BaseException:
        # run_scenario cannot own this session until this function returns.
        _tmux("kill-session", "-t", session, timeout=5)
        raise
    return session


def _capture(session: str) -> str:
    return _tmux("capture-pane", "-p", "-S", "-500", "-t", session, timeout=10).stdout


def _capture_visible(session: str) -> str:
    return _tmux("capture-pane", "-p", "-t", session, timeout=10).stdout


def _capture_for_turn(session: str, turn: dict[str, Any]) -> str:
    if turn.get("capture") == "visible":
        return _capture_visible(session)
    return _capture(session)


def _bottom_assertions_pass(capture: str, expected: list[str], *, window: int) -> bool:
    if not expected:
        return True
    tail = "\n".join(capture.splitlines()[-max(1, window) :])
    return all(item in tail for item in expected)


def _list_panes(session: str) -> list[str]:
    result = _tmux("list-panes", "-t", session, "-F", "#{pane_id}", timeout=10)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _capture_pane(pane_id: str) -> str:
    return _tmux("capture-pane", "-p", "-S", "-300", "-t", pane_id, timeout=10).stdout


def _capture_all_panes(session: str) -> dict[str, str]:
    return {pane_id: _capture_pane(pane_id) for pane_id in _list_panes(session)}


def _kill_tmux_pane_matching(session: str, marker: str) -> str | None:
    panes = _list_panes(session)
    pane_outputs = _capture_all_panes(session)
    for pane_id in panes[1:]:
        if marker in pane_outputs.get(pane_id, ""):
            result = _tmux("kill-pane", "-t", pane_id, timeout=10)
            if result.returncode != 0:
                return result.stderr.strip() or f"failed to kill {pane_id}"
            return None
    return f"no non-leader pane matched {marker!r}"


def _wait_for_prompt(session: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if not _session_exists(session):
            return last
        last = _capture(session)
        if "| ⚡ Koder |" in last and "│>" in last:
            return last
        time.sleep(0.5)
    return last


def _send(session: str, text: str) -> None:
    # Prompt-toolkit enables this mode when its input application is active.
    # A painted frame alone can still belong to the command that just finished.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = _tmux("display-message", "-p", "-t", session, "#{bracket_paste_flag}", timeout=5)
        if state.returncode == 0 and state.stdout.strip() == "1":
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("terminal input reader did not become ready")

    # Deliver long/multiline commands as one input event, not hundreds of
    # independent keystrokes racing completion callbacks and redraws.
    buffer_name = f"scenario-input-{uuid.uuid4().hex}"
    _tmux("set-buffer", "-b", buffer_name, "--", text, check=True, timeout=10)
    try:
        _tmux(
            "paste-buffer",
            "-p",
            "-d",
            "-b",
            buffer_name,
            "-t",
            session,
            check=True,
            timeout=10,
        )
    except BaseException:
        _tmux("delete-buffer", "-b", buffer_name, timeout=5)
        raise
    _tmux("send-keys", "-t", session, "Enter", timeout=10)


def _type_text(session: str, text: str) -> None:
    _tmux("send-keys", "-t", session, "-l", text, timeout=10)
    time.sleep(0.3)


def _send_key_sequence(session: str, keys: list[str]) -> None:
    for key in keys:
        _tmux("send-keys", "-t", session, key, timeout=10)
        time.sleep(0.2)


def _send_raw_hex(session: str, value: str) -> None:
    """Send one uninterrupted sequence of hexadecimal terminal bytes."""
    _tmux("send-keys", "-t", session, "-H", *value.split(), timeout=10)
    time.sleep(0.2)


def _dispatch_turn_input_actions(session: str, turn: dict[str, Any]) -> None:
    """Dispatch text and key actions in deterministic terminal-input order."""
    if turn.get("send"):
        _send(session, turn["send"])
    if turn.get("type"):
        _type_text(session, turn["type"])
    if turn.get("raw_hex"):
        _send_raw_hex(session, turn["raw_hex"])
    if turn.get("keys"):
        _send_key_sequence(session, turn["keys"])


def _turn_assertion_timeout(turn: dict[str, Any]) -> float:
    if "timeout" in turn:
        return float(turn["timeout"])
    command = turn.get("send")
    # Shell fixtures may start a fresh interpreter and SDK. Keep explicit
    # streaming/input timing contracts unchanged.
    return 45.0 if isinstance(command, str) and command.startswith("!") else 12.0


def _resize_window(session: str, *, width: int, height: int) -> None:
    _tmux("resize-window", "-t", session, "-x", str(width), "-y", str(height), timeout=10)
    time.sleep(0.6)


def _session_exists(session: str) -> bool:
    return _tmux("has-session", "-t", session, timeout=5).returncode == 0


def _tmux_pane_assertions_pass(session: str, turn: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    expected_min = turn.get("expect_tmux_panes_min")
    expect_any = turn.get("expect_tmux_any_pane_any", [])
    expect_all = turn.get("expect_tmux_any_pane_all", [])
    if expected_min is None and not expect_any and not expect_all:
        return True, {}

    pane_outputs = _capture_all_panes(session)
    if expected_min is not None and len(pane_outputs) < int(expected_min):
        return False, pane_outputs
    if expect_any and not any(
        any(item in output for item in expect_any) for output in pane_outputs.values()
    ):
        return False, pane_outputs
    if expect_all and not any(
        all(item in output for item in expect_all) for output in pane_outputs.values()
    ):
        return False, pane_outputs
    return True, pane_outputs


def _wait_for_assertions(
    session: str,
    turn: dict[str, Any],
    *,
    home: Path,
    repo: Path,
    timeout: float,
) -> tuple[bool, str]:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        exists = _session_exists(session)
        if turn.get("expect_session_dead"):
            if not exists:
                return True, last
        elif exists:
            last = _capture_for_turn(session, turn)
            assertion_text = _without_prompt_input(last) if turn.get("send") else last
            expect_all = _expected_strings(turn.get("expect_all", []), home=home, repo=repo)
            expect_any = _expected_strings(turn.get("expect_any", []), home=home, repo=repo)
            expect_regex = _expected_strings(turn.get("expect_regex", []), home=home, repo=repo)
            expect_not = _expected_strings(turn.get("expect_not", []), home=home, repo=repo)
            expect_bottom_all = _expected_strings(
                turn.get("expect_bottom_all", []), home=home, repo=repo
            )
            all_ok = all(item in assertion_text for item in expect_all)
            any_ok = True if not expect_any else any(item in assertion_text for item in expect_any)
            regex_ok = all(re.search(pattern, assertion_text) for pattern in expect_regex)
            not_ok = all(item not in last for item in expect_not)
            bottom_ok = _bottom_assertions_pass(
                last,
                expect_bottom_all,
                window=int(turn.get("expect_bottom_window", 6)),
            )
            pane_ok, _pane_outputs = _tmux_pane_assertions_pass(session, turn)
            if all_ok and any_ok and regex_ok and not_ok and bottom_ok and pane_ok:
                return True, last
        time.sleep(0.5)
    if _session_exists(session):
        last = _capture_for_turn(session, turn)
    return False, last


def _without_prompt_input(capture: str) -> str:
    """Do not accept a command's echoed source as evidence that it executed.

    Keep frame markers for layout assertions. Negative/privacy checks and
    explicit input/key assertions still inspect the original capture.
    """
    lines = []
    in_input = False
    for line in capture.splitlines():
        if line.startswith("┌") and "| ⚡ Koder |" in line:
            in_input = True
            lines.append(line)
        elif in_input and line.startswith("└"):
            in_input = False
            lines.append(line)
        elif in_input:
            if line.startswith("│>"):
                lines.append("│>")
        else:
            lines.append(line)
    return "\n".join(lines)


def _bind_fake_provider_url(scenario: ScenarioRef, url: str | None) -> ScenarioRef:
    if url is None:
        return scenario

    def bind(value):
        if isinstance(value, str):
            return value.replace("$FAKE_OPENAI_URL", url)
        if isinstance(value, list):
            return [bind(item) for item in value]
        if isinstance(value, dict):
            return {key: bind(item) for key, item in value.items()}
        return value

    return ScenarioRef(scenario.suite, scenario.name, bind(scenario.payload))


def _scenario_error_text(error: Exception) -> str:
    """Keep subprocess command arguments and captured output out of receipts."""
    if isinstance(error, subprocess.TimeoutExpired):
        detail = f"subprocess timed out after {error.timeout} seconds"
    elif isinstance(error, subprocess.CalledProcessError):
        detail = f"subprocess exited with status {error.returncode}"
    else:
        detail = str(error)
    return f"{type(error).__name__}: {detail}"


def run_scenario(scenario: ScenarioRef, *, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    error_path = output_dir / f"{scenario.suite}-{scenario.name}-error.txt"
    errors: list[str] = []
    ok = True

    def record_error(phase: str, error: Exception) -> None:
        nonlocal ok
        ok = False
        message = f"{phase}: {_scenario_error_text(error)}"
        errors.append(message)
        print(f"FAIL {scenario.suite}/{scenario.name}: {message}", file=sys.stderr)

    workspace = None
    phase = "workspace_setup"
    try:
        workspace = tempfile.TemporaryDirectory(prefix="koder-scenario-")
        home: Path | None = None
        repo: Path | None = None
        fake_openai_proc: subprocess.Popen | None = None
        fake_openai_url: str | None = None
        session: str | None = None
        try:
            home, repo = _prepare_workspace(Path(workspace.name))
            phase = "prelaunch"
            _write_prelaunch_files(scenario, home=home, repo=repo)
            phase = "provider_startup"
            fake_openai_proc, fake_openai_url = _start_fake_openai(scenario, home=home, repo=repo)
            scenario = _bind_fake_provider_url(scenario, fake_openai_url)
            phase = "cli_startup"
            session = _launch_session(
                home,
                repo,
                scenario,
                fake_openai_url=fake_openai_url,
            )
            for index, turn in enumerate(scenario.payload["turns"], start=1):
                phase = f"turn_{index}"
                _dispatch_turn_input_actions(session, turn)
                if turn.get("resize"):
                    resize = turn["resize"]
                    _resize_window(
                        session,
                        width=int(resize["width"]),
                        height=int(resize["height"]),
                    )
                if turn.get("kill_tmux_pane_matching"):
                    kill_error = _kill_tmux_pane_matching(session, turn["kill_tmux_pane_matching"])
                    if kill_error:
                        print(
                            f"FAIL {scenario.suite}/{scenario.name} turn {index}: {kill_error}",
                            file=sys.stderr,
                        )
                        ok = False
                        break
                if turn.get("wait"):
                    time.sleep(float(turn["wait"]))
                passed, output = _wait_for_assertions(
                    session,
                    turn,
                    home=home,
                    repo=repo,
                    timeout=_turn_assertion_timeout(turn),
                )
                capture = output_dir / f"{scenario.suite}-{scenario.name}-turn-{index}.txt"
                capture.write_text(output, encoding="utf-8")
                if any(
                    key in turn
                    for key in (
                        "expect_tmux_panes_min",
                        "expect_tmux_any_pane_any",
                        "expect_tmux_any_pane_all",
                    )
                ):
                    pane_capture = (
                        output_dir / f"{scenario.suite}-{scenario.name}-turn-{index}-panes.txt"
                    )
                    pane_capture.write_text(
                        "\n\n".join(
                            f"## {pane_id}\n{content}"
                            for pane_id, content in _capture_all_panes(session).items()
                        ),
                        encoding="utf-8",
                    )
                if not passed:
                    print(
                        f"FAIL {scenario.suite}/{scenario.name} turn {index}: {turn}",
                        file=sys.stderr,
                    )
                    ok = False
                    break
                time.sleep(0.8)
            if ok:
                phase = "post_assertions"
                post_failures = _run_post_assertions(scenario, home=home, repo=repo)
                if post_failures:
                    post_capture = output_dir / f"{scenario.suite}-{scenario.name}-post.txt"
                    post_capture.write_text("\n".join(post_failures), encoding="utf-8")
                    for failure in post_failures:
                        print(f"FAIL {scenario.suite}/{scenario.name}: {failure}", file=sys.stderr)
                    ok = False
        except Exception as error:
            record_error(phase, error)
        finally:
            # Each resource gets its cleanup attempt even if another one fails.
            cleanup_actions = []
            if session is not None:
                cleanup_actions.append(
                    ("tmux_cleanup", partial(_tmux, "kill-session", "-t", session, timeout=5))
                )
            if fake_openai_proc is not None:
                cleanup_actions.append(
                    ("provider_cleanup", partial(_stop_fake_openai, fake_openai_proc))
                )
            if home is not None and repo is not None:
                cleanup_actions.append(
                    (
                        "provider_log",
                        partial(
                            _copy_fake_openai_log,
                            scenario,
                            home=home,
                            repo=repo,
                            output_dir=output_dir,
                        ),
                    )
                )
            for cleanup_phase, cleanup in cleanup_actions:
                try:
                    cleanup()
                except Exception as error:
                    record_error(cleanup_phase, error)
    except Exception as error:
        record_error(phase, error)
    finally:
        if workspace is not None:
            try:
                workspace.cleanup()
            except Exception as error:
                record_error("workspace_cleanup", error)
    if errors:
        error_path.write_text("\n".join(errors) + "\n", encoding="utf-8")
    else:
        error_path.unlink(missing_ok=True)
    return ok


def select_scenarios(
    manifest: dict[str, Any], selectors: list[str], run_all: bool
) -> list[ScenarioRef]:
    refs = _scenario_refs(manifest)
    if run_all:
        return refs
    if not selectors:
        return []
    selected: list[ScenarioRef] = []
    for selector in selectors:
        for ref in refs:
            if selector in {ref.name, f"{ref.suite}/{ref.name}"}:
                selected.append(ref)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate or run tmux feature scenarios")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--check", action="store_true", help="Validate manifest coverage")
    parser.add_argument(
        "--strict-acceptance",
        action="store_true",
        help="Require acceptance scenarios to include acceptance metadata and stronger evidence",
    )
    parser.add_argument("--list", action="store_true", help="List scenario names")
    parser.add_argument(
        "--run", action="append", default=[], help="Run a scenario by name or suite/name"
    )
    parser.add_argument("--run-all", action="store_true", help="Run every scenario")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "koder-feature-scenarios",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    errors = validate_manifest(manifest, strict_acceptance=args.strict_acceptance)
    if args.check or not (args.list or args.run or args.run_all):
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print("scenario manifest covers all runtime slash commands and required feature suites")
        if args.check and not (args.list or args.run or args.run_all):
            return 0
    if args.list:
        for ref in _scenario_refs(manifest):
            print(
                f"{ref.suite}/{ref.name} [{_validation_level(ref.payload)}]: "
                f"{ref.payload.get('purpose', '')}"
            )
    selected = select_scenarios(manifest, args.run, args.run_all)
    if args.run or args.run_all:
        if errors:
            print("manifest validation failed; refusing to run scenarios", file=sys.stderr)
            return 1
        if not selected:
            print("no scenarios selected", file=sys.stderr)
            return 1
        if shutil.which("tmux") is None:
            print("tmux is not available", file=sys.stderr)
            return 2
        all_ok = True
        for scenario in selected:
            print(f"RUN {scenario.suite}/{scenario.name}")
            all_ok = run_scenario(scenario, output_dir=args.output_dir) and all_ok
        return 0 if all_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
