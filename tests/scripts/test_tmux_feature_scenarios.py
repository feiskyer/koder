from __future__ import annotations

import copy
import json
import sqlite3
from contextlib import closing

from scripts import tmux_feature_scenarios as scenarios
from scripts.fake_openai_chat_server import _Handler as FakeOpenAIHandler
from scripts.tmux_feature_scenarios import (
    DEFAULT_MANIFEST,
    VALIDATION_LEVELS,
    _bottom_assertions_pass,
    _load_manifest,
    validate_manifest,
)

IDLE_FIRST = (
    "idle-wrap-alpha-012345678901234567890123456789012345678901234567890123456789-idle-wrap-omega"
)
IDLE_SECOND = "idle-second-line"
QUEUED_FIRST = (
    "queued-wrap-alpha-012345678901234567890123456789012345678901234567890123456789-"
    "queued-wrap-omega"
)
QUEUED_SECOND = "queued-second-line"


def test_tui_feature_scenario_manifest_covers_all_runtime_commands():
    manifest = _load_manifest(DEFAULT_MANIFEST)

    errors = validate_manifest(manifest)

    assert errors == []


def test_bottom_assertions_only_scan_the_visible_capture_tail():
    capture = "old output\nmid prompt | ⚡ Koder |\n" + "\n".join(
        ["line 1", "line 2", "line 3", "line 4", "line 5", "status tail"]
    )

    assert not _bottom_assertions_pass(capture, ["| ⚡ Koder |"], window=3)
    assert _bottom_assertions_pass(capture, ["status tail"], window=3)


def test_raw_hex_turn_action_is_validated():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    turn = manifest["features"]["fixed-bottom-idle-tip"]["turns"][0]
    turn.pop("send", None)
    turn["raw_hex"] = "1b 5b 31 33 3b 32 75"

    assert validate_manifest(manifest) == []

    for invalid in ("", "1", "1b 5", 13):
        invalid_manifest = copy.deepcopy(manifest)
        invalid_manifest["features"]["fixed-bottom-idle-tip"]["turns"][0]["raw_hex"] = invalid
        assert any("raw_hex" in error for error in validate_manifest(invalid_manifest))


def test_send_raw_hex_uses_one_tmux_hex_call(monkeypatch):
    calls = []

    def fake_tmux(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(scenarios, "_tmux", fake_tmux)
    monkeypatch.setattr(scenarios.time, "sleep", lambda _seconds: None)

    scenarios._send_raw_hex("session", "1b 5b 31 33 3b 32 75")

    assert calls == [
        (
            (
                "send-keys",
                "-t",
                "session",
                "-H",
                "1b",
                "5b",
                "31",
                "33",
                "3b",
                "32",
                "75",
            ),
            {"timeout": 10},
        )
    ]


def test_dispatch_turn_input_actions_orders_type_raw_hex_then_named_keys(monkeypatch):
    actions = []
    monkeypatch.setattr(
        scenarios,
        "_send",
        lambda _session, value: actions.append(("send", value)),
    )
    monkeypatch.setattr(
        scenarios,
        "_type_text",
        lambda _session, value: actions.append(("type", value)),
    )
    monkeypatch.setattr(
        scenarios,
        "_send_raw_hex",
        lambda _session, value: actions.append(("raw_hex", value)),
    )
    monkeypatch.setattr(
        scenarios,
        "_send_key_sequence",
        lambda _session, value: actions.append(("keys", value)),
    )

    scenarios._dispatch_turn_input_actions(
        "session",
        {
            "type": "second line",
            "raw_hex": "1b 5b 31 33 3b 32 75",
            "keys": ["Enter"],
        },
    )

    assert actions == [
        ("type", "second line"),
        ("raw_hex", "1b 5b 31 33 3b 32 75"),
        ("keys", ["Enter"]),
    ]


def test_multiline_input_scenario_covers_idle_and_queued_shift_enter():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["multiline-input"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_MODEL": "openai/koder-fixture",
        "KODER_CONTEXT_WINDOW": "128000",
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "KODER_API_KEY": "multiline-input-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "multiline final response",
        "log_file": "$HOME/fake-openai-multiline-input.log",
        "ready_file": "$HOME/fake-openai-multiline-input.ready",
        "scenario": "streaming_tool_queue",
        "stream_delay": 0.08,
        "stream_lines": 160,
    }
    assert scenario["turns"][0]["resize"] == {"width": 72, "height": 40}
    assert scenario["turns"][1]["type"] == IDLE_FIRST
    assert scenario["turns"][1]["expect_regex"] == [
        "idle-wrap-alpha[^\\n]*\\n[^\\n]*idle-wrap-omega"
    ]
    assert scenario["turns"][2]["raw_hex"] == "1b 5b 31 33 3b 32 75"
    assert scenario["turns"][3]["type"] == IDLE_SECOND
    assert scenario["turns"][3]["keys"] == ["Enter"]
    assert scenario["turns"][4]["type"] == QUEUED_FIRST
    assert scenario["turns"][4]["expect_regex"] == [
        "queued-wrap-alpha[^\\n]*\\n[^\\n]*queued-wrap-omega"
    ]
    assert scenario["turns"][5]["raw_hex"] == "1b 5b 32 37 3b 32 3b 31 33 7e"
    assert scenario["turns"][6]["type"] == QUEUED_SECOND
    assert scenario["turns"][6]["keys"] == ["Enter"]
    assert scenario["turns"][7]["wait"] == 14
    assert "queued: queued-wrap-alpha" in scenario["turns"][7]["expect_not"]
    assert scenario["turns"][8]["type"] == "/sta"
    assert {"/status", "/statusline"} <= set(scenario["turns"][8]["expect_all"])
    for turn in scenario["turns"]:
        if turn.get("capture") == "visible":
            assert "Window too small" in turn.get("expect_not", [])
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/fake-openai-multiline-input.log",
                [
                    f"{IDLE_FIRST}\\n{IDLE_SECOND}",
                    "Queued user input",
                    f"{QUEUED_FIRST}\\n{QUEUED_SECOND}",
                ],
            ]
        }
    ]


def test_tui_feature_scenarios_are_multi_turn_and_not_placeholder_smoke_checks():
    manifest = _load_manifest(DEFAULT_MANIFEST)

    all_scenarios = []
    for suite_name in ("slash_commands", "agents", "teams", "skills", "features"):
        suite = manifest[suite_name]
        for name, payload in suite.items():
            all_scenarios.append((suite_name, name, payload))

    assert all_scenarios
    for suite_name, name, payload in all_scenarios:
        has_valid_level = payload["validation_level"] in VALIDATION_LEVELS
        assert has_valid_level, f"{suite_name}/{name} has invalid validation level"
        assert payload["purpose"].strip(), f"{suite_name}/{name} has no purpose"
        assert len(payload["turns"]) >= 2, f"{suite_name}/{name} is not multi-turn"
        assert any(
            turn.get("expect_any")
            or turn.get("expect_all")
            or turn.get("expect_regex")
            or turn.get("expect_not")
            or turn.get("expect_bottom_all")
            or turn.get("expect_session_dead")
            or turn.get("expect_tmux_panes_min")
            or turn.get("expect_tmux_any_pane_any")
            or turn.get("expect_tmux_any_pane_all")
            for turn in payload["turns"]
        ), f"{suite_name}/{name} has no assertions"


def test_clear_scenario_is_acceptance_backed_by_session_switch_and_history_reset():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["clear"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/rename clear-source",
        "expect_all": ["Session renamed to: clear-source"],
    }
    assert scenario["turns"][1] == {
        "send": "!printf 'clear_history_seed\\n'",
        "expect_all": ["Shell Mode", "clear_history_seed"],
    }
    assert scenario["turns"][2] == {
        "keys": ["Up"],
        "capture": "visible",
        "expect_all": ["!printf 'clear_history_seed\\n'"],
    }
    clear_turn = scenario["turns"][4]
    assert clear_turn["send"] == "/clear"
    assert clear_turn["capture"] == "visible"
    assert "Switched to session:" in clear_turn["expect_all"]
    assert {"clear_history_seed", "clear-source"} <= set(clear_turn["expect_not"])
    new_session_turn = scenario["turns"][5]
    assert new_session_turn["send"] == "/session"
    assert "session_id:" in new_session_turn["expect_all"]
    assert "clear-source" in new_session_turn["expect_not"]
    history_turn = scenario["turns"][6]
    assert history_turn["keys"] == ["Up"]
    assert {"clear_history_seed", "!printf"} <= set(history_turn["expect_not"])
    assert scenario["turns"][-2] == {
        "send": "/resume clear-source",
        "expect_all": ["Switched to session:"],
    }
    assert scenario["turns"][-1] == {
        "send": "/session",
        "expect_all": ["display_name: clear-source", "title: clear-source"],
    }
    assert scenario["post_assertions"] == [
        {
            "sqlite_contains": [
                "$HOME/.koder/koder.db",
                "select title from session_metadata where title = 'clear-source'",
                "clear-source",
            ]
        }
    ]


def test_commit_scenario_is_acceptance_backed_by_full_git_state_matrix():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["commit"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"].startswith("!git add AGENTS.md")
    assert "commit-fixture" in scenario["turns"][0]["expect_all"]
    dirty_turn = scenario["turns"][1]
    assert dirty_turn["send"] == "/commit"
    assert "Branch:" in dirty_turn["expect_all"]
    assert "Staged changes:" in dirty_turn["expect_all"]
    assert "staged.txt" in dirty_turn["expect_all"]
    assert "Unstaged changes:" in dirty_turn["expect_all"]
    assert "sample.txt" in dirty_turn["expect_all"]
    assert "1 untracked file(s):" in dirty_turn["expect_all"]
    assert "- untracked.txt" in dirty_turn["expect_all"]
    assert "Ready to commit." in dirty_turn["expect_all"]
    assert scenario["turns"][2] == {
        "send": "!git add sample.txt untracked.txt && git commit -m scenario-commit && echo commit-created",
        "expect_all": ["scenario-commit", "commit-created"],
    }
    assert scenario["turns"][3] == {
        "send": "/commit",
        "expect_all": [
            "Branch:",
            "No staged changes.",
            "Nothing to commit, working tree clean.",
        ],
    }
    assert scenario["turns"][4] == {
        "send": '!test -z "$(git status --short)" && git log -1 --pretty=%s && echo clean-status',
        "expect_all": ["scenario-commit", "clean-status"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/staged.txt", "staged"]},
        {"file_contains": ["$REPO/untracked.txt", "untracked"]},
        {"file_contains": ["$REPO/.git/COMMIT_EDITMSG", "scenario-commit"]},
    ]


def test_channels_scenario_is_acceptance_backed_by_launch_cli_args():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["channels"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["cli_args"] == [
        "--channels",
        "server:test-channel,plugin:team-chat@local",
        "--dangerously-load-development-channels",
        "server:dev-channel",
    ]
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    channels_turn = scenario["turns"][0]
    assert channels_turn["send"] == "/channels"
    assert channels_turn["capture"] == "visible"
    assert {
        "channels:",
        "enabled: true",
        "configured: 3",
        "development_channels: true",
        "usage: uv run koder --channels server:<name>",
        "plugin_usage: uv run koder --channels plugin:<name>@<marketplace>",
        "entries:",
        "- server:test-channel",
        "- plugin:team-chat@local",
        "- server:dev-channel [development]",
    } <= set(channels_turn["expect_all"])
    assert scenario["turns"][1] == {
        "send": "/channels help",
        "expect_all": [
            "Usage: /channels",
            "--channels server:<name>",
            "--channels plugin:<name>@<marketplace>",
        ],
    }
    assert scenario["turns"][2] == {
        "send": "/channels install team-chat",
        "expect_all": ["Usage: /channels"],
    }
    assert scenario["turns"][3] == {
        "send": "/mcp",
        "expect_all": ["No MCP servers configured."],
    }


def test_schedule_scenario_is_acceptance_backed_by_cron_registry_flow():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["schedule"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/schedule extra",
        "expect_all": ["Usage: /schedule"],
    }
    assert scenario["turns"][1] == {
        "send": "/schedule",
        "expect_all": ["No scheduled tasks", "cron_create", "/loop"],
    }
    create_turn = scenario["turns"][2]
    assert create_turn["send"].startswith(
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python -c'
    )
    assert "cron_create" in create_turn["send"]
    assert "schedule-create-ok" not in create_turn["send"]
    assert create_turn["expect_all"] == ["schedule-create-ok", "Invalid cron expression"]
    list_turn = scenario["turns"][3]
    assert list_turn["send"] == "/schedule"
    assert {
        "Scheduled tasks (2):",
        "cron: 0 9 * * *",
        "human_schedule: at 9:00",
        "recurring: true",
        "prompt: morning standup",
        "cron: 30 14 * * 1",
        "human_schedule: on Mon at 14:30",
        "recurring: false",
        "prompt: monday review",
    } <= set(list_turn["expect_all"])
    delete_turn = scenario["turns"][4]
    assert "cron_delete" in delete_turn["send"]
    assert "schedule-delete-ok" not in delete_turn["send"]
    assert delete_turn["expect_all"] == ["schedule-delete-ok"]
    assert scenario["turns"][5] == {
        "send": "/schedule",
        "expect_all": [
            "Scheduled tasks (1):",
            "cron: 30 14 * * 1",
            "recurring: false",
            "prompt: monday review",
        ],
    }
    assert scenario["turns"][6]["expect_all"] == ["schedule-malformed-fixture"]
    assert scenario["turns"][7] == {
        "send": "/schedule",
        "expect_all": [
            "schedule: failed to read scheduled task registry",
            "scheduled_tasks.json",
            "error:",
        ],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/schedule-proof.txt", "schedule-delete-ok"]},
        {"file_contains": ["$HOME/.koder/scheduled_tasks.json", "{not json"]},
    ]


def test_loop_scenario_is_acceptance_backed_by_cron_command_flow():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["loop"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/loop",
        "expect_all": ["No loop jobs", "Usage: /loop", "@every 5m"],
    }
    assert scenario["turns"][1] == {
        "send": "/loop @every 5m check build",
        "expect_all": [
            "Loop job created",
            "cron: */5 * * * *",
            "recurring: true",
            "prompt: check build",
        ],
    }
    assert scenario["turns"][2] == {
        "send": "/loop once 30 14 * * 1 monday review",
        "expect_all": [
            "Loop job created",
            "cron: 30 14 * * 1",
            "recurring: false",
            "prompt: monday review",
        ],
    }
    assert {
        "Loop jobs (2):",
        "cron: */5 * * * *",
        "prompt: check build",
        "cron: 30 14 * * 1",
        "recurring: false",
        "prompt: monday review",
    } <= set(scenario["turns"][3]["expect_all"])
    assert scenario["turns"][4] == {
        "send": "/loop @after-turn follow up",
        "expect_all": ["loop: unsupported schedule", "@after-turn"],
    }
    delete_turn = scenario["turns"][5]
    assert "cron_delete" in delete_turn["send"]
    assert "loop-delete-ok" not in delete_turn["send"]
    assert delete_turn["expect_all"] == ["loop-delete-ok"]
    assert scenario["turns"][6] == {
        "send": "/loop",
        "capture": "visible",
        "expect_all": [
            "Loop jobs (1):",
            "cron: 30 14 * * 1",
            "prompt: monday review",
        ],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/loop-proof.txt", "loop-delete-ok"]},
        {"file_contains": ["$HOME/.koder/scheduled_tasks.json", "monday review"]},
        {"file_not_contains": ["$HOME/.koder/scheduled_tasks.json", "check build"]},
    ]


def test_compact_scenario_is_acceptance_backed_by_persisted_session_rewrite():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["compact"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_API_KEY": "scenario-key",
        "OPENAI_API_KEY": "",
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
    }
    assert scenario["fake_openai"]["port"] == 0
    assert scenario["fake_openai"]["response"] == "Compacted 2 earlier messages"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    seed_turn = scenario["turns"][0]
    assert seed_turn["send"].startswith(
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python -c'
    )
    assert "compact older user" in seed_turn["send"]
    assert "compact kept assistant" in seed_turn["send"]
    assert "compact-fixture" not in seed_turn["send"]
    assert seed_turn["expect_all"] == ["compact-fixture"]
    assert scenario["turns"][1] == {
        "send": "/compact unexpected",
        "expect_all": ["Usage: /compact"],
    }
    compact_turn = scenario["turns"][2]
    assert compact_turn["send"] == "/compact"
    assert {
        "compacting...",
        "compacted, context size",
        "->",
    } <= set(compact_turn["expect_all"])
    proof_turn = scenario["turns"][3]
    assert "len(items)==3" in proof_turn["send"]
    assert "[Conversation compacted]" in proof_turn["send"]
    assert "compact-db-proof.txt" in proof_turn["send"]
    assert "compact-db-ok" not in proof_turn["send"]
    assert proof_turn["expect_all"] == ["compact-db-ok"]
    assert scenario["turns"][4] == {
        "send": "/summary",
        "expect_all": ["Session Summary:"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/compact-db-proof.txt", "compact-db-ok"]}
    ]


def test_bughunter_scenario_is_acceptance_backed_by_diff_evidence_and_clean_edge():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["bughunter"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    setup_turn = scenario["turns"][0]
    assert setup_turn["send"].startswith(
        "!git add AGENTS.md docs/runtime-notes.md .koder sample.txt"
    )
    assert "bughunter-fixture" in setup_turn["expect_all"]
    dirty_turn = scenario["turns"][1]
    assert dirty_turn["send"] == "/bughunter division regression"
    assert "working_tree: dirty" in dirty_turn["expect_all"]
    assert "M bughunter_target.py" in dirty_turn["expect_all"]
    assert "diff --git a/bughunter_target.py b/bughunter_target.py" in dirty_turn["expect_all"]
    assert "-    return numerator / denominator" in dirty_turn["expect_all"]
    assert "+    return numerator / 0" in dirty_turn["expect_all"]
    assert scenario["turns"][2] == {
        "send": "!git add bughunter_target.py && git commit -m bughunter-clean-edge && echo bughunter-cleaned",
        "expect_all": ["bughunter-clean-edge", "bughunter-cleaned"],
    }
    clean_turn = scenario["turns"][3]
    assert clean_turn["send"] == "/bughunter clean edge"
    assert "working_tree: clean" in clean_turn["expect_all"]
    assert "diff_evidence:" in clean_turn["expect_all"]
    assert "none" in clean_turn["expect_all"]
    assert scenario["turns"][4] == {
        "send": '!test -z "$(git status --short)" && git log -1 --pretty=%s && echo bughunter-final-clean',
        "expect_all": ["bughunter-clean-edge", "bughunter-final-clean"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/bughunter_target.py", "return numerator / 0"]},
        {"file_contains": ["$REPO/.git/COMMIT_EDITMSG", "bughunter-clean-edge"]},
    ]


def test_diff_scenario_is_acceptance_backed_by_git_and_conversation_edits():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["diff"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"].startswith('!uv --project "$KODER_SCENARIO_SOURCE_ROOT"')
    assert "diff-fixture" in scenario["turns"][0]["expect_all"]
    dirty_turn = scenario["turns"][1]
    assert dirty_turn["send"] == "/diff"
    assert "### Uncommitted changes" in dirty_turn["expect_all"]
    assert "sample.txt (+1 -0)" in dirty_turn["expect_all"]
    assert "### Conversation edits" in dirty_turn["expect_all"]
    assert 'Turn 1: "change conversation file"' in dirty_turn["expect_all"]
    assert "conversation.txt (+2 -1)" in dirty_turn["expect_all"]
    assert scenario["turns"][2] == {
        "send": "!git add sample.txt && git commit -m diff-clean && echo diff-clean-commit",
        "expect_all": ["diff-clean", "diff-clean-commit"],
    }
    clean_turn = scenario["turns"][3]
    assert clean_turn["send"] == "/diff"
    assert "No uncommitted changes." in clean_turn["expect_all"]
    assert 'Turn 1: "change conversation file"' in clean_turn["expect_all"]
    assert "conversation.txt (+2 -1)" in clean_turn["expect_all"]
    assert scenario["turns"][4] == {
        "send": "!git diff --quiet HEAD && echo clean-diff",
        "expect_all": ["clean-diff"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/diff-seed.txt", "seeded conversation diff"]}
    ]


def test_init_scenario_is_acceptance_backed_by_local_generation_flow():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["init"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/init",
        "expect_all": [
            "AGENTS.md already exists. Run /init-explore to improve it from the codebase."
        ],
    }
    assert scenario["turns"][1] == {
        "send": "/init extra",
        "expect_all": ["Usage: /init"],
    }
    remove_turn = scenario["turns"][2]
    assert remove_turn["send"].startswith("!mv AGENTS.md AGENTS.original")
    assert "init-removed" not in remove_turn["send"]
    assert remove_turn["expect_all"] == ["init-removed"]
    generate_turn = scenario["turns"][3]
    assert generate_turn["send"] == "/init"
    assert {
        "AGENTS.md generated.",
        "path:",
        "commands_detected: 0",
        "Found 1 magic doc(s):",
        "docs/runtime-notes.md: Runtime Notes",
        "tip: run /init-explore",
    } <= set(generate_turn["expect_all"])
    grep_turn = scenario["turns"][4]
    assert "This file provides guidance to Koder" in grep_turn["send"]
    assert "init-file-ok" not in grep_turn["send"]
    assert grep_turn["expect_all"] == [
        "This file provides guidance to Koder",
        "## Commands",
        "## Working Guidelines",
        "# Test project",
        "init-file-ok",
    ]
    assert scenario["turns"][5] == {
        "send": "/init",
        "expect_all": [
            "AGENTS.md already exists. Run /init-explore to improve it from the codebase."
        ],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$REPO/AGENTS.md",
                [
                    "# AGENTS.md",
                    "This file provides guidance to Koder",
                    "## Commands",
                    "## Working Guidelines",
                ],
            ]
        },
        {"file_contains": ["$REPO/AGENTS.original", "# Test project"]},
    ]


def test_init_verifiers_scenario_is_acceptance_backed_by_generated_skill_contract():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["init-verifiers"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_VERIFIER_MARKER": "verifier-skill-ok",
        "KODER_VERIFIER_FILE_MARKER": "verifier-skill-file-present",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    create_turn = scenario["turns"][0]
    assert create_turn["send"] == "/init-verifiers cli"
    assert "init-verifiers: created" in create_turn["expect_all"]
    assert "name: verifier-cli" in create_turn["expect_all"]
    assert "type: cli" in create_turn["expect_all"]
    grep_turn = scenario["turns"][1]
    assert grep_turn["send"].startswith("!grep -F 'name: verifier-cli'")
    assert '"$KODER_VERIFIER_MARKER"' in grep_turn["send"]
    assert "run_shell:tmux *" in grep_turn["expect_all"]
    assert "Report PASS or FAIL" in grep_turn["expect_all"]
    assert "verifier-skill-ok" in grep_turn["expect_all"]
    skills_turn = scenario["turns"][2]
    assert skills_turn["send"] == "/skills"
    assert "[project] verifier-cli" in skills_turn["expect_all"]
    assert "[project] demo-skill" in skills_turn["expect_all"]
    exists_turn = scenario["turns"][3]
    assert exists_turn["send"] == "/init-verifiers cli"
    assert "init-verifiers: exists" in exists_turn["expect_all"]
    file_turn = scenario["turns"][4]
    assert file_turn["send"].startswith("!test -f .koder/skills/verifier-cli/SKILL.md")
    assert '"$KODER_VERIFIER_FILE_MARKER"' in file_turn["send"]
    assert file_turn["expect_all"] == ["lines=39", "verifier-skill-file-present"]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$REPO/.koder/skills/verifier-cli/SKILL.md",
                [
                    "name: verifier-cli",
                    "description: Verify CLI and TUI behavior with tmux and multi-turn assertions",
                    "allowed-tools:",
                    "run_shell:tmux *",
                    "Report PASS or FAIL",
                ],
            ]
        }
    ]


def test_doctor_scenario_is_acceptance_backed_by_runtime_diagnostic_matrix():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["doctor"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"KODER_DOCTOR_MARKER": "doctor-shell-ok"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    doctor_turn = scenario["turns"][0]
    assert doctor_turn["send"] == "/doctor"
    assert doctor_turn["capture"] == "visible"
    assert {
        "cwd:",
        "python:",
        "invoked_binary: $RUNTIME_CLI",
        "invoked_binary:",
        "config_path:",
        "model: gpt-4.1",
        "provider: openai",
        "permission_mode: default",
        "mcp_servers: 0",
        "ripgrep_working:",
        "ripgrep_mode:",
        "ripgrep_path:",
    } <= set(doctor_turn["expect_all"])
    assert "cwd: .*/repo" in doctor_turn["expect_regex"]
    assert "installation_type: (development|local|unknown)" in doctor_turn["expect_regex"]
    assert "python: $RUNTIME_PYTHON_PATTERN" in doctor_turn["expect_regex"]
    assert "ripgrep_working: (true|false)" in doctor_turn["expect_regex"]
    shell_turn = scenario["turns"][1]
    assert shell_turn["send"].startswith('!test "$(basename "$PWD")" = repo')
    assert '"$KODER_DOCTOR_MARKER"' in shell_turn["send"]
    assert shell_turn["expect_all"] == ["doctor-shell-ok"]
    assert scenario["turns"][2] == {
        "send": "/version",
        "expect_all": ["version:", "package: koder", "cli_banner:", "(Koder)"],
    }
    assert scenario["turns"][3] == {
        "send": "/status",
        "expect_all": ["version:", "Runtime slash commands:", "Working directory:"],
    }


def test_mcp_scenario_is_acceptance_backed_by_project_config_round_trip():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["mcp"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/mcp",
        "expect_all": ["No MCP servers configured."],
    }
    assert scenario["turns"][1] == {
        "send": "/mcp unexpected",
        "expect_all": ["Usage: /mcp"],
    }
    add_turn = scenario["turns"][2]
    assert (
        'uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync koder mcp add-json'
        in (add_turn["send"])
    )
    assert "Path(sys.executable).resolve()" in add_turn["send"]
    assert "scenario-mcp" in add_turn["send"]
    assert "--scope project" in add_turn["send"]
    assert add_turn["expect_all"] == ["Added MCP server: scenario-mcp"]
    assert scenario["turns"][3] == {
        "send": "/mcp",
        "expect_all": [
            "scenario-mcp",
            "[project]",
            "stdio",
            "$RUNTIME_PYTHON_RESOLVED -m scenario_server",
        ],
    }
    get_turn = scenario["turns"][4]
    assert get_turn["send"] == (
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync koder mcp get scenario-mcp --scope project'
    )
    assert {
        '"name": "scenario-mcp"',
        '"transport_type": "stdio"',
        '"command": "$RUNTIME_PYTHON_RESOLVED"',
        '"env_vars": {',
        '"SCENARIO": "1"',
        '"scope": "project"',
    } <= set(get_turn["expect_all"])
    assert scenario["turns"][5] == {"send": "/doctor", "expect_all": ["mcp_servers: 1"]}
    assert scenario["turns"][6] == {
        "send": '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync koder mcp remove scenario-mcp --scope project',
        "expect_all": ["Removed MCP server: scenario-mcp"],
    }
    assert scenario["turns"][7] == {
        "send": "/mcp",
        "expect_all": ["No MCP servers configured."],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/.mcp.json", '"mcpServers": {}']}
    ]


def test_memory_scenario_is_acceptance_backed_by_project_user_and_remember_files():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["memory"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"KODER_MEMORY_MARKER": "memory-seed-ok"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/memory",
        "expect_all": ["No memories stored yet", "/remember skill"],
    }
    seed_turn = scenario["turns"][1]
    assert seed_turn["send"].startswith("!mkdir -p .koder/memory")
    assert "echo 'type: project'" in seed_turn["send"]
    assert "echo 'type: user'" in seed_turn["send"]
    assert "project-note.md" in seed_turn["send"]
    assert "user-note.md" in seed_turn["send"]
    assert '"$KODER_MEMORY_MARKER"' in seed_turn["send"]
    assert seed_turn["expect_all"] == ["memory-seed-ok"]
    seeded_listing = scenario["turns"][2]
    assert seeded_listing["send"] == "/memory"
    assert "Found 2 memory files" in seeded_listing["expect_all"]
    assert "[project] project-note.md: project memory marker" in seeded_listing["expect_all"]
    assert "[user] user-note.md: user memory marker" in seeded_listing["expect_all"]
    remember_turn = scenario["turns"][3]
    assert remember_turn["send"] == "/remember remembered memory marker"
    assert "remember: saved" in remember_turn["expect_all"]
    assert "index: .koder/memory/MEMORY.md" in remember_turn["expect_all"]
    final_listing = scenario["turns"][4]
    assert final_listing["send"] == "/memory"
    assert "Found 3 memory files" in final_listing["expect_all"]
    assert "remembered memory marker" in final_listing["expect_all"]
    candidate_listing = scenario["turns"][6]
    assert candidate_listing["send"] == "/memory candidates"
    assert "memory candidates: 2 pending" in candidate_listing["expect_all"]
    assert "scope=user" in candidate_listing["expect_all"]
    assert (
        "origin_project=/private/tmp/koder-memory-scenario-origin"
        in candidate_listing["expect_all"]
    )
    assert "origin_session=memory-scenario-session" in candidate_listing["expect_all"]
    assert scenario["turns"][7]["send"].startswith("/memory show ")
    assert scenario["turns"][8]["send"].startswith("/memory approve ")
    assert "memory candidate approved" in scenario["turns"][8]["expect_all"]
    assert scenario["turns"][9]["send"].startswith("/memory reject ")
    assert "candidate rejected" in scenario["turns"][9]["expect_all"]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$REPO/.koder/memory/project-note.md",
                ["type: project", "description: project memory marker", "project body"],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/memory/user-note.md",
                ["type: user", "description: user memory marker", "user body"],
            ]
        },
        {"file_contains": ["$REPO/.koder/memory/MEMORY.md", "remembered memory marker"]},
        {
            "file_glob_contains": [
                "$REPO/.koder/memory/*remembered-memory-marker.md",
                [
                    "type: project",
                    "description: remembered memory marker",
                    "remembered memory marker",
                ],
            ]
        },
        {
            "file_contains": [
                (
                    "$HOME/.koder/memory/auto-dream-"
                    "b5bfd0b63ffd1730fa6cc9074193e4388034798bde59f9fd740673df4602f6a6.md"
                ),
                [
                    "type: user",
                    "storage_scope: user",
                    (
                        "source_candidate: "
                        "b5bfd0b63ffd1730fa6cc9074193e4388034798bde59f9fd740673df4602f6a6"
                    ),
                    "description: candidate memory marker",
                    "approved candidate marker",
                ],
            ]
        },
        {
            "path_not_exists": (
                "$HOME/.koder/skill-candidates/pending/"
                "0499eb0d392f78a6997245ef10ed24d94af350edd039d7f0b0a1e50e46de58b6.json"
            )
        },
    ]


def test_ctx_viz_scenario_is_acceptance_backed_by_seeded_transcript_and_files():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["ctx_viz"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"KODER_CTX_VIZ_MARKER": "ctx-viz-fixture-ok"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    fresh_turn = scenario["turns"][0]
    assert fresh_turn == {
        "send": "/ctx_viz",
        "expect_all": [
            "Working directory:",
            "AGENTS.md content:",
            "# Test project",
            "Session messages: 0",
        ],
    }
    seed_turn = scenario["turns"][1]
    assert seed_turn["send"].startswith(
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python'
    )
    assert "ctx viz user prompt" in seed_turn["send"]
    assert "ctx viz assistant answer" in seed_turn["send"]
    assert "docs/runtime-notes.md" in seed_turn["send"]
    assert "KODER_CTX_VIZ_MARKER" in seed_turn["send"]
    assert seed_turn["expect_all"] == ["ctx-viz-fixture-ok"]
    seeded_turn = scenario["turns"][2]
    assert seeded_turn["send"] == "/ctx_viz"
    assert "Session messages: 2" in seeded_turn["expect_all"]
    assert "Files in session context:" in seeded_turn["expect_all"]
    assert "- AGENTS.md" in seeded_turn["expect_all"]
    assert "- docs/runtime-notes.md" in seeded_turn["expect_all"]
    assert "Recent transcript:" in seeded_turn["expect_all"]
    assert "user: ctx viz user prompt" in seeded_turn["expect_all"]
    assert "assistant: ctx viz assistant answer" in seeded_turn["expect_all"]
    context_turn = scenario["turns"][3]
    assert context_turn["send"] == "/context"
    assert {"Conversation", "Files", "Instructions", "### Files in context"} <= set(
        context_turn["expect_all"]
    )
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/ctx-viz-seed.txt", "seeded ctx viz context"]}
    ]


def test_context_scenario_is_acceptance_backed_by_exact_token_categories():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["context"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"KODER_CONTEXT_MARKER": "context-fixture-ok"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    fresh_turn = scenario["turns"][0]
    assert fresh_turn["send"] == "/context"
    assert "**Model:** gpt-4.1" in fresh_turn["expect_all"]
    assert "| Instructions | 9 |" in fresh_turn["expect_all"]
    seed_turn = scenario["turns"][1]
    assert seed_turn["send"].startswith(
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python'
    )
    assert "ctx viz user prompt" in seed_turn["send"]
    assert "ctx viz assistant answer" in seed_turn["send"]
    assert "docs/runtime-notes.md" in seed_turn["send"]
    assert "KODER_CONTEXT_MARKER" in seed_turn["send"]
    assert seed_turn["expect_all"] == ["context-fixture-ok"]
    seeded_context = scenario["turns"][2]
    assert seeded_context["send"] == "/context"
    assert "| Conversation | 30 |" in seeded_context["expect_all"]
    assert "| Files | 21 |" in seeded_context["expect_all"]
    assert "| Instructions | 9 |" in seeded_context["expect_all"]
    assert "- docs/runtime-notes.md" in seeded_context["expect_all"]
    ctx_viz_turn = scenario["turns"][3]
    assert ctx_viz_turn["send"] == "/ctx_viz"
    assert "Session messages: 2" in ctx_viz_turn["expect_all"]
    assert "user: ctx viz user prompt" in ctx_viz_turn["expect_all"]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/context-seed.txt", "seeded context accounting"]}
    ]


def test_brief_scenario_is_acceptance_backed_by_persisted_config_toggle():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["brief"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"] == [
        {"send": "/brief", "expect_all": ["Brief-only mode enabled"]},
        {
            "send": '!cat "$HOME/.koder/config.yaml"',
            "expect_all": ["harness:", "brief_mode_enabled: true"],
        },
        {"send": "/brief", "expect_all": ["Brief-only mode disabled"]},
        {
            "send": '!cat "$HOME/.koder/config.yaml"',
            "expect_all": ["harness:", "brief_mode_enabled: false"],
        },
        {"send": "/brief extra", "expect_all": ["Usage: /brief"]},
    ]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/config.yaml",
                ["harness:", "brief_mode_enabled: false"],
            ]
        }
    ]


def test_buddy_scenario_is_acceptance_backed_by_persisted_companion_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["buddy"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_BUDDY_SEED": "scenario-buddy",
        "KODER_BUDDY_FINAL_MARKER": "buddy-final-config",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/buddy status",
        "expect_all": ["buddy: no companion hatched yet."],
    }
    assert scenario["turns"][1] == {"send": "/clear", "expect_all": ["Switched to session:"]}
    hatch_turn = scenario["turns"][2]
    assert hatch_turn["send"] == "/buddy"
    for expected in ["buddy: hatched", "name:", "species:", "rarity:", "personality:"]:
        assert expected in hatch_turn["expect_all"]
    config_turn = scenario["turns"][3]
    assert config_turn["send"] == '!cat "$HOME/.koder/config.yaml"'
    assert "companion_muted: false" in config_turn["expect_all"]
    pet_turn = scenario["turns"][4]
    assert pet_turn["send"] == "/buddy"
    assert "buddy: pet" in pet_turn["expect_all"]
    assert "reaction:" in pet_turn["expect_all"]
    assert any(
        turn.get("send") == "/buddy mute" and "buddy: muted" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/buddy unmute" and "buddy: unmuted" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1]["send"].startswith("!grep -F 'companion_muted: false'")
    assert "KODER_BUDDY_FINAL_MARKER" in scenario["turns"][-1]["send"]
    assert "companion_muted: false" in scenario["turns"][-1]["expect_all"]
    assert "buddy-final-config" in scenario["turns"][-1]["expect_all"]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/config.yaml",
                ["companion:", "name:", "personality:", "companion_muted: false"],
            ]
        }
    ]


def test_feedback_scenario_is_acceptance_backed_by_redacted_local_event():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["feedback"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"KODER_FEEDBACK_SECRET": "feedback-secret-token-123456"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/feedback",
        "expect_all": [
            "feedback:",
            "saved: false",
            "usage: /feedback <message>",
            "repo:",
            "branch:",
            "cwd:",
        ],
    }
    saved_turn = scenario["turns"][1]
    assert saved_turn["send"] == "/feedback scenario feedback api_key=feedback-secret-token-123456"
    assert "feedback: saved" in saved_turn["expect_all"]
    assert "message: scenario feedback api_key=[REDACTED]" in saved_turn["expect_all"]
    assert "expect_not" not in saved_turn
    artifact_turn = scenario["turns"][2]
    assert artifact_turn["send"] == '!cat "$HOME/.koder/feedback/feedback.jsonl"'
    assert '"message": "scenario feedback api_key=[REDACTED]"' in artifact_turn["expect_all"]
    assert '"git_status":' in artifact_turn["expect_all"]
    assert "expect_not" not in artifact_turn
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/feedback/feedback.jsonl",
                [
                    '"message": "scenario feedback api_key=[REDACTED]"',
                    '"cwd":',
                    '"repo":',
                    '"branch":',
                    '"git_status":',
                ],
            ]
        },
        {
            "file_not_contains": [
                "$HOME/.koder/feedback/feedback.jsonl",
                "feedback-secret-token-123456",
            ]
        },
    ]


def test_branch_slash_scenario_is_acceptance_backed_by_git_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["branch"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/branch",
        "expect_all": ["branch:", "dirty: true", "sample.txt"],
    }
    assert any(
        turn.get("send") == "/branch bad..name"
        and "branch: invalid name bad..name" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/branch scenario-branch"
        and "action: created" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "!git branch --show-current"
        and "scenario-branch" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert {"file_contains": ["$REPO/.git/HEAD", "refs/heads/scenario-branch"]} in scenario[
        "post_assertions"
    ]


def test_hooks_slash_scenario_is_acceptance_backed_by_project_hook_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["hooks"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "hooks-fixture" in scenario["turns"][0]["expect_all"]
    hooks_turn = scenario["turns"][1]
    assert hooks_turn["send"] == "/hooks"
    assert "count: 2" in hooks_turn["expect_all"]
    assert "matcher=Bash" in hooks_turn["expect_all"]
    assert "echo guard" in hooks_turn["expect_all"]
    assert "echo done" in hooks_turn["expect_all"]
    assert {
        "file_contains": [
            "$REPO/.koder/settings.json",
            ["PreToolUse", "echo guard", "Stop", "echo done"],
        ]
    } in scenario["post_assertions"]


def test_keybindings_scenario_is_acceptance_backed_by_persisted_override():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["keybindings"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/keybindings",
        "expect_all": [
            "keybindings:",
            "settings_path:",
            "overrides: 0",
            "- submit: enter",
            "- complete: tab",
        ],
    }
    assert any(
        turn.get("send") == "/keybindings set complete c-space"
        and "key: c-space" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/keybindings set missing c-x"
        and "keybindings: unknown action" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/keybindings set submit definitely-not-a-key"
        and "keybindings: invalid key" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/keybindings.json", ['"complete": "c-space"']]},
        {"file_not_contains": ["$HOME/.koder/keybindings.json", "definitely-not-a-key"]},
    ]


def test_vim_scenario_is_acceptance_backed_by_persisted_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["vim"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/vim on",
        "expect_all": ["vim: enabled", "settings_path:"],
    }
    assert scenario["turns"][1] == {
        "send": '!cat "$HOME/.koder/vim_state.json"',
        "expect_all": ['"vim_enabled": true'],
    }
    assert any(
        turn.get("send") == "/output-style" and "vim_mode: true" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/vim maybe" and "Usage: /vim [on|off]" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1] == {
        "send": "/vim off",
        "expect_all": ["vim: disabled", "settings_path:"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/vim_state.json", '"vim_enabled": false']}
    ]


def test_theme_scenario_is_acceptance_backed_by_persisted_settings():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["theme"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/theme",
        "expect_all": ["theme: adaptive", "settings_path:"],
    }
    assert any(
        turn.get("send") == "/theme dark" and "theme: dark" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == '!cat "$HOME/.koder/settings.json"'
        and '"theme": "dark"' in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/output-style" and "theme: dark" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/theme ultraviolet"
        and "theme: invalid ultraviolet" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1] == {
        "send": "/theme adaptive",
        "expect_all": ["theme: adaptive", "settings_path:"],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/settings.json",
                ['"outputStyle"', '"theme": "adaptive"'],
            ]
        },
        {"file_not_contains": ["$HOME/.koder/settings.json", "ultraviolet"]},
    ]


def test_output_style_scenario_is_acceptance_backed_by_all_controls_and_reset():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["output-style"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/output-style",
        "expect_all": [
            "output-style:",
            "theme: adaptive",
            "color: default",
            "vim_mode: false",
            "statusline: not configured",
            "controls: /theme, /color, /statusline, /vim",
        ],
    }
    assert any(
        turn.get("send") == "/output-style theme dark"
        and "theme: dark" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/output-style color cyan"
        and "Session color set to: cyan" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/output-style vim on" and "vim: enabled" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any("output-style-fixture" in turn.get("expect_all", []) for turn in scenario["turns"])
    assert any(
        turn.get("send") == "/output-style"
        and "statusline: printf style-ready" in turn.get("expect_all", [])
        and "vim_mode: true" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/output-style reset"
        and "theme: adaptive" in turn.get("expect_all", [])
        and "vim_mode: false" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/settings.json",
                ['"outputStyle"', '"theme": "adaptive"'],
            ]
        },
        {"file_not_contains": ["$HOME/.koder/settings.json", '"statusLine"']},
        {"file_contains": ["$HOME/.koder/vim_state.json", '"vim_enabled": false']},
    ]


def test_statusline_scenario_is_acceptance_backed_by_import_clear_and_persistence():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["statusline"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_STATUSLINE_MARKER": "statusline-fixture-ok",
        "KODER_STATUSLINE_CLEAR_MARKER": "statusline-clear-ok",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    seed_turn = scenario["turns"][0]
    assert seed_turn["send"].startswith('!mkdir -p "$HOME"')
    assert 'PS1="koder:\\W $ "' in seed_turn["send"]
    assert seed_turn["expect_all"] == ["statusline-fixture-ok"]
    assert scenario["turns"][1] == {
        "send": "/statusline",
        "expect_all": [
            "statusline: configured from",
            ".zshrc",
            "settings_path:",
            "command: printf 'koder:%s'",
        ],
    }
    assert scenario["turns"][2] == {
        "send": "/output-style",
        "expect_all": [
            "output-style:",
            "statusline: printf 'koder:%s'",
            "controls: /theme, /color, /statusline, /vim",
        ],
    }
    assert scenario["turns"][3] == {
        "send": "/statusline clear",
        "expect_all": ["statusline: removed custom status line", ".koder/settings.json"],
    }
    assert scenario["turns"][4] == {
        "send": '!cat "$HOME/.koder/settings.json" && echo "$KODER_STATUSLINE_CLEAR_MARKER"',
        "expect_all": ["{}", "statusline-clear-ok"],
        "expect_not": ['"statusLine"'],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.zshrc", 'PS1="koder:\\W $ "']},
        {"file_not_contains": ["$HOME/.koder/settings.json", '"statusLine"']},
    ]


def test_voice_scenario_is_acceptance_backed_by_provider_toggle_and_secret_redaction():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["voice"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"OPENAI_API_KEY": "voice-secret-token-123456"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/voice status",
        "expect_all": [
            "voice_enabled: False",
            "voice_provider: None",
            "effective_provider: openai",
        ],
        "expect_not": ["voice-secret-token-123456"],
    }
    assert scenario["turns"][1] == {
        "send": "/voice provider openai",
        "expect_all": ["Voice provider set to: openai"],
    }
    assert scenario["turns"][2] == {
        "send": "/voice",
        "expect_all": [
            "Voice mode enabled.",
            "provider: openai",
            "status: provider-backed voice routing configured",
            "shortcut: double-space",
        ],
        "expect_not": ["voice-secret-token-123456"],
    }
    assert scenario["turns"][3] == {
        "send": "/voice status",
        "expect_all": [
            "voice_enabled: True",
            "voice_provider: openai",
            "effective_provider: openai",
        ],
        "expect_not": ["voice-secret-token-123456"],
    }
    assert scenario["turns"][4] == {
        "send": "/voice provider llama",
        "expect_all": ["Unsupported voice provider: llama."],
    }
    assert scenario["turns"][-3] == {
        "send": "/voice provider clear",
        "expect_all": ["Voice provider cleared."],
    }
    assert scenario["turns"][-2] == {
        "send": "/voice",
        "expect_all": ["Voice mode disabled."],
    }
    assert scenario["turns"][-1] == {
        "send": "/voice status",
        "expect_all": [
            "voice_enabled: False",
            "voice_provider: None",
            "effective_provider: openai",
        ],
        "expect_not": ["voice-secret-token-123456"],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/config.yaml",
                ["voice:", "enabled: false", "provider: null"],
            ]
        },
        {"file_not_contains": ["$HOME/.koder/config.yaml", "voice-secret-token-123456"]},
    ]


def test_magic_docs_slash_scenario_is_acceptance_backed_by_refresh_and_removed_header():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["magic-docs"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_MAGIC_DOC_MARKER": "magic-doc-content-ok",
        "KODER_MAGIC_DOC_REMOVED": "magic-doc-header-removed",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/magic-docs",
        "expect_all": [
            "magic_docs:",
            "discovered: 1",
            "tracked: 0",
            "docs/runtime-notes.md: Runtime Notes (discovered)",
        ],
    }
    assert scenario["turns"][1] == {
        "send": "/magic-docs nope",
        "expect_all": ["Usage: /magic-docs [status|refresh]"],
    }
    assert scenario["turns"][2] == {
        "send": "/magic-docs refresh",
        "expect_all": [
            "magic_docs: refresh",
            "checked: 1",
            "updated: 1",
            "docs/runtime-notes.md: updated",
            "managed section refreshed",
        ],
    }
    assert scenario["turns"][3]["send"] == "/magic-docs"
    assert "tracked: 1" in scenario["turns"][3]["expect_all"]
    grep_turn = scenario["turns"][4]
    assert grep_turn["send"].startswith("!grep -F 'koder-magic-docs:auto-refresh-start'")
    assert "magic-doc-content-ok" in grep_turn["expect_all"]
    remove_turn = scenario["turns"][5]
    assert remove_turn["send"].startswith("!printf '%s")
    assert remove_turn["expect_all"] == ["magic-doc-header-removed"]
    assert scenario["turns"][6] == {
        "send": "/magic-docs refresh",
        "expect_all": [
            "magic_docs: refresh",
            "checked: 1",
            "updated: 0",
            "docs/runtime-notes.md: removed",
            "header missing",
        ],
    }
    assert scenario["turns"][7] == {
        "send": "/magic-docs",
        "expect_all": ["magic_docs:", "discovered: 0", "tracked: 0", "docs: none"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/docs/runtime-notes.md", "No longer a Magic Doc."]},
        {"file_not_contains": ["$REPO/docs/runtime-notes.md", "# MAGIC DOC:"]},
    ]


def test_onboarding_scenario_is_acceptance_backed_by_real_state_transitions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["onboarding"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"] == "/env unset KODER_API_KEY"
    first_turn = scenario["turns"][1]
    assert first_turn["send"] == "/onboarding"
    assert "Configure API key: Set KODER_API_KEY" in first_turn["expect_all"]
    assert "Completed: API key=✗, Model=✓, Workspace=✓" in first_turn["expect_all"]
    assert scenario["turns"][2] == {
        "send": "/env KODER_API_KEY=scenario-onboarding-key",
        "expect_all": ["env: set KODER_API_KEY for this session."],
    }
    assert scenario["turns"][3] == {
        "send": "/onboarding",
        "expect_all": ["✓ Setup complete! All configuration is in place."],
    }
    assert scenario["turns"][5]["send"] == "/onboarding"
    assert (
        "Trust workspace: Initialize .koder/ directory in your project"
        in scenario["turns"][5]["expect_all"]
    )
    assert "Completed: API key=✓, Model=✓, Workspace=✗" in scenario["turns"][5]["expect_all"]
    assert scenario["turns"][-1] == {
        "send": "/env unset KODER_API_KEY",
        "expect_all": ["env: removed KODER_API_KEY from this session."],
    }
    assert scenario["post_assertions"] == [
        {"path_exists": "$REPO/.koder/skills/demo-skill/SKILL.md"},
        {"path_glob_exists": "$HOME/.koder/session-env/*.sh"},
        {"file_glob_not_contains": ["$HOME/.koder/session-env/*.sh", "scenario-onboarding-key"]},
    ]


def test_onboarding_session_env_startup_scenario_is_hermetic_and_selected_provider_aware():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["onboarding-session-env-startup"]
    session_id = "onboarding-session-env-startup"
    session_file = f"$HOME/.koder/session-env/{session_id}.sh"

    assert scenario["validation_level"] == "acceptance"
    assert scenario["cli_args"] == ["--session", session_id]
    assert scenario["env"] == {
        "KODER_CONTEXT_WINDOW": "128000",
        "KODER_API_KEY": "",
        "OPENAI_API_KEY": "synthetic-unrelated-openai-key",
        "ANTHROPIC_API_KEY": "",
        "GOOGLE_API_KEY": "",
        "GEMINI_API_KEY": "",
        "AZURE_API_KEY": "",
        "OPENROUTER_API_KEY": "",
    }
    assert scenario["prelaunch_files"] == [
        {
            "path": session_file,
            "content": (
                "export KODER_MODEL=openrouter/anthropic/claude-3-opus\n"
                "export OPENROUTER_API_KEY=synthetic-session-openrouter-key\n"
            ),
        }
    ]
    assert scenario["turns"][0]["capture"] == "visible"
    assert "Setup Recommended" in scenario["turns"][0]["expect_not"]
    assert scenario["turns"][1] == {
        "send": "/env unset OPENROUTER_API_KEY",
        "expect_all": ["env: removed OPENROUTER_API_KEY from this session."],
    }
    assert scenario["turns"][2]["send"] == "/onboarding"
    assert "Configure API key: Set KODER_API_KEY" in scenario["turns"][2]["expect_all"]
    assert "Completed: API key=✗, Model=✓, Workspace=✓" in scenario["turns"][2]["expect_all"]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                session_file,
                "export KODER_MODEL=openrouter/anthropic/claude-3-opus",
            ]
        },
        {
            "file_not_contains": [
                session_file,
                "synthetic-session-openrouter-key",
            ]
        },
    ]


def test_debug_tool_call_scenario_is_acceptance_backed_by_seeded_records_and_redaction():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["debug-tool-call"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"OPENAI_API_KEY": "debug-secret-value-12345"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/debug-tool-call",
        "expect_all": ["debug-tool-call: no recorded tool calls in this session"],
    }
    assert any(
        turn.get("send", "").startswith('!uv --project "$KODER_SCENARIO_SOURCE_ROOT"')
        and "debug-tool-fixture" in turn.get("expect_all", [])
        and "debug-secret-value-12345" in turn.get("expect_not", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/debug-tool-call"
        and "debug-tool-call: 2 recorded item(s)" in turn.get("expect_all", [])
        and "[REDACTED]" in turn.get("expect_all", [])
        and "debug-secret-value-12345" in turn.get("expect_not", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/debug-tool-call show 1"
        and "kind: call" in turn.get("expect_all", [])
        and "api_key" in turn.get("expect_all", [])
        and "[REDACTED]" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/debug-tool-call show 2"
        and "kind: output" in turn.get("expect_all", [])
        and "secret=[REDACTED]" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-2] == {
        "send": "/debug-tool-call show 99",
        "expect_all": ["debug-tool-call: number must be between 1 and 2"],
    }
    assert scenario["turns"][-1] == {
        "send": "/debug-tool-call nope",
        "expect_all": ["Usage: /debug-tool-call [list|show <number>]"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/debug-tool-fixture.txt", "seeded tool-call records"]}
    ]


def test_export_scenario_is_acceptance_backed_by_json_markdown_files_and_edges():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["export"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {"send": "/session", "expect_all": ["session_id:"]}
    assert any(
        turn.get("send", "").startswith('!uv --project "$KODER_SCENARIO_SOURCE_ROOT"')
        and "export-fixture" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/export"
        and "messages: 3" in turn.get("expect_all", [])
        and "assistant: export ready" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/export json export-session.json"
        and "format: json" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/export markdown export-session.md"
        and "format: markdown" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/export json missing-dir/export.json"
        and "export: parent directory not found" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1] == {
        "send": "/export .",
        "expect_all": ["export: target is a directory", "path: ."],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/export-seed.txt", "seeded export session"]},
        {
            "file_contains": [
                "$REPO/export-session.json",
                ['"session_id"', "Export Scenario", '"content": "export ready"'],
            ]
        },
        {
            "file_contains": [
                "$REPO/export-session.md",
                ["# Koder Session Export: Export Scenario", "assistant: export ready"],
            ]
        },
    ]


def test_btw_scenario_is_acceptance_backed_by_fake_provider_and_session_context():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["btw"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "OPENAI_API_KEY": "btw-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "btw-fixture-answer: check migration coverage.",
        "log_file": "$HOME/fake-openai-btw.log",
        "ready_file": "$HOME/fake-openai-btw.ready",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/btw",
        "expect_all": ["Usage: /btw <question>"],
    }
    assert scenario["turns"][1] == {
        "send": '!test -f "$HOME/fake-openai-btw.ready" && cat "$HOME/fake-openai-btw.ready"',
        "expect_all": ["ready $FAKE_OPENAI_URL"],
    }
    assert "btw seeded user context: retry validation checklist" in scenario["turns"][2]["send"]
    assert scenario["turns"][3] == {
        "send": "/btw what risk should I mention?",
        "expect_all": ["btw-fixture-answer: check migration coverage."],
        "expect_not": ["btw-secret-token"],
    }
    assert "Current session context:" in scenario["turns"][4]["send"]
    assert "Side question: what risk should I mention?" in scenario["turns"][4]["send"]
    assert scenario["turns"][5] == {
        "send": "/status",
        "expect_all": ["Runtime slash commands", "connectivity: local"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/btw-seed.txt", "seeded btw transcript"]},
        {
            "file_contains": [
                "$HOME/fake-openai-btw.log",
                [
                    "/v1/chat/completions",
                    "Current session context:",
                    "btw seeded user context: retry validation checklist",
                    "Side question: what risk should I mention?",
                ],
            ]
        },
        {"file_not_contains": ["$HOME/fake-openai-btw.log", "btw-secret-token"]},
    ]


def test_torch_scenario_is_acceptance_backed_by_fake_provider_prompt_shape():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["torch"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "OPENAI_API_KEY": "torch-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "torch-fixture-plan: inspect koder_agent/core/usage_tracker.py and search for context token accounting.",
        "log_file": "$HOME/fake-openai-torch.log",
        "ready_file": "$HOME/fake-openai-torch.ready",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/torch",
        "expect_all": ["Usage: /torch <topic>", "Deeply explores a codebase topic"],
    }
    assert scenario["turns"][1] == {
        "send": '!test -f "$HOME/fake-openai-torch.ready" && cat "$HOME/fake-openai-torch.ready"',
        "expect_all": ["ready $FAKE_OPENAI_URL"],
    }
    assert scenario["turns"][2] == {
        "send": "/torch context token accounting",
        "expect_all": [
            "Torch: Exploring 'context token accounting'",
            "torch-fixture-plan: inspect koder_agent/core/usage_tracker.py",
        ],
        "expect_not": ["torch-secret-token"],
    }
    assert (
        "The user wants to deeply explore: context token accounting" in scenario["turns"][3]["send"]
    )
    assert "Suggested search queries" in scenario["turns"][3]["send"]
    assert scenario["turns"][4] == {
        "send": "/status",
        "expect_all": ["Runtime slash commands", "connectivity: local"],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/fake-openai-torch.log",
                [
                    "/v1/chat/completions",
                    "code exploration assistant",
                    "The user wants to deeply explore: context token accounting",
                    "Suggested search queries",
                ],
            ]
        },
        {"file_not_contains": ["$HOME/fake-openai-torch.log", "torch-secret-token"]},
    ]


def test_ultraplan_scenario_is_acceptance_backed_by_fake_provider_and_no_writes():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["ultraplan"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "OPENAI_API_KEY": "ultraplan-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "ultraplan-fixture-plan: update koder_agent/core/usage_tracker.py, add tests, and run tmux validation.",
        "log_file": "$HOME/fake-openai-ultraplan.log",
        "ready_file": "$HOME/fake-openai-ultraplan.ready",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/ultraplan",
        "expect_all": [
            "Usage: /ultraplan <feature/task>",
            "Creates a detailed, comprehensive implementation plan",
        ],
    }
    assert scenario["turns"][1] == {
        "send": '!test -f "$HOME/fake-openai-ultraplan.ready" && cat "$HOME/fake-openai-ultraplan.ready"',
        "expect_all": ["ready $FAKE_OPENAI_URL"],
    }
    assert scenario["turns"][2] == {
        "send": '!git status --short > "$HOME/ultraplan-status-before.txt" && cat "$HOME/ultraplan-status-before.txt"',
        "expect_all": ["M sample.txt"],
    }
    assert scenario["turns"][3] == {
        "send": "/ultraplan add usage snapshot export",
        "expect_all": [
            "Ultra Plan: add usage snapshot export",
            "ultraplan-fixture-plan: update koder_agent/core/usage_tracker.py",
        ],
        "expect_not": ["ultraplan-secret-token"],
    }
    assert "You are a senior architect" in scenario["turns"][4]["send"]
    assert (
        "Create a comprehensive implementation plan for: add usage snapshot export"
        in scenario["turns"][4]["send"]
    )
    assert scenario["turns"][5]["expect_all"] == ["ultraplan-status-unchanged"]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/fake-openai-ultraplan.log",
                [
                    "/v1/chat/completions",
                    "You are a senior architect",
                    "Create a comprehensive implementation plan for: add usage snapshot export",
                ],
            ]
        },
        {"file_not_contains": ["$HOME/fake-openai-ultraplan.log", "ultraplan-secret-token"]},
        {"file_contains": ["$HOME/ultraplan-status-after.txt", "M sample.txt"]},
    ]


def test_files_scenario_is_acceptance_backed_by_seeded_context_and_missing_edge():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["files"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {"send": "/files", "expect_all": ["No files in context"]}
    assert any(
        turn.get("send", "").startswith('!uv --project "$KODER_SCENARIO_SOURCE_ROOT"')
        and "files-fixture" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    listing_turn = next(
        turn
        for turn in scenario["turns"]
        if turn.get("send") == "/files" and len(turn.get("expect_all", [])) > 1
    )
    assert "- AGENTS.md (exists)" in listing_turn["expect_all"]
    assert "- docs/runtime-notes.md (exists)" in listing_turn["expect_all"]
    assert "- missing-context.md (missing)" in listing_turn["expect_all"]
    assert "AGENTS.md (exists)\n- AGENTS.md (exists)" in listing_turn["expect_not"]
    assert any(
        turn.get("send") == "!test ! -f missing-context.md && cat files-seed.txt"
        and "seeded files context" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/files-seed.txt", "seeded files context"]},
        {"path_not_exists": "$REPO/missing-context.md"},
    ]


def test_cost_scenario_is_acceptance_backed_by_usage_snapshots_and_model_costs():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["cost"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/cost",
        "expect_all": [
            "requests: 0",
            "input_tokens: 0",
            "output_tokens: 0",
            "context_tokens: 0",
            "cost: 0.0000",
        ],
    }
    assert scenario["turns"][1]["send"] == (
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python '
        '"$KODER_SCENARIO_SOURCE_ROOT/scripts/seed_tmux_usage_fixture.py" known'
    )
    assert scenario["turns"][1]["expect_all"] == ["known-usage-fixture"]
    assert scenario["turns"][2] == {
        "send": "/cost",
        "expect_all": [
            "requests: 2",
            "input_tokens: 1500",
            "output_tokens: 2100",
            "context_tokens: 4900",
            "cost: 0.0198",
        ],
    }
    assert scenario["turns"][4]["expect_all"] == ["unknown-usage-fixture"]
    assert scenario["turns"][5] == {
        "send": "/cost",
        "expect_all": [
            "requests: 1",
            "input_tokens: 100",
            "output_tokens: 50",
            "context_tokens: 150",
            "cost: 0.0000",
        ],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/known-usage-seed.txt", "seeded known usage snapshot"]},
        {"file_contains": ["$REPO/unknown-usage-seed.txt", "seeded unknown usage snapshot"]},
        {
            "file_glob_contains": [
                "$HOME/.koder/usage/*.json",
                [
                    '"request_count": 1',
                    '"input_tokens": 100',
                    "totally-unknown-model-xyz-99999",
                ],
            ]
        },
    ]


def test_usage_scenario_is_acceptance_backed_by_persisted_usage_and_clear_edge():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["usage"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"] == "/usage"
    assert {
        "requests: 0",
        "input_tokens: 0",
        "output_tokens: 0",
        "last_input_tokens: 0",
        "last_output_tokens: 0",
        "context_tokens: 0",
        "cost: 0.0000",
        "rate_limit_status: unknown",
    } <= set(scenario["turns"][0]["expect_all"])
    assert scenario["turns"][1]["send"] == (
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python '
        '"$KODER_SCENARIO_SOURCE_ROOT/scripts/seed_tmux_usage_fixture.py" known'
    )
    assert scenario["turns"][1]["expect_all"] == ["known-usage-fixture"]
    assert {
        "requests: 2",
        "input_tokens: 1500",
        "output_tokens: 2100",
        "last_input_tokens: 500",
        "last_output_tokens: 100",
        "context_tokens: 4900",
        "cost: 0.0198",
        "rate_limit_status: unknown",
    } <= set(scenario["turns"][2]["expect_all"])
    assert scenario["turns"][4] == {
        "send": "/clear",
        "expect_all": ["Switched to session:"],
    }
    assert scenario["turns"][5]["send"] == "/usage"
    assert "requests: 0" in scenario["turns"][5]["expect_all"]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/known-usage-seed.txt", "seeded known usage snapshot"]},
        {
            "file_glob_contains": [
                "$HOME/.koder/usage/*.json",
                ['"request_count": 2', '"input_tokens": 1500', '"output_tokens": 2100', "gpt-4.1"],
            ]
        },
    ]


def test_insights_scenario_is_acceptance_backed_by_seeded_session_analytics():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["insights"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/insights",
        "expect_all": [
            "Session Insights:",
            "Transcript items: 0",
            "Messages: 0",
            "User messages: 0",
            "Assistant messages: 0",
            "Tool results: 0",
            "Tool calls: 0",
            "Files in context: 0",
            "Requests: 0",
            "Input tokens: 0",
            "Output tokens: 0",
            "Total cost: $0.0000",
        ],
    }
    assert any(
        turn.get("send", "").startswith('!uv --project "$KODER_SCENARIO_SOURCE_ROOT"')
        and "insights-fixture-ok" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    seeded_turn = scenario["turns"][2]
    assert seeded_turn["send"] == "/insights"
    for expected in [
        "Transcript items: 4",
        "Messages: 3",
        "User messages: 1",
        "Assistant messages: 1",
        "Tool results: 1",
        "Tool calls: 1",
        "Files in context: 2",
        "- AGENTS.md",
        "- docs/runtime-notes.md",
        "Total cost: $0.0000",
    ]:
        assert expected in seeded_turn["expect_all"]
    assert scenario["turns"][3] == {
        "send": "/debug-tool-call",
        "expect_all": [
            "debug-tool-call: 2 recorded item(s)",
            "call read_file",
            "output read_file",
        ],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/insights-seed.txt", "seeded insights session"]}
    ]


def test_summary_scenario_is_acceptance_backed_by_session_usage_and_git_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["summary"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/rename summary-fixture",
        "expect_all": ["Session renamed to: summary-fixture"],
    }
    summary_turn = scenario["turns"][1]
    assert summary_turn["send"] == "/summary"
    assert summary_turn["capture"] == "visible"
    for expected in [
        "Session Summary:",
        "Title: summary-fixture",
        "Requests: 0",
        "Tokens: 0 in / 0 out",
        "Uncommitted changes:",
        "sample.txt | 1 +",
        "1 file changed, 1 insertion(+)",
        "Recent commits:",
        "initial",
    ]:
        assert expected in summary_turn["expect_all"]
    assert scenario["post_assertions"] == [
        {
            "sqlite_contains": [
                "$HOME/.koder/koder.db",
                "select title from session_metadata where title = 'summary-fixture'",
                "summary-fixture",
            ]
        },
        {"file_contains": ["$REPO/sample.txt", "changed"]},
    ]


def test_thinkback_scenario_is_acceptance_backed_by_seeded_local_session():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["thinkback"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/thinkback nope",
        "expect_all": ["Usage: /thinkback [recent-turn-count]"],
    }
    assert scenario["turns"][1] == {
        "send": "/rename thinkback-fixture",
        "expect_all": ["Session renamed to: thinkback-fixture"],
    }
    seed_turn = scenario["turns"][2]
    assert seed_turn["send"].startswith(
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python'
    )
    assert "thinkback user prompt one" in seed_turn["send"]
    assert "thinkback assistant answer two" in seed_turn["send"]
    assert "thinkback tool output one" in seed_turn["send"]
    assert seed_turn["expect_all"] == ["thinkback-fixture"]
    thinkback_turn = scenario["turns"][3]
    assert thinkback_turn["send"] == "/thinkback 2"
    for expected in [
        "thinkback: session review",
        "title: thinkback-fixture",
        "messages: 6",
        "user_turns: 3",
        "assistant_turns: 2",
        "tool_outputs: 1",
        "recent_prompts:",
        "1. thinkback user prompt two",
        "2. thinkback user prompt three",
    ]:
        assert expected in thinkback_turn["expect_all"]
    assert "AuthenticationError" in thinkback_turn["expect_not"]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/thinkback-seed.txt", "seeded thinkback turns"]},
        {
            "sqlite_contains": [
                "$HOME/.koder/koder.db",
                "select count(*) from agent_messages where session_id = (select session_id from session_metadata where title = 'thinkback-fixture')",
                "6",
            ]
        },
    ]


def test_thinkback_play_scenario_is_acceptance_backed_by_seeded_replay():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["thinkback-play"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/thinkback-play",
        "expect_all": ["thinkback-play: no session turns available"],
        "expect_not": ["thinking...", "AuthenticationError"],
    }
    assert scenario["turns"][1] == {
        "send": "/thinkback-play nope",
        "expect_all": ["Usage: /thinkback-play [recent-turn-count]"],
    }
    seed_turn = scenario["turns"][2]
    assert seed_turn["send"].startswith(
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python'
    )
    assert "play user first" in seed_turn["send"]
    assert "play assistant second" in seed_turn["send"]
    assert "play tool third" in seed_turn["send"]
    assert seed_turn["expect_all"] == ["thinkback-play-fixture"]
    replay_turn = scenario["turns"][3]
    assert replay_turn["send"] == "/thinkback-play 3"
    for expected in [
        "thinkback-play: replaying 3 turn(s)",
        "assistant: play assistant second",
        "tool: play tool third",
        "user: play user fourth",
    ]:
        assert expected in replay_turn["expect_all"]
    assert "AuthenticationError" in replay_turn["expect_not"]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/thinkback-play-seed.txt", "seeded thinkback-play turns"]},
        {
            "sqlite_contains": [
                "$HOME/.koder/koder.db",
                "select count(*) from agent_messages where message_data like '%play user%' or message_data like '%play assistant%' or message_data like '%play tool%'",
                "4",
            ]
        },
    ]


def test_version_scenario_is_acceptance_backed_by_cli_version_contract():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["version"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"KODER_BUILD_TIME": "scenario-build"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/version",
        "expect_all": [
            "version: $RUNTIME_VERSION",
            "package: koder",
            "source: installed-package",
            "build_time: scenario-build",
            "cli_banner: $RUNTIME_VERSION (Koder)",
        ],
        "expect_not": ["python:"],
    }
    assert any(
        turn.get("send") == "/status" and "Runtime slash commands" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1] == {
        "send": (
            '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync --no-env-file koder --version '
            "| tee version-proof.txt && printf '%s%s\\n' 'version-cli-' 'finished'"
        ),
        "capture": "visible",
        "expect_all": ["$RUNTIME_VERSION (Koder)", "version-cli-finished"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/version-proof.txt", "$RUNTIME_VERSION (Koder)"]}
    ]


def test_project_agent_detail_scenario_is_acceptance_backed_by_lifecycle_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["agents"]["project-agent-detail"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert any(
        turn.get("send") == "/agents" and "reviewer" in turn.get("expect_any", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/agents" and "failed_files:" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/agents create project scenario-agent Scenario created agent"
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1] == {
        "send": "/agents show scenario-agent",
        "expect_all": ["agents: not found scenario-agent"],
    }
    assert scenario["post_assertions"] == [
        {"path_exists": "$REPO/.koder/agents/reviewer.md"},
        {"file_contains": ["$REPO/.koder/agents/broken.md", "broken-agent"]},
        {"path_not_exists": "$REPO/.koder/agents/scenario-agent.md"},
    ]


def test_agents_slash_scenario_is_acceptance_backed_by_lifecycle_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["agents"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/agents",
        "expect_any": ["reviewer", "Agents"],
    }
    assert scenario["turns"][1] == {
        "send": "/agents show reviewer",
        "expect_all": [
            "description: Reviews fixture changes",
            "tools: Read, Bash",
            "permission_mode: plan",
        ],
    }
    assert any(
        turn.get("send") == "/agents" and "failed_files:" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/agents create project scenario-agent Scenario created agent"
        and "agents: created" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/agents summary"
        and "agents: no runtime agents" in turn.get("expect_any", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1] == {
        "send": "/agents show scenario-agent",
        "expect_all": ["agents: not found scenario-agent"],
    }
    assert scenario["post_assertions"] == [
        {"path_exists": "$REPO/.koder/agents/reviewer.md"},
        {"file_contains": ["$REPO/.koder/agents/broken.md", "broken-agent"]},
        {"path_not_exists": "$REPO/.koder/agents/scenario-agent.md"},
    ]


def test_assistant_slash_scenario_is_acceptance_backed_by_profile_inspection():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["assistant"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    status_turn = scenario["turns"][0]
    assert status_turn["send"] == "/assistant"
    for expected in [
        "assistant:",
        "active_profile:",
        "model:",
        "provider:",
        "session_id:",
        "project_agents_dir:",
        "user_agents_dir:",
        "related_commands: /agents, /model, /session, /skills",
    ]:
        assert expected in status_turn["expect_all"]
    assert scenario["turns"][1] == {
        "send": "/assistant list",
        "expect_all": [
            "assistant_profiles:",
            "reviewer [projectSettings] model=sonnet",
            "general-purpose [built-in]",
        ],
    }
    assert scenario["turns"][2] == {
        "send": "/assistant show reviewer",
        "expect_all": [
            "agents: reviewer",
            "description: Reviews fixture changes",
            "tools: Read, Bash",
            "permission_mode: plan",
        ],
    }
    assert scenario["turns"][-2] == {
        "send": "/assistant show missing-profile",
        "expect_all": ["assistant: profile not found missing-profile"],
    }
    assert scenario["turns"][-1] == {
        "send": "/assistant nope",
        "expect_all": ["Usage: /assistant [list|show <agent-name>]"],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$REPO/.koder/agents/reviewer.md",
                [
                    "name: reviewer",
                    "description: Reviews fixture changes",
                    "permissionMode: plan",
                ],
            ]
        }
    ]


def test_tmux_pane_scenario_asserts_worker_pane_output():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    tmux_pane = manifest["teams"]["tmux-pane"]

    assert tmux_pane["validation_level"] == "acceptance"
    assert tmux_pane["acceptance_criteria"]
    assert tmux_pane["acceptance_artifacts"]
    spawn_turn = tmux_pane["turns"][1]
    assert spawn_turn["expect_tmux_panes_min"] == 2
    assert spawn_turn["expect_tmux_any_pane_all"] == [
        "Command Response",
        "model: gpt-4.1",
        "provider:",
    ]
    second_spawn_turn = tmux_pane["turns"][2]
    assert second_spawn_turn["expect_tmux_panes_min"] == 3
    assert second_spawn_turn["expect_tmux_any_pane_all"] == [
        "Command Response",
        "version:",
    ]
    show_turn = tmux_pane["turns"][3]
    assert show_turn["expect_all"] == [
        "team_id: pane-scenario",
        "member_count: 2",
        "name=pane-worker-a",
        "name=pane-worker-b",
        "pane_state=dead",
    ]
    assert tmux_pane["turns"][4] == {
        "kill_tmux_pane_matching": "model: gpt-4.1",
        "expect_tmux_panes_min": 2,
    }
    assert tmux_pane["turns"][5]["expect_all"] == [
        "team_id: pane-scenario",
        "member_count: 2",
        "name=pane-worker-a",
        "name=pane-worker-b",
        "pane_state=missing",
        "pane_state=dead",
    ]


def test_mailbox_and_task_scenario_is_acceptance_backed_by_state_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["teams"]["mailbox-and-task"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert any(
        turn.get("send") == "/peers inbox team-scenario worker-a --consume"
        and "peers: inbox empty" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["post_assertions"] == [
        {"path_exists": "$HOME/.koder/teams/team-scenario/config.json"},
        {
            "file_contains": [
                "$HOME/.koder/teams/team-scenario/inboxes/worker-a.json",
                ['"content": "team-message"', '"read": true'],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/teams/team-scenario/history.jsonl",
                [
                    '"event": "message_sent"',
                    '"event": "message_read"',
                    '"recipient": "worker-a"',
                ],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/tasks/team-scenario/1.json",
                [
                    '"subject": "check mailbox"',
                    '"status": "completed"',
                    '"owner": "worker-a"',
                ],
            ]
        },
    ]


def test_peers_slash_scenario_is_acceptance_backed_by_mailbox_and_task_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["peers"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"] == "/peers create peers-scenario"
    assert "effective_teammate_mode:" in scenario["turns"][0]["expect_all"]
    assert scenario["turns"][1] == {
        "send": "/peers show peers-scenario",
        "expect_all": [
            "peers: peers-scenario",
            "team_id: peers-scenario",
            "member_count: 0",
            "task_count: 0",
            "config_path:",
        ],
    }
    assert any(
        turn.get("send") == "/peers inbox peers-scenario worker-a --consume"
        and "peers: inbox consumed" in turn.get("expect_all", [])
        and "read: true" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/peers inbox peers-scenario worker-a --consume"
        and "peers: inbox empty" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/peers history peers-scenario"
        and "sent team-lead -> worker-a: peer-message" in turn.get("expect_all", [])
        and "read worker-a <= team-lead" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-1] == {
        "send": "/peers task list peers-scenario",
        "expect_all": ["- 1: verify mailbox status=completed owner=worker-a"],
    }
    assert scenario["post_assertions"] == [
        {"path_exists": "$HOME/.koder/teams/peers-scenario/config.json"},
        {
            "file_contains": [
                "$HOME/.koder/teams/peers-scenario/inboxes/worker-a.json",
                ['"content": "peer-message"', '"read": true'],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/teams/peers-scenario/history.jsonl",
                [
                    '"event": "message_sent"',
                    '"event": "message_read"',
                    '"recipient": "worker-a"',
                ],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/tasks/peers-scenario/1.json",
                [
                    '"subject": "verify mailbox"',
                    '"status": "completed"',
                    '"owner": "worker-a"',
                ],
            ]
        },
    ]


def test_team_memory_scenario_is_acceptance_backed_by_content_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["teams"]["team-memory"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][-1]["expect_tmux_any_pane_all"] == [
        "Shell Mode",
        "shared-team-note",
    ]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/.koder/team-memory/memory-team/MEMORY.md", "shared-team-note"]},
        {
            "file_contains": [
                "$HOME/.koder/teams/memory-team/memory/MEMORY.md",
                "shared-team-note",
            ]
        },
    ]


def test_tmux_discussion_scenario_is_acceptance_backed_by_history_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["teams"]["tmux-discussion"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["teammate_mode"] == "tmux"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][1]["expect_tmux_any_pane_all"] == [
        "Command Response",
        "recipient: team-lead",
        "sender: proposer-a",
    ]
    assert scenario["turns"][2]["expect_tmux_any_pane_all"] == [
        "Command Response",
        "recipient: team-lead",
        "sender: proposer-b",
    ]
    assert scenario["turns"][-1]["expect_all"] == [
        "peers: history",
        "sent proposer-a -> team-lead:",
        "PROPOSAL_A",
        "sent proposer-b -> team-lead:",
        "PROPOSAL_B",
        "read team-lead <= proposer-a",
        "read team-lead <= proposer-b",
    ]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/teams/discussion-team/config.json",
                ['"name": "proposer-a"', '"name": "proposer-b"', '"mode": "tmux"'],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/teams/discussion-team/history.jsonl",
                [
                    '"sender": "proposer-a"',
                    '"sender": "proposer-b"',
                    '"content": "PROPOSAL_A"',
                    '"content": "PROPOSAL_B"',
                    '"event": "message_read"',
                ],
            ]
        },
    ]


def test_in_process_discussion_scenario_is_acceptance_backed_by_history_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["teams"]["in-process-discussion"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["teammate_mode"] == "in-process"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["expect_all"] == [
        "peers: created",
        "team_id: inproc-discussion",
        "teammate_mode: in-process",
        "effective_teammate_mode: in-process",
    ]
    assert any(
        turn.get("send")
        == "/peers send inproc-discussion proposer-a /peers send inproc-discussion team-lead PROPOSAL_A"
        for turn in scenario["turns"]
    )
    assert scenario["turns"][-2]["expect_all"] == [
        "peers: history",
        "sent team-lead -> proposer-a:",
        "read proposer-a <= team-lead",
        "sent proposer-a -> team-lead: PROPOSAL_A",
        "sent proposer-b -> team-lead: PROPOSAL_B",
        "sent coordinator -> team-lead: SUMMARY_CHOOSE_A_AND_B",
        "run proposer-a state=completed source=mailbox",
        "run coordinator state=completed source=mailbox",
    ]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/teams/inproc-discussion/config.json",
                [
                    '"name": "coordinator"',
                    '"name": "proposer-a"',
                    '"name": "proposer-b"',
                ],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/teams/inproc-discussion/inboxes/proposer-a.json",
                [
                    '"content": "/peers send inproc-discussion team-lead PROPOSAL_A"',
                    '"read": true',
                ],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/teams/inproc-discussion/history.jsonl",
                [
                    '"sender": "proposer-a"',
                    '"sender": "proposer-b"',
                    '"sender": "coordinator"',
                    '"content": "PROPOSAL_A"',
                    '"content": "PROPOSAL_B"',
                    '"content": "SUMMARY_CHOOSE_A_AND_B"',
                    '"event": "message_read"',
                    '"event": "run_completed"',
                    '"source": "mailbox"',
                ],
            ]
        },
        {
            "file_glob_contains": [
                "$HOME/.koder/agents/agent-*.json",
                [
                    '"description": "Teammate: coordinator"',
                    '"description": "Teammate: proposer-a"',
                    '"description": "Teammate: proposer-b"',
                    '"state": "completed"',
                ],
            ]
        },
    ]


def test_background_fork_scenario_is_acceptance_backed_by_config_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["agents"]["background-fork"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_API_KEY": "scenario-secret-key",
        "KODER_BASE_URL": "https://scenario-base.invalid/v1",
        "KODER_REASONING_EFFORT": "high",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["expect_all"] == [
        "fork: launched background subagent",
        "agent_type: general-purpose",
        "context_mode: isolated",
        "model_config:",
        "model_override: inherit",
        "model_name: gpt-4.1",
        "base_url: https://scenario-base.invalid/v1",
        "api_key_present: True",
        "reasoning_effort: high",
    ]
    assert scenario["turns"][0]["expect_not"] == ["scenario-secret-key"]
    assert scenario["turns"][1]["expect_all"][:3] == [
        "fork: launched background subagent",
        "agent_type: general-purpose",
        "context_mode: fork",
    ]
    assert scenario["post_assertions"] == [
        {"path_glob_exists": "$HOME/.koder/agents/agent-*.json"},
        {
            "file_glob_contains": [
                "$HOME/.koder/agents/agent-*.json",
                [
                    '"model_config"',
                    '"model_override": "inherit"',
                    '"model_name": "gpt-4.1"',
                    '"base_url": "https://scenario-base.invalid/v1"',
                    '"api_key_present": true',
                    '"reasoning_effort": "high"',
                ],
            ]
        },
        {
            "file_glob_not_contains": [
                "$HOME/.koder/agents/agent-*.json",
                "scenario-secret-key",
            ]
        },
    ]


def test_fork_slash_scenario_is_acceptance_backed_by_config_assertions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["fork"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_API_KEY": "scenario-secret-key",
        "KODER_BASE_URL": "https://scenario-base.invalid/v1",
        "KODER_REASONING_EFFORT": "high",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/fork",
        "expect_all": ["fork: provide a prompt to run in a background subagent."],
    }
    isolated_turn = scenario["turns"][1]
    assert isolated_turn["send"] == "/fork general-purpose inspect fixture"
    assert isolated_turn["expect_all"] == [
        "fork: launched background subagent",
        "agent_type: general-purpose",
        "context_mode: isolated",
        "model_config:",
        "model_override: inherit",
        "model_name: gpt-4.1",
        "base_url: https://scenario-base.invalid/v1",
        "api_key_present: True",
        "reasoning_effort: high",
    ]
    assert isolated_turn["expect_not"] == ["scenario-secret-key"]
    fork_turn = scenario["turns"][2]
    assert fork_turn["send"].startswith("/fork --context fork general-purpose")
    assert "context_mode: fork" in fork_turn["expect_all"]
    assert "scenario-secret-key" in fork_turn["expect_not"]
    assert scenario["turns"][-1] == {
        "send": "/agents summary",
        "expect_all": ["agents: runtime summaries", "general-purpose"],
        "expect_any": ["Working:", "Completed:", "Failed:"],
    }
    assert scenario["post_assertions"] == [
        {"path_glob_exists": "$HOME/.koder/agents/agent-*.json"},
        {
            "file_glob_contains": [
                "$HOME/.koder/agents/agent-*.json",
                [
                    '"model_config"',
                    '"model_override": "inherit"',
                    '"model_name": "gpt-4.1"',
                    '"base_url": "https://scenario-base.invalid/v1"',
                    '"api_key_present": true',
                    '"reasoning_effort": "high"',
                ],
            ]
        },
        {
            "file_glob_not_contains": [
                "$HOME/.koder/agents/agent-*.json",
                "scenario-secret-key",
            ]
        },
    ]


def test_permissions_and_sandbox_scenario_is_acceptance_backed_by_decisions():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["permissions-and-sandbox"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert any(
        turn.get("send") == "/permissions check run_shell touch blocked.txt"
        and "requires_approval: true" in turn.get("expect_all", [])
        and "sandboxed shell command auto-allowed" in turn.get("expect_not", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/permissions check run_shell touch blocked.txt"
        and "requires_approval: true" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/permissions check run_shell rg TODO ."
        and "allowed: true" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$REPO/.koder/settings.local.json",
                [
                    '"enabled": false',
                    '"backend": "unix-local"',
                    '"excludedCommands"',
                    '"touch *"',
                ],
            ]
        }
    ]


def test_permission_slash_commands_are_acceptance_backed_by_runtime_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)

    for command in ("add-dir", "permissions", "sandbox"):
        scenario = manifest["slash_commands"][command]
        assert scenario["validation_level"] == "acceptance"
        assert scenario["acceptance_criteria"]
        assert scenario["acceptance_artifacts"]

    add_dir = manifest["slash_commands"]["add-dir"]
    assert any(
        turn.get("send") == "/add-dir /tmp"
        and "Added /private/tmp as a working directory" in turn.get("expect_all", [])
        for turn in add_dir["turns"]
    )
    assert any(
        turn.get("send") == "/permissions"
        and "working_directories: 2" in turn.get("expect_all", [])
        for turn in add_dir["turns"]
    )
    assert any(
        turn.get("send") == "/add-dir sample.txt"
        and "Did you mean to add the parent directory" in turn.get("expect_all", [])
        for turn in add_dir["turns"]
    )
    assert any(
        turn.get("send") == "/add-dir /tmp/koder-missing-directory-for-scenario"
        and "was not found." in turn.get("expect_all", [])
        for turn in add_dir["turns"]
    )

    permissions = manifest["slash_commands"]["permissions"]
    assert any(
        turn.get("send") == "/permissions check run_shell touch blocked.txt"
        and "requires_approval: true" in turn.get("expect_all", [])
        for turn in permissions["turns"]
    )
    assert any(
        turn.get("send") == "/permissions check run_shell touch blocked.txt"
        and "requires_approval: true" in turn.get("expect_all", [])
        and "sandboxed shell command auto-allowed" in turn.get("expect_not", [])
        for turn in permissions["turns"]
    )
    assert any(
        turn.get("send") == "/permissions check run_shell rg TODO ."
        and "allowed: true" in turn.get("expect_all", [])
        for turn in permissions["turns"]
    )
    assert {
        "file_contains": [
            "$REPO/.koder/settings.local.json",
            ['"enabled": true', '"backend": "unix-local"'],
        ]
    } in permissions["post_assertions"]

    sandbox = manifest["slash_commands"]["sandbox"]
    assert any(
        turn.get("send") == "/sandbox" and "sandbox_enabled: false" in turn.get("expect_all", [])
        for turn in sandbox["turns"]
    )
    assert any(
        turn.get("send") == "/sandbox enable"
        and "Available sandbox backends:" in turn.get("expect_all", [])
        for turn in sandbox["turns"]
    )
    assert any(
        turn.get("send") == "/sandbox exclude touch *"
        and "excluded_commands: 1" in turn.get("expect_all", [])
        for turn in sandbox["turns"]
    )
    assert {
        "file_contains": [
            "$REPO/.koder/settings.local.json",
            [
                '"enabled": false',
                '"backend": "unix-local"',
                '"excludedCommands"',
                '"touch *"',
            ],
        ]
    } in sandbox["post_assertions"]


def test_plan_slash_scenario_is_acceptance_backed_by_permission_checks():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["plan"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"] == "/plan"
    assert "permission_mode: plan" in scenario["turns"][0]["expect_all"]
    assert any(
        turn.get("send") == "/permissions check write_file plan-output.txt"
        and "plan mode: mutations not allowed" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/permissions check read_file sample.txt"
        and "read-only tool allowed in plan mode" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/plan" and "permission_mode: default" in turn.get("expect_all", [])
        for turn in scenario["turns"][1:]
    )
    assert {"path_not_exists": "$REPO/plan-output.txt"} in scenario["post_assertions"]


def test_help_slash_scenario_is_acceptance_backed_by_command_catalog():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["help"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"] == "/help"
    assert "Command Catalog" in scenario["turns"][0]["expect_all"]
    assert "List configured runtime hooks" in scenario["turns"][0]["expect_all"]
    assert "Execute /" in scenario["turns"][0]["expect_not"]
    assert any(
        turn.get("send") == "/"
        and "Add a workspace directory to the active session" in turn.get("expect_all", [])
        and "List, inspect, create, and manage local agents" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/help hooks"
        and turn.get("capture") == "visible"
        and "/hooks: List configured runtime hooks" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/help nope"
        and "help: unknown command /nope" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )


def test_exit_slash_scenario_is_acceptance_backed_by_dead_session_assertion():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["exit"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"] == "/status"
    assert scenario["turns"][1]["send"] == "/session"
    assert scenario["turns"][-1] == {"send": "/exit", "expect_session_dead": True}


def test_rename_slash_scenario_is_acceptance_backed_by_sqlite_metadata():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["rename"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/rename",
        "expect_all": ["Could not generate a name", "Usage: /rename <new title>"],
    }
    assert any(
        turn.get("send") == "/session" and "title: scenario-title" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select title from session_metadata where title = 'scenario-title'",
            "scenario-title",
        ]
    } in scenario["post_assertions"]


def test_color_slash_scenario_is_acceptance_backed_by_visible_and_sqlite_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["color"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/color",
        "expect_all": ["Please provide a color", "Available colors:"],
    }
    assert any(
        turn.get("send") == "/color chartreuse"
        and 'Invalid color "chartreuse"' in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/session" and "color: red" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/output-style" and "color: red" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/output-style" and "color: default" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select coalesce(color, '<none>') from session_metadata order by updated_at desc limit 1",
            "<none>",
        ]
    } in scenario["post_assertions"]


def test_session_slash_scenario_is_acceptance_backed_by_visible_and_sqlite_metadata():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["session"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"] == "/session"
    assert "hint: use /resume to switch sessions" in scenario["turns"][0]["expect_all"]
    assert any(turn.get("send") == "/rename session-check" for turn in scenario["turns"])
    assert any(
        turn.get("send") == "/session" and "title: session-check" in turn.get("expect_all", [])
        for turn in scenario["turns"][1:]
    )
    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select title from session_metadata where title = 'session-check'",
            ["session-check"],
        ]
    } in scenario["post_assertions"]


def test_resume_slash_scenario_is_acceptance_backed_by_existing_session_resolution():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["resume"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["send"].startswith("!uv --project")
    assert scenario["turns"][0]["timeout"] >= 45
    assert any(
        turn.get("send") == "/resume missing-session"
        and "Session missing-session was not found." in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/resume duplicate-title"
        and "Found 2 sessions matching duplicate-title" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/resume resume-title"
        and "Switched to session: resume-target" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/resume resume-duplicate-one"
        and "Switched to session: resume-duplicate-one" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select title from session_metadata where session_id in ('resume-target', 'resume-duplicate-one', 'resume-duplicate-two') order by session_id",
            ["resume-title", "duplicate-title"],
        ]
    } in scenario["post_assertions"]


def test_backfill_sessions_scenario_is_acceptance_backed_by_legacy_migration():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["backfill-sessions"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "create table if not exists ctx" in scenario["turns"][0]["send"]
    assert any(
        turn.get("send") == "/backfill-sessions"
        and "migrated: 1" in turn.get("expect_all", [])
        and "legacy-title" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/backfill-sessions" and "migrated: 0" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(
        turn.get("send") == "/resume legacy-title"
        and "Switched to session: legacy-session" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select title from session_metadata where session_id = 'legacy-session'",
            "legacy-title",
        ]
    } in scenario["post_assertions"]
    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select migrated_sessions from migration_status",
            "1",
        ]
    } in scenario["post_assertions"]


def test_memory_and_session_scenario_is_acceptance_backed_by_durable_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["memory-and-session"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert any(
        turn.get("send") == "/remember alpha-memory durable marker"
        and "remember: saved" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert any(turn.get("send") == "/clear" for turn in scenario["turns"])
    assert any(turn.get("send") == "/resume feature-session" for turn in scenario["turns"])
    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select title from session_metadata where title = 'feature-session'",
            "feature-session",
        ]
    } in scenario["post_assertions"]


def test_slash_completion_scenario_is_acceptance_backed_by_filtered_menu():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["slash-completion"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    first_turn = scenario["turns"][0]
    assert first_turn["type"] == "/sta"
    assert first_turn["capture"] == "visible"
    assert {"/status", "/statusline"} <= set(first_turn["expect_all"])
    assert "/help" in first_turn["expect_not"]
    assert any(turn.get("resize") == {"width": 72, "height": 13} for turn in scenario["turns"])
    assert scenario["turns"][-1]["keys"] == ["Enter"]
    assert "Runtime slash commands" in scenario["turns"][-1]["expect_all"]


def test_terminal_resize_reflow_scenario_is_acceptance_backed_by_prompt_survival():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["terminal-resize-reflow"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert any(turn.get("resize") == {"width": 60, "height": 12} for turn in scenario["turns"])
    assert any(turn.get("type") == "resize-survives" for turn in scenario["turns"])
    assert all("Window too small" in turn.get("expect_not", []) for turn in scenario["turns"])


def test_settings_bundle_scenario_is_acceptance_backed_by_bundle_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["settings-bundle"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]

    sends = [turn.get("send", "") for turn in scenario["turns"]]
    assert any("koder config export" in send and "--scope project" in send for send in sends)
    assert any("koder config import" in send and "--dry-run" in send for send in sends)
    assert any("test ! -f" in send and "settings-bundle-dry-run-ok" in send for send in sends)
    assert any(
        "config.yaml.bak-*" in send and "settings-bundle-backup-ok" in send for send in sends
    )
    assert scenario["turns"][-1]["send"] == "/config"
    assert "gpt-4.1" in scenario["turns"][-1]["expect_all"]

    assert {
        "file_not_contains": [
            "$HOME/.koder/settings-bundle-scenario.json",
            [
                "SECRET_TOKEN_SHOULD_NOT_EXPORT",
                "CACHE_SECRET_SHOULD_NOT_EXPORT",
                '"role": "tokens"',
                '"role": "cache"',
            ],
        ]
    } in scenario["post_assertions"]
    assert {
        "file_not_contains": [
            "$HOME/.koder/project-bundle-scenario.json",
            ['"role": "user_config"', '"role": "user_settings"', "user memory marker"],
        ]
    } in scenario["post_assertions"]
    assert {"path_glob_exists": "$HOME/.koder/config.yaml.bak-*"} in scenario["post_assertions"]


def test_auto_dream_task_scenario_is_acceptance_backed_by_runtime_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["auto-dream-task"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "run_auto_dream_from_messages" in scenario["turns"][0]["send"]
    assert '"type":"project"' in scenario["turns"][0]["send"]
    assert '"type":"user"' in scenario["turns"][0]["send"]
    assert {
        "auto-dream-runtime-ok",
        "task_id=1",
        "memories=2",
        "saved=",
        "project_scope_ok=True",
        "user_scope_ok=True",
        "cross_project_retrieved=False",
        "user_cross_project_retrieved=True",
    } <= set(scenario["turns"][0]["expect_all"])
    assert scenario["turns"][1]["send"] == "/tasks"
    assert {"auto-dream/1", "status=completed", "memories=2", "saved="} <= set(
        scenario["turns"][1]["expect_all"]
    )
    assert "broken.json" in scenario["turns"][2]["send"]
    assert {"auto-dream/malformed: broken.json", "memories=3", "errors=1"} <= set(
        scenario["turns"][-1]["expect_all"]
    )
    assert {
        "file_glob_contains": [
            "$HOME/auto-dream-project-a/.koder/memory/auto-dream-project-*.md",
            ["type: project", "storage_scope: project", "projectonlyzxq"],
        ]
    } in scenario["post_assertions"]
    assert {
        "file_glob_contains": [
            "$HOME/.koder/memory/auto-dream-user-*.md",
            ["type: user", "storage_scope: user", "userglobalqvx"],
        ]
    } in scenario["post_assertions"]
    assert {"file_glob_not_contains": ["$HOME/.koder/memory/*.md", "projectonlyzxq"]} in scenario[
        "post_assertions"
    ]
    assert {"path_not_exists": "$HOME/auto-dream-project-b/.koder/memory"} in scenario[
        "post_assertions"
    ]
    assert {
        "file_contains": [
            "$HOME/.koder/tasks/auto-dream/broken.json",
            "{not json",
        ]
    } in scenario["post_assertions"]


def test_tasks_slash_scenario_is_acceptance_backed_by_task_rows_and_errors():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["tasks"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/tasks",
        "expect_all": ["No runtime tasks tracked."],
    }
    assert "task-scenario.json" in scenario["turns"][1]["send"]
    assert "broken.json" in scenario["turns"][1]["send"]
    assert set(scenario["turns"][-1]["expect_all"]) >= {
        "auto-dream/malformed: broken.json",
        "auto-dream/task-scenario",
        "AutoDream memory consolidation",
        "status=completed",
        "memories=3",
        "errors=1",
    }
    assert {
        "file_contains": [
            "$HOME/.koder/tasks/auto-dream/task-scenario.json",
            ['"id":"task-scenario"', '"memories_written":3', "scenario warning"],
        ]
    } in scenario["post_assertions"]
    assert {
        "file_contains": [
            "$HOME/.koder/tasks/auto-dream/broken.json",
            "{not json",
        ]
    } in scenario["post_assertions"]


def test_model_config_status_effort_and_env_scenarios_are_acceptance_backed():
    manifest = _load_manifest(DEFAULT_MANIFEST)

    for command in ("model", "config", "status", "effort", "env"):
        scenario = manifest["slash_commands"][command]
        assert scenario["validation_level"] == "acceptance"
        assert scenario["acceptance_criteria"]
        assert scenario["acceptance_artifacts"]

    model = manifest["slash_commands"]["model"]
    assert any(
        turn.get("send") == "/model anthropic/claude-sonnet-4-6"
        and "effective_model: litellm/anthropic/claude-sonnet-4-6" in turn.get("expect_all", [])
        for turn in model["turns"]
    )
    assert any(
        turn.get("send") == "/fork general-purpose model inheritance marker"
        and "model_name: litellm/anthropic/claude-sonnet-4-6" in turn.get("expect_all", [])
        for turn in model["turns"]
    )
    assert {
        "file_glob_contains": [
            "$HOME/.koder/agents/agent-*.json",
            [
                '"model_name": "litellm/anthropic/claude-sonnet-4-6"',
                '"model_override": "inherit"',
            ],
        ]
    } in model["post_assertions"]

    effort = manifest["slash_commands"]["effort"]
    assert any(
        turn.get("send") == "/fork general-purpose effort inheritance marker"
        and "reasoning_effort: max" in turn.get("expect_all", [])
        for turn in effort["turns"]
    )
    assert any(turn.get("send") == "/effort xhigh" for turn in effort["turns"])
    assert any(turn.get("send") == "/effort max" for turn in effort["turns"])
    assert any(
        turn.get("send") == "/effort impossible"
        and "Invalid argument: impossible" in turn.get("expect_all", [])
        for turn in effort["turns"]
    )
    assert {
        "file_glob_contains": [
            "$HOME/.koder/agents/agent-*.json",
            '"reasoning_effort": "max"',
        ]
    } in effort["post_assertions"]

    env = manifest["slash_commands"]["env"]
    assert any(turn.get("send") == "/env DEMO_ENV=hello" for turn in env["turns"])
    assert any(
        turn.get("send") == "/env 1BAD=value"
        and "env: invalid variable name: 1BAD" in turn.get("expect_all", [])
        for turn in env["turns"]
    )
    assert {"path_glob_exists": "$HOME/.koder/session-env/*.sh"} in env["post_assertions"]

    config = manifest["slash_commands"]["config"]
    assert any(
        turn.get("send") == "/config" and "reasoning_effort: high" in turn.get("expect_all", [])
        for turn in config["turns"]
    )
    assert {
        "file_contains": [
            "$HOME/.koder/config.yaml",
            ["name: gpt-4.1", "provider: openai", "reasoning_effort: medium"],
        ]
    } in config["post_assertions"]

    status = manifest["slash_commands"]["status"]
    assert any(
        turn.get("send") == "/status"
        and "Model: litellm/anthropic/claude-sonnet-4-6" in turn.get("expect_all", [])
        for turn in status["turns"]
    )


def test_skills_slash_scenario_is_acceptance_backed_by_source_precedence_and_plugin_reload():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["skills"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"KODER_SKILLS_MARKER": "skills-user-fixture-ok"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    seed_turn = scenario["turns"][0]
    assert seed_turn["send"].startswith('!mkdir -p "$HOME/.koder/skills/demo-skill"')
    assert "User duplicate should be hidden" in seed_turn["send"]
    assert "User skill body." in seed_turn["send"]
    assert seed_turn["expect_all"] == ["skills-user-fixture-ok"]

    initial_listing = scenario["turns"][1]
    assert initial_listing["send"] == "/skills"
    assert initial_listing["capture"] == "visible"
    for expected in [
        "[project] demo-skill",
        "Demo skill for tmux validation",
        "[user] user-skill",
        "User skill fixture",
        "[plugin] demo-plugin:plugin-skill",
        "manual-only",
    ]:
        assert expected in initial_listing["expect_all"]
    assert "User duplicate should be hidden" in initial_listing["expect_not"]

    assert scenario["turns"][2] == {
        "send": "/plugin disable demo-plugin",
        "expect_all": ["Disabled demo-plugin"],
    }
    assert scenario["turns"][3] == {
        "send": "/reload-plugins",
        "expect_all": ["Reloaded 1 plugins."],
    }
    assert scenario["turns"][4] == {
        "send": "/clear",
        "expect_all": ["Switched to session:"],
    }
    hidden_listing = scenario["turns"][5]
    assert hidden_listing["send"] == "/skills"
    assert "- [plugin] demo-plugin:plugin-skill" in hidden_listing["expect_not"]
    assert scenario["turns"][-3] == {
        "send": "/plugin enable demo-plugin",
        "expect_all": ["Enabled demo-plugin"],
    }
    assert scenario["turns"][-2] == {
        "send": "/reload-plugins",
        "expect_all": ["Reloaded 1 plugins."],
    }
    restored_listing = scenario["turns"][-1]
    assert restored_listing["send"] == "/skills"
    assert "[plugin] demo-plugin:plugin-skill" in restored_listing["expect_all"]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/skills/user-skill/SKILL.md",
                ["name: user-skill", "User skill body."],
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/skills/demo-skill/SKILL.md",
                "User duplicate should be hidden",
            ]
        },
        {
            "file_contains": [
                "$HOME/.koder/plugins/state.json",
                ['"demo-plugin"', '"enabled": true'],
            ]
        },
    ]


def test_plugin_slash_scenario_is_acceptance_backed_by_lifecycle_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["plugin"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"] == [
        {
            "send": "/plugin",
            "expect_all": ["- demo-plugin v1.0.0 [user] (enabled) [skills]"],
        },
        {
            "send": "/plugin disable missing-plugin",
            "expect_all": ["Plugin 'missing-plugin' is not installed"],
        },
        {"send": "/plugin disable demo-plugin", "expect_all": ["Disabled demo-plugin"]},
        {
            "send": "/plugin",
            "expect_all": ["- demo-plugin v1.0.0 [user] (disabled) [skills]"],
        },
        {"send": "/plugin enable demo-plugin", "expect_all": ["Enabled demo-plugin"]},
        {
            "send": "/plugin",
            "expect_all": ["- demo-plugin v1.0.0 [user] (enabled) [skills]"],
        },
    ]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/plugins/state.json", ['"demo-plugin"', '"enabled": true']]}
    ]


def test_reload_plugins_slash_scenario_is_acceptance_backed_by_dynamic_plugin_visibility():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["reload-plugins"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_RELOAD_PLUGIN_MARKER": "reload-plugin-fixture-ok",
        "KODER_RELOAD_PLUGIN_REMOVED": "reload-plugin-removed-ok",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/reload-plugins",
        "expect_all": ["Reloaded 1 plugins."],
    }
    seed_turn = scenario["turns"][1]
    assert seed_turn["send"].startswith('!mkdir -p "$HOME/.koder/plugins/reload-plugin')
    assert "reload-plugin" in seed_turn["send"]
    assert "reload-skill" in seed_turn["send"]
    assert seed_turn["expect_all"] == ["reload-plugin-fixture-ok"]
    assert scenario["turns"][2] == {
        "send": "/reload-plugins",
        "expect_all": ["Reloaded 2 plugins."],
    }
    assert scenario["turns"][3] == {
        "send": "/plugin",
        "expect_all": [
            "- demo-plugin v1.0.0 [user] (enabled) [skills]",
            "- reload-plugin v1.0.0 [user] (enabled) [skills]",
        ],
    }
    skills_turn = scenario["turns"][4]
    assert skills_turn["send"] == "/skills"
    assert skills_turn["capture"] == "visible"
    assert "[plugin] reload-plugin:reload-skill" in skills_turn["expect_all"]
    assert scenario["turns"][5]["send"].startswith('!rm -rf "$HOME/.koder/plugins/reload-plugin"')
    assert scenario["turns"][5]["expect_all"] == ["reload-plugin-removed-ok"]
    assert scenario["turns"][6] == {
        "send": "/reload-plugins",
        "expect_all": ["Reloaded 1 plugins."],
    }
    assert scenario["turns"][7] == {
        "send": "/clear",
        "expect_all": ["Switched to session:"],
    }
    final_skills = scenario["turns"][8]
    assert final_skills["send"] == "/skills"
    assert "reload-plugin:reload-skill" in final_skills["expect_not"]
    assert scenario["post_assertions"] == [
        {"path_not_exists": "$HOME/.koder/plugins/reload-plugin"}
    ]


def test_release_notes_scenario_is_acceptance_backed_by_cached_changelog():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["release-notes"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["prelaunch_files"] == [
        {
            "path": "$HOME/.koder/cache/changelog.md",
            "content": "# Changelog\n\n## 0.4.13 - 2026-04-09\n- Added configurable statusline setup\n- Improved release notes command\n\n## 0.4.12 - 2026-04-01\n- Older performance improvements\n",
        },
        {
            "path": "$HOME/.koder/config.yaml",
            "content": "harness:\n  last_release_notes_seen: 0.4.12\n",
        },
    ]
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    release_turn = scenario["turns"][0]
    assert release_turn["send"] == "/release-notes"
    assert {
        "Version 0.4.13:",
        "· Added configurable statusline setup",
        "· Improved release notes command",
    } <= set(release_turn["expect_all"])
    assert {"Version 0.4.12:", "Older performance improvements"} <= set(release_turn["expect_not"])
    proof_turn = scenario["turns"][1]
    assert "resolve_runtime_version" in proof_turn["send"]
    assert "last_release_notes_seen: 0.4.13" not in proof_turn["send"]
    assert "release-config-ok" not in proof_turn["send"]
    assert proof_turn["expect_all"] == [
        "Shell Mode",
        "last_release_notes_seen: $RUNTIME_VERSION",
        "Added configurable statusline setup",
        "release-config-ok",
    ]
    assert scenario["turns"][2] == {
        "send": "/version",
        "expect_all": ["version: $RUNTIME_VERSION", "package: koder"],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/config.yaml",
                "last_release_notes_seen: $RUNTIME_VERSION",
            ]
        },
        {"file_contains": ["$HOME/.koder/cache/changelog.md", "Improved release notes command"]},
    ]


def test_pr_comments_scenarios_are_acceptance_backed_by_fake_gh_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    canonical = manifest["slash_commands"]["pr-comments"]
    alias = manifest["slash_commands"]["pr_comments"]

    for scenario in (canonical, alias):
        assert scenario["validation_level"] == "acceptance"
        assert scenario["env"] == {"PATH": "$REPO/bin:$PATH"}
        assert scenario["acceptance_criteria"]
        assert scenario["acceptance_artifacts"]
        assert "fake_gh_pr_comments.sh" in scenario["turns"][0]["send"]
        assert scenario["turns"][0]["expect_all"][0] not in scenario["turns"][0]["send"]
        assert scenario["post_assertions"] == [
            {
                "file_contains": [
                    "$HOME/.koder/pr-comments-gh.log",
                    [
                        "pr view --json number,headRepository",
                        "api /repos/octo/demo/issues/123/comments",
                        "api /repos/octo/demo/pulls/123/comments",
                    ],
                ]
            }
        ]

    assert canonical["turns"][1]["send"] == "/pr-comments"
    assert {
        "## Comments",
        "@alice PR conversation",
        "@bob src/app.py#42:",
        "```diff",
        "@carol:",
        "Agreed",
    } <= set(canonical["turns"][1]["expect_all"])
    assert canonical["turns"][3] == {
        "send": "/pr-comments",
        "expect_all": ["No comments found."],
    }
    assert canonical["turns"][5] == {
        "send": "/pr-comments",
        "expect_all": ["pr-comments: unable to fetch PR comments via gh."],
    }
    assert canonical["turns"][7] == {
        "send": "/pr-comments",
        "expect_all": ["pr-comments: unable to resolve current PR via gh."],
    }
    assert alias["turns"][1]["send"] == "/pr_comments"
    assert alias["turns"][2]["send"] == "/pr-comments"
    assert "pr-comments-alias-log-ok" in alias["turns"][3]["expect_all"]
    assert "pr-comments-alias-log-ok" not in alias["turns"][3]["send"]


def test_issue_scenario_is_acceptance_backed_by_fake_gh_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["issue"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"PATH": "$REPO/bin:$PATH"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "fake_gh_issue.sh" in scenario["turns"][0]["send"]
    assert "issue-gh-fixture" not in scenario["turns"][0]["send"]
    assert scenario["turns"][1] == {
        "send": "/issue",
        "expect_all": [
            "Recent issues:",
            "42",
            "Fix flaky tmux validation",
            "43",
            "Document release checklist",
        ],
    }
    assert scenario["turns"][2] == {
        "send": "/issue fixture title",
        "expect_all": [
            "To create an issue",
            'Create a GitHub issue titled "fixture title"',
        ],
    }
    assert scenario["turns"][4] == {
        "send": "/issue",
        "expect_all": ["No open issues found. Create one with: /issue <title>"],
    }
    assert scenario["turns"][6] == {
        "send": "/issue",
        "expect_all": ["Failed to fetch issues: gh issue list failed for scenario"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/issue-gh.log", "issue list --limit 10"]}
    ]


def test_subscribe_pr_scenario_is_acceptance_backed_by_fake_gh_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["subscribe-pr"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"PATH": "$REPO/bin:$PATH"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "fake_gh_subscribe_pr.sh" in scenario["turns"][0]["send"]
    assert "subscribe-pr-gh-fixture" not in scenario["turns"][0]["send"]
    assert scenario["turns"][1] == {
        "send": "/subscribe-pr",
        "expect_all": [
            "Open PRs (subscribe via GitHub notifications):",
            "17",
            "Improve tmux validation UX",
            "18",
            "Fix docs counter drift",
            "Use 'gh pr view <number>' for details.",
        ],
    }
    assert scenario["turns"][3] == {
        "send": "/subscribe-pr",
        "expect_all": ["No open PRs. PR subscription is managed via GitHub notifications."],
    }
    assert scenario["turns"][5] == {
        "send": "/subscribe-pr",
        "expect_all": ["Failed to fetch PRs: gh pr list failed for scenario"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/subscribe-pr-gh.log", "pr list --state open --limit 5"]}
    ]


def test_review_scenario_is_acceptance_backed_by_fake_gh_and_provider():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["review"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "OPENAI_API_KEY": "review-secret-token",
        "PATH": "$REPO/bin:$PATH",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "review-fixture-finding: verify the changed diff path and add regression coverage.",
        "log_file": "$HOME/fake-openai-review.log",
        "ready_file": "$HOME/fake-openai-review.ready",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "fake_gh_review.sh" in scenario["turns"][0]["send"]
    assert scenario["turns"][1] == {
        "send": "/review",
        "expect_all": [
            "Code Review (uncommitted changes):",
            "review-fixture-finding: verify the changed diff path",
        ],
        "expect_not": ["review-secret-token"],
        "timeout": 30,
    }
    assert "sample.txt" in scenario["turns"][2]["send"]
    assert "+changed" in scenario["turns"][2]["send"]
    assert scenario["turns"][3] == {
        "send": "/review #123",
        "expect_all": [
            "Code Review (PR #123):",
            "review-fixture-finding: verify the changed diff path",
        ],
        "expect_not": ["review-secret-token"],
        "timeout": 30,
    }
    assert "pr diff 123" in scenario["turns"][4]["send"]
    assert "pr_review.py" in scenario["turns"][4]["send"]
    assert scenario["turns"][6] == {
        "send": "/review",
        "expect_all": [
            "No changes to review. Make some changes or specify a PR number: /review #123"
        ],
    }
    assert scenario["turns"][7] == {
        "send": "/review #0",
        "expect_all": ["Failed to fetch PR #0: gh pr diff failed for scenario"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/review-gh.log", ["pr diff 123", "pr diff 0"]]},
        {
            "file_contains": [
                "$HOME/fake-openai-review.log",
                ["/v1/chat/completions", "uncommitted changes", "PR #123", "pr_review.py"],
            ]
        },
        {"file_not_contains": ["$HOME/fake-openai-review.log", "review-secret-token"]},
    ]


def test_security_review_scenario_is_acceptance_backed_by_fake_provider_and_clean_edge():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["security-review"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "OPENAI_API_KEY": "security-review-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "# Security Review\n\nNo high-confidence security findings.",
        "log_file": "$HOME/fake-openai-security-review.log",
        "ready_file": "$HOME/fake-openai-security-review.ready",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/security-review",
        "expect_all": ["# Security Review", "No high-confidence security findings."],
        "expect_not": ["security-review-secret-token"],
        "timeout": 30,
    }
    assert "You are a senior security engineer" in scenario["turns"][1]["send"]
    assert "BASE RANGE: HEAD" in scenario["turns"][1]["send"]
    assert "sample.txt" in scenario["turns"][1]["send"]
    assert "+changed" in scenario["turns"][1]["send"]
    assert scenario["turns"][3] == {
        "send": "/security-review",
        "expect_all": ["security-review: no pending changes to review."],
    }
    assert scenario["turns"][4] == {
        "send": "/status",
        "expect_all": ["Runtime slash commands", "connectivity: local"],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/fake-openai-security-review.log",
                ["/v1/chat/completions", "senior security engineer", "DIFF CONTENT", "sample.txt"],
            ]
        },
        {
            "file_not_contains": [
                "$HOME/fake-openai-security-review.log",
                "security-review-secret-token",
            ]
        },
    ]


def test_advisor_scenario_is_acceptance_backed_by_fake_provider_and_no_context_edge():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["advisor"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "OPENAI_API_KEY": "advisor-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "# Advisor Review\n\n## Assessment\nadvisor-fixture: add regression coverage.",
        "log_file": "$HOME/fake-openai-advisor.log",
        "ready_file": "$HOME/fake-openai-advisor.ready",
    }
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": '!test -f "$HOME/fake-openai-advisor.ready" && cat "$HOME/fake-openai-advisor.ready"',
        "expect_all": ["ready $FAKE_OPENAI_URL"],
    }
    assert "advisor seeded user request: review validation plan" in scenario["turns"][1]["send"]
    assert scenario["turns"][2] == {
        "send": "/advisor focus on tmux validation",
        "expect_all": ["# Advisor Review", "advisor-fixture: add regression coverage."],
        "expect_not": ["advisor-secret-token"],
    }
    assert "focus on tmux validation" in scenario["turns"][3]["send"]
    assert "gpt-5.5" in scenario["turns"][3]["send"]
    assert "advisor seeded user request: review validation plan" in scenario["turns"][3]["send"]
    assert "sample.txt" in scenario["turns"][3]["send"]
    assert scenario["turns"][6] == {
        "send": "/advisor",
        "expect_all": ["advisor: no current session or pending changes to review."],
    }
    assert scenario["turns"][7]["expect_all"] == ["advisor-single-call-ok"]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/advisor-seed.txt", "seeded advisor transcript"]},
        {
            "file_contains": [
                "$HOME/fake-openai-advisor.log",
                [
                    "/v1/chat/completions",
                    "Advisor Review",
                    "focus on tmux validation",
                    "advisor seeded user request",
                    "sample.txt",
                ],
            ]
        },
        {"file_not_contains": ["$HOME/fake-openai-advisor.log", "advisor-secret-token"]},
    ]


def test_autofix_pr_scenario_is_acceptance_backed_by_fake_gh_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["autofix-pr"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {"PATH": "$REPO/bin:$PATH"}
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "fake_gh_autofix_pr.sh" in scenario["turns"][0]["send"]
    assert "autofix-pr-gh-fixture" not in scenario["turns"][0]["send"]
    assert scenario["turns"][1] == {
        "send": "/autofix-pr",
        "expect_all": [
            "Usage: /autofix-pr #<PR-number>",
            "Inspects the PR diff size and tells you how to request an automated fix.",
        ],
    }
    assert scenario["turns"][2] == {
        "send": "/autofix-pr #123",
        "expect_all": ["PR #123: 5 diff lines", "review and fix PR #123"],
    }
    assert scenario["turns"][4] == {
        "send": "/autofix-pr #456",
        "expect_all": ["PR #456 has no changes."],
    }
    assert scenario["turns"][6] == {
        "send": "/autofix-pr #0",
        "expect_all": ["Failed to fetch PR #0: gh pr diff failed for scenario"],
    }
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/.koder/autofix-pr-gh.log",
                ["pr diff 123", "pr diff 456", "pr diff 0"],
            ]
        },
        {"path_not_exists": "$REPO/autofix-pr-output.patch"},
    ]


def test_oauth_refresh_scenario_is_acceptance_backed_by_seeded_tokens():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["oauth-refresh"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0] == {
        "send": "/oauth-refresh",
        "expect_all": [
            "oauth_refresh:",
            "providers: none",
            "login_command: koder auth login <provider>",
        ],
    }
    seed_turn = scenario["turns"][1]
    assert "google@example.invalid" in seed_turn["send"]
    assert "claude@example.invalid" in seed_turn["send"]
    assert "oauth-refresh-token-fixture" not in seed_turn["send"]
    assert scenario["turns"][2] == {
        "send": "/oauth-refresh",
        "expect_all": [
            "oauth_refresh:",
            "providers:",
            "- claude: expired",
            "email: claude@example.invalid",
            "expires_at: 1",
            "- google: valid",
            "email: google@example.invalid",
            "expires_at: 4102444800000",
            "refresh_command: koder auth login <provider>",
        ],
    }
    assert scenario["turns"][3] == {
        "send": "/oauth-refresh now",
        "expect_all": ["Usage: /oauth-refresh"],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/tokens/google.json", "google@example.invalid"]},
        {"file_contains": ["$HOME/.koder/tokens/claude.json", "claude@example.invalid"]},
        {"file_contains": ["$HOME/.koder/tokens/chatgpt.json", "{not json"]},
    ]


def test_commit_push_pr_scenario_is_acceptance_backed_by_git_readiness_state():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["commit-push-pr"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert "commit-push-pr-fixture" not in scenario["turns"][0]["send"]
    assert "commit-push-pr-baseline" in scenario["turns"][0]["expect_all"]
    assert scenario["turns"][1] == {
        "send": "/commit-push-pr",
        "expect_all": [
            "branch:",
            "remote: No git remote configured.",
            "status:",
            "M sample.txt",
            "A  publish-ready.txt",
            "staged_diff:",
            "publish-ready.txt",
            "unstaged_diff:",
            "sample.txt",
        ],
    }
    assert scenario["turns"][3] == {
        "send": "/commit-push-pr",
        "expect_all": [
            "branch:",
            "remote: No git remote configured.",
            "status:",
            "Clean working tree.",
            "staged_diff:",
            "No staged diff.",
            "unstaged_diff:",
            "No working tree diff.",
        ],
    }
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/.git/COMMIT_EDITMSG", "commit-push-pr-clean"]},
        {"path_not_exists": "$REPO/.git/refs/remotes/origin"},
    ]


def test_rewind_scenario_is_acceptance_backed_by_prompt_restore_and_db_trim():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["slash_commands"]["rewind"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    seed_turn = scenario["turns"][0]
    assert seed_turn["send"].startswith(
        '!uv --project "$KODER_SCENARIO_SOURCE_ROOT" run --no-sync python -c'
    )
    assert "first prompt" in seed_turn["send"]
    assert "second prompt" in seed_turn["send"]
    assert "rewind-seed" not in seed_turn["send"]
    assert seed_turn["expect_all"] == ["rewind-seed"]
    assert scenario["turns"][1] == {
        "send": "/rewind",
        "expect_all": [
            "Rewind targets",
            "1. second prompt",
            "2. first prompt",
            "Use /rewind <number> [conversation|code|both].",
        ],
    }
    assert scenario["turns"][2] == {
        "send": "/rewind help",
        "expect_all": ["Usage: /rewind [number] [conversation|code|both]"],
    }
    assert scenario["turns"][3] == {
        "send": "/rewind nope",
        "expect_all": ["Usage: /rewind [number] [conversation|code|both]"],
    }
    assert scenario["turns"][4] == {
        "send": "/rewind 99",
        "expect_all": ["Rewind target must be between 1 and 2."],
    }
    assert scenario["turns"][5] == {
        "send": "/rewind 1",
        "expect_all": [
            "Rewound conversation to prompt 1.",
            "Restored input: second prompt",
        ],
    }
    assert scenario["turns"][6] == {
        "type": " plus",
        "capture": "visible",
        "expect_all": ["second prompt plus"],
        "expect_not": ["Window too small"],
    }
    assert scenario["turns"][7] == {
        "keys": ["C-d"],
        "capture": "visible",
        "expect_all": ["| ⚡ Koder |"],
        "expect_not": ["second prompt plus"],
    }
    proof_turn = scenario["turns"][8]
    assert "rewind-db-proof.txt" in proof_turn["send"]
    assert "rewind-db-ok" not in proof_turn["send"]
    assert proof_turn["expect_all"] == ["rewind-db-ok"]
    assert scenario["post_assertions"] == [
        {"file_contains": ["$REPO/rewind-db-proof.txt", "rewind-db-ok"]}
    ]


def test_manual_skill_scenario_is_acceptance_backed_by_no_model_error():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["skills"]["manual-skill-command"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    first_turn = scenario["turns"][0]
    assert first_turn["send"] == "/demo-skill"
    assert "Use deterministic local output." in first_turn["expect_all"]
    assert {"thinking...", "AuthenticationError", "Execution error"} <= set(
        first_turn["expect_not"]
    )
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$REPO/.koder/skills/demo-skill/SKILL.md",
                ["disable-model-invocation: true", "Use deterministic local output."],
            ]
        }
    ]


def test_project_and_plugin_skill_scenario_is_acceptance_backed_by_lifecycle():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["skills"]["project-and-plugin-skill-listing"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["expect_all"] == [
        "[project] demo-skill",
        "[plugin] demo-plugin:plugin-skill",
    ]
    assert any(turn.get("send") == "/plugin disable demo-plugin" for turn in scenario["turns"])
    hidden_turn = next(turn for turn in scenario["turns"] if turn.get("capture") == "visible")
    assert "- [plugin] demo-plugin:plugin-skill" in hidden_turn["expect_not"]
    assert any(
        turn.get("send") == "/demo-plugin:plugin-skill"
        and "Unknown command 'demo-plugin:plugin-skill'" in turn.get("expect_all", [])
        for turn in scenario["turns"]
    )
    assert scenario["post_assertions"] == [
        {"file_contains": ["$HOME/.koder/plugins/state.json", ['"demo-plugin"', '"enabled": true']]}
    ]


def test_prompt_suggestion_scenario_is_acceptance_backed_by_accept_and_clear():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["prompt-suggestion"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["acceptance_criteria"]
    assert scenario["acceptance_artifacts"]
    assert scenario["turns"][0]["expect_all"] == ["Shell Mode", "tests failed"]
    assert any(turn.get("keys") == ["Right"] for turn in scenario["turns"])
    assert any("Run the tests now" in turn.get("expect_all", []) for turn in scenario["turns"])
    assert any(turn.get("send") == "/clear" for turn in scenario["turns"])
    assert scenario["turns"][-1]["type"] == "unrelated"
    assert "Run the tests" in scenario["turns"][-1]["expect_not"]


def test_validation_levels_are_checked():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["teams"]["tmux-pane"]["validation_level"] = "maybe"

    errors = validate_manifest(manifest)

    assert any("validation_level must be one of" in error for error in errors)


def test_teammate_mode_is_checked():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["teams"]["tmux-discussion"]["teammate_mode"] = "sidecar"

    errors = validate_manifest(manifest)

    assert any("teammate_mode must be auto, in-process, or tmux" in error for error in errors)


def test_cli_args_are_checked():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["slash_commands"]["channels"]["cli_args"] = ["--channels", 1]

    errors = validate_manifest(manifest)

    assert any("cli_args must contain strings" in error for error in errors)


def test_prelaunch_files_are_checked():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["slash_commands"]["release-notes"]["prelaunch_files"] = [
        {"path": "$HOME/.koder/config.yaml"}
    ]

    errors = validate_manifest(manifest)

    assert any("prelaunch_file 1 needs content" in error for error in errors)


def test_prelaunch_source_is_copied_from_repository(tmp_path):
    home, repo = tmp_path / "home", tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    ref = scenarios.ScenarioRef(
        "features",
        "mcp-fixture",
        {
            "prelaunch_files": [
                {"path": "$REPO/peer.py", "source": "tests/fixtures/mcp_stdio_server.py"}
            ]
        },
    )
    scenarios._write_prelaunch_files(ref, home=home, repo=repo)
    assert (repo / "peer.py").read_bytes() == (
        scenarios.PROJECT_ROOT / "tests/fixtures/mcp_stdio_server.py"
    ).read_bytes()


def test_prelaunch_source_cannot_escape_repository():
    import pytest

    for path in ("/etc/passwd", "../outside.py", "AGENTS.md", ""):
        with pytest.raises(ValueError):
            scenarios._prelaunch_source(path)


def test_fake_openai_fixture_is_checked():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["slash_commands"]["btw"]["fake_openai"] = {
        "port": "19081",
        "stream_lines": "many",
    }

    errors = validate_manifest(manifest)

    assert "slash_commands/btw: fake_openai.port must be an integer" in errors
    assert "slash_commands/btw: fake_openai.response is required" in errors
    assert "slash_commands/btw: fake_openai.log_file is required" in errors
    assert "slash_commands/btw: fake_openai.ready_file is required" in errors
    assert "slash_commands/btw: fake_openai.stream_lines must be a positive integer" in errors


def test_fake_openai_fixture_accepts_dynamic_port_and_rejects_invalid_range():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    fixture = manifest["features"]["fixed-bottom-subagent-progress"]["fake_openai"]
    assert fixture["port"] == 0
    assert validate_manifest(manifest) == []

    fixture["port"] = 65536
    errors = validate_manifest(manifest)

    assert (
        "features/fixed-bottom-subagent-progress: fake_openai.port must be from 0 through 65535"
        in errors
    )


def test_streaming_tool_queue_fixture_continuation_uses_latest_user_turn():
    completed_previous_tool_turn = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "content": "sample.txt", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "final answer"},
            {"role": "user", "content": "second"},
        ]
    }
    active_tool_continuation = {
        "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "content": "sample.txt", "tool_call_id": "call_1"},
        ]
    }

    assert not FakeOpenAIHandler._request_has_tool_output(None, completed_previous_tool_turn)
    assert FakeOpenAIHandler._request_has_tool_output(None, active_tool_continuation)


def test_sandbox_function_tool_scenario_reaches_model_function_tool_path():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["sandbox-function-tool-approval"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["fake_openai"]["scenario"] == "sandbox_shell_tool"
    assert scenario["turns"][0]["expect_all"] == [
        "Permission required: run_shell",
        "touch model-tool-created.txt",
        "[y]allow once  [a]always allow  [n]deny",
    ]
    assert scenario["turns"][1]["send"] == "n"
    assert scenario["post_assertions"][0] == {"path_not_exists": "$REPO/model-tool-created.txt"}


def test_fake_openai_sandbox_scenario_emits_run_shell_call():
    handler = object.__new__(FakeOpenAIHandler)
    handler.scenario = "sandbox_shell_tool"

    payload = handler._tool_call_payload()

    assert payload["function"]["name"] == "run_shell"
    assert json.loads(payload["function"]["arguments"]) == {
        "command": "touch model-tool-created.txt"
    }


def test_fixed_bottom_queued_input_scenario_uses_streaming_tool_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["fixed-bottom-queued-input"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_MODEL": "openai/koder-fixture",
        "KODER_CONTEXT_WINDOW": "128000",
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "KODER_API_KEY": "fixed-bottom-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "final answer after queued input",
        "log_file": "$HOME/fake-openai-fixed-bottom-queue.log",
        "ready_file": "$HOME/fake-openai-fixed-bottom-queue.ready",
        "scenario": "streaming_tool_queue",
        "stream_delay": 0.08,
        "stream_lines": 80,
    }
    assert scenario["turns"][0]["expect_all"] == [
        "streaming fixture long line 050",
        "| ⚡ Koder |",
    ]
    assert scenario["turns"][0]["expect_bottom_all"] == ["| ⚡ Koder |"]
    assert scenario["turns"][0]["expect_not"] == ["streaming fixture long line 001"]
    assert scenario["turns"][1]["expect_all"] == [
        "queued: queued from tmux while streaming",
        "| ⚡ Koder |",
    ]
    assert scenario["turns"][1]["expect_bottom_all"] == ["| ⚡ Koder |"]
    assert scenario["turns"][1]["expect_regex"] == [
        "queued: queued from tmux while streaming\\n\\u250c.*\\| \\u26a1 Koder \\|"
    ]
    assert scenario["turns"][1]["expect_not"] == ["Tip:"]
    assert scenario["turns"][2]["expect_bottom_all"] == ["| ⚡ Koder |"]
    assert scenario["turns"][2]["expect_not"] == ["queued: queued from tmux while streaming"]
    assert scenario["turns"][3]["send"] == "hello after prior output"
    assert scenario["turns"][3]["wait"] == 1.1
    assert scenario["turns"][3]["expect_all"] == [
        "streaming fixture long line 010",
        "final answer after queued input",
        "| ⚡ Koder |",
    ]
    assert scenario["turns"][3]["expect_bottom_all"] == ["| ⚡ Koder |"]
    assert scenario["turns"][3]["timeout"] == 0.8
    assert scenario["turns"][4]["expect_all"] == [
        "final answer after queued input",
        "| ⚡ Koder |",
    ]
    assert scenario["turns"][4]["expect_bottom_all"] == ["Tip:", "| ⚡ Koder |"]
    assert scenario["turns"][4]["expect_regex"] == ["Tip: .*\\n\\u250c.*\\| \\u26a1 Koder \\|"]
    assert scenario["post_assertions"] == [
        {
            "file_contains": [
                "$HOME/fake-openai-fixed-bottom-queue.log",
                ["Queued user input", "queued from tmux while streaming", "sample.txt"],
            ]
        }
    ]


def test_fixed_bottom_subagent_progress_uses_compact_dynamic_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["fixed-bottom-subagent-progress"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"]["KODER_BASE_URL"] == "$FAKE_OPENAI_URL"
    assert scenario["fake_openai"]["port"] == 0
    assert scenario["fake_openai"]["scenario"] == "streaming_subagent_tool"
    assert scenario["turns"][0]["resize"] == {"width": 60, "height": 12}
    assert "2 running" in scenario["turns"][1]["expect_all"]
    assert "Succeeded (2 agents, 2 tool uses)" in scenario["turns"][2]["expect_all"]
    assert scenario["post_assertions"] == [
        {
            "file_occurrences": [
                "$HOME/fake-openai-subagent-display.log",
                {
                    "subagent-alpha-read_file-request": 1,
                    "subagent-alpha-read_file-result": 1,
                    "subagent-beta-read_file-request": 1,
                    "subagent-beta-read_file-result": 1,
                    "call_koder_subagent_alpha_read_file": 2,
                    "call_koder_subagent_beta_read_file": 2,
                },
            ]
        }
    ]


def test_fake_openai_subagent_scenario_emits_child_specific_read_calls():
    handler = object.__new__(FakeOpenAIHandler)
    handler.scenario = "streaming_subagent_tool"

    alpha_body = {
        "messages": [
            {
                "role": "user",
                "content": "Read sample.txt. Marker: KODER_SUBAGENT_ALPHA.",
            }
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }
    payload = handler._tool_call_payload(alpha_body)

    assert payload["id"] == "call_koder_subagent_alpha_read_file"
    assert payload["function"]["name"] == "read_file"
    assert json.loads(payload["function"]["arguments"]) == {"path": "sample.txt"}


def test_fixed_bottom_error_history_scenario_uses_failing_stream_fixture():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["fixed-bottom-error-history"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["env"] == {
        "KODER_MODEL": "openai/koder-fixture",
        "KODER_CONTEXT_WINDOW": "128000",
        "KODER_BASE_URL": "$FAKE_OPENAI_URL",
        "KODER_API_KEY": "fixed-bottom-error-secret-token",
    }
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "fixture stream failure",
        "log_file": "$HOME/fake-openai-fixed-bottom-error.log",
        "ready_file": "$HOME/fake-openai-fixed-bottom-error.ready",
        "scenario": "streaming_tool_error",
        "stream_delay": 0.01,
        "stream_lines": 2,
    }
    first_turn = scenario["turns"][0]
    assert first_turn["capture"] == "visible"
    assert {
        "streaming fixture line 1",
        "streaming fixture line 2",
        "read_file",
        "Execution error",
        "fixture stream failure",
        "Please provide new instructions.",
        "| ⚡ Koder |",
    } <= set(first_turn["expect_all"])
    second_turn = scenario["turns"][1]
    assert second_turn["send"] == "!echo tui-still-usable-after-error"
    assert {"streaming fixture line 1", "Execution error"} <= set(second_turn["expect_all"])


def test_fixed_bottom_idle_tip_scenario_checks_tip_with_prompt():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    scenario = manifest["features"]["fixed-bottom-idle-tip"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["fake_openai"] == {
        "port": 0,
        "response": "final answer after queued input",
        "log_file": "$HOME/fake-openai-fixed-bottom-tip.log",
        "ready_file": "$HOME/fake-openai-fixed-bottom-tip.ready",
        "scenario": "streaming_tool_queue",
        "stream_delay": 0.01,
        "stream_lines": 1,
    }
    for turn in scenario["turns"]:
        assert turn["capture"] == "visible"
        assert turn["expect_bottom_all"] == ["Tip:", "| ⚡ Koder |"]
        assert turn["expect_regex"] == ["Tip: .*\\n\\u250c.*\\| \\u26a1 Koder \\|"]


def test_strict_acceptance_requires_acceptance_metadata():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["slash_commands"]["advisor"].pop("acceptance_criteria")
    manifest["slash_commands"]["advisor"].pop("acceptance_artifacts")

    errors = validate_manifest(manifest, strict_acceptance=True)

    assert "slash_commands/advisor: acceptance_criteria is required" in errors
    assert "slash_commands/advisor: acceptance_artifacts is required" in errors


def test_strict_acceptance_accepts_complete_metadata():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["features"]["prompt-suggestion"].update(
        {
            "validation_level": "acceptance",
            "acceptance_criteria": ["prompt suggestion renders after a local shell turn"],
            "acceptance_artifacts": ["leader capture"],
            "post_assertions": [{"path_exists": "$REPO/sample.txt"}],
        }
    )
    manifest["features"]["prompt-suggestion"]["turns"][0]["expect_all"] = ["tests failed"]

    errors = validate_manifest(manifest, strict_acceptance=True)

    assert errors == []


def test_post_assertions_are_validated():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["teams"]["tmux-pane"]["post_assertions"] = [{"file_contains": ["$HOME/x.txt"]}]

    errors = validate_manifest(manifest)

    assert any("file_contains needs [path, text-or-text-list]" in error for error in errors)


def test_file_occurrence_post_assertions_are_validated():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["teams"]["tmux-pane"]["post_assertions"] = [
        {"file_occurrences": ["$HOME/x.txt", {"marker": -1}]}
    ]

    errors = validate_manifest(manifest)

    assert any(
        "file_occurrences needs [path, {text: exact-nonnegative-count}]" in error
        for error in errors
    )


def test_sqlite_post_assertions_are_validated():
    manifest = copy.deepcopy(_load_manifest(DEFAULT_MANIFEST))
    manifest["features"]["memory-and-session"]["post_assertions"] = [
        {"sqlite_contains": ["$HOME/.koder/koder.db", "delete from session_metadata", "x"]}
    ]

    errors = validate_manifest(manifest)

    assert any("sqlite_contains query must be SELECT-only" in error for error in errors)


def test_post_assertions_check_filesystem_state(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    marker = home / ".koder" / "teams" / "demo" / "history.jsonl"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"event": "message_read", "read": true}\n', encoding="utf-8")
    scenario = scenarios.ScenarioRef(
        suite="teams",
        name="demo",
        payload={
            "post_assertions": [
                {"path_exists": "$HOME/.koder/teams/demo/history.jsonl"},
                {
                    "file_contains": [
                        "$HOME/.koder/teams/demo/history.jsonl",
                        ['"event": "message_read"', '"read": true'],
                    ]
                },
                {"file_not_contains": ["$HOME/.koder/teams/demo/history.jsonl", "unread"]},
            ]
        },
    )

    failures = scenarios._run_post_assertions(scenario, home=home, repo=repo)

    assert failures == []


def test_post_assertions_check_exact_file_occurrences(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    log_file = home / "provider.log"
    log_file.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    counts = {"alpha": 2, "beta": 1}
    scenario = scenarios.ScenarioRef(
        suite="features",
        name="subagents",
        payload={"post_assertions": [{"file_occurrences": ["$HOME/provider.log", counts]}]},
    )

    assert scenarios._run_post_assertions(scenario, home=home, repo=repo) == []

    counts["alpha"] = 1
    [failure] = scenarios._run_post_assertions(scenario, home=home, repo=repo)

    assert "'alpha': expected 1, found 2" in failure


def test_post_assertions_check_globbed_filesystem_state(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    home.mkdir()
    repo.mkdir()
    first = home / ".koder" / "agents" / "agent-one.json"
    first.parent.mkdir(parents=True)
    first.write_text(
        '{"model_config": {"model_name": "litellm/openai/gpt-4.1", "api_key_present": true}}\n',
        encoding="utf-8",
    )
    scenario = scenarios.ScenarioRef(
        suite="agents",
        name="demo",
        payload={
            "post_assertions": [
                {"path_glob_exists": "$HOME/.koder/agents/agent-*.json"},
                {
                    "file_glob_contains": [
                        "$HOME/.koder/agents/agent-*.json",
                        ["litellm/openai/gpt-4.1", '"api_key_present": true'],
                    ]
                },
                {
                    "file_glob_not_contains": [
                        "$HOME/.koder/agents/agent-*.json",
                        "secret-key",
                    ]
                },
            ]
        },
    )

    failures = scenarios._run_post_assertions(scenario, home=home, repo=repo)

    assert failures == []


def test_post_assertions_check_sqlite_state(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    database = home / ".koder" / "koder.db"
    database.parent.mkdir(parents=True)
    repo.mkdir()
    with closing(sqlite3.connect(database)) as conn, conn:
        conn.execute("create table session_metadata(session_id text, title text)")
        conn.execute("insert into session_metadata values (?, ?)", ("s1", "feature-session"))
    scenario = scenarios.ScenarioRef(
        suite="features",
        name="memory-and-session",
        payload={
            "post_assertions": [
                {
                    "sqlite_contains": [
                        "$HOME/.koder/koder.db",
                        "select title from session_metadata where session_id = 's1'",
                        "feature-session",
                    ]
                }
            ]
        },
    )

    failures = scenarios._run_post_assertions(scenario, home=home, repo=repo)

    assert failures == []


def test_scenario_env_expansion_replaces_repo_home_and_path(monkeypatch, tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    value = scenarios._expand_scenario_env_value(
        "$REPO/bin:$HOME/tools:$PATH", home=home, repo=repo
    )

    assert value == f"{repo}/bin:{home}/tools:/usr/bin:/bin"


def test_scenario_env_expansion_replaces_dynamic_fake_openai_url(tmp_path):
    home = tmp_path / "home"
    repo = tmp_path / "repo"

    value = scenarios._expand_scenario_env_value(
        "$FAKE_OPENAI_URL/chat",
        home=home,
        repo=repo,
        fake_openai_url="http://127.0.0.1:43210/v1",
    )

    assert value == "http://127.0.0.1:43210/v1/chat"


def test_tmux_pane_assertion_helper_matches_any_worker_pane(monkeypatch):
    monkeypatch.setattr(
        scenarios,
        "_capture_all_panes",
        lambda _session: {
            "%0": "leader pane",
            "%1": "Command Response\nmodel: gpt-4o\nprovider: openai",
        },
    )

    passed, outputs = scenarios._tmux_pane_assertions_pass(
        "session",
        {
            "expect_tmux_panes_min": 2,
            "expect_tmux_any_pane_all": ["Command Response", "model:"],
        },
    )

    assert passed is True
    assert sorted(outputs) == ["%0", "%1"]


def test_kill_tmux_pane_matching_skips_leader_and_kills_worker(monkeypatch):
    calls = []
    monkeypatch.setattr(scenarios, "_list_panes", lambda _session: ["%0", "%1", "%2"])
    monkeypatch.setattr(
        scenarios,
        "_capture_all_panes",
        lambda _session: {
            "%0": "leader marker",
            "%1": "worker marker",
            "%2": "other worker",
        },
    )

    def fake_tmux(*args, **_kwargs):
        calls.append(args)
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(scenarios, "_tmux", fake_tmux)

    error = scenarios._kill_tmux_pane_matching("session", "worker marker")

    assert error is None
    assert calls == [("kill-pane", "-t", "%1")]


def test_legacy_slash_shell_test_delegates_to_scenario_runner():
    script = (scenarios.PROJECT_ROOT / "tests/e2e/test_all_slash_commands.sh").read_text(
        encoding="utf-8"
    )

    assert "tmux_feature_scenarios.py" in script
    assert "Testing all 119 commands" not in script
