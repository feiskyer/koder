"""Contracts learned from real, isolated terminal scenario executions."""

import json

from scripts.tmux_feature_scenarios import DEFAULT_MANIFEST, _load_manifest


def test_resume_seed_cannot_pass_on_the_echoed_shell_command():
    turn = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["resume"]["turns"][0]

    assert turn["expect_all"] == ["resume-fixture"]
    assert "resume-fixture" not in turn["send"]
    assert turn["timeout"] >= 45


def test_config_fixture_explicitly_selects_synthetic_native_model_routing():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["config"]

    assert scenario["env"]["KODER_API_KEY"] == "synthetic-config-key"
    assert scenario["env"]["OPENAI_API_KEY"] == ""
    assert any(
        "effective_model: gpt-4.1" in turn.get("expect_all", []) for turn in scenario["turns"]
    )


def test_config_fixture_checks_live_status_label_after_each_model_switch():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["config"]
    switched_config, restored_config = [
        turn for turn in scenario["turns"] if turn.get("send") == "/config"
    ][1:]
    assert "Model: claude-sonnet-4-6" in switched_config["expect_all"]
    assert "Model: gpt-4.1" in restored_config["expect_all"]
    assert switched_config["expect_bottom_all"] == ["Model: claude-sonnet-4-6"]
    assert restored_config["expect_bottom_all"] == ["Model: gpt-4.1"]


def test_permission_fixture_has_an_isolated_auxiliary_classifier():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["permissions"]

    assert scenario["fake_openai"]["port"] == 0
    assert scenario["env"]["KODER_BASE_URL"] == "$FAKE_OPENAI_URL"
    assert scenario["env"]["KODER_MODEL"] == "openai/koder-fixture"
    assert scenario["env"]["KODER_CONTEXT_WINDOW"] == "128000"
    assert any(
        "Classify this shell command: touch blocked.txt" in assertion.get("file_contains", [])
        for assertion in scenario["post_assertions"]
    )


def test_goal_clear_has_a_durable_empty_store_assertion():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["goal"]

    assert {
        "sqlite_contains": [
            "$HOME/.koder/koder.db",
            "select count(*) from session_goals",
            "0",
        ]
    } in scenario["post_assertions"]


def test_streaming_cancel_scenario_checks_recovery_and_persistence():
    scenario = _load_manifest(DEFAULT_MANIFEST)["features"]["streaming-cancel-recovery"]

    assert scenario["validation_level"] == "acceptance"
    assert scenario["fake_openai"]["port"] == 0
    assert scenario["env"]["KODER_BASE_URL"] == "$FAKE_OPENAI_URL"
    assert scenario["env"]["KODER_CONTEXT_WINDOW"] == "128000"
    assert any(turn.get("keys") == ["Escape"] for turn in scenario["turns"])
    assert any(
        "Operation cancelled by user" in turn.get("expect_all", []) for turn in scenario["turns"]
    )
    assert any(
        turn.get("send", "").startswith("!printf")
        and "cancel-recovery-ok" in turn.get("expect_all", [])
        and "cancel-recovery-ok" not in turn["send"]
        for turn in scenario["turns"]
    )
    assert any("sqlite_contains" in assertion for assertion in scenario["post_assertions"])
    assert any("file_contains" in assertion for assertion in scenario["post_assertions"])


def test_streaming_error_fixture_declares_its_synthetic_model_context_window():
    scenario = _load_manifest(DEFAULT_MANIFEST)["features"]["fixed-bottom-error-history"]

    assert scenario["env"]["KODER_CONTEXT_WINDOW"] == "128000"


def test_all_synthetic_model_scenarios_declare_their_context_window():
    from scripts.tmux_feature_scenarios import _scenario_refs

    for scenario in _scenario_refs(_load_manifest(DEFAULT_MANIFEST)):
        env = scenario.payload.get("env", {})
        if env.get("KODER_MODEL") == "openai/koder-fixture":
            assert env.get("KODER_CONTEXT_WINDOW") == "128000", scenario.name


def test_session_environment_onboarding_declares_its_fixture_context_window():
    scenario = _load_manifest(DEFAULT_MANIFEST)["features"]["onboarding-session-env-startup"]

    assert scenario["env"].get("KODER_CONTEXT_WINDOW") == "128000"


def test_model_and_status_fixtures_use_the_native_openai_model_identity():
    manifest = _load_manifest(DEFAULT_MANIFEST)
    for name in ("model", "status"):
        scenario = manifest["slash_commands"][name]
        assert "litellm/openai/gpt-4.1" not in json.dumps(scenario)
        assert "litellm/anthropic/claude-sonnet-4-6" in json.dumps(scenario)


def test_mcp_fixture_uses_the_resolved_interpreter_for_project_admission():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["mcp"]
    assert "Path(sys.executable).resolve()" in scenario["turns"][2]["send"]
    assert "$RUNTIME_PYTHON_RESOLVED -m scenario_server" in scenario["turns"][3]["expect_all"]


def test_ultraplan_status_witnesses_are_outside_the_worktree():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["ultraplan"]
    before = scenario["turns"][2]["send"]
    after = scenario["turns"][5]["send"]

    assert '> "$HOME/ultraplan-status-before.txt"' in before
    assert '> "$HOME/ultraplan-status-after.txt"' in after
    assert 'cmp "$HOME/ultraplan-status-before.txt" "$HOME/ultraplan-status-after.txt"' in after
    assert "--untracked-files=no" not in before + after


def test_onboarding_clears_its_seeded_session_key_before_the_missing_key_check():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["onboarding"]

    assert scenario["env"]["KODER_API_KEY"] == ""
    assert scenario["cli_args"] == ["--session", "onboarding-scenario"]
    assert scenario["prelaunch_files"][0]["path"].endswith("/onboarding-scenario.sh")
    assert scenario["turns"][0]["send"] == "/env unset KODER_API_KEY"
    assert scenario["turns"][1]["send"] == "/onboarding"


def test_local_sandbox_fixture_checks_both_denial_and_explicit_degradation_consent():
    scenario = _load_manifest(DEFAULT_MANIFEST)["features"]["sandbox-unix-local-shell"]

    assert "sandboxed: false" in scenario["turns"][0]["expect_all"]
    assert "executed: false" in scenario["turns"][0]["expect_all"]
    approved = scenario["turns"][0]
    assert "sandbox_unavailable_approval=" in approved["send"]
    assert "explicit_degradation_approvals: 1" in approved["expect_all"]
    assert "Operation not permitted" in approved["expect_all"]
    assert {"file_contains": ["$REPO/sandbox-inside.txt", "sandbox-inside"]} in scenario[
        "post_assertions"
    ]


def test_protected_path_fixture_does_not_claim_a_sandbox_or_process_was_created():
    scenario = _load_manifest(DEFAULT_MANIFEST)["features"]["sandbox-protected-paths"]

    assert scenario["turns"][1]["send"].startswith("/permissions check run_shell")
    denied = scenario["turns"][0]
    assert "sandbox_unavailable_approval=" in denied["send"]
    assert {"sandboxed: false", "created: false", "executed: false"} <= set(denied["expect_all"])
    assert "sandbox violation" in denied["expect_all"]


def test_unavailable_backend_fixture_requires_consent_and_proves_no_fallback_execution():
    scenario = _load_manifest(DEFAULT_MANIFEST)["features"]["sandbox-unavailable-backend"]

    assert "requires_approval: true" in scenario["turns"][1]["expect_all"]
    fallback = scenario["turns"][2]
    assert fallback["send"] == "!touch unavailable-denied.txt"
    assert "executed: false" in fallback["expect_all"]
    assert (
        "unable to capture exact sandbox fallback state; host execution was blocked"
        in fallback["expect_all"]
    )


def test_version_scenario_checks_values_across_all_three_surfaces():
    scenario = _load_manifest(DEFAULT_MANIFEST)["slash_commands"]["version"]
    version_turn, status_turn, cli_turn = scenario["turns"]
    assert "version: $RUNTIME_VERSION" in version_turn["expect_all"]
    assert "source: installed-package" in version_turn["expect_all"]
    assert "cli_banner: $RUNTIME_VERSION (Koder)" in version_turn["expect_all"]
    assert "version: $RUNTIME_VERSION" in status_turn["expect_all"]
    assert cli_turn["expect_all"] == ["$RUNTIME_VERSION (Koder)", "version-cli-finished"]
    assert "--project" in cli_turn["send"]
    assert "| tee version-proof.txt" in cli_turn["send"]
    assert "version-cli-finished" not in cli_turn["send"]
    assert {"file_contains": ["$REPO/version-proof.txt", "$RUNTIME_VERSION (Koder)"]} in scenario[
        "post_assertions"
    ]
