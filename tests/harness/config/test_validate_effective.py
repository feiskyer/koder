from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from koder_agent.config.manager import ConfigManager
from koder_agent.harness.config import commands
from koder_agent.harness.config.service import RuntimeConfigService

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK_DELEGATE_ENV_VARS = (
    "KODER_TASK_DELEGATE_MAX_BATCH_SIZE",
    "KODER_TASK_DELEGATE_MAX_CONCURRENCY",
)


def _patch_validate_service(monkeypatch, config_path):
    """Make _handle_config_validate use a RuntimeConfigService at config_path."""
    monkeypatch.setattr(
        commands,
        "RuntimeConfigService",
        lambda *a, **k: RuntimeConfigService(config_path=config_path),
    )


def _run_real_config_validate_cli(
    tmp_path,
    *,
    env_overrides: dict[str, str],
) -> subprocess.CompletedProcess:
    config_path = tmp_path / ".koder" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump({"harness": {"task_delegate_max_batch_size": 3}}),
        encoding="utf-8",
    )
    # A real CLI process needs a synthetic profile AND cwd. Do not inherit
    # provider credentials or project-local settings from the developer shell.
    inherited_names = (
        "PATH",
        "SYSTEMROOT",
        "COMSPEC",
        "TMPDIR",
        "TEMP",
        "TMP",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_NO_SYNC",
    )
    env = {name: os.environ[name] for name in inherited_names if name in os.environ}
    env["HOME"] = str(tmp_path)
    env["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    env["OPENAI_AGENTS_DISABLE_TRACING"] = "1"
    env.update(env_overrides)
    return subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(PROJECT_ROOT),
            "--no-env-file",
            "koder",
            "config",
            "validate",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_config_validate_valid(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": {"name": "gpt-4.1", "provider": "openai"}}),
        encoding="utf-8",
    )
    _patch_validate_service(monkeypatch, config_path)
    exit_code = commands._handle_config_validate()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Config valid" in out


def test_config_validate_invalid_schema(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    # reasoning_effort accepts only a constrained literal set; a bogus value fails.
    config_path.write_text(
        yaml.safe_dump({"model": {"reasoning_effort": "turbo"}}),
        encoding="utf-8",
    )
    _patch_validate_service(monkeypatch, config_path)
    exit_code = commands._handle_config_validate()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Config invalid" in out


def test_config_validate_bad_yaml(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: [unbalanced\n", encoding="utf-8")
    _patch_validate_service(monkeypatch, config_path)
    exit_code = commands._handle_config_validate()
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Config invalid" in out


def test_config_validate_missing_file_ok(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "missing.yaml"
    _patch_validate_service(monkeypatch, config_path)
    exit_code = commands._handle_config_validate()
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "defaults are valid" in out


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("KODER_TASK_DELEGATE_MAX_BATCH_SIZE", "not-an-int"),
        ("KODER_TASK_DELEGATE_MAX_CONCURRENCY", "0"),
    ],
)
def test_config_validate_rejects_invalid_effective_task_delegate_env(
    monkeypatch,
    tmp_path,
    capsys,
    env_name,
    env_value,
):
    config_path = tmp_path / "missing.yaml"
    _patch_validate_service(monkeypatch, config_path)
    monkeypatch.setenv(env_name, env_value)

    exit_code = commands._handle_config_validate()
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "Config invalid: effective configuration" in out
    assert env_name in out
    assert "koder config validate" in out


def test_config_validate_rejects_effective_concurrency_above_batch(
    monkeypatch,
    tmp_path,
    capsys,
):
    config_path = tmp_path / "missing.yaml"
    _patch_validate_service(monkeypatch, config_path)
    monkeypatch.setenv("KODER_TASK_DELEGATE_MAX_BATCH_SIZE", "3")
    monkeypatch.setenv("KODER_TASK_DELEGATE_MAX_CONCURRENCY", "4")

    exit_code = commands._handle_config_validate()
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "KODER_TASK_DELEGATE_MAX_BATCH_SIZE" in out
    assert "KODER_TASK_DELEGATE_MAX_CONCURRENCY" in out
    assert "less than or equal" in out


def test_real_cli_validates_env_precedence_before_task_delegate_relation(tmp_path):
    proc = _run_real_config_validate_cli(
        tmp_path,
        env_overrides={"KODER_TASK_DELEGATE_MAX_CONCURRENCY": "2"},
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode == 0, output
    assert "Config valid:" in proc.stdout
    assert "Fatal Error" not in output
    assert "Traceback" not in output


@pytest.mark.parametrize(
    ("env_name", "env_value", "expected_error"),
    [
        (
            "KODER_TASK_DELEGATE_MAX_BATCH_SIZE",
            "not-an-int",
            "expected an integer between 1 and 32",
        ),
        (
            "KODER_TASK_DELEGATE_MAX_CONCURRENCY",
            "not-an-int",
            "expected an integer between 1 and 32",
        ),
        ("KODER_TASK_DELEGATE_MAX_CONCURRENCY", "4", "less than or equal"),
    ],
)
def test_real_cli_reports_invalid_task_delegate_env_without_fatal_startup(
    tmp_path,
    env_name,
    env_value,
    expected_error,
):
    proc = _run_real_config_validate_cli(
        tmp_path,
        env_overrides={env_name: env_value},
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode == 1, output
    assert "Config invalid: effective configuration" in proc.stdout
    assert env_name in proc.stdout
    assert expected_error in proc.stdout
    assert "Fatal Error" not in output
    assert "Traceback" not in output


@pytest.mark.parametrize(
    ("env_overrides", "invalid_env"),
    [
        (
            {"KODER_TASK_DELEGATE_MAX_CONCURRENCY": "2.0"},
            "KODER_TASK_DELEGATE_MAX_CONCURRENCY",
        ),
        (
            {
                "KODER_TASK_DELEGATE_MAX_BATCH_SIZE": "3.0",
                "KODER_TASK_DELEGATE_MAX_CONCURRENCY": "2",
            },
            "KODER_TASK_DELEGATE_MAX_BATCH_SIZE",
        ),
    ],
)
def test_real_cli_rejects_decimal_task_delegate_env_with_shared_grammar(
    tmp_path,
    env_overrides,
    invalid_env,
):
    proc = _run_real_config_validate_cli(
        tmp_path,
        env_overrides=env_overrides,
    )
    output = proc.stdout + proc.stderr

    assert proc.returncode == 1, output
    assert "Config invalid: effective configuration" in proc.stdout
    assert invalid_env in proc.stdout
    assert "expected an integer between 1 and 32" in proc.stdout
    assert "Fatal Error" not in output
    assert "Traceback" not in output


@pytest.mark.asyncio
async def test_config_show_effective_overlays_env(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": {"name": "file-model", "provider": "openai"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(commands, "get_config_manager", lambda: ConfigManager(config_path))
    monkeypatch.setenv("KODER_MODEL", "env-model")

    args = argparse.Namespace(config_action="show", effective=True)
    exit_code = await commands.handle_config_subcommand(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    dumped = yaml.safe_load(out)
    assert dumped["model"]["name"] == "env-model"


@pytest.mark.asyncio
async def test_config_show_without_effective_ignores_env(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"model": {"name": "file-model", "provider": "openai"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(commands, "get_config_manager", lambda: ConfigManager(config_path))
    monkeypatch.setenv("KODER_MODEL", "env-model")

    args = argparse.Namespace(config_action="show", effective=False)
    exit_code = await commands.handle_config_subcommand(args)
    out = capsys.readouterr().out
    assert exit_code == 0
    dumped = yaml.safe_load(out)
    assert dumped["model"]["name"] == "file-model"
