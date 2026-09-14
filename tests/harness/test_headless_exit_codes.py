"""Automation must not receive successful exit codes for failed work."""

import json
from unittest.mock import AsyncMock

import pytest

from koder_agent.harness import session_flow
from koder_agent.harness.tools.shell_executor import ShellExecutionResult
from tests.harness.test_session_flow_stdin import _FakeScheduler, _patch_session_flow


@pytest.fixture(autouse=True)
def isolated_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)


def _set_turn_result(monkeypatch, *, status, response):
    async def handle(self, _prompt, **_kwargs):
        self._last_turn_errored = status == "error"
        self._last_turn_cancelled = status == "cancelled"
        return response

    monkeypatch.setattr(_FakeScheduler, "handle", handle)
    monkeypatch.setattr(_FakeScheduler, "handle_stream_json", handle, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "json", "stream-json", "verbose-stream-json"])
@pytest.mark.parametrize("status,expected", [("success", 0), ("error", 1), ("cancelled", 130)])
async def test_print_mode_propagates_turn_status(monkeypatch, capsys, mode, status, expected):
    # Status, not the wording of a model response, determines success.
    response = "Error: quoted example" if status == "success" else "stopped response"
    _set_turn_result(monkeypatch, status=status, response=response)
    output_format = "stream-json" if mode == "verbose-stream-json" else mode
    argv = ["--bare", "--output-format", output_format]
    if mode == "verbose-stream-json":
        argv.append("--verbose")
    argv.extend(["--print", "request"])

    code = await session_flow.run_harness_session_flow(first_arg=None, argv=argv)

    assert code == expected
    if output_format != "text":
        payload = json.loads(capsys.readouterr().out)
        assert payload["result"] == response
        assert payload.get("is_error", False) is (expected != 0)
        if expected:
            assert payload["exit_code"] == expected
        if output_format == "stream-json":
            assert payload["type"] == "result"
    assert _FakeScheduler.instances[-1].cleanup_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [("error", 1), ("cancelled", 130)])
async def test_positional_prompt_propagates_turn_status(monkeypatch, status, expected):
    _set_turn_result(monkeypatch, status=status, response="stopped")
    code = await session_flow.run_harness_session_flow(first_arg=None, argv=["--bare", "request"])
    assert code == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "json", "stream-json"])
@pytest.mark.parametrize(
    "result,expected",
    [
        (ShellExecutionResult(status="success", output="done", exit_code=0), 0),
        (ShellExecutionResult(status="error", output="failed", exit_code=7), 7),
        (ShellExecutionResult(status="error", output="signalled", exit_code=-15), 143),
        (ShellExecutionResult(status="error", output="permission denied"), 1),
        (
            ShellExecutionResult(
                status="error", output="command finished; sandbox cleanup failed", exit_code=0
            ),
            1,
        ),
        (ShellExecutionResult(status="background", output="started", shell_id="shell-1"), 0),
    ],
)
async def test_shell_print_preserves_execution_status(monkeypatch, capsys, mode, result, expected):
    execute = AsyncMock(return_value=result)
    monkeypatch.setattr(session_flow, "execute_shell_command", execute)

    code = await session_flow.run_harness_session_flow(
        first_arg=None,
        argv=["--bare", "--output-format", mode, "--print", "!fixture-command"],
    )

    assert code == expected
    execute.assert_awaited_once()
    if mode in {"json", "stream-json"}:
        payload = json.loads(capsys.readouterr().out)
        assert payload["result"] == result.output
        assert payload.get("is_error", False) is (expected != 0)
        if expected:
            assert payload["exit_code"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["text", "json"])
async def test_empty_shell_command_is_an_error(capsys, mode):
    code = await session_flow.run_harness_session_flow(
        first_arg=None, argv=["--bare", "--output-format", mode, "--print", "!"]
    )
    assert code == 1
    if mode == "json":
        assert json.loads(capsys.readouterr().out)["is_error"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [("error", 1), ("cancelled", 130)])
async def test_failed_turn_is_not_replaced_by_a_schema_validation_error(
    monkeypatch, capsys, status, expected
):
    _set_turn_result(monkeypatch, status=status, response="original execution failure")
    schema = json.dumps({"type": "object", "required": ["answer"]})

    code = await session_flow.run_harness_session_flow(
        first_arg=None,
        argv=[
            "--bare",
            "--output-format",
            "json",
            "--json-schema",
            schema,
            "--print",
            "request",
        ],
    )

    assert code == expected
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "original execution failure"
    assert payload["is_error"] is True
    assert "structured_output" not in payload


@pytest.mark.asyncio
async def test_schema_validation_failure_keeps_json_output_machine_readable(monkeypatch, capsys):
    _set_turn_result(monkeypatch, status="success", response='{"answer": "wrong type"}')
    schema = json.dumps(
        {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
    )
    code = await session_flow.run_harness_session_flow(
        first_arg=None,
        argv=["--bare", "--output-format", "json", "--json-schema", schema, "--print", "request"],
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["is_error"] is True
    assert "did not match" in payload["result"]


@pytest.mark.asyncio
async def test_successful_schema_validation_preserves_structured_output(monkeypatch, capsys):
    _set_turn_result(monkeypatch, status="success", response='{"answer": 42}')
    code = await session_flow.run_harness_session_flow(
        first_arg=None,
        argv=[
            "--bare",
            "--output-format",
            "json",
            "--json-schema",
            '{"type":"object"}',
            "--print",
            "request",
        ],
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["structured_output"] == {"answer": 42}
    assert not payload.get("is_error")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["json", "stream-json"])
async def test_failed_session_switch_is_not_json_success(monkeypatch, capsys, mode):
    handler = session_flow.HarnessInteractiveCommandHandler
    monkeypatch.setattr(handler, "is_slash_command", lambda _self, _prompt: True)
    monkeypatch.setattr(
        handler,
        "handle_slash_input",
        AsyncMock(return_value="session_switch:missing"),
        raising=False,
    )
    monkeypatch.setattr(
        session_flow, "_switch_active_session", AsyncMock(side_effect=ValueError("cannot restore"))
    )

    code = await session_flow.run_harness_session_flow(
        first_arg=None,
        argv=["--bare", "--output-format", mode, "--print", "/resume missing"],
    )

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"] == "cannot restore"
    assert payload["is_error"] is True
