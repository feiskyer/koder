"""The real CLI setup path must not block the active event loop."""

import pytest

from koder_agent import utils
from koder_agent.harness import session_flow
from tests.harness.test_session_flow_stdin import _patch_session_flow


@pytest.mark.asyncio
async def test_cli_client_setup_keeps_event_loop_responsive(
    tmp_path, monkeypatch, event_loop_progress_probe
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _patch_session_flow(monkeypatch, "", stdin_is_tty=True)
    wrap, observed = event_loop_progress_probe
    monkeypatch.setattr(utils, "setup_openai_client", wrap(lambda: None))

    code = await session_flow.run_harness_session_flow(
        first_arg=None, argv=["--bare", "--print", "fixture prompt"]
    )

    assert code == 0
    assert observed == [True], "client setup blocked the CLI event loop"
