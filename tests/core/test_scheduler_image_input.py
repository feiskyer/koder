"""Tests that AgentScheduler.handle threads multimodal image input to Runner.run.

The ``-i/--image`` flow builds a multimodal ``input`` (image blocks + text) via
``koder_agent.utils.image_input.build_multimodal_input`` and passes it to
``scheduler.handle(..., multimodal_input=...)``. These tests prove the scheduler
forwards that list as the actual ``Runner.run`` input on the first turn, while
the plain-text path (no images) still passes a string.
"""

import base64
from unittest.mock import AsyncMock, patch

import pytest

from koder_agent.utils.image_input import build_multimodal_input

# Minimal valid 1x1 PNG so build_multimodal_input's magic-byte validation passes.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)


def _make_scheduler(session_id: str):
    """Build an AgentScheduler with a stubbed session/agent for turn tests."""
    from koder_agent.core.scheduler import AgentScheduler

    mock_session = AsyncMock()
    mock_session.session_id = session_id
    # Empty history so this counts as a first turn (title/memory path).
    mock_session.get_items = AsyncMock(return_value=[])
    mock_session.db_path = ":memory:"

    with (
        patch("koder_agent.core.scheduler.get_all_tools", return_value=[]),
        patch("koder_agent.core.scheduler.get_display_hooks"),
        patch("koder_agent.core.scheduler.ApprovalHooks"),
        patch("koder_agent.core.scheduler.EnhancedSQLiteSession", return_value=mock_session),
    ):
        scheduler = AgentScheduler(session_id=session_id)

    scheduler.dev_agent = object()
    scheduler._agent_initialized = True
    scheduler._migration_done = True
    # Avoid touching the network/usage/magic-doc/memory subsystems.
    scheduler._capture_usage = AsyncMock()
    scheduler._refresh_magic_docs_after_turn = AsyncMock()
    scheduler._load_memory_context = AsyncMock(return_value="")
    scheduler._repair_unreplayable_session_items = AsyncMock()
    scheduler.session = mock_session
    return scheduler


class _Result:
    final_output = "ok"
    context_wrapper = None


@pytest.mark.asyncio
async def test_handle_passes_multimodal_list_to_runner_run(tmp_path):
    """When multimodal_input is provided, Runner.run receives the LIST, not text."""
    path = tmp_path / "tiny.png"
    path.write_bytes(_PNG_1X1)

    mm = build_multimodal_input("describe this", [str(path)])
    assert isinstance(mm, list)  # sanity: images produced a list

    captured = {}

    async def fake_run(_agent, run_input, **_kwargs):
        captured["input"] = run_input
        return _Result()

    scheduler = _make_scheduler("image-turn")
    try:
        with (
            patch("koder_agent.core.scheduler.Runner.run", side_effect=fake_run),
            patch("koder_agent.core.scheduler.get_companion", return_value=None),
        ):
            await scheduler.handle("describe this", render_output=False, multimodal_input=mm)
    finally:
        await scheduler.cleanup()

    # The scheduler must forward the multimodal list verbatim as the model input.
    assert isinstance(captured["input"], list)
    assert captured["input"] is mm
    content = captured["input"][0]["content"]
    assert content[0]["type"] == "input_image"
    assert content[-1] == {"type": "input_text", "text": "describe this"}


@pytest.mark.asyncio
async def test_handle_plain_text_path_passes_string_to_runner_run():
    """Without multimodal_input, Runner.run still receives the plain text string."""
    captured = {}

    async def fake_run(_agent, run_input, **_kwargs):
        captured["input"] = run_input
        return _Result()

    scheduler = _make_scheduler("text-turn")
    try:
        with (
            patch("koder_agent.core.scheduler.Runner.run", side_effect=fake_run),
            patch("koder_agent.core.scheduler.get_companion", return_value=None),
        ):
            await scheduler.handle("just text", render_output=False)
    finally:
        await scheduler.cleanup()

    assert isinstance(captured["input"], str)
    assert captured["input"] == "just text"


@pytest.mark.asyncio
async def test_streaming_path_passes_multimodal_list_to_run_streamed(tmp_path):
    """The streaming path also forwards the multimodal list as run input."""
    path = tmp_path / "tiny.png"
    path.write_bytes(_PNG_1X1)

    mm = build_multimodal_input("look", [str(path)])
    captured = {}

    scheduler = _make_scheduler("image-stream")
    scheduler.streaming = True

    # Replace the streaming body with a light stub that records the run_input it
    # is handed, so the test does not depend on Rich Live / SDK streaming.
    async def fake_handle_streaming(user_input, *, streaming_ui=None, run_input=None):
        captured["user_input"] = user_input
        captured["run_input"] = run_input
        return "ok"

    scheduler._handle_streaming = fake_handle_streaming

    try:
        with patch("koder_agent.core.scheduler.get_companion", return_value=None):
            await scheduler.handle("look", render_output=False, multimodal_input=mm)
    finally:
        await scheduler.cleanup()

    # Bookkeeping string stays the plain text; run input is the multimodal list.
    assert captured["user_input"] == "look"
    assert captured["run_input"] is mm
    assert isinstance(captured["run_input"], list)
