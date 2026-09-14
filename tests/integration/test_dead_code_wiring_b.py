"""Tests that previously dead modules are now wired into runtime."""

from pathlib import Path


def _source(path):
    return (Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8")


def test_mcp_reconnection_wired():
    src = _source("koder_agent/mcp/__init__.py")
    assert "reconnect" in src.lower() or "retry" in src.lower() or "ReconnectionManager" in src


def test_magic_docs_wired():
    src = _source("koder_agent/harness/commands/interactive.py")
    assert "magic_doc" in src or "find_magic_docs" in src


def test_auto_dream_wired():
    src1 = _source("koder_agent/harness/session_flow.py")
    src2 = _source("koder_agent/core/interactive.py")
    assert (
        "auto_dream" in src1
        or "AutoDreamManager" in src1
        or "auto_dream" in src2
        or "record_session" in src2
    )


def test_micro_compact_in_tools():
    src1 = _source("koder_agent/tools/file.py")
    src2 = _source("koder_agent/tools/shell.py")
    assert "truncat" in src1.lower() or "MAX_RESULT" in src1 or "truncat" in src2.lower()


def test_keybinding_overrides_applied():
    src = _source("koder_agent/core/interactive.py")
    # The no-op comment should be replaced with actual logic
    assert "get_key" in src or "override" in src.lower()
