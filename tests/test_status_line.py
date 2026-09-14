from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from koder_agent.core.status_line import StatusLine


class _UsageTracker:
    def __init__(self):
        self.model = "gpt-5.4"
        self.session_usage = type(
            "_Usage",
            (),
            {
                "request_count": 1,
                "input_tokens": 12,
                "output_tokens": 8,
                "total_cost": 0.01,
                "last_input_tokens": 12,
                "last_output_tokens": 8,
                "current_context_tokens": 40,
            },
        )()


@pytest.mark.parametrize("source", ["environment", "config"])
def test_custom_model_context_override_in_both_status_paths(source, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KODER_CONTEXT_WINDOW", raising=False)
    monkeypatch.setattr(
        "koder_agent.utils.client.get_config",
        lambda: SimpleNamespace(
            model=SimpleNamespace(context_window=128000 if source == "config" else None)
        ),
    )
    if source == "environment":
        monkeypatch.setenv("KODER_CONTEXT_WINDOW", "128000")
    _wide_terminal(monkeypatch)
    tracker = _UsageTracker()
    tracker.model = "litellm/openai/koder-fixture"
    tracker.session_usage.current_context_tokens = 64000
    status = StatusLine(tracker, "custom-context")

    rendered = "".join(text for _, text in status.get_formatted_text())
    assert "64k/128k" in rendered
    assert "(50.0%)" in rendered
    payload = status._build_statusline_payload()
    assert payload["context_window"]["context_window_size"] == 128000
    assert payload["context_window"]["used_percentage"] == 50.0


def test_unknown_model_without_context_override_still_fails(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("KODER_CONTEXT_WINDOW", raising=False)
    monkeypatch.setattr(
        "koder_agent.utils.client.get_config",
        lambda: SimpleNamespace(model=SimpleNamespace(context_window=None)),
    )
    tracker = _UsageTracker()
    tracker.model = "litellm/openai/koder-fixture"
    status = StatusLine(tracker, "unknown-context")
    for render in (status.get_formatted_text, status._build_statusline_payload):
        with pytest.raises(ValueError, match="Unknown context window"):
            render()


def test_status_model_refreshes_without_losing_session_usage(tmp_path, monkeypatch):
    from koder_agent.core.usage_tracker import UsageTracker

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KODER_CONTEXT_WINDOW", "128000")
    active_model = "gpt-4.1"
    monkeypatch.setattr("koder_agent.core.usage_tracker.get_model_name", lambda: active_model)
    _wide_terminal(monkeypatch)
    tracker = UsageTracker()
    tracker.session_usage.total_cost = 0.42
    tracker.session_usage.input_tokens = 1000
    status = StatusLine(tracker, "switch-model")
    assert "gpt-4.1" in "".join(text for _, text in status.get_formatted_text())

    active_model = "litellm/openai/koder-fixture"
    rendered = "".join(text for _, text in status.get_formatted_text())
    assert "koder-fixture" in rendered
    assert "gpt-4.1" not in rendered
    assert "~$0.42" in rendered
    assert status._build_statusline_payload()["model"]["id"] == active_model
    assert tracker.session_usage.input_tokens == 1000


def test_mixed_unpriced_usage_is_marked_in_default_and_custom_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("KODER_CONTEXT_WINDOW", "128000")
    tracker = _make_tracker(total_cost=0.42, input_tokens=1000, cost_unavailable=True)
    status = StatusLine(tracker, "mixed-pricing")
    assert "$?" in status._format_token_cost_segment()
    payload = status._build_statusline_payload()
    assert payload["cost"]["total_cost_usd"] == 0.42
    assert payload["cost"]["cost_unavailable"] is True


def test_status_line_uses_configured_command_output(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = tmp_path / ".koder" / "settings.json"
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(
            {
                "statusLine": {
                    "type": "command",
                    "command": 'python -c \'import sys, json; data=json.load(sys.stdin); print("custom:" + data["model"]["display_name"])\'',
                    "padding": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    status_line = StatusLine(usage_tracker=_UsageTracker(), session_id="custom-status-session")
    fragments = status_line.get_formatted_text()
    rendered = "".join(fragment for _, fragment in fragments)

    assert "  custom:gpt-5.4" in rendered
    assert "Model:" not in rendered


def test_status_line_compacts_default_output_to_terminal_width(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback: type("S", (), {"columns": 80})()
    )

    status_line = StatusLine(
        usage_tracker=_UsageTracker(),
        session_id="2026-05-05T19:49:00-long-session-id",
    )
    fragments = status_line.get_formatted_text()
    rendered = "".join(fragment for _, fragment in fragments)

    assert len(rendered) <= 80
    assert "M: " in rendered
    assert "Dir: " in rendered
    assert "Tok: " in rendered


def _make_tracker(model="gpt-5.4", **usage_kwargs):
    """Build a UsageTracker-like double whose summary() reflects usage_kwargs."""
    from koder_agent.core.usage_tracker import UsageTracker

    class FixedModelTracker(UsageTracker):
        @property
        def model(self):
            return model

    tracker = FixedModelTracker()
    for key, value in usage_kwargs.items():
        setattr(tracker.session_usage, key, value)
    return tracker


def _wide_terminal(monkeypatch, columns=160):
    monkeypatch.setattr(
        "shutil.get_terminal_size", lambda fallback: type("S", (), {"columns": columns})()
    )


def test_token_cost_segment_known_pricing(monkeypatch):
    tracker = _make_tracker(input_tokens=40000, output_tokens=5000, cache_read_tokens=12000)
    tracker.get_model_costs = Mock(return_value=(0.000003, 0.000009))
    tracker.session_usage.total_cost = 0.14

    status_line = StatusLine(usage_tracker=tracker, session_id="s")
    segment = status_line._format_token_cost_segment()

    assert segment.startswith("▽ ")
    assert "45k tok" in segment
    assert "(12k cached)" in segment
    assert "~$0.14" in segment
    assert "$?" not in segment


def test_token_cost_segment_unknown_pricing(monkeypatch):
    tracker = _make_tracker(input_tokens=40000, output_tokens=5000, cache_read_tokens=12000)
    tracker.get_model_costs = Mock(return_value=(0.0, 0.0))
    tracker.session_usage.total_cost = 0.0

    status_line = StatusLine(usage_tracker=tracker, session_id="s")
    segment = status_line._format_token_cost_segment()

    assert "45k tok" in segment
    assert "(12k cached)" in segment
    # Cost marked unavailable, not a misleading $0.00.
    assert "$?" in segment
    assert "$0.00" not in segment


def test_token_cost_segment_omits_cached_when_zero():
    tracker = _make_tracker(input_tokens=1000, output_tokens=500, cache_read_tokens=0)
    tracker.get_model_costs = Mock(return_value=(0.000003, 0.000009))
    tracker.session_usage.total_cost = 0.01

    status_line = StatusLine(usage_tracker=tracker, session_id="s")
    segment = status_line._format_token_cost_segment()
    assert "cached" not in segment


def test_absolute_token_warning_fires_past_limit():
    tracker = _make_tracker()
    status_line = StatusLine(usage_tracker=tracker, session_id="s")

    # Default 200k threshold.
    assert status_line.absolute_token_warning(150_000) is None
    warning = status_line.absolute_token_warning(250_000)
    assert warning is not None
    assert "250k" in warning
    assert "200k" in warning


def test_absolute_token_warning_respects_explicit_limit():
    tracker = _make_tracker()
    status_line = StatusLine(usage_tracker=tracker, session_id="s")

    assert status_line.absolute_token_warning(5_000, limit=10_000) is None
    assert status_line.absolute_token_warning(15_000, limit=10_000) is not None


def test_absolute_token_warning_env_override(monkeypatch):
    monkeypatch.setenv("KODER_TOKEN_WARN_LIMIT", "50000")
    tracker = _make_tracker()
    status_line = StatusLine(usage_tracker=tracker, session_id="s")

    # 60k exceeds the overridden 50k limit even though below the 200k default.
    assert status_line.absolute_token_warning(60_000) is not None
    assert status_line.absolute_token_warning(40_000) is None


def test_wide_statusline_renders_token_cost_segment(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _wide_terminal(monkeypatch, columns=200)

    tracker = _make_tracker(
        input_tokens=40000,
        output_tokens=5000,
        cache_read_tokens=12000,
        request_count=3,
        current_context_tokens=45000,
    )
    tracker.get_model_costs = Mock(return_value=(0.000003, 0.000009))
    tracker.session_usage.total_cost = 0.14

    status_line = StatusLine(usage_tracker=tracker, session_id="s")
    fragments = status_line.get_formatted_text()
    rendered = "".join(fragment for _, fragment in fragments)

    assert "▽" in rendered
    assert "45k tok" in rendered
    assert "(12k cached)" in rendered
    assert "~$0.14" in rendered


def test_wide_statusline_renders_absolute_token_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _wide_terminal(monkeypatch, columns=220)
    # Force a huge context window so the percentage warning would NOT trigger,
    # proving the absolute-threshold warning is independent of context %.
    monkeypatch.setattr(
        "koder_agent.core.status_line.get_configured_context_window", lambda model: 2_000_000
    )

    tracker = _make_tracker(
        input_tokens=250000,
        output_tokens=10000,
        request_count=5,
        current_context_tokens=250000,
    )
    tracker.get_model_costs = Mock(return_value=(0.000003, 0.000009))
    tracker.session_usage.total_cost = 1.23

    status_line = StatusLine(usage_tracker=tracker, session_id="s")
    fragments = status_line.get_formatted_text()
    rendered = "".join(fragment for _, fragment in fragments)

    assert "⚠" in rendered
    assert "250k tokens" in rendered
    # Context percentage is tiny (250k / 2M = 12.5%) so this warning is purely absolute.
    assert "(12.5%)" in rendered


def test_wide_statusline_marks_cost_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    _wide_terminal(monkeypatch, columns=200)

    tracker = _make_tracker(
        input_tokens=40000,
        output_tokens=5000,
        request_count=3,
        current_context_tokens=45000,
    )
    tracker.get_model_costs = Mock(return_value=(0.0, 0.0))
    tracker.session_usage.total_cost = 0.0

    status_line = StatusLine(usage_tracker=tracker, session_id="s")
    fragments = status_line.get_formatted_text()
    rendered = "".join(fragment for _, fragment in fragments)

    assert "$?" in rendered


# --- M11: CJK/emoji display width tests ---


def test_display_width_ascii():
    """ASCII characters are each 1 column wide."""
    assert StatusLine._display_width("hello") == 5
    assert StatusLine._display_width("") == 0
    assert StatusLine._display_width("abc123") == 6


def test_display_width_cjk():
    """CJK ideographs occupy 2 columns each."""
    assert StatusLine._display_width("中文") == 4
    assert StatusLine._display_width("日本語") == 6
    assert StatusLine._display_width("a中b") == 4  # 1 + 2 + 1


def test_display_width_fullwidth():
    """Fullwidth Latin letters occupy 2 columns each."""
    # U+FF21 = FULLWIDTH LATIN CAPITAL LETTER A
    assert StatusLine._display_width("ＡＢ") == 4


def test_truncate_respects_cjk_width():
    """Truncation accounts for double-width CJK characters."""
    status_line = StatusLine(usage_tracker=_UsageTracker(), session_id="s")
    # "中文测试" is 8 columns wide. Truncating to max_len=7 should produce
    # a result fitting in 7 columns.
    result = status_line._truncate("中文测试", 7)
    assert StatusLine._display_width(result) <= 7
    assert "..." in result


def test_truncate_cjk_no_truncation_needed():
    """When CJK string fits, it's returned unchanged."""
    status_line = StatusLine(usage_tracker=_UsageTracker(), session_id="s")
    result = status_line._truncate("中文", 4)
    assert result == "中文"


def test_truncate_cjk_from_start():
    """From-start truncation also respects CJK width."""
    status_line = StatusLine(usage_tracker=_UsageTracker(), session_id="s")
    result = status_line._truncate("中文测试数据", 9, from_start=True)
    assert StatusLine._display_width(result) <= 9
    assert result.endswith("...")


def test_compact_cwd_cjk_path(tmp_path, monkeypatch):
    """CJK path segments are measured by display width, not character count."""
    status_line = StatusLine(usage_tracker=_UsageTracker(), session_id="s")
    # A path with CJK characters: "中文" = 4 cols but len() = 2
    cjk_path = "/home/用户/项目/代码"
    # display width = 1+4+1+2+1+4+1+2+1+4 = all ASCII separators (/) + CJK
    # Actually: /home/用户/项目/代码 = 18 display cols
    # With max_len=10, it must truncate
    result = status_line._compact_cwd(cjk_path, 10)
    assert StatusLine._display_width(result) <= 10
