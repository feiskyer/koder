"""Tests for enhanced per-model cost tracking."""

import pytest

from koder_agent.core.usage_tracker import (
    ModelUsage,
    UsageTracker,
)


def test_model_usage_dataclass():
    mu = ModelUsage(model="claude-sonnet-4-6")
    assert mu.input_tokens == 0
    assert mu.output_tokens == 0
    assert mu.cache_read_tokens == 0
    assert mu.cache_write_tokens == 0
    assert mu.cost == 0.0
    assert mu.request_count == 0


def test_record_with_model():
    tracker = UsageTracker()
    tracker.record_usage(1000, 500, model="claude-sonnet-4-6")
    tracker.record_usage(2000, 800, model="claude-sonnet-4-6")
    tracker.record_usage(500, 200, model="gpt-4o")

    assert tracker.session_usage.input_tokens == 3500
    assert tracker.session_usage.output_tokens == 1500
    assert tracker.session_usage.request_count == 3

    # Per-model breakdown
    models = tracker.get_per_model_usage()
    assert "claude-sonnet-4-6" in models
    assert models["claude-sonnet-4-6"].input_tokens == 3000
    assert models["claude-sonnet-4-6"].output_tokens == 1300
    assert models["claude-sonnet-4-6"].request_count == 2
    assert "gpt-4o" in models
    assert models["gpt-4o"].input_tokens == 500


def test_record_cache_tokens():
    tracker = UsageTracker()
    tracker.record_usage(
        1000,
        500,
        model="claude-sonnet-4-6",
        cache_read_tokens=800,
        cache_write_tokens=200,
    )

    models = tracker.get_per_model_usage()
    assert models["claude-sonnet-4-6"].cache_read_tokens == 800
    assert models["claude-sonnet-4-6"].cache_write_tokens == 200


def test_backward_compatible_record():
    """record_usage without model param should still work."""
    tracker = UsageTracker()
    tracker.record_usage(1000, 500)
    assert tracker.session_usage.input_tokens == 1000
    assert tracker.session_usage.output_tokens == 500


def test_save_and_load(tmp_path):
    tracker = UsageTracker()
    tracker.record_usage(1000, 500, model="claude-sonnet-4-6")
    tracker.record_usage(2000, 800, model="gpt-4o")

    save_path = tmp_path / "usage.json"
    tracker.save(save_path)
    assert save_path.exists()

    # Load into new tracker
    tracker2 = UsageTracker()
    tracker2.load(save_path)
    assert tracker2.session_usage.input_tokens == 3000
    assert tracker2.session_usage.request_count == 2
    models = tracker2.get_per_model_usage()
    assert "claude-sonnet-4-6" in models
    assert "gpt-4o" in models


def test_save_creates_parent_dirs(tmp_path):
    tracker = UsageTracker()
    tracker.record_usage(100, 50, model="test")
    save_path = tmp_path / "deep" / "nested" / "usage.json"
    tracker.save(save_path)
    assert save_path.exists()


def test_load_nonexistent_file(tmp_path):
    tracker = UsageTracker()
    tracker.load(tmp_path / "nonexistent.json")
    # Should not crash, just stay empty
    assert tracker.session_usage.request_count == 0


def test_format_summary():
    tracker = UsageTracker()
    tracker.record_usage(10000, 5000, model="claude-sonnet-4-6")
    tracker.record_usage(3000, 1000, model="gpt-4o")

    summary = tracker.format_summary()
    assert "claude-sonnet-4-6" in summary or "sonnet" in summary.lower()
    assert isinstance(summary, str)
    assert len(summary) > 0


@pytest.fixture
def model_switch(monkeypatch):
    active = ["model-a"]
    monkeypatch.setattr("koder_agent.core.usage_tracker.get_model_name", lambda: active[0])
    monkeypatch.setattr(
        "koder_agent.core.usage_tracker.litellm.model_cost",
        {
            "model-a": {"input_cost_per_token": 0.01, "output_cost_per_token": 0.02},
            "model-b": {"input_cost_per_token": 0.03, "output_cost_per_token": 0.04},
        },
    )
    return active


def test_switch_refreshes_cached_pricing_without_repricing_history(model_switch):
    tracker = UsageTracker()
    tracker.record_usage(10, 5, model="model-a")
    assert tracker.model == "model-a"
    assert tracker.get_model_costs() == (0.01, 0.02)
    old_cost = tracker.session_usage.total_cost
    model_switch[0] = "model-b"

    # Exercise the cost lookup before reading the model label.
    assert tracker.get_model_costs() == (0.03, 0.04)
    assert tracker.model == "model-b"
    assert tracker.session_usage.total_cost == old_cost
    tracker.record_usage(10, 5, model="model-b")
    assert tracker.session_usage.total_cost == pytest.approx(0.7)
    assert tracker.get_per_model_usage()["model-a"].cost == pytest.approx(0.2)
    assert tracker.get_per_model_usage()["model-b"].cost == pytest.approx(0.5)


def test_explicit_request_model_controls_price_not_active_model(model_switch):
    tracker = UsageTracker()
    tracker.record_usage(10, 5, model="model-b")
    assert tracker.model == "model-a"
    assert tracker.session_usage.total_cost == pytest.approx(0.5)
    assert tracker.get_per_model_usage()["model-b"].cost == pytest.approx(0.5)


def test_implicit_model_usage_is_attributed_across_switches(model_switch):
    tracker = UsageTracker()
    tracker.record_usage(10, 5)
    model_switch[0] = "model-b"
    tracker.record_usage(10, 5)
    assert set(tracker.get_per_model_usage()) == {"model-a", "model-b"}
    assert tracker.session_usage.total_cost == pytest.approx(0.7)


def test_unknown_pricing_stays_incomplete_after_switch_and_reload(model_switch, tmp_path):
    tracker = UsageTracker()
    tracker.record_usage(10, 5, model="unknown-fixture")
    tracker.record_usage(10, 5, model="model-a")
    assert tracker.summary().cost_unavailable is True
    assert tracker.session_usage.total_cost == pytest.approx(0.2)
    rendered = tracker.format_summary()
    assert "Total Cost: unavailable" in rendered
    assert "Cost: $0.2000" in rendered

    path = tmp_path / "usage.json"
    tracker.save(path)
    loaded = UsageTracker()
    loaded.load(path)
    model_switch[0] = "model-b"
    assert loaded.summary().cost_unavailable is True
    assert loaded.session_usage.total_cost == pytest.approx(0.2)
    assert loaded.get_per_model_usage()["model-a"].cost == pytest.approx(0.2)
    loaded.reset()
    assert loaded.summary().cost_unavailable is False
    assert loaded.get_per_model_usage() == {}


@pytest.mark.parametrize("with_model_breakdown", [False, True])
def test_legacy_unpriced_usage_is_not_made_free_by_switching_models(
    model_switch, tmp_path, with_model_breakdown
):
    import json

    data = {
        "session_usage": {
            "request_count": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_cost": 0.0,
        },
        "per_model": {},
    }
    if with_model_breakdown:
        data["per_model"]["unknown-fixture"] = {
            "model": "unknown-fixture",
            "request_count": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "cost": 0.0,
        }
    path = tmp_path / "legacy-usage.json"
    path.write_text(json.dumps(data))
    tracker = UsageTracker()
    tracker.load(path)

    assert tracker.model == "model-a"
    assert tracker.summary().cost_unavailable is True
    assert "Cost: $0.0000" not in tracker.format_summary()


def test_cached_rates_are_keyed_by_model_and_reused(model_switch, monkeypatch):
    from unittest.mock import Mock

    import litellm

    prices = Mock(wraps=litellm.model_cost)
    monkeypatch.setattr(litellm, "model_cost", prices)
    tracker = UsageTracker()
    assert tracker.get_model_costs() == (0.01, 0.02)
    model_switch[0] = "model-b"
    assert tracker.get_model_costs() == (0.03, 0.04)
    model_switch[0] = "model-a"
    assert tracker.get_model_costs() == (0.01, 0.02)
    assert prices.get.call_count == 2
