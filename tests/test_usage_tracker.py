"""Tests for UsageTracker and cost calculation."""

from unittest.mock import Mock, patch

import pytest

from koder_agent.core.usage_tracker import (
    SessionUsage,
    UsageSummary,
    UsageTracker,
    usage_snapshot_path,
)


class TestSessionUsage:
    """Tests for SessionUsage dataclass."""

    def test_default_values(self):
        """Test default values for SessionUsage."""
        usage = SessionUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_cost == 0.0
        assert usage.request_count == 0
        assert usage.last_input_tokens == 0
        assert usage.last_output_tokens == 0
        assert usage.current_context_tokens == 0

    def test_custom_values(self):
        """Test custom values for SessionUsage."""
        usage = SessionUsage(
            input_tokens=1000,
            output_tokens=500,
            total_cost=0.05,
            request_count=5,
            last_input_tokens=200,
            last_output_tokens=100,
            current_context_tokens=1500,
        )
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 500
        assert usage.total_cost == 0.05
        assert usage.request_count == 5


class TestUsageTracker:
    """Tests for UsageTracker class."""

    def test_initialization(self):
        """Test UsageTracker initializes with empty session usage."""
        tracker = UsageTracker()
        assert tracker.session_usage.input_tokens == 0
        assert tracker.session_usage.output_tokens == 0
        assert tracker.session_usage.total_cost == 0.0
        assert tracker.session_usage.request_count == 0

    def test_model_property_observes_active_configuration(self):
        """Model selection is live; only per-model prices are cached."""
        tracker = UsageTracker()
        with patch(
            "koder_agent.core.usage_tracker.get_model_name", side_effect=["gpt-4o", "gpt-4.1"]
        ):
            assert tracker.model == "gpt-4o"
            assert tracker.model == "gpt-4.1"


def test_usage_snapshot_path_escapes_session_ids(tmp_path):
    path = usage_snapshot_path("session/with spaces", home=tmp_path)

    assert path == tmp_path / ".koder" / "usage" / "session%2Fwith%20spaces.json"


class TestGetModelCosts:
    """Tests for get_model_costs method."""

    def test_costs_cached_after_first_lookup(self):
        """Test that costs are cached after first lookup."""
        tracker = UsageTracker()

        # First call
        costs1 = tracker.get_model_costs("gpt-4o")
        # Second call should use cache
        costs2 = tracker.get_model_costs("gpt-4o")

        assert costs1 == costs2
        assert tracker._cached_costs["gpt-4o"] == costs1

    def test_unknown_model_returns_zero_costs(self):
        """Test that unknown models return zero costs."""
        tracker = UsageTracker()
        input_cost, output_cost = tracker.get_model_costs("totally-unknown-model-xyz-99999")
        assert input_cost == 0.0
        assert output_cost == 0.0

    def test_dot_to_hyphen_model_lookup(self, monkeypatch):
        """Test that models with dots are looked up with hyphens."""
        import litellm

        # Mock litellm.model_cost to have the hyphenated version
        mock_model_cost = {
            "claude-opus-4-5": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000025,
            }
        }
        monkeypatch.setattr(litellm, "model_cost", mock_model_cost)

        tracker = UsageTracker()
        input_cost, output_cost = tracker.get_model_costs("claude-opus-4.5")
        # Should find the model via hyphen variant
        assert input_cost == 0.000005
        assert output_cost == 0.000025

    def test_prefixed_model_finds_costs(self, monkeypatch):
        """Test that prefixed model names find costs through variants."""
        import litellm

        # Mock litellm.model_cost to have the base model name
        mock_model_cost = {
            "claude-opus-4-5": {
                "input_cost_per_token": 0.000005,
                "output_cost_per_token": 0.000025,
            }
        }
        monkeypatch.setattr(litellm, "model_cost", mock_model_cost)

        tracker = UsageTracker()
        input_cost, output_cost = tracker.get_model_costs("litellm/github_copilot/claude-opus-4.5")
        # Should find costs via the variant "claude-opus-4-5"
        assert input_cost == 0.000005
        assert output_cost == 0.000025


class TestCalculateCost:
    """Tests for calculate_cost method."""

    def test_calculate_cost_with_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.00001, 0.00003))

        cost = tracker.calculate_cost(0, 0)
        assert cost == 0.0

    def test_calculate_cost_with_known_rates(self):
        """Test cost calculation with known rates."""
        tracker = UsageTracker()
        # Set known rates: $10/1M input, $30/1M output
        tracker.get_model_costs = Mock(return_value=(0.00001, 0.00003))

        cost = tracker.calculate_cost(1000, 500)
        # 1000 * 0.00001 + 500 * 0.00003 = 0.01 + 0.015 = 0.025
        assert cost == pytest.approx(0.025)

    def test_calculate_cost_with_zero_rates(self):
        """Test cost calculation with zero rates (unknown model)."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        cost = tracker.calculate_cost(10000, 5000)
        assert cost == 0.0


class TestRecordUsage:
    """Tests for record_usage method."""

    def test_record_usage_accumulates_tokens(self):
        """Test that record_usage accumulates tokens correctly."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        tracker.record_usage(100, 50)
        tracker.record_usage(200, 100)

        assert tracker.session_usage.input_tokens == 300
        assert tracker.session_usage.output_tokens == 150
        assert tracker.session_usage.request_count == 2

    def test_record_usage_tracks_last_call(self):
        """Test that record_usage tracks the last call's tokens."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        tracker.record_usage(100, 50)
        tracker.record_usage(200, 100)

        assert tracker.session_usage.last_input_tokens == 200
        assert tracker.session_usage.last_output_tokens == 100

    def test_record_usage_accumulates_cost(self):
        """Test that record_usage accumulates costs."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.00001, 0.00003))

        tracker.record_usage(1000, 500)  # 0.01 + 0.015 = 0.025
        tracker.record_usage(1000, 500)  # 0.025 more

        assert tracker.session_usage.total_cost == pytest.approx(0.05)

    def test_record_usage_with_explicit_context_tokens(self):
        """Test record_usage with explicit context_tokens parameter."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        tracker.record_usage(100, 50, context_tokens=500)
        assert tracker.session_usage.current_context_tokens == 500

    def test_record_usage_defaults_context_to_input_plus_output(self):
        """Test that context_tokens defaults to input + output."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        tracker.record_usage(100, 50)
        assert tracker.session_usage.current_context_tokens == 150  # 100 + 50


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_session_usage(self):
        """Test that reset clears all session usage data."""
        tracker = UsageTracker()
        tracker.get_model_costs("gpt-4o")
        assert tracker._cached_costs

        # Record some usage
        tracker.record_usage(1000, 500)
        tracker.record_usage(2000, 1000)

        # Reset
        tracker.reset()

        assert tracker.session_usage.input_tokens == 0
        assert tracker.session_usage.output_tokens == 0
        assert tracker.session_usage.total_cost == 0.0
        assert tracker.session_usage.request_count == 0
        assert tracker._cached_costs == {}


class TestSessionCacheTokens:
    """Tests that cache-read/write tokens accumulate at the session level."""

    def test_record_usage_accumulates_session_cache_tokens(self):
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        tracker.record_usage(100, 50, cache_read_tokens=800, cache_write_tokens=40, model="m")
        tracker.record_usage(200, 100, cache_read_tokens=1200, cache_write_tokens=0, model="m")

        assert tracker.session_usage.cache_read_tokens == 2000
        assert tracker.session_usage.cache_write_tokens == 40

    def test_session_cache_defaults_to_zero(self):
        usage = SessionUsage()
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0


class TestPricingKnown:
    """Tests for the pricing_known helper."""

    def test_pricing_known_true_with_rates(self):
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.00001, 0.00003))
        assert tracker.pricing_known() is True

    def test_pricing_known_false_with_zero_rates(self):
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))
        assert tracker.pricing_known() is False


class TestSummary:
    """Tests for the /cost-style UsageSummary snapshot."""

    def test_summary_includes_cache_read_split(self):
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.00001, 0.00003))

        tracker.record_usage(
            1000, 500, cache_read_tokens=4000, cache_write_tokens=200, model="gpt-x"
        )

        summary = tracker.summary()
        assert isinstance(summary, UsageSummary)
        assert summary.input_tokens == 1000
        assert summary.output_tokens == 500
        assert summary.cache_read_tokens == 4000
        assert summary.cache_write_tokens == 200
        assert summary.fresh_input_tokens == 1000
        # 1000 * 0.00001 + 500 * 0.00003 = 0.025
        assert summary.total_cost == pytest.approx(0.025)
        assert summary.cost_unavailable is False

    def test_summary_marks_cost_unavailable_for_unknown_pricing(self):
        """Subscription/OAuth-style: tokens flow but per-token price is 0."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        tracker.record_usage(5000, 2000, cache_read_tokens=1000, model="oauth-model")

        summary = tracker.summary()
        assert summary.input_tokens == 5000
        assert summary.output_tokens == 2000
        assert summary.cache_read_tokens == 1000
        assert summary.total_cost == 0.0
        assert summary.cost_unavailable is True

    def test_summary_cost_available_when_positive_even_if_pricing_lookup_zero(self):
        """A recorded positive cost should not be flagged unavailable."""
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))
        # Simulate a cost that was recorded some other way.
        tracker.session_usage.total_cost = 0.42

        summary = tracker.summary()
        assert summary.cost_unavailable is False
        assert summary.total_cost == pytest.approx(0.42)

    def test_summary_backfills_cache_from_per_model_snapshot(self, tmp_path):
        """Older snapshots may lack session-level cache counters; backfill them."""
        tracker = UsageTracker()
        # Simulate a loaded snapshot: session-level cache is zero, per-model has data.
        tracker.session_usage = SessionUsage(input_tokens=100, output_tokens=50)
        from koder_agent.core.usage_tracker import ModelUsage

        tracker._per_model = {
            "m": ModelUsage(model="m", cache_read_tokens=777, cache_write_tokens=11),
        }
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))

        summary = tracker.summary()
        assert summary.cache_read_tokens == 777
        assert summary.cache_write_tokens == 11


class TestFormatSummaryText:
    """Tests for the format_summary text output."""

    def test_format_summary_shows_cache_read_and_unavailable_cost(self):
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))
        tracker.record_usage(3000, 1000, cache_read_tokens=1500, model="oauth-model")

        text = tracker.format_summary()
        assert "Cache Read Tokens: 1,500" in text
        assert "unavailable" in text
        assert "$0.0000" not in text

    def test_format_summary_shows_dollar_cost_when_known(self):
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.00001, 0.00003))
        tracker.record_usage(1000, 500, model="gpt-x")

        text = tracker.format_summary()
        assert "$0.0250" in text
        assert "unavailable" not in text


class TestSnapshotRoundTripWithCache:
    """Tests that cache tokens survive save/load."""

    def test_save_and_load_preserves_session_cache_tokens(self, tmp_path):
        tracker = UsageTracker()
        tracker.get_model_costs = Mock(return_value=(0.0, 0.0))
        tracker.record_usage(100, 50, cache_read_tokens=900, cache_write_tokens=30, model="m")

        path = tmp_path / "usage.json"
        tracker.save(path)

        loaded = UsageTracker()
        loaded.load(path)
        assert loaded.session_usage.cache_read_tokens == 900
        assert loaded.session_usage.cache_write_tokens == 30

    def test_load_legacy_snapshot_without_cache_fields(self, tmp_path):
        """A snapshot written before session cache fields existed still loads."""
        import json

        path = tmp_path / "legacy.json"
        path.write_text(
            json.dumps(
                {
                    "session_usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_cost": 0.0,
                        "request_count": 1,
                        "last_input_tokens": 10,
                        "last_output_tokens": 5,
                        "current_context_tokens": 15,
                    },
                    "per_model": {},
                }
            )
        )

        tracker = UsageTracker()
        tracker.load(path)
        assert tracker.session_usage.input_tokens == 10
        # Missing cache fields default to zero.
        assert tracker.session_usage.cache_read_tokens == 0
        assert tracker.session_usage.cache_write_tokens == 0
