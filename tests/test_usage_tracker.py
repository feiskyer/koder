"""Tests for UsageTracker and cost calculation."""

from unittest.mock import patch

import pytest

from koder_agent.core.usage_tracker import SessionUsage, UsageTracker


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

    def test_model_property_caches_value(self):
        """Test that model property caches the model name."""
        tracker = UsageTracker()
        with patch("koder_agent.core.usage_tracker.get_model_name", return_value="gpt-4o") as mock:
            _ = tracker.model
            _ = tracker.model  # Second call should use cache
            assert mock.call_count == 1  # Only called once due to caching


class TestGetModelCosts:
    """Tests for get_model_costs method."""

    def test_costs_cached_after_first_lookup(self):
        """Test that costs are cached after first lookup."""
        tracker = UsageTracker()
        tracker._model = "gpt-4o"

        # First call
        costs1 = tracker.get_model_costs()
        # Second call should use cache
        costs2 = tracker.get_model_costs()

        assert costs1 == costs2
        assert tracker._cached_costs is not None

    def test_unknown_model_returns_zero_costs(self):
        """Test that unknown models return zero costs."""
        tracker = UsageTracker()
        tracker._model = "totally-unknown-model-xyz-99999"

        input_cost, output_cost = tracker.get_model_costs()
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
        tracker._model = "claude-opus-4.5"  # dot version

        input_cost, output_cost = tracker.get_model_costs()
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
        tracker._model = "litellm/github_copilot/claude-opus-4.5"

        input_cost, output_cost = tracker.get_model_costs()
        # Should find costs via the variant "claude-opus-4-5"
        assert input_cost == 0.000005
        assert output_cost == 0.000025


class TestCalculateCost:
    """Tests for calculate_cost method."""

    def test_calculate_cost_with_zero_tokens(self):
        """Test cost calculation with zero tokens."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.00001, 0.00003)

        cost = tracker.calculate_cost(0, 0)
        assert cost == 0.0

    def test_calculate_cost_with_known_rates(self):
        """Test cost calculation with known rates."""
        tracker = UsageTracker()
        # Set known rates: $10/1M input, $30/1M output
        tracker._cached_costs = (0.00001, 0.00003)

        cost = tracker.calculate_cost(1000, 500)
        # 1000 * 0.00001 + 500 * 0.00003 = 0.01 + 0.015 = 0.025
        assert cost == pytest.approx(0.025)

    def test_calculate_cost_with_zero_rates(self):
        """Test cost calculation with zero rates (unknown model)."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.0, 0.0)

        cost = tracker.calculate_cost(10000, 5000)
        assert cost == 0.0


class TestRecordUsage:
    """Tests for record_usage method."""

    def test_record_usage_accumulates_tokens(self):
        """Test that record_usage accumulates tokens correctly."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.0, 0.0)  # Zero cost for simplicity

        tracker.record_usage(100, 50)
        tracker.record_usage(200, 100)

        assert tracker.session_usage.input_tokens == 300
        assert tracker.session_usage.output_tokens == 150
        assert tracker.session_usage.request_count == 2

    def test_record_usage_tracks_last_call(self):
        """Test that record_usage tracks the last call's tokens."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.0, 0.0)

        tracker.record_usage(100, 50)
        tracker.record_usage(200, 100)

        assert tracker.session_usage.last_input_tokens == 200
        assert tracker.session_usage.last_output_tokens == 100

    def test_record_usage_accumulates_cost(self):
        """Test that record_usage accumulates costs."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.00001, 0.00003)

        tracker.record_usage(1000, 500)  # 0.01 + 0.015 = 0.025
        tracker.record_usage(1000, 500)  # 0.025 more

        assert tracker.session_usage.total_cost == pytest.approx(0.05)

    def test_record_usage_with_explicit_context_tokens(self):
        """Test record_usage with explicit context_tokens parameter."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.0, 0.0)

        tracker.record_usage(100, 50, context_tokens=500)
        assert tracker.session_usage.current_context_tokens == 500

    def test_record_usage_defaults_context_to_input_plus_output(self):
        """Test that context_tokens defaults to input + output."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.0, 0.0)

        tracker.record_usage(100, 50)
        assert tracker.session_usage.current_context_tokens == 150  # 100 + 50


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_session_usage(self):
        """Test that reset clears all session usage data."""
        tracker = UsageTracker()
        tracker._cached_costs = (0.00001, 0.00003)
        tracker._model = "gpt-4o"

        # Record some usage
        tracker.record_usage(1000, 500)
        tracker.record_usage(2000, 1000)

        # Reset
        tracker.reset()

        assert tracker.session_usage.input_tokens == 0
        assert tracker.session_usage.output_tokens == 0
        assert tracker.session_usage.total_cost == 0.0
        assert tracker.session_usage.request_count == 0
        assert tracker._model is None
        assert tracker._cached_costs is None
