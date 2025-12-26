"""Tests for model_info utilities."""

import pytest

from koder_agent.utils.model_info import (
    get_context_window_size,
    get_maximum_output_tokens,
    get_model_name_variants_for_lookup,
    get_summarization_threshold,
)


class TestGetModelNameVariantsForLookup:
    """Tests for get_model_name_variants_for_lookup function."""

    def test_simple_model_name(self):
        """Test with a simple model name without prefixes."""
        variants = get_model_name_variants_for_lookup("gpt-4o")
        assert variants == ["gpt-4o"]

    def test_simple_model_name_with_uppercase(self):
        """Test that lowercase variant is added."""
        variants = get_model_name_variants_for_lookup("GPT-4o")
        assert "GPT-4o" in variants
        assert "gpt-4o" in variants

    def test_model_with_single_prefix(self):
        """Test model with provider prefix (e.g., openai/gpt-4)."""
        variants = get_model_name_variants_for_lookup("openai/gpt-4o")
        assert "openai/gpt-4o" in variants
        assert "gpt-4o" in variants

    def test_model_with_multiple_prefixes(self):
        """Test model with multiple prefixes (e.g., litellm/openai/gpt-4)."""
        variants = get_model_name_variants_for_lookup("litellm/openai/gpt-4o")
        assert "litellm/openai/gpt-4o" in variants
        assert "gpt-4o" in variants  # last part
        assert "openai/gpt-4o" in variants  # provider/model

    def test_github_copilot_model(self):
        """Test GitHub Copilot style model name."""
        variants = get_model_name_variants_for_lookup("litellm/github_copilot/claude-opus-4.5")
        assert "litellm/github_copilot/claude-opus-4.5" in variants
        assert "claude-opus-4.5" in variants
        assert "github_copilot/claude-opus-4.5" in variants

    def test_dot_to_hyphen_conversion(self):
        """Test that dots are converted to hyphens for litellm compatibility."""
        variants = get_model_name_variants_for_lookup("claude-opus-4.5")
        assert "claude-opus-4.5" in variants
        assert "claude-opus-4-5" in variants  # dot -> hyphen

    def test_dot_to_hyphen_with_prefixes(self):
        """Test dot-to-hyphen conversion with prefixed model names."""
        variants = get_model_name_variants_for_lookup("litellm/github_copilot/claude-opus-4.5")
        # Original variants
        assert "claude-opus-4.5" in variants
        # Hyphenated variants
        assert "claude-opus-4-5" in variants
        assert "litellm/github_copilot/claude-opus-4-5" in variants
        assert "github_copilot/claude-opus-4-5" in variants

    def test_multiple_dots_converted(self):
        """Test that multiple dots are all converted to hyphens."""
        variants = get_model_name_variants_for_lookup("model-1.2.3")
        assert "model-1.2.3" in variants
        assert "model-1-2-3" in variants

    def test_no_duplicates(self):
        """Test that duplicates are removed from variants."""
        variants = get_model_name_variants_for_lookup("gpt-4o")
        # Should not have duplicates
        assert len(variants) == len(set(variants))

    def test_openrouter_model(self):
        """Test OpenRouter style model name."""
        variants = get_model_name_variants_for_lookup("openrouter/anthropic/claude-3-opus")
        assert "openrouter/anthropic/claude-3-opus" in variants
        assert "claude-3-opus" in variants
        assert "anthropic/claude-3-opus" in variants

    def test_model_without_dots_no_extra_variants(self):
        """Test that models without dots don't get extra hyphen variants."""
        variants = get_model_name_variants_for_lookup("claude-3-opus")
        # Should not have duplicate entries from dot-to-hyphen conversion
        assert variants.count("claude-3-opus") == 1


class TestGetContextWindowSize:
    """Tests for get_context_window_size function."""

    def test_custom_max_context_size_override(self):
        """Test that custom max_context_size is returned when provided."""
        result = get_context_window_size("any-model", max_context_size=100000)
        assert result == 100000

    def test_fallback_to_default(self, monkeypatch):
        """Test fallback to default when model not in litellm registry."""
        # Use a fake model name that won't exist
        result = get_context_window_size("nonexistent-model-xyz-12345")
        # Should return the fallback default (32000)
        assert result == 32000

    def test_known_model_returns_correct_size(self, monkeypatch):
        """Test that known models return their context window size from litellm."""
        import litellm

        # Mock litellm.model_cost to have a known model
        mock_model_cost = {"gpt-4o": {"max_input_tokens": 128000}}
        monkeypatch.setattr(litellm, "model_cost", mock_model_cost)

        result = get_context_window_size("gpt-4o")
        assert result == 128000


class TestGetMaximumOutputTokens:
    """Tests for get_maximum_output_tokens function."""

    def test_output_tokens_calculated_from_context(self):
        """Test that max output tokens is calculated from context size."""
        # For a model with 32000 context, max_output should be floor(32000/5) = 6400
        result = get_maximum_output_tokens("nonexistent-model-xyz-12345")
        assert result == 6400  # floor(32000 / 5)

    def test_output_tokens_capped_at_64000(self):
        """Test that max output tokens is capped at 64000."""
        # For very large context models, output should be capped
        result = get_maximum_output_tokens("gpt-4o")
        assert result <= 64000


class TestGetSummarizationThreshold:
    """Tests for get_summarization_threshold function."""

    def test_default_threshold_ratio(self):
        """Test default threshold ratio of 0.8."""
        # For 32000 context, threshold should be 32000 * 0.8 = 25600
        result = get_summarization_threshold("nonexistent-model-xyz-12345")
        assert result == 25600

    def test_custom_threshold_ratio(self):
        """Test custom threshold ratio."""
        result = get_summarization_threshold("nonexistent-model-xyz-12345", threshold_ratio=0.5)
        assert result == 16000  # 32000 * 0.5
