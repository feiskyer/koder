"""Token usage and cost tracking for API calls."""

from dataclasses import dataclass
from typing import Optional

import litellm

from ..utils.client import get_model_name
from ..utils.model_info import get_model_name_variants_for_lookup


@dataclass
class SessionUsage:
    """Tracks cumulative token usage and cost for a session."""

    input_tokens: int = 0  # Cumulative input tokens
    output_tokens: int = 0  # Cumulative output tokens
    total_cost: float = 0.0
    request_count: int = 0
    last_input_tokens: int = 0  # Last API call's input tokens
    last_output_tokens: int = 0  # Last API call's output tokens
    current_context_tokens: int = 0  # Estimated current context size (tokens)


class UsageTracker:
    """Tracks and calculates token usage and costs."""

    def __init__(self):
        self.session_usage = SessionUsage()
        self._model: Optional[str] = None
        self._cached_costs: Optional[tuple[float, float]] = None

    @property
    def model(self) -> str:
        """Get the current model name, caching for performance."""
        if self._model is None:
            self._model = get_model_name()
        return self._model

    def get_model_costs(self) -> tuple[float, float]:
        """
        Get input and output cost per token for current model.

        Returns:
            Tuple of (input_cost_per_token, output_cost_per_token)
        """
        # Return cached costs if available (costs don't change per request)
        if self._cached_costs is not None:
            return self._cached_costs

        variants = get_model_name_variants_for_lookup(self.model)

        for name in variants:
            try:
                info = litellm.model_cost.get(name, {})
                input_cost = info.get("input_cost_per_token", 0.0)
                output_cost = info.get("output_cost_per_token", 0.0)
                if input_cost > 0 or output_cost > 0:
                    self._cached_costs = (input_cost, output_cost)
                    return self._cached_costs
            except Exception:
                continue

        self._cached_costs = (0.0, 0.0)
        return self._cached_costs

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost for given token counts.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Total cost in USD
        """
        input_cost_per_token, output_cost_per_token = self.get_model_costs()
        return (input_tokens * input_cost_per_token) + (output_tokens * output_cost_per_token)

    def record_usage(
        self, input_tokens: int, output_tokens: int, context_tokens: Optional[int] = None
    ) -> None:
        """
        Record usage from an API call.

        Args:
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            context_tokens: Optional explicit context size (if different from input+output)
        """
        cost = self.calculate_cost(input_tokens, output_tokens)
        self.session_usage.input_tokens += input_tokens
        self.session_usage.output_tokens += output_tokens
        self.session_usage.total_cost += cost
        self.session_usage.request_count += 1
        self.session_usage.last_input_tokens = input_tokens
        self.session_usage.last_output_tokens = output_tokens

        if context_tokens is not None:
            self.session_usage.current_context_tokens = context_tokens
        else:
            # Fallback: assume context is roughly input + output of the run
            self.session_usage.current_context_tokens = input_tokens + output_tokens

    def reset(self) -> None:
        """Reset session usage (for new session)."""
        self.session_usage = SessionUsage()
        self._model = None
        self._cached_costs = None
