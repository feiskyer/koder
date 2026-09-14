"""Model information utilities for context window and token management."""

from math import floor
from typing import Optional

from ..litellm_cost_map import get_litellm

litellm = get_litellm()


class UnknownModelContextWindowError(ValueError):
    """Raised when no authoritative model context window is available."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Unknown context window for model '{model}'. Configure model.context_window "
            "(or KODER_CONTEXT_WINDOW); auxiliary small models can use "
            "model.small_model_context_window (or KODER_SMALL_MODEL_CONTEXT_WINDOW)."
        )


# Model aliases: short names → full model IDs
# Updated to latest models as of 2026-04
MODEL_ALIASES: dict[str, str] = {
    # Claude family
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "best": "claude-opus-4-6",
    # 1M context variants
    "sonnet[1m]": "claude-sonnet-4-6",  # Sonnet doesn't have separate 1M ID
    "opus[1m]": "claude-opus-4-6[1m]",
    # GPT family shortcuts
    "gpt4o": "gpt-4o",
    "gpt5": "gpt-5.1-codex",
    "o3": "o3",
    "o4-mini": "o4-mini",
}


def resolve_model_alias(model: str) -> str:
    """Resolve a model alias to its full model ID.

    Supports short aliases like 'sonnet', 'opus', 'haiku', 'best'.
    Also supports [1m] suffix for 1M context variants.
    Unknown model names pass through unchanged.
    """
    # Try exact match (case-insensitive)
    lower = model.lower()
    if lower in MODEL_ALIASES:
        return MODEL_ALIASES[lower]

    # Try with [1m] suffix
    if lower.endswith("[1m]"):
        base = lower[:-4]  # Remove [1m]
        if base in MODEL_ALIASES:
            base_model = MODEL_ALIASES[base]
            # Append [1m] to the resolved model if not already there
            if not base_model.endswith("[1m]"):
                return f"{base_model}[1m]"
            return base_model

    # Pass through unknown model names unchanged
    return model


def get_model_name_variants_for_lookup(model: str) -> list[str]:
    """
    Generate model name variants to try when looking up in litellm.model_cost.
    Returns a list of names to try in order: exact, lowercase, without prefix, etc.
    """
    names_to_try = [model, model.lower()]

    # If there's a prefix (e.g., litellm/openai/gpt-4), try without prefixes
    if "/" in model:
        # Try the last part (model name only)
        base_model = model.rsplit("/", 1)[1]
        names_to_try.extend([base_model, base_model.lower()])

        # If there are multiple slashes, try intermediate parts too
        parts = model.split("/")
        if len(parts) > 2:
            # Try provider/model (e.g., openai/gpt-4)
            provider_model = "/".join(parts[-2:])
            names_to_try.extend([provider_model, provider_model.lower()])

    # Add variants with dots replaced by hyphens (e.g., claude-opus-4.5 -> claude-opus-4-5)
    # litellm uses hyphens in model names like "claude-opus-4-5"
    dot_to_hyphen_variants = []
    for name in names_to_try:
        if "." in name:
            dot_to_hyphen_variants.append(name.replace(".", "-"))
    names_to_try.extend(dot_to_hyphen_variants)

    # Remove duplicates while preserving order
    return list(dict.fromkeys(names_to_try))


def get_context_window_size(model: str, max_context_size: Optional[int] = None) -> int:
    """
    Get the context window size for a model.

    Priority:
    1. Custom max_context_size argument (if provided)
    2. LiteLLM's model registry (litellm.model_cost)
    3. Fail closed for unknown/custom models

    Args:
        model: The model name/identifier
        max_context_size: Optional custom context size override

    Returns:
        Context window size in tokens
    """
    if max_context_size is not None:
        return max_context_size

    name_variants = get_model_name_variants_for_lookup(model)

    for name in name_variants:
        try:
            context_size = litellm.model_cost[name]["max_input_tokens"]
            if isinstance(context_size, int) and context_size > 0:
                return context_size
        except KeyError:
            continue
        except Exception:
            continue

    raise UnknownModelContextWindowError(model)


def get_maximum_output_tokens(model: str, max_context_size: Optional[int] = None) -> int:
    """
    Get the maximum output tokens for a model.

    Calculates a reasonable output limit as: floor(min(64000, context_size / 5))
    Then checks if LiteLLM has a lower max_output_tokens and uses that.

    Args:
        model: The model name/identifier

    Returns:
        Maximum output tokens
    """
    context_size = get_context_window_size(model, max_context_size=max_context_size)
    max_output_tokens = floor(min(64000, context_size / 5))

    name_variants = get_model_name_variants_for_lookup(model)
    for name in name_variants:
        try:
            litellm_max = litellm.model_cost[name]["max_output_tokens"]
            if litellm_max < max_output_tokens:
                max_output_tokens = litellm_max
            return max_output_tokens
        except Exception:
            continue

    return max_output_tokens


def get_summarization_threshold(
    model: str,
    threshold_ratio: float = 0.8,
    max_context_size: Optional[int] = None,
) -> int:
    """
    Get the token threshold at which context summarization should be triggered.

    Args:
        model: The model name/identifier
        threshold_ratio: Ratio of context window to use as threshold (default 0.8 = 80%)

    Returns:
        Token count threshold for triggering summarization
    """
    context_size = get_context_window_size(model, max_context_size=max_context_size)
    return int(context_size * threshold_ratio)


def should_use_reasoning_param() -> bool:
    """
    Check if the current provider/model configuration supports the Reasoning parameter.

    The Reasoning object from openai.types.shared is only compatible with:
    - Native OpenAI API calls (not through LiteLLM)
    - Models that actually support reasoning (o1, o3, o4, gpt-5 series, etc.)

    When using LiteLLM with providers like GitHub Copilot, Anthropic, etc.,
    the Reasoning object causes schema validation errors.

    Returns:
        True if Reasoning parameter should be used, False otherwise
    """
    # Import here to avoid circular imports
    from .client import is_native_openai_provider

    # Only use Reasoning parameter when using native OpenAI provider
    # For LiteLLM-based providers (including GitHub Copilot), skip it
    return is_native_openai_provider()
