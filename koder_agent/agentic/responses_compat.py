"""Small compatibility helpers for Koder's LiteLLM Responses transport.

Keep provider payloads out of failure messages. These helpers mirror the SDK's
terminal-response, usage, and stream-lifetime contracts without depending on
private runner state or changing Koder's provider/permission architecture.
"""

import asyncio
import inspect
import logging
from collections.abc import Mapping
from typing import Any

from agents import ModelBehaviorError
from agents.usage import Usage
from openai import NotGiven, Omit
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from ..harness.memory.budget import ContextPreflightError, estimate_model_request_preflight
from ..utils.model_info import get_maximum_output_tokens

logger = logging.getLogger(__name__)
_cleanup_tasks: set[asyncio.Task] = set()


def present_request_value(value: Any) -> Any:
    """Treat all SDK omission sentinels as absent, not as request content."""
    return None if isinstance(value, (Omit, NotGiven)) else value


def check_responses_request_budget(
    create_kwargs: Mapping[str, Any], *, context_window: int, model: str
) -> None:
    """Check the fully assembled request immediately before provider I/O."""
    reserve = present_request_value(create_kwargs.get("max_output_tokens"))
    if reserve is None:
        reserve = get_maximum_output_tokens(model, max_context_size=context_window)
    extra_payload = {
        key: present_request_value(create_kwargs.get(key))
        for key in ("prompt", "extra_body", "context_management")
        if present_request_value(create_kwargs.get(key)) is not None
    }
    estimate = estimate_model_request_preflight(
        context_window=context_window,
        response_reserve=int(reserve),
        instructions=present_request_value(create_kwargs.get("instructions")),
        input_items=present_request_value(create_kwargs.get("input")),
        tools=present_request_value(create_kwargs.get("tools")),
        response_format=present_request_value(create_kwargs.get("text")),
        extra_payload=extra_payload,
        model=model,
    )
    if not estimate.fits:
        raise ContextPreflightError(estimate, subject="Provider request")


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def responses_usage(raw_usage: Any) -> Usage:
    """Count a completed request even when the provider omits token usage."""
    input_details = _field(raw_usage, "input_tokens_details")
    output_details = _field(raw_usage, "output_tokens_details")
    return Usage(
        requests=1,
        input_tokens=_field(raw_usage, "input_tokens", 0) or 0,
        output_tokens=_field(raw_usage, "output_tokens", 0) or 0,
        total_tokens=_field(raw_usage, "total_tokens", 0) or 0,
        input_tokens_details=InputTokensDetails(
            cached_tokens=_field(input_details, "cached_tokens", 0) or 0,
            cache_write_tokens=_field(input_details, "cache_write_tokens", 0) or 0,
        ),
        output_tokens_details=OutputTokensDetails(
            reasoning_tokens=_field(output_details, "reasoning_tokens", 0) or 0
        ),
    )


def terminal_response_error(
    response: Any = None, *, event_type: str | None = None
) -> ModelBehaviorError | None:
    """Return a payload-free SDK error for terminal provider failures."""
    status = _field(response, "status")
    if status in {"failed", "incomplete", "cancelled"}:
        return ModelBehaviorError(f"Responses API returned terminal status '{status}'.")
    if event_type in {"response.failed", "response.incomplete", "error", "response.error"}:
        return ModelBehaviorError(f"Responses API returned terminal event '{event_type}'.")
    return None


def _cleanup_done(task: asyncio.Task) -> None:
    _cleanup_tasks.discard(task)
    if not task.cancelled() and (error := task.exception()) is not None:
        logger.debug("Provider stream cleanup failed (%s)", type(error).__name__)


async def close_provider_stream(stream: Any, *, suppress_errors: bool = False) -> None:
    """Close provider transport once; let an in-flight close finish on cancel.

    This helper is only for the raw provider transport, not SDK generators with
    task-local tracing scopes. Those must be closed in their consuming task.
    """
    close = getattr(stream, "aclose", None)
    if not callable(close):
        close = getattr(stream, "close", None)
    if not callable(close):
        return

    async def finish_close():
        result = close()
        if inspect.isawaitable(result):
            await result

    task = asyncio.create_task(finish_close())
    _cleanup_tasks.add(task)
    task.add_done_callback(_cleanup_done)
    try:
        await asyncio.shield(task)
    except Exception:
        if not suppress_errors:
            raise
