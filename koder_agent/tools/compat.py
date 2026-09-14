"""Compatibility helpers for SDK-decorated tools."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any

from agents import FunctionTool
from agents import function_tool as _agents_function_tool
from agents.tool import default_tool_error_function
from agents.tool_context import ToolContext

from ..core.display_context import tool_display_call_scope
from ..harness.execution_context import capture_tool_directory, get_execution_cwd
from .permission_context import (
    begin_tool_invocation,
    enforce_tool_permission,
    reset_tool_invocation,
)

logger = logging.getLogger(__name__)

# Default maximum number of characters returned by a string-valued tool.
# Configurable via ``KODER_MAX_TOOL_OUTPUT_CHARS``.
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 30000
_TOOL_OUTPUT_TOO_LARGE_MINIMAL = '{"error":"tool_output_too_large"}'
MIN_TOOL_OUTPUT_ERROR_CHARS = len(_TOOL_OUTPUT_TOO_LARGE_MINIMAL)
_MAX_JSON_DEPTH = 64
_MAX_JSON_COLLECTION_ITEMS = 10000
_MAX_JSON_TOTAL_ITEMS = 100000
_MAX_JSON_PARSE_CHARS = 1_000_000


class StructuredOutputTooLargeError(ValueError):
    """Raised when structured data exceeds safe parsing or serialization limits."""


class DuplicateJSONKeyError(ValueError):
    """Raised when duplicate object keys are forbidden by the caller."""

    def __init__(self, key: str) -> None:
        super().__init__(f"duplicate JSON object key: {key}")
        self.key = key


class InvalidJSONConstantError(ValueError):
    """Raised for non-standard JSON constants such as NaN and Infinity."""


class _JSONObjectPairs(list[tuple[str, Any]]):
    """Object representation that preserves duplicate keys during validation."""


class _JSONNumber(str):
    """Validated number lexeme that avoids Python's huge-integer conversion limit."""


def _get_max_tool_output_chars() -> int:
    """Resolve the tool-output truncation threshold from the environment.

    Falls back to :data:`DEFAULT_MAX_TOOL_OUTPUT_CHARS` when the env var is unset,
    empty, non-numeric, or non-positive.
    """

    raw = os.environ.get("KODER_MAX_TOOL_OUTPUT_CHARS")
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_TOOL_OUTPUT_CHARS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_TOOL_OUTPUT_CHARS
    if value <= 0:
        return DEFAULT_MAX_TOOL_OUTPUT_CHARS
    return value


def _truncate_text_output(result: str, max_chars: int) -> str:
    """Fit plain text within ``max_chars`` using a head/tail marker."""
    total = len(result)
    if total <= max_chars:
        return result

    kept = max_chars
    marker = ""
    for _ in range(4):
        removed = total - kept
        marker = f"\n...[truncated {removed} characters]...\n"
        next_kept = max(0, max_chars - len(marker))
        if next_kept == kept:
            break
        kept = next_kept

    if not marker or len(marker) > max_chars:
        short_marker = f"...[truncated {total} chars]..."
        return short_marker[:max_chars]

    head_len = (kept * 7) // 10
    tail_len = kept - head_len
    head = result[:head_len]
    tail = result[total - tail_len :] if tail_len > 0 else ""
    return f"{head}{marker}{tail}"


def tool_output_too_large_json(max_chars: int, *, original_chars: int | None = None) -> str:
    """Return an explicit error, using a fixed minimum for impossibly tiny caps."""
    payload: dict[str, Any] = {"error": "tool_output_too_large"}
    if original_chars is not None:
        payload.update(original_chars=original_chars, max_chars=max_chars)
        detailed = json.dumps(payload, separators=(",", ":"))
        if len(detailed) <= max_chars:
            return detailed
    return _TOOL_OUTPUT_TOO_LARGE_MINIMAL


def _validate_native_json(value: Any) -> None:
    """Reject deep, wide, cyclic, or non-JSON-native values without recursion."""
    stack = [(value, 0)]
    seen_containers: set[int] = set()
    total_items = 0

    while stack:
        current, depth = stack.pop()
        if isinstance(current, (str, int, float, bool)) or current is None:
            continue
        if not isinstance(current, (dict, list)):
            raise StructuredOutputTooLargeError
        if depth >= _MAX_JSON_DEPTH:
            raise StructuredOutputTooLargeError

        identity = id(current)
        if identity in seen_containers:
            raise StructuredOutputTooLargeError
        seen_containers.add(identity)

        size = len(current)
        if size > _MAX_JSON_COLLECTION_ITEMS:
            raise StructuredOutputTooLargeError
        total_items += size
        if total_items > _MAX_JSON_TOTAL_ITEMS:
            raise StructuredOutputTooLargeError

        if isinstance(current, dict):
            if not all(isinstance(key, str) for key in current):
                raise StructuredOutputTooLargeError
            stack.extend((item, depth + 1) for item in current.values())
        else:
            stack.extend((item, depth + 1) for item in current)


def serialize_bounded_json(value: Any, max_chars: int) -> str:
    """Serialize native JSON once, replacing unsafe or oversized values with an error."""
    try:
        _validate_native_json(value)
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (MemoryError, OverflowError, RecursionError, TypeError, ValueError):
        return tool_output_too_large_json(max_chars)
    if len(serialized) > max_chars:
        return tool_output_too_large_json(max_chars, original_chars=len(serialized))
    return serialized


def parse_json_with_limits(data: str, *, reject_duplicate_keys: bool = False) -> Any:
    """Validate a bounded JSON document without collapsing duplicate keys.

    Number hooks retain the already-validated JSON lexeme instead of converting
    arbitrarily large integers to Python ``int`` objects. Generic callers receive
    objects as ordered key/value pairs, preserving duplicates. Callers that need
    normal native objects may request duplicate rejection, in which case objects
    are converted to dictionaries only after uniqueness is established.
    """
    if len(data) > _MAX_JSON_PARSE_CHARS:
        raise StructuredOutputTooLargeError

    object_items = 0

    def _bounded_object(pairs: list[tuple[str, Any]]) -> dict[str, Any] | _JSONObjectPairs:
        nonlocal object_items
        object_items += len(pairs)
        if len(pairs) > _MAX_JSON_COLLECTION_ITEMS or object_items > _MAX_JSON_TOTAL_ITEMS:
            raise StructuredOutputTooLargeError
        seen: set[str] = set()
        for key, _value in pairs:
            if key in seen and reject_duplicate_keys:
                raise DuplicateJSONKeyError(key)
            seen.add(key)
        if reject_duplicate_keys:
            return dict(pairs)
        return _JSONObjectPairs(pairs)

    def _invalid_constant(_value: str) -> None:
        raise InvalidJSONConstantError

    try:
        parsed = json.loads(
            data,
            object_pairs_hook=_bounded_object,
            parse_int=_JSONNumber,
            parse_float=_JSONNumber,
            parse_constant=_invalid_constant,
        )
    except (
        json.JSONDecodeError,
        DuplicateJSONKeyError,
        InvalidJSONConstantError,
        StructuredOutputTooLargeError,
    ):
        raise
    except (MemoryError, OverflowError, RecursionError, ValueError) as exc:
        raise StructuredOutputTooLargeError from exc

    stack = [(parsed, 0)]
    total_items = 0
    while stack:
        current, depth = stack.pop()
        if isinstance(current, (str, int, float, bool)) or current is None:
            continue
        if depth >= _MAX_JSON_DEPTH:
            raise StructuredOutputTooLargeError
        if isinstance(current, _JSONObjectPairs):
            size = len(current)
            total_items += size
            stack.extend((value, depth + 1) for _key, value in current)
        elif isinstance(current, dict):
            size = len(current)
            total_items += size
            stack.extend((value, depth + 1) for value in current.values())
        elif isinstance(current, list):
            size = len(current)
            total_items += size
            stack.extend((value, depth + 1) for value in current)
        else:
            raise StructuredOutputTooLargeError
        if size > _MAX_JSON_COLLECTION_ITEMS or total_items > _MAX_JSON_TOTAL_ITEMS:
            raise StructuredOutputTooLargeError
    return parsed


def _bound_tool_output(result: Any, max_chars: int) -> Any:
    """Bound strings while preserving SDK-native structured output types."""
    if not isinstance(result, str):
        return result
    if len(result) <= max_chars:
        return result

    if not _has_plausible_json_root(result):
        return _truncate_text_output(result, max_chars)

    try:
        parse_json_with_limits(result)
    except (json.JSONDecodeError, InvalidJSONConstantError):
        return _truncate_text_output(result, max_chars)
    except (MemoryError, RecursionError, StructuredOutputTooLargeError):
        return tool_output_too_large_json(max_chars, original_chars=len(result))

    # JSON scalars are intentionally treated the same as objects and lists. A
    # syntactically valid all-digit identifier is therefore JSON, deterministically,
    # and receives an explicit error instead of being corrupted by text truncation.
    return tool_output_too_large_json(max_chars, original_chars=len(result))


def _has_plausible_json_root(data: str) -> bool:
    """Return whether ``data`` starts like a JSON value after JSON whitespace.

    This lightweight classification runs before the structured parser's work
    limit. Clearly ordinary oversized text therefore keeps the useful head/tail
    contract instead of being mislabeled as an oversized structured payload.
    """
    index = 0
    while index < len(data) and data[index] in " \t\r\n":
        index += 1
    if index >= len(data):
        return False

    root = data[index]
    if root in '{["0123456789':
        return True
    if root == "-":
        return index + 1 < len(data) and data[index + 1].isdigit()
    return data.startswith(("true", "false", "null"), index)


def _validate_declared_tool_arguments(tool: FunctionTool, input_json: str) -> str | None:
    """Reject undeclared top-level fields before permission checks or invocation.

    The SDK emits ``additionalProperties: false`` in strict schemas, but its
    generated Pydantic model ignores extras during direct ``on_invoke_tool``
    calls. Enforce the advertised schema here so an ignored field cannot steer
    permission extraction while a different declared field reaches the tool.
    """
    try:
        arguments = json.loads(input_json) if input_json else {}
    except (TypeError, ValueError):
        return None  # Let the SDK produce its normal malformed-JSON error.
    if not isinstance(arguments, dict):
        return None

    schema = tool.params_json_schema if isinstance(tool.params_json_schema, dict) else {}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None
    undeclared = sorted(set(arguments) - set(properties))
    if not undeclared:
        return None

    fields = ", ".join(undeclared)
    return f"Invalid JSON input for tool {tool.name}: undeclared argument(s): {fields}"


def _wrap_none_context(tool: FunctionTool) -> FunctionTool:
    """Allow direct test/harness invocation of SDK tools without a runner context."""

    original_on_invoke_tool = tool.on_invoke_tool

    @wraps(original_on_invoke_tool)
    async def _on_invoke_tool(ctx: ToolContext[Any] | None, input_json: str) -> Any:
        invocation_token = begin_tool_invocation()
        try:
            max_chars = _get_max_tool_output_chars()
            if ctx is None:
                ctx = ToolContext(
                    context=None,
                    tool_name=tool.name,
                    tool_call_id=f"manual-{tool.name}",
                    tool_arguments=input_json,
                )
            with (
                capture_tool_directory(),
                tool_display_call_scope(tool.name, getattr(ctx, "tool_call_id", None)),
            ):
                argument_error = _validate_declared_tool_arguments(tool, input_json)
                if argument_error is not None:
                    return _bound_tool_output(argument_error, max_chars)
                # Argument-level permission enforcement for the main agent chain. Returns a
                # denial string (fed back to the model) when the active permission service
                # blocks this specific call; None when allowed or when no service is active.
                blocked = await enforce_tool_permission(tool.name, input_json)
                if blocked is not None:
                    return _bound_tool_output(blocked, max_chars)
                result = await original_on_invoke_tool(ctx, input_json)
                return _bound_tool_output(result, max_chars)
        finally:
            reset_tool_invocation(invocation_token)

    tool.on_invoke_tool = _on_invoke_tool
    return tool


def _dispatching_tool_error_function(ctx: Any, error: Exception) -> str:
    """SDK ``failure_error_function`` that dispatches ``PostToolUseFailure`` hooks.

    The SDK converts tool exceptions into a model-facing error string via this
    callback; it is the one place on the main agent chain that observes every
    tool failure, so the hook event is dispatched here. Hook problems never
    mask the original tool error.
    """
    tool_name = getattr(ctx, "tool_name", None) or "unknown"
    tool_input: dict[str, Any] = {}
    raw_args = getattr(ctx, "tool_arguments", None)
    if raw_args:
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                tool_input = parsed
        except (json.JSONDecodeError, TypeError):
            pass
    try:
        from koder_agent.harness.hooks.runtime import dispatch_command_hooks

        dispatch_command_hooks(
            cwd=get_execution_cwd(),
            event_name="PostToolUseFailure",
            match_value=tool_name,
            payload={
                "event": "PostToolUseFailure",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "error": str(error),
            },
        )
    except Exception:
        logger.debug("PostToolUseFailure hook dispatch failed for %s", tool_name, exc_info=True)
    return default_tool_error_function(ctx, error)


def function_tool(func: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Wrap ``agents.function_tool`` while preserving legacy ``ctx=None`` invocations."""

    kwargs.setdefault("failure_error_function", _dispatching_tool_error_function)
    decorated = _agents_function_tool(func, **kwargs)
    if isinstance(decorated, FunctionTool):
        return _wrap_none_context(decorated)

    def _decorator(real_func: Callable[..., Any]) -> FunctionTool:
        return _wrap_none_context(decorated(real_func))

    return _decorator
