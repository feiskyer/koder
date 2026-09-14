"""Deterministic transcript compaction helpers."""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from koder_agent.utils.client import llm_completion

from .budget import estimate_messages_tokens


@dataclass(frozen=True)
class CompactionResult:
    """Result of transcript compaction."""

    summary: str | None
    kept_messages: list[dict]
    token_count: int
    original_count: int


_EASY_MESSAGE_ROLES = {"user", "assistant", "system", "developer"}
_TYPED_MESSAGE_ROLES = {"user", "assistant", "system", "developer"}
_COMPACTED_PREFIX = "[Conversation compacted]"
_POST_COMPACT_ATTACHMENT_PREFIX = "[Post-compact file restoration]"


def is_replayable_session_item(item: Any) -> bool:
    """Return whether a session item can be replayed by the SDK."""
    if not isinstance(item, dict):
        return False

    role = item.get("role")
    item_type = item.get("type")

    if set(item.keys()) == {"role", "content"}:
        return role in _EASY_MESSAGE_ROLES

    if item_type == "message":
        return role in _TYPED_MESSAGE_ROLES and "content" in item
    if item_type == "function_call":
        return all(key in item for key in ("call_id", "name", "arguments"))
    if item_type == "function_call_output":
        return all(key in item for key in ("call_id", "output"))
    if item_type == "file_search_call":
        return "id" in item
    if item_type == "reasoning":
        return True

    return False


def replayable_session_items(items: list[Any]) -> list[dict]:
    """Filter session history down to valid SDK input items.

    Guarantees pair-consistency: a ``function_call`` is only kept if its
    matching ``function_call_output`` (same ``call_id``) also passes the
    replayability filter, and vice versa. Orphaned halves trigger HTTP 400
    from providers.
    """
    candidates = [item for item in items if is_replayable_session_item(item)]
    call_ids_with_call = {
        item.get("call_id")
        for item in candidates
        if item.get("type") == "function_call" and item.get("call_id")
    }
    call_ids_with_output = {
        item.get("call_id")
        for item in candidates
        if item.get("type") == "function_call_output" and item.get("call_id")
    }
    paired_ids = call_ids_with_call & call_ids_with_output
    return [
        item
        for item in candidates
        if item.get("type") not in ("function_call", "function_call_output")
        or item.get("call_id") in paired_ids
    ]


def is_compactable_session_item(item: Any) -> bool:
    """Return whether a session item can be summarized during compaction."""
    if not isinstance(item, dict) or _is_post_compact_attachment(item):
        return False
    if is_replayable_session_item(item):
        return True

    role = item.get("role")
    if role in _EASY_MESSAGE_ROLES and "content" in item:
        return True
    if role == "tool" and ("content" in item or "tool_call_id" in item):
        return True
    return False


def compactable_session_items(items: list[Any]) -> list[dict]:
    """Filter session history down to valid source items for compaction."""
    return [item for item in items if is_compactable_session_item(item)]


def _is_post_compact_attachment(item: dict) -> bool:
    if item.get("role") != "system":
        return False
    content = item.get("content")
    return isinstance(content, str) and content.startswith(_POST_COMPACT_ATTACHMENT_PREFIX)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"text", "input_text", "output_text"}:
                    text_parts.append(str(block.get("text", "")))
                elif block_type in {"image_url", "input_image"}:
                    text_parts.append("[image]")
                elif block_type == "refusal":
                    text_parts.append(str(block.get("refusal", "")))
        return " ".join(part for part in text_parts if part).strip()
    return str(content) if content is not None else ""


def _plain_context_message_from_item(item: dict) -> dict | None:
    role = item.get("role")
    item_type = item.get("type")

    if item_type == "message":
        if role not in _TYPED_MESSAGE_ROLES:
            return None
        content = _content_to_text(item.get("content", ""))
    elif role in _EASY_MESSAGE_ROLES:
        content = _content_to_text(item.get("content", ""))
    else:
        return None

    if role == "system" and content.startswith(_POST_COMPACT_ATTACHMENT_PREFIX):
        return None
    if not content.strip():
        return None
    return {"role": role, "content": content}


def _recent_plain_context_items(
    messages: list[dict],
    max_messages: int | None,
) -> list[tuple[int, dict]]:
    plain_items = [
        (index, plain)
        for index, message in enumerate(messages)
        if (plain := _plain_context_message_from_item(message)) is not None
    ]
    if max_messages is None:
        return plain_items
    return plain_items[-max_messages:]


def _is_already_compacted_context(messages: list[dict], keep_recent: int) -> bool:
    if not messages:
        return False
    # A previously-compacted context may carry a trailing run of preserved
    # tool pairs. Recognize its conversational head independently of preserved
    # system/developer instructions; those do not consume keep_recent slots.
    tail = _trailing_replayable_tool_items(messages)
    head = messages[: len(messages) - len(tail)] if tail else messages
    conversation = [item for item in head if item.get("role") not in {"system", "developer"}]
    plain_items = _recent_plain_context_items(conversation, None)
    if not plain_items or len(plain_items) != len(conversation):
        return False
    first = plain_items[0][1]
    return first["content"].startswith(_COMPACTED_PREFIX) and len(plain_items) <= keep_recent + 1


def _already_compacted_kept_messages(messages: list[dict]) -> list[dict]:
    """Preserve original item shapes when the history needs no compaction.

    The textual projection is only for recognizing the context, not for
    replacing it: normalization here would erase images and typed-message
    fields while reporting a no-op.
    """
    return list(messages)


def _summary_message(summary: str) -> dict:
    return {"role": "user", "content": f"{_COMPACTED_PREFIX}\n\n{summary}"}


def build_compacted_session_items(result: CompactionResult) -> list[dict]:
    """Compose persisted history with preserved instructions before the summary."""
    instructions = [
        item for item in result.kept_messages if item.get("role") in {"system", "developer"}
    ]
    remainder = [
        item for item in result.kept_messages if item.get("role") not in {"system", "developer"}
    ]
    summary_items = [_summary_message(result.summary)] if result.summary else []
    return [*instructions, *summary_items, *remainder]


def _is_replayable_tool_item(item: Any) -> bool:
    """Whether an item is a replayable function_call / function_call_output."""
    if not is_replayable_session_item(item):
        return False
    return isinstance(item, dict) and item.get("type") in {
        "function_call",
        "function_call_output",
    }


def _trailing_replayable_tool_items(messages: list[dict]) -> list[dict]:
    """Return complete, ordered pairs from the trailing contiguous tool run.

    The kept tail must stay replayable, so any trailing ``function_call_output``
    is paired back to its originating ``function_call`` via ``call_id``. We walk
    backwards while items are replayable function_call / function_call_output
    items and stop at the first non-tool item. Outputs without an earlier call
    and calls without a retained output are excluded from the replayed tail;
    llm_compact_messages still includes those items in its summary source.
    """
    if not messages:
        return []

    start = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        if _is_replayable_tool_item(messages[index]):
            start = index
        else:
            break

    tail = messages[start:]
    if not tail:
        return []

    # Keep pairing intact: only emit a function_call_output when its matching
    # function_call is present earlier in the captured tail; an unpaired output
    # is not replayable on its own.
    seen_call_ids: set = set()
    trimmed: list[dict] = []
    for item in tail:
        item_type = item.get("type")
        call_id = item.get("call_id")
        if item_type == "function_call":
            if call_id is not None:
                seen_call_ids.add(call_id)
            trimmed.append(item)
        elif item_type == "function_call_output":
            if call_id is not None and call_id in seen_call_ids:
                trimmed.append(item)
            # else: leading orphan output whose call is outside the tail — skip.
    # The ordered pass above only guarantees output -> call consistency.
    # Apply the session's two-way contract as well: an unanswered call must be
    # summarized, not replayed and then silently dropped by next-turn repair.
    return replayable_session_items(trimmed)


def _item_role(message: dict) -> str:
    role = message.get("role")
    if isinstance(role, str) and role:
        return role

    item_type = message.get("type")
    if item_type in {"function_call", "file_search_call"}:
        return "assistant"
    if item_type == "function_call_output":
        return "tool"
    if item_type == "reasoning":
        return "assistant-reasoning"
    return str(item_type or "unknown")


def _build_summary(messages: list[dict]) -> str | None:
    if not messages:
        return None
    roles = list(OrderedDict.fromkeys(_item_role(message) for message in messages))
    roles_text = ", ".join(roles)
    return f"Compacted {len(messages)} earlier messages across roles: {roles_text}."


def _trim_to_token_budget(messages: list[dict], max_tokens: int) -> list[dict]:
    kept = list(messages)
    while kept and estimate_messages_tokens(kept) > max_tokens:
        kept.pop(0)
    return kept


def compact_messages(
    messages: list[dict],
    *,
    max_messages: int | None = None,
    max_tokens: int | None = None,
) -> CompactionResult:
    """Compact a transcript into a summary plus recent plain text messages."""
    original_count = len(messages)
    if max_messages is not None and _is_already_compacted_context(messages, max_messages):
        kept_messages = _already_compacted_kept_messages(messages)
        return CompactionResult(
            summary=None,
            kept_messages=kept_messages,
            token_count=estimate_messages_tokens(kept_messages),
            original_count=original_count,
        )

    kept_pairs = _recent_plain_context_items(messages, max_messages)
    kept_messages = [message for _, message in kept_pairs]
    kept_indices = {index for index, _ in kept_pairs}

    if max_tokens is not None:
        kept_messages = _trim_to_token_budget(kept_messages, max_tokens)
        kept_ids = {id(message) for message in kept_messages}
        kept_indices = {index for index, message in kept_pairs if id(message) in kept_ids}

    dropped_messages = [
        message for index, message in enumerate(messages) if index not in kept_indices
    ]
    summary = _build_summary(dropped_messages)
    token_count = estimate_messages_tokens(kept_messages)

    return CompactionResult(
        summary=summary,
        kept_messages=kept_messages,
        token_count=token_count,
        original_count=original_count,
    )


COMPACTION_SUMMARY_PROMPT = """You are summarizing a conversation between a user and an AI coding assistant to preserve critical context while reducing token usage.

First, write an <analysis> section where you think through the conversation, identify patterns, and determine what's most important. This section will be stripped from the final output.

Then, write a <summary> section with exactly these 9 numbered sections:

Use the analysis and summary tags only as section delimiters. Do not nest or repeat them inside a section; escape them when quoting examples.

1. **Primary Request and Intent**: What is the user's main goal or problem they're trying to solve?

2. **Key Technical Concepts**: What frameworks, libraries, APIs, or technical patterns are central to this conversation?

3. **Files and Code Sections**: What specific files, modules, functions, or code sections were discussed or modified? Include file paths.

4. **Errors and Fixes**: What errors occurred and how were they resolved? Include stack traces or error messages if critical.

5. **Problem Solving**: What approaches were tried? What worked and what didn't? What debugging steps were taken?

6. **All User Messages**: List every user message verbatim or as close as possible. This ensures no user intent is lost.

7. **Pending Tasks**: What tasks or action items remain unfinished or were mentioned for later?

8. **Current Work**: What is the current state? What was just completed or is in progress?

9. **Optional Next Step**: Based on the conversation flow, what is the likely next step or question the user might ask?

Keep each section concise but complete. If a section doesn't apply, write "N/A" for that section.

The conversation to summarize is below. Produce your <analysis> followed by <summary>."""


def _strip_images_from_message(msg: dict) -> dict:
    """Strip image content blocks from a message, replacing with [image] placeholder."""
    if not isinstance(msg.get("content"), list):
        return msg

    stripped_content = []
    for block in msg["content"]:
        if isinstance(block, dict):
            if block.get("type") in {"image_url", "input_image"}:
                stripped_content.append({"type": "text", "text": "[image]"})
            else:
                stripped_content.append(block)
        else:
            stripped_content.append(block)

    new_msg = msg.copy()
    new_msg["content"] = stripped_content
    return new_msg


def _format_message_for_summary(msg: dict) -> str:
    """Format a message for inclusion in the summary prompt."""
    item_type = msg.get("type")
    if item_type == "function_call":
        name = msg.get("name", "unknown")
        arguments = msg.get("arguments", "")
        call_id = msg.get("call_id", "")
        return f"assistant tool call (id={call_id}): {name}({arguments})"
    if item_type == "function_call_output":
        return f"tool result (id={msg.get('call_id', '')}): {msg.get('output', '')}"
    if item_type == "file_search_call":
        queries = msg.get("queries", [])
        status = msg.get("status", "")
        return f"assistant tool call (id={msg.get('id', '')}): file_search_call(queries={queries}, status={status})"
    if item_type == "reasoning":
        content = msg.get("content", [])
        text_parts = []
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text") or block.get("summary_text")
                    if text:
                        text_parts.append(str(text))
        return "assistant reasoning: " + (
            " ".join(text_parts) if text_parts else "[reasoning omitted]"
        )

    role = _item_role(msg)
    content = _content_to_text(msg.get("content", ""))

    # Include tool calls if present
    if msg.get("tool_calls"):
        tool_info = []
        for tc in msg["tool_calls"]:
            func = tc.get("function", {})
            tool_info.append(f"{func.get('name', 'unknown')}({func.get('arguments', '')})")
        content = f"{content}\n[Tool calls: {', '.join(tool_info)}]"

    # Include tool call ID if this is a tool result
    if msg.get("tool_call_id"):
        return f"tool result (id={msg['tool_call_id']}): {content}"

    return f"{role}: {content}"


def _extract_compaction_summary(response: str) -> str:
    """Accept a final summary, never analysis or ambiguous section markup."""
    # Remove analysis before looking for a summary: examples inside the
    # discarded analysis section are not the final answer.
    content = re.sub(r"<analysis>.*?</analysis>", "", response, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</?analysis\b", content, re.IGNORECASE):
        raise ValueError("Compaction summary contains incomplete analysis markup")

    match = re.search(r"<summary>(.*?)</summary>", content, re.DOTALL | re.IGNORECASE)
    if match:
        summary = match.group(1).strip()
        remainder = content[: match.start()] + content[match.end() :]
    else:
        # Untagged final summaries remain supported for compatible models.
        summary = content.strip()
        remainder = ""
    if any(re.search(r"</?summary\b", part, re.IGNORECASE) for part in (summary, remainder)):
        raise ValueError("Compaction summary markup is incomplete or ambiguous")
    if not summary:
        raise ValueError("Compaction summary is empty; original history must be preserved")
    return summary


async def llm_compact_messages(
    messages: list[dict],
    *,
    keep_recent: int = 2,
) -> CompactionResult:
    """
    Compact messages using LLM-based summarization.

    Generates a structured 9-section summary while preserving recent plain text
    messages. Most tool calls and tool outputs are summarized rather than
    replayed, but the trailing contiguous run of replayable function_call /
    function_call_output items is ALSO preserved verbatim (appended after the
    summary and plain-text tail) so the kept context stays replayable and the
    most recent tool interaction is not reduced to prose. Tool_call/tool_result
    pairing is preserved via call_id.

    Args:
        messages: List of conversation messages to compact
        keep_recent: Number of recent messages to keep (default 2)

    Returns:
        CompactionResult with LLM-generated summary and kept messages
    """
    original_count = len(messages)

    if _is_already_compacted_context(messages, keep_recent):
        return CompactionResult(
            summary=None,
            kept_messages=_already_compacted_kept_messages(messages),
            token_count=estimate_messages_tokens(messages),
            original_count=original_count,
        )

    # System/developer items are authoritative instructions and are never
    # rewritten into prose. ``keep_recent`` applies only to conversational
    # messages so pinned instructions cannot crowd out the latest user intent.
    instruction_indices = {
        index
        for index, message in enumerate(messages)
        if message.get("role") in {"system", "developer"}
    }
    recent_conversation_pairs = [
        (index, message)
        for index, message in _recent_plain_context_items(messages, None)
        if message.get("role") not in {"system", "developer"}
    ][-keep_recent:]
    kept_indices = instruction_indices | {index for index, _message in recent_conversation_pairs}
    to_keep = [message for index, message in enumerate(messages) if index in kept_indices]

    # Preserve the trailing tool_call/result pair verbatim so it stays
    # replayable. These items are appended after the plain-text tail and are
    # excluded from summarization (deduped by identity) to avoid representing
    # them both verbatim and in prose.
    tail_tool_items = _trailing_replayable_tool_items(messages)
    tail_ids = {id(item) for item in tail_tool_items}

    def _keep_tail(base: list[dict]) -> list[dict]:
        # Append tail items not already present (deduped by identity), keeping
        # tool_call/tool_result pairing intact.
        existing = {id(item) for item in base}
        return base + [item for item in tail_tool_items if id(item) not in existing]

    to_summarize = [
        message
        for index, message in enumerate(messages)
        if index not in kept_indices and id(message) not in tail_ids
    ]

    if not to_summarize:
        kept_messages = _keep_tail(to_keep)
        return CompactionResult(
            summary=None,
            kept_messages=kept_messages,
            token_count=estimate_messages_tokens(kept_messages),
            original_count=original_count,
        )

    try:
        # Strip images from messages being summarized
        stripped_to_summarize = [_strip_images_from_message(msg) for msg in to_summarize]

        # Build conversation text
        conversation_text = "\n\n".join(
            _format_message_for_summary(msg) for msg in stripped_to_summarize
        )

        # Call LLM
        summary_messages = [
            {
                "role": "user",
                "content": f"{COMPACTION_SUMMARY_PROMPT}\n\n{conversation_text}",
            }
        ]

        response = await llm_completion(
            summary_messages,
            use_small=True,
            response_reserve=4_096,
        )

        # An API response is not itself proof of a usable final summary.
        # Reject invalid output before authorizing a replacement of the source.
        summary = _extract_compaction_summary(response)

        kept_messages = _keep_tail(to_keep)
        return CompactionResult(
            summary=summary,
            kept_messages=kept_messages,
            token_count=estimate_messages_tokens([_summary_message(summary), *kept_messages]),
            original_count=original_count,
        )

    except Exception:
        # Destructive compaction must never claim coverage when the complete
        # source was not accepted by the summarizer. Propagate the failure so
        # scheduler/manual callers leave the original session untouched.
        raise
