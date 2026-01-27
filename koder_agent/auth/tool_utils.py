"""Tool calling utilities for OAuth providers.

Shared utilities for converting, cleaning, and managing tool calls across
different OAuth providers (Google, Claude, ChatGPT, Antigravity).
"""

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def merge_optional_params(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge LiteLLM optional_params into kwargs without mutating the result."""
    optional_params = kwargs.pop("optional_params", {})
    return {**kwargs, **optional_params}


# =============================================================================
# THOUGHT SIGNATURE CACHE (Gemini 3)
# Stores thoughtSignature values to survive LiteLLM's streaming handler
# Thread-safe via threading.Lock for concurrent access
# =============================================================================

# Cache: {(call_id, function_name): (thought_signature, timestamp)}
_thought_signature_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
# Most recent thought_signature for parallel calls fallback (per function name)
_last_thought_signature_by_name: Dict[str, Tuple[str, float]] = {}
# Global last signature for cross-function parallel calls (legacy fallback)
_last_thought_signature: Optional[Tuple[str, float]] = None
# Session-based signature cache: session_id -> (signature, message_count, timestamp, model_family?)
_session_signature_cache: Dict[str, Tuple[str, int, float, Optional[str]]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes TTL
_SESSION_CACHE_TTL_SECONDS = 2 * 60 * 60  # 2 hours TTL
_BATCH_WINDOW_SECONDS = 2  # Window for parallel call batch (used for fallback)
# Lock for thread-safe cache access (required for multi-step operations)
_cache_lock = threading.Lock()
_MAX_CACHE_SIZE = 1000  # Prevent unbounded growth


def _normalize_function_name(name: str) -> str:
    """Normalize function name by stripping known prefixes.

    Handles prefixes like: default_api:, tools:, functions:, etc.
    """
    known_prefixes = ("default_api:", "tools:", "functions:", "tool:")
    for prefix in known_prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def cache_thought_signature(call_id: str, function_name: str, thought_signature: str) -> None:
    """Cache a thoughtSignature for later retrieval.

    Also stores as "last signature" for parallel calls that don't have their own.
    Thread-safe via _cache_lock.

    Args:
        call_id: Tool call ID
        function_name: Function name
        thought_signature: The thoughtSignature to cache
    """
    global _last_thought_signature

    key = (call_id, function_name)
    now = time.time()
    normalized_name = _normalize_function_name(function_name)

    with _cache_lock:
        # Store in main cache
        _thought_signature_cache[key] = (thought_signature, now)

        # Also store normalized version for prefix-agnostic lookup
        if normalized_name != function_name:
            _thought_signature_cache[(call_id, normalized_name)] = (thought_signature, now)

        # Store per-function-name fallback for parallel calls
        _last_thought_signature_by_name[normalized_name] = (thought_signature, now)

        # Clean up old entries and enforce size limit
        _cleanup_thought_signature_cache_unlocked()


def cache_thinking_signature(thought_signature: str) -> None:
    """Cache a thoughtSignature from a thinking block.

    Gemini 3 may emit thoughtSignature only on thinking parts (not functionCall).
    Store it as the global fallback so tool calls without explicit signatures can reuse it.
    """
    global _last_thought_signature

    if not thought_signature or not isinstance(thought_signature, str):
        return

    now = time.time()
    with _cache_lock:
        _last_thought_signature = (thought_signature, now)
        _cleanup_thought_signature_cache_unlocked()


def _normalize_model_family(model: Optional[str]) -> Optional[str]:
    if not model or not isinstance(model, str):
        return None
    lower = model.lower()
    if "claude-opus" in lower:
        return "claude-opus"
    if "claude-sonnet" in lower:
        return "claude-sonnet"
    if "claude-haiku" in lower:
        return "claude-haiku"
    if "gemini-3" in lower:
        return "gemini-3"
    if "gemini-2" in lower:
        return "gemini-2"
    return lower.split("-thinking")[0]


def cache_session_signature(
    session_id: str,
    thought_signature: str,
    message_count: int,
    model: Optional[str] = None,
) -> None:
    """Cache the latest thinking signature for a session (Gemini 3 tool loops).

    Prefer most recent signatures, detect rewind (message_count decreases),
    and only store sufficiently long signatures.
    """
    if not session_id or not thought_signature or not isinstance(thought_signature, str):
        return
    if len(thought_signature) < MIN_SIGNATURE_LENGTH:
        return

    now = time.time()
    model_family = _normalize_model_family(model)
    with _cache_lock:
        existing = _session_signature_cache.get(session_id)
        should_store = False
        if existing is None:
            should_store = True
        else:
            if len(existing) == 4:
                _, existing_count, ts, existing_family = existing
            else:
                _, existing_count, ts = existing  # type: ignore[misc]
                existing_family = None
            # Expired entry
            if now - ts >= _SESSION_CACHE_TTL_SECONDS:
                should_store = True
            # Rewind detected
            elif message_count < existing_count:
                should_store = True
            # Same message count: prefer longer signature
            elif message_count == existing_count:
                should_store = len(thought_signature) > len(existing[0])
            else:
                should_store = True

        if should_store:
            _session_signature_cache[session_id] = (
                thought_signature,
                message_count,
                now,
                model_family if model_family else (existing_family if existing else None),
            )

        _cleanup_thought_signature_cache_unlocked()


def get_session_signature(session_id: str, model: Optional[str] = None) -> Optional[str]:
    """Retrieve cached session signature if valid and not expired."""
    if not session_id:
        return None

    now = time.time()
    with _cache_lock:
        entry = _session_signature_cache.get(session_id)
        if not entry:
            return None
        if len(entry) == 4:
            sig, _count, ts, family = entry
        else:
            sig, _count, ts = entry  # type: ignore[misc]
            family = None
        if now - ts >= _SESSION_CACHE_TTL_SECONDS:
            return None
        if model and family:
            target_family = _normalize_model_family(model)
            if target_family and family != target_family:
                return None
        return sig


def _extract_part_thought_signature(part: Dict[str, Any]) -> Optional[str]:
    """Extract thoughtSignature from a Gemini part (direct or metadata.google)."""
    sig = part.get("thoughtSignature")
    if isinstance(sig, str) and sig:
        return sig

    func_call = part.get("functionCall")
    if isinstance(func_call, dict):
        sig = func_call.get("thoughtSignature") or func_call.get("thought_signature")
        if isinstance(sig, str) and sig:
            return sig

        func_meta = func_call.get("metadata")
        if isinstance(func_meta, dict):
            google_meta = func_meta.get("google")
            if isinstance(google_meta, dict):
                sig = google_meta.get("thoughtSignature")
                if isinstance(sig, str) and sig:
                    return sig

    metadata = part.get("metadata")
    if isinstance(metadata, dict):
        google_meta = metadata.get("google")
        if isinstance(google_meta, dict):
            sig = google_meta.get("thoughtSignature")
            if isinstance(sig, str) and sig:
                return sig

    return None


def inject_thought_signature_for_function_calls(
    contents: List[Dict[str, Any]], session_id: str, model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Inject thoughtSignature into functionCall parts when missing.

    Uses session-scoped cached signatures (no fake signatures).
    """
    if not contents or not session_id:
        return contents

    sig = get_session_signature(session_id, model)
    if not sig:
        return contents

    updated_contents = []
    for content in contents:
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            updated_contents.append(content)
            continue

        parts = []
        for part in content["parts"]:
            if not isinstance(part, dict):
                parts.append(part)
                continue

            if "functionCall" in part and not _extract_part_thought_signature(part):
                new_part = dict(part)
                new_part["thoughtSignature"] = sig
                parts.append(new_part)
            else:
                parts.append(part)

        updated_contents.append({**content, "parts": parts})

    return updated_contents


def get_cached_thought_signature(call_id: str, function_name: str) -> Optional[str]:
    """Retrieve a cached thoughtSignature.

    Handles various function name formats via normalization:
    - Direct name: "grep_search"
    - With API prefix: "default_api:grep_search", "tools:grep_search", etc.

    For parallel function calls, Gemini only provides thoughtSignature for the
    first call. This function falls back to the most recent signature if no
    exact match is found (within the batch window).

    Thread-safe via _cache_lock.

    Args:
        call_id: Tool call ID
        function_name: Function name (may include prefix)

    Returns:
        The cached thoughtSignature, or None if not found or expired
    """
    now = time.time()
    normalized_name = _normalize_function_name(function_name)

    with _cache_lock:
        # Strategy 1: Try exact match first
        key = (call_id, function_name)
        if key in _thought_signature_cache:
            sig, timestamp = _thought_signature_cache[key]
            if now - timestamp < _CACHE_TTL_SECONDS:
                return sig

        # Strategy 2: Try normalized name (handles all prefixes)
        if normalized_name != function_name:
            key = (call_id, normalized_name)
            if key in _thought_signature_cache:
                sig, timestamp = _thought_signature_cache[key]
                if now - timestamp < _CACHE_TTL_SECONDS:
                    return sig

        # NOTE: Do not use per-function/global fallbacks for Gemini 3 tool calls.
        # Thought signatures must match the specific functionCall in the response.

    # No valid signature found - log warning for debugging (only for Gemini 3 models)
    # This is outside the lock since logging doesn't need synchronization
    logger.debug(
        f"No thoughtSignature found for call_id={call_id}, function={function_name}. "
        "This is expected for non-Gemini-3 models."
    )
    return None


def _cleanup_thought_signature_cache_unlocked() -> None:
    """Remove expired entries from the cache and enforce size limits.

    MUST be called while holding _cache_lock.
    Also cleans up the per-function-name fallback cache.
    """
    global _last_thought_signature

    now = time.time()

    # Clean main cache - remove expired entries
    expired_keys = [
        key for key, (_, ts) in _thought_signature_cache.items() if now - ts >= _CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        del _thought_signature_cache[key]

    # Clean per-function-name fallback cache
    expired_names = [
        name
        for name, (_, ts) in _last_thought_signature_by_name.items()
        if now - ts >= _CACHE_TTL_SECONDS
    ]
    for name in expired_names:
        del _last_thought_signature_by_name[name]

    # Clear global fallback if expired
    if _last_thought_signature:
        _, timestamp = _last_thought_signature
        if now - timestamp >= _CACHE_TTL_SECONDS:
            _last_thought_signature = None

    # Enforce size limit on main cache (LRU-style: remove oldest)
    if len(_thought_signature_cache) > _MAX_CACHE_SIZE:
        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(_thought_signature_cache.items(), key=lambda x: x[1][1])
        entries_to_remove = len(_thought_signature_cache) - _MAX_CACHE_SIZE
        for key, _ in sorted_entries[:entries_to_remove]:
            del _thought_signature_cache[key]

    # Clean session cache - remove expired entries
    expired_sessions = [
        key
        for key, entry in _session_signature_cache.items()
        if now - (entry[2] if len(entry) >= 3 else 0) >= _SESSION_CACHE_TTL_SECONDS
    ]
    for key in expired_sessions:
        del _session_signature_cache[key]


def clear_thought_signature_cache() -> None:
    """Clear all cached thought signatures.

    Useful for testing and when switching between providers.
    Thread-safe.
    """
    global _last_thought_signature

    with _cache_lock:
        _thought_signature_cache.clear()
        _last_thought_signature_by_name.clear()
        _last_thought_signature = None
        _session_signature_cache.clear()


# Placeholder for tools with empty schemas (Claude VALIDATED mode requires at least one property)
EMPTY_SCHEMA_PLACEHOLDER_NAME = "_placeholder"
EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION = "Placeholder. Always pass true."

# =============================================================================
# CLAUDE TOOL HARDENING CONSTANTS
# =============================================================================

# Sentinel value to bypass thought signature validation
# When a thinking block has an invalid/missing signature, use this to skip validation
SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

# System instruction for Claude tool usage hardening
# Prevents hallucinated parameters by explicitly stating the rules
CLAUDE_TOOL_SYSTEM_INSTRUCTION = """CRITICAL TOOL USAGE INSTRUCTIONS:
You are operating in a custom environment where tool definitions differ from your training data.
You MUST follow these rules strictly:

1. DO NOT use your internal training data to guess tool parameters
2. ONLY use the exact parameter structure defined in the tool schema
3. Parameter names in schemas are EXACT - do not substitute with similar names from your training
4. Array parameters have specific item types - check the schema's 'items' field for the exact structure
5. When you see "STRICT PARAMETERS" in a tool description, those type definitions override any assumptions
6. Tool use in agentic workflows is REQUIRED - you must call tools with the exact parameters specified

If you are unsure about a tool's parameters, YOU MUST read the schema definition carefully."""

# Template for parameter signature injection into tool descriptions
CLAUDE_DESCRIPTION_PROMPT = "\n\n⚠️ STRICT PARAMETERS: {params}."

# Minimum signature length to be considered valid (from TypeScript MIN_SIGNATURE_LENGTH)
MIN_SIGNATURE_LENGTH = 50

# Unsupported constraint keywords that should be moved to description hints
# Claude/Gemini reject these in VALIDATED mode
UNSUPPORTED_CONSTRAINTS = [
    "minLength",
    "maxLength",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "pattern",
    "minItems",
    "maxItems",
    "format",
    "default",
    "examples",
]

# Keywords that should be removed after hint extraction
UNSUPPORTED_KEYWORDS = [
    *UNSUPPORTED_CONSTRAINTS,
    "$schema",
    "$defs",
    "definitions",
    "const",
    "$ref",
    "additionalProperties",
    "propertyNames",
    "title",
    "$id",
    "$comment",
]


# =============================================================================
# JSON SCHEMA CLEANING FOR LLM TOOL CALLS
# Cleans JSON schemas for providers that don't support advanced features
# like $ref/$defs (GitHub Copilot, Antigravity, etc.)
# =============================================================================


def _append_description_hint(schema: Dict[str, Any], hint: str) -> Dict[str, Any]:
    """Append a hint to a schema's description field."""
    if not schema or not isinstance(schema, dict):
        return schema

    existing = schema.get("description", "")
    if isinstance(existing, str) and existing:
        new_description = f"{existing} ({hint})"
    else:
        new_description = hint

    return {**schema, "description": new_description}


def _convert_refs_to_hints(schema: Any) -> Any:
    """Phase 1a: Convert $ref to description hints.

    $ref: "#/$defs/Foo" → { type: "object", description: "See: Foo" }
    """
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_convert_refs_to_hints(item) for item in schema]

    # If this object has $ref, replace it with a hint
    if isinstance(schema.get("$ref"), str):
        ref_val = schema["$ref"]
        def_name = ref_val.split("/")[-1] if "/" in ref_val else ref_val
        hint = f"See: {def_name}"
        existing_desc = schema.get("description", "")
        new_description = f"{existing_desc} ({hint})" if existing_desc else hint
        return {"type": "object", "description": new_description}

    # Recursively process all properties
    result = {}
    for key, value in schema.items():
        result[key] = _convert_refs_to_hints(value)
    return result


def _convert_const_to_enum(schema: Any) -> Any:
    """Phase 1b: Convert const to enum.

    { const: "foo" } → { enum: ["foo"] }
    """
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_convert_const_to_enum(item) for item in schema]

    result = {}
    for key, value in schema.items():
        if key == "const" and "enum" not in schema:
            result["enum"] = [value]
        else:
            result[key] = _convert_const_to_enum(value)
    return result


def _add_enum_hints(schema: Any) -> Any:
    """Phase 1c: Add enum hints to description.

    { enum: ["a", "b", "c"] } → adds "(Allowed: a, b, c)" to description
    """
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_add_enum_hints(item) for item in schema]

    result = dict(schema)

    # Add enum hint if enum has 2-10 items
    if isinstance(result.get("enum"), list) and 1 < len(result["enum"]) <= 10:
        vals = ", ".join(str(v) for v in result["enum"])
        result = _append_description_hint(result, f"Allowed: {vals}")

    # Recursively process nested objects
    for key, value in list(result.items()):
        if key != "enum" and isinstance(value, dict):
            result[key] = _add_enum_hints(value)
        elif key != "enum" and isinstance(value, list):
            result[key] = [_add_enum_hints(item) for item in value]

    return result


def _add_additional_properties_hints(schema: Any) -> Any:
    """Phase 1d: Add additionalProperties hints.

    { additionalProperties: false } → adds "(No extra properties allowed)" to description
    """
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_add_additional_properties_hints(item) for item in schema]

    result = dict(schema)

    if result.get("additionalProperties") is False:
        result = _append_description_hint(result, "No extra properties allowed")

    # Recursively process nested objects
    for key, value in list(result.items()):
        if key != "additionalProperties" and isinstance(value, dict):
            result[key] = _add_additional_properties_hints(value)
        elif key != "additionalProperties" and isinstance(value, list):
            result[key] = [_add_additional_properties_hints(item) for item in value]

    return result


def _move_constraints_to_description(schema: Any) -> Any:
    """Phase 1e: Move unsupported constraints to description hints.

    { minLength: 1, maxLength: 100 } → adds "(minLength: 1) (maxLength: 100)" to description
    """
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_move_constraints_to_description(item) for item in schema]

    result = dict(schema)

    # Move constraint values to description
    for constraint in UNSUPPORTED_CONSTRAINTS:
        if constraint in result and not isinstance(result[constraint], dict):
            result = _append_description_hint(result, f"{constraint}: {result[constraint]}")

    # Recursively process nested objects
    for key, value in list(result.items()):
        if isinstance(value, dict):
            result[key] = _move_constraints_to_description(value)
        elif isinstance(value, list):
            result[key] = [_move_constraints_to_description(item) for item in value]

    return result


def _merge_all_of(schema: Any) -> Any:
    """Phase 2a: Merge allOf schemas into a single object."""
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_merge_all_of(item) for item in schema]

    result = dict(schema)

    if isinstance(result.get("allOf"), list):
        all_of = result.pop("allOf")
        merged_properties = {}
        merged_required = []

        for sub_schema in all_of:
            if isinstance(sub_schema, dict):
                if isinstance(sub_schema.get("properties"), dict):
                    merged_properties.update(sub_schema["properties"])
                if isinstance(sub_schema.get("required"), list):
                    merged_required.extend(sub_schema["required"])

        if merged_properties:
            existing_props = result.get("properties", {})
            result["properties"] = {**existing_props, **merged_properties}

        if merged_required:
            existing_req = result.get("required", [])
            result["required"] = list(set(existing_req + merged_required))

    # Recursively process nested objects
    for key, value in list(result.items()):
        if isinstance(value, dict):
            result[key] = _merge_all_of(value)
        elif isinstance(value, list):
            result[key] = [_merge_all_of(item) for item in value]

    return result


def _remove_unsupported_keywords(schema: Any) -> Any:
    """Phase 3: Remove unsupported keywords from schema."""
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_remove_unsupported_keywords(item) for item in schema]

    result = {}
    for key, value in schema.items():
        if key in UNSUPPORTED_KEYWORDS:
            continue
        if isinstance(value, dict):
            result[key] = _remove_unsupported_keywords(value)
        elif isinstance(value, list):
            result[key] = [_remove_unsupported_keywords(item) for item in value]
        else:
            result[key] = value

    return result


def _fix_required_fields(schema: Any) -> Any:
    """Fix required fields that reference non-existent properties.

    Gemini API validates that all entries in 'required' array have
    corresponding properties defined. This removes invalid entries.
    """
    if not schema or not isinstance(schema, dict):
        return schema

    result = dict(schema)

    # Fix required at this level
    if isinstance(result.get("required"), list) and isinstance(result.get("properties"), dict):
        valid_props = set(result["properties"].keys())
        result["required"] = [r for r in result["required"] if r in valid_props]
        # Remove empty required array
        if not result["required"]:
            del result["required"]

    # Recursively fix nested schemas
    for key, value in list(result.items()):
        if isinstance(value, dict):
            result[key] = _fix_required_fields(value)
        elif isinstance(value, list):
            result[key] = [
                _fix_required_fields(item) if isinstance(item, dict) else item for item in value
            ]

    return result


def clean_json_schema(schema: Any) -> Dict[str, Any]:
    """Clean JSON schema for LLM providers that don't support advanced features.

    Some providers (GitHub Copilot, Antigravity, etc.) don't support $ref/$defs
    or other advanced JSON Schema features. This function cleans the schema
    by converting unsupported features to description hints.

    Applies the following transformations:
    1a. Convert $ref to description hints
    1b. Convert const to enum
    1c. Add enum hints to descriptions
    1d. Add additionalProperties hints
    1e. Move unsupported constraints to description
    2a. Merge allOf schemas
    3. Remove unsupported keywords ($ref, $defs, additionalProperties, etc.)
    4. Fix required fields referencing non-existent properties

    Args:
        schema: JSON schema to clean

    Returns:
        Cleaned schema compatible with all LLM providers
    """
    if not schema or not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    # Phase 1: Convert and hint
    result = _convert_refs_to_hints(schema)
    result = _convert_const_to_enum(result)
    result = _add_enum_hints(result)
    result = _add_additional_properties_hints(result)
    result = _move_constraints_to_description(result)

    # Phase 2: Merge
    result = _merge_all_of(result)

    # Phase 3: Remove unsupported keywords
    result = _remove_unsupported_keywords(result)

    # Phase 4: Fix required fields referencing non-existent properties
    result = _fix_required_fields(result)

    # Ensure type is object
    if "type" not in result:
        result["type"] = "object"

    return result


# =============================================================================
# CLAUDE SCHEMA CLEANING
# =============================================================================

# Whitelist of allowed fields for Claude VALIDATED mode
# Any field not in this list will be removed
CLAUDE_ALLOWED_SCHEMA_FIELDS = {
    "type",
    "description",
    "properties",
    "required",
    "items",
    "enum",
    "title",
}

# Constraints to move to description hints
CLAUDE_CONSTRAINT_FIELDS = [
    ("minLength", "minLen"),
    ("maxLength", "maxLen"),
    ("pattern", "pattern"),
    ("minimum", "min"),
    ("maximum", "max"),
    ("multipleOf", "multipleOf"),
    ("exclusiveMinimum", "exclMin"),
    ("exclusiveMaximum", "exclMax"),
    ("minItems", "minItems"),
    ("maxItems", "maxItems"),
    ("propertyNames", "propertyNames"),
    ("format", "format"),
]


def _score_schema_option(schema: Any) -> int:
    """Score a schema option for anyOf/oneOf selection.

    Higher scores are preferred:
    - Object with properties: 3
    - Array: 2
    - Scalar type (non-null): 1
    - Null or invalid: 0
    """
    if not schema or not isinstance(schema, dict):
        return 0

    # Object with properties scores highest
    if schema.get("properties") or schema.get("type") == "object":
        return 3

    # Array scores medium
    if schema.get("items") or schema.get("type") == "array":
        return 2

    # Non-null scalar types score low
    type_val = schema.get("type")
    if type_val and type_val != "null":
        return 1

    return 0


def _extract_best_schema_from_union(union_array: List[Any]) -> Optional[Dict[str, Any]]:
    """Extract the best schema option from anyOf/oneOf array.

    Selects the schema with the highest complexity score.
    """
    best_option = None
    best_score = -1

    for item in union_array:
        score = _score_schema_option(item)
        if score > best_score:
            best_score = score
            best_option = item

    return dict(best_option) if isinstance(best_option, dict) else None


def _flatten_anyof_oneof_for_claude(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten anyOf/oneOf by selecting the best option and merging properties."""
    result = dict(schema)

    # Check for anyOf or oneOf
    for union_key in ("anyOf", "oneOf"):
        union_array = result.get(union_key)
        if not isinstance(union_array, list) or not union_array:
            continue

        # Only process if type is not already set or is "object"
        current_type = result.get("type")
        if current_type and current_type not in ("object", "null"):
            # Type already set, just remove anyOf/oneOf
            result.pop(union_key, None)
            continue

        # Find best option
        best_branch = _extract_best_schema_from_union(union_array)
        if not best_branch:
            result.pop(union_key, None)
            continue

        # Merge properties from best branch
        if best_branch.get("properties"):
            target_props = result.setdefault("properties", {})
            for prop_key, prop_val in best_branch.get("properties", {}).items():
                if prop_key not in target_props:
                    target_props[prop_key] = prop_val

        # Merge required
        if best_branch.get("required"):
            target_req = result.setdefault("required", [])
            for req_item in best_branch.get("required", []):
                if req_item not in target_req:
                    target_req.append(req_item)

        # Copy type if not set
        if not result.get("type") and best_branch.get("type"):
            result["type"] = best_branch["type"]

        # Remove the union key
        result.pop(union_key, None)

    return result


def _flatten_type_array_for_claude(schema: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Flatten type array to single type with nullable hint.

    E.g., ["string", "null"] -> "string" (returns is_nullable=True)

    Returns:
        Tuple of (modified schema, is_nullable)
    """
    result = dict(schema)
    is_nullable = False

    type_val = result.get("type")
    if isinstance(type_val, list):
        has_null = "null" in type_val
        non_null_types = [t for t in type_val if t != "null" and t]

        # Select first non-null type, or "string" as fallback
        first_type = non_null_types[0] if non_null_types else "string"
        result["type"] = first_type.lower()  # Keep lowercase for Claude

        # Track nullable for description hint
        if has_null:
            is_nullable = True
    elif isinstance(type_val, str):
        # Ensure lowercase
        result["type"] = type_val.lower()
        if type_val.lower() == "null":
            is_nullable = True
            result["type"] = "string"  # Fallback

    return result, is_nullable


def _add_constraint_hints_for_claude(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Move constraint fields to description hints."""
    result = dict(schema)
    hints = []

    for field, label in CLAUDE_CONSTRAINT_FIELDS:
        val = result.get(field)
        if val is not None:
            val_str = str(val) if not isinstance(val, str) else val
            hints.append(f"{label}: {val_str}")

    if hints:
        suffix = f" [Constraint: {', '.join(hints)}]"
        desc = result.get("description", "")
        if suffix not in desc:
            result["description"] = desc + suffix if desc else suffix.strip()

    return result


def _apply_whitelist_for_claude(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Apply whitelist filtering - remove all fields not in CLAUDE_ALLOWED_SCHEMA_FIELDS."""
    return {key: value for key, value in schema.items() if key in CLAUDE_ALLOWED_SCHEMA_FIELDS}


def _add_empty_object_placeholder_for_claude(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add placeholder property for empty object schemas.

    Claude VALIDATED mode requires at least one property in object schemas.
    """
    result = dict(schema)

    if result.get("type") == "object":
        props = result.get("properties")
        has_props = props and isinstance(props, dict) and len(props) > 0

        if not has_props:
            result["properties"] = {
                "reason": {
                    "type": "string",
                    "description": "Reason for calling this tool",
                }
            }
            result["required"] = ["reason"]

    return result


def _fix_required_for_claude(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Remove required entries that don't exist in properties."""
    result = dict(schema)

    if isinstance(result.get("required"), list) and isinstance(result.get("properties"), dict):
        valid_props = set(result["properties"].keys())
        result["required"] = [r for r in result["required"] if r in valid_props]
        if not result["required"]:
            del result["required"]

    return result


def clean_json_schema_for_claude(schema: Any) -> Dict[str, Any]:
    """Clean JSON schema for Claude VALIDATED mode via Antigravity.

    This is a comprehensive cleaning function that ensures schemas are compatible
    with Claude's strict JSON Schema draft 2020-12 validation.

    Key transformations:
    1. Flatten $ref references
    2. Merge allOf schemas
    3. Flatten anyOf/oneOf to best option
    4. Flatten type arrays (["string", "null"] -> "string")
    5. Move constraints to description hints
    6. Apply whitelist filtering (only allowed fields kept)
    7. Add placeholder for empty object schemas
    8. Fix required fields

    Args:
        schema: JSON schema to clean

    Returns:
        Cleaned schema compatible with Claude VALIDATED mode
    """
    if not schema or not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Reason for calling this tool",
                }
            },
            "required": ["reason"],
        }

    result = dict(schema)

    # Phase 1: Resolve $ref (reuse existing function)
    result = _resolve_refs(result)

    # Phase 2: Merge allOf
    result = _merge_all_of(result)

    # Phase 3: Flatten anyOf/oneOf
    result = _flatten_anyof_oneof_for_claude(result)

    # Phase 4: Flatten type arrays and track nullable
    result, is_nullable = _flatten_type_array_for_claude(result)

    # Phase 5: Add constraint hints
    result = _add_constraint_hints_for_claude(result)

    # Phase 6: Apply whitelist
    result = _apply_whitelist_for_claude(result)

    # Phase 7: Add placeholder for empty objects
    result = _add_empty_object_placeholder_for_claude(result)

    # Phase 8: Fix required fields
    result = _fix_required_for_claude(result)

    # Add nullable hint if needed
    if is_nullable:
        desc = result.get("description", "")
        if "(nullable)" not in desc:
            result["description"] = f"{desc} (nullable)" if desc else "(nullable)"

    # Ensure type is set
    if "type" not in result:
        result["type"] = "object"

    # Recursively clean nested schemas
    if isinstance(result.get("properties"), dict):
        cleaned_props = {}
        nullable_keys = set()
        for prop_key, prop_val in result["properties"].items():
            if isinstance(prop_val, dict):
                cleaned = clean_json_schema_for_claude(prop_val)
                cleaned_props[prop_key] = cleaned
                # Track nullable properties
                if "(nullable)" in cleaned.get("description", ""):
                    nullable_keys.add(prop_key)
            else:
                cleaned_props[prop_key] = prop_val
        result["properties"] = cleaned_props

        # Remove nullable fields from required
        if nullable_keys and isinstance(result.get("required"), list):
            result["required"] = [r for r in result["required"] if r not in nullable_keys]
            if not result["required"]:
                del result["required"]

    if isinstance(result.get("items"), dict):
        result["items"] = clean_json_schema_for_claude(result["items"])

    return result


# =============================================================================
# GEMINI SCHEMA TRANSFORMATION
# =============================================================================

# Fields that Gemini API rejects - must be removed from schemas
GEMINI_UNSUPPORTED_SCHEMA_FIELDS = {
    "additionalProperties",
    "$schema",
    "$id",
    "$comment",
    "$ref",
    "$defs",
    "definitions",
    "const",
    "contentMediaType",
    "contentEncoding",
    "if",
    "then",
    "else",
    "not",
    "patternProperties",
    "unevaluatedProperties",
    "unevaluatedItems",
    "dependentRequired",
    "dependentSchemas",
    "propertyNames",
    "minContains",
    "maxContains",
}


def _resolve_refs(schema: Any, defs: Optional[Dict[str, Any]] = None) -> Any:
    """Resolve $ref references in JSON Schema.

    Args:
        schema: JSON Schema that may contain $ref
        defs: Dictionary of definitions (from $defs or definitions)

    Returns:
        Schema with $ref replaced by actual definitions
    """
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [_resolve_refs(item, defs) for item in schema]

    # Extract definitions from schema if present
    local_defs = schema.get("$defs") or schema.get("definitions") or {}
    if local_defs:
        defs = {**defs, **local_defs} if defs else local_defs

    # Check for $ref
    ref = schema.get("$ref")
    if ref and isinstance(ref, str) and defs:
        # Parse ref like "#/$defs/TodoItem" or "#/definitions/TodoItem"
        if ref.startswith("#/$defs/"):
            ref_name = ref[8:]  # Remove "#/$defs/"
            if ref_name in defs:
                # Return resolved definition (recursively resolve any nested refs)
                return _resolve_refs(defs[ref_name], defs)
        elif ref.startswith("#/definitions/"):
            ref_name = ref[14:]  # Remove "#/definitions/"
            if ref_name in defs:
                return _resolve_refs(defs[ref_name], defs)

    # Recursively resolve refs in nested structures
    result: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("$defs", "definitions"):
            # Don't include definitions in output
            continue
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                prop_name: _resolve_refs(prop_schema, defs)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items":
            result[key] = _resolve_refs(value, defs)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            result[key] = [_resolve_refs(item, defs) for item in value]
        else:
            result[key] = value

    return result


def _convert_anyof_to_nullable(anyof_items: List[Any]) -> Optional[Dict[str, Any]]:
    """Convert anyOf pattern with null to Gemini's nullable format.

    Common JSON Schema patterns for nullable fields:
    - anyOf: [{type: 'integer'}, {type: 'null'}]

    Gemini expects:
    - {type: 'INTEGER', nullable: true}

    Args:
        anyof_items: List of anyOf schema items

    Returns:
        Gemini-compatible schema with nullable, or None if not a nullable pattern
    """
    if len(anyof_items) != 2:
        return None

    null_item = None
    non_null_item = None

    for item in anyof_items:
        if isinstance(item, dict):
            item_type = item.get("type", "").lower()
            if item_type == "null":
                null_item = item
            else:
                non_null_item = item

    if null_item is not None and non_null_item is not None:
        # This is a nullable pattern: convert to Gemini format
        result = to_gemini_schema(non_null_item)
        if isinstance(result, dict):
            result["nullable"] = True
        return result

    return None


def to_gemini_schema(schema: Any) -> Any:
    """Transform a JSON Schema to Gemini-compatible format.

    Based on @google/genai SDK's processJsonSchema() function.

    Key transformations:
    - Converts type values to uppercase (object -> OBJECT, string -> STRING)
    - Removes unsupported fields ($schema, additionalProperties, etc.)
    - Recursively processes nested schemas (properties, items, anyOf, etc.)
    - Converts anyOf with null to nullable: true (Gemini format)
    - Ensures arrays have 'items' field (Gemini API requires it)

    Args:
        schema: A JSON Schema object or primitive value

    Returns:
        Gemini-compatible schema with uppercase types
    """
    # Return primitives and arrays as-is
    if not schema or not isinstance(schema, dict):
        return schema

    if isinstance(schema, list):
        return [to_gemini_schema(item) for item in schema]

    result: Dict[str, Any] = {}

    # First pass: collect all property names for required validation
    property_names: set[str] = set()
    if isinstance(schema.get("properties"), dict):
        property_names = set(schema["properties"].keys())

    # Check for anyOf pattern that represents nullable - convert early
    anyof = schema.get("anyOf")
    if isinstance(anyof, list):
        nullable_result = _convert_anyof_to_nullable(anyof)
        if nullable_result is not None:
            return nullable_result

    for key, value in schema.items():
        # Skip unsupported fields that Gemini API rejects
        if key in GEMINI_UNSUPPORTED_SCHEMA_FIELDS:
            continue

        if key == "type" and isinstance(value, str):
            # Convert type to uppercase for Gemini API
            result[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            # Recursively transform nested property schemas
            props: Dict[str, Any] = {}
            for prop_name, prop_schema in value.items():
                props[prop_name] = to_gemini_schema(prop_schema)
            result[key] = props
        elif key == "items" and isinstance(value, dict):
            # Transform array items schema
            result[key] = to_gemini_schema(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            # For non-nullable anyOf/oneOf/allOf, skip them as Gemini doesn't support these
            # We've already handled the nullable pattern above
            # Just use the first non-null type as a fallback
            for item in value:
                if isinstance(item, dict) and item.get("type", "").lower() != "null":
                    transformed = to_gemini_schema(item)
                    if isinstance(transformed, dict):
                        for k, v in transformed.items():
                            if k not in result:
                                result[k] = v
                    break
            # Don't include anyOf/oneOf/allOf in result - Gemini doesn't support them
            continue
        elif key == "enum" and isinstance(value, list):
            # Keep enum values as-is
            result[key] = value
        elif key in ("default", "examples"):
            # Keep default and examples as-is
            result[key] = value
        elif key == "required" and isinstance(value, list):
            # Filter required array to only include properties that exist
            # This fixes: "parameters.required[X]: property is not defined"
            if property_names:
                valid_required = [
                    prop for prop in value if isinstance(prop, str) and prop in property_names
                ]
                if valid_required:
                    result[key] = valid_required
                # If no valid required properties, omit the required field entirely
            else:
                # If there are no properties, keep required as-is
                result[key] = value
        else:
            result[key] = value

    # Issue #80: Ensure array schemas have an 'items' field
    # Gemini API requires: "parameters.properties[X].items: missing field"
    if result.get("type") == "ARRAY" and "items" not in result:
        result["items"] = {"type": "STRING"}

    return result


def ensure_tool_has_properties(
    schema: Dict[str, Any], uppercase_types: bool = True
) -> Dict[str, Any]:
    """Ensure tool schema has at least one property for VALIDATED mode.

    Claude/Antigravity VALIDATED mode requires tool parameters to be an object schema
    with at least one property. This adds a placeholder if needed.

    Args:
        schema: Tool parameter schema
        uppercase_types: If True, use uppercase types (OBJECT, BOOLEAN) for Gemini/Antigravity

    Returns:
        Schema with guaranteed properties
    """
    obj_type = "OBJECT" if uppercase_types else "object"
    bool_type = "BOOLEAN" if uppercase_types else "boolean"

    if not schema or not isinstance(schema, dict):
        return {
            "type": obj_type,
            "properties": {
                EMPTY_SCHEMA_PLACEHOLDER_NAME: {
                    "type": bool_type,
                    "description": EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION,
                }
            },
            "required": [EMPTY_SCHEMA_PLACEHOLDER_NAME],
        }

    result = dict(schema)
    # Ensure type is uppercase for Gemini/Antigravity
    if uppercase_types:
        result["type"] = "OBJECT"
    else:
        result["type"] = "object"

    # Check if properties exist and are non-empty
    has_properties = isinstance(result.get("properties"), dict) and len(result["properties"]) > 0

    if not has_properties:
        result["properties"] = {
            EMPTY_SCHEMA_PLACEHOLDER_NAME: {
                "type": bool_type,
                "description": EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION,
            }
        }
        required = result.get("required", [])
        if isinstance(required, list):
            result["required"] = list(set(required + [EMPTY_SCHEMA_PLACEHOLDER_NAME]))
        else:
            result["required"] = [EMPTY_SCHEMA_PLACEHOLDER_NAME]

    return result


# =============================================================================
# TOOL HARDENING FUNCTIONS
# =============================================================================


def _format_type_hint(prop_data: Dict[str, Any]) -> str:
    """Format a type hint string from property schema."""
    prop_type = prop_data.get("type", "any")

    if prop_type == "array":
        items = prop_data.get("items", {})
        if isinstance(items, dict):
            items_type = items.get("type", "any")
            return f"array of {items_type}"
        return "array"

    if prop_type == "object":
        return "object"

    return str(prop_type)


def inject_parameter_signatures(
    tools: List[Dict[str, Any]],
    prompt_template: str = CLAUDE_DESCRIPTION_PROMPT,
) -> List[Dict[str, Any]]:
    """Inject parameter signatures into tool descriptions (tool hardening).

    Appends a strict parameter signature to each tool description to prevent
    Claude from hallucinating parameters from training data.

    Args:
        tools: Gemini-style tools array (with functionDeclarations)
        prompt_template: Template for the signature (default: CLAUDE_DESCRIPTION_PROMPT)

    Returns:
        Modified tools array with signatures injected
    """
    if not tools or not isinstance(tools, list):
        return tools

    result = []
    for tool in tools:
        if not isinstance(tool, dict):
            result.append(tool)
            continue

        declarations = tool.get("functionDeclarations")
        if not isinstance(declarations, list):
            result.append(tool)
            continue

        new_declarations = []
        for decl in declarations:
            if not isinstance(decl, dict):
                new_declarations.append(decl)
                continue

            # Skip if signature already injected
            description = decl.get("description", "")
            if "STRICT PARAMETERS:" in description:
                new_declarations.append(decl)
                continue

            schema = decl.get("parameters") or decl.get("parametersJsonSchema")
            if not schema or not isinstance(schema, dict):
                new_declarations.append(decl)
                continue

            required = schema.get("required", [])
            if not isinstance(required, list):
                required = []

            properties = schema.get("properties", {})
            if not isinstance(properties, dict) or not properties:
                new_declarations.append(decl)
                continue

            # Build parameter list
            param_list = []
            for prop_name, prop_data in properties.items():
                if prop_name == EMPTY_SCHEMA_PLACEHOLDER_NAME:
                    continue
                if not isinstance(prop_data, dict):
                    continue
                type_hint = _format_type_hint(prop_data)
                is_required = prop_name in required
                param_str = f"{prop_name} ({type_hint}{', REQUIRED' if is_required else ''})"
                param_list.append(param_str)

            if not param_list:
                new_declarations.append(decl)
                continue

            # Inject signature
            sig_str = prompt_template.replace("{params}", ", ".join(param_list))
            new_decl = dict(decl)
            new_decl["description"] = (description or "") + sig_str
            new_declarations.append(new_decl)

        result.append({**tool, "functionDeclarations": new_declarations})

    return result


def inject_tool_hardening_instruction(
    payload: Dict[str, Any],
    instruction_text: str = CLAUDE_TOOL_SYSTEM_INSTRUCTION,
) -> None:
    """Inject a tool hardening system instruction into the request payload.

    Prepends the instruction to the systemInstruction to prevent Claude from
    hallucinating tool parameters.

    Args:
        payload: The Gemini request payload (modified in place)
        instruction_text: The instruction text to inject
    """
    if not instruction_text:
        return

    existing = payload.get("systemInstruction")

    # Skip if instruction already present
    if existing and isinstance(existing, dict) and "parts" in existing:
        parts = existing.get("parts", [])
        if isinstance(parts, list) and any(
            isinstance(p, dict) and "CRITICAL TOOL USAGE INSTRUCTIONS" in (p.get("text") or "")
            for p in parts
        ):
            return

    instruction_part = {"text": instruction_text}

    if payload.get("systemInstruction"):
        if isinstance(existing, dict) and "parts" in existing:
            parts = existing.get("parts", [])
            if isinstance(parts, list):
                parts.insert(0, instruction_part)
        elif isinstance(existing, str):
            payload["systemInstruction"] = {
                "role": "user",
                "parts": [instruction_part, {"text": existing}],
            }
        else:
            payload["systemInstruction"] = {
                "role": "user",
                "parts": [instruction_part],
            }
    else:
        payload["systemInstruction"] = {
            "role": "user",
            "parts": [instruction_part],
        }


# =============================================================================
# THINKING SIGNATURE MANAGEMENT (Claude multi-turn conversations)
# =============================================================================


def has_signed_thinking_part(part: Dict[str, Any]) -> bool:
    """Check if a part has a valid signed thinking signature."""
    if not part or not isinstance(part, dict):
        return False

    # Gemini format: thought=true + thoughtSignature
    if part.get("thought") is True:
        sig = part.get("thoughtSignature", "")
        return isinstance(sig, str) and len(sig) >= MIN_SIGNATURE_LENGTH

    # Anthropic format: type=thinking/redacted_thinking + signature
    part_type = part.get("type", "")
    if part_type in ("thinking", "redacted_thinking"):
        sig = part.get("signature", "")
        return isinstance(sig, str) and len(sig) >= MIN_SIGNATURE_LENGTH

    return False


def ensure_thinking_signature_for_tool_use(
    contents: List[Dict[str, Any]],
    session_key: str,
    allow_sentinel: bool = True,
) -> List[Dict[str, Any]]:
    """Ensure thinking blocks have signatures before tool use parts.

    For Claude thinking models, tool use requires signed thinking to precede it.
    If no valid signed thinking is found, optionally inject the SKIP_THOUGHT_SIGNATURE sentinel.
    Sentinel injection must be disabled for Vertex AI, which rejects it.

    Args:
        contents: Gemini-style contents array
        session_key: Session key for signature caching

    Returns:
        Modified contents with thinking signatures ensured
    """
    result = []

    for content in contents:
        if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
            result.append(content)
            continue

        role = content.get("role")
        if role not in ("model", "assistant"):
            result.append(content)
            continue

        parts = content["parts"]

        # Check if this content has tool use (functionCall)
        has_tool_use = any(
            isinstance(p, dict) and ("functionCall" in p or "tool_use" in p or "toolUse" in p)
            for p in parts
        )
        if not has_tool_use:
            result.append(content)
            continue

        # Separate thinking parts from other parts
        thinking_parts = []
        other_parts = []

        for part in parts:
            if not isinstance(part, dict):
                other_parts.append(part)
                continue

            is_thinking = part.get("thought") is True or part.get("type") in (
                "thinking",
                "reasoning",
                "redacted_thinking",
            )
            if is_thinking:
                thinking_parts.append(part)
            else:
                other_parts.append(part)

        # Check if we have a valid signed thinking
        has_signed = any(has_signed_thinking_part(p) for p in thinking_parts)

        if has_signed:
            # Put thinking parts first, then other parts
            result.append({**content, "parts": thinking_parts + other_parts})
        else:
            # No valid signed thinking. For Vertex AI, skip sentinel injection.
            if not allow_sentinel:
                if thinking_parts:
                    result.append({**content, "parts": thinking_parts + other_parts})
                else:
                    result.append(content)
                logger.debug(
                    "Skipping sentinel signature injection for tool use (session=%s)",
                    session_key,
                )
                continue

            # Non-Vertex: inject sentinel signature to bypass validation
            existing_thinking = thinking_parts[0] if thinking_parts else {}
            thinking_text = existing_thinking.get("thinking") or existing_thinking.get("text") or ""

            sentinel_block = {
                "type": "thinking",
                "thinking": thinking_text,
                "signature": SKIP_THOUGHT_SIGNATURE,
            }

            logger.debug("Injecting sentinel signature for tool use (session=%s)", session_key)
            result.append({**content, "parts": [sentinel_block] + other_parts})

    return result


def has_valid_signature_for_function_calls(
    contents: List[Dict[str, Any]],
    session_id: Optional[str],
    model: Optional[str] = None,
) -> bool:
    """Check if any valid thought signature exists for tool calls."""
    if session_id:
        sig = get_session_signature(session_id, model)
        if sig and len(sig) >= MIN_SIGNATURE_LENGTH:
            return True

    for content in contents:
        if not isinstance(content, dict) or content.get("role") not in ("model", "assistant"):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue
        for part in parts:
            if not isinstance(part, dict):
                continue
            sig = _extract_part_thought_signature(part)
            if sig and len(sig) >= MIN_SIGNATURE_LENGTH:
                return True
            if has_signed_thinking_part(part):
                return True

    return False


def strip_tool_call_signatures(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove thought_signature fields from OpenAI-style tool_calls."""
    if not messages:
        return messages

    cleaned: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            cleaned.append(msg)
            continue
        tool_calls = msg.get("tool_calls")
        if msg.get("role") != "assistant" or not tool_calls:
            cleaned.append(msg)
            continue

        new_msg = dict(msg)
        new_calls = []
        for call in tool_calls:
            if isinstance(call, dict):
                call_copy = dict(call)
                call_copy.pop("thought_signature", None)
                call_copy.pop("thoughtSignature", None)
                new_calls.append(call_copy)
            else:
                new_calls.append(call)
        new_msg["tool_calls"] = new_calls
        cleaned.append(new_msg)

    return cleaned


# =============================================================================
# TOOL DEFINITION CONVERSION
# =============================================================================


def convert_tools_to_gemini_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Gemini functionDeclarations format.

    OpenAI format:
        [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Gemini format:
        [{"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}]

    Key transformations for Gemini compatibility:
    - Types converted to uppercase (string -> STRING, object -> OBJECT)
    - Unsupported fields removed ($schema, additionalProperties, etc.)
    - Arrays ensured to have 'items' field
    - Required fields validated against existing properties

    Args:
        tools: OpenAI-style tool definitions

    Returns:
        Gemini-style tool definitions with uppercase type schemas
    """
    if not tools:
        return []

    function_declarations = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        # Handle OpenAI function tool format
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            func = tool["function"]
            declaration: Dict[str, Any] = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            if "parameters" in func:
                # Resolve $ref references first, then transform to Gemini format
                resolved_params = _resolve_refs(func["parameters"])
                gemini_params = to_gemini_schema(resolved_params)
                declaration["parameters"] = gemini_params
            else:
                # Provide default placeholder schema for tools without parameters
                declaration["parameters"] = {
                    "type": "OBJECT",
                    "properties": {
                        "_placeholder": {
                            "type": "BOOLEAN",
                            "description": "Placeholder. Always pass true.",
                        }
                    },
                    "required": ["_placeholder"],
                }
            function_declarations.append(declaration)

        # Handle direct function declaration
        elif "name" in tool:
            declaration = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
            params = tool.get("parameters") or tool.get("input_schema") or tool.get("inputSchema")
            if params:
                # Resolve $ref references first, then transform to Gemini format
                resolved_params = _resolve_refs(params)
                gemini_params = to_gemini_schema(resolved_params)
                declaration["parameters"] = gemini_params
            else:
                # Provide default placeholder schema
                declaration["parameters"] = {
                    "type": "OBJECT",
                    "properties": {
                        "_placeholder": {
                            "type": "BOOLEAN",
                            "description": "Placeholder. Always pass true.",
                        }
                    },
                    "required": ["_placeholder"],
                }
            function_declarations.append(declaration)

        # Handle already wrapped functionDeclarations
        elif "functionDeclarations" in tool:
            # Transform schemas in existing declarations
            for decl in tool["functionDeclarations"]:
                cleaned_decl = dict(decl)
                if "parameters" in cleaned_decl:
                    # Resolve $ref references first, then transform to Gemini format
                    resolved_params = _resolve_refs(cleaned_decl["parameters"])
                    cleaned_decl["parameters"] = to_gemini_schema(resolved_params)
                else:
                    cleaned_decl["parameters"] = {
                        "type": "OBJECT",
                        "properties": {},
                    }
                function_declarations.append(cleaned_decl)

    if not function_declarations:
        return []

    return [{"functionDeclarations": function_declarations}]


def convert_tools_to_claude_format(
    tools: List[Dict[str, Any]],
    inject_signatures: bool = True,
) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Claude format with schema cleaning.

    Applies:
    - Conversion to functionDeclarations format
    - JSON schema cleaning for VALIDATED mode
    - Optional parameter signature injection (tool hardening)

    Args:
        tools: OpenAI-style tool definitions
        inject_signatures: Whether to inject parameter signatures into descriptions

    Returns:
        Claude-style tool definitions with cleaned schemas
    """
    if not tools:
        return []

    function_declarations = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        # Extract function info
        func = None
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            func = tool["function"]
        elif "name" in tool:
            func = tool
        elif "functionDeclarations" in tool:
            # Already in Claude format, just clean schemas
            for decl in tool["functionDeclarations"]:
                cleaned = dict(decl)
                if "parameters" in cleaned:
                    # Use Claude-specific schema cleaning for VALIDATED mode
                    cleaned["parameters"] = clean_json_schema_for_claude(cleaned["parameters"])
                function_declarations.append(cleaned)
            continue

        if not func:
            continue

        # Build declaration
        name = str(func.get("name", "")).replace(" ", "_")[:64]
        description = func.get("description", "")

        # Get parameters
        params = (
            func.get("parameters")
            or func.get("input_schema")
            or func.get("inputSchema")
            or {"type": "object", "properties": {}}
        )

        # Clean schema using Claude-specific cleaning for VALIDATED mode
        cleaned_params = clean_json_schema_for_claude(params)

        # Optionally inject parameter signatures into description (tool hardening)
        if inject_signatures and cleaned_params.get("properties"):
            sig_parts = []
            for param_name, param_schema in cleaned_params["properties"].items():
                if param_name == EMPTY_SCHEMA_PLACEHOLDER_NAME:
                    continue
                param_type = param_schema.get("type", "any")
                sig_parts.append(f"{param_name}: {param_type}")
            if sig_parts:
                signature = f"({', '.join(sig_parts)})"
                if description:
                    description = f"{description} {signature}"
                else:
                    description = signature

        declaration = {
            "name": name,
            "description": description,
            "parameters": cleaned_params,
        }
        function_declarations.append(declaration)

    if not function_declarations:
        return []

    return [{"functionDeclarations": function_declarations}]


def convert_tools_to_codex_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to ChatGPT Codex (Responses API) format.

    OpenAI Chat Completions format:
        [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    OpenAI Responses format (Codex API):
        [{"type": "function", "name": "...", "description": "...", "parameters": {...}}]

    The key difference is that Responses format has name/description at the top level
    alongside type, not nested under a "function" key.

    Args:
        tools: OpenAI Chat Completions-style tool definitions

    Returns:
        OpenAI Responses-style tool definitions for Codex API
    """
    if not tools:
        return []

    codex_tools = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        # Handle OpenAI Chat Completions format (nested under "function")
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            func = tool["function"]
            codex_tool: Dict[str, Any] = {
                "type": "function",
                "name": func.get("name", ""),
            }
            if "description" in func:
                codex_tool["description"] = func["description"]
            if "parameters" in func:
                codex_tool["parameters"] = func["parameters"]
            if "strict" in func:
                codex_tool["strict"] = func["strict"]
            codex_tools.append(codex_tool)

        # Handle already-converted Responses format or direct format
        elif "type" in tool and "name" in tool:
            # Already in Responses format, pass through
            codex_tools.append(tool)

        # Handle direct function declaration (no type wrapper)
        elif "name" in tool:
            codex_tool = {
                "type": "function",
                "name": tool.get("name", ""),
            }
            if "description" in tool:
                codex_tool["description"] = tool["description"]
            params = tool.get("parameters") or tool.get("input_schema")
            if params:
                codex_tool["parameters"] = params
            codex_tools.append(codex_tool)

    return codex_tools


# =============================================================================
# TOOL MESSAGE CONVERSION (OpenAI → Gemini/Claude format)
# =============================================================================


def convert_tool_message_to_gemini_part(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI tool result message to a Gemini functionResponse part.

    OpenAI format:
        {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

    Gemini format:
        {"functionResponse": {"name": "...", "response": {...}, "id": "..."}}

    Note: The 'id' field is REQUIRED for Antigravity+Claude to match tool results
    with their corresponding tool calls. Without it, you get:
    "messages.X.content.0.tool_result.tool_use_id: Field required"

    Args:
        msg: OpenAI-style tool result message

    Returns:
        Gemini-style functionResponse part
    """
    name = msg.get("name", msg.get("tool_name", "unknown_function"))
    content = msg.get("content", "")
    # Get tool_call_id - this is REQUIRED for Claude via Antigravity
    tool_call_id = msg.get("tool_call_id") or msg.get("tool_use_id") or msg.get("id", "")

    # Try to parse content as JSON
    response_data: Any
    if isinstance(content, str):
        try:
            response_data = json.loads(content)
        except json.JSONDecodeError:
            response_data = {"result": content}
    elif isinstance(content, dict):
        response_data = content
    else:
        response_data = {"result": str(content)}

    func_response: Dict[str, Any] = {
        "name": name,
        "response": response_data,
    }
    # Include id if available (required for Claude via Antigravity)
    if tool_call_id:
        func_response["id"] = tool_call_id

    return {"functionResponse": func_response}


def convert_tool_calls_to_gemini_parts(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    """Convert OpenAI tool_calls to Gemini functionCall parts.

    OpenAI format:
        [{"id": "...", "function": {"name": "...", "arguments": "..."}, "thought_signature": "..."}]

    Gemini format:
        [{"functionCall": {"name": "...", "args": {...}}, "thoughtSignature": "..."}]

    Note: thoughtSignature is required for Gemini 3 models. It's preserved from the
    original response and must be passed back exactly as received. If not found in
    the tool call, attempts to retrieve from cache.

    Args:
        tool_calls: OpenAI-style tool calls (may be dicts or ChatCompletionMessageToolCall objects)

    Returns:
        Gemini-style functionCall parts (with thoughtSignature if present or cached)
    """
    parts = []
    for call in tool_calls:
        # Handle both dict and ChatCompletionMessageToolCall objects
        call_id = None
        if isinstance(call, dict):
            call_id = call.get("id")
            func = call.get("function", {})
            if isinstance(func, dict):
                name = func.get("name", call.get("name", ""))
                args_str = func.get("arguments", call.get("arguments", "{}"))
            else:
                # func might be a Function object
                name = getattr(func, "name", call.get("name", ""))
                args_str = getattr(func, "arguments", call.get("arguments", "{}"))
            # Get thought_signature from dict
            thought_sig = call.get("thought_signature") or call.get("thoughtSignature")
        else:
            # Handle ChatCompletionMessageToolCall objects (from LiteLLM)
            call_id = getattr(call, "id", None)
            func = getattr(call, "function", None)
            if func:
                name = getattr(func, "name", "")
                args_str = getattr(func, "arguments", "{}")
            else:
                name = getattr(call, "name", "")
                args_str = getattr(call, "arguments", "{}")
            # Try to get thought_signature from object attributes
            thought_sig = getattr(call, "thought_signature", None) or getattr(
                call, "thoughtSignature", None
            )

        # If thought_signature not found, try to retrieve from cache (call_id required)
        if not thought_sig and call_id and name:
            thought_sig = get_cached_thought_signature(call_id, name)

        # Parse arguments
        if isinstance(args_str, str):
            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {}
        else:
            args = args_str or {}

        # Build functionCall with ID for proper pairing
        func_call: Dict[str, Any] = {
            "name": name,
            "args": args,
        }
        # CRITICAL: Include ID for tool pairing (Claude via Antigravity requires this)
        # The ID must match the tool_call_id in the subsequent functionResponse
        if call_id:
            func_call["id"] = call_id

        part: Dict[str, Any] = {"functionCall": func_call}

        # Preserve thoughtSignature for Gemini 3 models
        if thought_sig:
            part["thoughtSignature"] = thought_sig

        parts.append(part)

    return parts


# =============================================================================
# TOOL ID MANAGEMENT (Claude/Antigravity)
# =============================================================================


def assign_tool_call_ids(
    contents: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Pass 1: Assign IDs to functionCall parts.

    Generates unique IDs for functionCall parts that don't have them.
    Returns the modified contents and a dict of pending call IDs by function name.

    Args:
        contents: Gemini-style contents array

    Returns:
        Tuple of (modified contents, pending IDs by function name)
    """
    pending_by_name: Dict[str, List[str]] = defaultdict(list)
    result = []

    for content in contents:
        if not isinstance(content, dict):
            result.append(content)
            continue

        parts = content.get("parts")
        if not isinstance(parts, list):
            result.append(content)
            continue

        new_parts = []
        for part in parts:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue

            if "functionCall" in part:
                call = dict(part["functionCall"])
                if not call.get("id"):
                    call["id"] = f"tool-call-{uuid.uuid4().hex[:8]}"
                name = call.get("name", "unknown")
                pending_by_name[name].append(call["id"])
                new_parts.append({**part, "functionCall": call})
            else:
                new_parts.append(part)

        result.append({**content, "parts": new_parts})

    return result, dict(pending_by_name)


def match_tool_response_ids(
    contents: List[Dict[str, Any]],
    pending_by_name: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Pass 2: Match functionResponse IDs using FIFO queue per function name.

    Assigns IDs to functionResponse parts by matching them to pending
    functionCall IDs in FIFO order.

    Args:
        contents: Gemini-style contents array (already processed by pass 1)
        pending_by_name: Dict of pending call IDs by function name

    Returns:
        Modified contents with matched response IDs
    """
    # Create a mutable copy of pending queues
    queues = {name: list(ids) for name, ids in pending_by_name.items()}
    result = []

    for content in contents:
        if not isinstance(content, dict):
            result.append(content)
            continue

        parts = content.get("parts")
        if not isinstance(parts, list):
            result.append(content)
            continue

        new_parts = []
        for part in parts:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue

            if "functionResponse" in part:
                resp = dict(part["functionResponse"])
                name = resp.get("name", "unknown")

                # Try to match with pending call
                if not resp.get("id") and name in queues and queues[name]:
                    resp["id"] = queues[name].pop(0)  # FIFO

                new_parts.append({**part, "functionResponse": resp})
            else:
                new_parts.append(part)

        result.append({**content, "parts": new_parts})

    return result


def fix_tool_response_grouping(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pass 3: Orphan recovery - fix mismatched tool call/response pairs.

    Handles cases where functionResponse parts don't have matching IDs
    by attempting to pair them with unmatched functionCall parts.

    Args:
        contents: Gemini-style contents array

    Returns:
        Modified contents with orphan responses fixed
    """
    # First, collect all call IDs and response IDs
    all_call_ids: Dict[str, List[str]] = defaultdict(list)  # name -> [ids]
    all_response_ids: Dict[str, List[str]] = defaultdict(list)  # name -> [ids]
    call_names: Dict[str, str] = {}  # id -> name

    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            continue

        for part in parts:
            if not isinstance(part, dict):
                continue
            if "functionCall" in part:
                call = part["functionCall"]
                call_id = call.get("id")
                name = call.get("name", "unknown")
                if call_id:
                    all_call_ids[name].append(call_id)
                    call_names[call_id] = name
            elif "functionResponse" in part:
                resp = part["functionResponse"]
                resp_id = resp.get("id")
                name = resp.get("name", "unknown")
                if resp_id:
                    all_response_ids[name].append(resp_id)

    # Find orphaned responses (responses without matching calls)
    result = []
    for content in contents:
        if not isinstance(content, dict):
            result.append(content)
            continue

        parts = content.get("parts", [])
        if not isinstance(parts, list):
            result.append(content)
            continue

        new_parts = []
        for part in parts:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue

            if "functionResponse" in part:
                resp = dict(part["functionResponse"])
                resp_id = resp.get("id")
                name = resp.get("name", "unknown")

                # Check if this response's ID matches any call
                if resp_id and resp_id not in call_names:
                    # Try to find an unmatched call with the same name
                    if name in all_call_ids and all_call_ids[name]:
                        unmatched = [
                            cid
                            for cid in all_call_ids[name]
                            if cid not in all_response_ids.get(name, [])
                        ]
                        if unmatched:
                            resp["id"] = unmatched[0]
                            all_response_ids[name].append(unmatched[0])

                new_parts.append({**part, "functionResponse": resp})
            else:
                new_parts.append(part)

        result.append({**content, "parts": new_parts})

    return result


def apply_tool_pairing_fixes(
    payload: Dict[str, Any],
    is_claude: bool = False,
) -> Dict[str, Any]:
    """Apply all tool pairing fixes to a request payload.

    Runs the full 3-pass tool ID management pipeline:
    1. Assign IDs to functionCall parts
    2. Match functionResponse IDs using FIFO
    3. Orphan recovery

    Args:
        payload: Request payload with 'contents' array
        is_claude: Whether this is for Claude (enables additional validation)

    Returns:
        Modified payload with fixed tool pairing
    """
    if "contents" not in payload or not isinstance(payload["contents"], list):
        return payload

    result = dict(payload)

    # Pass 1: Assign IDs
    contents_with_ids, pending = assign_tool_call_ids(result["contents"])

    # Pass 2: Match response IDs
    contents_matched = match_tool_response_ids(contents_with_ids, pending)

    # Pass 3: Orphan recovery
    result["contents"] = fix_tool_response_grouping(contents_matched)

    return result


# =============================================================================
# TOOL CALL EXTRACTION FROM RESPONSES
# =============================================================================


def remap_function_call_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Remap Claude's hallucinated tool arguments to match actual schemas.

    Claude sometimes hallucinates incorrect parameter names. This function
    remaps common hallucinations to the correct parameter names.

    Args:
        tool_name: Name of the tool being called
        args: Original arguments dict from Claude

    Returns:
        Remapped arguments dict with corrected parameter names
    """
    if not isinstance(args, dict):
        return args

    result = dict(args)
    name_lower = tool_name.lower()

    # grep/search tools: description→pattern, query→pattern, paths[]→path
    if name_lower in ("grep", "grep_search", "search", "ripgrep"):
        # Remap description → pattern
        if "description" in result and "pattern" not in result:
            result["pattern"] = result.pop("description")
        # Remap query → pattern
        if "query" in result and "pattern" not in result:
            result["pattern"] = result.pop("query")
        # CRITICAL: paths[] → path (string) - Claude often returns array
        if "paths" in result and "path" not in result:
            paths_val = result.pop("paths")
            if isinstance(paths_val, list) and paths_val:
                result["path"] = paths_val[0]
            elif isinstance(paths_val, str):
                result["path"] = paths_val
            else:
                result["path"] = "."
        # Remap directory → path
        if "directory" in result and "path" not in result:
            result["path"] = result.pop("directory")

    # glob/glob_search tools: paths[]→path, pattern can stay
    elif name_lower in ("glob", "glob_search", "find_files"):
        # CRITICAL: paths[] → path (string)
        if "paths" in result and "path" not in result:
            paths_val = result.pop("paths")
            if isinstance(paths_val, list) and paths_val:
                result["path"] = paths_val[0]
            elif isinstance(paths_val, str):
                result["path"] = paths_val
            else:
                result["path"] = "."
        # Remap directory → path
        if "directory" in result and "path" not in result:
            result["path"] = result.pop("directory")

    # read/read_file tools: path→file_path
    elif name_lower in ("read", "read_file", "cat", "view_file"):
        # Remap path → file_path (Claude Code schema uses file_path)
        if "path" in result and "file_path" not in result:
            result["file_path"] = result.pop("path")
        # Remap filename → file_path
        if "filename" in result and "file_path" not in result:
            result["file_path"] = result.pop("filename")

    # write/write_file tools: path→file_path
    elif name_lower in ("write", "write_file", "create_file"):
        # Remap path → file_path
        if "path" in result and "file_path" not in result:
            result["file_path"] = result.pop("path")
        # Remap filename → file_path
        if "filename" in result and "file_path" not in result:
            result["file_path"] = result.pop("filename")

    # edit/edit_file tools: path→file_path
    elif name_lower in ("edit", "edit_file", "modify_file", "patch_file"):
        # Remap path → file_path
        if "path" in result and "file_path" not in result:
            result["file_path"] = result.pop("path")

    # list_directory/ls tools: path→directory
    elif name_lower in ("list_directory", "ls", "list_dir", "dir"):
        # Remap path → directory (some schemas use directory)
        if "path" in result and "directory" not in result:
            directory = result.get("directory")
            if not directory:
                result["directory"] = result.pop("path")

    # run_shell/bash/execute tools: cmd→command
    elif name_lower in ("run_shell", "bash", "shell", "execute", "run_command"):
        # Remap cmd → command
        if "cmd" in result and "command" not in result:
            result["command"] = result.pop("cmd")
        # Remap script → command
        if "script" in result and "command" not in result:
            result["command"] = result.pop("script")

    return result


def extract_tool_calls_from_gemini_response(
    data: Dict[str, Any],
    session_id: Optional[str] = None,
    message_count: Optional[int] = None,
    model: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Extract functionCall parts from Gemini/Antigravity response.

    Preserves thoughtSignature for Gemini 3 models - this must be passed back
    exactly as received when sending tool results. Also caches the signature
    for retrieval during follow-up calls (since LiteLLM streaming may lose it).

    Args:
        data: Response data from Gemini API

    Returns:
        List of OpenAI-style tool calls (with thought_signature if present),
        or None if no tool calls found
    """
    # Handle wrapped response
    if "response" in data:
        data = data["response"]

    candidates = data.get("candidates", [])
    if not candidates:
        return None

    tool_calls = []
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        current_thought_sig: Optional[str] = None

        for part in parts:
            if not isinstance(part, dict):
                continue

            # Cache thoughtSignature from thinking parts (Gemini 3)
            thought_sig = _extract_part_thought_signature(part)
            if thought_sig and (
                part.get("thought") is True or part.get("type") in ("thinking", "reasoning")
            ):
                current_thought_sig = thought_sig
                cache_thinking_signature(thought_sig)
                if session_id and message_count is not None:
                    cache_session_signature(session_id, thought_sig, message_count, model=model)

            if "functionCall" in part:
                func_call = part["functionCall"]
                name = func_call.get("name", "")
                args = func_call.get("args", {})
                call_id = func_call.get("id", f"call_{uuid.uuid4().hex[:8]}")

                # Remap hallucinated arguments to match actual tool schemas
                # Only apply for Claude models to avoid affecting Gemini flows
                is_claude = model and "claude" in model.lower()
                if isinstance(args, dict) and is_claude:
                    args = remap_function_call_args(name, args)

                # Convert to OpenAI format
                tool_call: Dict[str, Any] = {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    },
                }

                # Preserve thoughtSignature for Gemini 3 models
                # Store as snake_case in OpenAI format for consistency
                thought_sig = _extract_part_thought_signature(part) or current_thought_sig
                if thought_sig:
                    tool_call["thought_signature"] = thought_sig
                    # Also cache for retrieval during follow-up calls
                    cache_thought_signature(call_id, name, thought_sig)
                    if session_id and message_count is not None:
                        cache_session_signature(session_id, thought_sig, message_count, model=model)

                tool_calls.append(tool_call)

    return tool_calls if tool_calls else None


def extract_tool_calls_from_codex_response(
    data: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Extract function_call items from ChatGPT Codex response.

    Args:
        data: Response data from Codex API (from response.done event)

    Returns:
        List of OpenAI-style tool calls, or None if no tool calls found
    """
    output = data.get("output", [])
    if not isinstance(output, list):
        return None

    tool_calls = []
    for item in output:
        if not isinstance(item, dict):
            continue

        if item.get("type") == "function_call":
            call_id = item.get("call_id", f"call_{uuid.uuid4().hex[:8]}")
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")

            # Ensure arguments is a string
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments)

            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

    return tool_calls if tool_calls else None


def build_tool_calls_response_message(
    tool_calls: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build an assistant message with tool_calls for ModelResponse.

    Args:
        tool_calls: List of OpenAI-style tool calls

    Returns:
        Assistant message dict with tool_calls
    """
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
