"""Tool calling utilities for OAuth providers.

Shared utilities for converting, cleaning, and managing tool calls across
different OAuth providers (Google, Claude, ChatGPT, Antigravity).

Reference implementations:
- /tmp/oauth-providers/antigravity-auth/src/plugin/request-helpers.ts
- /tmp/oauth-providers/antigravity-auth/src/plugin/transform/claude.ts
- /tmp/oauth-providers/chatgpt-auth/lib/request/request-transformer.ts
"""

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# THOUGHT SIGNATURE CACHE (Gemini 3)
# Stores thoughtSignature values to survive LiteLLM's streaming handler
# Thread-safe via threading.Lock for concurrent access
# =============================================================================

# Cache: {(call_id, function_name): (thought_signature, timestamp)}
_thought_signature_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}
# Most recent thought_signature for parallel calls fallback (per function name)
_last_thought_signature_by_name: Dict[str, Tuple[str, float]] = {}
# Global last signature for cross-function parallel calls
_last_thought_signature: Optional[Tuple[str, float]] = None
_CACHE_TTL_SECONDS = 300  # 5 minutes TTL
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

        # Also store global last signature for parallel call fallback
        _last_thought_signature = (thought_signature, now)

        # Clean up old entries and enforce size limit
        _cleanup_thought_signature_cache_unlocked()


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

        # Strategy 3: Per-function-name fallback for parallel calls
        # Use batch window (stricter than TTL) to avoid cross-request pollution
        if normalized_name in _last_thought_signature_by_name:
            sig, timestamp = _last_thought_signature_by_name[normalized_name]
            if now - timestamp < _BATCH_WINDOW_SECONDS:
                logger.debug(
                    f"Using per-function fallback signature for {function_name} "
                    f"(age: {now - timestamp:.2f}s)"
                )
                return sig

        # Strategy 4: Global fallback for parallel calls within batch window only
        # Gemini only provides thoughtSignature for the first parallel call
        if _last_thought_signature:
            sig, timestamp = _last_thought_signature
            if now - timestamp < _BATCH_WINDOW_SECONDS:
                logger.debug(
                    f"Using global fallback signature for {function_name} "
                    f"(age: {now - timestamp:.2f}s)"
                )
                return sig

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


# Placeholder for tools with empty schemas (Claude VALIDATED mode requires at least one property)
EMPTY_SCHEMA_PLACEHOLDER_NAME = "_placeholder"
EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION = "Placeholder. Always pass true."

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


def ensure_tool_has_properties(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure tool schema has at least one property for VALIDATED mode.

    Claude VALIDATED mode requires tool parameters to be an object schema
    with at least one property. This adds a placeholder if needed.
    """
    if not schema or not isinstance(schema, dict):
        return {
            "type": "object",
            "properties": {
                EMPTY_SCHEMA_PLACEHOLDER_NAME: {
                    "type": "boolean",
                    "description": EMPTY_SCHEMA_PLACEHOLDER_DESCRIPTION,
                }
            },
            "required": [EMPTY_SCHEMA_PLACEHOLDER_NAME],
        }

    result = dict(schema)
    result["type"] = "object"

    # Check if properties exist and are non-empty
    has_properties = isinstance(result.get("properties"), dict) and len(result["properties"]) > 0

    if not has_properties:
        result["properties"] = {
            EMPTY_SCHEMA_PLACEHOLDER_NAME: {
                "type": "boolean",
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
# TOOL DEFINITION CONVERSION
# =============================================================================


def convert_tools_to_gemini_format(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style tools to Gemini functionDeclarations format.

    OpenAI format:
        [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

    Gemini format:
        [{"functionDeclarations": [{"name": "...", "description": "...", "parameters": {...}}]}]

    Args:
        tools: OpenAI-style tool definitions

    Returns:
        Gemini-style tool definitions with cleaned schemas
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
            declaration = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
            }
            if "parameters" in func:
                # Clean schema for Gemini compatibility
                cleaned_params = clean_json_schema(func["parameters"])
                declaration["parameters"] = cleaned_params
            function_declarations.append(declaration)

        # Handle direct function declaration
        elif "name" in tool:
            declaration = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
            params = tool.get("parameters") or tool.get("input_schema") or tool.get("inputSchema")
            if params:
                # Clean schema for Gemini compatibility
                cleaned_params = clean_json_schema(params)
                declaration["parameters"] = cleaned_params
            function_declarations.append(declaration)

        # Handle already wrapped functionDeclarations
        elif "functionDeclarations" in tool:
            # Clean schemas in existing declarations
            for decl in tool["functionDeclarations"]:
                cleaned_decl = dict(decl)
                if "parameters" in cleaned_decl:
                    cleaned_decl["parameters"] = clean_json_schema(cleaned_decl["parameters"])
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
                    cleaned["parameters"] = clean_json_schema(cleaned["parameters"])
                    cleaned["parameters"] = ensure_tool_has_properties(cleaned["parameters"])
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

        # Clean schema
        cleaned_params = clean_json_schema(params)
        cleaned_params = ensure_tool_has_properties(cleaned_params)

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
        {"functionResponse": {"name": "...", "response": {...}}}

    Args:
        msg: OpenAI-style tool result message

    Returns:
        Gemini-style functionResponse part
    """
    name = msg.get("name", msg.get("tool_name", "unknown_function"))
    content = msg.get("content", "")

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

    return {
        "functionResponse": {
            "name": name,
            "response": response_data,
        }
    }


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

        # If thought_signature not found, try to retrieve from cache
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

        part: Dict[str, Any] = {
            "functionCall": {
                "name": name,
                "args": args,
            }
        }

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


def extract_tool_calls_from_gemini_response(
    data: Dict[str, Any],
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

        for part in parts:
            if not isinstance(part, dict):
                continue

            if "functionCall" in part:
                func_call = part["functionCall"]
                name = func_call.get("name", "")
                args = func_call.get("args", {})
                call_id = func_call.get("id", f"call_{uuid.uuid4().hex[:8]}")

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
                thought_sig = part.get("thoughtSignature")
                if thought_sig:
                    tool_call["thought_signature"] = thought_sig
                    # Also cache for retrieval during follow-up calls
                    cache_thought_signature(call_id, name, thought_sig)

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
