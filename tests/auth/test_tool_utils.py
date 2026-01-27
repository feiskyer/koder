"""Unit tests for tool_utils.py."""

import json

from koder_agent.auth.tool_utils import (
    apply_tool_pairing_fixes,
    # Tool ID management
    assign_tool_call_ids,
    build_tool_calls_response_message,
    # Schema cleaning
    clean_json_schema,
    convert_tool_calls_to_gemini_parts,
    # Tool message conversion
    convert_tool_message_to_gemini_part,
    convert_tools_to_claude_format,
    convert_tools_to_codex_format,
    # Tool format conversion
    convert_tools_to_gemini_format,
    ensure_tool_has_properties,
    extract_tool_calls_from_codex_response,
    # Tool call extraction
    extract_tool_calls_from_gemini_response,
    match_tool_response_ids,
)


class TestCleanJsonSchemaForAntigravity:
    """Tests for clean_json_schema."""

    def test_removes_defs_and_refs(self):
        """Should remove $defs and convert $ref to description hints."""
        schema = {
            "type": "object",
            "$defs": {"MyType": {"type": "string", "description": "A custom type"}},
            "properties": {"field": {"$ref": "#/$defs/MyType"}},
        }
        result = clean_json_schema(schema)

        # $defs should be removed
        assert "$defs" not in result
        # $ref should be converted to description hint
        assert "$ref" not in result["properties"]["field"]
        assert "See: MyType" in result["properties"]["field"].get("description", "")

    def test_converts_const_to_enum(self):
        """Should convert const values to enum arrays."""
        schema = {"type": "object", "properties": {"status": {"const": "active"}}}
        result = clean_json_schema(schema)

        assert "const" not in result["properties"]["status"]
        assert result["properties"]["status"].get("enum") == ["active"]

    def test_adds_enum_hints(self):
        """Should add enum hints to descriptions."""
        schema = {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        }
        result = clean_json_schema(schema)

        description = result["properties"]["color"].get("description", "")
        assert "Allowed: red, green, blue" in description

    def test_removes_unsupported_constraints(self):
        """Should move unsupported constraints to description."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 100}},
        }
        result = clean_json_schema(schema)

        prop = result["properties"]["name"]
        # Constraints should be in description, not as keywords
        assert "minLength" not in prop or "minLength: 1" in prop.get("description", "")
        assert "maxLength" not in prop or "maxLength: 100" in prop.get("description", "")

    def test_removes_schema_keyword(self):
        """Should remove $schema keyword."""
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {},
        }
        result = clean_json_schema(schema)

        assert "$schema" not in result

    def test_fixes_invalid_required_fields(self):
        """Should remove required fields that reference non-existent properties."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name", "nonexistent"],
        }
        result = clean_json_schema(schema)

        # Only "name" should remain in required
        assert "name" in result.get("required", [])
        assert "nonexistent" not in result.get("required", [])

    def test_handles_empty_schema(self):
        """Should handle empty or None schema."""
        assert clean_json_schema(None) == {"type": "object", "properties": {}}
        assert clean_json_schema({}) == {"type": "object", "properties": {}}


class TestEnsureToolHasProperties:
    """Tests for ensure_tool_has_properties."""

    def test_adds_placeholder_for_empty_properties(self):
        """Should add placeholder for empty properties (uppercase for Gemini/Antigravity)."""
        schema = {"type": "object", "properties": {}}
        result = ensure_tool_has_properties(schema)

        assert "_placeholder" in result["properties"]
        # Default is uppercase_types=True for Gemini/Antigravity compatibility
        assert result["properties"]["_placeholder"]["type"] == "BOOLEAN"

    def test_preserves_existing_properties(self):
        """Should preserve existing properties."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = ensure_tool_has_properties(schema)

        assert result["properties"]["name"]["type"] == "string"
        assert "_placeholder" not in result["properties"]

    def test_handles_none_schema(self):
        """Should handle None schema."""
        result = ensure_tool_has_properties(None)

        assert "_placeholder" in result["properties"]


class TestConvertToolsToGeminiFormat:
    """Tests for convert_tools_to_gemini_format."""

    def test_converts_openai_format(self):
        """Should convert OpenAI-style tools to Gemini format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        result = convert_tools_to_gemini_format(tools)

        assert len(result) == 1
        assert "functionDeclarations" in result[0]
        decl = result[0]["functionDeclarations"][0]
        assert decl["name"] == "get_weather"
        assert decl["description"] == "Get weather for a location"
        assert "parameters" in decl

    def test_cleans_schemas(self):
        """Should clean schemas during conversion."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "Test",
                    "parameters": {
                        "type": "object",
                        "$schema": "http://...",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        ]
        result = convert_tools_to_gemini_format(tools)

        params = result[0]["functionDeclarations"][0]["parameters"]
        assert "$schema" not in params

    def test_handles_empty_tools(self):
        """Should handle empty tools list."""
        assert convert_tools_to_gemini_format([]) == []
        assert convert_tools_to_gemini_format(None) == []


class TestConvertToolsToClaudeFormat:
    """Tests for convert_tools_to_claude_format."""

    def test_injects_signatures(self):
        """Should inject parameter signatures into description."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    },
                },
            }
        ]
        result = convert_tools_to_claude_format(tools, inject_signatures=True)

        decl = result[0]["functionDeclarations"][0]
        assert "(a: number, b: number)" in decl["description"]

    def test_without_signatures(self):
        """Should skip signature injection when disabled."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add numbers",
                    "parameters": {"type": "object", "properties": {"a": {"type": "number"}}},
                },
            }
        ]
        result = convert_tools_to_claude_format(tools, inject_signatures=False)

        decl = result[0]["functionDeclarations"][0]
        assert decl["description"] == "Add numbers"


class TestConvertToolsToCodexFormat:
    """Tests for convert_tools_to_codex_format."""

    def test_converts_chat_completions_to_responses(self):
        """Should convert Chat Completions format to Responses format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }
        ]
        result = convert_tools_to_codex_format(tools)

        assert len(result) == 1
        # Responses format: name at top level, not nested
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "read_file"
        assert result[0]["description"] == "Read a file"
        assert "function" not in result[0]

    def test_passthrough_responses_format(self):
        """Should pass through already-converted Responses format."""
        tools = [{"type": "function", "name": "test", "parameters": {}}]
        result = convert_tools_to_codex_format(tools)

        assert result == tools

    def test_handles_empty_tools(self):
        """Should handle empty tools list."""
        assert convert_tools_to_codex_format([]) == []
        assert convert_tools_to_codex_format(None) == []


class TestConvertToolMessageToGeminiPart:
    """Tests for convert_tool_message_to_gemini_part."""

    def test_converts_string_content(self):
        """Should convert string content to functionResponse."""
        msg = {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "get_weather",
            "content": "Sunny, 72F",
        }
        result = convert_tool_message_to_gemini_part(msg)

        assert "functionResponse" in result
        assert result["functionResponse"]["name"] == "get_weather"
        assert result["functionResponse"]["response"] == {"result": "Sunny, 72F"}

    def test_converts_json_content(self):
        """Should parse JSON content."""
        msg = {"role": "tool", "name": "get_data", "content": '{"temperature": 72, "unit": "F"}'}
        result = convert_tool_message_to_gemini_part(msg)

        assert result["functionResponse"]["response"]["temperature"] == 72

    def test_converts_dict_content(self):
        """Should handle dict content directly."""
        msg = {"role": "tool", "name": "test", "content": {"value": 42}}
        result = convert_tool_message_to_gemini_part(msg)

        assert result["functionResponse"]["response"]["value"] == 42


class TestConvertToolCallsToGeminiParts:
    """Tests for convert_tool_calls_to_gemini_parts."""

    def test_converts_tool_calls(self):
        """Should convert OpenAI tool_calls to functionCall parts."""
        tool_calls = [
            {
                "id": "call_123",
                "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
            }
        ]
        result = convert_tool_calls_to_gemini_parts(tool_calls)

        assert len(result) == 1
        assert "functionCall" in result[0]
        assert result[0]["functionCall"]["name"] == "get_weather"
        assert result[0]["functionCall"]["args"] == {"location": "NYC"}


class TestToolIdManagement:
    """Tests for tool ID management functions."""

    def test_assign_tool_call_ids(self):
        """Should assign IDs to functionCall parts."""
        contents = [{"role": "model", "parts": [{"functionCall": {"name": "test", "args": {}}}]}]
        result, pending = assign_tool_call_ids(contents)

        call = result[0]["parts"][0]["functionCall"]
        assert "id" in call
        assert call["id"].startswith("tool-call-")
        assert call["id"] in pending["test"]

    def test_match_tool_response_ids(self):
        """Should match functionResponse IDs with pending calls."""
        contents = [
            {"role": "user", "parts": [{"functionResponse": {"name": "test", "response": {}}}]}
        ]
        pending = {"test": ["call-123"]}
        result = match_tool_response_ids(contents, pending)

        resp = result[0]["parts"][0]["functionResponse"]
        assert resp["id"] == "call-123"

    def test_apply_tool_pairing_fixes(self):
        """Should apply all tool pairing fixes."""
        payload = {
            "contents": [
                {"role": "model", "parts": [{"functionCall": {"name": "test", "args": {}}}]},
                {
                    "role": "user",
                    "parts": [{"functionResponse": {"name": "test", "response": {"result": "ok"}}}],
                },
            ]
        }
        result = apply_tool_pairing_fixes(payload)

        call_id = result["contents"][0]["parts"][0]["functionCall"]["id"]
        resp_id = result["contents"][1]["parts"][0]["functionResponse"]["id"]
        assert call_id == resp_id


class TestExtractToolCallsFromGeminiResponse:
    """Tests for extract_tool_calls_from_gemini_response."""

    def test_extracts_function_calls(self):
        """Should extract functionCall parts from Gemini response."""
        data = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {"functionCall": {"name": "get_weather", "args": {"location": "Tokyo"}}}
                        ],
                    }
                }
            ]
        }
        result = extract_tool_calls_from_gemini_response(data)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"location": "Tokyo"}

    def test_returns_none_for_no_tool_calls(self):
        """Should return None when no tool calls found."""
        data = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}}]}
        result = extract_tool_calls_from_gemini_response(data)

        assert result is None


class TestExtractToolCallsFromCodexResponse:
    """Tests for extract_tool_calls_from_codex_response."""

    def test_extracts_function_calls(self):
        """Should extract function_call items from Codex response."""
        data = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc123",
                    "name": "read_file",
                    "arguments": '{"path": "test.txt"}',
                }
            ]
        }
        result = extract_tool_calls_from_codex_response(data)

        assert len(result) == 1
        assert result[0]["id"] == "call_abc123"
        assert result[0]["function"]["name"] == "read_file"

    def test_returns_none_for_text_only(self):
        """Should return None for text-only output."""
        data = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}]
        }
        result = extract_tool_calls_from_codex_response(data)

        assert result is None


class TestBuildToolCallsResponseMessage:
    """Tests for build_tool_calls_response_message."""

    def test_builds_message(self):
        """Should build assistant message with tool_calls."""
        tool_calls = [
            {"id": "call_123", "type": "function", "function": {"name": "test", "arguments": "{}"}}
        ]
        result = build_tool_calls_response_message(tool_calls)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert result["tool_calls"] == tool_calls


class TestThoughtSignatureCache:
    """Tests for thought signature caching (Gemini 3 support)."""

    def setup_method(self):
        """Clear cache before each test."""
        from koder_agent.auth.tool_utils import clear_thought_signature_cache

        clear_thought_signature_cache()

    def test_cache_and_retrieve_exact_match(self):
        """Should cache and retrieve signature with exact call_id and function_name."""
        from koder_agent.auth.tool_utils import (
            cache_thought_signature,
            get_cached_thought_signature,
        )

        cache_thought_signature("call_123", "read_file", "sig_abc")
        result = get_cached_thought_signature("call_123", "read_file")

        assert result == "sig_abc"

    def test_retrieve_with_normalized_name(self):
        """Should retrieve signature when function name has prefix."""
        from koder_agent.auth.tool_utils import (
            cache_thought_signature,
            get_cached_thought_signature,
        )

        # Cache with bare name
        cache_thought_signature("call_456", "grep_search", "sig_def")

        # Retrieve with prefix should still work
        result = get_cached_thought_signature("call_456", "default_api:grep_search")
        assert result == "sig_def"

    def test_retrieve_strips_various_prefixes(self):
        """Should handle multiple prefix formats."""
        from koder_agent.auth.tool_utils import (
            cache_thought_signature,
            get_cached_thought_signature,
        )

        cache_thought_signature("call_789", "write_file", "sig_ghi")

        # Should work with various prefixes
        assert get_cached_thought_signature("call_789", "tools:write_file") == "sig_ghi"
        assert get_cached_thought_signature("call_789", "functions:write_file") == "sig_ghi"

    def test_no_per_function_fallback_for_different_call_id(self):
        """Should NOT use fallback for different call_id (Gemini 3 requires exact match)."""
        from koder_agent.auth.tool_utils import (
            cache_thought_signature,
            get_cached_thought_signature,
        )

        # Cache signature for a function
        cache_thought_signature("call_001", "read_file", "sig_parallel")

        # Different call_id - no fallback, signatures must match exact functionCall
        result = get_cached_thought_signature("call_002", "read_file")
        assert result is None  # No fallback - Gemini 3 requires exact match

    def test_no_global_fallback_for_different_function(self):
        """Should NOT use global fallback for different function (Gemini 3 requires exact match)."""
        from koder_agent.auth.tool_utils import (
            cache_thought_signature,
            get_cached_thought_signature,
        )

        # Cache signature
        cache_thought_signature("call_100", "read_file", "sig_global")

        # Different call_id AND different function - no fallback
        result = get_cached_thought_signature("call_101", "grep_search")
        assert result is None  # No fallback - Gemini 3 requires exact match

    def test_batch_window_constant_is_reasonable(self):
        """Verify batch window constant is set correctly."""
        from koder_agent.auth.tool_utils import _BATCH_WINDOW_SECONDS

        # Batch window should be short to prevent cross-request pollution
        assert _BATCH_WINDOW_SECONDS == 2

    def test_cache_ttl_constant(self):
        """Verify TTL constant is set correctly."""
        from koder_agent.auth.tool_utils import _CACHE_TTL_SECONDS

        # TTL should be 5 minutes
        assert _CACHE_TTL_SECONDS == 300

    def test_cache_stores_entries(self):
        """Should store entries in cache."""
        from koder_agent.auth.tool_utils import (
            _thought_signature_cache,
            cache_thought_signature,
        )

        cache_thought_signature("call_300", "test_func", "sig_test")

        # Verify it's in cache
        assert ("call_300", "test_func") in _thought_signature_cache

    def test_max_cache_size_constant(self):
        """Verify max cache size constant is set."""
        from koder_agent.auth.tool_utils import _MAX_CACHE_SIZE

        assert _MAX_CACHE_SIZE == 1000

    def test_normalize_function_name(self):
        """Should normalize function names by stripping prefixes."""
        from koder_agent.auth.tool_utils import _normalize_function_name

        assert _normalize_function_name("grep_search") == "grep_search"
        assert _normalize_function_name("default_api:grep_search") == "grep_search"
        assert _normalize_function_name("tools:read_file") == "read_file"
        assert _normalize_function_name("functions:write_file") == "write_file"
        assert _normalize_function_name("tool:edit_file") == "edit_file"
        # Unknown prefix should be kept
        assert _normalize_function_name("unknown:func") == "unknown:func"

    def test_clear_cache_removes_all_entries(self):
        """Should clear all cache entries."""
        from koder_agent.auth.tool_utils import (
            _last_thought_signature_by_name,
            _thought_signature_cache,
            cache_thought_signature,
            clear_thought_signature_cache,
        )

        # Add some entries
        cache_thought_signature("call_1", "func_1", "sig_1")
        cache_thought_signature("call_2", "func_2", "sig_2")

        # Clear
        clear_thought_signature_cache()

        # Verify all cleared
        assert len(_thought_signature_cache) == 0
        assert len(_last_thought_signature_by_name) == 0
