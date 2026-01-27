"""Unit tests for OAuth provider tool handling utilities.

Tests for:
- JSON schema cleaning for Antigravity/Claude
- Tool definition conversion (OpenAI → Gemini/Claude format)
- Tool ID management (assignment, matching, orphan recovery)
- Tool call extraction from provider responses
"""

import json

from koder_agent.auth.tool_utils import (
    EMPTY_SCHEMA_PLACEHOLDER_NAME,
    apply_tool_pairing_fixes,
    # Tool ID management
    assign_tool_call_ids,
    build_tool_calls_response_message,
    # Schema cleaning
    clean_json_schema,
    convert_tool_calls_to_gemini_parts,
    convert_tool_message_to_gemini_part,
    convert_tools_to_claude_format,
    convert_tools_to_gemini_format,
    ensure_tool_has_properties,
    extract_tool_calls_from_codex_response,
    # Tool call extraction
    extract_tool_calls_from_gemini_response,
    match_tool_response_ids,
)

# =============================================================================
# JSON SCHEMA CLEANING TESTS
# =============================================================================


class TestCleanJsonSchemaForAntigravity:
    """Tests for clean_json_schema function."""

    def test_empty_schema(self):
        """Empty/invalid schemas should return basic object schema."""
        result = clean_json_schema(None)
        assert result["type"] == "object"
        assert "properties" in result

        result = clean_json_schema({})
        assert result["type"] == "object"

    def test_removes_ref(self):
        """$ref should be converted to description hint."""
        schema = {"type": "object", "properties": {"user": {"$ref": "#/$defs/User"}}}
        result = clean_json_schema(schema)
        # $ref should be removed and converted to description
        assert "$ref" not in json.dumps(result)
        assert "See: User" in json.dumps(result)

    def test_converts_const_to_enum(self):
        """const should be converted to enum."""
        schema = {"type": "object", "properties": {"action": {"const": "delete"}}}
        result = clean_json_schema(schema)
        action_prop = result["properties"]["action"]
        assert "enum" in action_prop
        assert action_prop["enum"] == ["delete"]
        assert "const" not in action_prop

    def test_adds_enum_hints(self):
        """enum values should be added to description."""
        schema = {
            "type": "object",
            "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
        }
        result = clean_json_schema(schema)
        desc = result["properties"]["color"].get("description", "")
        assert "Allowed: red, green, blue" in desc

    def test_moves_constraints_to_description(self):
        """Unsupported constraints should be moved to description."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string", "minLength": 1, "maxLength": 100}},
        }
        result = clean_json_schema(schema)
        desc = result["properties"]["name"].get("description", "")
        assert "minLength: 1" in desc
        assert "maxLength: 100" in desc
        # The keywords should be removed
        assert "minLength" not in result["properties"]["name"]
        assert "maxLength" not in result["properties"]["name"]

    def test_removes_unsupported_keywords(self):
        """Unsupported keywords should be removed."""
        schema = {
            "$schema": "https://json-schema.org/draft-07/schema#",
            "$id": "test",
            "type": "object",
            "additionalProperties": False,
            "properties": {"value": {"type": "string"}},
        }
        result = clean_json_schema(schema)
        assert "$schema" not in result
        assert "$id" not in result
        assert "additionalProperties" not in result

    def test_merges_allof(self):
        """allOf schemas should be merged."""
        schema = {
            "allOf": [
                {"properties": {"name": {"type": "string"}}},
                {"properties": {"age": {"type": "integer"}}, "required": ["age"]},
            ]
        }
        result = clean_json_schema(schema)
        assert "allOf" not in result
        assert "name" in result.get("properties", {})
        assert "age" in result.get("properties", {})
        assert "age" in result.get("required", [])


class TestEnsureToolHasProperties:
    """Tests for ensure_tool_has_properties function."""

    def test_empty_schema_gets_placeholder(self):
        """Empty schema should get placeholder property."""
        result = ensure_tool_has_properties({})
        assert EMPTY_SCHEMA_PLACEHOLDER_NAME in result["properties"]
        assert EMPTY_SCHEMA_PLACEHOLDER_NAME in result.get("required", [])

    def test_schema_with_properties_unchanged(self):
        """Schema with properties should be unchanged (except type uses uppercase for Gemini)."""
        schema = {"properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = ensure_tool_has_properties(schema)
        assert "name" in result["properties"]
        # Default is uppercase_types=True for Gemini/Antigravity compatibility
        assert result["type"] == "OBJECT"
        # Should not have placeholder
        assert EMPTY_SCHEMA_PLACEHOLDER_NAME not in result["properties"]


# =============================================================================
# TOOL FORMAT CONVERSION TESTS
# =============================================================================


class TestConvertToolsToGeminiFormat:
    """Tests for convert_tools_to_gemini_format function."""

    def test_openai_function_format(self):
        """OpenAI function format should be converted."""
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
        decls = result[0]["functionDeclarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["description"] == "Get weather for a location"

    def test_direct_declaration_format(self):
        """Direct declaration format should be converted."""
        tools = [
            {
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        result = convert_tools_to_gemini_format(tools)
        assert len(result) == 1
        decls = result[0]["functionDeclarations"]
        assert decls[0]["name"] == "search"

    def test_already_wrapped_format(self):
        """Already wrapped functionDeclarations should be preserved."""
        tools = [
            {
                "functionDeclarations": [
                    {"name": "tool1", "description": "desc1"},
                    {"name": "tool2", "description": "desc2"},
                ]
            }
        ]
        result = convert_tools_to_gemini_format(tools)
        assert len(result[0]["functionDeclarations"]) == 2

    def test_empty_tools(self):
        """Empty tools list should return empty list."""
        assert convert_tools_to_gemini_format([]) == []
        assert convert_tools_to_gemini_format(None) == []


class TestConvertToolsToClaudeFormat:
    """Tests for convert_tools_to_claude_format function."""

    def test_schema_cleaning_applied(self):
        """Schema should be cleaned for Claude."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "Test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string", "minLength": 1}},
                    },
                },
            }
        ]
        result = convert_tools_to_claude_format(tools)
        decls = result[0]["functionDeclarations"]
        params = decls[0]["parameters"]
        # minLength should be removed (moved to description)
        assert "minLength" not in params.get("properties", {}).get("value", {})

    def test_signature_injection(self):
        """Parameter signatures should be injected into description."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet someone",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    },
                },
            }
        ]
        result = convert_tools_to_claude_format(tools, inject_signatures=True)
        desc = result[0]["functionDeclarations"][0]["description"]
        assert "name: string" in desc
        assert "age: integer" in desc

    def test_no_signature_injection(self):
        """Signature injection can be disabled."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "Original description",
                    "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
                },
            }
        ]
        result = convert_tools_to_claude_format(tools, inject_signatures=False)
        desc = result[0]["functionDeclarations"][0]["description"]
        assert desc == "Original description"


class TestConvertToolMessageToGeminiPart:
    """Tests for convert_tool_message_to_gemini_part function."""

    def test_string_content(self):
        """String content should be wrapped in response."""
        msg = {"role": "tool", "name": "get_weather", "content": "Sunny, 72°F"}
        result = convert_tool_message_to_gemini_part(msg)
        assert "functionResponse" in result
        assert result["functionResponse"]["name"] == "get_weather"
        assert result["functionResponse"]["response"]["result"] == "Sunny, 72°F"

    def test_json_content(self):
        """JSON string content should be parsed."""
        msg = {"role": "tool", "name": "get_data", "content": '{"value": 42}'}
        result = convert_tool_message_to_gemini_part(msg)
        assert result["functionResponse"]["response"]["value"] == 42

    def test_dict_content(self):
        """Dict content should be preserved."""
        msg = {"role": "tool", "name": "func", "content": {"key": "value"}}
        result = convert_tool_message_to_gemini_part(msg)
        assert result["functionResponse"]["response"]["key"] == "value"


class TestConvertToolCallsToGeminiParts:
    """Tests for convert_tool_calls_to_gemini_parts function."""

    def test_openai_format(self):
        """OpenAI tool_calls should be converted to functionCall parts."""
        tool_calls = [
            {
                "id": "call_123",
                "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'},
            }
        ]
        result = convert_tool_calls_to_gemini_parts(tool_calls)
        assert len(result) == 1
        assert "functionCall" in result[0]
        assert result[0]["functionCall"]["name"] == "get_weather"
        assert result[0]["functionCall"]["args"]["location"] == "Tokyo"


# =============================================================================
# TOOL ID MANAGEMENT TESTS
# =============================================================================


class TestAssignToolCallIds:
    """Tests for assign_tool_call_ids function."""

    def test_assigns_ids_to_calls_without_ids(self):
        """functionCall parts without IDs should get IDs assigned."""
        contents = [{"role": "model", "parts": [{"functionCall": {"name": "func1", "args": {}}}]}]
        result, pending = assign_tool_call_ids(contents)
        # Should have an ID now
        call = result[0]["parts"][0]["functionCall"]
        assert "id" in call
        assert call["id"].startswith("tool-call-")
        # Should be in pending queue
        assert "func1" in pending
        assert len(pending["func1"]) == 1

    def test_preserves_existing_ids(self):
        """functionCall parts with existing IDs should be preserved."""
        contents = [
            {
                "role": "model",
                "parts": [{"functionCall": {"name": "func1", "args": {}, "id": "existing-id"}}],
            }
        ]
        result, pending = assign_tool_call_ids(contents)
        call = result[0]["parts"][0]["functionCall"]
        assert call["id"] == "existing-id"
        assert pending["func1"] == ["existing-id"]


class TestMatchToolResponseIds:
    """Tests for match_tool_response_ids function."""

    def test_matches_response_ids_fifo(self):
        """functionResponse parts should be matched with pending IDs in FIFO order."""
        contents = [
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "func1", "response": {"result": "a"}}},
                    {"functionResponse": {"name": "func1", "response": {"result": "b"}}},
                ],
            }
        ]
        pending = {"func1": ["id-1", "id-2"]}
        result = match_tool_response_ids(contents, pending)

        parts = result[0]["parts"]
        assert parts[0]["functionResponse"]["id"] == "id-1"  # First response gets first ID
        assert parts[1]["functionResponse"]["id"] == "id-2"  # Second response gets second ID


class TestApplyToolPairingFixes:
    """Tests for apply_tool_pairing_fixes function."""

    def test_full_pipeline(self):
        """Full 3-pass pipeline should work."""
        payload = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "get_weather", "args": {"location": "NYC"}}}
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {"functionResponse": {"name": "get_weather", "response": {"temp": 72}}}
                    ],
                },
            ]
        }
        result = apply_tool_pairing_fixes(payload, is_claude=True)

        # Call should have ID
        call = result["contents"][0]["parts"][0]["functionCall"]
        assert "id" in call

        # Response should have matching ID
        response = result["contents"][1]["parts"][0]["functionResponse"]
        assert response["id"] == call["id"]


# =============================================================================
# TOOL CALL EXTRACTION TESTS
# =============================================================================


class TestExtractToolCallsFromGeminiResponse:
    """Tests for extract_tool_calls_from_gemini_response function."""

    def test_extracts_function_calls(self):
        """functionCall parts should be extracted as OpenAI tool_calls."""
        data = {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_weather",
                                    "args": {"location": "London"},
                                    "id": "call_123",
                                }
                            }
                        ],
                    }
                }
            ]
        }
        result = extract_tool_calls_from_gemini_response(data)
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "call_123"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"location": "London"}

    def test_handles_wrapped_response(self):
        """Wrapped response format should be handled."""
        data = {
            "response": {
                "candidates": [
                    {"content": {"parts": [{"functionCall": {"name": "test", "args": {}}}]}}
                ]
            }
        }
        result = extract_tool_calls_from_gemini_response(data)
        assert result is not None
        assert result[0]["function"]["name"] == "test"

    def test_returns_none_for_no_tool_calls(self):
        """Should return None when no tool calls present."""
        data = {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}
        result = extract_tool_calls_from_gemini_response(data)
        assert result is None


class TestExtractToolCallsFromCodexResponse:
    """Tests for extract_tool_calls_from_codex_response function."""

    def test_extracts_function_calls(self):
        """function_call items should be extracted."""
        data = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                }
            ]
        }
        result = extract_tool_calls_from_codex_response(data)
        assert result is not None
        assert len(result) == 1
        assert result[0]["id"] == "call_abc"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["arguments"] == '{"location": "Paris"}'

    def test_handles_dict_arguments(self):
        """Dict arguments should be converted to JSON string."""
        data = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "func",
                    "arguments": {"key": "value"},
                }
            ]
        }
        result = extract_tool_calls_from_codex_response(data)
        assert result[0]["function"]["arguments"] == '{"key": "value"}'

    def test_ignores_non_function_call_items(self):
        """Non function_call items should be ignored."""
        data = {
            "output": [
                {"type": "message", "content": "Hello"},
                {"type": "function_call", "call_id": "c1", "name": "func", "arguments": "{}"},
            ]
        }
        result = extract_tool_calls_from_codex_response(data)
        assert len(result) == 1

    def test_returns_none_for_no_function_calls(self):
        """Should return None when no function_call items."""
        data = {"output": [{"type": "message", "content": "Hello"}]}
        result = extract_tool_calls_from_codex_response(data)
        assert result is None


class TestBuildToolCallsResponseMessage:
    """Tests for build_tool_calls_response_message function."""

    def test_builds_assistant_message(self):
        """Should build assistant message with tool_calls."""
        tool_calls = [
            {"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
        ]
        result = build_tool_calls_response_message(tool_calls)
        assert result["role"] == "assistant"
        assert result["content"] is None
        assert result["tool_calls"] == tool_calls
