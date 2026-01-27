"""ChatGPT OAuth provider implementation.

Provides OAuth authentication for ChatGPT Plus/Pro subscription using
PKCE authorization code flow. Uses stateless mode (store:false)
as required by ChatGPT backend.

Note: This OAuth provider is named 'chatgpt' to avoid conflict with
the 'openai' API key provider.
"""

import base64
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import aiohttp
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse, Usage

from koder_agent.auth.base import OAuthProvider, OAuthResult, OAuthTokens
from koder_agent.auth.client_integration import get_oauth_token
from koder_agent.auth.codex_prompts import CODEX_PROMPTS
from koder_agent.auth.constants import (
    CHATGPT_CODEX_BASE,
    CHATGPT_CODEX_HEADERS,
    DEFAULT_CODEX_INSTRUCTIONS,
    JWT_CLAIM_PATH,
    OPENAI_AUTH_URL,
    OPENAI_CALLBACK_PATH,
    OPENAI_CALLBACK_PORT,
    OPENAI_CLIENT_ID,
    OPENAI_MODELS_URL,
    OPENAI_REDIRECT_URI,
    OPENAI_SCOPES,
    OPENAI_TOKEN_URL,
)
from koder_agent.auth.tool_utils import merge_optional_params

logger = logging.getLogger(__name__)


class ChatGPTOAuthProvider(OAuthProvider):
    """ChatGPT OAuth provider for ChatGPT Plus/Pro subscription access.

    Uses OpenAI's OAuth 2.0 with PKCE to authenticate users
    and obtain access tokens for ChatGPT API via subscription.

    Note: ChatGPT backend requires store:false (stateless mode)
    which means full message history must be sent in every request.
    """

    provider_id = "chatgpt"
    auth_url = OPENAI_AUTH_URL
    token_url = OPENAI_TOKEN_URL
    redirect_uri = OPENAI_REDIRECT_URI
    client_id = OPENAI_CLIENT_ID
    scopes = OPENAI_SCOPES
    callback_port = OPENAI_CALLBACK_PORT
    callback_path = OPENAI_CALLBACK_PATH

    def _build_auth_params(self) -> Dict[str, str]:
        """Build OpenAI-specific authorization parameters."""
        params = super()._build_auth_params()

        # OpenAI-specific parameters for Codex CLI compatibility
        params["id_token_add_organizations"] = "true"
        params["codex_cli_simplified_flow"] = "true"
        params["originator"] = "codex_cli_rs"

        return params

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process OpenAI token response.

        Args:
            token_data: Token response from OpenAI

        Returns:
            OAuthResult with tokens or error
        """
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            return OAuthResult(success=False, error="Missing access_token in response")

        if not refresh_token:
            return OAuthResult(success=False, error="Missing refresh_token in response")

        # Calculate expiry timestamp
        expires_at = int(time.time() * 1000) + (expires_in * 1000)

        # Extract email from JWT if present
        email = self._extract_email_from_jwt(access_token)

        tokens = OAuthTokens(
            provider=self.provider_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            email=email,
        )

        return OAuthResult(success=True, tokens=tokens)

    def _extract_email_from_jwt(self, token: str) -> Optional[str]:
        """Extract email from JWT access token.

        Args:
            token: JWT access token

        Returns:
            Email string or None if not found
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            # Decode payload (second part)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)
            return data.get("email") or data.get("sub")
        except Exception:
            return None

    def get_auth_headers(self, access_token: str) -> Dict[str, str]:
        """Get OpenAI authorization headers.

        Returns headers suitable for ChatGPT API requests.
        """
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration for OpenAI requests.

        Returns config needed for stateless ChatGPT API mode.
        """
        return {
            "store": False,  # Required for ChatGPT backend
            "stream": True,
        }

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available OpenAI models.

        Note: Falls back to known models if API call fails.

        Args:
            access_token: Valid ChatGPT OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' and optional 'error')
        """
        models = []
        status = {"source": "api", "error": None}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPENAI_MODELS_URL,
                    headers=self.get_auth_headers(access_token),
                ) as response:
                    if response.ok:
                        data = await response.json()
                        for model in data.get("data", []):
                            model_id = model.get("id", "")
                            # Filter for chat completion models
                            if model_id and any(
                                prefix in model_id for prefix in ["gpt-", "o1", "o3", "chatgpt"]
                            ):
                                models.append(f"{self.provider_id}/{model_id}")
                    else:
                        error_text = await response.text()
                        status["error"] = f"API returned {response.status}: {error_text[:200]}"
        except Exception as e:
            status["error"] = f"API request failed: {str(e)}"

        # Fallback to known models if API call fails or returns empty
        if not models:
            status["source"] = "fallback"
            models = [
                # GPT-5.2 Codex models
                f"{self.provider_id}/gpt-5.2-codex",
                # GPT-5.2 general purpose
                f"{self.provider_id}/gpt-5.2",
                # GPT-5.1 Codex Max models
                f"{self.provider_id}/gpt-5.1-codex-max",
                # GPT-5.1 Codex models
                f"{self.provider_id}/gpt-5.1-codex",
                f"{self.provider_id}/gpt-5.1-codex-mini",
                # GPT-5.1 general purpose
                f"{self.provider_id}/gpt-5.1",
                # Legacy GPT-4 models (for compatibility)
                f"{self.provider_id}/gpt-4o",
                f"{self.provider_id}/gpt-4o-mini",
            ]

        return models, status


class ChatGPTOAuthLLM(CustomLLM):
    """Custom LiteLLM handler for ChatGPT/OpenAI OAuth access.

    Uses ChatGPT Codex Backend API (chatgpt.com/backend-api) with Bearer token
    authentication and stateless mode (store=false). Uses the Codex "Responses"
    format with input array instead of messages array.
    """

    def __init__(self):
        """Initialize ChatGPT OAuth LLM handler."""
        super().__init__()
        self.provider_id = "chatgpt"
        self._account_id: Optional[str] = None

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token for ChatGPT."""
        tokens = get_oauth_token(self.provider_id)
        if tokens:
            return tokens.access_token
        return None

    def _require_access_token(self) -> str:
        """Return a valid OAuth token or raise a helpful error."""
        access_token = self._get_access_token()
        if not access_token:
            raise ValueError(
                "No OAuth token available for ChatGPT. "
                "Run 'koder auth login chatgpt' to authenticate."
            )
        return access_token

    def _extract_account_id_from_jwt(self, token: str) -> Optional[str]:
        """Extract ChatGPT account ID from JWT access token.

        The account ID is stored at ["https://api.openai.com/auth"]["chatgpt_account_id"]
        in the JWT payload.

        Args:
            token: JWT access token

        Returns:
            Account ID string or None
        """
        # Return cached account ID
        if self._account_id:
            return self._account_id

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            # Decode payload (second part)
            payload = parts[1]
            # Add padding if needed
            padding = 4 - len(payload) % 4
            if padding != 4:
                payload += "=" * padding

            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)

            # Extract account ID from JWT claim path
            auth_claim = data.get(JWT_CLAIM_PATH, {})
            account_id = auth_claim.get("chatgpt_account_id")
            if account_id:
                self._account_id = account_id
            return account_id
        except Exception:
            return None

    def _normalize_model(self, model: str) -> str:
        """Normalize model name to ChatGPT Codex-supported variants."""
        # Strip provider prefix
        model_id = model.split("/")[-1] if "/" in model else model
        model_lower = model_id.lower()

        # GPT-5.2 Codex variants (newest, supports xhigh)
        if "gpt-5.2-codex" in model_lower or "gpt 5.2 codex" in model_lower:
            return "gpt-5.2-codex"

        # GPT-5.2 general purpose (supports none/low/medium/high/xhigh)
        if "gpt-5.2" in model_lower or "gpt 5.2" in model_lower:
            return "gpt-5.2"

        # GPT-5.1 Codex Max (supports xhigh)
        if "gpt-5.1-codex-max" in model_lower or "codex-max" in model_lower:
            return "gpt-5.1-codex-max"

        # GPT-5.1 Codex Mini
        if (
            "gpt-5.1-codex-mini" in model_lower
            or "codex-mini" in model_lower
            or "codex_mini" in model_lower
        ):
            return "gpt-5.1-codex-mini"

        # Legacy Codex Mini
        if "codex-mini-latest" in model_lower or "gpt-5-codex-mini" in model_lower:
            return "gpt-5.1-codex-mini"

        # GPT-5.1 Codex (standard)
        if "gpt-5.1-codex" in model_lower or "gpt 5.1 codex" in model_lower:
            return "gpt-5.1-codex"

        # GPT-5.1 general purpose
        if "gpt-5.1" in model_lower or "gpt 5.1" in model_lower:
            return "gpt-5.1"

        # Legacy GPT-5 Codex -> map to GPT-5.1 Codex
        if "gpt-5-codex" in model_lower or "gpt 5 codex" in model_lower:
            return "gpt-5.1-codex"

        # Generic codex -> default to GPT-5.1 Codex
        if "codex" in model_lower:
            return "gpt-5.1-codex"

        # Legacy GPT-5 -> map to GPT-5.1
        if "gpt-5" in model_lower or "gpt 5" in model_lower:
            return "gpt-5.1"

        # Return as-is for unknown models
        return model_id

    def _coerce_input_item(self, item: Any) -> Optional[Dict[str, Any]]:
        """Coerce an input item to a dict, if possible."""
        if item is None:
            return None
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "model_dump"):
            try:
                return item.model_dump()
            except Exception:
                return None
        if hasattr(item, "dict"):
            try:
                return item.dict()
            except Exception:
                return None
        try:
            return dict(item)
        except Exception:
            return None

    def _sanitize_input_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter unsupported items and strip IDs for stateless Codex mode."""
        sanitized: List[Dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "item_reference":
                continue
            cleaned = dict(item)
            cleaned.pop("id", None)
            sanitized.append(cleaned)

        # Handle orphaned function_call_output items (missing matching function_call)
        call_ids = {
            it.get("call_id")
            for it in sanitized
            if it.get("type") == "function_call" and it.get("call_id")
        }
        fixed: List[Dict[str, Any]] = []
        for item in sanitized:
            if item.get("type") == "function_call_output":
                call_id = item.get("call_id")
                if call_id and call_id not in call_ids:
                    tool_name = item.get("name") or "tool"
                    output = item.get("output")
                    if output is None:
                        output_text = ""
                    elif isinstance(output, str):
                        output_text = output
                    else:
                        try:
                            output_text = json.dumps(output)
                        except Exception:
                            output_text = str(output)
                    if len(output_text) > 16000:
                        output_text = output_text[:16000] + "\n...[truncated]"
                    fixed.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        f"[Previous {tool_name} result; "
                                        f"call_id={call_id}]: {output_text}"
                                    ),
                                }
                            ],
                        }
                    )
                    continue
            fixed.append(item)

        return fixed

    def _get_model_family(self, normalized_model: str) -> str:
        """Determine Codex prompt family from normalized model name."""
        if "gpt-5.2-codex" in normalized_model or "gpt 5.2 codex" in normalized_model:
            return "gpt-5.2-codex"
        if "codex-max" in normalized_model:
            return "codex-max"
        if "codex" in normalized_model or normalized_model.startswith("codex-"):
            return "codex"
        if "gpt-5.2" in normalized_model:
            return "gpt-5.2"
        return "gpt-5.1"

    async def _get_codex_instructions(self, normalized_model: str) -> str:
        """Load bundled Codex system instructions."""
        model_family = self._get_model_family(normalized_model)
        instructions = CODEX_PROMPTS.get(model_family)
        if instructions:
            return instructions

        logger.error("Missing bundled instructions for %s; using default.", model_family)
        return DEFAULT_CODEX_INSTRUCTIONS

    def _convert_content_to_input(self, content: Any, role: str = "user") -> List[Dict[str, Any]]:
        """Convert message content into Codex input content array.

        Args:
            content: Message content (string, list, or other)
            role: Message role - determines content type (input_text vs output_text)
        """
        # Determine content type based on role
        # Assistant messages use output_text, all others use input_text
        content_type = "output_text" if role == "assistant" else "input_text"

        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": content_type, "text": content}]
        if isinstance(content, list):
            content_array = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        content_array.append({"type": content_type, "text": item.get("text", "")})
                    elif item_type in ("input_text", "output_text"):
                        # Preserve existing type if it's already correct format
                        content_array.append(item)
                    else:
                        content_array.append(item)
                else:
                    content_array.append({"type": content_type, "text": str(item)})
            return content_array
        return [{"type": content_type, "text": str(content)}]

    def _convert_messages_to_input(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to Codex input format.

        Codex API uses 'input' array with different structure:
        - type: "message"
        - role: "user" | "assistant" | "developer" (instead of "system")
        - content: array of content objects

        Args:
            messages: OpenAI-style messages

        Returns:
            Codex-style input array
        """
        input_items = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Convert role names
            if role == "system":
                role = "developer"

            # Tool output messages -> function_call_output items
            if role == "tool":
                call_id = msg.get("tool_call_id") or msg.get("call_id") or msg.get("id")
                output = content
                if output is None:
                    output_text = ""
                elif isinstance(output, str):
                    output_text = output
                else:
                    try:
                        output_text = json.dumps(output)
                    except Exception:
                        output_text = str(output)
                item: Dict[str, Any] = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
                tool_name = msg.get("name") or msg.get("tool_name")
                if tool_name:
                    item["name"] = tool_name
                input_items.append(item)
                continue

            content_array = self._convert_content_to_input(content, role)
            if content_array:
                input_items.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": content_array,
                    }
                )

            # Convert tool calls to function_call items
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function") or {}
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id") or call.get("call_id") or msg.get("id"),
                            "name": fn.get("name") or call.get("name"),
                            "arguments": fn.get("arguments") or call.get("arguments"),
                        }
                    )

            # Legacy function_call field
            function_call = msg.get("function_call")
            if isinstance(function_call, dict):
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": msg.get("id") or msg.get("call_id"),
                        "name": function_call.get("name"),
                        "arguments": function_call.get("arguments"),
                    }
                )

        return self._sanitize_input_items(input_items)

    async def _build_request_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build ChatGPT Codex API request body."""
        model_name = self._normalize_model(model)

        # Use Responses-style input if provided; otherwise convert messages
        raw_input = kwargs.get("input")
        if raw_input is not None:
            input_items: List[Dict[str, Any]] = []
            if isinstance(raw_input, list):
                for item in raw_input:
                    coerced = self._coerce_input_item(item)
                    if coerced is not None:
                        input_items.append(coerced)
            else:
                coerced = self._coerce_input_item(raw_input)
                if coerced is not None:
                    input_items.append(coerced)
        else:
            input_items = self._convert_messages_to_input(messages)

        # Filter unsupported items and strip IDs
        input_items = self._sanitize_input_items(input_items)

        body: Dict[str, Any] = {
            "model": model_name,
            "input": input_items,
            "store": False,  # Required for ChatGPT backend (stateless mode)
            "stream": True,  # Always stream for Codex API
        }

        # Use Codex CLI instructions (required by ChatGPT backend)
        body["instructions"] = await self._get_codex_instructions(model_name)

        # Add include for encrypted reasoning content (required for stateless mode)
        include = kwargs.get("include")
        if include is None:
            include = ["reasoning.encrypted_content"]
        else:
            include = list(include)
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
        body["include"] = include

        # Add text verbosity (default: medium, matches Codex CLI)
        text = kwargs.get("text")
        if text is None:
            text = {}
        elif hasattr(text, "model_dump"):
            try:
                text = text.model_dump()
            except Exception:
                text = {}
        elif not isinstance(text, dict):
            text = {}
        verbosity = kwargs.get("verbosity") or text.get("verbosity") or "medium"
        text["verbosity"] = verbosity
        body["text"] = text

        # Add reasoning config based on model family
        reasoning_payload = kwargs.get("reasoning")
        reasoning: Dict[str, Any] = {}
        if reasoning_payload is not None:
            if isinstance(reasoning_payload, dict):
                reasoning = dict(reasoning_payload)
            elif hasattr(reasoning_payload, "model_dump"):
                try:
                    reasoning = reasoning_payload.model_dump()
                except Exception:
                    reasoning = {}
            elif hasattr(reasoning_payload, "dict"):
                try:
                    reasoning = reasoning_payload.dict()
                except Exception:
                    reasoning = {}

        reasoning_effort = reasoning.get("effort") or kwargs.get("reasoning_effort", None)

        normalized_name = model_name.lower()
        is_gpt52_codex = "gpt-5.2-codex" in normalized_name or "gpt 5.2 codex" in normalized_name
        is_gpt52_general = (
            "gpt-5.2" in normalized_name or "gpt 5.2" in normalized_name
        ) and not is_gpt52_codex
        is_codex_max = "codex-max" in normalized_name or "codex max" in normalized_name
        is_codex_mini = (
            "codex-mini" in normalized_name
            or "codex mini" in normalized_name
            or "codex_mini" in normalized_name
            or "codex-mini-latest" in normalized_name
        )
        is_codex = "codex" in normalized_name and not is_codex_mini
        is_lightweight = not is_codex_mini and (
            "nano" in normalized_name or "mini" in normalized_name
        )
        is_gpt51_general = (
            ("gpt-5.1" in normalized_name or "gpt 5.1" in normalized_name)
            and not is_codex
            and not is_codex_max
            and not is_codex_mini
        )

        supports_xhigh = is_gpt52_general or is_gpt52_codex or is_codex_max
        supports_none = is_gpt52_general or is_gpt51_general

        if reasoning_effort is None:
            if is_codex_mini:
                reasoning_effort = "medium"
            elif supports_xhigh:
                reasoning_effort = "high"
            elif is_lightweight:
                reasoning_effort = "minimal"
            else:
                reasoning_effort = "medium"

        if is_codex_mini:
            if reasoning_effort in ["minimal", "low", "none"]:
                reasoning_effort = "medium"
            elif reasoning_effort == "xhigh":
                reasoning_effort = "high"
            elif reasoning_effort not in ["medium", "high"]:
                reasoning_effort = "medium"

        if not supports_xhigh and reasoning_effort == "xhigh":
            reasoning_effort = "high"

        if not supports_none and reasoning_effort == "none":
            reasoning_effort = "low"

        if is_codex and reasoning_effort == "minimal":
            reasoning_effort = "low"

        reasoning["effort"] = reasoning_effort
        reasoning.setdefault("summary", "auto")
        body["reasoning"] = reasoning

        # Convert and add tools if present (Chat Completions → Responses format)
        tools = kwargs.get("tools")
        if tools:
            from koder_agent.auth.tool_utils import convert_tools_to_codex_format

            codex_tools = convert_tools_to_codex_format(tools)
            if codex_tools:
                body["tools"] = codex_tools

        # Pass through supported Responses-style fields when present (excluding tools)
        passthrough_fields = [
            "tool_choice",
            "parallel_tool_calls",
            "metadata",
            "truncation",
            "previous_response_id",
            "prompt",
            "temperature",
            "top_p",
            "prompt_cache_key",
        ]
        for field in passthrough_fields:
            value = kwargs.get(field)
            if value is not None:
                body[field] = value

        # Remove unsupported parameters
        body.pop("max_output_tokens", None)
        body.pop("max_completion_tokens", None)

        return body

    async def _iter_sse_lines(self, content: aiohttp.StreamReader) -> AsyncIterator[str]:
        """Yield SSE lines from an aiohttp stream, handling chunk boundaries."""
        buffer = ""
        async for chunk in content:
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                yield line.rstrip("\r")
        if buffer:
            yield buffer.rstrip("\r")

    def _parse_codex_response(
        self,
        response_lines: List[str],
        model: str,
    ) -> ModelResponse:
        """Parse Codex SSE response into LiteLLM ModelResponse format.

        Codex API always returns SSE format, even for non-streaming requests.
        We need to aggregate all chunks and extract the final content and tool calls.
        """
        from koder_agent.auth.tool_utils import extract_tool_calls_from_codex_response

        full_content = ""
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        usage_data = {}
        tool_calls: List[Dict[str, Any]] = []
        final_response_obj: Optional[Dict[str, Any]] = None

        for line in response_lines:
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)

                # Extract response ID
                if "id" in data:
                    response_id = data["id"]

                # Extract content from output_text events
                event_type = data.get("type", "")
                if event_type == "response.output_text.delta":
                    delta = data.get("delta", "")
                    if delta:
                        full_content += delta
                elif event_type in ("response.completed", "response.done"):
                    # Extract usage and final response object
                    response_obj = data.get("response", {})
                    final_response_obj = response_obj
                    usage_data = response_obj.get("usage", {})
                    # Extract output content and tool calls
                    output = response_obj.get("output", [])
                    for item in output:
                        if item.get("type") == "message":
                            for content_block in item.get("content", []):
                                if content_block.get("type") == "output_text":
                                    full_content = content_block.get("text", full_content)
            except json.JSONDecodeError:
                continue

        # Extract tool calls from final response
        if final_response_obj:
            extracted_calls = extract_tool_calls_from_codex_response(final_response_obj)
            if extracted_calls:
                tool_calls = extracted_calls

        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=(usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)),
        )

        # Build message based on whether we have tool calls
        if tool_calls:
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": full_content if full_content else None,
                "tool_calls": tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": full_content,
            }
            finish_reason = "stop"

        return ModelResponse(
            id=response_id,
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            usage=usage,
        )

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Synchronous completion (not implemented - use async)."""
        raise NotImplementedError("Use acompletion for ChatGPT OAuth")

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using ChatGPT Codex Backend API with OAuth."""
        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        # Extract account ID from JWT
        account_id = self._extract_account_id_from_jwt(access_token)
        if not account_id:
            raise ValueError(
                "Failed to extract ChatGPT account ID from token. "
                "Please re-authenticate with 'koder auth login chatgpt'."
            )

        body = await self._build_request_body(model, messages, **merged_kwargs)

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "chatgpt-account-id": account_id,
            **CHATGPT_CODEX_HEADERS,
        }
        prompt_cache_key = body.get("prompt_cache_key") or merged_kwargs.get("prompt_cache_key")
        if prompt_cache_key:
            headers["conversation_id"] = prompt_cache_key
            headers["session_id"] = prompt_cache_key

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{CHATGPT_CODEX_BASE}/codex/responses",
                json=body,
                headers=headers,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"ChatGPT OAuth API error ({response.status}): {error_text}")

                # Collect all SSE lines
                response_lines = []
                async for line in self._iter_sse_lines(response.content):
                    line_str = line.strip()
                    if line_str:
                        response_lines.append(line_str)

                return self._parse_codex_response(response_lines, model)

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming (not implemented - use async)."""
        raise NotImplementedError("Use astreaming for ChatGPT OAuth")

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming using ChatGPT Codex Backend API with OAuth."""
        from koder_agent.auth.tool_utils import extract_tool_calls_from_codex_response

        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        # Extract account ID from JWT
        account_id = self._extract_account_id_from_jwt(access_token)
        if not account_id:
            raise ValueError(
                "Failed to extract ChatGPT account ID from token. "
                "Please re-authenticate with 'koder auth login chatgpt'."
            )

        body = await self._build_request_body(model, messages, **merged_kwargs)

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "chatgpt-account-id": account_id,
            **CHATGPT_CODEX_HEADERS,
        }
        prompt_cache_key = body.get("prompt_cache_key") or merged_kwargs.get("prompt_cache_key")
        if prompt_cache_key:
            headers["conversation_id"] = prompt_cache_key
            headers["session_id"] = prompt_cache_key

        url = f"{CHATGPT_CODEX_BASE}/codex/responses"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"ChatGPT OAuth API error ({response.status}): {error_text}")

                chunk_index = 0
                has_tool_calls = False
                final_response_obj: Optional[Dict[str, Any]] = None

                async for line in self._iter_sse_lines(response.content):
                    line_str = line.strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue

                    data_str = line_str[5:].strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type", "")

                        # Handle output_text delta events
                        if event_type == "response.output_text.delta":
                            delta = data.get("delta", "")
                            if delta:
                                yield GenericStreamingChunk(
                                    text=delta,
                                    is_finished=False,
                                    finish_reason=None,
                                    usage=None,
                                    index=chunk_index,
                                )
                                chunk_index += 1
                        # Handle function call events
                        elif event_type == "response.function_call_arguments.delta":
                            # Tool call argument streaming - track but don't yield text yet
                            has_tool_calls = True
                        elif event_type in ("response.completed", "response.done"):
                            final_response_obj = data.get("response", {})
                            # Extract and yield tool calls from final response
                            if final_response_obj:
                                extracted = extract_tool_calls_from_codex_response(
                                    final_response_obj
                                )
                                if extracted:
                                    has_tool_calls = True
                                    # Yield each tool call as a separate chunk
                                    for tool_call in extracted:
                                        yield GenericStreamingChunk(
                                            text="",
                                            is_finished=False,
                                            finish_reason=None,
                                            usage=None,
                                            index=chunk_index,
                                            tool_use={
                                                "id": tool_call.get(
                                                    "id", f"call_{uuid.uuid4().hex[:8]}"
                                                ),
                                                "type": "function",
                                                "function": tool_call.get("function", {}),
                                            },
                                        )
                                        chunk_index += 1
                            break
                    except json.JSONDecodeError:
                        continue

                # Final chunk with appropriate finish reason
                finish_reason = "tool_calls" if has_tool_calls else "stop"
                yield GenericStreamingChunk(
                    text="",
                    is_finished=True,
                    finish_reason=finish_reason,
                    usage=None,
                    index=chunk_index,
                )
