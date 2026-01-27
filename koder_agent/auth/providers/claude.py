"""Claude OAuth provider implementation.

Provides OAuth authentication for Claude Max subscription using
PKCE authorization code flow. Supports both Claude Pro/Max
subscription access and API key creation.

Note: This OAuth provider is named 'claude' to avoid conflict with
the 'anthropic' API key provider.
"""

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import aiohttp
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import GenericStreamingChunk, ModelResponse, Usage

from koder_agent.auth.base import OAuthProvider, OAuthResult, OAuthTokens
from koder_agent.auth.client_integration import get_oauth_token
from koder_agent.auth.constants import (
    ANTHROPIC_API_BASE,
    ANTHROPIC_AUTH_URL_CONSOLE,
    ANTHROPIC_AUTH_URL_MAX,
    ANTHROPIC_BETA_HEADERS,
    ANTHROPIC_CLIENT_ID,
    ANTHROPIC_CREATE_API_KEY_URL,
    ANTHROPIC_REDIRECT_URI,
    ANTHROPIC_SCOPES,
    ANTHROPIC_TOKEN_URL,
    CLAUDE_CODE_SYSTEM_PREFIX,
)
from koder_agent.auth.tool_utils import merge_optional_params


class ClaudeOAuthProvider(OAuthProvider):
    """Claude OAuth provider for Claude Max subscription access.

    Uses Anthropic's OAuth 2.0 with PKCE to authenticate users
    and obtain access tokens for Claude API via subscription.

    Supports two modes:
    - "max": Claude Pro/Max subscription (uses claude.ai OAuth)
    - "console": Create API key (uses console.anthropic.com OAuth)
    """

    provider_id = "claude"
    token_url = ANTHROPIC_TOKEN_URL
    redirect_uri = ANTHROPIC_REDIRECT_URI
    client_id = ANTHROPIC_CLIENT_ID
    scopes = ANTHROPIC_SCOPES

    def __init__(self, mode: str = "max"):
        """Initialize Claude OAuth provider.

        Args:
            mode: OAuth mode - "max" for Claude Pro/Max, "console" for API key
        """
        super().__init__()
        self.mode = mode
        self.auth_url = ANTHROPIC_AUTH_URL_MAX if mode == "max" else ANTHROPIC_AUTH_URL_CONSOLE

    def _build_auth_params(self) -> Dict[str, str]:
        """Build Anthropic-specific authorization parameters."""
        params = {
            "code": "true",  # Anthropic-specific parameter
            "client_id": self.client_id,
            "response_type": "code",
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
            "code_challenge": self._pkce.challenge,
            "code_challenge_method": "S256",
            "state": self._pkce.verifier,  # Anthropic uses verifier as state
        }
        return params

    def _build_token_request(self, code: str, verifier: str) -> Dict[str, str]:
        """Build Anthropic token exchange request.

        Anthropic expects JSON body instead of form-encoded.
        """
        # Handle code format: code#state or just code
        if "#" in code:
            auth_code, state = code.split("#", 1)
        else:
            auth_code = code
            state = verifier

        return {
            "code": auth_code,
            "state": state,
            "grant_type": "authorization_code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "code_verifier": verifier,
        }

    async def exchange_code(self, code: str, verifier: str) -> OAuthResult:
        """Exchange authorization code for tokens.

        Anthropic uses JSON request body instead of form-encoded.
        """
        try:
            data = self._build_token_request(code, verifier)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    json=data,  # JSON body for Anthropic
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        return OAuthResult(
                            success=False,
                            error=f"Token exchange failed ({response.status}): {error_text}",
                        )

                    token_data = await response.json()
                    return await self._process_token_response(token_data)

        except Exception as e:
            return OAuthResult(success=False, error=str(e))

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process Anthropic token response.

        Args:
            token_data: Token response from Anthropic

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

        tokens = OAuthTokens(
            provider=self.provider_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            extra={"mode": self.mode},
        )

        return OAuthResult(success=True, tokens=tokens)

    async def refresh_tokens(self, refresh_token: str) -> OAuthResult:
        """Refresh Anthropic access token.

        Anthropic uses JSON request body for token refresh.
        """
        try:
            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.token_url,
                    json=data,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        return OAuthResult(
                            success=False,
                            error=f"Token refresh failed ({response.status}): {error_text}",
                        )

                    token_data = await response.json()

                    # Use existing refresh token if not returned
                    if "refresh_token" not in token_data:
                        token_data["refresh_token"] = refresh_token

                    return await self._process_token_response(token_data)

        except Exception as e:
            return OAuthResult(success=False, error=str(e))

    async def create_api_key(self, access_token: str) -> Optional[str]:
        """Create an API key using OAuth access token.

        Only works in "console" mode.

        Args:
            access_token: Valid OAuth access token

        Returns:
            API key string or None if failed
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    ANTHROPIC_CREATE_API_KEY_URL,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {access_token}",
                    },
                ) as response:
                    if response.ok:
                        data = await response.json()
                        return data.get("raw_key")
        except Exception:
            pass
        return None

    def get_auth_headers(self, access_token: str) -> Dict[str, str]:
        """Get Anthropic authorization headers.

        Includes OAuth beta headers required for API access.
        """
        return {
            "Authorization": f"Bearer {access_token}",
            "anthropic-beta": ",".join(ANTHROPIC_BETA_HEADERS),
        }

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available Claude models.

        Note: Anthropic doesn't provide a public models listing API.
        Returns commonly available models for Claude Max subscription.

        Args:
            access_token: Valid Claude OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' info)
        """
        # Anthropic doesn't have a models listing API
        # Return commonly available models for Claude Max subscription
        # Model names must match actual Anthropic API model identifiers
        status = {
            "source": "hardcoded",
            "error": None,
            "reason": "Anthropic has no public models API",
        }
        models = [
            # Claude 4.5 models (latest versions)
            f"{self.provider_id}/claude-sonnet-4-5-20250929",
            f"{self.provider_id}/claude-opus-4-5-20251101",
            # Aliases without date suffix
            f"{self.provider_id}/claude-sonnet-4-5",
            f"{self.provider_id}/claude-opus-4-5",
            # Claude 4.1 models
            f"{self.provider_id}/claude-opus-4-1-20250805",
            f"{self.provider_id}/claude-opus-4-1",
            # Claude 4 models
            f"{self.provider_id}/claude-sonnet-4-20250514",
            f"{self.provider_id}/claude-opus-4-20250514",
            # Legacy Claude 3.5/3 models
            f"{self.provider_id}/claude-3-5-sonnet-20241022",
            f"{self.provider_id}/claude-3-opus-20240229",
        ]
        return models, status


class ClaudeOAuthLLM(CustomLLM):
    """Custom LiteLLM handler for Claude/Anthropic OAuth access.

    Uses Anthropic API with Bearer token authentication. Claude Pro/Max OAuth
    requires the system field as an array of content blocks with the Claude Code
    identification string as the first block.
    """

    def __init__(self):
        """Initialize Claude OAuth LLM handler."""
        super().__init__()
        self.provider_id = "claude"

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token for Claude."""
        tokens = get_oauth_token(self.provider_id)
        if tokens:
            return tokens.access_token
        return None

    def _require_access_token(self) -> str:
        """Return a valid OAuth token or raise a helpful error."""
        access_token = self._get_access_token()
        if not access_token:
            raise ValueError(
                "No OAuth token available for Claude. "
                "Run 'koder auth login claude' to authenticate."
            )
        return access_token

    async def _build_request_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build Anthropic API request body."""
        from koder_agent.auth.tool_utils import clean_json_schema, ensure_tool_has_properties

        # Extract system messages and build anthropic messages
        system_parts = []
        anthropic_messages: List[Dict[str, Any]] = []
        pending_tool_results: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_content = content if isinstance(content, str) else str(content)
                system_parts.append(system_content)
                continue

            # Handle tool result messages (role="tool")
            if role == "tool":
                tool_use_id = msg.get("tool_call_id", msg.get("tool_use_id", ""))
                result_content = msg.get("content", "")
                pending_tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": (
                            result_content
                            if isinstance(result_content, str)
                            else json.dumps(result_content)
                        ),
                    }
                )
                continue

            # Flush pending tool results as a user message before the next non-tool message
            if pending_tool_results:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": pending_tool_results,
                    }
                )
                pending_tool_results = []

            # Handle assistant messages with tool_calls
            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                content_blocks: List[Dict[str, Any]] = []
                # Add text content if present
                if isinstance(content, str) and content:
                    content_blocks.append({"type": "text", "text": content})
                # Convert tool_calls to tool_use blocks
                for call in tool_calls:
                    func = call.get("function", {})
                    name = func.get("name", call.get("name", ""))
                    args_str = func.get("arguments", call.get("arguments", "{}"))
                    call_id = call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    # Parse arguments
                    if isinstance(args_str, str):
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = args_str or {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": call_id,
                            "name": name,
                            "input": args,
                        }
                    )
                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content_blocks,
                    }
                )
            else:
                anthropic_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

        # Flush any remaining tool results
        if pending_tool_results:
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": pending_tool_results,
                }
            )

        # Extract model name without provider prefix
        model_name = model
        if "/" in model:
            model_name = model.split("/")[-1]

        # Build system prompt as array of content blocks (required for Claude OAuth)
        system_blocks: List[Dict[str, str]] = [{"type": "text", "text": CLAUDE_CODE_SYSTEM_PREFIX}]

        for part in system_parts:
            if part.strip() == CLAUDE_CODE_SYSTEM_PREFIX.strip():
                continue
            if part.startswith(CLAUDE_CODE_SYSTEM_PREFIX):
                remainder = part[len(CLAUDE_CODE_SYSTEM_PREFIX) :].strip()
                if remainder:
                    system_blocks.append({"type": "text", "text": remainder})
            else:
                system_blocks.append({"type": "text", "text": part})

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "system": system_blocks,
        }

        if "temperature" in kwargs:
            body["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            body["top_p"] = kwargs["top_p"]

        # Convert and add tools
        tools = kwargs.get("tools")
        if tools:
            anthropic_tools = []
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                # Handle OpenAI function tool format
                if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                    func = tool["function"]
                    name = func.get("name", "")
                    description = func.get("description", "")
                    params = func.get("parameters", {"type": "object", "properties": {}})
                elif "name" in tool:
                    name = tool.get("name", "")
                    description = tool.get("description", "")
                    params = (
                        tool.get("parameters")
                        or tool.get("input_schema")
                        or {"type": "object", "properties": {}}
                    )
                else:
                    continue
                # Clean schema for Claude
                cleaned_params = clean_json_schema(params)
                cleaned_params = ensure_tool_has_properties(cleaned_params)
                anthropic_tools.append(
                    {
                        "name": name,
                        "description": description,
                        "input_schema": cleaned_params,
                    }
                )
            if anthropic_tools:
                body["tools"] = anthropic_tools

        return body

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> ModelResponse:
        """Parse Anthropic response into LiteLLM ModelResponse format."""
        # Extract content and tool_use blocks
        content = ""
        tool_calls: List[Dict[str, Any]] = []
        content_blocks = response_data.get("content", [])

        for block in content_blocks:
            block_type = block.get("type", "")
            if block_type == "text":
                content += block.get("text", "")
            elif block_type == "tool_use":
                # Convert tool_use to OpenAI tool_calls format
                tool_calls.append(
                    {
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        # Extract usage
        usage_data = response_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=(usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)),
        )

        # Build message with optional tool_calls
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": content if content else None,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        # Map stop_reason to finish_reason
        stop_reason = response_data.get("stop_reason", "end_turn")
        if stop_reason == "tool_use" or tool_calls:
            finish_reason = "tool_calls"
        elif stop_reason == "end_turn":
            finish_reason = "stop"
        else:
            finish_reason = stop_reason

        return ModelResponse(
            id=response_data.get("id", f"msg_{uuid.uuid4().hex}"),
            created=int(time.time()),
            model=response_data.get("model", model),
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
        raise NotImplementedError("Use acompletion for Claude OAuth")

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using Anthropic API with OAuth."""
        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        body = await self._build_request_body(model, messages, **merged_kwargs)

        headers = {
            "authorization": f"Bearer {access_token}",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ",".join(ANTHROPIC_BETA_HEADERS),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ANTHROPIC_API_BASE}/messages",
                json=body,
                headers=headers,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"Claude OAuth API error ({response.status}): {error_text}")

                response_data = await response.json()
                return self._parse_response(response_data, model)

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming (not implemented - use async)."""
        raise NotImplementedError("Use astreaming for Claude OAuth")

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming using Anthropic API with OAuth."""
        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        body = await self._build_request_body(model, messages, **merged_kwargs)
        body["stream"] = True

        headers = {
            "authorization": f"Bearer {access_token}",
            "content-type": "application/json",
            "accept": "text/event-stream",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": ",".join(ANTHROPIC_BETA_HEADERS),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{ANTHROPIC_API_BASE}/messages",
                json=body,
                headers=headers,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"Claude OAuth API error ({response.status}): {error_text}")

                chunk_index = 0
                has_tool_calls = False
                # Track current tool_use block being streamed
                current_tool_use: Optional[Dict[str, Any]] = None
                current_tool_json = ""

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue

                    if line_str == "data: [DONE]":
                        break

                    try:
                        data = json.loads(line_str[5:].strip())
                        event_type = data.get("type", "")

                        if event_type == "content_block_start":
                            # Check if this is a tool_use block
                            content_block = data.get("content_block", {})
                            if content_block.get("type") == "tool_use":
                                has_tool_calls = True
                                current_tool_use = {
                                    "id": content_block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                    "name": content_block.get("name", ""),
                                }
                                current_tool_json = ""

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            delta_type = delta.get("type", "")

                            if delta_type == "text_delta":
                                yield GenericStreamingChunk(
                                    text=delta.get("text", ""),
                                    is_finished=False,
                                    finish_reason=None,
                                    usage=None,
                                    index=chunk_index,
                                )
                                chunk_index += 1
                            elif delta_type == "input_json_delta" and current_tool_use:
                                # Accumulate JSON for tool_use input
                                current_tool_json += delta.get("partial_json", "")

                        elif event_type == "content_block_stop":
                            # Finalize tool_use block if we were building one
                            if current_tool_use:
                                yield GenericStreamingChunk(
                                    text="",
                                    is_finished=False,
                                    finish_reason=None,
                                    usage=None,
                                    index=chunk_index,
                                    tool_use={
                                        "id": current_tool_use["id"],
                                        "type": "function",
                                        "function": {
                                            "name": current_tool_use["name"],
                                            "arguments": current_tool_json or "{}",
                                        },
                                    },
                                )
                                chunk_index += 1
                                current_tool_use = None
                                current_tool_json = ""

                        elif event_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue

                # Final chunk with appropriate finish_reason
                finish_reason = "tool_calls" if has_tool_calls else "stop"
                yield GenericStreamingChunk(
                    text="",
                    is_finished=True,
                    finish_reason=finish_reason,
                    usage=None,
                    index=chunk_index,
                )
