"""Google OAuth provider implementation.

Provides OAuth authentication for Gemini CLI (free with Google account)
using PKCE authorization code flow.

Note: This OAuth provider is named 'google' to avoid conflict with
the 'gemini' API key provider.
"""

import asyncio
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
    CODE_ASSIST_HEADERS,
    GEMINI_CODE_ASSIST_ENDPOINT,
    GOOGLE_AUTH_URL,
    GOOGLE_CALLBACK_PATH,
    GOOGLE_CALLBACK_PORT,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_MODELS_URL,
    GOOGLE_REDIRECT_URI,
    GOOGLE_SCOPES,
    GOOGLE_TOKEN_URL,
    GOOGLE_USERINFO_URL,
)
from koder_agent.auth.tool_utils import merge_optional_params


class GoogleOAuthProvider(OAuthProvider):
    """Google OAuth provider for Gemini API access.

    Uses Google's OAuth 2.0 with PKCE to authenticate users
    and obtain access tokens for Gemini API.
    """

    provider_id = "google"
    auth_url = GOOGLE_AUTH_URL
    token_url = GOOGLE_TOKEN_URL
    redirect_uri = GOOGLE_REDIRECT_URI
    client_id = GOOGLE_CLIENT_ID
    client_secret = GOOGLE_CLIENT_SECRET
    scopes = GOOGLE_SCOPES
    callback_port = GOOGLE_CALLBACK_PORT
    callback_path = GOOGLE_CALLBACK_PATH

    def _build_auth_params(self) -> Dict[str, str]:
        """Build Google-specific authorization parameters."""
        params = super()._build_auth_params()

        # Google-specific parameters
        params["access_type"] = "offline"  # Request refresh token
        params["prompt"] = "consent"  # Force consent to get refresh token

        return params

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process Google token response.

        Args:
            token_data: Token response from Google

        Returns:
            OAuthResult with tokens or error
        """
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in", 3600)

        if not access_token:
            return OAuthResult(success=False, error="Missing access_token in response")

        if not refresh_token:
            return OAuthResult(
                success=False,
                error="Missing refresh_token in response. Try revoking access and re-authenticating.",
            )

        # Calculate expiry timestamp
        expires_at = int(time.time() * 1000) + (expires_in * 1000)

        # Fetch user info
        email = None
        user_info = await self.get_user_info(access_token)
        if user_info:
            email = user_info.get("email")

        tokens = OAuthTokens(
            provider=self.provider_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            email=email,
        )

        return OAuthResult(success=True, tokens=tokens)

    async def get_user_info(self, access_token: str) -> Optional[Dict[str, Any]]:
        """Fetch user info from Google.

        Args:
            access_token: Valid access token

        Returns:
            User info dict with email and profile info
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{GOOGLE_USERINFO_URL}?alt=json",
                    headers=self.get_auth_headers(access_token),
                ) as response:
                    if response.ok:
                        return await response.json()
        except Exception:
            pass
        return None

    async def revoke_token(self, token: str) -> bool:
        """Revoke a Google OAuth token.

        Args:
            token: Access or refresh token to revoke

        Returns:
            True if revocation succeeded
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://oauth2.googleapis.com/revoke",
                    data={"token": token},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                ) as response:
                    return response.ok
        except Exception:
            return False

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available Gemini models.

        Note: The OAuth scope may not include model listing permission.
        Falls back to known models if API call fails.

        Args:
            access_token: Valid Google OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' and optional 'error')
        """
        models = []
        status = {"source": "api", "error": None}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    GOOGLE_MODELS_URL,
                    headers=self.get_auth_headers(access_token),
                ) as response:
                    if response.ok:
                        data = await response.json()
                        for model in data.get("models", []):
                            model_name = model.get("name", "")
                            # Extract model ID from "models/gemini-xxx" format
                            if model_name.startswith("models/"):
                                model_id = model_name[7:]  # Remove "models/" prefix
                                # Filter for generation models (exclude embedding, etc.)
                                if "generateContent" in model.get("supportedGenerationMethods", []):
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
                # Gemini 3 models (preview)
                f"{self.provider_id}/gemini-3-pro-preview",
                f"{self.provider_id}/gemini-3-flash-preview",
                # Gemini 2.5 models
                f"{self.provider_id}/gemini-2.5-pro",
                f"{self.provider_id}/gemini-2.5-flash",
                # Gemini 2.0 models
                f"{self.provider_id}/gemini-2.0-flash",
                # Legacy models
                f"{self.provider_id}/gemini-1.5-pro",
                f"{self.provider_id}/gemini-1.5-flash",
            ]

        return models, status


class GoogleOAuthLLM(CustomLLM):
    """Custom LiteLLM handler for Google/Gemini OAuth access.

    Uses Gemini Code Assist endpoint with Bearer token authentication
    instead of API key authentication.
    """

    def __init__(self):
        """Initialize Google OAuth LLM handler."""
        super().__init__()
        self.provider_id = "google"
        self._managed_project_id: Optional[str] = None
        self._stored_project_id: Optional[str] = None
        self._session_id = f"agent-{uuid.uuid4()}"
        self._last_request_message_count: int = 0
        self._last_request_model: Optional[str] = None

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token for Google."""
        tokens = get_oauth_token(self.provider_id)
        if tokens:
            return tokens.access_token
        return None

    def _require_access_token(self) -> str:
        """Return a valid OAuth token or raise a helpful error."""
        access_token = self._get_access_token()
        if not access_token:
            raise ValueError(
                "No OAuth token available for Google. "
                "Run 'koder auth login google' to authenticate."
            )
        return access_token

    async def _ensure_project_context(self, access_token: str, prefer_managed: bool = True) -> str:
        """Ensure we have a valid managed project ID.

        Calls loadCodeAssist API to get the user's managed project.
        If not onboarded, calls onboardUser to create one.

        Args:
            access_token: Valid OAuth access token
            prefer_managed: If True, prefer managed project; if False, prefer stored

        Returns:
            Managed project ID
        """
        # Return cached project ID if available
        if prefer_managed and self._managed_project_id:
            return self._managed_project_id
        if not prefer_managed and self._stored_project_id:
            return self._stored_project_id

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **CODE_ASSIST_HEADERS,
        }

        # Try to load existing project
        async with aiohttp.ClientSession() as session:
            # First, try loadCodeAssist
            load_body = {
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                }
            }

            async with session.post(
                f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                json=load_body,
                headers=headers,
            ) as response:
                if response.ok:
                    data = await response.json()
                    if data.get("cloudaicompanionProject"):
                        self._managed_project_id = data["cloudaicompanionProject"]
                        return self._managed_project_id

            # If no project, try to onboard with FREE tier
            onboard_body = {
                "tierId": "FREE",
                "metadata": {
                    "ideType": "IDE_UNSPECIFIED",
                    "platform": "PLATFORM_UNSPECIFIED",
                    "pluginType": "GEMINI",
                },
            }

            for _ in range(5):  # Retry up to 5 times
                async with session.post(
                    f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                    json=onboard_body,
                    headers=headers,
                ) as response:
                    if response.ok:
                        data = await response.json()
                        if data.get("done"):
                            project_id = (
                                data.get("response", {})
                                .get("cloudaicompanionProject", {})
                                .get("id")
                            )
                            if project_id:
                                self._managed_project_id = project_id
                                return self._managed_project_id

                # Wait before retrying
                await asyncio.sleep(3)

        raise ValueError(
            "Failed to get managed project for Google OAuth. "
            "Please ensure your Google account has access to Gemini."
        )

    def _build_request_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        project_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build Gemini Code Assist request body.

        Wraps the request in the format expected by cloudcode-pa.googleapis.com.

        Args:
            model: Model name
            messages: OpenAI-style messages
            project_id: Managed project ID from loadCodeAssist
            **kwargs: Additional generation config
        """
        from koder_agent.auth.tool_utils import (
            convert_tool_calls_to_gemini_parts,
            convert_tool_message_to_gemini_part,
            convert_tools_to_gemini_format,
        )

        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            if role == "system":
                # Extract system instruction
                if isinstance(content, str):
                    system_instruction = {"parts": [{"text": content}]}
                continue

            # Handle tool result messages
            if role == "tool":
                # Tool results go as user role with functionResponse part
                tool_part = convert_tool_message_to_gemini_part(msg)
                contents.append({"role": "user", "parts": [tool_part]})
                continue

            # Map roles: user->user, assistant->model
            gemini_role = "model" if role == "assistant" else "user"
            parts: List[Dict[str, Any]] = []

            # Convert content
            if isinstance(content, str) and content:
                parts.append({"text": content})
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # Handle image content
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, dict):
                                url = image_url.get("url", "")
                                if url.startswith("data:"):
                                    # Base64 image
                                    parts.append({"inline_data": {"data": url}})

            # Handle tool calls from assistant messages
            if tool_calls and isinstance(tool_calls, list):
                tool_call_parts = convert_tool_calls_to_gemini_parts(tool_calls)
                parts.extend(tool_call_parts)

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        # Build request payload
        request_payload: Dict[str, Any] = {"contents": contents}

        if system_instruction:
            request_payload["systemInstruction"] = system_instruction

        # Add generation config
        generation_config: Dict[str, Any] = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]

        if generation_config:
            request_payload["generationConfig"] = generation_config

        # Convert and add tools if provided
        tools = kwargs.get("tools")
        if tools:
            gemini_tools = convert_tools_to_gemini_format(tools)
            if gemini_tools:
                request_payload["tools"] = gemini_tools

        # Extract model name without provider prefix
        model_name = model
        if "/" in model:
            model_name = model.split("/")[-1]

        return {
            "project": project_id,
            "model": model_name,
            "request": request_payload,
        }

    def _parse_response(self, response_data: Dict[str, Any]) -> ModelResponse:
        """Parse Gemini response into LiteLLM ModelResponse format."""
        from koder_agent.auth.tool_utils import extract_tool_calls_from_gemini_response

        # Handle wrapped response format
        if "response" in response_data:
            response_data = response_data["response"]

        # Extract content and tool calls from candidates
        content = ""
        tool_calls: Optional[List[Dict[str, Any]]] = None

        if "candidates" in response_data:
            candidates = response_data["candidates"]
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                if "content" in candidate:
                    parts = candidate["content"].get("parts", [])
                    for part in parts:
                        if "text" in part:
                            content += part["text"]

        # Extract tool calls
        tool_calls = extract_tool_calls_from_gemini_response(
            response_data,
            session_id=self._session_id,
            message_count=self._last_request_message_count,
            model=self._last_request_model,
        )

        # Extract usage
        usage_data = response_data.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )

        # Build message based on whether we have tool calls
        if tool_calls:
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": content,
            }
            finish_reason = "stop"

        return ModelResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=response_data.get("model", "gemini"),
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
        raise NotImplementedError("Use acompletion for Google OAuth")

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using Gemini Code Assist endpoint."""
        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        # Get managed project ID
        project_id = await self._ensure_project_context(access_token)

        # Determine if streaming
        stream = merged_kwargs.get("stream", False)

        # Build request with managed project ID
        body = self._build_request_body(model, messages, project_id, **merged_kwargs)

        # Determine action
        action = "streamGenerateContent" if stream else "generateContent"
        url = f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:{action}"
        if stream:
            url += "?alt=sse"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **CODE_ASSIST_HEADERS,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"Google OAuth API error ({response.status}): {error_text}")

                if stream:
                    # Handle streaming response
                    return await self._handle_streaming_response(response, model)
                else:
                    response_data = await response.json()
                    return self._parse_response(response_data)

    async def _handle_streaming_response(
        self,
        response: aiohttp.ClientResponse,
        model: str,
    ) -> ModelResponse:
        """Handle SSE streaming response and return complete ModelResponse."""
        from koder_agent.auth.tool_utils import extract_tool_calls_from_gemini_response

        full_content = ""
        all_tool_calls: List[Dict[str, Any]] = []
        last_data: Optional[Dict[str, Any]] = None

        async for line in response.content:
            line_str = line.decode("utf-8").strip()
            if not line_str or not line_str.startswith("data:"):
                continue

            try:
                data = json.loads(line_str[5:].strip())
                last_data = data
                # Handle wrapped response
                if "response" in data:
                    data = data["response"]

                if "candidates" in data:
                    for candidate in data["candidates"]:
                        if "content" in candidate:
                            for part in candidate["content"].get("parts", []):
                                if "text" in part:
                                    full_content += part["text"]
            except json.JSONDecodeError:
                continue

        # Extract tool calls from final response
        if last_data:
            tool_calls = extract_tool_calls_from_gemini_response(last_data)
            if tool_calls:
                all_tool_calls = tool_calls

        # Build message based on whether we have tool calls
        if all_tool_calls:
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": full_content if full_content else None,
                "tool_calls": all_tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            message = {
                "role": "assistant",
                "content": full_content,
            }
            finish_reason = "stop"

        return ModelResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        )

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming (not implemented - use async)."""
        raise NotImplementedError("Use astreaming for Google OAuth")

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming using Gemini Code Assist endpoint."""
        from koder_agent.auth.tool_utils import extract_tool_calls_from_gemini_response

        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        # Get managed project ID (required for API access)
        project_id = await self._ensure_project_context(access_token)

        # Build request with streaming enabled and managed project ID
        merged_kwargs["stream"] = True
        body = self._build_request_body(model, messages, project_id, **merged_kwargs)

        url = f"{GEMINI_CODE_ASSIST_ENDPOINT}/v1internal:streamGenerateContent?alt=sse"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **CODE_ASSIST_HEADERS,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=body,
                headers=headers,
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    raise ValueError(f"Google OAuth API error ({response.status}): {error_text}")

                chunk_index = 0
                has_tool_calls = False
                last_data: Optional[Dict[str, Any]] = None

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()
                    if not line_str or not line_str.startswith("data:"):
                        continue

                    try:
                        data = json.loads(line_str[5:].strip())
                        last_data = data
                        # Handle wrapped response
                        if "response" in data:
                            data = data["response"]

                        if "candidates" in data:
                            for candidate in data["candidates"]:
                                if "content" in candidate:
                                    for part in candidate["content"].get("parts", []):
                                        if "text" in part:
                                            yield GenericStreamingChunk(
                                                text=part["text"],
                                                is_finished=False,
                                                finish_reason=None,
                                                usage=None,
                                                index=chunk_index,
                                            )
                                            chunk_index += 1
                                        elif "functionCall" in part:
                                            # Yield tool call
                                            has_tool_calls = True
                                            func_call = part["functionCall"]
                                            name = func_call.get("name", "")
                                            args = func_call.get("args", {})
                                            call_id = func_call.get(
                                                "id", f"call_{uuid.uuid4().hex[:8]}"
                                            )
                                            yield GenericStreamingChunk(
                                                text="",
                                                is_finished=False,
                                                finish_reason=None,
                                                usage=None,
                                                index=chunk_index,
                                                tool_use={
                                                    "id": call_id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": name,
                                                        "arguments": (
                                                            json.dumps(args)
                                                            if isinstance(args, dict)
                                                            else str(args)
                                                        ),
                                                    },
                                                },
                                            )
                                            chunk_index += 1
                    except json.JSONDecodeError:
                        continue

                # Check for tool calls in final data (in case not detected in stream)
                if last_data and not has_tool_calls:
                    tool_calls = extract_tool_calls_from_gemini_response(last_data)
                    if tool_calls:
                        has_tool_calls = True
                        # Yield any tool calls from final data
                        for tc in tool_calls:
                            yield GenericStreamingChunk(
                                text="",
                                is_finished=False,
                                finish_reason=None,
                                usage=None,
                                index=chunk_index,
                                tool_use={
                                    "id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                    "type": "function",
                                    "function": tc.get("function", {}),
                                },
                            )
                            chunk_index += 1

                # Final chunk with appropriate finish reason
                finish_reason = "tool_calls" if has_tool_calls else "stop"
                yield GenericStreamingChunk(
                    text="",
                    is_finished=True,
                    finish_reason=finish_reason,
                    usage=None,
                    index=chunk_index,
                )
