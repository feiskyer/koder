"""Antigravity OAuth provider implementation.

Provides OAuth authentication for Antigravity which gives access to
Gemini 3 and Claude models via Google OAuth credentials.
Uses dual quota system: Antigravity quota + Gemini CLI quota.
"""

import asyncio
import datetime
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
from koder_agent.auth.constants import (
    ANTIGRAVITY_API_BASE,
    ANTIGRAVITY_AUTH_URL,
    ANTIGRAVITY_CALLBACK_PATH,
    ANTIGRAVITY_CALLBACK_PORT,
    ANTIGRAVITY_CLIENT_ID,
    ANTIGRAVITY_CLIENT_SECRET,
    ANTIGRAVITY_DEFAULT_PROJECT_ID,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_FALLBACKS,
    ANTIGRAVITY_ENDPOINT_PROD,
    ANTIGRAVITY_HEADERS,
    ANTIGRAVITY_REDIRECT_URI,
    ANTIGRAVITY_SCOPES,
    ANTIGRAVITY_SYSTEM_INSTRUCTION,
    ANTIGRAVITY_TOKEN_URL,
    CLAUDE_ENDPOINT_FALLBACKS,
    CLAUDE_INTERLEAVED_THINKING_HINT,
    CLAUDE_THINKING_BUDGETS,
    CLAUDE_THINKING_MAX_OUTPUT_TOKENS,
    DEFAULT_RATE_LIMIT_DELAY_SECONDS,
    GEMINI_CLI_HEADERS,
    GOOGLE_USERINFO_URL,
    MAX_RATE_LIMIT_RETRIES,
    MAX_SIGNATURE_RETRIES,
    REPAIR_PROMPT,
    SIGNATURE_ERROR_PATTERNS,
)
from koder_agent.auth.tool_utils import merge_optional_params

logger = logging.getLogger(__name__)

# Antigravity internal API endpoints
CLOUD_CODE_BASE_URL = "https://cloudcode-pa.googleapis.com"
LOAD_CODE_ASSIST_URL = f"{CLOUD_CODE_BASE_URL}/v1internal:loadCodeAssist"
FETCH_MODELS_URL = f"{CLOUD_CODE_BASE_URL}/v1internal:fetchAvailableModels"
ANTIGRAVITY_USER_AGENT = "antigravity/1.11.3 Darwin/arm64"
DEFAULT_PROJECT_ID = "bamboo-precept-lgxtn"


class AntigravityOAuthProvider(OAuthProvider):
    """Antigravity OAuth provider for Gemini 3 + Claude access.

    Uses Google OAuth to authenticate and access models through
    Antigravity's quota system. Supports multi-account rotation
    for higher combined quotas.

    Available models via Antigravity:
    - gemini-3-flash, gemini-3-pro-low, gemini-3-pro-high
    - claude-sonnet-4-5, claude-opus-4-5 (with thinking variants)
    """

    provider_id = "antigravity"
    auth_url = ANTIGRAVITY_AUTH_URL
    token_url = ANTIGRAVITY_TOKEN_URL
    redirect_uri = ANTIGRAVITY_REDIRECT_URI
    client_id = ANTIGRAVITY_CLIENT_ID
    client_secret = ANTIGRAVITY_CLIENT_SECRET
    scopes = ANTIGRAVITY_SCOPES
    callback_port = ANTIGRAVITY_CALLBACK_PORT
    callback_path = ANTIGRAVITY_CALLBACK_PATH

    def __init__(self):
        """Initialize Antigravity OAuth provider."""
        super().__init__()
        self._accounts: List[OAuthTokens] = []

    def _build_auth_params(self) -> Dict[str, str]:
        """Build Antigravity-specific authorization parameters."""
        params = super()._build_auth_params()

        # Request refresh token and force consent
        params["access_type"] = "offline"
        params["prompt"] = "consent"

        return params

    async def _process_token_response(self, token_data: Dict[str, Any]) -> OAuthResult:
        """Process Antigravity token response.

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
                error="Missing refresh_token. Try revoking access and re-authenticating.",
            )

        # Calculate expiry timestamp
        expires_at = int(time.time() * 1000) + (expires_in * 1000)

        # Fetch user info
        email = None
        user_info = await self.get_user_info(access_token)
        if user_info:
            email = user_info.get("email")

        # Fetch project ID for reuse (best-effort)
        project_id, _tier_id = await self._fetch_project_id(access_token)

        tokens = OAuthTokens(
            provider=self.provider_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            email=email,
            extra={
                "quota_type": "antigravity",  # Primary quota
                "fallback_quota": "gemini_cli",  # Fallback quota
                "project_id": project_id,
            },
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
        """Revoke an Antigravity OAuth token.

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

    def get_model_mapping(self) -> Dict[str, str]:
        """Get Antigravity model name to API model mapping.

        Returns:
            Dict mapping user model names to API model names
        """
        return {
            # Gemini 3 models (Antigravity quota)
            "antigravity-gemini-3-flash": "gemini-3-flash",
            "antigravity-gemini-3-pro-low": "gemini-3-pro-low",
            "antigravity-gemini-3-pro-high": "gemini-3-pro-high",
            # Claude models (Antigravity quota)
            "antigravity-claude-sonnet-4-5": "claude-sonnet-4-5",
            "antigravity-claude-sonnet-4-5-thinking-low": "claude-sonnet-4-5-thinking-low",
            "antigravity-claude-sonnet-4-5-thinking-medium": "claude-sonnet-4-5-thinking-medium",
            "antigravity-claude-sonnet-4-5-thinking-high": "claude-sonnet-4-5-thinking-high",
            "antigravity-claude-opus-4-5-thinking-low": "claude-opus-4-5-thinking-low",
            "antigravity-claude-opus-4-5-thinking-medium": "claude-opus-4-5-thinking-medium",
            "antigravity-claude-opus-4-5-thinking-high": "claude-opus-4-5-thinking-high",
            # Gemini CLI quota fallback models
            "gemini-2.5-flash": "gemini-2.5-flash",
            "gemini-2.5-pro": "gemini-2.5-pro",
            "gemini-3-flash-preview": "gemini-3-flash-preview",
            "gemini-3-pro-preview": "gemini-3-pro-preview",
        }

    def get_api_base_url(self, model: str) -> str:
        """Get API base URL for a model.

        Args:
            model: Model name

        Returns:
            API base URL
        """
        return ANTIGRAVITY_API_BASE

    async def _fetch_project_id(self, access_token: str) -> tuple[Optional[str], Optional[str]]:
        """Fetch project ID and subscription tier from loadCodeAssist API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    LOAD_CODE_ASSIST_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "User-Agent": "google-api-nodejs-client/9.15.1",
                        "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
                        "Client-Metadata": '{"ideType":"IDE_UNSPECIFIED","platform":"PLATFORM_UNSPECIFIED","pluginType":"GEMINI"}',
                    },
                    json={
                        "metadata": {
                            "ideType": "IDE_UNSPECIFIED",
                            "platform": "PLATFORM_UNSPECIFIED",
                            "pluginType": "GEMINI",
                        }
                    },
                ) as response:
                    if response.ok:
                        data = await response.json()
                        project_id = data.get("cloudaicompanionProject")
                        paid_tier = data.get("paidTier", {})
                        current_tier = data.get("currentTier", {})
                        tier_id = paid_tier.get("id") or current_tier.get("id")
                        return project_id, tier_id
        except Exception:
            pass
        return None, None

    async def _fetch_available_models(
        self, access_token: str, project_id: str
    ) -> tuple[list[str], dict]:
        """Fetch available models from fetchAvailableModels API."""
        models = []
        status = {"source": "api", "error": None, "project_id": project_id}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    FETCH_MODELS_URL,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json",
                        "User-Agent": ANTIGRAVITY_USER_AGENT,
                    },
                    json={"project": project_id},
                ) as response:
                    if response.ok:
                        data = await response.json()
                        models_data = data.get("models", {})

                        for model_name, model_info in models_data.items():
                            # Skip internal models
                            if model_info.get("isInternal"):
                                continue
                            # Only include models with quota info
                            if model_info.get("quotaInfo"):
                                models.append(f"{self.provider_id}/{model_name}")

                        status["model_count"] = len(models_data)
                        status["filtered_count"] = len(models)
                    else:
                        error_text = await response.text()
                        status["error"] = f"API returned {response.status}: {error_text[:200]}"
                        status["source"] = "fallback"
        except Exception as e:
            status["error"] = f"API request failed: {str(e)}"
            status["source"] = "fallback"

        return models, status

    async def list_models(self, access_token: str, verbose: bool = False) -> tuple[list[str], dict]:
        """List available Antigravity models from live API.

        Args:
            access_token: Valid Antigravity OAuth access token
            verbose: If True, return detailed status info

        Returns:
            Tuple of (model list, status dict with 'source' info)
        """
        # Fetch project ID first
        project_id, tier_id = await self._fetch_project_id(access_token)
        project_id = project_id or DEFAULT_PROJECT_ID

        # Fetch models from API
        models, status = await self._fetch_available_models(access_token, project_id)

        if tier_id:
            status["subscription_tier"] = tier_id

        # Fallback to hardcoded list if API fails
        if not models:
            status["source"] = "fallback"
            if not status.get("error"):
                status["error"] = "API returned no models"
            models = [
                f"{self.provider_id}/gemini-3-flash",
                f"{self.provider_id}/gemini-3-pro-low",
                f"{self.provider_id}/gemini-3-pro-high",
                f"{self.provider_id}/claude-sonnet-4-5",
                f"{self.provider_id}/claude-sonnet-4-5-thinking",
                f"{self.provider_id}/claude-opus-4-5-thinking",
                f"{self.provider_id}/gemini-2.5-flash",
                f"{self.provider_id}/gemini-2.5-pro",
            ]

        return models, status


class AntigravityOAuthLLM(CustomLLM):
    """Custom LiteLLM handler for Antigravity OAuth access.

    Uses Antigravity (Gemini Code Assist) endpoint to access both
    Gemini 3 and Claude models through Google OAuth.

    Supports endpoint fallback: daily → autopush → prod
    Uses managed project from loadCodeAssist or falls back to default.
    """

    def __init__(self):
        """Initialize Antigravity OAuth LLM handler."""
        super().__init__()
        self.provider_id = "antigravity"
        self._managed_project_id: Optional[str] = None
        self._managed_project_profile: Optional[str] = None
        self._stored_project_id: Optional[str] = None
        self._session_id = f"agent-{uuid.uuid4()}"
        self._last_request_message_count: int = 0
        self._last_request_model: Optional[str] = None

    def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token for Antigravity."""
        tokens = get_oauth_token(self.provider_id)
        if tokens:
            return tokens.access_token
        return None

    def _require_access_token(self) -> str:
        """Return a valid OAuth token or raise a helpful error."""
        access_token = self._get_access_token()
        if not access_token:
            raise ValueError(
                "No OAuth token available for Antigravity. "
                "Run 'koder auth login antigravity' to authenticate."
            )
        return access_token

    @staticmethod
    def _strip_provider_prefix(model: str) -> str:
        """Strip provider prefixes/suffixes from model name."""
        model_name = model.split("/")[-1] if "/" in model else model
        if model_name.endswith(":antigravity"):
            model_name = model_name[: -len(":antigravity")]
        if model_name.startswith("antigravity-"):
            model_name = model_name[len("antigravity-") :]
        return model_name

    def _select_header_style(self, model: str) -> str:
        """Choose header style (antigravity vs gemini-cli) for a model."""
        raw_lower = model.lower()
        if "claude" in raw_lower:
            return "antigravity"
        if raw_lower.startswith("antigravity/") or "antigravity-" in raw_lower:
            return "antigravity"
        if raw_lower.endswith(":antigravity"):
            return "antigravity"
        # Prefer antigravity quota for Gemini 3 (non-preview)
        if "gemini-3" in raw_lower and "preview" not in raw_lower:
            return "antigravity"
        return "gemini-cli"

    def _header_style_candidates(self, model: str) -> List[str]:
        """Return header styles to try in order."""
        primary = self._select_header_style(model)
        raw_lower = model.lower()
        if "claude" in raw_lower:
            return [primary]
        if (
            raw_lower.startswith("antigravity/")
            or "antigravity-" in raw_lower
            or raw_lower.endswith(":antigravity")
        ):
            return [primary]
        alternate = "gemini-cli" if primary == "antigravity" else "antigravity"
        return [primary, alternate]

    def _resolve_model_for_header_style(
        self, model: str, header_style: str
    ) -> tuple[str, Optional[str], Optional[int]]:
        """Resolve model name for a specific header style."""
        import re

        model_name = self._strip_provider_prefix(model)
        lower = model_name.lower()
        is_gemini3 = "gemini-3" in lower

        if is_gemini3:
            if header_style == "antigravity":
                # Preview -> non-preview, add tier for gemini-3-pro if missing
                model_name = re.sub(r"-preview$", "", model_name, flags=re.IGNORECASE)
                is_gemini3_pro = model_name.lower().startswith("gemini-3-pro")
                has_tier_suffix = re.search(r"-(low|medium|high)$", model_name, re.IGNORECASE)
                if is_gemini3_pro and not has_tier_suffix:
                    model_name = f"{model_name}-low"
            else:
                # gemini-cli uses preview models without tier suffix
                model_name = re.sub(r"-(low|medium|high)$", "", model_name, flags=re.IGNORECASE)
                if not model_name.lower().endswith("-preview"):
                    model_name = f"{model_name}-preview"

        return self._resolve_model_with_tier(model_name)

    def _resolve_model_with_tier(self, model: str) -> tuple[str, Optional[str], Optional[int]]:
        """Resolve model name and extract thinking tier configuration.

        Claude models use tier suffixes (-low/-medium/-high) for thinking budget
        configuration, but the API model name doesn't include these suffixes.
        """
        import re

        # Strip provider prefix/suffix
        model_name = self._strip_provider_prefix(model)

        # Check for Claude thinking models with tier suffix
        tier_match = re.match(
            r"^(claude-(?:sonnet|opus)-\d+-\d+(?:-thinking)?)-" r"(low|medium|high)$",
            model_name,
            re.IGNORECASE,
        )

        if tier_match:
            base_model = tier_match.group(1)
            tier = tier_match.group(2).lower()
            thinking_budget = CLAUDE_THINKING_BUDGETS.get(tier)
            return (base_model, tier, thinking_budget)

        # No tier suffix - return as-is
        return (model_name, None, None)

    def _is_claude_thinking_model(self, model: str) -> bool:
        """Check if model is a Claude thinking model."""
        lower = model.lower()
        return "claude" in lower and "thinking" in lower

    def _append_system_instruction_hint(self, request_payload: Dict[str, Any], hint: str) -> None:
        """Append a hint to systemInstruction in various formats."""
        existing = request_payload.get("systemInstruction")

        if isinstance(existing, str):
            request_payload["systemInstruction"] = (
                f"{existing}\n\n{hint}" if existing.strip() else hint
            )
            return

        if isinstance(existing, dict):
            parts = existing.get("parts")
            if isinstance(parts, list) and parts:
                appended = False
                for idx in range(len(parts) - 1, -1, -1):
                    part = parts[idx]
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = f"{part['text']}\n\n{hint}"
                        appended = True
                        break
                if not appended:
                    parts.append({"text": hint})
            else:
                existing["parts"] = [{"text": hint}]
            request_payload["systemInstruction"] = existing
            return

        request_payload["systemInstruction"] = {"parts": [{"text": hint}]}

    @staticmethod
    def _is_signature_error(status_code: int, error_text: Optional[str]) -> bool:
        """Detect invalid thought signature errors for self-healing retries."""
        if status_code != 400 or not error_text:
            return False
        return any(pattern in error_text for pattern in SIGNATURE_ERROR_PATTERNS)

    @staticmethod
    def _parse_retry_delay_seconds(value: Optional[str]) -> Optional[float]:
        if not value or not isinstance(value, str):
            return None
        try:
            if value.endswith("ms"):
                return float(value[:-2]) / 1000.0
            if value.endswith("s"):
                return float(value[:-1])
            return float(value)
        except (ValueError, TypeError):
            return None

    @classmethod
    def _extract_rate_limit_delay(cls, error_text: Optional[str]) -> Optional[float]:
        if not error_text:
            return None
        try:
            payload = json.loads(error_text)
        except json.JSONDecodeError:
            return None

        error = payload.get("error") or {}
        details = error.get("details") or []
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    delay = cls._parse_retry_delay_seconds(detail.get("retryDelay"))
                    if delay is not None:
                        return delay
                metadata = detail.get("metadata")
                if isinstance(metadata, dict):
                    delay = cls._parse_retry_delay_seconds(metadata.get("quotaResetDelay"))
                    if delay is not None:
                        return delay
                    reset_at = metadata.get("quotaResetTimeStamp")
                    if isinstance(reset_at, str):
                        try:
                            if reset_at.endswith("Z"):
                                reset_at = reset_at[:-1] + "+00:00"
                            target = datetime.datetime.fromisoformat(reset_at)
                            now = datetime.datetime.now(datetime.timezone.utc)
                            delta = (target - now).total_seconds()
                            if delta > 0:
                                return delta
                        except ValueError:
                            pass
        return None

    @staticmethod
    def _inject_repair_prompt(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Append repair prompt to last user message without mutating input."""
        if not messages:
            return messages

        cloned: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, dict):
                cloned.append(dict(msg))
            else:
                cloned.append(msg)

        for msg in reversed(cloned):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                msg["content"] = content + REPAIR_PROMPT
            elif isinstance(content, list):
                updated = list(content)
                updated.append({"type": "text", "text": REPAIR_PROMPT})
                msg["content"] = updated
            else:
                msg["content"] = REPAIR_PROMPT
            break

        return cloned

    def _inject_antigravity_system_instruction(self, request_payload: Dict[str, Any]) -> None:
        """Inject Antigravity system instruction with role=user."""
        existing = request_payload.get("systemInstruction")

        if isinstance(existing, dict):
            existing["role"] = "user"
            parts = existing.get("parts")
            if isinstance(parts, list) and parts:
                first = parts[0]
                if isinstance(first, dict) and isinstance(first.get("text"), str):
                    first["text"] = f"{ANTIGRAVITY_SYSTEM_INSTRUCTION}\n\n{first['text']}"
                else:
                    parts.insert(0, {"text": ANTIGRAVITY_SYSTEM_INSTRUCTION})
            else:
                existing["parts"] = [{"text": ANTIGRAVITY_SYSTEM_INSTRUCTION}]
            request_payload["systemInstruction"] = existing
            return

        if isinstance(existing, str):
            request_payload["systemInstruction"] = {
                "role": "user",
                "parts": [
                    {"text": f"{ANTIGRAVITY_SYSTEM_INSTRUCTION}\n\n{existing}"},
                ],
            }
            return

        request_payload["systemInstruction"] = {
            "role": "user",
            "parts": [{"text": ANTIGRAVITY_SYSTEM_INSTRUCTION}],
        }

    async def _iter_sse_payloads(self, response: aiohttp.ClientResponse) -> AsyncIterator[str]:
        """Yield SSE data payloads from a streaming response."""
        buffer = ""
        data_lines: List[str] = []

        async for chunk in response.content.iter_chunked(4096):
            if not chunk:
                continue
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.rstrip("\r")
                if line == "":
                    if data_lines:
                        payload = "\n".join(data_lines).strip()
                        data_lines = []
                        if payload and payload != "[DONE]":
                            yield payload
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())

        tail = buffer.strip()
        if tail:
            if tail.startswith("data:"):
                data_lines.append(tail[5:].lstrip())
            elif tail == "" and data_lines:
                payload = "\n".join(data_lines).strip()
                data_lines = []
                if payload and payload != "[DONE]":
                    yield payload

        if data_lines:
            payload = "\n".join(data_lines).strip()
            if payload and payload != "[DONE]":
                yield payload

    @staticmethod
    def _filter_response_headers(headers: aiohttp.typedefs.LooseHeaders) -> Dict[str, str]:
        """Filter response headers for safe debug logging."""
        filtered: Dict[str, str] = {}
        for key, value in headers.items():
            lower = key.lower()
            if (
                lower
                in {
                    "content-type",
                    "content-length",
                    "date",
                    "server",
                    "retry-after",
                    "x-request-id",
                    "x-goog-request-id",
                    "x-goog-trace-id",
                }
                or lower.startswith("x-")
                or lower.startswith("grpc-")
            ):
                filtered[key] = value
        return filtered

    @staticmethod
    def _project_profile_for_model(model_name: Optional[str]) -> str:
        """Resolve managed-project profile for a model."""
        if model_name and "claude" in model_name.lower():
            return "antigravity"
        return "gemini"

    async def _load_managed_project(
        self,
        access_token: str,
        fallback_project_id: Optional[str] = None,
        profile: str = "gemini",
    ) -> Optional[str]:
        """Load managed project from Code Assist API."""
        if profile == "antigravity":
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "User-Agent": "antigravity/1.11.9 windows/amd64",
                "Host": "cloudcode-pa.googleapis.com",
            }
            metadata: Dict[str, str] = {"ideType": "ANTIGRAVITY"}
        else:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "User-Agent": "google-api-nodejs-client/9.15.1",
                "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
                "Client-Metadata": ANTIGRAVITY_HEADERS["Client-Metadata"],
            }
            metadata = {
                "ideType": "IDE_UNSPECIFIED",
                "platform": "PLATFORM_UNSPECIFIED",
                "pluginType": "GEMINI",
            }
        if fallback_project_id:
            metadata["duetProject"] = fallback_project_id

        load_body = {"metadata": metadata}

        load_endpoints = [
            ANTIGRAVITY_ENDPOINT_PROD,
            ANTIGRAVITY_ENDPOINT_DAILY,
            ANTIGRAVITY_ENDPOINT_AUTOPUSH,
        ]

        async with aiohttp.ClientSession() as session:
            for endpoint in load_endpoints:
                try:
                    async with session.post(
                        f"{endpoint}/v1internal:loadCodeAssist",
                        json=load_body,
                        headers=headers,
                    ) as response:
                        if response.ok:
                            data = await response.json()
                            project = data.get("cloudaicompanionProject")
                            if isinstance(project, str):
                                return project
                            if isinstance(project, dict) and project.get("id"):
                                return project["id"]
                except Exception:
                    continue

        return None

    async def _ensure_project_context(
        self,
        access_token: str,
        prefer_managed: bool = False,
        model: Optional[str] = None,
    ) -> str:
        """Ensure we have a valid project ID for Antigravity."""
        profile = self._project_profile_for_model(model)

        if self._managed_project_id and self._managed_project_profile == profile:
            return self._managed_project_id

        tokens = get_oauth_token(self.provider_id)
        if tokens:
            managed_project = (
                tokens.extra.get("managed_project_id")
                or tokens.extra.get("managedProjectId")
                or tokens.extra.get("managedProject")
            )
            if managed_project:
                self._managed_project_id = managed_project
                self._managed_project_profile = profile
                if prefer_managed:
                    return managed_project

            stored_project = tokens.extra.get("project_id") or tokens.extra.get("projectId")
            if stored_project:
                self._stored_project_id = stored_project
                if not prefer_managed:
                    return stored_project

        project_id = await self._load_managed_project(
            access_token, ANTIGRAVITY_DEFAULT_PROJECT_ID, profile=profile
        )
        if project_id:
            self._managed_project_id = project_id
            self._managed_project_profile = profile
            return project_id

        if (
            not prefer_managed
            and self._managed_project_id
            and self._managed_project_profile == profile
        ):
            return self._managed_project_id

        logger.debug("Using default Antigravity project ID: %s", ANTIGRAVITY_DEFAULT_PROJECT_ID)
        return ANTIGRAVITY_DEFAULT_PROJECT_ID

    def _build_request_body(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        project_id: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build Antigravity request body."""
        from koder_agent.auth.tool_utils import (
            apply_tool_pairing_fixes,
            convert_tool_calls_to_gemini_parts,
            convert_tool_message_to_gemini_part,
            convert_tools_to_claude_format,
            convert_tools_to_gemini_format,
            ensure_thinking_signature_for_tool_use,
            has_valid_signature_for_function_calls,
            inject_parameter_signatures,
            inject_thought_signature_for_function_calls,
            inject_tool_hardening_instruction,
            strip_tool_call_signatures,
        )

        header_style = kwargs.pop("header_style", "antigravity")
        session_id = kwargs.pop("session_id", None)
        is_retry = kwargs.pop("is_retry", False)

        if is_retry:
            messages = strip_tool_call_signatures(messages)

        model_name, _tier, thinking_budget = self._resolve_model_for_header_style(
            model, header_style
        )
        self._last_request_model = model_name
        is_claude = "claude" in model_name.lower()
        is_claude_thinking = self._is_claude_thinking_model(model_name)

        contents: List[Dict[str, Any]] = []
        system_instruction = None
        pending_tool_responses: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, str):
                    system_instruction = {"parts": [{"text": content}]}
                continue

            if role == "tool":
                tool_part = convert_tool_message_to_gemini_part(msg)
                pending_tool_responses.append(tool_part)
                continue

            if pending_tool_responses:
                contents.append({"role": "user", "parts": pending_tool_responses})
                pending_tool_responses = []

            gemini_role = "model" if role == "assistant" else "user"

            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                parts = convert_tool_calls_to_gemini_parts(tool_calls)
                if isinstance(content, str) and content:
                    parts.insert(0, {"text": content})
                contents.append({"role": gemini_role, "parts": parts})
            elif isinstance(content, str):
                contents.append({"role": gemini_role, "parts": [{"text": content}]})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append({"text": item.get("text", "")})
                if parts:
                    contents.append({"role": gemini_role, "parts": parts})

        if pending_tool_responses:
            contents.append({"role": "user", "parts": pending_tool_responses})

        request_payload: Dict[str, Any] = {"contents": contents}

        has_tool_calls = any(
            isinstance(content, dict)
            and isinstance(content.get("parts"), list)
            and any(isinstance(part, dict) and "functionCall" in part for part in content["parts"])
            for content in contents
        )
        enable_thinking = is_claude_thinking
        if enable_thinking and has_tool_calls:
            signature_session_id = None if is_retry else session_id
            if not has_valid_signature_for_function_calls(
                contents, signature_session_id, model=model_name
            ):
                logger.warning(
                    "Disabling Claude thinking: no valid thoughtSignature for tool calls (session=%s)",
                    session_id,
                )
                enable_thinking = False

        self._last_request_message_count = len(contents)

        if system_instruction:
            request_payload["systemInstruction"] = system_instruction

        tools = kwargs.get("tools")
        if tools:
            if is_claude:
                converted_tools = convert_tools_to_claude_format(tools)
                converted_tools = inject_parameter_signatures(converted_tools)
            else:
                converted_tools = convert_tools_to_gemini_format(tools)
            if converted_tools:
                request_payload["tools"] = converted_tools

            if enable_thinking:
                self._append_system_instruction_hint(
                    request_payload, CLAUDE_INTERLEAVED_THINKING_HINT
                )

            if is_claude:
                inject_tool_hardening_instruction(request_payload)

        generation_config: Dict[str, Any] = {}
        if "temperature" in kwargs:
            generation_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            generation_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            generation_config["topP"] = kwargs["top_p"]

        if enable_thinking:
            thinking_config: Dict[str, Any] = {"include_thoughts": True}
            if thinking_budget:
                thinking_config["thinking_budget"] = thinking_budget
            generation_config["thinkingConfig"] = thinking_config
            current_max = generation_config.get("maxOutputTokens", 0)
            if not current_max or current_max <= (thinking_budget or 0):
                generation_config["maxOutputTokens"] = CLAUDE_THINKING_MAX_OUTPUT_TOKENS

        if generation_config:
            request_payload["generationConfig"] = generation_config

        if is_claude and request_payload.get("tools"):
            request_payload["toolConfig"] = {"functionCallingConfig": {"mode": "VALIDATED"}}

        if is_claude:
            request_payload = apply_tool_pairing_fixes(request_payload, is_claude=True)

        if enable_thinking and request_payload.get("contents"):
            session_key = session_id or f"antigravity-{uuid.uuid4().hex[:8]}"
            request_payload["contents"] = ensure_thinking_signature_for_tool_use(
                request_payload["contents"], session_key, allow_sentinel=False
            )

        if session_id and request_payload.get("contents") and not is_retry:
            request_payload["contents"] = inject_thought_signature_for_function_calls(
                request_payload["contents"], session_id, model=model_name
            )

        if header_style == "antigravity":
            self._inject_antigravity_system_instruction(request_payload)

        # Build envelope matching Antigravity-Manager's structure
        return {
            "project": project_id,
            "requestId": f"agent-{uuid.uuid4()}",
            "request": request_payload,
            "model": model_name,
            "userAgent": "antigravity",
            "requestType": "claude" if is_claude else "gemini",
        }

    def _parse_response(self, response_data: Dict[str, Any], model: str) -> ModelResponse:
        """Parse Antigravity response into LiteLLM ModelResponse format."""
        from koder_agent.auth.tool_utils import (
            extract_tool_calls_from_gemini_response,
            remap_function_call_args,
        )

        if "response" in response_data:
            response_data = response_data["response"]

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

        tool_calls = extract_tool_calls_from_gemini_response(
            response_data,
            session_id=self._session_id,
            message_count=self._last_request_message_count,
            model=self._last_request_model,
        )

        # Remap hallucinated arguments for Claude models
        is_claude_model = "claude" in model.lower()
        if tool_calls and is_claude_model:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args_str = func.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    args = remap_function_call_args(name, args)
                    func["arguments"] = json.dumps(args)
                except (json.JSONDecodeError, TypeError):
                    pass

        usage_data = response_data.get("usageMetadata", {})
        usage = Usage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            total_tokens=usage_data.get("totalTokenCount", 0),
        )

        if tool_calls:
            message: Dict[str, Any] = {
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tool_calls,
            }
            finish_reason = "tool_calls"
        else:
            message = {"role": "assistant", "content": content}
            finish_reason = "stop"

        return ModelResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=response_data.get("model", model),
            choices=[{"index": 0, "message": message, "finish_reason": finish_reason}],
            usage=usage,
        )

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Synchronous completion (not implemented - use async)."""
        raise NotImplementedError("Use acompletion for Antigravity OAuth")

    async def acompletion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using Antigravity endpoint with endpoint fallback."""
        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        project_id = await self._ensure_project_context(access_token, model=model)

        header_styles = self._header_style_candidates(model)
        is_claude_model = "claude" in model.lower()
        endpoint_order = (
            CLAUDE_ENDPOINT_FALLBACKS if is_claude_model else ANTIGRAVITY_ENDPOINT_FALLBACKS
        )

        last_error = ""
        for sig_retry in range(MAX_SIGNATURE_RETRIES + 1):
            is_retry = sig_retry > 0
            effective_messages = self._inject_repair_prompt(messages) if is_retry else messages

            for header_style in header_styles:
                body = self._build_request_body(
                    model,
                    effective_messages,
                    project_id,
                    header_style=header_style,
                    session_id=self._session_id,
                    is_retry=is_retry,
                    **merged_kwargs,
                )
                resolved_model = body["model"]

                base_headers = (
                    ANTIGRAVITY_HEADERS if header_style == "antigravity" else GEMINI_CLI_HEADERS
                )
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    **base_headers,
                }

                for endpoint in endpoint_order:
                    for rate_retry in range(MAX_RATE_LIMIT_RETRIES + 1):
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    f"{endpoint}/v1internal:generateContent",
                                    json=body,
                                    headers=headers,
                                ) as response:
                                    if response.ok:
                                        response_data = await response.json()
                                        return self._parse_response(response_data, resolved_model)

                                    error_text = await response.text()
                                    last_error = f"({response.status}): {error_text[:500]}"

                                    if self._is_signature_error(response.status, error_text):
                                        logger.warning(
                                            "Antigravity signature error (retry=%s): %s",
                                            sig_retry,
                                            error_text[:200],
                                        )
                                        break

                                    if (
                                        response.status == 429
                                        and rate_retry < MAX_RATE_LIMIT_RETRIES
                                    ):
                                        delay = (
                                            self._extract_rate_limit_delay(error_text)
                                            or DEFAULT_RATE_LIMIT_DELAY_SECONDS
                                        )
                                        logger.debug(
                                            "Antigravity rate limit (retry=%s delay=%.1fs)",
                                            rate_retry,
                                            delay,
                                        )
                                        await asyncio.sleep(delay)
                                        continue

                                    break
                        except Exception as e:
                            last_error = str(e)
                            break

        raise ValueError(f"Antigravity OAuth API error (all endpoints failed): {last_error}")

    def streaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> Iterator[GenericStreamingChunk]:
        """Synchronous streaming (not implemented - use async)."""
        raise NotImplementedError("Use astreaming for Antigravity OAuth")

    async def astreaming(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming using Antigravity endpoint with endpoint fallback."""
        from koder_agent.auth.tool_utils import (
            _extract_part_thought_signature,
            cache_session_signature,
            cache_thought_signature,
            remap_function_call_args,
        )

        access_token = self._require_access_token()
        merged_kwargs = merge_optional_params(kwargs)

        project_id = await self._ensure_project_context(access_token, model=model)

        header_styles = self._header_style_candidates(model)
        is_claude_model = "claude" in model.lower()
        endpoint_order = (
            CLAUDE_ENDPOINT_FALLBACKS if is_claude_model else ANTIGRAVITY_ENDPOINT_FALLBACKS
        )

        last_error = ""
        for sig_retry in range(MAX_SIGNATURE_RETRIES + 1):
            is_retry = sig_retry > 0
            effective_messages = self._inject_repair_prompt(messages) if is_retry else messages

            for header_style in header_styles:
                body = self._build_request_body(
                    model,
                    effective_messages,
                    project_id,
                    header_style=header_style,
                    session_id=self._session_id,
                    is_retry=is_retry,
                    **merged_kwargs,
                )
                # resolved_model = body["model"]  # Available if needed for debug

                base_headers = (
                    ANTIGRAVITY_HEADERS if header_style == "antigravity" else GEMINI_CLI_HEADERS
                )
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                    **base_headers,
                }

                for endpoint in endpoint_order:
                    for rate_retry in range(MAX_RATE_LIMIT_RETRIES + 1):
                        try:
                            async with aiohttp.ClientSession() as session:
                                async with session.post(
                                    f"{endpoint}/v1internal:streamGenerateContent?alt=sse",
                                    json=body,
                                    headers=headers,
                                ) as response:
                                    if not response.ok:
                                        error_text = await response.text()
                                        last_error = f"({response.status}): {error_text[:500]}"

                                        if self._is_signature_error(response.status, error_text):
                                            logger.warning(
                                                "Antigravity streaming signature error (retry=%s)",
                                                sig_retry,
                                            )
                                            break

                                        if (
                                            response.status == 429
                                            and rate_retry < MAX_RATE_LIMIT_RETRIES
                                        ):
                                            delay = (
                                                self._extract_rate_limit_delay(error_text)
                                                or DEFAULT_RATE_LIMIT_DELAY_SECONDS
                                            )
                                            await asyncio.sleep(delay)
                                            continue
                                        break

                                    chunk_index = 0
                                    has_tool_calls = False
                                    yielded_tool_call_ids: set = set()
                                    pending_thought_sig: Optional[str] = None

                                    async for payload in self._iter_sse_payloads(response):
                                        try:
                                            data = json.loads(payload)
                                            if "response" in data:
                                                data = data["response"]

                                            if "candidates" in data:
                                                for candidate in data["candidates"]:
                                                    if "content" in candidate:
                                                        for part in candidate["content"].get(
                                                            "parts", []
                                                        ):
                                                            thought_sig = (
                                                                _extract_part_thought_signature(
                                                                    part
                                                                )
                                                            )
                                                            if thought_sig:
                                                                cache_session_signature(
                                                                    self._session_id,
                                                                    thought_sig,
                                                                    self._last_request_message_count,
                                                                    model=self._last_request_model,
                                                                )
                                                                pending_thought_sig = thought_sig

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
                                                                has_tool_calls = True
                                                                func_call = part["functionCall"]
                                                                name = func_call.get("name", "")
                                                                args = func_call.get("args", {})
                                                                call_id = func_call.get(
                                                                    "id",
                                                                    f"call_{uuid.uuid4().hex[:8]}",
                                                                )
                                                                if (
                                                                    isinstance(args, dict)
                                                                    and is_claude_model
                                                                ):
                                                                    args = remap_function_call_args(
                                                                        name, args
                                                                    )
                                                                tool_use_data: Dict[str, Any] = {
                                                                    "id": call_id,
                                                                    "type": "function",
                                                                    "function": {
                                                                        "name": name,
                                                                        "arguments": (
                                                                            json.dumps(args)
                                                                            if isinstance(
                                                                                args, dict
                                                                            )
                                                                            else str(args)
                                                                        ),
                                                                    },
                                                                }
                                                                thought_sig = (
                                                                    _extract_part_thought_signature(
                                                                        part
                                                                    )
                                                                    or pending_thought_sig
                                                                )
                                                                if thought_sig:
                                                                    tool_use_data[
                                                                        "thought_signature"
                                                                    ] = thought_sig
                                                                    cache_session_signature(
                                                                        self._session_id,
                                                                        thought_sig,
                                                                        self._last_request_message_count,
                                                                        model=self._last_request_model,
                                                                    )
                                                                    cache_thought_signature(
                                                                        call_id, name, thought_sig
                                                                    )
                                                                if (
                                                                    call_id
                                                                    not in yielded_tool_call_ids
                                                                ):
                                                                    yielded_tool_call_ids.add(
                                                                        call_id
                                                                    )
                                                                    yield GenericStreamingChunk(
                                                                        text="",
                                                                        is_finished=False,
                                                                        finish_reason=None,
                                                                        usage=None,
                                                                        index=chunk_index,
                                                                        tool_use=tool_use_data,
                                                                    )
                                                                    chunk_index += 1
                                        except json.JSONDecodeError:
                                            continue

                                    finish_reason = "tool_calls" if has_tool_calls else "stop"
                                    yield GenericStreamingChunk(
                                        text="",
                                        is_finished=True,
                                        finish_reason=finish_reason,
                                        usage=None,
                                        index=chunk_index,
                                    )
                                    return
                        except Exception as e:
                            last_error = str(e)
                            break

        raise ValueError(
            f"Antigravity OAuth API streaming error (all endpoints failed): {last_error}"
        )
