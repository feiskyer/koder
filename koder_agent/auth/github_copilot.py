"""Asynchronous device login using the installed SDK's Copilot cache contract."""

from __future__ import annotations

import asyncio
import json
import math
import numbers
import os
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TypeVar

import httpx
from filelock import FileLock

from koder_agent.utils.async_tasks import await_owned_task, run_sync_owned
from koder_agent.utils.atomic_file import write_text_atomic


class CopilotLoginError(Exception):
    """A safe, user-visible authentication error without response/token contents."""


@dataclass(frozen=True)
class CopilotLoginResult:
    token_dir: str
    api_endpoint: str


_HTTPResource = TypeVar("_HTTPResource", httpx.AsyncClient, httpx.Response)


@asynccontextmanager
async def _owned_http(resource: _HTTPResource) -> AsyncIterator[_HTTPResource]:
    """Close an HTTP resource through repeated cancellation without masking failure."""
    try:
        yield resource
    except BaseException:
        with suppress(Exception, asyncio.CancelledError):
            await await_owned_task(asyncio.create_task(resource.aclose()))
        raise
    else:
        await await_owned_task(asyncio.create_task(resource.aclose()))


def _positive_seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise CopilotLoginError("OAuth timeout and polling intervals must be positive numbers")
    if not math.isfinite(value) or value <= 0:
        raise CopilotLoginError("OAuth timeout and polling intervals must be positive and finite")
    return float(value)


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise CopilotLoginError(f"GitHub Copilot response is missing a valid {field}")
    return value


async def _request_json(client: httpx.AsyncClient, method: str, url: str, **kwargs) -> dict:
    try:
        request = client.build_request(method, url, **kwargs)
        response = await client.send(request, stream=True)
        async with _owned_http(response):
            response.raise_for_status()
            await response.aread()
            result = response.json()
    except httpx.HTTPStatusError as exc:
        raise CopilotLoginError(
            f"GitHub Copilot authentication service returned HTTP {exc.response.status_code}"
        ) from None
    except httpx.RequestError as exc:
        raise CopilotLoginError(
            f"GitHub Copilot authentication request failed ({type(exc).__name__})"
        ) from None
    except ValueError:
        raise CopilotLoginError("GitHub Copilot returned invalid JSON") from None
    if not isinstance(result, dict):
        raise CopilotLoginError("GitHub Copilot returned an invalid response object")
    return result


def _publish_cache(authenticator, access_token: str, api_key_info: dict) -> None:
    """Publish each SDK cache file atomically under a short cooperating-writer lock.

    This is not a two-file or power-loss transaction. The async owner joins this
    entire operation before returning, including when cancellation races a write.
    """
    access_path = Path(authenticator.access_token_file)
    api_path = Path(authenticator.api_key_file)
    if access_path.resolve() == api_path.resolve():
        raise CopilotLoginError("GitHub Copilot cache paths must be distinct")
    paths = (access_path, api_path)
    if any(path.is_symlink() for path in paths):
        raise CopilotLoginError("Refusing a symbolic-link GitHub Copilot cache file")
    serialized = json.dumps(api_key_info, allow_nan=False)
    lock_path = Path(authenticator.token_dir) / ".koder-login.lock"
    if lock_path.is_symlink():
        raise CopilotLoginError("Refusing a symbolic-link GitHub Copilot cache lock")
    with FileLock(str(lock_path), timeout=5, mode=0o600):
        for path in paths:
            if path.is_symlink():
                raise CopilotLoginError("Refusing a symbolic-link GitHub Copilot cache file")
            if path.exists():
                path.chmod(0o600)
        write_text_atomic(access_path, access_token)
        write_text_atomic(api_path, serialized)


async def login(
    timeout: float, *, on_device_code: Callable[[str, str], None]
) -> CopilotLoginResult:
    """Own device polling, token exchange and final cache publication."""
    timeout = _positive_seconds(timeout)
    from litellm.llms.github_copilot import authenticator as sdk

    async with asyncio.timeout(timeout) as deadline:
        authenticator = await run_sync_owned(sdk.Authenticator)
        client_id = os.getenv("GITHUB_COPILOT_CLIENT_ID", sdk.DEFAULT_GITHUB_CLIENT_ID)
        device_url = os.getenv("GITHUB_COPILOT_DEVICE_CODE_URL", sdk.DEFAULT_GITHUB_DEVICE_CODE_URL)
        token_url = os.getenv(
            "GITHUB_COPILOT_ACCESS_TOKEN_URL", sdk.DEFAULT_GITHUB_ACCESS_TOKEN_URL
        )
        api_key_url = os.getenv("GITHUB_COPILOT_API_KEY_URL", sdk.DEFAULT_GITHUB_API_KEY_URL)
        headers = authenticator._get_github_headers()
        async with _owned_http(
            httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        ) as client:
            device = await _request_json(
                client,
                "POST",
                device_url,
                headers=headers,
                json={"client_id": client_id, "scope": "read:user"},
            )
            device_code = _required_string(device, "device_code")
            user_code = _required_string(device, "user_code")
            verification_uri = _required_string(device, "verification_uri")
            interval = _positive_seconds(device.get("interval", 5))
            if "expires_in" in device:
                expires = asyncio.get_running_loop().time() + _positive_seconds(
                    device["expires_in"]
                )
                deadline.reschedule(min(deadline.when(), expires))
            on_device_code(verification_uri, user_code)
            while True:
                token = await _request_json(
                    client,
                    "POST",
                    token_url,
                    headers=headers,
                    json={
                        "client_id": client_id,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
                if "access_token" in token:
                    access_token = _required_string(token, "access_token")
                    break
                error = token.get("error")
                if error == "slow_down":
                    interval += 5
                elif error != "authorization_pending":
                    raise CopilotLoginError("GitHub Copilot device authorization was rejected")
                await asyncio.sleep(interval)
            api_key_info = await _request_json(
                client,
                "GET",
                api_key_url,
                headers=authenticator._get_github_headers(access_token),
            )
        _required_string(api_key_info, "token")
        if "expires_at" in api_key_info:
            _positive_seconds(api_key_info["expires_at"])
        endpoints = api_key_info.get("endpoints", {})
        if not isinstance(endpoints, dict):
            raise CopilotLoginError("GitHub Copilot returned invalid endpoint metadata")
        api_endpoint = endpoints.get("api")
        if api_endpoint is not None and not isinstance(api_endpoint, str):
            raise CopilotLoginError("GitHub Copilot returned an invalid API endpoint")
        await run_sync_owned(_publish_cache, authenticator, access_token, api_key_info)
        return CopilotLoginResult(str(authenticator.token_dir), api_endpoint or "default")
