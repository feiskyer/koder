"""OAuth provider implementations.

OAuth providers use different names than API key providers to avoid conflicts:
- google (OAuth) → Gemini CLI subscription
- claude (OAuth) → Claude Max subscription
- chatgpt (OAuth) → ChatGPT Plus/Pro subscription
- antigravity (OAuth) → Antigravity (Gemini 3 + Claude)
"""

import logging
import os
import threading
from typing import Optional

from koder_agent.auth.base import OAuthProvider
from koder_agent.auth.constants import SUPPORTED_PROVIDERS
from koder_agent.auth.providers.antigravity import (
    AntigravityOAuthLLM,
    AntigravityOAuthProvider,
)
from koder_agent.auth.providers.chatgpt import ChatGPTOAuthLLM, ChatGPTOAuthProvider
from koder_agent.auth.providers.claude import ClaudeOAuthLLM, ClaudeOAuthProvider
from koder_agent.auth.providers.google import GoogleOAuthLLM, GoogleOAuthProvider
from koder_agent.litellm_cost_map import get_litellm

litellm = get_litellm()
logger = logging.getLogger(__name__)

# OAuth provider identifiers - single source of truth
OAUTH_PROVIDER_IDS = tuple(SUPPORTED_PROVIDERS)
_REGISTRATION_LOCK = threading.RLock()
_OWNED_HANDLERS_ATTRIBUTE = "_koder_oauth_handlers"


def _reset_registration_lock_after_fork() -> None:
    global _REGISTRATION_LOCK
    _REGISTRATION_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_registration_lock_after_fork)

__all__ = [
    # OAuth Providers
    "GoogleOAuthProvider",
    "ClaudeOAuthProvider",
    "ChatGPTOAuthProvider",
    "AntigravityOAuthProvider",
    # LLM Handlers
    "GoogleOAuthLLM",
    "ClaudeOAuthLLM",
    "ChatGPTOAuthLLM",
    "AntigravityOAuthLLM",
    # Utility functions
    "get_provider",
    "list_providers",
    "register_oauth_providers",
    "get_oauth_model_prefix",
    "is_oauth_provider",
    # Constants
    "OAUTH_PROVIDER_IDS",
]


# Singleton instances for LiteLLM custom handlers
_google_oauth_llm = GoogleOAuthLLM()
_claude_oauth_llm = ClaudeOAuthLLM()
_chatgpt_oauth_llm = ChatGPTOAuthLLM()
_antigravity_oauth_llm = AntigravityOAuthLLM()


def list_providers() -> list[str]:
    """List all available OAuth provider IDs.

    Returns:
        List of provider ID strings
    """
    return list(OAUTH_PROVIDER_IDS)


def get_provider(provider_id: str) -> OAuthProvider:
    """Get OAuth provider instance by ID.

    Args:
        provider_id: Provider identifier (google, claude, chatgpt, antigravity)

    Returns:
        OAuth provider instance

    Raises:
        ValueError: If provider is not supported
    """
    providers = {
        "google": GoogleOAuthProvider,
        "claude": ClaudeOAuthProvider,
        "chatgpt": ChatGPTOAuthProvider,
        "antigravity": AntigravityOAuthProvider,
    }

    if provider_id not in providers:
        raise ValueError(
            f"Unsupported provider: {provider_id}. Supported: {', '.join(providers.keys())}"
        )

    return providers[provider_id]()


def _merge_provider_ids(existing, *, field: str, owned_ids: tuple[str, ...]) -> list[str]:
    if existing is None and field == "_custom_providers":
        existing = []
    if not isinstance(existing, list) or any(
        not isinstance(name, str) or not name for name in existing
    ):
        raise ValueError(f"Invalid LiteLLM provider registry: {field}")
    merged: list[str] = []
    seen_owned: set[str] = set()
    for name in existing:
        if name not in owned_ids or name not in seen_owned:
            merged.append(name)
        if name in owned_ids:
            seen_owned.add(name)
    merged.extend(name for name in owned_ids if name not in seen_owned)
    return merged


def register_oauth_providers() -> None:
    """Explicit legacy SDK registration; Koder requests use private wire routes.

    Only current handlers or identities previously installed by Koder may be
    replaced for its reserved aliases. A conflicting handler fails before any
    mutation. The ownership receipt is process-local coordination, not a trust
    boundary against code that can modify the SDK's globals. Native SDK model
    catalogs are never changed to force selection of a custom handler.
    """
    _register_handler_mapping(_oauth_handlers(), ownership_attribute=_OWNED_HANDLERS_ATTRIBUTE)


def _oauth_handlers() -> dict:
    return {
        "google": _google_oauth_llm,
        "claude": _claude_oauth_llm,
        "chatgpt": _chatgpt_oauth_llm,
        "antigravity": _antigravity_oauth_llm,
    }


def _register_handler_mapping(handlers: dict, *, ownership_attribute: str) -> None:
    """Publish one owned handler group without changing other SDK groups."""
    with _REGISTRATION_LOCK:
        owned_ids = tuple(handlers)
        entries = litellm.custom_provider_map
        if not isinstance(entries, list):
            raise ValueError("Invalid LiteLLM provider registry: custom_provider_map")
        previous = getattr(litellm, ownership_attribute, {})
        if not isinstance(previous, dict):
            previous = {}
        merged_entries = []
        seen_owned: set[str] = set()
        for entry in entries:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("provider"), str)
                or not entry["provider"]
                or entry.get("custom_handler") is None
            ):
                raise ValueError("Invalid LiteLLM provider registry entry")
            name = entry["provider"]
            if name not in handlers:
                merged_entries.append(entry)
                continue
            active = entry["custom_handler"]
            if active is not handlers[name] and active is not previous.get(name):
                raise ValueError(
                    f"OAuth provider registration conflict for reserved prefix: {name}"
                )
            if name not in seen_owned:
                merged_entries.append(
                    entry
                    if active is handlers[name]
                    else {**entry, "custom_handler": handlers[name]}
                )
                seen_owned.add(name)
        merged_entries.extend(
            {"provider": name, "custom_handler": handler}
            for name, handler in handlers.items()
            if name not in seen_owned
        )
        provider_ids = _merge_provider_ids(
            litellm.provider_list, field="provider_list", owned_ids=owned_ids
        )
        custom_ids = _merge_provider_ids(
            litellm._custom_providers, field="_custom_providers", owned_ids=owned_ids
        )
        unchanged = (
            len(entries) == len(merged_entries)
            and all(old is new for old, new in zip(entries, merged_entries))
            and litellm.provider_list == provider_ids
            and litellm._custom_providers == custom_ids
        )
        if unchanged:
            setattr(litellm, ownership_attribute, handlers)
            return

        # Publish only after validation. Retain public list identities for
        # callers holding references to the SDK's registration containers.
        entries[:] = merged_entries
        litellm.provider_list[:] = provider_ids
        if litellm._custom_providers is None:
            litellm._custom_providers = custom_ids
        else:
            litellm._custom_providers[:] = custom_ids
        setattr(litellm, ownership_attribute, handlers)
        logger.info("Registered OAuth provider handlers with LiteLLM: %s", ", ".join(handlers))


def get_oauth_model_prefix(provider: str) -> Optional[str]:
    """Get the LiteLLM model prefix for an OAuth provider.

    Args:
        provider: OAuth provider name (google, claude, chatgpt, antigravity)

    Returns:
        LiteLLM model prefix for OAuth access (same as provider name)
    """
    provider_lower = provider.lower()
    return provider_lower if provider_lower in OAUTH_PROVIDER_IDS else None


def is_oauth_provider(provider: str) -> bool:
    """Check if a provider name is an OAuth provider.

    Args:
        provider: Provider name to check

    Returns:
        True if this is an OAuth provider
    """
    return provider.lower() in OAUTH_PROVIDER_IDS
