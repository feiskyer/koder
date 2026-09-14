"""Thin compatibility wrapper around the existing provider/auth stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .models import ProviderAuthSnapshot, ResolvedModelClient


@dataclass
class ProviderCompat:
    """Runtime-facing adapter for providers and auth state."""

    oauth_providers: list[str] = field(default_factory=list)

    @classmethod
    def from_current_runtime(cls) -> "ProviderCompat":
        from koder_agent.auth.client_integration import list_oauth_provider_ids

        return cls(oauth_providers=list_oauth_provider_ids())

    def resolve_model_client(self, model_name: Optional[str] = None) -> ResolvedModelClient:
        from koder_agent.utils.client import get_model_client_snapshot

        snapshot = get_model_client_snapshot(model_name)
        return ResolvedModelClient(
            model_name=snapshot["model_name"],
            api_key=snapshot["api_key"],
            base_url=snapshot["base_url"],
            litellm_kwargs=dict(snapshot["litellm_kwargs"]),
            native_openai=snapshot["native_openai"],
        )

    def auth_snapshot(self, provider: str) -> ProviderAuthSnapshot:
        from koder_agent.auth.client_integration import get_provider_auth_info

        api_key, headers, is_oauth = get_provider_auth_info(provider)
        return ProviderAuthSnapshot(
            provider=provider,
            api_key=api_key,
            headers=headers or {},
            is_oauth=is_oauth,
        )
