#!/usr/bin/env python3
"""Check Antigravity model quotas and subscription status.

Usage:
    uv run python scripts/antigravity_quota.py           # Show quota summary
    uv run python scripts/antigravity_quota.py --raw     # Show raw API responses
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import aiohttp

from koder_agent.auth.providers import get_provider
from koder_agent.auth.token_storage import get_token_storage

# API configuration
API_BASE = "https://cloudcode-pa.googleapis.com"
USER_AGENT = "antigravity/1.11.3 Darwin/arm64"
DEFAULT_PROJECT_ID = "bamboo-precept-lgxtn"


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


async def get_access_token() -> str:
    """Get valid Antigravity access token, refreshing if needed."""
    storage = get_token_storage()
    tokens = storage.load("antigravity")

    if not tokens:
        raise RuntimeError(
            "No Antigravity tokens. Run 'uv run koder auth login antigravity' first."
        )

    if tokens.is_expired():
        print(f"{Colors.YELLOW}Refreshing expired token...{Colors.RESET}")
        provider = get_provider("antigravity")
        result = await provider.refresh_tokens(tokens.refresh_token)
        if result.success and result.tokens:
            storage.save(result.tokens)
            return result.tokens.access_token
        raise RuntimeError(f"Token refresh failed: {result.error}")

    return tokens.access_token


async def fetch_project_info(session: aiohttp.ClientSession, token: str) -> dict:
    """Fetch project ID and subscription tier."""
    async with session.post(
        f"{API_BASE}/v1internal:loadCodeAssist",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        json={"metadata": {"ideType": "ANTIGRAVITY"}},
    ) as resp:
        if resp.ok:
            return await resp.json()
        return {}


async def fetch_models(session: aiohttp.ClientSession, token: str, project: str) -> dict:
    """Fetch available models with quota info."""
    async with session.post(
        f"{API_BASE}/v1internal:fetchAvailableModels",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        json={"project": project},
    ) as resp:
        if resp.ok:
            return await resp.json()
        return {}


def render_quota_bar(fraction: float) -> str:
    """Render a visual quota bar."""
    pct = int(fraction * 100)
    filled = pct // 10
    bar = "█" * filled + "░" * (10 - filled)

    if pct >= 80:
        color = Colors.GREEN
    elif pct >= 30:
        color = Colors.YELLOW
    else:
        color = Colors.RED

    return f"{color}[{bar}]{Colors.RESET} {pct}%"


def format_reset_time(reset_time: str) -> str:
    """Format reset time for display."""
    if not reset_time or reset_time == "N/A":
        return "N/A"
    # Extract time portion for brevity
    if "T" in reset_time:
        return reset_time.split("T")[1].replace("Z", " UTC")
    return reset_time


async def main():
    parser = argparse.ArgumentParser(description="Check Antigravity model quotas")
    parser.add_argument("--raw", action="store_true", help="Show raw API responses")
    parser.add_argument("--no-color", action="store_true", help="Disable colors")
    args = parser.parse_args()

    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    print(f"{Colors.BOLD}Antigravity Quota Status{Colors.RESET}\n")

    # Get access token
    try:
        token = await get_access_token()
    except RuntimeError as e:
        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
        sys.exit(1)

    async with aiohttp.ClientSession() as session:
        # Fetch project info
        project_data = await fetch_project_info(session, token)

        if args.raw:
            print(f"{Colors.DIM}=== loadCodeAssist ==={Colors.RESET}")
            print(json.dumps(project_data, indent=2))
            print()

        project_id = project_data.get("cloudaicompanionProject", DEFAULT_PROJECT_ID)
        paid_tier = project_data.get("paidTier", {})
        current_tier = project_data.get("currentTier", {})
        tier_id = paid_tier.get("id") or current_tier.get("id") or "unknown"
        tier_name = paid_tier.get("name") or current_tier.get("name") or tier_id

        print(f"  Project:      {Colors.CYAN}{project_id}{Colors.RESET}")
        print(f"  Subscription: {Colors.GREEN}{tier_name}{Colors.RESET} ({tier_id})")

        # Fetch models
        models_data = await fetch_models(session, token, project_id)

        if args.raw:
            print(f"\n{Colors.DIM}=== fetchAvailableModels ==={Colors.RESET}")
            print(json.dumps(models_data, indent=2))
            print()

        models = models_data.get("models", {})
        if not models:
            print(f"\n{Colors.YELLOW}No models returned from API{Colors.RESET}")
            return

        # Group models by provider
        google_models = []
        anthropic_models = []
        openai_models = []
        other_models = []

        for name, info in models.items():
            if info.get("isInternal"):
                continue

            quota = info.get("quotaInfo", {})
            remaining = quota.get("remainingFraction")
            reset = quota.get("resetTime", "")
            display = info.get("displayName", name)

            entry = {
                "name": name,
                "display": display,
                "remaining": remaining,
                "reset": reset,
            }

            provider = info.get("modelProvider", "")
            if "GOOGLE" in provider:
                google_models.append(entry)
            elif "ANTHROPIC" in provider:
                anthropic_models.append(entry)
            elif "OPENAI" in provider:
                openai_models.append(entry)
            else:
                other_models.append(entry)

        def print_model_group(title: str, models: list):
            if not models:
                return
            print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
            for m in sorted(models, key=lambda x: x["name"]):
                if m["remaining"] is not None:
                    bar = render_quota_bar(m["remaining"])
                    reset = format_reset_time(m["reset"])
                    print(f"  {m['display']:<35} {bar}  {Colors.DIM}reset: {reset}{Colors.RESET}")
                else:
                    print(f"  {m['display']:<35} {Colors.DIM}(no quota info){Colors.RESET}")

        print_model_group("Google Models", google_models)
        print_model_group("Anthropic Models", anthropic_models)
        print_model_group("OpenAI Models", openai_models)
        print_model_group("Other Models", other_models)

        # Summary
        total = len(google_models) + len(anthropic_models) + len(openai_models) + len(other_models)
        print(f"\n{Colors.DIM}Total: {total} models available{Colors.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
