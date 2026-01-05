#!/usr/bin/env python3
"""List available models from OAuth providers.

Uses stored OAuth tokens to fetch model lists from each provider.
Supports Google, Claude, ChatGPT, and Antigravity providers.

Usage:
    uv run python scripts/list_oauth_models.py                    # List all providers
    uv run python scripts/list_oauth_models.py antigravity        # Single provider
    uv run python scripts/list_oauth_models.py google chatgpt     # Multiple providers
    uv run python scripts/list_oauth_models.py --refresh          # Force refresh tokens
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from koder_agent.auth.providers import get_provider, list_providers
from koder_agent.auth.token_storage import get_token_storage


# ANSI color codes
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


async def refresh_token_if_needed(provider_id: str, tokens, storage) -> str:
    """Refresh access token if expired.

    Args:
        provider_id: Provider identifier
        tokens: Current OAuth tokens
        storage: Token storage instance

    Returns:
        Valid access token
    """
    if not tokens.is_expired():
        return tokens.access_token

    print(f"  {Colors.YELLOW}Token expired, refreshing...{Colors.RESET}")
    provider = get_provider(provider_id)
    result = await provider.refresh_tokens(tokens.refresh_token)

    if result.success and result.tokens:
        storage.save(result.tokens)
        return result.tokens.access_token
    else:
        raise RuntimeError(f"Failed to refresh token: {result.error}")


async def list_models_for_provider(
    provider_id: str, storage, force_refresh: bool = False
) -> tuple[list[str], dict]:
    """List models for a single provider.

    Args:
        provider_id: Provider identifier (google, claude, chatgpt, antigravity)
        storage: Token storage instance
        force_refresh: Force refresh the access token

    Returns:
        Tuple of (model list, status dict)
    """
    tokens = storage.load(provider_id)
    if not tokens:
        raise RuntimeError(
            f"No tokens found for {provider_id}. Run 'uv run koder auth login {provider_id}' first."
        )

    # Get valid access token
    if force_refresh or tokens.is_expired():
        access_token = await refresh_token_if_needed(provider_id, tokens, storage)
    else:
        access_token = tokens.access_token

    # Get provider and list models
    provider = get_provider(provider_id)
    models, status = await provider.list_models(access_token)

    return models, status


def get_source_label(status: dict) -> str:
    """Get colored source label for display."""
    source = status.get("source", "unknown")
    if source == "api":
        return f"{Colors.GREEN}API{Colors.RESET}"
    elif source == "fallback":
        return f"{Colors.YELLOW}fallback{Colors.RESET}"
    elif source == "hardcoded":
        return f"{Colors.CYAN}hardcoded{Colors.RESET}"
    return source


def print_status_error(status: dict) -> None:
    """Print error details if present."""
    source = status.get("source", "unknown")
    error = status.get("error")
    reason = status.get("reason")

    if source == "fallback" and error:
        print(f"  {Colors.YELLOW}⚠ API failed, using prefilled models{Colors.RESET}")
        print(f"  {Colors.RED}  Error: {error}{Colors.RESET}")
    elif source == "hardcoded" and reason:
        print(f"  {Colors.CYAN}ℹ {reason}{Colors.RESET}")


async def main():
    parser = argparse.ArgumentParser(
        description="List available models from OAuth providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python scripts/list_oauth_models.py                    # All providers
    uv run python scripts/list_oauth_models.py antigravity        # Single provider
    uv run python scripts/list_oauth_models.py google chatgpt     # Multiple providers
    uv run python scripts/list_oauth_models.py --refresh          # Force token refresh
        """,
    )
    parser.add_argument(
        "providers",
        nargs="*",
        default=["all"],
        help="Provider(s) to list models for. Use 'all' for all providers. (default: all)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh OAuth tokens before listing models",
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        Colors.RED = ""
        Colors.GREEN = ""
        Colors.YELLOW = ""
        Colors.BLUE = ""
        Colors.CYAN = ""
        Colors.RESET = ""
        Colors.BOLD = ""

    # Determine which providers to query
    available_providers = list_providers()
    if "all" in args.providers:
        target_providers = available_providers
    else:
        # Validate provider names
        for p in args.providers:
            if p not in available_providers:
                print(f"Error: Unknown provider '{p}'. Available: {', '.join(available_providers)}")
                sys.exit(1)
        target_providers = args.providers

    storage = get_token_storage()
    results = {}

    for provider_id in target_providers:
        print(f"\n{'=' * 60}")
        print(f"{Colors.BOLD}Provider: {provider_id.upper()}{Colors.RESET}")
        print("=" * 60)

        try:
            models, status = await list_models_for_provider(provider_id, storage, args.refresh)
            results[provider_id] = {"success": True, "models": models, "status": status}

            # Print error details if any
            print_status_error(status)

            # Print models with source label
            source_label = get_source_label(status)
            if models:
                print(f"  Found {Colors.BOLD}{len(models)}{Colors.RESET} models ({source_label}):")
                for model in sorted(models):
                    print(f"    - {model}")
            else:
                print(f"  No models found ({source_label})")

        except RuntimeError as e:
            results[provider_id] = {
                "success": False,
                "error": str(e),
                "status": {"source": "error"},
            }
            print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
        except Exception as e:
            results[provider_id] = {
                "success": False,
                "error": str(e),
                "status": {"source": "error"},
            }
            print(f"  {Colors.RED}Unexpected error: {e}{Colors.RESET}")

    # Print JSON output if requested
    if args.json:
        import json

        print("\n" + "=" * 60)
        print("JSON Output:")
        print("=" * 60)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
