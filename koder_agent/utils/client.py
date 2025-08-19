import os

from agents import (
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()


def get_model_name():
    """Get the appropriate model name based on the configured client type."""
    model = os.environ.get("KODER_MODEL", "gpt-4.1")
    if os.environ.get("OPENAI_API_KEY"):
        return model

    # LiteLLM - use litellm format like "litellm/gemini/gemini-2.5-pro"
    if not model.startswith("litellm/"):
        model = f"litellm/{model}"
    return model


def setup_openai_client():
    """Set up the OpenAI client with priority: OpenAI native > LiteLLM."""
    set_tracing_disabled(True)
    model = get_model_name()
    # Try OpenAI native integration first
    if os.environ.get("OPENAI_API_KEY"):
        client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL"),  # Optional custom base URL
        )
        set_default_openai_client(client)
        console.print(
            Panel(
                f"[green]✅ Using OpenAI {model}[/green]",
                title="Model Configuration",
                border_style="green",
            )
        )
        return client

    # Fall back to LiteLLM integration
    # LiteLLM models use format: "litellm/provider/model-name"
    console.print(
        Panel(
            f"[green]✅ Using LiteLLM {model}[/green]",
            title="Model Configuration",
            border_style="green",
        )
    )
    return None
