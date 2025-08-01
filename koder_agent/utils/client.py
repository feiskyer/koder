import os

from agents import (
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()


def get_model_name():
    """Get the appropriate model name based on the configured client type."""
    model = os.environ.get("KODER_MODEL", "gpt-4.1")
    # 1. OpenAI native
    if os.environ.get("OPENAI_API_KEY"):
        return model

    # 2. Azure OpenAI - use deployment name
    azure_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    if all(os.environ.get(var) for var in azure_vars):
        return os.environ.get("AZURE_OPENAI_DEPLOYMENT", model)

    # 3. LiteLLM - use litellm format like "litellm/gemini/gemini-2.5-pro"
    if model == "gpt-4.1":
        model = "gemini/gemini-2.5-pro"
    if not model.startswith("litellm/"):
        model = f"litellm/{model}"
    return model


def setup_openai_client():
    """Set up the OpenAI client with priority: OpenAI native > Azure > LiteLLM."""
    set_tracing_disabled(True)
    model = get_model_name()
    # 1. Try OpenAI native integration first
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

    # 2. Try Azure OpenAI integration
    azure_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    if all(os.environ.get(var) for var in azure_vars):
        client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT", model),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )
        set_default_openai_client(client)
        console.print(
            Panel(
                f"[green]✅ Using Azure OpenAI {model}[/green]",
                title="Model Configuration",
                border_style="green",
            )
        )
        return client

    # 3. Fall back to LiteLLM integration
    # LiteLLM models use format: "litellm/provider/model-name"
    console.print(
        Panel(
            f"[green]✅ Using LiteLLM {model}[/green]",
            title="Model Configuration",
            border_style="green",
        )
    )
    return None
