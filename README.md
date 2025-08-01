# Koder

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![PyPI Downloads](https://static.pepy.tech/badge/koder)](https://pepy.tech/projects/koder)

An intuitive AI coding assistant and interactive CLI tool that boosts developer productivity with intelligent automation and context-aware support.

## 🚀 Overview

Koder is a universal, provider-agnostic CLI assistant. It supports OpenAI, Anthropic Claude, Google Gemini, Azure OpenAI, GitHub Copilot (device flow), and 100+ providers via LiteLLM.

- Universal provider support with intelligent auto-detection
- Persistent context across sessions with smart token management
- Rich toolset: file operations, search, shell commands, and web access
- Zero-config start: set one API key and go (streaming supported)
- Session management per project

## 📋 Requirements

- Python 3.9 or higher
- API key and optional base URL from your chosen AI provider

## 🛠️ Installation

### Using uv (Recommended)

```sh
uv tool install koder
```

### Using pip

```bash
pip install koder
```

## ⚡ Quick Start (Minimal)

```bash
# 1) Install
uv tool install koder

# 2) Configure one provider (example: OpenAI)
export OPENAI_API_KEY="your-openai-api-key"
export KODER_MODEL="gpt-4o"

# 3) Run
koder -s demo "Help me scaffold a FastAPI service with a /health endpoint"
```

## 🤖 Provider Configuration

Koder auto-detects the provider based on environment variables. If multiple providers are configured, KODER_MODEL is used to route requests. If KODER_MODEL is omitted, Koder picks a sensible default for the detected provider.

**Model Selection**

The `KODER_MODEL` environment variable controls which model to use:

```bash
# OpenAI models
export KODER_MODEL="gpt-4.1"

# Claude models (via LiteLLM)
export KODER_MODEL="claude-opus-4-20250514"

# Google models (via LiteLLM)
export KODER_MODEL="gemini/gemini-2.5-pro"
```

**AI Providers:**

<details>

<summary>OpenAI</summary>

```bash
# Required
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Custom OpenAI-compatible endpoint
export OPENAI_BASE_URL="https://api.openai.com/v1"  # Default

# Optional: Specify model (default: gpt-4.1)
export KODER_MODEL="gpt-4o"
```

</details>

<details>

<summary>Gemini</summary>

```bash
# Required
export GEMINI_API_KEY="your-gemini-api-key"

# Optional: Specify model (default: gemini/gemini-2.5-pro)
export KODER_MODEL="gemini/gemini-2.5-pro"
```

</details>

<details>

<summary>Anthropic Claude</summary>

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"
export KODER_MODEL="claude-opus-4-20250514"
```

</details>

<details>

<summary>GitHub Copilot</summary>

```bash
export KODER_MODEL="github_copilot/claude-sonnet-4"
```

On first run you will see a device code in the terminal. Visit <https://github.com/login/device> and enter the code to authenticate.

</details>

<details>

<summary>Azure OpenAI</summary>

```bash
# Required
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"

# Your Azure deployment name (often different from the base model name)
export AZURE_OPENAI_DEPLOYMENT="my-gpt4o-deployment"

# Choose a model that maps to your deployment
export KODER_MODEL="gpt-4o"

# Optional: API version
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"
```

Tips:

- Ensure the endpoint hostname matches your Azure resource.
- AZURE_OPENAI_DEPLOYMENT must match the deployed model name in Azure.

</details>

<details>

<summary>Other AI providers (via LiteLLM)</summary>

[LiteLLM](https://docs.litellm.ai/docs/providers) supports 100+ providers including Anthropic, Google, Cohere, Hugging Face, and more:

```bash
# Google Vertex AI
export GOOGLE_APPLICATION_CREDENTIALS="your-sa-path.json"
export VERTEXAI_LOCATION="<your-region>"
export KODER_MODEL="vertex_ai/claude-sonnet-4@20250514"

# Custom OpenAI-compatible endpoints
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"
export KODER_MODEL="openai/<your-model-name>"
```

</details>

## 🔐 Security

- Never commit API keys. Prefer shell profiles, direnv (.envrc), 1Password CLI, or CI secrets.
- Rotate keys regularly and scope permissions minimally.
- For CI, store secrets in the platform’s secret manager and inject at runtime.

## 📦 Usage Examples

```bash
# Run in interactive mode
koder

# Execute a single prompt in a named session
koder -s my-project "Help me implement a new feature"

# Use an explicit session flag
koder --session my-project "Your prompt here"

# Enable streaming mode
koder --stream "Your prompt here"
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone and setup
git clone https://github.com/feiskyer/koder.git
cd koder
uv sync

uv run koder
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run formatting and linting
5. Commit your changes: `git commit -am 'feat: add your feature'`
6. Push to the branch: `git push origin feature/your-feature`
7. Submit a pull request

## 🌐 Code of Conduct

This project follows a Code of Conduct based on the Contributor Covenant. Be kind and respectful. If you observe unacceptable behavior, please open an issue.

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

Use of third-party AI services is governed by their respective provider terms.
