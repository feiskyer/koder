# Koder

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An intuitive AI coding assistant and interactive CLI tool that boosts developer productivity with intelligent automation and context-aware support.

## 🚀 Why this project

Yet another vibe coding assistant that aims to provide:

- Universal AI provider support - Works with OpenAI, Claude, Gemini, and 100+ providers through intelligent auto-detection.
- Persistent context - Remembers your conversations across sessions with smart token management.
- Rich toolset - File operations, search, shell commands, and web access in one unified interface.
- Zero-config start - Just set your API key and go, with automatic model selection and streaming support.
- Session management - Organize work by project with isolated conversation histories.

## 📋 Requirements

- Python 3.9 or higher.
- API key and optional baseURL from OpenAI, Gemini, Anthropic or other AI providers.

## 🛠️ Installation

### Using uv (Recommended)

```sh
uv tool install koder
```

### Using pip

```bash
pip install koder
```

## 🤖 AI Provider Configuration

Koder automatically detects and uses the LLM provider from environment variables.

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
export GEMINI_API_KEY="your-openai-api-key"

# Specify model (default: gemini/gemini-2.5-pro)
export KODER_MODEL="gemini/gemini-2.5-pro"
```

</details>

<details>

<summary>Azure OpenAI</summary>

```bash
# Required
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export KODER_MODEL="gpt-4o"  # Your Azure deployment model

# Optional
export AZURE_OPENAI_DEPLOYMENT="your-deployment-name"  # Defaults to KODER_MODEL
export AZURE_OPENAI_API_VERSION="2025-04-01-preview"   # Default version
```

</details>

<details>

<summary>Other AI providers (via LiteLLM)</summary>

[LiteLLM](https://docs.litellm.ai/docs/providers) supports 100+ providers including Anthropic, Google, Cohere, Hugging Face, and more:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-key"
export KODER_MODEL="claude-opus-4-20250514"

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



## ⚡ Quick Start

### Basic Usage

```bash
# Run in interactive mode
koder

# Execute a single prompt
koder -s my-project "Help me implement a new feature"

# Use a specific session
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

# Type checking
mypy .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run formating and linting: `black . && ruff check .`
5. Commit your changes: `git commit -am 'Add your feature'`
6. Push to the branch: `git push origin feature/your-feature`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
