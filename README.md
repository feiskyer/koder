# Koder

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![PyPI Downloads](https://static.pepy.tech/badge/koder)](https://pepy.tech/projects/koder)

Koder is an experimental, universal AI coding assistant designed to explore how to build an advanced terminal-based AI coding assistant. Written entirely in Python, it serves as both a functional tool and a learning playground for AI agent development.

**🎯 Project Status**: Under active vibe coding! This is a learning-focused project where we explore building AI coding agents.

## ✨ Features

- **🤖 Universal AI Support**: Works with OpenAI, Anthropic, Google, GitHub Copilot, and 100+ providers via LiteLLM with intelligent auto-detection
- **💾 Smart Context Management**: Persistent sessions with SQLite storage and automatic token-aware compression (50k token limit)
- **🔄 Real-time Streaming**: Rich Live displays with intelligent terminal cleanup for responsive user experience
- **🛠️ Comprehensive Toolset**: file operations, search, shell, task delegation and todos.
- **🔌 MCP Integration**: Model Context Protocol support with stdio, SSE, and HTTP transports for extensible tool ecosystem
- **🛡️ Enterprise Security**: SecurityGuard validation, output filtering, permission system, and input sanitization
- **🎯 Zero Configuration**: Automatic provider detection with fallback defaults

## 🛠️ Installation

### Using uv (Recommended)

```sh
uv tool install koder
```

### Using pip

```bash
pip install koder
```

## ⚡ Quick Start

Simply run Koder with your question or request:

```bash
# Configure one provider (example: OpenAI)
export OPENAI_API_KEY="your-openai-api-key"
export KODER_MODEL="gpt-4o"

# Run in interactive mode
koder

# Run with prompt
koder "create a Python function to calculate fibonacci numbers"

# Execute a single prompt in a named session
koder -s my-project "Help me implement a new feature"

# Use an explicit session flag
koder --session my-project "Your prompt here"
```

## 🤖 Configuration

### Environment Variables

Koder automatically detects your AI provider based on available environment variables. The `KODER_MODEL` environment variable controls which model to use:

```bash
# OpenAI models
export KODER_MODEL="gpt-4.1"

# Claude models (via LiteLLM)
export KODER_MODEL="claude-opus-4-20250514"

# Google models (via LiteLLM)
export KODER_MODEL="gemini/gemini-2.5-pro"

# Github Copilot (via LiteLLM)
export KODER_MODEL="github_copilot/claude-sonnet-4"
```

### Supported Providers

<details>
<summary><b>OpenAI</b></summary>

```bash
export OPENAI_API_KEY=your-api-key

# Optional: Use custom endpoint
export OPENAI_API_BASE=https://your-endpoint.com

# Optional: Specify model (default: gpt-4.1)
export KODER_MODEL="gpt-4o"
```

</details>

<details>
<summary><b>Anthropic</b></summary>

```bash
export ANTHROPIC_API_KEY=your-api-key
export KODER_MODEL="claude-opus-4-20250514"

```

</details>

<details>
<summary><b>Google (Gemini)</b></summary>

```bash
export GOOGLE_API_KEY=your-api-key
# Optional: Specify model (default: gemini/gemini-2.5-pro)
export KODER_MODEL="gemini/gemini-2.5-pro"
```

</details>

<details>
<summary><b>GitHub Copilot</b></summary>

```bash
export KODER_MODEL="github_copilot/claude-sonnet-4"
```

On first run you will see a device code in the terminal. Visit <https://github.com/login/device> and enter the code to authenticate.

</details>

<details>
<summary><b>Azure OpenAI</b></summary>

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
<summary><b>Other AI providers (via LiteLLM)</b></summary>

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

## 🛠️ Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/feiskyer/koder.git
cd koder

uv sync
uv run koder
```

### Code Quality

```bash
# Code formatting
black .

# Linting
ruff check --fix

# pylint
pylint koder_agent/ --disable=C,R,W --errors-only
```

## 🔒 Security

- **API Keys**: All API keys are stored in environment variables and never in code.
- **Local Storage**: Sessions are stored locally in your home directory.
- **No Telemetry**: Koder doesn't send any data besides API requests to your chosen provider.
- **Code Execution**: Shell commands require explicit user confirmation.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 🌐 Code of Conduct

This project follows a Code of Conduct based on the Contributor Covenant. Be kind and respectful. If you observe unacceptable behavior, please open an issue.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Use of third-party AI services is governed by their respective provider terms.
