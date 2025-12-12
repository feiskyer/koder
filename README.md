# Koder

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![PyPI Downloads](https://static.pepy.tech/badge/koder)](https://pepy.tech/projects/koder)

Koder is an experimental, universal AI coding assistant designed to explore how to build an advanced terminal-based AI coding assistant. Written entirely in Python, it serves as both a functional tool and a learning playground for AI agent development.

**🎯 Project Status**: Alpha! This is a learning-focused project where we explore building AI coding agents.

## ✨ Features

- **🤖 Universal AI Support**: Works with OpenAI, Anthropic, Google, GitHub Copilot, and 100+ providers via LiteLLM with intelligent auto-detection
- **💾 Smart Context Management**: Persistent sessions with SQLite storage and automatic token-aware compression (50k token limit)
- **🔄 Real-time Streaming**: Rich Live displays with intelligent terminal cleanup for responsive user experience
- **🛠️ Comprehensive Toolset**: file operations, search, shell, task delegation, todos, and skills
- **📚 Progressive Disclosure Skills**: Load specialized knowledge on-demand with 90%+ token savings
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
koder -s my-project "Your prompt here"

# Use high reasoning effort for complex problems (OpenAI reasoning models)
koder --reasoning high "Solve this complex algorithm problem"

# Use low reasoning for simple tasks
koder --reasoning low "Add a print statement"
```

## 🤖 Configuration

Koder supports flexible configuration through three mechanisms (in order of priority):

1. **CLI Arguments** - Highest priority, for runtime overrides
2. **Environment Variables** - For secrets and runtime configuration
3. **Config File** - For persistent defaults (`~/.koder/config.yaml`)

### Quick Setup

```bash
# Minimal setup - just set your API key and go
export OPENAI_API_KEY="your-api-key"
koder

# Or use a different provider
export ANTHROPIC_API_KEY="your-api-key"
export KODER_MODEL="claude-opus-4-20250514"
koder
```

### Config File

Koder uses a YAML config file at `~/.koder/config.yaml` for persistent settings.

#### Config File Format

```yaml
# ~/.koder/config.yaml

# Model configuration
model:
  name: "gpt-4.1"              # Model name (default: gpt-4.1)
  provider: "openai"           # Provider name (default: openai)
  api_key: null                # API key (prefer env vars for security)
  base_url: null               # Custom API endpoint (optional)

  # Reasoning effort for OpenAI reasoning models (o1, o3, gpt-5.1, etc.)
  reasoning_effort: null       # none, minimal, low, medium, high, or null (default: null)

# CLI defaults
cli:
  session: null                # Default session name (auto-generated if null)
  stream: true                 # Enable streaming output (default: true)

# MCP servers for extended functionality
mcp_servers: []
```

### Environment Variables

#### Core Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `KODER_MODEL` | Model selection (highest priority) | `gpt-4o`, `claude-opus-4-20250514` |
| `KODER_REASONING_EFFORT` | Reasoning effort for reasoning models | `medium`, `high`, `low`, `null` |
| `EDITOR` | Editor for `koder config edit` | `vim`, `code` |

#### Provider API Keys

| Provider | API Key Variable | Additional Variables |
|----------|------------------|---------------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Anthropic | `ANTHROPIC_API_KEY` | - |
| Google/Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | - |
| Azure | `AZURE_API_KEY` | `AZURE_API_BASE`, `AZURE_API_VERSION` |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` | `VERTEXAI_LOCATION` |
| GitHub Copilot | `GITHUB_TOKEN` | - |
| Groq | `GROQ_API_KEY` | - |
| Together AI | `TOGETHERAI_API_KEY` | - |
| OpenRouter | `OPENROUTER_API_KEY` | - |
| Mistral | `MISTRAL_API_KEY` | - |
| Cohere | `COHERE_API_KEY` | - |
| Bedrock | `AWS_ACCESS_KEY_ID` | `AWS_SECRET_ACCESS_KEY` |

### Supported Providers

<details>
<summary><b>OpenAI</b></summary>

```bash
export OPENAI_API_KEY=your-api-key
export KODER_MODEL="gpt-4o"  # Optional, default: gpt-4.1

# Optional: Custom endpoint
export OPENAI_BASE_URL=https://your-endpoint.com/v1

koder
```

</details>

<details>
<summary><b>Anthropic</b></summary>

```bash
export ANTHROPIC_API_KEY=your-api-key
export KODER_MODEL="claude-opus-4-20250514"
koder
```

</details>

<details>
<summary><b>Google Gemini</b></summary>

```bash
export GOOGLE_API_KEY=your-api-key
export KODER_MODEL="gemini/gemini-2.5-pro"
koder
```

</details>

<details>
<summary><b>GitHub Copilot</b></summary>

```bash
export KODER_MODEL="github_copilot/claude-sonnet-4"
koder
```

On first run you will see a device code in the terminal. Visit <https://github.com/login/device> and enter the code to authenticate.

</details>

<details>
<summary><b>Azure OpenAI</b></summary>

```bash
export AZURE_API_KEY="your-azure-api-key"
export AZURE_API_BASE="https://your-resource.openai.azure.com"
export AZURE_API_VERSION="2025-04-01-preview"
export KODER_MODEL="azure/gpt-4"
koder
```

Or configure in `~/.koder/config.yaml`:

```yaml
model:
  name: "gpt-4"
  provider: "azure"
  azure_api_version: "2025-04-01-preview"
```

</details>

<details>
<summary><b>Google Vertex AI</b></summary>

```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
export VERTEXAI_LOCATION="us-central1"
export KODER_MODEL="vertex_ai/claude-sonnet-4@20250514"
koder
```

Or configure in `~/.koder/config.yaml`:

```yaml
model:
  name: "claude-sonnet-4@20250514"
  provider: "vertex_ai"
  vertex_ai_location: "us-central1"
  vertex_ai_credentials_path: "path/to/service-account.json"
```

</details>

<details>
<summary><b>Other Providers (100+ via LiteLLM)</b></summary>

[LiteLLM](https://docs.litellm.ai/docs/providers) supports 100+ providers. Use the format `provider/model`:

```bash
# Groq
export GROQ_API_KEY=your-key
export KODER_MODEL="groq/llama-3.3-70b-versatile"

# Together AI
export TOGETHERAI_API_KEY=your-key
export KODER_MODEL="together_ai/meta-llama/Llama-3-70b-chat-hf"

# OpenRouter
export OPENROUTER_API_KEY=your-key
export KODER_MODEL="openrouter/anthropic/claude-3-opus"

# Custom OpenAI-compatible endpoints
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://your-custom-endpoint.com/v1"
export KODER_MODEL="openai/your-model-name"

koder
```

</details>

### MCP Server Configuration

Model Context Protocol (MCP) servers extend Koder's capabilities with additional tools.

#### MCP CLI Commands

```bash
# Add an MCP server (stdio transport)
koder mcp add myserver "python -m my_mcp_server" --transport stdio

# Add with environment variables
koder mcp add myserver "python -m server" -e API_KEY=xxx -e DEBUG=true

# Add HTTP/SSE server
koder mcp add webserver --transport http --url http://localhost:8000

# List all MCP servers
koder mcp list

# Get server details
koder mcp get myserver

# Remove a server
koder mcp remove myserver
```

#### MCP Config Format

```yaml
# In ~/.koder/config.yaml

mcp_servers:
  # stdio transport (runs a local command)
  - name: "filesystem"
    transport_type: "stdio"
    command: "python"
    args: ["-m", "mcp.server.filesystem"]
    env_vars:
      ROOT_PATH: "/home/user/projects"
    cache_tools_list: true
    allowed_tools:          # Optional: whitelist specific tools
      - "read_file"
      - "write_file"

  # HTTP transport (connects to remote server)
  - name: "web-tools"
    transport_type: "http"
    url: "http://localhost:8000"
    headers:
      Authorization: "Bearer token123"

  # SSE transport (server-sent events)
  - name: "streaming-server"
    transport_type: "sse"
    url: "http://localhost:9000/sse"
```

### Skills

Skills provide specialized knowledge and guidance that Koder can load on-demand. This uses a **Progressive Disclosure** pattern to minimize token usage - only skill metadata is loaded at startup, with full content fetched when needed.

#### Skills Directory Structure

Skills are loaded from two locations (project skills take priority):

1. **Project skills**: `.koder/skills/` in your current directory
2. **User skills**: `~/.koder/skills/` for personal skills

Each skill lives in its own directory with a `SKILL.md` file:

```
.koder/skills/
├── api-design/
│   └── SKILL.md
├── code-review/
│   ├── SKILL.md
│   └── checklist.md    # Supplementary resource
└── testing/
    └── SKILL.md
```

#### Creating a Skill

Create a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: api-design
description: Best practices for designing RESTful APIs
allowed_tools:
  - read_file
  - write_file
---

# API Design Guidelines

## RESTful Principles

Use nouns for resources, HTTP verbs for actions...

## Versioning

Always version your APIs using URL path (`/v1/users`)...

## Error Handling

Return consistent error responses with status codes...
```

#### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier |
| `description` | Yes | Brief description (shown in metadata) |
| `allowed_tools` | No | Tools the skill recommends using |

#### How Skills Work

1. **Startup**: Only skill names and descriptions are loaded (Level 1 - minimal tokens)
2. **On-demand**: When Koder needs a skill, it calls `get_skill(name)` to load full content (Level 2)
3. **Supplementary**: Skills can reference additional files that Koder reads with `read_file` (Level 3)

This progressive approach saves **90%+ tokens** compared to loading all skill content at startup.

#### Skills Configuration

```yaml
# ~/.koder/config.yaml
skills:
  enabled: true                        # Enable/disable skills (default: true)
  project_skills_dir: ".koder/skills"  # Project skills location
  user_skills_dir: "~/.koder/skills"   # User skills location
```

### Example Configurations

<details>
<summary><b>Minimal (OpenAI)</b></summary>

```yaml
# ~/.koder/config.yaml
model:
  name: "gpt-4o"
  provider: "openai"
```

```bash
export OPENAI_API_KEY="sk-..."
koder
```

</details>

<details>
<summary><b>Enterprise Azure Setup</b></summary>

```yaml
# ~/.koder/config.yaml
model:
  name: "gpt-4"
  provider: "azure"
  azure_api_version: "2025-04-01-preview"

cli:
  session: "enterprise-project"
  stream: true

mcp_servers:
  - name: "company-tools"
    transport_type: "http"
    url: "https://internal-mcp.company.com"
    headers:
      X-API-Key: "${COMPANY_API_KEY}"
```

```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com"
koder
```

</details>

<details>
<summary><b>Multi-Provider Development</b></summary>

```yaml
# ~/.koder/config.yaml - set a default
model:
  name: "gpt-4o"
  provider: "openai"
```

```bash
# Override at runtime with KODER_MODEL
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Use default (OpenAI)
koder

# Switch to Claude for specific tasks
KODER_MODEL="claude-opus-4-20250514" koder "complex reasoning task"
```

</details>

### Configuration Priority

When the same setting is defined in multiple places, the priority is:

```
CLI Arguments  >  Environment Variables  >  Config File  >  Defaults
```

**Example:**

```yaml
# ~/.koder/config.yaml
model:
  name: "gpt-4o"
```

```bash
# Environment variable overrides config file
export KODER_MODEL="claude-opus-4-20250514"
koder  # Uses claude-opus-4-20250514
```

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
