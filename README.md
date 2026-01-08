# Koder

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/koder)](https://pepy.tech/projects/koder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An experimental, universal AI coding assistant for the terminal. Built in Python, Koder works with OpenAI, Anthropic, Google, GitHub Copilot, and 100+ providers via LiteLLM.

🎯 **Status**: Alpha - This is a learning-focused project exploring AI agent development.

## Features

- **Universal AI Support** - Works with ChatGPT/Gemini/Claude subscriptions and API keys via OpenAI, Anthropic, Google, GitHub Copilot, and 100+ providers
- **Smart Context** - Persistent sessions with SQLite storage and automatic token-aware compression
- **Real-time Streaming** - Rich terminal displays with live output
- **Comprehensive Tools** - File operations, search, shell, task delegation, todos, and skills
- **MCP Integration** - Extensible tool ecosystem via Model Context Protocol
- **Zero Config** - Automatic provider detection with sensible defaults

## Installation

### Using uv (Recommended)

```bash
uv tool install koder
```

### Using pip

```bash
pip install koder
```

## Quick Start

```bash
# 1. Set your API key (works with any provider)
export KODER_API_KEY="your-api-key"

# 2. Run Koder
koder
```

That's it! `KODER_API_KEY` works with any provider - no need to remember provider-specific variable names.

### Basic Usage

```bash
# Interactive mode
koder

# Single prompt
koder "create a Python function to calculate fibonacci numbers"

# Named session (persists conversation)
koder -s my-project "help me implement a new feature"

# Use a different model
KODER_MODEL="claude-opus-4-20250514" koder "your prompt"
```

## Configuration

Koder can be configured via (in priority order):

1. **CLI arguments** - Highest priority
2. **Environment variables** - Universal `KODER_*` vars override everything
3. **Config file** - `~/.koder/config.yaml`

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `KODER_API_KEY` | Universal API key (works with any provider) | `sk-...` |
| `KODER_MODEL` | Model to use | `gpt-4o`, `claude-opus-4-20250514` |
| `KODER_BASE_URL` | Custom API endpoint | `http://localhost:8080/v1` |
| `KODER_REASONING_EFFORT` | For reasoning models | `low`, `medium`, `high` |

`KODER_API_KEY` and `KODER_BASE_URL` take priority over provider-specific variables (like `OPENAI_API_KEY`) and config file settings.

### Providers

Use `KODER_API_KEY` for any provider, or provider-specific variables:

| Provider | Environment Variable | Model Example |
|----------|---------------------|---------------|
| Any | `KODER_API_KEY` | Works with all providers |
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4.1` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-20250514` |
| Google | `GOOGLE_API_KEY` | `gemini/gemini-2.5-pro` |
| GitHub Copilot | *(device auth)* | `github_copilot/claude-sonnet-4` |
| Azure | `AZURE_API_KEY` | `azure/gpt-4` |

See [Configuration Guide](docs/configuration.md) for all 100+ providers.

### OAuth Providers (Subscription-Based)

Use your existing subscriptions (Claude Max, ChatGPT Plus/Pro, Google Gemini) without API keys:

```bash
# Authenticate with a provider
koder auth login google      # Google/Gemini CLI (free with Google account)
koder auth login claude      # Claude Max subscription
koder auth login chatgpt     # ChatGPT Plus/Pro subscription
koder auth login antigravity # Antigravity (Gemini 3 + Claude models)

# Check authentication status
koder auth list              # List configured providers
koder auth status            # Show detailed token status

# Revoke access
koder auth revoke google     # Remove stored tokens
```

**OAuth Providers:**

| OAuth Provider | Subscription | Description |
|----------------|--------------|-------------|
| `google` | Google Gemini | Access to Gemini models via Google account |
| `claude` | Claude Pro/Max | Access to Claude models via subscription |
| `chatgpt` | ChatGPT Plus/Pro | Access to GPT models via subscription |
| `antigravity` | Antigravity | Access to Gemini 3 + Claude models |

Available models are fetched from each provider's API after login and cached locally (1 day TTL).
Use `koder auth list` to see all available models for your authenticated providers.

**Usage after authentication:**

```bash
# Use OAuth-authenticated models (OAuth provider prefix required)
KODER_MODEL="google/gemini-3-pro-preview" koder "your prompt"                # Google OAuth
KODER_MODEL="claude/claude-opus-4-5-20250514" koder "your prompt"            # Claude Max
KODER_MODEL="chatgpt/gpt-5.2" koder "your prompt"                            # ChatGPT OAuth
KODER_MODEL="antigravity/antigravity-gemini-3-pro-high" koder "your prompt"  # Antigravity
```

OAuth tokens and cached models are stored in `~/.koder/tokens/` and automatically refreshed before expiry.

### Config File Example

```yaml
# ~/.koder/config.yaml

model:
  name: "gpt-4o"
  provider: "openai"
  reasoning_effort: null    # For reasoning models: low, medium, high

cli:
  session: null             # Default session name
  stream: true              # Enable streaming output

mcp_servers: []             # MCP server configurations
```

### Commands

```bash
koder config show          # Show current config
koder config edit          # Edit config file
koder -s SESSION_NAME      # Use named session
```

## MCP Servers

Model Context Protocol (MCP) servers extend Koder with additional tools.

### CLI Commands

```bash
# Add servers
koder mcp add myserver "python -m my_mcp_server"
koder mcp add myserver "python -m server" -e API_KEY=xxx

# HTTP/SSE transport
koder mcp add webserver --transport http --url http://localhost:8000

# Manage servers
koder mcp list
koder mcp get myserver
koder mcp remove myserver
```

### Config Example

```yaml
# In ~/.koder/config.yaml
mcp_servers:
  # stdio transport (local command)
  - name: "filesystem"
    transport_type: "stdio"
    command: "python"
    args: ["-m", "mcp.server.filesystem"]
    env_vars:
      ROOT_PATH: "/home/user/projects"
    cache_tools_list: true
    allowed_tools:
      - "read_file"
      - "write_file"

  # HTTP transport (remote server)
  - name: "web-tools"
    transport_type: "http"
    url: "http://localhost:8000"
    headers:
      Authorization: "Bearer token123"

  # SSE transport
  - name: "streaming-server"
    transport_type: "sse"
    url: "http://localhost:9000/sse"
```

## Skills

Skills provide specialized knowledge loaded on-demand, saving 90%+ tokens via progressive disclosure.

### Directory Structure

Skills are loaded from (project skills take priority):

1. **Project**: `.koder/skills/`
2. **User**: `~/.koder/skills/`

```
.koder/skills/
├── api-design/
│   └── SKILL.md
├── code-review/
│   ├── SKILL.md
│   └── checklist.md
└── testing/
    └── SKILL.md
```

### Creating a Skill

Create a `SKILL.md` with YAML frontmatter:

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
```

### How Skills Work

1. **Startup**: Only skill names and descriptions are loaded (minimal tokens)
2. **On-demand**: Full content fetched when needed via `get_skill(name)`
3. **Supplementary**: Skills can reference additional files

## Architecture

```
koder_agent/
├── agentic/        # Agent creation, hooks, and approval system
├── cli.py          # Main CLI entry point
├── config/         # Configuration management
├── core/           # Scheduler, context, streaming, security
├── mcp/            # Model Context Protocol integration
├── tools/          # Tool implementations
└── utils/          # Helpers and utilities
```

### Core Flow

1. **CLI** (`cli.py`) parses arguments, initializes session
2. **AgentScheduler** (`core/scheduler.py`) orchestrates execution with streaming
3. **Agent** (`agentic/agent.py`) builds agent with tools, MCP servers, model settings
4. **Tools** (`tools/engine.py`) register tools, validate inputs, filter output
5. **Context** (`core/context.py`) persists conversations in SQLite

### Data Storage

- **Database**: `~/.koder/koder.db` (SQLite)
- **Config**: `~/.koder/config.yaml`
- **Skills**: `~/.koder/skills/` or `.koder/skills/`

## Development

### Setup

```bash
git clone https://github.com/feiskyer/koder.git
cd koder
uv sync
uv run koder
```

### Code Quality

```bash
uv run black .              # Format
uv run ruff check --fix     # Lint
uv run pytest               # Test
```

## Security

- **API Keys**: Stored in environment variables, never in code
- **Local Storage**: Sessions stored in `~/.koder/`
- **No Telemetry**: Only API requests to your chosen provider
- **Shell Commands**: Require explicit user confirmation

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with Python and curiosity</sub>
</p>
