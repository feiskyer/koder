# Configuration Guide

Koder supports flexible configuration through three mechanisms (in order of priority):

1. **CLI Arguments** - Highest priority, for runtime overrides
2. **Environment Variables** - For secrets and runtime configuration
3. **Config File** - For persistent defaults (`~/.koder/config.yaml`)

## Table of Contents

- [Config File](#config-file)
- [Environment Variables](#environment-variables)
- [Provider Setup](#provider-setup)
- [MCP Servers](#mcp-servers)
- [Skills](#skills)
- [Example Configurations](#example-configurations)

## Config File

Koder uses a YAML config file at `~/.koder/config.yaml` for persistent settings.

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

## Environment Variables

### Core Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `KODER_MODEL` | Model selection (highest priority) | `gpt-4o`, `claude-opus-4-20250514` |
| `KODER_REASONING_EFFORT` | Reasoning effort for reasoning models | `medium`, `high`, `low`, `null` |
| `EDITOR` | Editor for `koder config edit` | `vim`, `code` |

### Provider API Keys

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

## Provider Setup

### OpenAI

```bash
export OPENAI_API_KEY=your-api-key
export KODER_MODEL="gpt-4o"  # Optional, default: gpt-4.1

# Optional: Custom endpoint
export OPENAI_BASE_URL=https://your-endpoint.com/v1

koder
```

### Anthropic

```bash
export ANTHROPIC_API_KEY=your-api-key
export KODER_MODEL="claude-opus-4-20250514"
koder
```

### Google Gemini

```bash
export GOOGLE_API_KEY=your-api-key
export KODER_MODEL="gemini/gemini-2.5-pro"
koder
```

### GitHub Copilot

```bash
export KODER_MODEL="github_copilot/claude-sonnet-4"
koder
```

On first run you will see a device code in the terminal. Visit <https://github.com/login/device> and enter the code to authenticate.

### Azure OpenAI

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

### Google Vertex AI

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

### Other Providers (100+ via LiteLLM)

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

## MCP Servers

Model Context Protocol (MCP) servers extend Koder's capabilities with additional tools.

### CLI Commands

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

### Config Format

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

## Skills

Skills provide specialized knowledge and guidance that Koder can load on-demand. This uses a **Progressive Disclosure** pattern to minimize token usage - only skill metadata is loaded at startup, with full content fetched when needed.

### Directory Structure

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

### Creating a Skill

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

### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier |
| `description` | Yes | Brief description (shown in metadata) |
| `allowed_tools` | No | Tools the skill recommends using |

### How Skills Work

1. **Startup**: Only skill names and descriptions are loaded (Level 1 - minimal tokens)
2. **On-demand**: When Koder needs a skill, it calls `get_skill(name)` to load full content (Level 2)
3. **Supplementary**: Skills can reference additional files that Koder reads with `read_file` (Level 3)

This progressive approach saves **90%+ tokens** compared to loading all skill content at startup.

### Configuration

```yaml
# ~/.koder/config.yaml
skills:
  enabled: true                        # Enable/disable skills (default: true)
  project_skills_dir: ".koder/skills"  # Project skills location
  user_skills_dir: "~/.koder/skills"   # User skills location
```

## Example Configurations

### Minimal (OpenAI)

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

### Enterprise Azure Setup

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

### Multi-Provider Development

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

## Configuration Priority

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
