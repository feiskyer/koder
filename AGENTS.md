# AGENTS.md

This file provides guidance to KODER and Agentic AI when working with code in this repository.

## Commands

### Development Setup

```bash
uv sync                                    # Install dependencies
uv run koder                               # Run in interactive mode
uv run koder "Your prompt"                 # Single prompt
uv run koder -s my-session "Your prompt"   # Named session
uv run koder --resume                      # Resume previous session
```

### Code Quality

```bash
uv run black .                                          # Format
uv run ruff check --fix                                 # Lint with auto-fix
uv run pylint koder_agent/ --disable=C,R,W --errors-only  # Error-only check
```

### Testing

```bash
uv run pytest                          # All tests
uv run pytest tests/test_file_tools.py # Single file
uv run pytest -v -k "test_name"        # Single test by name
```

### MCP Server Management

```bash
uv run koder mcp list                              # List servers
uv run koder mcp add myserver "python -m server"   # Add server
uv run koder mcp remove myserver                   # Remove server
```

## Architecture

Koder is a terminal-based AI coding assistant built on the `openai-agents` library with multi-provider support via LiteLLM.

### Package Structure

```md
koder_agent/
├── agentic/        # Agent creation, hooks, guardrails, approval system
├── cli.py          # Main CLI entry point
├── config/         # Configuration management (YAML, env vars)
├── core/           # Scheduler, context, streaming, security, interactive mode, commands
├── mcp/            # Model Context Protocol server integration (stdio, SSE, HTTP)
├── tools/          # Tool implementations (file, search, shell, web, task, todo, skill)
└── utils/          # Client setup, prompts, sessions, model info, terminal theme
```

### Core Flow

1. **CLI Entry** (`cli.py`) → parses args, initializes session
2. **AgentScheduler** (`core/scheduler.py`) → orchestrates execution with streaming and usage tracking
3. **Agent Creation** (`agentic/agent.py`) → builds agent with tools, MCP servers, model settings
4. **Tool Engine** (`tools/engine.py`) → registers tools, validates inputs, filters sensitive output
5. **Context Manager** (`core/context.py`) → persists conversations in SQLite with token-aware compression

### Key Design Patterns

- **Provider Abstraction**: `utils/client.py` detects providers from environment; uses native OpenAI client for OpenAI/Azure, LiteLLM wrapper for others
- **RetryingLitellmModel**: `agentic/agent.py` wraps LiteLLM with exponential backoff retry (3-5 attempts) for rate limits and transient errors
- **Progressive Disclosure Skills**: `tools/skill.py` loads skill metadata at startup (Level 1), full content on-demand (Level 2), saving 90%+ tokens
- **Skill Restrictions**: `tools/skill_context.py` + `agentic/skill_guardrail.py` limit tool access when specific skills are active
- **Streaming Display**: `core/streaming_display.py` manages Rich Live displays for real-time output
- **Approval Hooks**: `agentic/approval_hooks.py` wraps tool execution with permission checks
- **Security Guard**: `core/security.py` validates shell commands before execution
- **Background Shells**: `tools/shell.py` `BackgroundShellManager` tracks async shell commands

### Tool Categories

| Category | Tools                                                                   |
|----------|-------------------------------------------------------------------------|
| File     | `read_file`, `write_file`, `append_file`, `edit_file`, `list_directory` |
| Search   | `glob_search`, `grep_search`                                            |
| Shell    | `run_shell`, `shell_output`, `shell_kill`, `git_command`                |
| Web      | `web_search`, `web_fetch`                                               |
| Task     | `task_delegate`, `todo_read`, `todo_write`                              |
| Skills   | `get_skill`                                                             |

### Configuration Priority

CLI Arguments > Environment Variables > Config File (`~/.koder/config.yaml`) > Defaults

Key environment variables:

- `KODER_MODEL` - Model name (e.g., `gpt-4o`, `claude-opus-4-20250514`, `github_copilot/gpt-5.1-codex`)
- `KODER_REASONING_EFFORT` - Reasoning effort for o1/o3/gpt-5.1 models (`low`, `medium`, `high`)
- Provider API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GITHUB_TOKEN`, etc.

### Database

SQLite at `~/.koder/koder.db` stores:

- Conversation history with token-aware compression (50k token limit via tiktoken)
- Session metadata with auto-generated titles
- MCP server configurations

### Project Context

The CLI loads `AGENTS.md` from the working directory as project-specific context for the agent.

### Skills System

Skills are loaded from `.koder/skills/` (project) and `~/.koder/skills/` (user). Each skill has a `SKILL.md` with YAML frontmatter defining `name`, `description`, and optional `allowed_tools`.
