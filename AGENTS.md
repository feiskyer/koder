# CLAUDE.md

This file provides guidance to KODER and Agentic AI when working with code in this repository.

## Commands

### Development Setup

```bash
# Install dependencies
uv sync

# Upgrade dependencies
uv sync --upgrade

# Run the CLI in interactive mode
uv run koder

# Run with a single prompt
uv run koder "Help me implement a new feature"

# Run with specific session
uv run koder --session my-session "Your prompt"

# Enable streaming mode (default)
uv run koder --stream "Your prompt"

# Disable streaming mode
uv run koder --no-stream "Your prompt"

# Resume a previous session
uv run koder --resume

# MCP server management
uv run koder mcp list
uv run koder mcp add myserver command arg1 arg2
uv run koder mcp remove myserver
```

### Development Commands

```bash
# Code formatting
black .

# Linting
ruff check --fix

# pylint
pylint koder_agent/ --disable=C,R,W --errors-only
```

## Architecture

### Core Components

**Agent Scheduler (`koder_agent/core/scheduler.py`)**: Central orchestrator that manages agent execution, handles streaming with Rich Live displays, and coordinates context management. Uses semaphores for concurrency control and includes intelligent cleanup of streaming content.

**Context Manager (`koder_agent/core/context.py`)**: Manages conversation history with SQLite storage at `~/.koder/koder.db`, implements token-aware compression (50k token limit using tiktoken), and handles session management across multiple projects.

**Tool Engine (`koder_agent/tools/engine.py`)**: Registers and executes tools with Pydantic validation, filters sensitive information from outputs, enforces security checks via SecurityGuard, and maintains an allowed tools list.

**Permission Manager (`koder_agent/core/permissions.py`)**: Handles tool execution permissions and approval workflows, supporting both automatic and interactive approval modes.

**MCP Server Manager (`koder_agent/mcp/server_manager.py`)**: Manages Model Context Protocol (MCP) server configurations stored in SQLite, supporting stdio, SSE, and HTTP transports.

### Agent System

The project uses the `openai-agents` library with multi-provider support:

- **Main Agent**: `create_dev_agent()` - Primary development agent with full tool access and MCP integration
- **Model Selection**: Automatically chooses appropriate model via `get_model_name()` supporting OpenAI, Anthropic Claude, Google Gemini, Azure OpenAI, GitHub Copilot, and 100+ providers via LiteLLM
- **Provider Detection**: Auto-detects provider based on environment variables with fallback defaults

### Tool Categories

1. **File Operations**: `read_file`, `write_file`, `append_file`, `list_directory`
2. **Search Operations**: `glob_search`, `grep_search`
3. **Shell Operations**: `run_shell`, `git_command`
4. **Web Operations**: `web_search`, `web_fetch`
5. **Task Management**: `todo_read`, `todo_write`, `task_delegate`

### Key Features

- **Streaming Support**: Real-time output with Rich Live displays and intelligent terminal cleanup
- **Context Persistence**: Conversation history across sessions with automatic compression
- **Tool Validation**: Pydantic schemas with security checks and output filtering
- **Interactive CLI**: Rich-formatted panels, prompts, and session management
- **MCP Integration**: Support for Model Context Protocol servers with multiple transport types
- **Multi-Provider AI**: Universal provider support with intelligent auto-detection
- **Session Management**: Per-project session isolation with resume capability

### Configuration

- **Context Loading**: The CLI looks for `AGENTS.md` in the working directory to load project-specific context
- **Database**: SQLite database at `~/.koder/koder.db` stores conversation history and MCP server configs
- **AI Provider Setup**: Set API credentials via environment variables (see README.md for provider-specific setup)
- **Model Selection**: Use `KODER_MODEL` environment variable to specify which model to use across providers

### Security Features

- **Command Validation**: SecurityGuard validates shell commands before execution
- **Output Filtering**: Automatic filtering of API keys, tokens, and sensitive information
- **Permission System**: Tool execution requires permission checks, with approval hooks for interactive mode
- **Input Sanitization**: Validation of all tool inputs using Pydantic schemas

### Entry Points

- **CLI Entry**: `koder_agent.cli:run` - Main CLI interface with argument parsing and session management
- **Interactive Mode**: `koder_agent/core/interactive.py` - Rich-based interactive prompt with command completion
- **Slash Commands**: `koder_agent/core/commands.py` - Built-in commands for session management and utilities