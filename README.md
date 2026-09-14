# Koder

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyPI Downloads](https://static.pepy.tech/badge/koder)](https://pepy.tech/projects/koder)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Koder is an experimental, open-source AI coding assistant for the terminal. It combines a streaming TUI, persistent local sessions, durable goals and scheduled loops, repository-aware tools, extensible skills, MCP integrations, sandbox-aware permissions, and multi-agent workflows in one Python runtime.

Status: Alpha. Koder is a learning-focused project for exploring agentic coding systems. Expect rapid iteration and occasional sharp edges.

## Start Here

```bash
uv tool install koder
export KODER_API_KEY="your-api-key"
export KODER_MODEL="gpt-4o"
koder
```

Prefer a provider-specific key or subscription login? Jump to [Model Configuration](#model-configuration).

## Why Koder

Koder is designed for developers who want a local-first coding agent they can inspect, extend, and run across model providers without changing their workflow.

- Bring your own model: OpenAI, Anthropic, Google/Gemini, GitHub Copilot, Azure, OAuth-backed subscriptions, OpenRouter, and 100+ LiteLLM providers.
- Stay in the terminal: use slash commands, file mentions, shell mode, history search, live usage output, and optional voice dictation.
- Keep long work honest: set session goals with optional token budgets, pause or resume them, and schedule recurring prompts with local cron-backed loops.
- Keep runtime state local: sessions, transcripts, memories, task records, settings, skills, agents, and team state live under Koder-owned local paths. Model requests still go to the provider you configure.
- Extend the runtime: add project skills, user skills, plugins, MCP servers, channels, and Magic Docs.
- Delegate carefully: run background subagents and local teams while the main session stays responsible for integration.
- Study the system: the Python codebase is organized to make agent scheduling, provider routing, permissions, tools, and TUI behavior easy to trace.

## How It Fits

Koder is in the same broad family as terminal-first and open-source AI coding agents. Its emphasis is different:

| If you care about... | Koder emphasizes |
|---|---|
| Provider choice | Universal `KODER_*` variables, provider-specific keys, OAuth-backed subscriptions, custom base URLs, and LiteLLM provider routing. |
| Terminal ergonomics | A persistent Rich TUI with slash commands, shell mode, file mentions, usage/cost output, voice input, and resume. |
| Long-running work | Session goals, token budgets, automatic continuation, and cron-backed loop prompts. |
| Local inspectability | SQLite sessions, local goals, local memory, local settings, local tokens, and documented privacy boundaries. |
| Extension experiments | Skills, verifier skills, plugins, MCP servers, channels, Magic Docs, and project instructions. |
| Multi-agent workflows | Background subagents, local teams, tmux teammates, mailbox/task records, and team memory. |
| Safety research | Permission rules, managed settings, sandbox policy, backend diagnostics, and explicit local/remote data boundaries. |

## Highlights

| Area | What You Get |
|---|---|
| Interactive TUI | Streaming output, slash completion, shell mode, file mentions, status line, reverse history search, and multi-line prompts. |
| Model routing | Universal `KODER_*` variables, provider-specific keys, custom base URLs, reasoning effort, and subscription-backed OAuth providers. |
| Durable context | SQLite sessions, named sessions, goals, resume, export, compaction, rewind, thinkback, local memories, and AutoDream consolidation. |
| Coding tools | File operations, search, shell execution, git helpers, notebooks, web fetch/search, todos, and local code intelligence. |
| Workflows | Goals, scheduled loops, review, security review, advisor, planning, commit readiness, PR comments, GitHub Actions setup, release notes, and verification summaries. |
| Agents and teams | Project/user agents, `task_delegate`, `/fork`, `/peers`, in-process teammates, tmux teammates, mailbox routing, tasks, and team memory. |
| Extensions | Skills, verifier skills, plugins, MCP servers, channels, and Magic Docs. |
| Safety controls | Permission rules, sandbox policy, managed settings, workspace roots, privacy diagnostics, and local storage boundaries. |

## Installation

Install the published CLI with `uv`:

```bash
uv tool install koder
```

Or use `pip`:

```bash
pip install koder
```

For local development from this repository:

```bash
git clone https://github.com/feiskyer/koder.git
cd koder
uv sync
uv run koder
```

Requirements: Python 3.11 or newer and a model provider credential or subscription login.
Python 3.10 is no longer supported; task cancellation relies on Python 3.11's asyncio APIs.

## Quick Start

Set a model credential and start the interactive TUI:

```bash
export KODER_API_KEY="your-api-key"
export KODER_MODEL="gpt-4o"
koder
```

Run a single prompt from the shell:

```bash
koder "summarize the current repository"
```

Use a named session when you want durable context for a project or feature:

```bash
koder -s billing-refactor
koder -s billing-refactor "continue the failing test investigation"
koder --resume
```

Good first commands inside the TUI:

```bash
/onboarding
/status
/model
/files
/permissions
/help
```

Good first prompts:

```text
Inspect this repository and explain the test command.
Find the smallest safe fix for the failing test, then run the focused test.
Review the current diff for correctness and missing verification.
```

## Common Usage

| Task | Command |
|---|---|
| Open the TUI | `koder` |
| Run one prompt | `koder "fix the failing test"` |
| Print script-friendly output | `koder --print "summarize"` |
| Use a named session | `koder -s my-project` |
| Resume previous work | `koder --resume` or `koder --continue` |
| Inspect runtime state | `/status`, `/summary`, `/stats`, `/doctor` |
| Inspect context | `/files`, `/context`, `/ctx_viz` |
| Review changes | `/diff`, `/review`, `/security-review` |
| Check usage and cost | `/usage`, `/cost` |
| Manage agents and teams | `/agents`, `/fork`, `/peers`, `/tasks` |
| Manage goals and loops | `/goal`, `/loop`, `/schedule` |
| Manage extensions | `/skills`, `/plugin`, `/mcp`, `/channels` |
| Manage permissions | `/permissions`, `/sandbox`, `/add-dir` |

See the [Command Reference](docs/commands.md) for the complete slash-command catalog.

## Trust Boundaries

Koder is local-first, not offline-only:

- Koder stores product state under `~/.koder/` and project `.koder/` paths by default.
- Koder does not upload sessions to a Koder-hosted service.
- Prompts, selected context, and tool results needed for a turn are sent to the model provider you configure.
- Goals are stored with local session state, and scheduled loop prompts are stored locally before they are run.
- MCP servers, plugins, shell commands, and teammate processes can access whatever their local permissions allow unless constrained by Koder policy and sandbox settings.
- OAuth tokens are stored under `~/.koder/tokens/`; avoid committing secrets into project files.

Inspect the current boundary with:

```bash
/privacy-settings
/permissions
/sandbox status
/files
/context
```

## Model Configuration

Koder resolves configuration in this order:

1. CLI arguments
2. Environment variables
3. `~/.koder/config.yaml`
4. Built-in defaults

Universal environment variables work across providers:

| Variable | Purpose | Example |
|---|---|---|
| `KODER_API_KEY` | Universal API key | `sk-...` |
| `KODER_MODEL` | Active model | `gpt-4o`, `claude-opus-4-20250514` |
| `KODER_BASE_URL` | Custom OpenAI-compatible endpoint | `http://localhost:8080/v1` |
| `KODER_REASONING_EFFORT` | Reasoning effort | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `KODER_REASONING_DISPLAY` | Reasoning display mode | `off`, `summary`, `full` |
| `KODER_ENFORCE_TOOL_APPROVAL` | Approval-required tool calls when no interactive approver is wired. Unset = TTY-aware (fail-closed when non-interactive, allow+log when a TTY is attached) | `1`/`true` (fail closed), `0`/`false` (allow+log) |
| `KODER_GOAL_MAX_CONTINUATIONS` | Max hidden goal continuation turns per turn before the loop stops; `0` disables continuations | `25` (default), `0`, `10` |
| `KODER_BG_SHELL_MAX_LINES` | Max retained output lines per background shell (older lines are evicted) | `10000` (default) |
| `KODER_SKILL_INLINE_COMMANDS` | Whether skills may execute inline ``` !`cmd` ``` at render time | `1` (default), `0`/`false` (disable) |

Provider-specific examples:

```bash
OPENAI_API_KEY="sk-..." KODER_MODEL="gpt-4o" koder
ANTHROPIC_API_KEY="..." KODER_MODEL="claude-opus-4-20250514" koder
GOOGLE_API_KEY="..." KODER_MODEL="gemini/gemini-2.5-pro" koder
KODER_BASE_URL="http://localhost:8080/v1" KODER_MODEL="openai/local-model" koder
```

Subscription-backed providers use local OAuth token stores:

```bash
koder auth login google
koder auth login claude
koder auth login chatgpt
koder auth login antigravity
koder auth login github-copilot
koder auth list
```

After login, select an OAuth-backed model with its provider prefix:

```bash
KODER_MODEL="google/gemini-3-pro-preview" koder
KODER_MODEL="claude/claude-opus-4-5-20250514" koder
KODER_MODEL="chatgpt/gpt-5.2" koder
```

OAuth tokens and cached model lists are stored under `~/.koder/tokens/`.

## Configuration File

Persistent defaults live in `~/.koder/config.yaml`:

```yaml
model:
  name: "gpt-4o"
  provider: "openai"
  reasoning_effort: medium

cli:
  session: null
  stream: true

mcp_servers: []

voice:
  enabled: false
  provider: null
  model: null

harness:
  reasoning_display: "off"
```

Useful configuration commands:

```bash
koder config show
koder config edit
koder config export ~/koder-settings.json
koder config import ~/koder-settings.json --dry-run
```

See the [Configuration Guide](docs/configuration.md) for provider setup, OAuth, settings bundles, managed settings, MCP configuration, and voice routing.

## Extending Koder

Koder can be extended at the project, user, and plugin levels.

### Skills

Skills are local instruction bundles loaded on demand:

```text
.koder/skills/api-review/SKILL.md
~/.koder/skills/personal-style/SKILL.md
```

```markdown
---
name: api-review
description: Review API changes for compatibility and error handling
allowed_tools:
  - read_file
  - grep_search
---

Review public API changes for request shape, response shape, status codes, and migration notes.
```

Inspect skills with `/skills`. Create verifier skills with `/init-verifiers`.

### MCP Servers

MCP servers add external tools to the runtime:

```bash
koder mcp add filesystem "python -m mcp.server.filesystem" --scope project
koder mcp add api --transport http --url http://localhost:8000
koder mcp list
```

### Plugins And Channels

Plugins can contribute skills, commands, MCP servers, channels, and dependencies:

```bash
koder plugin install ./my-plugin --scope project
koder plugin list
koder --channels server:my-channel
koder --channels plugin:team-chat@local
koder /channels
```

See [Skills, Plugins, and MCP](docs/extensions.md) for the full extension model.

## Architecture

```text
koder_agent/
├── agentic/        # Agent creation, hooks, guardrails, approvals
├── auth/           # OAuth providers, token storage, provider-specific routing
├── cli.py          # Main CLI entry point
├── config/         # YAML, environment, and settings management
├── core/           # Scheduler, sessions, goals, streaming, security, TUI prompt
├── harness/        # Runtime commands, cron loops, plugins, memory, permissions, teams, UI scaffolding
├── mcp/            # Model Context Protocol integration
├── providers/      # Provider routing metadata
├── tools/          # Tool implementations
└── utils/          # Client setup, prompts, sessions, model info, terminal theme
```

Runtime flow:

1. `cli.py` parses arguments and builds a runtime request.
2. `HarnessRuntime` loads permissions and dispatches interactive, prompt, and subcommand modes.
3. Session flow wires context, hooks, plugins, agents, slash commands, and scheduler execution.
4. `AgentScheduler` streams model execution, usage tracking, queued input, and goal continuations.
5. Tool and permission layers validate file, shell, MCP, skill, and teammate operations.
6. `EnhancedSQLiteSession` persists transcripts and session metadata in `~/.koder/koder.db`.

## Development

Set up the repository:

```bash
uv sync
uv run koder
```

Code quality and tests:

```bash
uv run black . && uv run ruff format && uv run ruff check --fix
uv run --frozen --no-env-file python scripts/run_isolated_tests.py -q
```

Focused test examples:

```bash
uv run --frozen --no-env-file python scripts/run_isolated_tests.py --test tests/test_file_tools.py -q
uv run --frozen --no-env-file python scripts/run_isolated_tests.py -v -k "test_name"
```

The isolated runner uses disposable application state, drops inherited provider
credentials before collection, and guards ordinary Python child processes. It
excludes real TUI and manual/live-provider tests; it is not an OS sandbox for
untrusted code. See [test validation](CONTRIBUTING.md#isolated-test-validation)
for result artifacts and exact boundaries, and
[wheel validation](CONTRIBUTING.md#wheel-validation) for distribution checks.

## Security And Privacy

- API keys should live in environment variables or local user config, not project files.
- OAuth tokens and cached provider model lists are stored under `~/.koder/tokens/`.
- Sessions, transcripts, goals, memories, scheduled tasks, agents, and teams are stored locally under `~/.koder/` and project `.koder/` paths.
- Koder does not upload sessions to a Koder-hosted service. Model requests still go to the provider you configure.
- Shell, file, MCP, and teammate operations are mediated by local permission and sandbox policy. Foreground shell commands can use a real sandbox backend when `/sandbox status` reports that backend as available.

Use these commands to inspect boundaries:

```bash
/privacy-settings
/permissions
/sandbox
/managed-settings
/files
/context
```

See [Sandbox Guide](docs/sandbox.md) and [Permissions and Privacy](docs/permissions-and-privacy.md) for details.

## Documentation

- [Feature Guide](docs/features.md) - topic map for Koder's main user-facing capabilities
- [Getting Started](docs/getting-started.md) - first install, provider setup, and a safe first session
- [Interactive TUI](docs/interactive-tui.md) - prompt controls, slash commands, mentions, shell mode, and voice input
- [Configuration Guide](docs/configuration.md) - config files, environment variables, providers, OAuth, and settings bundles
- [Sessions and Memory](docs/sessions-and-memory.md) - named sessions, goals, scheduled loops, resume, compaction, rewind, memory, and local storage
- [Agents and Teams](docs/agents-and-teams.md) - background subagents, project agents, teams, teammate modes, and team memory
- [Workflows](docs/workflows.md) - review, planning, Git, GitHub, release, and verification workflows
- [Skills, Plugins, and MCP](docs/extensions.md) - extending Koder with skills, plugins, MCP servers, channels, and Magic Docs
- [Permissions and Privacy](docs/permissions-and-privacy.md) - approvals, sandbox policy, managed settings, workspace roots, and data boundaries
- [Hooks](docs/hooks.md) - run your own commands on session, tool, permission, and agent lifecycle events
- [Sandbox Guide](docs/sandbox.md) - shell sandbox backends, status fields, configuration, and troubleshooting
- [Voice Mode](docs/voice-mode.md) - voice dictation setup and provider-specific notes
- [Command Reference](docs/commands.md) - complete slash-command catalog

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/amazing-feature`.
3. Make a focused change with tests or validation.
4. Commit your changes: `git commit -m 'Add amazing feature'`.
5. Push the branch and open a pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License. See [LICENSE](LICENSE) for details.
