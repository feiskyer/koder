# AGENTS.md

This file provides guidance to Koder and other AI agents when working with code in this repository.

## Commands

### Development Setup

```bash
uv sync                                    # Install dependencies
uv run koder                               # Run in interactive mode
uv run koder "Your prompt"                 # Single prompt
uv run koder -s my-session "Your prompt"   # Named session
uv run koder --resume                      # Resume previous session
uv run koder -c                            # Continue most recent session in cwd
uv run koder -p "Your prompt"              # Print mode (respond and exit)
```

### Code Quality

```bash
uv run black .                                          # Format code
uv run ruff format                                      # Format imports/style
uv run ruff check --fix                                 # Lint with auto-fix
uv run pylint koder_agent/ --disable=C,R,W --errors-only # Error-only check
```

### Testing

Use the installed-runtime entrypoint for primary unit/non-E2E integration
validation. It discovers the Python behind `koder` on PATH and reads its current
dependencies on every invocation, adds test tools without changing that install,
and rejects dependency mismatches before collection. Do not substitute an old
`uv.lock` environment for the user's actual runtime. An alternate install can be
selected with `--runtime-python /path/to/bin/python`.

The underlying isolated runner uses temporary application state and blocks
accidental Python access to real profiles and remote services. It is not an OS
sandbox. Dependency alignment does not mean using the user's real configuration.

```bash
bash scripts/test.sh -q
bash scripts/test.sh --test tests/core -q
bash scripts/test.sh --runtime-report /tmp/koder-runtime.json --junitxml /tmp/koder-tests.xml -q
```

Direct pytest commands do not provide those collection-time isolation guards:

```bash
uv run pytest                                   # All tests
uv run pytest tests/test_file_tools.py          # Single file
uv run pytest -v -k "test_name"                 # Single test by name
uv run pytest tests/integration/                # Integration tests
uv run pytest tests/e2e/                        # E2E tmux tests (requires tmux)
```

Validate a built wheel without importing the application:

```bash
uv run --no-project --no-env-file python scripts/verify_wheel.py dist/koder-0.6.4-py3-none-any.whl --source . --json
```

### CLI Subcommands

```bash
# MCP server management
uv run koder mcp list                              # List servers
uv run koder mcp add myserver "python -m server"   # Add stdio server
uv run koder mcp add-json myserver '{"command":"..."}' # Add from JSON
uv run koder mcp get myserver                      # Get server details
uv run koder mcp remove myserver                   # Remove server
uv run koder mcp serve                             # Expose koder tools as MCP server
uv run koder mcp reset-project-choices             # Reset project MCP approvals

# Configuration
uv run koder config show                           # Show current config
uv run koder config set model.name gpt-4.1         # Set a config value
uv run koder config edit                           # Open in editor
uv run koder config init                           # Initialize defaults
uv run koder config validate                       # Validate config schema
uv run koder config export bundle.json             # Export settings bundle
uv run koder config import bundle.json             # Import settings bundle

# Authentication (OAuth providers)
uv run koder auth login google                     # Login via OAuth
uv run koder auth list                             # List configured providers
uv run koder auth status                           # Show token status
uv run koder auth revoke google                    # Revoke tokens

# Plugins
uv run koder plugin list                           # List installed plugins
uv run koder plugin install ./my-plugin            # Install from directory
uv run koder plugin uninstall my-plugin            # Uninstall
uv run koder plugin enable/disable my-plugin       # Toggle plugin
uv run koder plugin validate ./my-plugin           # Validate manifest
uv run koder plugin marketplace list               # List marketplaces

# Other subcommands
uv run koder agents                                # List configured agents
uv run koder doctor                                # Run environment diagnostics
uv run koder review --base main                    # Headless code review
uv run koder completion bash                       # Print shell completion script
uv run koder upgrade                               # Self-upgrade
```

## Architecture

Koder is a terminal-based AI coding assistant built on the `openai-agents` library with multi-provider support via LiteLLM.

### Package Structure

```
koder_agent/
├── agentic/           # Agent creation, retry model, guardrails (plan, skill, hook), approval hooks
├── auth/              # OAuth providers (Google, Claude, ChatGPT, Antigravity, GitHub Copilot),
│   └── providers/     #   token storage, callback server, tool conversion utilities
├── cli.py             # Main CLI entry point, argument parsing, subcommand wiring
├── config/            # Configuration management (YAML schema, Pydantic models, manager)
├── core/              # Scheduler, session storage, streaming display, security, interactive mode,
│                      #   goals, status line, usage tracking, @ autocomplete, keyboard listener,
│                      #   vim mode, notifications, file index
├── data/              # Vendored LiteLLM model cost map (model_prices_and_context_window.json)
├── harness/           # Runtime orchestration layer:
│   ├── agents/        #   Agent definitions, team service, teammate execution (in-process & tmux)
│   ├── auth/          #   CLI auth command handlers
│   ├── channels/      #   Push events from MCP servers into running sessions
│   ├── cli/           #   Entrypoint routing, headless mode, shell completion
│   ├── commands/      #   Interactive slash commands, builtins, prompt commands
│   ├── config/        #   Runtime config service, settings bundle import/export, schema
│   ├── cron/          #   Scheduled task execution (cron expressions, loop, storage)
│   ├── hooks/         #   Command hooks runtime (PreToolUse, PostToolUse, SessionStart, etc.)
│   ├── mcp/           #   MCP CLI command handlers
│   ├── memory/        #   Conversation compaction (auto, micro, dream), recovery, retrieval
│   ├── permissions/   #   Permission service, shell classifier, path policy, rule hierarchy
│   ├── plan/          #   Plan mode state management
│   ├── plugins/       #   Plugin lifecycle, manifest, marketplace, discovery
│   ├── sandbox/       #   Sandboxed execution backends (local, Docker, Modal, E2B, Vercel)
│   ├── skills/        #   Bundled skill loading and discovery
│   ├── tasks/         #   Task service, storage, output
│   ├── tools/         #   Harness-level tool implementations (file ops, shell executor, search, web)
│   ├── voice/         #   Voice dictation service (OpenAI, Google)
│   └── worktree/      #   Git worktree management for parallel development
├── litellm_cost_map.py # Local cost-map mode and explicit model-boundary SDK initialization
├── mcp/               # MCP server integration: stdio/SSE/HTTP transports, OAuth, reconnection,
│                      #   server factory/manager, elicitation, prompts, serve mode
├── providers/         # Provider model definitions and compatibility shims
├── tools/             # Tool implementations (file, search, shell, web, task, todo, skill, goal,
│                      #   cron, plan mode, worktree, agent spawn, code intelligence, etc.)
└── utils/             # Client setup, system prompts, session helpers, model info/deprecation,
                       #   image input, terminal theme
```

`koder_agent/version.py` supplies the shared lightweight version resolver for the
package and CLI. Importing the root package keeps local cost-map mode enabled but
does not import LiteLLM; model consumers use `litellm_cost_map.get_litellm()` to
initialize the SDK and publish the vendored map before model/pricing use.

### Core Flow

1. **CLI Entry** (`cli.py`) parses args and builds a `RuntimeRequest` via `harness/cli/entrypoint.py`.
2. **HarnessRuntime** (`harness/runtime.py`) loads the permission hierarchy (project → local → user settings), creates the `PermissionService` and `AiShellClassifier`, builds command and tool registries, and dispatches to the appropriate mode (interactive, prompt, subcommand, help, version).
3. **Session Flow** (`harness/session_flow.py`) wires context, hooks, plugins, agents, slash commands, memory retrieval, and scheduler execution. Handles session switching, resume, headless/print modes, and agent mentions.
4. **AgentScheduler** (`core/scheduler.py`) orchestrates agent execution with streaming, usage tracking, token-aware compaction (via `harness/memory/`), goal management, and cancellation support.
5. **Agent Creation** (`agentic/agent.py`) builds the agent with tools, MCP servers, model settings, provider routing, guardrails (plan mode, skill restriction, hook pre-tool), and the `RetryingLitellmModel` wrapper.
6. **Tool Registration** (`tools/__init__.py` → `get_all_tools()`) collects all `FunctionTool` instances, attaches guardrails, and returns the full tool list. Tool wrappers in `tools/compat.py` handle output truncation.
7. **Harness Tool Execution** (`harness/tools/`) provides the actual implementations for file, shell, search, web, code intelligence, and MCP operations. `harness/tools/shell_executor.py` runs shell commands with security checks and permission prompts.
8. **Session Storage** (`core/session.py`) persists conversations in SQLite with token-aware compression helpers.

### Key Design Patterns

- **Provider Abstraction**: `utils/client.py` resolves providers from environment/config and selects native OpenAI clients or LiteLLM wrappers. API keys flow through explicit client/request arguments, not newly exported child-process environment variables; base URLs and non-secret provider settings keep their documented precedence.
- **OAuth Request Routing**: `auth/oauth_routing.py` maps public OAuth model names to private SDK wire routes, restores model identity, and keeps provider credentials/endpoints with Koder handlers. `agentic/oauth_model.py` preserves SDK chat conversion and per-request stream ownership. Do not steer OAuth by deleting SDK-native model catalog entries.
- **OAuth Providers**: `auth/providers/` supports Google, Claude, ChatGPT, Antigravity, and GitHub Copilot subscription-backed model access. Tokens stored under `~/.koder/tokens/`.
- **Keychain Boundary**: `auth/keychain_backend.py` is a standalone, lazy native helper for existing file-based Keychain items. Values travel through bounded pipes, not argv; checked operations use full OSStatus values. Never infer absence from a process exit code. Native references/buffers and the helper's interaction policy are restored/released. TokenStorage propagates uncertain mutation results instead of publishing a fallback as if the write certainly failed. Native tests use adapters and must not access real Keychains or profiles.
- **MCP OAuth Callbacks**: `mcp/oauth.py` owns the loopback listener and accepted connections for each authorization flow. Shutdown interrupts active reads, closes the listener and joins its server thread off the event loop; cleanup preserves the original authorization error or cancellation. Idle callback reads have a finite timeout, and failed thread startup closes the allocated listener.
- **MCP SDK Connections**: `mcp/connection.py` owns each transport/session on one lifetime task. Connect, reconnection retirement, and cleanup callers signal/join it; never unwind AnyIO scopes from a different task. MCP 2 uses the public `Client` with numeric seconds; MCP 1 uses its duration-based `ClientSession` contract. Channel admission uses Koder's explicit stream/session binding, not SDK-private read-stream fields. Test actual SDK negotiation over subprocess stdio, not just mocked configuration lists.
- **MCP Elicitation**: `mcp/elicitation.py` runs hooks and owned form prompts asynchronously. Serialize terminal readers per event loop; cancellation joins prompt cleanup. Preserve arrays, validate responses with an offline schema registry, omit content on decline/cancel, and never read stdin without an interactive terminal. URL mode is explicitly declined, not opened automatically.
- **Manual OAuth Input**: `harness/auth/commands.py` owns an asynchronous authorization-code prompt. Timeout and cancellation stop and join the prompt before returning; do not use an executor-backed `input()` that can keep consuming stdin after its waiter exits. Empty/closed input never reaches token exchange or storage.
- **Copilot Device Login**: `auth/github_copilot.py` owns asynchronous device polling using the installed SDK's endpoints, headers and cache paths. Validate finite positive deadlines, honor provider expiry/slow-down responses, close HTTP resources through cancellation, and publish only after both credentials are obtained. Cache writes are owned and atomic per file, not a two-file/power-loss transaction.
- **RetryingLitellmModel**: `agentic/agent.py` wraps LiteLLM with exponential backoff retry (3-5 attempts) for rate limits and transient errors.
- **Progressive Disclosure Skills**: `tools/skill.py` loads skill metadata at startup (Level 1) and full content on demand (Level 2), saving 90%+ tokens.
- **Bundled Skills**: `harness/skills/bundled_skills/` ships 14 built-in skills: `batch`, `code-review`, `debug`, `fewer-permission-prompts`, `init-explore`, `loop`, `remember`, `review-spec`, `run`, `security-review`, `simplify`, `stuck`, `update-config`, `verify`.
- **Skill Restrictions**: `tools/skill_context.py` and `agentic/skill_guardrail.py` limit tool access when restricted skills are active.
- **Goals System**: `core/goals.py` provides persistent session objectives with optional token budgets. Tools (`tools/goal.py`) create/update/query goals; `core/goal_runtime.py` manages lifecycle.
- **Hooks System**: `harness/hooks/runtime.py` dispatches command hooks for events like `PreToolUse`, `PostToolUse`, `SessionStart`, `FileChanged`, `CwdChanged`, etc. Hooks are configured in `.koder/settings.json`.
- **Hook Results**: Matching another hook retains accumulated results unless its output explicitly replaces a field. HTTP hook failures close their response bodies before returning diagnostics.
- **Memory & Compaction**: `harness/memory/` handles token-aware compaction, dream modes and retrieval. Its standalone `TranscriptStore` and SQLite recovery helpers are not wired to the main SDK-backed Session; no automatic `.bak` fallback is implied.
- **Compaction Safety**: Empty, analysis-only or malformed summary output must fail before history replacement. Preserve instruction items, recent multimodal messages and complete trailing tool pairs; unfinished tool items stay in summary source. Recognize both Responses and Chat content blocks. An already-compacted no-op keeps original item shapes and does not count pinned system/developer instructions against the recent-conversation budget.
- **File Read Ownership**: Managed file tools use the actual `EnhancedSQLiteSession.file_read_state`, never another session's or the legacy unscoped cache. Capture this state before file side effects; detached, retired or closed managed bindings cannot borrow unscoped read evidence. Session replacement, clear/pop, micro-truncation and close invalidate only their owner's state. Exact replacement invalidates after commit wins; failed atomic replacement retains the original evidence.
- **SQLite Ownership**: Session metadata and legacy MCP migration use `utils/sqlite_connections.py` to retain connection ownership through failed opening and repeated cancellation. Close and join the SQLite worker before returning; raw `async with aiosqlite.connect(...)` does not cover failed-opening cleanup. GoalStore reuses the same close helper for its cached connection.
- **Metadata Schema**: Session metadata reads/writes and listing share `core/session.py`'s schema helper. Already-current schemas use a read-only fast path; migration rechecks columns after `BEGIN IMMEDIATE` and commits the upgrade atomically. Do not reintroduce separate check-then-ALTER paths that race concurrent openers.
- **Streaming Display**: `core/streaming_display.py` manages Rich Live displays for real-time output with reasoning display support.
- **Approval Hooks**: `agentic/approval_hooks.py` and `harness/permissions/` wrap tool execution with permission checks. Shell commands are classified by `permissions/shell_classifier.py` and `permissions/ai_classifier.py`.
- **Shell Executable Identity**: Positive readonly/allow decisions preserve explicit executable paths; `./ls` is not trusted merely because its basename is `ls`. Path-qualified runners are not transparently unwrapped for allow matching. Deny and hazard checks retain a separate conservative basename projection. Never reuse that deny projection to grant execution, or infer an allow from failed parsing/unresolved normalization.
- **Plan Mode**: `agentic/plan_guardrail.py` restricts write operations during exploration/planning.
- **Security Guard**: `core/security.py` and `core/bash_security.py` validate shell commands before execution.
- **Shell Runner Projection**: `harness/permissions/shell_runners.py` recognizes only known argument-preserving wrapper forms for automatic read/allow decisions. Unknown flags, environment edits, `xargs`, `nohup`, and session-detaching wrappers do not inherit inner-command trust. Rule serialization preserves word boundaries; deny-only projections may conservatively recognize paths and assignments but never grant an allow.
- **Shell Segment Identity**: `harness/permissions/shell_segments.py` supplies the shared literal segment parser for classification, matching and saved-prefix derivation. Keep original quote spelling separate from decoded argv, consume only actual redirection operands, and treat `#` inside a word as literal. Saved rules retain the approved executable path; unsupported syntax stays approval-gated.
- **Sandbox**: `harness/sandbox/` supports sandboxed execution via multiple backends (unix-local, Docker, Modal, E2B, Vercel) with filesystem policies.
- **Background Shells**: `tools/shell.py` `BackgroundShellManager` tracks async shell commands.
- **Agent Teams**: `harness/agents/teams/` supports in-process and tmux-backed teammate execution with memory sync and permission bridging.
- **Team Tool Ownership**: `harness/agents/teams/tool_runtime.py` retains leader selections per actual main Session on the owning AgentService. Public team tools and interactive commands share the in-process runner; an idle teammate is still live work. Close its consumer before releasing the service. Generation-scoped TeamService handles prevent stale leader tools from acting on a deleted-and-recreated team.
- **Fork Context**: `harness/agents/runtime_context.py` binds the actual Session during a managed turn. `agent_tool(context="fork")` reads that parent instead of guessing a session ID/database, filters incomplete tool pairs, and rejects unavailable/retired parents. Detached agent tasks do not retain a live parent Session binding; each agent execution owns and closes its own Session.
- **Channels**: `harness/channels/` enables MCP servers and plugins to push real-time events into running sessions. Each managed `HarnessRuntime.run()` owns its channel policy and MCP notification handler. Admission retains that owner's policy across reader tasks, and runtime exit revokes it. A downstream turn cancellation does not retire the session's consumer; session shutdown cancels and joins it. Direct unscoped helper calls retain the legacy default context; this is not a guarantee that all other process-global runtime state is isolated.
- **Channel Inbox**: `harness/channels/inbox.py` stages bounded payloads in a private per-runtime directory and retains only entry descriptors in memory. Admission must not wait for model consumption on the MCP receive path. Reject overload explicitly; preserve unfinished files without automatic replay. Stop admission and join the consumer before finalizing the inbox. A scheduler-reported failure or cancellation is not successful delivery.
- **Plugin System**: `harness/plugins/` manages plugin lifecycle, discovery, marketplace integration, and manifest validation.
- **Code Intelligence**: `harness/code_intelligence.py` provides LSP-style operations (document symbols, workspace symbols, definition, references, diagnostics).
- **Buddy Companion**: `harness/buddy.py` implements an interactive companion personality with rarities and speech bubbles.
- **Status Line**: `core/status_line.py` renders a rich interactive status bar showing model, session, tokens, cost, and PR status.

### Tool Categories

| Category | Tools |
|----------|-------|
| File | `read_file`, `write_file`, `append_file`, `edit_file`, `list_directory`, `notebook_edit` |
| Search | `glob_search`, `grep_search`, `code_intelligence` |
| Shell | `run_shell`, `shell_output`, `shell_kill`, `git_command`, `run_powershell` (Windows) |
| Web | `web_search`, `web_fetch` |
| Task | `task_delegate`, `task_create`, `task_update`, `task_get`, `task_list`, `task_output`, `task_stop` |
| Todo | `todo_read`, `todo_write` |
| Skills | `get_skill` (loads bundled, project, and user skills) |
| Goals | `get_goal`, `create_goal`, `update_goal` |
| Agent | `agent_tool` (spawn sub-agents), `send_message`, `team_create`, `team_delete` |
| Cron | `cron_create`, `cron_delete`, `cron_list` |
| Plan | `enter_plan_mode`, `exit_plan_mode` |
| Worktree | `enter_worktree`, `exit_worktree` |
| Runtime | `config_tool`, `tool_search`, `ask_user_question`, `structured_output`, `sleep_tool`, `list_mcp_resources`, `read_mcp_resource` |

### Configuration Priority

CLI Arguments > Environment Variables > Config File (`~/.koder/config.yaml`) > Defaults

Key environment variables:

- `KODER_API_KEY` — Universal API key, overrides provider-specific keys.
- `KODER_BASE_URL` — Custom API endpoint, overrides provider-specific base URLs.
- `KODER_MODEL` — Model name, e.g. `gpt-4.1`, `claude-opus-4-20250514`, `github_copilot/gpt-5.1-codex`.
- `KODER_REASONING_EFFORT` — Reasoning effort for reasoning models (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
- `KODER_SMALL_MODEL` — Small/cheap model for auxiliary LLM calls (title generation, compaction).
- `KODER_PERMISSION_MODE` — Permission mode override (`default`, `plan`, etc.).
- Provider API keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `GITHUB_TOKEN`, `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`, `GROQ_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`, and other LiteLLM-supported provider variables.

### Settings Hierarchy

Permission rules, hooks, and other harness settings are loaded from a three-tier hierarchy (later tiers override earlier):

1. **Project** — `.koder/settings.json` (committed, shared with team)
2. **Local** — `.koder/settings.local.json` (gitignored, machine-specific overrides)
3. **User** — `~/.koder/settings.json` (global user preferences)

### Database

SQLite at `~/.koder/koder.db` stores:

- Conversation history with token-aware compression (tiktoken-based compaction)
- Session metadata with auto-generated titles
- MCP server configurations
- Goal state (one active goal per session)

### Project Context

The CLI loads `AGENTS.md` from the working directory as project-specific context for the agent. The `@` mention system supports referencing files and directories for inline context.

### Skills System

Skills are loaded from three sources:

1. **Bundled** — `koder_agent/harness/skills/bundled_skills/` (14 built-in skills)
2. **Project** — `.koder/skills/` (project-specific skills)
3. **User** — `~/.koder/skills/` (user-level skills)

Each skill has a `SKILL.md` (or `<name>.md` for bundled) with YAML frontmatter defining `name`, `description`, and optional `allowed_tools`.

### Test Structure

Tests mirror the source package structure:

```
tests/
├── agentic/           # Agent, guardrail, and hook tests
├── auth/              # OAuth provider tests
├── core/              # Scheduler, session, streaming tests
├── e2e/               # End-to-end tmux-based TUI tests
├── fixtures/          # Shared test fixtures
├── harness/           # Harness subsystem tests (agents, channels, cli, commands, config,
│                      #   cron, hooks, memory, mcp, permissions, plan, plugins, sandbox,
│                      #   skills, tasks, tools, voice, worktree)
├── integration/       # Integration tests (wiring, OAuth flows, agent teams)
├── mcp/               # MCP server tests
├── providers/         # Provider compatibility tests
├── scripts/           # Script tests (tmux scenarios, branding, docs)
├── tools/             # Tool-level tests (file ops, grep, notebook, goals, etc.)
├── utils/             # Utility tests (prompts, model info)
└── test_*.py          # Top-level tests (file tools, security, shell, skills, streaming, etc.)
```

## Requirements

- Always run `uv run black . && uv run ruff format && uv run ruff check --fix` and fix warnings/errors whenever code changes are made.
- Always use `uv run` whenever you need to run, evaluate, or test Python scripts.
- Before claiming TUI behavior, validate the real terminal flow with scenario-based tmux checks from `tests/e2e/tui_feature_scenarios.json`. Validate scenarios with `uv run scripts/tmux_feature_scenarios.py --check`; run focused scenarios with `uv run scripts/tmux_feature_scenarios.py --run <name>` or the full suite with `--run-all`.
- TUI acceptance must also use the actual installed runtime's interpreter/dependencies, with isolated HOME, a private tmux socket, and synthetic providers. Pin the selected `UV_PYTHON`/`UV_PROJECT_ENVIRONMENT` for that invocation to prevent child CLI drift; this is not permission to reuse a historical dependency lock.
- When adding a new tool, register it in `tools/__init__.py` → `get_all_tools()` and update the Tool Categories table above.
- When adding new CLI subcommands or flags, update `cli.py` parser and `harness/cli/entrypoint.py` SUBCOMMANDS set.
- When adding new hook events, update `harness/hooks/runtime.py` HOOK_EVENTS set.
- Test files should mirror the source directory structure (e.g., `koder_agent/harness/memory/` → `tests/harness/memory/`).
