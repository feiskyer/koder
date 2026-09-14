# Configuration Guide

Koder supports flexible configuration through three mechanisms (in order of priority):

1. **CLI Arguments** - Highest priority, for runtime overrides
2. **Environment Variables** - For secrets and runtime configuration
3. **Config File** - For persistent defaults (`~/.koder/config.yaml`)

## Table of Contents

- [Config File](#config-file)
- [Environment Variables](#environment-variables)
- [Settings Files](#settings-files)
- [Settings Bundles](#settings-bundles)
- [Managed Settings](#managed-settings)
- [Provider Setup](#provider-setup)
- [MCP Servers](#mcp-servers)
- [Skills](#skills)
- [Voice Mode](#voice-mode)
- [Example Configurations](#example-configurations)

## Config File

Koder uses a YAML config file at `~/.koder/config.yaml` for persistent settings.

The `koder` executable enters through the product runtime entrypoint with config at `~/.koder/config.yaml`.

`koder --help`, `koder --version`, and `koder config ...` are maintenance paths:
they do not require successful model/permission initialization. In particular,
`koder config validate` can explain malformed YAML instead of failing while
trying to initialize the agent from that same file.

CLI and interactive configuration writes share the same persistence path.
Writes publish a complete UTF-8 snapshot atomically; new files are private, and
existing file modes and config symlinks are preserved. `ConfigChange` hooks see
the proposed file. If a hook rejects the change or raises, the previous file is
restored and the rejected in-memory cache is invalidated. External side effects
performed by a hook cannot be rolled back. Configuration remains a whole-document,
replacement interface. Cooperating writers serialize their changes and reject
stale loaded snapshots; this is not a transaction guarantee against direct
external writers or process crashes.

```yaml
# ~/.koder/config.yaml

# Model configuration
model:
  name: "gpt-4.1"              # Model name (default: gpt-4.1)
  provider: "openai"           # Provider name (default: openai)
  api_key: null                # API key (prefer env vars for security)
  base_url: null               # Custom API endpoint (optional)

  # Reasoning effort for OpenAI reasoning models (o1, o3, gpt-5.1, etc.)
  reasoning_effort: medium     # none, minimal, low, medium, high, xhigh, or max (default: medium)

# CLI defaults
cli:
  session: null                # Default session name (auto-generated if null)
  stream: true                 # Enable streaming output (default: true)

# MCP servers for extended functionality
mcp_servers: []

# Voice dictation
voice:
  enabled: false               # Enable interactive voice dictation
  provider: null               # openai, chatgpt, google, gemini, azure
  model: null                  # Optional transcription model override
  api_key: null                # Optional voice-specific API key
  base_url: null               # Optional voice-specific base URL
  api_version: null            # Optional API version (currently Azure)

# Runtime harness settings
harness:
  reasoning_display: "off"     # off, summary, or full (default: off)
  auto_dream_write_mode: review # off, review, or automatic (default: review)
```

`automatic` is the only value that enables direct AutoDream memory writes. Legacy
`auto_dream_enabled: true` migrates to the review queue, while `false` migrates to
`off`; boolean and `yes`/`on`/`enabled` aliases are not accepted for the new field.
Both review approval and automatic mode keep `user` memories in the global user store
and route `feedback`, `project`, and `reference` memories to the originating workspace.

## Voice Mode

Voice dictation is configured with top-level `voice.*` settings in `~/.koder/config.yaml`.

See [Voice Mode](voice-mode.md) for:

- interactive usage
- `/voice` commands
- provider-specific configuration
- Azure OpenAI examples
- troubleshooting

## Environment Variables

### Universal Variables

These `KODER_*` variables work with any provider and override provider-specific settings:

| Variable | Purpose | Example |
|----------|---------|---------|
| `KODER_API_KEY` | Universal API key (works with all providers) | `sk-...`, `your-api-key` |
| `KODER_BASE_URL` | Custom API endpoint | `http://localhost:8080/v1` |
| `KODER_MODEL` | Model selection | `gpt-4o`, `claude-opus-4-20250514` |
| `KODER_REASONING_EFFORT` | Reasoning effort for reasoning models | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` |
| `KODER_REASONING_DISPLAY` | Reasoning display mode | `off`, `summary`, `full` |
| `EDITOR` | Editor for `koder config edit` | `vim`, `code` |

**Priority:** `KODER_API_KEY` and `KODER_BASE_URL` override provider-specific variables (like `OPENAI_API_KEY`) and config file settings.

### Provider-Specific API Keys

Use these if you need different keys for different providers:

| Provider | API Key Variable | Additional Variables |
|----------|------------------|---------------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| Anthropic | `ANTHROPIC_API_KEY` | - |
| Google/Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | - |
| Azure | `AZURE_API_KEY` | `AZURE_API_BASE`, `AZURE_API_VERSION` |
| Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` | `VERTEXAI_LOCATION` |
| GitHub Copilot | `koder auth login github_copilot` | `GITHUB_COPILOT_TOKEN_DIR` (optional token cache location) |
| Groq | `GROQ_API_KEY` | - |
| Together AI | `TOGETHERAI_API_KEY` | - |
| OpenRouter | `OPENROUTER_API_KEY` | - |
| Mistral | `MISTRAL_API_KEY` | - |
| Cohere | `COHERE_API_KEY` | - |
| Bedrock | `AWS_ACCESS_KEY_ID` | `AWS_SECRET_ACCESS_KEY` |

## Settings Files

Besides `config.yaml`, Koder reads JSON settings files for hooks, permission rules, sandbox policy, and the status line:

| File | Scope |
|---|---|
| `~/.koder/settings.json` | User, all projects |
| `.koder/settings.json` | Project, committed to git |
| `.koder/settings.local.json` | Project, this machine only (gitignored) |

Permission rules live under `permissions.allow` / `permissions.deny` as `"tool_name(content)"` strings (deny rules win over allow rules). Hook configuration lives under `hooks` — see the [Hooks](hooks.md) guide for events, matchers, and examples.

## Settings Bundles

`config edit` runs the editor as an argument vector, edits a private temporary
candidate, and publishes only a successful, valid edit. Paths with spaces are
passed as one argument. If another writer changed the configuration while the
editor was open, the stale candidate is rejected without overwriting the winner.
Ordinary saves and migrations coordinate publication, ConfigChange decisions and
rollback using the same per-file writer lock.

Koder can export and import a local settings bundle for machine-to-machine setup or backup. Bundles include known Koder config, settings, keybindings, user memories, project memories, and project session notes. Token stores, model caches, transcripts, task records, plugin caches, and arbitrary files are not included.

```bash
koder config export ~/koder-settings.json
koder config export ~/koder-project-settings.json --scope project
koder config import ~/koder-settings.json --dry-run
koder config import ~/koder-settings.json
```

Export and import scopes are `all`, `user`, and `project`. Import writes into the current `HOME` and current project, validates checksums, rejects unsafe relative paths, and creates timestamped backups before replacing existing files.

Import validates every selected entry before publishing changes, including its
role/scope, content type, checksum and target. Duplicate targets and symlinked
storage boundaries are rejected. A write failure triggers rollback of targets
already changed; retained backups support recovery if rollback itself fails.
This is exception recovery, not an all-or-nothing transaction across process
crashes or concurrent writers.

Import captures the existing hook definitions before writing any destination.
After publishing the complete candidate bundle, it dispatches `ConfigChange`
for each changed configuration file using that same pre-import snapshot:

| Imported file | `ConfigChange` source and matcher |
|---|---|
| User `config.yaml`, `settings.json`, or `keybindings.json` | `user_settings` |
| Project `.koder/settings.json` | `project_settings` |
| Project `.koder/settings.local.json` | `local_settings` |

Newly imported hooks or `disableAllHooks` cannot replace the rules judging that
import. A synchronous blocking hook or raised exception triggers rollback of the
whole applied batch while its writer locks are held. A confirmed newer external
file is preserved and incomplete rollback is reported. New project hook content
does not inherit the previous content's approval. No-op imports and `--dry-run`
do not execute hooks; imported memory documents are not `ConfigChange` events.

The snapshot freezes hook definitions, not arbitrary scripts they reference.
Existing `async`, `once` and exit-code semantics still apply: async hooks cannot
veto the import, and hook side effects cannot be undone. Lock-free readers can
observe provisional files before a rollback; import is not a crash-safe,
serializable multi-file transaction.

Exported bundles are written with private `0600` permissions on supported local
filesystems. They are not sanitized sharing artifacts: configuration and memory
can still contain secrets, even though dedicated token stores are excluded.

## Managed Settings

Koder reads an optional local managed policy file at `~/.koder/managed-settings.json`. It can define high-priority hook settings and sandbox policy keys such as `enabled`, `mode`, `backend`, and `autoAllowBashIfSandboxed`.

```bash
koder /hooks
koder /sandbox status
```

Managed settings are local files. Koder does not fetch a hosted managed-settings service; the policy is read from the local file currently present on disk and surfaces in `/hooks` and `/sandbox status`.

See the [Sandbox Guide](sandbox.md) for sandbox backend selection, enable/disable commands, and status fields.

## Provider Setup

Koder's OAuth model names remain `google/...`, `claude/...`, `chatgpt/...` and
`antigravity/...`. Agent and auxiliary requests translate these names at the
request boundary to private SDK routing identifiers. The raw model name is
restored before the Koder handler runs, and response/stream model fields retain
the public name. Koder no longer removes GPT/Codex names from LiteLLM's native
model catalogs to influence dispatch.

OAuth endpoints and credentials belong to the selected Koder handler. An
unrelated API key, base URL or API client is not forwarded for these requests;
request options cannot override the selected provider. Private-name collisions
with SDK-native routes or other registered handlers fail explicitly. These
identifiers coordinate routing, not authorization against other Python code
in the same process.

Ordinary Koder setup does not claim public provider aliases in the SDK registry.
For integrations explicitly calling the legacy `register_oauth_providers()`,
unrelated custom-provider entries are retained and conflicting public aliases
raise before mutation; that low-level registration does not override SDK-native
dispatch rules. Private route registration similarly checks current SDK state
instead of trusting an old success flag.

Streaming requests own their transports separately. EOF, errors, cancellation
and explicit close release that request's stream without sharing lifecycle state
with another request. When SDK tracing is enabled, consume and close the SDK event
iterator in the same async context; transport cleanup is cancellation-safe.
These contracts do not substitute for live acceptance against each provider.

### OAuth Refresh and Cancellation

The four OAuth handlers acquire credentials asynchronously. CLI client setup,
agent model snapshots, auxiliary model resolution, and async auth commands move
blocking credential I/O to owned worker threads. Cancelling a caller waits for
its started I/O to finish; it does not leave a token write running after that
caller exits. Synchronous compatibility APIs remain blocking.

Cooperating refreshes for the same provider and token directory share a separate
private refresh lock. After acquiring it, each caller rereads the credentials:
it reuses a newer valid token or stops if the account was removed or replaced
with an expired snapshot. Network work does not hold the short token mutation
lock, so local deletion can still proceed. A refresh result is saved only if its
original token snapshot remains current; it cannot recreate a deleted account
or overwrite a different login.

Refresh coordination and the provider request share a 30-second deadline.
Provider cancellation cleanup and local I/O are still joined, so this is not a
hard upper bound on elapsed time. A caller cancelled while waiting for the
refresh lock stops before sending a request. If refresh has already started,
Koder waits for its result and conditional publication before propagating the
caller cancellation, because the provider may have rotated the refresh token.

These locks coordinate the same local token directory, not old clients or
direct credential writers. Different token directories do not coordinate a
shared native credential backend. Exact same-value delete/re-login cycles are
not distinguishable by snapshot comparison; process termination, failed
publication, or a response lost after upstream rotation can still require a new
login. These guarantees do not cover every synchronous metadata path, manual
stdin input, or actual provider/Keychain acceptance.

### OAuth Records in Keychain

On macOS, Koder accesses the existing file-based Keychain through a short-lived
native helper. Service, account and credential data travel through a private
stdin/stdout protocol, not child-process arguments or credential environment
variables. UTF-8 values retain whitespace, newlines and embedded NULs; generic
hexadecimal-looking passwords are not decoded as hexadecimal.
Service/account identifiers must be nonempty and NUL-free. Empty identifiers
must not silently become unconstrained native lookup or deletion predicates.

Checked reads/deletions recognize only the complete native `errSecItemNotFound`
status as absence. A process exit code, timeout, malformed response, locked
keychain or denied access is not proof of absence. Each helper call has a
five-second timeout. If a mutation's result is lost, its outcome is reported as
uncertain; TokenStorage does not then overwrite or remove an existing fallback
file as though it knew the native write had failed.

The helper keeps the file-based store and search-list behavior used by the old
`security` CLI backend. It updates existing items rather than deleting and
recreating them, preserving their ACLs. It does not change search lists or grant
broader access. The helper suppresses interactive prompts for its operation and
restores its previous interaction setting before exit.

Existing item ACLs may not authorize the Python helper identity. Such access
failures are reported rather than silently treated as missing credentials.
Known unavailable/failed writes retain the existing private-file fallback
policy; an indeterminate mutation is not a known failure. Native ACL/UI
compatibility is platform-dependent and is not established by synthetic tests.

The file-based compatibility APIs are deprecated by Apple. Switching to the
data-protection keychain requires a separate storage/signing/migration design;
this backend does not silently move existing credentials to it. This is not
isolation from arbitrary native code running as the same user.

OAuth records retain single-line ASCII JSON for legacy readers. Older pretty
JSON and legacy hex-encoded OAuth records remain readable at the OAuth boundary.
Reading them does not rewrite them. Empty or malformed records fail strict
refresh comparison and cannot authorize replacement.

### Quick Start (Any Provider)

The simplest way to configure Koder - use universal `KODER_*` variables:

```bash
# Works with any provider
export KODER_API_KEY="your-api-key"
export KODER_MODEL="gpt-4o"  # or "claude-opus-4-20250514", etc.

# Optional: custom endpoint
export KODER_BASE_URL="https://your-endpoint.com/v1"

koder
```

### OpenAI

```bash
# Using universal variable (recommended)
export KODER_API_KEY=your-api-key
koder

# Or provider-specific variable
export OPENAI_API_KEY=your-api-key
export KODER_MODEL="gpt-4o"  # Optional, default: gpt-4.1
koder
```

### Anthropic

```bash
# Using universal variable (recommended)
export KODER_API_KEY=your-api-key
export KODER_MODEL="claude-opus-4-20250514"
koder

# Or provider-specific variable
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

Run `koder auth login github_copilot` to start the GitHub device login. Visit <https://github.com/login/device> and enter the code shown in the terminal. LiteLLM stores Copilot tokens under `~/.config/litellm/github_copilot` by default. If refresh fails with `Failed to refresh API key`, Koder will ask you to run the login command again.

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

Koder vendors LiteLLM's `model_prices_and_context_window.json` under
`koder_agent/data/` and forces LiteLLM's local cost-map mode at startup. This avoids
runtime fetches from GitHub when LiteLLM imports. Before publishing a release, refresh
the vendored map with:

```bash
uv run scripts/update_litellm_model_cost_map.py
```

Importing `koder_agent` or its basic goal/cancellation utilities does not initialize
LiteLLM, load the price table, or trigger the SDK's dotenv loader. The existing
local-cost-map environment flag is retained. Model-facing modules initialize the
SDK through a shared boundary and install the vendored table before using model
metadata or pricing. Unknown custom model entries survive that installation.
Provider initialization can still have its SDK's own effects; this is not a
promise that importing every high-level model or CLI module is side-effect-free.

The public package version, CLI banner and `/version` share installed distribution
metadata. Without installed metadata, the resolver can read this checkout's static
`pyproject.toml`; if neither source is available, it reports `unknown` rather than
a stale hard-coded release. On supported Python versions, source parsing uses
standard-library `tomllib`. Version resolution itself does not query a provider
or fetch release information.

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

### Minimal (Any Provider)

```bash
# Just set your API key and go!
export KODER_API_KEY="your-api-key"
koder
```

Or with config file:

```yaml
# ~/.koder/config.yaml
model:
  name: "gpt-4o"
  provider: "openai"
```

```bash
export KODER_API_KEY="sk-..."
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
      X-API-Key: "your-company-api-key"
```

```bash
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com"
koder
```

Note: `config.yaml` values are used literally; `${VAR}` placeholders are not expanded there. Environment-variable interpolation in MCP server definitions works only in a project `.mcp.json` file — put `"X-API-Key": "${COMPANY_API_KEY}"` there and the real value stays in your environment. For YAML-configured servers, use `koder mcp add --header` or write the literal value.

Project `.mcp.json` files and inline `mcpServers` in project agent frontmatter must be reviewed with `koder mcp approve` before Koder connects them. Approval is bound to the repository root, source path, fixed execution directory, and a digest of the fully expanded executable values. Environment changes therefore invalidate approval without storing the expanded secret values.

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
