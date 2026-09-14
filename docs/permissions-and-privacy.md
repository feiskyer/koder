# Permissions And Privacy

Koder is a local terminal assistant. It can read and edit files, run shell commands, call configured model providers, and connect to configured local extensions. This guide explains how to inspect and control those boundaries.

## Local Data Paths

Default paths:

| Path | Purpose |
|---|---|
| `~/.koder/config.yaml` | User configuration. |
| `~/.koder/settings.json` | User permission and runtime settings. |
| `.koder/settings.json` | Project permission and runtime settings. |
| `~/.koder/koder.db` | Sessions and transcripts. |
| `~/.koder/tokens/` | OAuth token stores and model caches. |
| `~/.koder/agents/`, `~/.koder/teams/`, `~/.koder/tasks/` | Agent, team, and task state. |
| `.koder/memory/`, `~/.koder/memory/` | Project and user memory files. |

Koder does not upload sessions to a Koder-hosted service. Model requests still go to the configured model provider, and any enabled MCP server, plugin, shell command, teammate process, or external tool can have its own local or network behavior.

## Permission Commands

Inspect the active policy:

```bash
/permissions
/sandbox
/sandbox status
```

Change sandbox policy:

```bash
/sandbox enable
/sandbox enable unix-local
/sandbox disable
```

The permission layer protects shell, file, tool, and teammate operations. Foreground `run_shell` commands can also run through a real OpenAI Agents SDK sandbox backend when sandbox policy is enabled and the configured backend is available. `/sandbox status` reports the active backend and whether it is available.

Sandbox policy supports these high-level modes:

| Mode | Behavior |
|---|---|
| `read-only` | Mutating shell commands are denied before execution. |
| `workspace-write` | Foreground shell commands run in the selected backend and can write the workspace while outside-workspace writes are blocked by supported backends. |
| `danger-full-access` | Sandbox disabled; shell commands use the normal local executor. |

Koder currently treats file tools, MCP servers, teammate processes, and background shell commands as permission-backed unless their execution path is explicitly routed through a sandbox backend. Hosted backends are listed with missing dependency or credential hints; credentials are not printed. See the [Sandbox Guide](sandbox.md) for setup, status fields, backend options, and troubleshooting.

Hooks extend this policy layer with your own commands: a `PermissionRequest` hook can approve or deny a call programmatically, and `PermissionDenied` hooks can log every denial. See [Hooks](hooks.md).

The main SDK tool path checks shell commands, file reads/writes, URL fetches,
and MCP resource reads using their actual arguments before invoking the tool.
Target-specific deny rules apply to reads as well as mutations. If the permission
evaluator fails, the guarded invocation is refused with a model-visible error;
the tool body is not executed. An evaluator failure is not an approval request
and cannot be converted into an allow by an approval hook.

Rules are tool-specific. A rule for `read_file` is not a blanket operating-system
filesystem policy for every search tool, shell process, or third-party extension.
Use supported sandbox controls for process-level confinement.

## Workspace Directories

Koder starts from the current working directory. Add another workspace root only when a task needs it:

```bash
/add-dir /path/to/other/workspace
```

Use `/files`, `/context`, and `/ctx_viz` to see what the session has loaded.

An in-process subagent binds its own execution directory without changing the
parent process's cwd. Its relative file targets and shell/sandbox execution are
checked against that directory. The main session can still work in a subdirectory
of its allowed workspace without losing access to allowed sibling paths.
Absolute-path deny rules also match equivalent relative tool targets.

These are tool-routing and permission guarantees, not an OS confinement claim:
worktrees alone do not sandbox shell processes, MCP servers, or third-party code.

## Managed Settings

Managed settings are local high-priority policy files:

```text
~/.koder/managed-settings.json
```

Inspect the resulting policy with:

```bash
/hooks
/sandbox status
```

Koder does not fetch a hosted managed-settings service. Policy is read from the local file currently present on disk.

## Shell Commands

Shell commands can be run from the TUI with `!` or by the model through shell
tools. Model-requested mutations may require approval depending on policy.
Explicit `!` commands use the manual shell path: model-tool permission modes
such as `dontAsk` do not provide an extra approval gate for that direct user
request. The shell executor's security and sandbox checks still apply.

Examples:

```bash
!git status --short
!uv run pytest tests/test_file_tools.py
```

Background commands can be started with `&` and monitored or stopped by shell tooling.

An explicit executable path does not inherit the read-only classification or
allow rule of its basename. For example, `./ls` and `/some/path/ls` are not
automatically trusted as the ordinary `ls` command. Approval or an applicable
rule naming that path is required. Known bare runners can be normalized without
discarding the inner executable path: `env ./tool` can match `./tool:*`, not an
unrelated `tool:*` rule.

Only recognized argument-preserving wrapper forms are transparent to this
normalization, such as `env command`, `timeout 5 command`, and
`nice -n 1 command`. Unknown options, `env -S`, environment edits,
stdin-built `xargs` arguments, `nohup` output and session-detaching `setsid`
require their own authorization. A prefix rule for `ls` does not authorize
`PATH=... ls` or `env PATH=... ls`: the environment can change which code runs.
Use an explicit full-command rule when that behavior is intended.

Rule matching preserves word boundaries. A single executable named `ls suffix`
does not inherit an `ls:*` rule just because the normalized display contains a
space.

Classification, rule matching and “always allow” derivation share literal
command segmentation. Quotes remain significant within a chain: `FOO=bar ls`
is not the same command as `'FOO=bar' ls`. A `#` inside a word is part of that
word; an input-redirection filename is not an executable. Unsupported grouping,
multiline quoting and control characters require explicit approval.

Saved prefix rules retain the path you approved. Approving
`./tools/npm test` can produce `./tools/npm test:*`, not `npm test:*`.
Background or pipeline chains do not widen to a rule for just one component.
Unusual spellings that cannot reuse a stable prefix remain exact approvals.

Deny and dangerous-command checks remain conservative and may recognize
basenames, including path-qualified runners. That stronger deny projection is
never used to grant permission. Unparseable commands cannot inherit a raw allow
prefix, and unresolved wrapper chains cannot gain permission from a normalized
prefix. These are command-identity checks, not binary-signature verification or
attestation of the user's configured PATH.

Git command names alone do not establish read-only behavior. The static
classifier treats reflog expiry/deletion/drop/write, `fsck --lost-found`, and
branch mutation options as approval-requiring operations. It recognizes long
option abbreviations, `--option=value`, and short mutation flags with attached
arguments. Without an overriding allow rule, `dontAsk` denies these operations;
the read-only sandbox preflight also rejects them.

Ordinary queries such as `git reflog show`, plain `git fsck`, branch listings
and log searches remain read-only classifications. An ambiguous option that
could select a write mode requires approval rather than being auto-run. This
classification is not a complete Git/shell parser or process-confinement
guarantee; explicit permission rules and sandbox enforcement remain separate.

Sed is also checked as a program, not merely for `-i`. Literal print/filter
commands, ordinary address ranges, substitutions without write/execution flags,
transliteration and bounded command groups can remain read-only. Script writes
(`w`/`W`), execution (`e`), substitution write/execution flags, script files
(`-f`/`--file`), active parameter/glob expansion and unrecognized or oversized
programs require approval. Shell quoting is retained for this check: a literal
`'$p'` end-address is different from `"$SED_PROGRAM"`.

Put options before operands, or use `--` for literal filenames beginning with
`-`; argument ordering that can be interpreted differently by BSD and GNU sed
is not auto-approved. These are conservative static checks, not executable
provenance verification or a replacement for sandbox confinement. Approved
commands retain the full capabilities of the selected sed implementation.

Other nominally read-only tools also have effectful modes. Ripgrep preprocessors
and hostname programs, `file` magic-database compilation, `sort` output files or
compression programs, and explicit Git external-diff/textconv modes require
approval unless an applicable rule explicitly authorizes them. Argument
terminators and recognized option values remain data, not executable options.

Static read-only inference also requires literal arguments. Active shell
variable, parameter, arithmetic and glob expansions can add effectful options
or execute hidden work, so they do not auto-run solely because the visible
program name is read-only. Quoted literal regexes and escaped dollar signs
remain supported. This is a bounded static check of known modes, not inspection
of pre-existing executable binaries, environment or tool configuration.

## Secrets

Prefer environment variables for secrets:

```bash
export KODER_API_KEY="sk-..."
export KODER_BASE_URL="https://your-endpoint.example/v1"
```

Avoid putting API keys in project files. OAuth tokens are stored under `~/.koder/tokens/` and refreshed locally by provider-specific auth flows.

Stored OAuth records are type-checked before use. Empty access tokens,
non-numeric/non-finite timestamps and malformed model/metadata fields are not
usable credentials; rejecting a record does not rewrite or delete its file.
Structurally valid expired tokens remain readable for refresh. Legacy null
values in optional refresh-token, model-list and extra-metadata fields use
explicit empty defaults.

## Privacy Checks

Use these commands when you want to verify what Koder can see:

```bash
/status
/files
/context
/memory
/agents summary
/tasks
```

Settings bundles can move local settings and memory between machines, but they intentionally exclude token stores, model caches, transcripts, task records, and plugin caches:

```bash
koder config export ~/koder-settings.json
koder config import ~/koder-settings.json --dry-run
```

## Practical Boundary Checklist

Before giving Koder a sensitive repository, check:

- Run `/files` and `/context` to see which workspace files are loaded.
- Run `/memory` to inspect project and user memories that may influence prompts.
- Run `/permissions` and `/sandbox status` to see command policy and backend availability.
- Run `/mcp`, `/plugin`, and `/channels` to inspect external tool surfaces.
- Run `/hooks` to see which hook commands run automatically on session and tool events.
- Keep API keys in environment variables or user config; do not store them in repository files.
- Treat model-provider requests as remote requests to that provider, even though Koder's own product state is local.
