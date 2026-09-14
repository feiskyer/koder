# Sandbox Guide

Koder can run foreground shell commands through a sandbox backend. The user-facing model is intentionally small: turn sandbox on, choose one backend, and check whether that backend is available.

## Quick Start

Check the current state:

```bash
/sandbox status
```

Enable sandboxing:

```bash
/sandbox enable
```

Koder prints the available backend choices. Pick one and save it to the project-local settings file:

```bash
/sandbox enable unix-local
```

Disable sandboxing:

```bash
/sandbox disable
```

## Backends

Koder exposes six backend choices:

| Backend | Use it when | Common requirement |
|---|---|---|
| `unix-local` | You want the local backend on macOS. | Requires `/usr/bin/sandbox-exec`. Koder marks this backend unavailable on Linux and other non-macOS hosts because the pinned backend does not provide kernel confinement there. |
| `docker` | You want container-backed local execution. | Python `docker` package and a reachable Docker daemon. |
| `cloudflare` | You want a Cloudflare-hosted sandbox. | `CLOUDFLARE_SANDBOX_WORKER_URL`, optional `CLOUDFLARE_SANDBOX_API_KEY`. |
| `e2b` | You want an E2B-hosted sandbox. | `E2B_API_KEY`, optional `KODER_SANDBOX_E2B_TYPE`. |
| `modal` | You want a Modal-hosted sandbox. | Modal sandbox extra and `python -m modal setup` or `modal token new`; optional `KODER_SANDBOX_MODAL_APP_NAME`. |
| `vercel` | You want a Vercel-hosted sandbox. | Vercel sandbox extra and `VERCEL_TOKEN`; optional `KODER_SANDBOX_VERCEL_PROJECT_ID` and `KODER_SANDBOX_VERCEL_TEAM_ID`. |

List backend availability at any time:

```bash
/sandbox backends
```

Koder shows environment variable names and dependency hints, not secret values.

## Status Fields

`/sandbox status` prints the fields most users need:

| Field | Meaning |
|---|---|
| `sandbox_enabled` | Whether sandbox policy is turned on. |
| `backend` | The configured backend Koder will use when sandboxing is enabled. |
| `backend_available` | Whether that backend can run on this machine right now. |
| `backend_reason` | Why the backend is available or unavailable. |
| `mode` | The high-level filesystem mode. The default enabled mode is `workspace-write`. |
| `network_policy_enforcement` | Whether the selected backend enforces network policy. `unsupported` means network settings are metadata only. |
| `allowed_domains` / `denied_domains` | Domain lists from policy. Marked `(policy metadata, not enforced)` because no current backend enforces them. |
| `protected_paths` | Metadata paths Koder blocks before sandbox execution. |
| `settings_path` | The project-local settings file Koder writes when you run `/sandbox enable` or `/sandbox disable`. |
| `backend_options` | The supported backend names. |

When sandbox is enabled, Koder uses the configured backend. Availability alone
does not prove that the backend enforces every requested policy. Missing
guarantees require a separate, explicit degradation approval; a generic command
approval does not accept those losses. If the backend is unavailable, host
fallback is denied unless an exact, state-bound fallback approval is given.
There is no silent fallback.

## What Is Sandboxed

The first sandboxed execution path is foreground `run_shell` commands and the child processes they start. This covers model-requested shell commands that go through Koder's foreground shell executor.

These surfaces are still protected by permissions, but are not currently routed through a sandbox backend:

| Surface | Current behavior |
|---|---|
| File tools such as `read_file`, `write_file`, and `edit_file` | Permission and workspace-root checks. |
| Background shell commands | Denied while sandbox is enabled. The denial message suggests running in the foreground, adding a `/sandbox exclude` rule, or `/sandbox disable`. |
| PowerShell commands | No sandbox backend is used. While sandbox policy is enabled, host execution requires a separate, exact state-bound fallback approval in addition to ordinary tool authorization; otherwise it is blocked. |
| MCP servers | Deliberately permission-only. MCP server processes are not routed through sandbox backends. |
| Skills | Skill metadata is read normally. Helper shell commands only inherit sandboxing when they go through foreground `run_shell`. |
| Agent and teammate processes | The processes themselves are not sandboxed. Their foreground shell tool calls still use the same shell executor and configured sandbox policy; other tool surfaces retain their own permission checks. |

Note that commands matched by a `/sandbox exclude` rule bypass these sandbox denials entirely and fall back to the normal permission flow.

Explicit TUI `!` commands use the same shell executor, but the manual path does
not supply a sandbox-degradation approver. If the selected backend needs
degradation or host fallback, that command is blocked instead of silently
accepting the missing guarantees. Inspect `/sandbox status` before changing the
policy; a direct user command is not implicit consent to weaker confinement.

## Request Cleanup And Outcomes

Each foreground sandbox request owns its session from creation through context
exit and provider deletion. SDK context exit can persist workspace state and
close dependencies; provider deletion is a separate operation. Cancellation
stops further command startup and waits for owned cleanup. Repeated cancellation
does not detach deletion or interrupt an already-started context exit.

Once a command has run, cleanup failure does not erase its output or actual
command exit code. The overall shell result is still an error. In print mode,
a successful command followed by failed cleanup exits Koder with code 1 and
marks JSON/stream-JSON output as an error. An existing nonzero command exit code
is retained. A completed sandbox command is not replayed on the host to recover
from cleanup failure.

Read the execution flags as lifecycle observations: `created` records that a
session handle was returned, and `executed` records that command execution was
attempted. They are not an exactly-once guarantee or evidence that cancellation
undid completed effects. A provider that allocates a resource but fails before
returning its handle may leave state Koder cannot confirm or delete.

Provider deletion has one five-second timeout. The adapter cancels a timed-out
delete and waits for it to settle; a noncooperating SDK operation can therefore
outlast that timeout. Context exit also depends on the SDK completing its own
cleanup. Koder does not promise forced cleanup after process death or for a
provider that never returns. A reported cleanup failure means provider resources
may remain; inspect that provider before retrying work with external effects.

## Configuration

`/sandbox enable <backend>` writes `.koder/settings.local.json` in the current project. A typical enabled config looks like this:

```json
{
  "sandbox": {
    "enabled": true,
    "mode": "workspace-write",
    "backend": "unix-local"
  }
}
```

You can also set sandbox policy in these files:

| File | Typical use |
|---|---|
| `~/.koder/settings.json` | User default policy. |
| `.koder/settings.json` | Shared project policy. |
| `.koder/settings.local.json` | Local project override, usually not committed. |
| `~/.koder/managed-settings.json` | High-priority local managed policy file. |

Older backend aliases are normalized to the current backend names in status output.

## Excluded Commands

Excluded commands bypass sandbox execution and return to the normal permission flow. Use this only for commands that cannot run inside the sandbox and that you still want Koder to evaluate explicitly.

```bash
/sandbox exclude "touch *"
```

An excluded mutating command should still require approval unless your permission rules allow it. Exclusions are not a blanket allowlist.

## Filesystem Policy

The default protected metadata paths are:

```text
.git
.koder
.agents
.codex
```

Koder checks directly visible write targets under those paths before starting
a sandboxed command. This preflight is best effort: scripts, indirect paths and
filesystem changes after checking are not a race-free enforcement boundary.
The selected backend must supply any required process/filesystem confinement.
On macOS, `unix-local` uses `/usr/bin/sandbox-exec` for its write boundary, but
Koder does not treat availability as proof of complete host, repository-sync,
protected-subpath or network isolation.

## Network Policy

Sandboxed shell network access defaults to disabled in policy. `allowedDomains` and `deniedDomains` are accepted in settings and shown in status, but no current backend enforces them: `/sandbox status` reports `network_policy_enforcement: unsupported` for `unix-local` and labels the domain lists as `(policy metadata, not enforced)`. The `networkAccess` flag itself is also not enforced by the `unix-local` backend. Koder does not currently provide a domain proxy for local shell commands.

Shell network policy is separate from model-provider access and Koder's own web search or web fetch tools.

## Verify A Backend

Developers and advanced users can run no-model smoke checks from the repository:

```bash
uv run scripts/sandbox_backend_smoke.py --backend unix-local
uv run scripts/sandbox_backend_smoke.py --backend unix-local --case protected-paths
uv run scripts/sandbox_backend_smoke.py --all --skip-unconfigured
```

Inspect the result of each selected case: the suite includes `pwd`, workspace
writes, escape attempts and timeout checks. A passing single case does not prove
the others, and skipped or unconfigured cases are not passing runtime evidence.

## Troubleshooting

If sandbox commands are denied unexpectedly, run:

```bash
/sandbox status
/sandbox backends
/permissions
```

Common states:

| Symptom | Meaning | What to do |
|---|---|---|
| `backend_available: false` | The configured backend cannot run right now. | Install or configure that backend, or select another one with `/sandbox enable <backend>`. |
| `backend_reason: unknown backend` | Settings request a backend id Koder does not know. | Change `sandbox.backend` to one listed by `/sandbox backends`. |
| A mutating command is auto-allowed | It may be allowed by ordinary permission rules. Sandbox-based auto-approval additionally requires proven host/workspace isolation, repository synchronization and every active policy guarantee. | Inspect the permission decision; neither `backend_available` nor `autoAllowBashIfSandboxed` alone grants approval. |
| Hosted backend says dependencies or credentials are missing | The optional provider is not configured. | Install the named provider package, set the named environment variable, or use `unix-local`. |
| Network domain lists do not block a local command | Domain filtering is policy-only for the selected backend. | Treat domain lists as status metadata until an enforcing backend or proxy is available. |

Choose a backend for the guarantees the task needs. Confirm availability and
the reported enforcement limits, and read any degradation/fallback approval
before accepting it. The current backend registry does not claim all guarantees
required for sandbox-based auto-approval.
