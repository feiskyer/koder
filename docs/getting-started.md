# Getting Started

This guide gets Koder from install to a useful first coding session.

## Install

Use `uv tool install` for a clean command-line install:

```bash
uv tool install koder
```

Koder requires Python 3.11 or newer.

For local development from this repository:

```bash
uv sync
uv run koder
```

## Configure A Model

The shortest setup is a universal API key and a model name:

```bash
export KODER_API_KEY="your-api-key"
export KODER_MODEL="gpt-4o"
koder
```

`KODER_API_KEY`, `KODER_BASE_URL`, `KODER_MODEL`, and `KODER_REASONING_EFFORT` override provider-specific settings and `~/.koder/config.yaml`.

Provider-specific examples:

```bash
OPENAI_API_KEY="sk-..." KODER_MODEL="gpt-4o" koder
ANTHROPIC_API_KEY="..." KODER_MODEL="claude-opus-4-20250514" koder
GOOGLE_API_KEY="..." KODER_MODEL="gemini/gemini-2.5-pro" koder
KODER_BASE_URL="http://localhost:8080/v1" KODER_MODEL="openai/local-model" koder
```

Subscription-backed providers use `koder auth`:

```bash
koder auth login google
koder auth login claude
koder auth login chatgpt
koder auth login antigravity
koder auth list
```

For the manual authorization-code flow, paste the entire displayed code,
including any state suffix. A supplied state suffix must match the current
login request; a mismatch is rejected before token exchange. Codes without a
suffix retain the local-verifier compatibility path. Empty or closed input fails authentication.
Timeout and cancellation release the input reader before the flow returns;
they do not leave a background `input()` consuming later terminal input.

GitHub Copilot device login also honors `--timeout`. It polls asynchronously,
respects device-code expiry and slow-down responses, and rejects non-positive
or non-finite deadlines. Cancellation during HTTP work closes the owned clients
and does not start a background cache writer. The existing LiteLLM cache paths
and file formats are retained.

Once final cache publication has begun, cancellation waits for the file work
to finish; the login may already have taken effect. Each cache file is replaced
atomically, but the two files are not one transaction across process failure.

After login, use the provider prefix in `KODER_MODEL`, for example `google/gemini-3-pro-preview`, `claude/claude-opus-4-5-20250514`, or `chatgpt/gpt-5.2`.

See [Configuration Guide](configuration.md) for the full provider matrix.

## Pick A Runtime Style

Koder can be used in three common ways:

| Style | Command | Use it when |
|---|---|---|
| Interactive TUI | `koder` | You want normal coding work with streamed output, slash commands, shell mode, file mentions, and resume. |
| Single prompt | `koder "summarize the current git diff"` | You know the task and want one recorded turn from your shell. |
| Print mode | `koder --print "summarize"` | You want script-friendly output for automation or logs. |

### Automation Results

Main-agent prompts in single-prompt and print mode exit with `0` on success,
`1` on a reported execution/preflight failure, and `130` on a reported
cancellation. Returned failures in JSON or JSONL result records include
`"is_error": true` and a numeric `exit_code`; successful result records keep their
existing shape. Use these statuses rather than matching words in `result`.

Foreground `!` shell commands preserve the child exit code (or `128 + signal`
when terminated by a signal). Permission rejection and an empty shell command
return `1`. A background launch returning `0` means it was accepted, not that
the background work has finished.

When a response fails `--json-schema` validation, print mode returns a JSON error
record and exits with `1`. A failed or cancelled model turn is not validated as
successful structured output, so its original diagnostic remains available.

## Start A Session

Interactive mode is the normal daily workflow:

```bash
koder
```

Single prompt mode is useful from scripts or when you already know the task:

```bash
koder "summarize the current git diff"
```

Named sessions keep a durable conversation attached to a project or topic:

```bash
koder -s billing-refactor
koder -s billing-refactor "continue the failing test investigation"
```

Resume previous work with:

```bash
koder --resume
koder --continue
```

## First Workspace Check

Inside a project, run these commands before asking for large edits:

```bash
/onboarding
/status
/model
/files
/permissions
```

They show the active provider, session, workspace directory, loaded context, and permission policy. If something looks wrong, fix it before delegating substantial work.

## A Safe First Task

Try a read-only task first:

```bash
koder "inspect this repo and explain the test command"
```

Then move to a small edit:

```bash
koder "fix the failing test in tests/test_example.py and run the focused test"
```

Koder will use the project instructions from `AGENTS.md` when present, and it stores session state locally under `~/.koder/`.

Before allowing larger edits, ask Koder to show the boundary:

```bash
koder "report the current workspace, loaded project instructions, configured model, and whether any sandbox backend is active. Do not edit files."
```

Then run a small implementation task with explicit verification:

```bash
koder "fix the smallest failing test you can identify, run only the focused test first, and summarize changed files."
```

## What To Read Next

- [Interactive TUI](interactive-tui.md) for keyboard controls and slash commands.
- [Sessions and Memory](sessions-and-memory.md) for durable context and cleanup.
- [Workflows](workflows.md) for review, planning, Git, and PR workflows.
- [Permissions and Privacy](permissions-and-privacy.md) for local data and tool approval behavior.
