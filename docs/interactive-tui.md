# Interactive TUI

Interactive mode is the main Koder experience. Start it with:

```bash
koder
```

The TUI combines a prompt, live model output, command completion, shell mode, file mentions, session state, and local status information.

## Prompt Controls

Common controls:

| Control | Action |
|---|---|
| `Enter` | Send the current prompt. |
| `Shift+Enter` (`Ctrl+J` or `Alt+Enter` fallback) | Insert a newline for multi-line prompts. |
| `/` | Open slash-command completion. |
| `@` | Mention files, agents, or MCP resources when available. |
| `Ctrl+R` | Search prompt history. |
| `Up` / `Down` | Navigate recent prompt history. |
| `Right` or `Tab` | Accept a visible ghost suggestion. |

Use `/clear` when you want a fresh session state without carrying the previous prompt history forward.

## Slash Commands

Slash commands are the fastest way to inspect or control the runtime. Type `/` to complete commands or run `/help` for the full list.

Useful first commands:

```bash
/help
/status
/model
/config
/files
/diff
/usage
/permissions
```

Command families:

| Family | Examples |
|---|---|
| Session | `/session`, `/resume`, `/rename`, `/clear`, `/export` |
| Model and UI | `/model`, `/effort`, `/theme`, `/color`, `/vim`, `/statusline`, `/output-style` |
| Context | `/files`, `/context`, `/ctx_viz`, `/summary`, `/insights`, `/doctor` |
| Workflows | `/review`, `/security-review`, `/advisor`, `/brief`, `/branch` |
| Memory and goals | `/memory`, `/remember`, `/thinkback`, `/thinkback-play`, `/compact`, `/rewind`, `/goal` |
| Agents and loops | `/agents`, `/fork`, `/peers`, `/tasks`, `/loop`, `/schedule` |
| Extensions | `/skills`, `/plugin`, `/reload-plugins`, `/mcp`, `/channels`, `/magic-docs` |
| Permissions | `/permissions`, `/sandbox`, `/add-dir` |
| GitHub | `/pr-comments`, `/release-notes`, `/subscribe-pr`, `/autofix-pr` |

See [Command Reference](commands.md) for the complete catalog.

## Shell Mode

Prefix a line with `!` to run a shell command from the TUI:

```bash
!git status --short
!uv run pytest tests/test_file_tools.py
```

Use `&` for a background command:

```bash
!uv run python -m http.server 8000 &
```

Background commands can be inspected or stopped with shell tools in the agent runtime. Mutating commands may trigger permission checks depending on your policy.

## Mentions And Context

Use `@` to mention available workspace files, agents, and MCP resources. Mentions make the intended context explicit and help avoid asking the model to rediscover a path.

Helpful context commands:

```bash
/files
/context
/ctx_viz
/magic-docs
```

`/files` shows tracked session files. `/context` and `/ctx_viz` show what Koder is injecting into the active turn.

## Status And Usage

The status line can show model, session, usage, and workspace details. These commands are useful when you suspect a stale model, session, or cost display:

```bash
/status
/usage
/cost
```

Changing the model updates the status label without clearing accumulated usage.
Each request is attributed and priced using its request model; selecting another
model does not reprice earlier tokens. Unknown pricing is shown as unavailable
(`$?`), including sessions that mix known and unknown prices. Known amounts remain
a subtotal, not a complete vendor bill.

An explicit context-window override is used by both the default status line and
custom status commands, including for custom model names.

During streaming, `Escape` cancels the active turn. In fixed-bottom mode, partial
output and the cancellation notice remain in the transcript, and the next prompt
can continue the same session. A later turn waits for pending streamed-session
persistence before starting. Cancellation does not mean that already-completed
external side effects have been undone.

You can customize terminal style with:

```bash
/theme dark
/color blue
/vim on
/statusline clear
/output-style
```

## Voice Input

When voice mode is enabled, double-tap `Space` to start recording. Press `Space` or `Enter` to stop. Koder transcribes the audio and sends the transcript as the user message.

Configure it with:

```bash
/voice
/voice status
/voice provider openai
```

See [Voice Mode](voice-mode.md) for provider setup and troubleshooting.
