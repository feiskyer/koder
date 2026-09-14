# Hooks

Hooks run your own commands automatically when specific events happen inside a Koder session: before a tool runs, after a file edit, when a session starts, when compaction happens, and more. Use them to enforce policy, run formatters, write audit logs, or notify external systems without asking the model to remember anything.

Inspect the active configuration at any time:

```bash
/hooks
```

## Where Hooks Are Configured

Hooks live under a `hooks` key in Koder settings files. All matching scopes run; identical hooks are deduplicated within one event dispatch.

| File | Scope |
|---|---|
| `~/.koder/settings.json` | User, all projects |
| `.koder/settings.json` | Project, committed to git |
| `.koder/settings.local.json` | Project, this machine only (gitignored) |
| `~/.koder/managed-settings.json` | Managed local policy, highest priority |
| Plugin `hooks/hooks.json` | Contributed by an enabled plugin |
| Skill frontmatter `hooks:` | Active only while that skill runs |

Set `"disableAllHooks": true` in user or project settings to disable everything except managed-policy hooks; setting it in `managed-settings.json` disables all hooks. `koder --bare` skips the `SessionStart`, `Setup`, and `InstructionsLoaded` events (no `AGENTS.md` is loaded) — other hook events still fire; use `disableAllHooks` to turn hooks off.

## Configuration Shape

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "edit_file|write_file",
        "hooks": [
          { "type": "command", "command": "npx prettier --write ." }
        ]
      }
    ]
  }
}
```

Each event maps to a list of groups. A group has an optional `matcher` and a list of hooks. Every hook receives the event payload as JSON on stdin.

Command input is a complete JSON document, not an interactive input stream.
Koder prepares it in a private temporary file so timeout and cancellation
polling cannot interrupt delivery to slow readers. The input file is closed
on success, failure, timeout, or cancellation; stdout and stderr remain pipes.

### Matchers

`matcher` is a regular expression tested against the event's match value — usually the tool name (`run_shell`, `edit_file`), and for some events another identifier (agent type for `SubagentStart`/`SubagentStop`, `auto`/`manual` for compaction events, the watched file name for `FileChanged`). An empty matcher or `"*"` matches everything.

### Hook Fields

| Field | Meaning |
|---|---|
| `type` | `command`, `http`, `prompt`, or `agent`. |
| `command` | Shell command for `command` hooks. Payload arrives on stdin. |
| `url` | Endpoint for `http` hooks. Payload is POSTed as JSON. |
| `prompt` | Instruction for `prompt` (model-only) and `agent` (model with tools) hooks. |
| `timeout` | Seconds before the hook is killed (command/http). |
| `shell` | Run the command via a specific shell, e.g. `"bash"`. |
| `if` | Extra argument-level condition, e.g. `"run_shell(git push*)"` — tool pattern plus a glob over the command/path argument. |
| `once` | `true` fires this hook at most once per session. |
| `async` | `true` runs the command in the background without blocking the turn. |
| `headers`, `allowedEnvVars` | For `http` hooks: extra headers, with `${VAR}` expansion limited to the allowed list. |
| `model` | Model override for `prompt`/`agent` hooks. |

### Hook Environment

Command hooks inherit your environment plus:

| Variable | Content |
|---|---|
| `KODER_PROJECT_DIR` | Absolute path of the dispatching workspace. |
| `KODER_SESSION_ID` | Active session id, when the payload carries one. |
| `KODER_ENV_FILE` | Path to the session env file, provided to `SessionStart` hooks; write `export` lines to it to inject session-scoped variables into the session. |
| `KODER_SKILL_DIR` | Skill/plugin root, for hooks contributed by a skill or plugin. |

## Events

Every payload includes an `event` field with the event name.

### Tool lifecycle

| Event | Fires | Match value | Payload extras |
|---|---|---|---|
| `PreToolUse` | Before each tool call. | Tool name | `tool_name`, `tool_input` |
| `PostToolUse` | After each tool call completes (on failure, `result` carries the error text and `PostToolUseFailure` fires first). | Tool name | `tool_name`, `tool_input`, `result` |
| `PostToolUseFailure` | When a tool raises an error. | Tool name | `tool_name`, `tool_input`, `error` |
| `PermissionRequest` | A guarded tool call needs approval. | Tool name | `tool_name`, `tool_input`, `reason` |
| `PermissionDenied` | A guarded tool call is denied. | Tool name | `tool_name`, `tool_input`, `reason` |
| `Notification` | Approval prompt notifications. | `permission_prompt` | `notification_type`, `tool_name`, `reason` |

A `PermissionRequest` hook can resolve the approval itself by printing a decision (see Hook Output below).

### Session lifecycle

| Event | Fires | Payload extras |
|---|---|---|
| `SessionStart` | Once at startup (skipped with `--bare`). Blocking aborts startup. | `source` (`startup`/`resume`), `session_id` |
| `SessionEnd` | On session exit. | `session_id` |
| `UserPromptSubmit` | On every submitted prompt. Blocking rejects the prompt. | `prompt`, `session_id` |
| `Stop` | When the agent finishes a turn. Blocking halts the turn with an error carrying the block reason. | `agent_type`, `last_assistant_message`, `stop_hook_active` |
| `StopFailure` | When a turn fails with an error. | `last_assistant_message` (the error text), `session_id` |
| `Setup` | At startup when onboarding steps are missing. | `missing_steps` |

### Context and workspace

| Event | Fires | Payload extras |
|---|---|---|
| `PreCompact` / `PostCompact` | Around conversation compaction. Matcher is the trigger: `manual` (/compact) or `auto`. | `trigger`, `session_id`, compaction stats |
| `InstructionsLoaded` | When `AGENTS.md` is loaded at session start. Blocking skips the content. | `reason`, `file_path` |
| `CwdChanged` | When the working directory changes (e.g. `/resume` restores a session's recorded directory). | `old_cwd`, `cwd` |
| `FileChanged` | When a watched file changes (register paths via `watchPaths`, polled between turns). Matcher is the file name. | `file_path` |
| `ConfigChange` | When runtime config is saved (e.g. `/brief`) or a settings bundle imports changed configuration files. A synchronous block triggers rollback. Matcher is `user_settings`, `project_settings`, or `local_settings`. | `source`, `file_path` |

For `config import`, all selected entries are validated first. The complete
candidate bundle is then published, and one `ConfigChange` event is dispatched
per changed configuration file using the hook definitions captured **before**
the import. User config, settings and keybindings match `user_settings`; project
settings match `project_settings`, and project-local settings match `local_settings`.
The `file_path` is the absolute destination path. Imported memories do not emit
this event, and dry-run or unchanged imports execute no hooks.

An imported hook definition or `disableAllHooks` cannot judge its own import or
remove an existing veto. A blocking hook or raised exception rolls back the
applied batch, preserving any detected newer external file and reporting
incomplete rollback. The snapshot does not freeze arbitrary scripts referenced
by hooks, extend project approvals to new hook content, or undo hook side effects.
Existing async and once-only behavior is unchanged; async hooks do not veto
configuration changes.

### Agents, teams, and worktrees

| Event | Fires | Match value | Payload extras |
|---|---|---|---|
| `SubagentStart` / `SubagentStop` | Around a subagent run (`/fork`, `@agent` mentions, agent-definition runs). | Agent type | `agent_type`, `output` (stop) |
| `TaskCreated` / `TaskCompleted` | Team task lifecycle (`/peers` tasks). Blocking reverts the task change. | Task id | `task` fields |
| `TeammateIdle` | A teammate becomes idle. | Teammate name | `team_name`, `agent_id`, `agent_name` |
| `WorktreeCreate` | An isolated agent worktree is created. A hook can print `worktreePath` to relocate it. | — | `branch`, `worktree_path` |
| `WorktreeRemove` | A clean agent worktree is removed after completion. | — | `worktree_path` |

### MCP interactions

| Event | Fires | Payload extras |
|---|---|---|
| `Elicitation` | An MCP server requests user input. A hook can auto-respond (see below). | `message`, `mode`, `requestedSchema` |
| `ElicitationResult` | After the elicitation resolves. | `action`, `source` (`hook`/`user`), `field_names` |

## Hook Output

Command hooks communicate back through their exit code and stdout.

**Exit codes:** `0` means success. `2` blocks the triggering action (stderr becomes the block reason). Anything else is a non-blocking failure.

**Structured stdout:** print a JSON object to control behavior precisely:

```json
{
  "hookSpecificOutput": {
    "decision": { "behavior": "deny", "message": "pushes are release-managed" },
    "watchPaths": ["docs/spec.md"],
    "worktreePath": "/custom/worktree/location",
    "action": "accept",
    "content": { "name": "auto-filled" }
  }
}
```

| Key | Used by | Effect |
|---|---|---|
| `decision` | `PermissionRequest` (and blocking events) | `{"behavior": "allow"}` approves the call; `{"behavior": "deny", "message": ...}` denies it. |
| `watchPaths` | `SessionStart`, `InstructionsLoaded`, `CwdChanged`, `FileChanged` | Registers files for `FileChanged` polling. |
| `worktreePath` | `WorktreeCreate` | Overrides the worktree location. |
| `action` / `content` | `Elicitation` | Auto-responds `accept`/`decline`/`cancel` with optional form content. |

The top-level form `{"decision": "block", "reason": "..."}` also blocks.

## Examples

Format after every file edit:

```json
{
  "hooks": {
    "PostToolUse": [
      { "matcher": "edit_file|write_file",
        "hooks": [ { "type": "command", "command": "npx prettier --write .", "timeout": 120 } ] }
    ]
  }
}
```

Block pushes from Koder sessions:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "run_shell",
        "hooks": [ { "type": "command",
                     "if": "run_shell(git push*)",
                     "command": "echo 'pushes are done from CI' >&2; exit 2" } ] }
    ]
  }
}
```

Audit-log every shell command:

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "run_shell",
        "hooks": [ { "type": "command",
                     "command": "python3 -c \"import sys,datetime;open('.koder/shell-audit.log','a').write(datetime.datetime.now().isoformat()+' '+sys.stdin.read()+'\\n')\"" } ] }
    ]
  }
}
```

Notify a webhook when a session ends:

```json
{
  "hooks": {
    "SessionEnd": [
      { "hooks": [ { "type": "http", "url": "https://example.internal/koder-done",
                     "headers": { "Authorization": "Bearer ${WEBHOOK_TOKEN}" },
                     "allowedEnvVars": ["WEBHOOK_TOKEN"] } ] }
    ]
  }
}
```

Auto-answer MCP elicitations in CI:

```json
{
  "hooks": {
    "Elicitation": [
      { "hooks": [ { "type": "command",
                     "command": "echo '{\"hookSpecificOutput\":{\"action\":\"accept\",\"content\":{\"confirm\":true}}}'" } ] }
    ]
  }
}
```

## Troubleshooting

| Symptom | Likely cause | Check |
|---|---|---|
| Hook never fires | Wrong event name or matcher | `/hooks` lists what is loaded; event names are case-sensitive. |
| Hook fires but nothing blocks | Exit code is not 2 and stdout has no deny decision | Print the structured decision or `exit 2`. |
| No hooks run at all | `disableAllHooks` is set | `/hooks` and your settings files. |
| Hook output ignored | stdout is not valid JSON | Only structured JSON stdout is interpreted; other output is treated as text. |
| Session vars not visible | Wrong event | `KODER_ENV_FILE` is provided to `SessionStart` hooks. |

Hook payloads stay on your machine: command hooks run locally, and only `http` hooks send the payload to the URL you configure.
