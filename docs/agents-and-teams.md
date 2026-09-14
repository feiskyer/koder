# Agents And Teams

Koder can delegate work to local background agents and teams. Use this when a task can be split into focused investigations, reviews, or implementation branches while the main session keeps coordinating.

## Project And User Agents

Agents are loaded from:

1. `.koder/agents/` in the current project
2. `~/.koder/agents/` for user-level agents

Inspect them with:

```bash
/agents
/agents summary
```

Use a specific agent from the CLI:

```bash
koder --agent reviewer "review the current diff"
```

Project agents are useful for repeated roles such as reviewer, test fixer, documentation writer, or release assistant.

## Background Subagents

Use `/fork` for a background subagent:

```bash
/fork "investigate why the import-order test is failing"
/fork --context "review the current diff for docs regressions"
```

Default subagent context is isolated from the main agent. The main session receives the result and can inspect runtime summaries. When you need the subagent to see current conversation context, pass an explicit context-bearing mode or prompt that names the files and facts it needs.

The model-facing `agent_tool(context="fork")` snapshots the active parent
Session's stored conversation, including when that Session uses a non-default
database. Incomplete Chat Completions and Responses tool-call pairs are filtered
before seeding the child. A missing, retired or unreadable parent produces an
error instead of silently launching without context; a valid empty parent is
still allowed. The child uses its own Session and closes it on completion,
failure or cancellation.

Subagents inherit the active model configuration, including model name, base URL, reasoning effort, and OAuth-style provider routing where supported. That keeps background work on the same provider family as the main session unless you intentionally override it.

Model tools and interactive commands share the CLI run's agent service. Named
agents therefore remain addressable across consecutive tool calls and main
session switches in that CLI run. A duplicate/ambiguous name is not a reliable
identifier; use the returned full agent ID.

`send_message` acknowledges queueing, not completion of the requested work.
Running agents process queued user-level messages at run boundaries using the
same agent conversation. Messages for a stopped agent remain pending until an
explicit `agent_tool` call supplies `resume` with its ID or registered name.
One execution drains a bounded batch of follow-ups; remaining work stays queued
with a delayed state. Interrupted setup or model work still needs explicit
recovery and is not an exactly-once external-effect guarantee.

Execution leases prevent cooperating runtimes from concurrently resuming the
same agent. An unowned historical record is not a live delivery handle. Waiting
can be cancelled without cancelling the observed agent; cancelling an already
completed agent does not erase its result.

On CLI exit, teammate consumers stop before the shared agent service joins its
owned work. A retired main scheduler does not close that shared service while
the CLI run continues. Embedded callers that supply their own shared service are
responsible for its asynchronous shutdown.

An agent definition with `isolation: worktree` in its frontmatter runs in its own
Git worktree under `.koder/worktrees/`. Relative file, notebook and search paths,
code-intelligence requests, and shell subprocesses use that agent's execution
directory. In-process agents keep separate task-local directories; Koder does
not change the whole process's cwd to switch between them. Permission checks,
sandbox snapshots and project extension lookup use the same execution directory.

A worktree separates checkouts; it is not an operating-system sandbox. Explicit
absolute paths and subprocess behavior still depend on the configured permission
rules and sandbox backend. See [Permissions And Privacy](permissions-and-privacy.md).

Automatic cleanup preserves uncommitted or ignored files, unmerged commits, and
checkouts whose ownership or clean state cannot be established. It does not
force-remove a worktree or reset an existing branch. A later run can reuse a
preserved worktree or create a new isolated worktree if the previous clean one
was removed. See [Hooks](hooks.md) for the `SubagentStart`, `SubagentStop`,
`WorktreeCreate`, and `WorktreeRemove` events.

Interactive `enter_worktree` / `exit_worktree` handles belong to the runtime that
created them. Another main session or subagent cannot consume that handle, even
if its displayed identity is the same. Session switches and background resumes
that reuse the retained runtime keep the owner's handle; creating a new runtime
or restarting the process does not automatically restore an interactive handle.
Creating a worktree returns its path rather than changing every subsequent tool's
directory implicitly.

## Task Delegate Tool

During normal model execution, Koder may expose `task_delegate` for bounded background work. Treat it like `/fork`: delegate concrete tasks with clear outputs, keep the main session responsible for integration, and avoid asking a subagent to guess unstated context.

Good prompts:

```text
Inspect docs/configuration.md and report any provider setup gaps. Do not edit files.
```

```text
Update only tests/test_sessions.py to cover resume-by-title ambiguity. Run the focused test and report changed files.
```

## Agent Teams

Use `/peers` for team workflows:

```bash
/peers create migration-review
/peers spawn migration-review general-purpose reviewer "check the docs links"
/peers spawn migration-review general-purpose tester "run focused docs tests"
/peers inbox migration-review
/peers history migration-review
/peers task list migration-review
```

Teams provide local records for members, mailbox messages, task history, and shared memory. They are useful for multi-agent discussion, coordinator-plus-reviewer workflows, or repeated project teams.

### Model-facing team tools

`team_create(team_name="migration-review")` selects that team for the current
main Session. The selection is owned by the running AgentService, so it survives
separate SDK tool-call tasks without leaking into another main Session or
AgentService. It is not a process-restart recovery mechanism.

`agent_tool(team_name="migration-review", name="reviewer", ...)` registers a real
member and uses the same in-process teammate runner as interactive commands.
Model, plan-mode, fork-history and worktree inputs are forwarded to execution.
With `run_in_background=True`, the tool returns after launch; otherwise it waits
for the initial result. In both cases the teammate remains available for follow-up
mail and shared tasks until it is shut down. Disabling background tasks prevents
team launches rather than silently creating an unrelated ordinary agent.

In a selected team, `send_message` uses the team's durable mailbox. A running
member sends with its own identity, not the leader's. Resuming a live teammate
queues another unit of team work through its existing consumer. Resume requires
the recorded agent definition, model and isolation settings, and keeps the
member's name, conversation and permission mode. Incompatible changes are
rejected before queueing; launch a new member to use a different definition.
An acknowledgement of a queued message is not proof that the work has finished.
If a mailbox follow-up cannot start, its failed outcome is retained in team
history and reported to the leader before the consumer stops.

`team_delete()` deletes the selected team only after all members have shut down.
For interactive shutdown, use `/peers shutdown approve <team-id> <agent-id>`.
Idle teammates still count as live members. CLI/service shutdown also joins
their consumers before releasing agent resources. A saved selection cannot
silently act on a different team created later with the same name.

## Teammate Modes

Koder supports two teammate execution modes:

| Mode | Start With | Best For |
|---|---|---|
| `in-process` | `koder --teammate-mode in-process` | Fast local teammate execution inside the Koder process. This is the default for ordinary team work. |
| `tmux` | `koder --teammate-mode tmux` | Visible teammate panes, debugging team UX, or watching separate agents work in real terminal sessions. |

Keep `tmux` for cases where you need terminal-pane visibility. Use the default in-process mode for most user workflows.

## Team Memory

Team memory can be synchronized between project-local files and runtime state:

```bash
/peers memory <team-id> sync
```

Project memory path:

```text
.koder/team-memory/<team-id>/
```

Runtime memory path:

```text
~/.koder/teams/<team-id>/memory/
```

Use team memory for decisions, shared constraints, investigation findings, or handoff notes that several teammates should see.

## Inspect And Clean Up

Useful commands:

```bash
/agents summary
/tasks
/peers history <team-id>
/peers inbox <team-id>
/peers shutdown approve <team-id> <agent-id>
/peers cleanup <team-id>
```

Persistent runtime state normally lives under `~/.koder/agents/`, `~/.koder/teams/`, and `~/.koder/tasks/`. Tests and temporary harnesses should use temporary directories instead of writing product state into the repository root.
