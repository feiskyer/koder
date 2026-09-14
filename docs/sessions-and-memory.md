# Sessions And Memory

Koder keeps conversation state and local memory so you can resume work without rebuilding context by hand.

## Local Storage

Default storage paths:

| Path | Purpose |
|---|---|
| `~/.koder/koder.db` | SQLite session metadata and transcripts. |
| `~/.koder/koder.db` (`session_goals`) | Durable goal state for each session. |
| `~/.koder/memory/` | User-level memories. |
| `.koder/memory/` | Project-level memories. |
| `.koder/session-memory/` | Project-local session notes. |
| `~/.koder/scheduled_tasks.json` | Cron-backed loop and scheduled prompt records. |
| `~/.koder/tasks/` | Runtime task records, including maintenance tasks. |

Koder does not require a hosted session service for these features.

The active CLI store is `EnhancedSQLiteSession` over the Agents SDK's SQLite
tables. The separate synchronous `TranscriptStore` and snapshot/restore helpers
are explicit-call components, not an automatically selected fallback backend.
The CLI does not automatically create `.bak` snapshots or restore one over the
primary database. Snapshot restoration requires closed primary connections and
coordinated writers; it must not be attached to live-session startup casually.

Within managed runs, file-read deduplication and read-before-write evidence
belong to the actual Session object, not a process-global cache or session-id
string. Another Session, including a fork or a newly opened handle, must obtain
its own fresh reads. A retired or closed managed binding cannot fall back to
standalone helper state.

Session-metadata operations and legacy MCP migration wait for their SQLite
connections and worker threads to close before returning. This also applies
when opening fails or the caller is cancelled repeatedly. Cancellation can
therefore wait for database work already running on the worker to finish.

Concurrent opens of an older metadata schema serialize the migration inside a
SQLite transaction and recheck columns after acquiring the writer lock. Session
listing uses the same migration path as cwd, agent and title operations.
Cancellation before the migration commits rolls it back before the connection
is released; a later opener can retry without a partially published upgrade.

## Named Sessions

Use named sessions when you want durable work streams:

```bash
koder -s api-migration
koder -s api-migration "continue the serializer cleanup"
koder --resume
koder --continue
```

Inside the TUI:

```bash
/session
/rename api-migration
/resume api-migration
/clear
```

`/clear` switches to a fresh workflow state. The previous named session remains resumable unless you explicitly remove local state outside Koder.

When `/resume` switches sessions inside the TUI, Koder also restores the working directory recorded for that session (and fires the `CwdChanged` hook, see [Hooks](hooks.md)) so the resumed session continues where it actually ran.

Session discovery includes the SDK's stored sessions and messages, legacy
`items` records, and Koder metadata. A generated title or metadata row is not
required to find and resume an existing conversation by its session id.

## Inspecting Session State

Use these commands to understand what the active session contains:

```bash
/status
/summary
/insights
/usage
/cost
/files
/context
/ctx_viz
```

`/summary` is a compact local status report. `/insights` focuses on transcript roles, tool activity, context files, and usage counters.

## Legacy Session Import

Legacy `ctx` histories, also importable with `/backfill-sessions`, are copied with
their completion marker in one database transaction. Failed writes do not leave
a partial import marked complete. A retry preserves an already-imported history
prefix, any newer messages, and user-renamed titles. Conflicting or malformed
histories abort the import with an error instead of silently merging or skipping
them; the original `ctx` records remain intact for inspection and repair.

During automatic startup import, these legacy-data errors produce a warning on
stderr and defer the import without blocking unrelated modern sessions. The
explicit `/backfill-sessions` command still reports the import failure. Database
I/O errors, unexpected failures, and cancellation are not silently ignored.

## Goals

Use goals when a session has a concrete long-running objective that should persist across turns:

```bash
/goal improve benchmark coverage --budget 50000
/goal
/goal pause
/goal resume
/goal edit improve benchmark and scheduler coverage
/goal budget 75000
/goal clear
```

Goals track objective text, status, elapsed time, token usage, and an optional token budget. Active goals can trigger continuation turns until the goal is completed, paused, blocked, or budget-limited.

Goal writes and their returned state snapshot share a transaction. If a write
fails or is cancelled, any pending changes are rolled back before the connection
can be reused; a later operation cannot accidentally commit them. Cancellation
racing an already-completed commit does not undo that commit. Connection setup
and shutdown also wait for owned database cleanup before releasing resources.

## Scheduled Loops

Use local loop jobs for recurring prompts that should run through the active scheduler:

```bash
/loop @every 5m check build
/loop once 30 14 * * 1 monday review
/loop list
/loop delete <id>
/schedule
```

Loop jobs are stored in `~/.koder/scheduled_tasks.json`. `/schedule` is the read-only registry view; `/loop` creates, lists, and deletes jobs.

Storage mutations are locked across threads/processes and publish complete JSON
snapshots atomically. Concurrent create/delete operations cannot lose unrelated
jobs or bypass the job limit. Invalid storage is reported, not silently replaced
with an empty registry.

A one-shot job is removed only after its prompt was accepted and the scheduled
turn completed without an error or cancellation. A rendered error message is not
a successful acknowledgment. Cooperating Koder processes using the same local
store coordinate delivery through per-job OS locks and persisted occurrence
receipts. A claimed job remains locked through queueing and execution; an
acknowledged occurrence is not dispatched again by another cooperating process.

Process exit releases the lock but leaves an unacknowledged claim recoverable.
This is at-least-once recovery, not exactly-once external effects: a crash after
a side effect but before acknowledgment can cause a retry, so such prompts
should be idempotent. Polling runs only while an interactive Koder process is
running; this is not an operating-system scheduler. Minutes never observed by
a running scheduler are not backfilled.

A stored job deleted while still waiting in the prompt queue is skipped when
dequeued. Deleting a job does not interrupt an already-dispatched prompt.
Shutdown discards queued in-memory deliveries while preserving unacknowledged
durable claims; after restart, those claims remain eligible for recovery.

Task records under `~/.koder/tasks/` use positive numeric IDs, not filesystem
paths. Reads and mutations reject path traversal, symlink targets, and records
whose embedded ID disagrees with their filename. Task snapshots and the ID
high-water mark are written atomically under a cross-platform lock. Related
dependency edits publish a shared redo journal before updating individual task
files. Public storage operations hold the same lock and replay committed intent
before returning data or making another change.

An I/O failure after journal publication does not imply rollback: a later
operation can finish the committed edit. Individual task files remain separate
snapshots, not one database transaction or an isolation guarantee for external
programs that bypass the storage API.

## Compaction And Rewind

Long sessions can be compacted to keep the useful parts while reducing context size:

```bash
/compact
```

Automatic compaction and `/compact` preserve system/developer instructions,
recent conversation messages in their original form, and complete trailing
tool-call/result pairs. Unfinished tool calls are included in the summary source
instead of being replayed without results. Older image payloads become `[image]`
placeholders in that source, while their accompanying text is retained.

An empty, analysis-only, or malformed summary response leaves the original
history unchanged. Plain untagged summaries are still supported; tagged output
must contain one unambiguous final summary. This checks the response structure,
not the semantic completeness of the model's summary.

Already-compacted context within its retention budget does not request another
summary just because system/developer instructions precede it. The no-op path
keeps image blocks and typed-message fields intact.

After a successful history replacement, the Session invalidates its own
file-read state before callers update usage statistics. A later read can therefore
return file contents removed by compaction instead of claiming they remain in
context. No-op and pre-replacement summary failures keep valid read state.
The same ownership boundary covers rewind/exact replacement. Clear/pop and
micro-truncated writes conservatively invalidate their Session's evidence,
including when a worker may finish after cancellation. Closing a Session
releases its cached content without clearing another Session's reads.

Use rewind when a recent turn sent the session down the wrong path:

```bash
/rewind
```

`/rewind` lists recent prompt targets with the number of newer transcript items each restore would remove, restores the selected prompt into input, and trims later session history.

## Exporting

Use local export commands when you want a durable artifact from a session:

```bash
/export
```

## Memory Commands

Memory files are local markdown files. User memory is shared across projects; project memory stays in the workspace.

```bash
/memory
/remember prefers focused tests before full suites
/thinkback
/thinkback-play
```

`/thinkback` summarizes recent local session context and prompt counts without running a model request. `/thinkback-play` replays recent turns from the active session.

## AutoDream

AutoDream is a best-effort cleanup-time memory consolidation task. When its local cadence threshold is met, it asks the configured provider to separate durable factual memories from reusable procedural skill candidates. The default `harness.auto_dream_write_mode: review` stages them under `~/.koder/memory-candidates/` and `~/.koder/skill-candidates/`; retrieval and skill discovery do not read those queues. Every candidate binds its storage scope, canonical origin workspace, and origin session into the reviewed record and content-derived ID.

Review candidates with `/memory candidates`, `/memory show <id>`, `/memory approve <id>`, or `/memory reject <id>`. `user` memories are user-scoped and approve to `~/.koder/memory/`; `feedback`, `project`, and `reference` memories are project-scoped and approve only to the candidate's recorded origin workspace. Factual approval and disabled skill-draft approval use the same private, exclusive, no-follow writer with full candidate IDs; skill drafts remain outside normal discovery under `~/.koder/skill-drafts/`. Set the mode to the exact value `automatic` only to opt into direct factual-memory writes. Automatic mode applies the same scope routing and writes separate files for each actual memory type/scope group. Set the mode to `off` to disable AutoDream extraction. Legacy `auto_dream_enabled: true` migrates to review rather than automatic writes. `/remember` remains an immediate user-authorized project memory write.

AutoDream records task metadata under `~/.koder/tasks/auto-dream/`.

Inspect recent runtime tasks with:

```bash
/tasks
```

## Settings Bundles

Use settings bundles to move configuration and local notes between machines:

```bash
koder config export ~/koder-settings.json
koder config import ~/koder-settings.json --dry-run
koder config import ~/koder-settings.json
```

Bundles include known Koder config, settings, keybindings, user memories, project memories, and project session notes. They exclude token stores, model caches, transcripts, task records, and plugin caches.
