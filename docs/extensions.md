# Skills, Plugins, And MCP

Koder can be extended with skills, plugins, MCP servers, session channels, and Magic Docs. Use the lightest extension that solves the problem.

## Skills

Skills are local instruction bundles loaded with progressive disclosure. At startup Koder loads skill names and descriptions; full content is loaded only when the skill is invoked or selected by the agent.

Skill search order:

1. `.koder/skills/` in the current project
2. `~/.koder/skills/`
3. Installed plugins
4. Bundled runtime skills

Create a project skill:

```text
.koder/skills/api-review/SKILL.md
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

Inspect available skills:

```bash
/skills
```

Invoke a manual skill by its command name when the skill exposes one, or ask Koder to use it in plain language.

Subsequent `get_skill` calls refresh the merged cache when discovered skill files
are added, removed, renamed or edited. The cache tracks each file's path,
identity, size and timestamps, so a newer unrelated skill cannot hide those
changes. Nested reference documents are not skill definitions and do not trigger
this cache refresh. This is a filesystem-change check, not a content hash or an
atomic snapshot of concurrent edits.

Refreshing discovery does not recall instructions or restrictions already
activated in an ongoing invocation. Later invocations load the current policy.
The `paths` frontmatter field is retained as metadata; the runtime does not
currently use it to automatically load a skill or activate its hooks or tool
restrictions. Explicit/model-selected skill loading remains the activation path.

## Verifier Skills

Use `/init-verifiers` to create project-local verifier skills:

```bash
/init-verifiers cli
/init-verifiers web
/init-verifiers api
```

Verifier skills are useful when a project has a repeatable acceptance workflow that should be explained in one place and loaded on demand.

## Plugins

Plugins can contribute skills, commands, MCP servers, channels, and dependencies.

```bash
koder plugin install ./my-plugin --scope project
koder plugin list
koder plugin enable my-plugin
koder plugin disable my-plugin
koder plugin validate ./my-plugin
koder plugin marketplace list
```

Installing `name@marketplace` records the selected catalog and its source
fingerprint alongside the installed plugin. That record survives enable/disable,
upgrades and rollback; a direct local replacement does not inherit it. Repository
caches are source-specific and are only updated when their configured Git origin
matches the requested repository.

Inside the TUI:

```bash
/plugin
/reload-plugins
/skills
```

Use `--plugin-dir` for a session-only plugin directory while developing a plugin locally.

The session overlay is bound to its scheduler: model-time skills, agent
definitions, hooks and MCP discovery use the same directory as command discovery.
Concurrent runtimes keep distinct roots, and a local override does not inherit
the installed copy's marketplace authorization.

## MCP Servers

MCP servers add external tools to the Koder runtime.

```bash
koder mcp add filesystem "python -m mcp.server.filesystem" --scope project
koder mcp add api --transport http --url http://localhost:8000 --header "Authorization: Bearer token"
koder mcp list
koder mcp get filesystem
koder mcp remove filesystem --scope project
koder mcp approve
koder mcp reset-project-choices
koder mcp serve
```

MCP configuration can live in user, project, or local scopes depending on the command flags. Use project scope when the server is part of the repository workflow; use user scope for personal tools.

Repository-controlled MCP definitions, including inline `mcpServers` in project agent frontmatter, are fail-closed until reviewed. Run `koder mcp approve` to inspect each source path, fixed execution directory, server target, and current expanded-configuration digest, then approve it interactively. Use `--source PATH` to limit the review or `--yes` only after reviewing the displayed source. Changes to the source, workspace root, execution directory, or environment-expanded values require a new approval. Approval records contain only paths, the decision, and digests—not expanded secrets.

See [Configuration Guide](configuration.md) for the YAML config format.

### MCP form requests

Connected servers can request a non-sensitive form. Koder supports text,
integer, number, boolean, enum and string-array answers, and checks the submitted
content against the requested JSON Schema before returning it. Arrays stay
arrays; invalid enum input does not silently choose the first option.
Invalid/unsupported responses are cancelled, including invalid hook-provided
answers. Schema references cannot fetch remote documents.

Form readers and hook dispatch are asynchronous. Concurrent form requests on
the event loop share the terminal one at a time. EOF and Ctrl+C dismiss a form;
task cancellation stops and joins its reader before propagating. Without an
interactive terminal, an unanswered request is cancelled without consuming
stdin. Configured `Elicitation` hooks can still provide schema-valid answers.
Decline/cancel responses contain no submitted data. Unsupported URL-mode
requests return `decline` without opening a browser.

## Channels

Channels are MCP or plugin-backed session integrations enabled at startup:

```bash
koder --channels server:my-channel
koder --channels plugin:team-chat@local
koder --dangerously-load-development-channels server:dev-channel
```

Inspect active channel entries:

```bash
/channels
/channels help
```

`/channels` is read-only. It reports active entries, the supported startup forms,
and the current inbox's delivery counts, capacity, rejection count and path.

Channel notifications are admitted only after the initialized server has been
published, advertises the required capability, and remains enabled for the
session. Events from retired connections or closed owners are discarded;
ordinary non-channel MCP operations remain available.

### Bounded channel delivery

Admitted notifications are staged as private JSON files under
`~/.koder/channel-inbox/run-*/`. Each running Koder instance owns a separate
directory; it does not load another run's retained messages. Waiting payloads
stay on disk, while memory holds only bounded entry descriptors. File I/O runs
off the event loop and is joined before cancellation returns.

The defaults are:

| Setting | Environment variable | Default |
| --- | --- | --- |
| Retained message count | `KODER_CHANNEL_MAX_PENDING_MESSAGES` | 4,096 |
| Retained serialized JSON bytes | `KODER_CHANNEL_MAX_PENDING_BYTES` | 64 MiB |
| Single serialized JSON record | `KODER_CHANNEL_MAX_MESSAGE_BYTES` | 1 MiB |

Limits must be positive integers; the single-record limit must not exceed the
total byte limit. Reservations waiting for disk I/O and pending, running, failed,
cancelled and interrupted records
all occupy capacity until successfully handled. Filesystem allocation overhead,
temporary staging and the final metadata manifest are additional to the JSON
byte count. These limits bound the retained inbox, not the underlying MCP
transport's decoding of an incoming frame.

Capacity is reserved before waiting for the disk lock, so concurrent connections
cannot accumulate unbounded staging buffers. Reserved admissions remain owned
through receiver cancellation and are joined before inbox closure.

Admission waits for local file publication, **not for the model to consume a
message or free a queue slot**. This matters because channel notifications share
the MCP receive path with ordinary responses. A full inbox rejects new work
explicitly: warnings are rate-limited, but the rejection count and reasons remain
available in status and the retained manifest. Storage failures stop further
admission and are reported; they do not trigger an unbounded admission-task queue.
Local disk latency still contributes to receive latency.

The delivery states are visible in filenames and the retained manifest:

- `pending`: staged but not handed to a turn.
- `running`: handed off; a crash can leave its final outcome uncertain.
- `failed` / `cancelled`: a turn did not complete successfully, including failures
  rendered by the scheduler as text rather than raised as exceptions.
- `interrupted`: runtime shutdown interrupted an in-flight delivery.

Successfully handled records are removed. On orderly shutdown the runtime first
unregisters admission, then cancels and joins its consumer and file operations.
Unfinished records are retained with `manifest.json`, and the directory is
reported. An entirely successful inbox is removed. Records contain the full
incoming content in plaintext; new directories/files use private POSIX modes,
not encryption or an operating-system sandbox.

Retained messages are **not automatically replayed**. Review them explicitly
before choosing to retry, since an interrupted turn may already have performed
external actions. A staging log entry is not an acknowledgement that model work
finished. The inbox preserves ordinary runtime/shutdown outcomes; it is not an
exactly-once, crash-recovery or power-loss transaction across model actions.
Retained archives across separate runs are not a global disk quota.

Messages still target the conversation active when the consumer dispatches them,
rather than pinning the conversation that happened to be active on receipt.

The `plugin:team-chat@local` form requires an installer-owned receipt from the
registered `local` marketplace. Its stored source fingerprint must still match
the current registration. Legacy/local installations, malformed receipts and
removed or rebound marketplace names do not silently gain this authorization.
Reinstall from the registered marketplace to establish provenance, or use
`server:<actual MCP name>` to explicitly enable that installed server.

A plugin manifest or a `plugin:` prefix in a server name cannot establish this
receipt. It identifies the configured installation source, not a publisher code
signature or an operating-system security boundary.

### Capability boundary

Channel delivery does not enable remote tool approval. The permission-relay
ID/callback helpers are standalone and have no production approval consumer;
they do not authenticate a sender. Their registrations reject duplicate pending
IDs and keep old unsubscribe handles from retiring new requests, but those
properties alone are not an authorization protocol. Model tool calls still
use the normal permission service and configured approval path.

## Magic Docs

Magic Docs are markdown files whose first line is:

```markdown
# MAGIC DOC: Project Runtime Notes
```

An optional italic line directly after the header becomes refresh guidance. When Koder reads a Magic Doc through `read_file`, it tracks the file and refreshes a managed `## Koder Session Notes` section after completed turns.

Commands:

```bash
/magic-docs
/magic-docs refresh
```

Use Magic Docs for local project notes that should stay current across a Koder session. The refresh is local and deterministic; it preserves the header and guidance line and replaces only the managed section.
