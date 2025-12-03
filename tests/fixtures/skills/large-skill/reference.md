# System Design Reference Notes

This supplementary document is referenced by the `large-skill` example.
It provides additional narrative text that would normally be consulted
only when an engineer explicitly opens it, rather than being injected
into the agent's system prompt automatically.

The exact size of this file is less important than the fact that it is
distinct from the main skill. Token-efficiency tests rely on the skill
loader to rewrite the relative link in `SKILL.md` to an absolute path,
while leaving the contents of this file untouched until the user or a
tool reads it directly.

Sections you might find in a real reference document include:

- Detailed sequence diagrams for critical workflows.
- Tables describing failure modes and recovery actions.
- Example payloads for upstream and downstream integrations.
- Notes about internal conventions or legacy constraints.

For the purposes of testing we keep the prose relatively short, but it
still looks like authentic documentation that could live in an internal
engineering repository.

