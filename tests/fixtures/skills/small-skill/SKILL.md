---
name: small-skill
description: Small skill with basic content for token testing
license: MIT
---

# Small Skill: Code Formatting Guidelines

This skill describes a compact but realistic set of code formatting
guidelines. It is used in tests that measure token counts for skill
metadata versus full content. The text is intentionally longer than the
template skill so that the savings from progressive disclosure are
obvious when only the lightweight metadata is injected into the system
prompt.

## Goals of Consistent Formatting

- Make code bases readable by any team member.
- Reduce cognitive load when switching between files.
- Minimize unnecessary diffs during code review.
- Allow automated tools such as `black` and `ruff` to do most of the work.

When formatting is predictable, reviewers can focus on the substance of
a change instead of spending time commenting on indentation, line
length, or import ordering. A good rule of thumb is that humans should
spend their attention on naming, structure, and correctness, while
automated tools enforce the small mechanical rules.

## Editor Integration

Most developers experience formatting rules through their editor or IDE.
To keep friction low:

1. Enable "format on save" for the primary language server.
2. Configure the same line length in both the formatter and linter.
3. Make sure the editor uses UTF‑8 and Unix line endings by default.
4. Share editor configuration files (such as `.editorconfig`) in the repository.

The important part is that everyone on the team runs the same tools with
the same configuration. If a project uses `black`, developers should not
disable it locally or override its defaults in surprising ways.

## Handling Legacy Code

In real projects there is almost always some legacy code that does not
match the current formatting rules. A practical approach is:

- Avoid reformatting the entire repository in a single massive commit.
- Instead, reformat files as they are touched for real changes.
- For large refactors, coordinate with the team so that merges remain manageable.

This incremental strategy allows teams to converge toward a consistent
style without breaking every open feature branch. The tests in this
repository treat this document as a "small" skill that still contains
enough natural language to produce a meaningful token count.

