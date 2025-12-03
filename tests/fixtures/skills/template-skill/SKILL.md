---
name: template-skill
description: A minimal template skill for testing
---

This is a minimal template skill for testing purposes.
It is intentionally small so that tests can easily reason
about token counts while still working with realistic content.

Use this skill when you want to verify that:

- YAML frontmatter is parsed correctly.
- The body content is separated from metadata.
- Progressive disclosure only loads the content when requested.

The text does not reference any other skills and does not rely
on external files. When the loader reads this file it should
see a short but structured document that looks like a typical
skill definition used by the Koder agent.

