---
name: medium-skill
description: Medium-sized skill for comprehensive token testing
allowed_tools:
  - read_file
  - write_file
---

# Medium Skill: Practical API Integration Playbook

This skill represents a medium-sized document that an engineering team
might maintain to describe best practices for integrating with external
APIs. The content is intentionally richer and longer than the previous
skills, providing enough material for tests that compare token usage
between metadata-only prompts and fully expanded skill prompts.

## Establishing Clear Contracts

Before writing the first line of integration code, clarify the API
contract:

- Define which operations are required for the product's first release.
- Identify rate limits, quota policies, and billing implications.
- Capture authentication requirements in a short checklist.
- Decide which fields are mandatory and which are optional.

These details should live in version-controlled documentation so that
they can evolve alongside the code. Engineers can reference this skill
while writing tests or reviewing changes to the integration layer.

## Authentication and Secrets

Most APIs require some form of credential such as an API key, OAuth
token, or signed JWT. Good practices include:

1. Store secrets in a secure vault rather than in source control.
2. Provide a lightweight helper that loads credentials from a single place.
3. Rotate credentials regularly and document the rotation procedure.
4. Log authentication failures with enough context to debug, but never log secrets.

When tests in this repository count tokens, they treat this section as
part of the "full content" that is only loaded when the agent explicitly
requests the medium‑sized skill.

## Error Handling and Retries

API integrations fail in many subtle ways: network timeouts, malformed
responses, authentication drift, and upstream outages. The integration
layer should:

- Distinguish between transient and permanent failures.
- Use exponential backoff with jitter for retries.
- Prefer idempotent operations when possible.
- Surface clear error messages to both logs and users.

The goal is to avoid tight retry loops that amplify outages while still
providing a smooth experience when brief network issues occur. Unit
tests often simulate these failures by stubbing HTTP clients and
asserting that the integration layer behaves as described here.

## Pagination and Partial Results

Many APIs paginate large collections. Instead of loading every record at
once, integrations should stream or page through results:

- Respect server-provided cursors or continuation tokens.
- Choose sensible default page sizes for the client.
- Allow callers to stop early when they have enough data.
- Validate that pagination state is not lost across retries.

By structuring the integration code around clear data flows and
reusable helpers, teams can keep this logic understandable even when
working with complex external APIs.

