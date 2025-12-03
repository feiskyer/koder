---
name: large-skill
description: Large skill with extensive content and supplementary files
---

# Large Skill: System Design Field Guide

This skill contains a longer narrative that resembles the kind of
internal field guide teams use when designing new systems. It is not
intended to be exhaustive, but it is intentionally verbose so that
token-counting tests have a substantial body of text to work with.

## 1. Clarify the Problem

Every system design effort should begin with a careful restatement of
the problem. Engineers should write down:

- Who the primary users are and what they are trying to achieve.
- The most important success metrics for the system.
- Explicit non-goals that are deliberately out of scope.
- Constraints related to latency, throughput, security, or compliance.

Capturing these items up front makes it easier to evaluate trade‑offs
later. When the design diverges from the original problem statement,
the team can either adjust expectations or update the requirements.

## 2. Identify Boundaries and Responsibilities

System diagrams are useful, but they must be backed by well-defined
boundaries. Each component should have a crisp responsibility such as
"serves read‑only product data" or "handles authenticated user
sessions." When responsibilities are vague, coupling tends to increase
and it becomes difficult to reason about failures.

Consider documenting:

- Which components own which data sets.
- Which APIs are stable contracts versus internal details.
- How requests flow through the system from entry point to storage.
- Where cross-cutting concerns such as logging and metrics live.

These notes rarely appear in user-facing documentation, but they are
precisely the kind of content that internal skills can capture.

## 3. Data Modeling and Storage

Choosing the right data model is often more important than picking a
particular database product. For each major entity, describe:

- The fields and their types.
- How entities relate to one another.
- Which access patterns must be fast and which can be slower.
- How historical data should be archived or deleted.

During testing, this section serves simply as additional text that
increases the size of the large skill. The token counter does not need
to understand the semantics; it only needs to see that this content is
substantially larger than the metadata header used at progressive
disclosure level one.

## 4. Performance and Scalability

Performance planning starts with realistic assumptions about traffic.
Design documents should include back‑of‑the‑envelope calculations for:

- Requests per second in steady state.
- Expected spikes during peak events or batch jobs.
- Storage growth over one, two, and three years.
- The cost of serving an average request.

Engineers can use these estimates to decide whether to introduce queues,
caching layers, background workers, or read replicas. When documenting
these decisions in a skill, it is helpful to explain the reasoning in
plain language so that future maintainers can understand why certain
trade‑offs were made.

## 5. Resilience and Operations

No system is perfect, so designs must include a plan for failure. This
section should answer questions such as:

- What happens when a dependency is slow or unavailable?
- How are partial failures reported to users?
- Which alerts are critical and which are informational?
- How do we perform controlled rollbacks when a deployment goes wrong?

Operational guidance may also cover runbooks, on‑call expectations, and
playbooks for common incidents. Again, from the perspective of token
efficiency tests, these paragraphs simply provide more natural language
for the encoder to process.

## 6. Supplementary Reference Material

Longer design documents often refer to additional material such as
sequence diagrams, API schemas, or architectural decision records. In
this skill we model that pattern using a separate reference file:

See the extended notes in the [reference](reference.md) document for
more detailed examples, diagrams, and checklists.

The loader should treat this as a normal markdown link. Tests that
exercise progressive disclosure at level three can assert that the
content of the reference file is not automatically inlined when the
skill is loaded; instead, only the path to the document is expanded to
an absolute filesystem path.

