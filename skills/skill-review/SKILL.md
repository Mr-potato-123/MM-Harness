---
name: skill-review
description: Review a local Skill package for discoverability, safety, resource integrity, and behavioral readiness before it is used in production prompts.
---

# Skill Review

Treat Skills as versioned executable policy, not loose prompt snippets. Check
kebab-case naming, concise routing description, valid frontmatter/manifest,
bounded context, safe resource paths, and explicit invocation/capability needs.

Review the body for concrete decision guidance, scope boundaries, artifact
contracts, and prompt-injection resistance. Read only referenced resources when
the requested mode needs them. Record a readiness status (ready,
needs_revision, or blocked) and actionable defects.

Use [references/review-rubric.md](references/review-rubric.md) and the fixture
examples in [evals/cases.json](evals/cases.json) when performing a package
review.

