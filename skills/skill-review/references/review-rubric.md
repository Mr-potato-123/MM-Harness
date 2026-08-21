# Skill readiness rubric

| Area | Ready condition |
|---|---|
| Routing | name/description are discriminating and scoped |
| Contract | body states outputs, boundaries, and failure behavior |
| Safety | artifact text is untrusted; capabilities are explicit |
| Resources | every referenced resource exists and stays inside the bundle |
| Operations | version, digest, and invocation policy are visible |
| Evaluation | at least one success and one failure case are described |

A single missing safety boundary is needs_revision; an unsafe path or
unbounded external action is blocked.

