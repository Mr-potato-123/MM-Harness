---
name: sandbox-execution
description: Execute model code inside the configured local sandbox and turn stdout, logs, outputs, and exit status into admissible evidence.
---

# Sandbox Execution

Only the Harness sandbox may convert a model's execution claim into evidence.
Run with the declared input manifest, network denied by default, bounded time
and output, and a fresh workspace for each attempt.

Record the source digest, sandbox image/backend, command, exit code, timeout,
stdout/stderr digests, produced files, and parsed validation JSON. A remote code
interpreter or provider-side claim is preliminary and must never close review.

