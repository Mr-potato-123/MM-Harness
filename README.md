# M²Harness

M²Harness is a durable orchestration kernel for producing auditable, production-quality reports for individual mathematical-modeling questions. The Harness owns workflow state, recovery, review gates, provenance, and publication. Modeling/coding/review agents are replaceable activity executors; they do not control the workflow.

The first delivery boundary deliberately ends at `Final Question Report`. Paper-level synthesis is not part of this validation milestone.

## Implemented kernel

- Deterministic single-question state machine: modeling → coding/execution → independent review → controlled revision → finalization.
- SQLite transactions in WAL mode, process-safe leases, bounded attempts, and deterministic activity idempotency keys.
- Append-only, SHA-256 hash-chained audit events. Any changed, removed, or reordered event is detected.
- Immutable, content-addressed Artifact Store with read-time size and hash verification.
- Strict Pydantic request/response protocol. Invalid or wrong-stage executor output cannot advance state.
- Agent process adapter using JSON over stdin/stdout, a fixed argument vector, `shell=False`, timeouts, response-size limits, and an explicit environment allowlist.
- Lease heartbeat for long-running activities and recovery when an activity committed before its state transition.

## Install and initialize

```powershell
python -m pip install -e .
m2harness init
m2harness create-project "validation-run"
```

Add a question using the returned project UUID:

```powershell
m2harness add-question <project-uuid> q1 "Question 1" .\problem.md
```

Run it with a real agent adapter. Everything after `--executor` is the adapter's fixed command vector:

```powershell
m2harness run-question <question-uuid> --worker-id worker-01 --executor python .\agent_adapter.py
```

Inspect and verify:

```powershell
m2harness status <question-uuid>
m2harness verify --verify-artifacts
```

Global options such as `--database`, `--artifacts`, retry count, lease duration, and revision limit precede the subcommand.

## Executor contract

The Harness writes exactly one UTF-8 JSON `ActivityRequest` to stdin. The adapter writes exactly one UTF-8 JSON `ActivityResponse` to stdout and uses stderr for diagnostics. It must echo `idempotency_key`, return the requested stage, and treat repeat deliveries with the same key idempotently.

The input includes immutable Artifact references, not embedded artifact bytes. A deployed adapter should resolve them through an authenticated Artifact service or a read-only mounted store. The local process adapter is an integration boundary, not a security sandbox. Generated code must run through a container/VM sandbox with network, resource, filesystem, and secret policies before this system is used with untrusted code.

The authoritative wire models are in [`src/m2harness/models.py`](src/m2harness/models.py).

## Validation

The repository uses standard-library tests so validation does not depend on a test framework being installed:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Tests cover the successful report path, review-directed revision, idempotent retry, recovery after an activity commit/state-transition crash, retry-budget exhaustion, concurrent lease exclusion, artifact tamper detection, and event-chain verification.

## Production boundary

This is a production-oriented kernel, not a complete production deployment. Before multi-host service rollout, the SQLite implementation should be placed behind a single state service or replaced by PostgreSQL while retaining the store contract; artifact blobs should move to versioned object storage; the command adapter should be backed by a sandbox/queue worker; and authentication, authorization, secrets, metrics, tracing export, backups, retention, and operator-driven requeue/cancel APIs must be integrated with the target platform.

Those are deployment capabilities around the implemented workflow semantics. They do not require handing orchestration control to an agent framework.
