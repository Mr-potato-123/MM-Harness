# M²Harness

M²Harness is a durable Agent Harness for producing auditable, production-quality reports for individual mathematical-modeling questions. The Main Harness Agent owns process state, DAG/TODO planning, recovery, review gates, provenance, and publication. Its primary problem-solving primitive is the `solve_problem` Tool; Model Agent exploration, optional multi-route synthesis, Code Harness execution, and Model review communicate inside that Tool through reports.

The first delivery boundary is a reviewed `Final Question Report` plus one
structurally validated LaTeX publication artifact. TeX compilation and
claim-to-paper parity are explicit follow-up capabilities. This is still a
single-question report workflow; cross-question paper synthesis and paper
review remain outside the milestone.

Deployment scope is intentionally local: control plane, storage, skills, tools, media processing, context, sandbox, execution, review, and reports run locally. Only allowlisted model/API Provider calls and the dedicated `web_search` capability may leave the machine. See [M2Harness_Scope_v0.1.md](M2Harness_Scope_v0.1.md).

The repository is currently in a controlled rebuild. The original durable stage workflow remains under `m2harness.store/workflow` as a compatibility path; the target semantics live under `domain`, `ports`, `application`, `infrastructure`, and `runtime`. The architecture/tool-flow audit is tracked in [M2Harness_Current_Architecture_and_Toolflow_Report_v0.2.md](M2Harness_Current_Architecture_and_Toolflow_Report_v0.2.md), and the reference-derived Research/HMML design is in [M2Harness_Reference_Adoption_and_Research_Design_v0.1.md](M2Harness_Reference_Adoption_and_Research_Design_v0.1.md).

## Implemented kernel

- Main Harness DAG/TODO kernel that dispatches each executable problem node through one `solve_problem` Tool and unlocks dependent nodes only after a completed SolveProblemReport.
- Explicit `rollback()` and `replan()` operations invalidate downstream state, preserve historical reports/files, and append durable process decisions.
- MM-Agent-style dependency projection: each downstream solve receives its own subproblem text, a compact accepted-solution projection for direct predecessors, and a purpose-labelled read-only file manifest. DAG nodes can name required upstream keys through `dependency_outputs`.
- Progressive file disclosure: listing a workspace-relative path does not disclose its content. A Model Agent must return `requested_file_paths`; only allowlisted, digest-verified UTF-8 files are then added to `disclosed_text_files`. Binary problem inputs continue through the explicit multimodal channel.
- Provider-boundary `compact` middleware preserves system/recent turns and replaces older history with an auditable continuation snapshot containing objectives, decisions, evidence, open issues, file paths, and next actions.
- Bounded `solve_problem` loop: optional preliminary multi-route modeling → Model Agent synthesis → Code Harness realization → Model Agent approve/revise, with strict report contracts.
- Review selects the narrowest `revision_target`: code-only revisions reuse the accepted Modeling Report; model/full revisions repeat exploration and synthesis.
- Strict `SolveProblemTask`, `ModelingReport`, `CodingReport`, `Review`, and final problem report protocols; the default runtime fails closed when real Model Agent/Code Harness ports are not configured.
- Model Agent context can run a local-first DeepResearch pass before preliminary modeling. The read-only `knowledge_search` Tool returns source-linked HMML findings and gaps; it never grants Harness instructions.
- SQLite transactions in WAL mode, process-safe leases, bounded attempts, and deterministic activity idempotency keys.
- Append-only, SHA-256 hash-chained audit events. Any changed, removed, or reordered event is detected.
- Immutable, content-addressed Artifact Store with read-time size and hash verification.
- Strict Pydantic request/response protocol. Invalid or wrong-stage executor output cannot advance state.
- Agent process adapter using JSON over stdin/stdout, a fixed argument vector, `shell=False`, timeouts, response-size limits, and an explicit environment allowlist.
- Lease heartbeat for long-running activities and recovery when an activity committed before its state transition.
- Versioned domain Event Envelope with SQLite optimistic aggregate versions and transactional outbox.
- Descriptor-first Skill Registry with manifest/digest validation and immutable per-session Skill snapshots.
- Capability resolution and Tool Runtime middleware for authorization, input schemas, idempotency, result budgets, and redaction.
- Native Qwen multimodal projection tests for PDF `file_data` and image `image_url`; the provider adapter reads credentials only from the environment.
- Harness-owned DAG task table supports serial or planner-selected dependency graphs. `canonical_main_harness_dag()` creates `solve_problem -> publish-paper`; the older `ingest -> model -> code -> review -> publish-paper` graph remains only as a compatibility path.
- Final publication gate requiring a `final_latex_paper` `.tex` artifact,
  structural safety validation, and Markdown/LaTeX claim parity at the
  workflow boundary.
- The local runtime discovers MM-Agent's HMML export from `M2HARNESS_HMML_PATH`,
  `knowledge/HMML.json`, or the checked-in reference path; set the environment
  variable for a production-managed index.
- `MainHarness.generate_paper()` is the terminal global-context merge boundary;
  `QwenPaperComposer` can produce the reviewed report plus the single
  `final_latex_paper` artifact after all `solve_problem` nodes are complete.
- Every solve attempt is materialized under `workspace/reports/runs/<run-id>/tasks/<task-id>/attempt-<n>/` with preliminary/modeling/coding/final reports, generated source/log artifacts, digests, sizes, purposes, and a persisted report-file index in Main Harness state.

Context is intentionally layered for long-horizon work: current problem text and compact direct-dependency conclusions are loaded first; full upstream `solution_report.md`, coding outputs, and generated files remain path-only until requested. Final paper composition receives accepted final reports and artifact metadata rather than complete Model/Code iteration transcripts.

## Rebuilt runtime smoke check

The new runtime can be composed without starting a provider or a Sandbox:

```powershell
$env:PYTHONPATH = "src"
python -c "from m2harness import build_local_runtime; print([s.name for s in build_local_runtime().skills.list()])"
```

This verifies registry composition, capability-gated planning, restart-safe Workflow state, and the local tool catalog. Generated code still requires a native container/VM sandbox for an untrusted multi-tenant deployment.

Inspect the effective local Skill/Tool/Capability catalog before starting a worker:

```powershell
m2harness catalog
```

Run a deployment preflight (for this trusted single-machine scope the host backend is the default):

```powershell
m2harness preflight --input .\2026B.pdf
```

Add `--qwen` when the intended run is the Qwen provider path; this makes the runtime secret mandatory in the preflight result.

The catalog includes bounded local artifact/workspace/media/data/Python/validation/report tools plus the `dag_task_table` planning validator. Tool reservations and hash-chain audit records use the configured database. Python/validation execution uses the trusted local host backend by default, with fixed argv, scrubbed environment, workspace confinement, timeouts, and output budgets. Set `M2HARNESS_SANDBOX_BACKEND=docker` for isolation or `=none` to fail closed.

To opt into Docker isolation when Docker Desktop/Engine is available:

```powershell
# Preload the image once; the Harness never pulls images during a run.
docker pull python:3.12-slim
$env:M2HARNESS_SANDBOX_BACKEND = "docker"
$env:M2HARNESS_SANDBOX_IMAGE = "python:3.12-slim"
```

At startup the Docker backend records the resolved local image ID (`sha256:...`) in execution evidence; the configured tag is only the lookup reference. Keep the image preloaded and pin/approve that ID in deployment configuration.

## Install and initialize

```powershell
python -m pip install -e .
# Optional local PDF/image/analytics parsers:
python -m pip install -e ".[media,analytics]"
m2harness init
m2harness create-project "validation-run"
```

The target Main Harness path can be run after injecting the provider secret
(`DASHSCOPE_API_KEY`) without putting it in source:

```powershell
m2harness run-main-qwen .\2026B.pdf --problem "Solve the attached problem" --max-iterations 3
```

This executes `Main Harness → solve_problem → Model/Code/Review → global paper
composer`; the local sandbox and HMML index remain on the machine.

The wheel carries the complete 19-skill bundled library (including referenced
checklists/evals) under `share/m2harness/skills`; an installed CLI does not
depend on the source checkout's `skills/` directory. The repository path is
`D:\Project\MM_Harness\skills`; the older `D:\Project\MM\_Harness\skills`
path is not present in this checkout.

Add a question using the returned project UUID:

```powershell
m2harness add-question <project-uuid> q1 "Question 1" .\problem.md
```

Run it with a real agent adapter. Everything after `--executor` is the adapter's fixed command vector:

```powershell
m2harness run-question <question-uuid> --worker-id worker-01 --executor python .\agent_adapter.py
```

For the bundled Qwen3.8-Max multimodal adapter (the API key must be supplied at runtime through `DASHSCOPE_API_KEY`):

```powershell
$env:DASHSCOPE_API_KEY = "<key from your secret manager>"
$env:QWEN_MAX_OUTPUT_TOKENS = "12000"  # optional hard output budget
m2harness run-qwen <question-uuid> --worker-id worker-01
```

`run-qwen` runs the proposed source code through the local host backend by default. The Harness materializes verified inputs under the activity workspace (`inputs/`), `SandboxedActivityExecutor` runs it locally, and stdout/stderr/result files are persisted as evidence. `--allow-remote-code-interpreter` explicitly enables the provider's remote code capability only for preliminary provider/projection testing; its result is still rejected by the production evidence gate.

Inspect and verify:

```powershell
m2harness status <question-uuid>
m2harness verify --verify-artifacts
m2harness export-audit <question-uuid> .\audit-bundle.json
```

`export-audit` writes a versioned, atomically published bundle containing the
question, activity protocol records, immutable artifact metadata, claim/evidence
links, and the verified event count. Existing files are never overwritten unless
`--overwrite` is supplied.

When a run completes, filter the artifact list for `kind=final_latex_paper` to
obtain the `.tex` source. The `publish-paper` DAG task is the only completion
boundary: the Harness validates document boundaries and rejects
shell/file-write primitives; it does not claim that TeX compiled unless a
separately authorized compiler activity records that evidence.

Global options such as `--database`, `--artifacts`, retry count, lease duration, and revision limit precede the subcommand.

## Executor contract

The Harness writes exactly one UTF-8 JSON `ActivityRequest` to stdin. The adapter writes exactly one UTF-8 JSON `ActivityResponse` to stdout and uses stderr for diagnostics. It must echo `idempotency_key`, return the requested stage, and treat repeat deliveries with the same key idempotently.

The input includes immutable Artifact references, not embedded artifact bytes. A deployed adapter should resolve them through an authenticated Artifact service or a read-only mounted store. The local host backend is suitable only for this trusted single-user scope; generated code must run through a container/VM sandbox with network, resource, filesystem, and secret policies before this system is exposed to untrusted or multi-tenant code.

The authoritative wire models are in [`src/m2harness/models.py`](src/m2harness/models.py).

## Validation

The repository uses standard-library tests so validation does not depend on a test framework being installed:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Tests cover the successful report and LaTeX publication path, DAG terminal and
cycle/reachability validation, review-directed revision, idempotent retry,
recovery after an activity commit/state-transition crash, retry-budget
exhaustion, concurrent lease exclusion, artifact tamper detection, multimodal
projection, skill resource integrity, and event-chain verification.

## Production boundary

This is a production-oriented kernel, not a complete production deployment. Before multi-host service rollout, the SQLite implementation should be placed behind a single state service or replaced by PostgreSQL while retaining the store contract; artifact blobs should move to versioned object storage; the command adapter should be backed by a sandbox/queue worker; and authentication, authorization, secrets, metrics, tracing export, backups, retention, and operator-driven requeue/cancel APIs must be integrated with the target platform.

Those are deployment capabilities around the implemented workflow semantics. They do not require handing orchestration control to an agent framework.
