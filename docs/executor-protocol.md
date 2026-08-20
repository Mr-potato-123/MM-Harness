# Activity executor protocol v1

An activity is the only side-effecting unit delivered to an external agent. Delivery is at least once; the stable `idempotency_key` gives the adapter the information required to make the effect exactly once from its own perspective.

## Required behavior

1. Read one JSON request from stdin and validate `protocol_version == 1`.
2. Resolve only the listed immutable inputs and verify each referenced SHA-256 digest.
3. Execute only the requested stage. Persist any provider/run identifier against `idempotency_key` before returning.
4. Emit one matching response to stdout. Do not mix logs into stdout.
5. On a repeated key, return the recorded response or resume the same run; do not start an unrelated run.

Stage outputs are discriminated by `output.stage`:

- `modeling`: report, validation contract, expected outputs.
- `coding`: execution report, validation outcomes, metrics, and produced evidence.
- `review`: independent verdict, accepted claims, or concrete revision instructions.
- `finalize`: final single-question report and structured downstream outputs.

Every stage publishes its report as an immutable Artifact. Extra textual or binary artifacts use `text` or validated Base64 content respectively. Review is a Harness gate: only `approve` can lead to finalization, and revision count is bounded by policy.

## Failure semantics

- Non-zero process exit, timeout, malformed JSON, schema mismatch, incorrect key, or incorrect stage fails the attempt.
- A failed attempt does not advance question state.
- An expired activity lease permits redelivery with the same idempotency key and a higher attempt number.
- Once the activity result and Artifact metadata commit, recovery reuses them and performs only the missing state transition.
