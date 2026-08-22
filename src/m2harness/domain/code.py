"""Code proposal contract used by the local Code Harness adapter."""

from __future__ import annotations

from pydantic import Field

from m2harness.models import StrictModel


class CodeProposal(StrictModel):
    source: str = Field(min_length=1, max_length=2 * 1024 * 1024)
    logical_name: str = Field(default="generated_solution.py", pattern=r"^[A-Za-z0-9._-]+\.py$", max_length=200)
    # Local mathematical-model execution has no fixed wall-clock budget.
    # ``None`` means observe until natural completion; an explicit value is
    # reserved for an operator-requested emergency stop.
    timeout_seconds: int | None = Field(default=None, ge=1, le=86_400)
    expected_validations: tuple[str, ...] = ()
    # Provider-neutral runtime metadata.  The execution contract never trusts
    # this field for correctness, but it lets a mature agent runtime expose
    # its session/event artifacts to independent review.
    metadata: dict[str, str] = Field(default_factory=dict)
