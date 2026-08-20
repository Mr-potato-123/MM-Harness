"""Stable error taxonomy for the harness API and worker runtime."""


class HarnessError(Exception):
    """Base class for expected harness failures."""


class ConfigurationError(HarnessError):
    """Configuration is incomplete or unsafe."""


class NotFoundError(HarnessError):
    """A requested domain object does not exist."""


class ConflictError(HarnessError):
    """A concurrent owner or incompatible state prevents the operation."""


class InvalidTransitionError(HarnessError):
    """A workflow transition violates the state machine."""


class LeaseLostError(ConflictError):
    """A worker no longer owns the workflow lease."""


class ArtifactIntegrityError(HarnessError):
    """Stored artifact bytes no longer match their immutable identity."""


class ActivityExecutionError(HarnessError):
    """An external activity failed or returned an invalid result."""


class RevisionLimitExceeded(HarnessError):
    """The configured modeling/code revision budget was exhausted."""

