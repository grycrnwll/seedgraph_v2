"""Exception hierarchy for seedgraph."""


class SeedgraphError(Exception):
    """Base class for all seedgraph errors."""


class ConfigError(SeedgraphError):
    """Raised when configuration is missing, malformed, or self-inconsistent."""


class ValidationError(SeedgraphError):
    """Raised when an input fails validation (e.g. a malformed project slug)."""
