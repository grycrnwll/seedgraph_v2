"""Golden-set errors."""

from __future__ import annotations


class GoldenError(Exception):
    """A golden-set operation failed loudly (bad seed, ambiguous hash, unsafe path...)."""
