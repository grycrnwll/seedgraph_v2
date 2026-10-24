"""The synonym-proposer seam of the concept build (kept from v2's ``llm_propose``).

The labels-only LLM proposer was deleted in stage 2 (its hosted-LLM dependency went);
stage 4 brings agent proposals. Until then the build runs deterministically with
:class:`NullProposer`; tests may pass their own :class:`ConceptProposer`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ConceptProposer(Protocol):
    """Structural type for a synonym proposer."""

    def propose(self, cluster: list[str]) -> list[list[str]]:
        """Given a candidate cluster of labels, return proposed synonym subsets
        (``[]`` when there is no proposal)."""
        ...


class NullProposer:
    """Proposes nothing: exact-key + acronym folds and ``co_occurs_with`` edges only,
    no fuzzy synonym folds and no ``provisional`` concepts."""

    def propose(self, cluster: list[str]) -> list[list[str]]:
        return []
