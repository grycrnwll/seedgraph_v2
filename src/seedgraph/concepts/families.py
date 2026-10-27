"""Type families (stage 4 design §1.1). Pure constants.

The type fence applies to a **family**, not to a claim type: the three assumption claim
types are one family, as are ``method`` and ``model``. ``paper_scalar`` claims
(research question, main contribution, setting) are never concepts; paper pages read
them from ``current_claims`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Family:
    name: str
    claim_types: tuple[str, ...]
    #: Bridge pass order (1 first); ``None`` = only on request.
    bridge_order: int | None
    #: ``False`` only for ``paper_scalar``: no concepts, no bridges.
    concept: bool = True


FAMILIES: tuple[Family, ...] = (
    Family("assumption", ("identification_assumption", "general_assumption", "regularity_condition"), 1),
    Family("estimand", ("estimand",), 2),
    Family("method", ("method", "model"), 3),
    Family("result", ("result",), 4),
    Family("data", ("data",), 5),
    Family("limitation", ("limitation",), None),
    Family("robustness_check", ("robustness_check",), None),
    Family("open_question", ("open_question",), None),
    Family("paper_scalar", ("research_question", "main_contribution", "setting"), None, concept=False),
)
BY_NAME: dict[str, Family] = {f.name: f for f in FAMILIES}
_BY_CLAIM_TYPE: dict[str, str] = {t: f.name for f in FAMILIES for t in f.claim_types}
#: Families that carry concepts, in table order.
CONCEPT_FAMILIES: tuple[str, ...] = tuple(f.name for f in FAMILIES if f.concept)


def family_of(claim_type: str | None) -> str | None:
    """The family of a claim type; ``None`` for an unknown type (the claim is skipped)."""
    if not claim_type:
        return None
    return _BY_CLAIM_TYPE.get(claim_type.strip().lower().replace(" ", "_").replace("-", "_"))


def is_concept_family(family: str | None) -> bool:
    return family in BY_NAME and BY_NAME[family].concept
