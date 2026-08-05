"""Budget-enforcement seam (spec §8) — fail-closed on unverified pricing.

Phase 0 owns the single seam every later cost-generating phase routes through;
enforcement itself (estimate-before-run, stop-on-exceeded) is wired in later
phases. This module ships only the gate + a note.

The rule: a USD-spend-limited task MUST NOT be enforced against a model whose
``llm_capabilities.yaml`` ``pricing_status != "verified"`` — an unverified price
could under-count cost and quietly overshoot the cap. Default is fail-closed
(refuse); the only override is ``LlmBudget.allow_unverified_pricing`` (CLI
``--allow-unverified-pricing``), which proceeds but emits a logged WARNING. This
gates ONLY the USD path; context-window (D4) and other capability reads are
unaffected, and a task with no USD limit is likewise unaffected.
"""

from __future__ import annotations

import logging

from .config.models import LlmBudget, ModelCapability
from .errors import ConfigError

logger = logging.getLogger("seedgraph.budget")

VERIFIED = "verified"


def check_pricing_allowed(model_id: str, capability: ModelCapability, budget: LlmBudget) -> None:
    """Enforce the fail-closed unverified-pricing rule for USD-limited tasks.

    No-op when there is no USD spend limit, or the model's pricing is verified.
    """
    if budget.usd_limit is None:
        return  # no USD limit configured -> nothing to fail closed on
    if capability.pricing_status == VERIFIED:
        return
    if not budget.allow_unverified_pricing:
        raise ConfigError(
            f"refusing a USD-limited task against model '{model_id}': pricing_status="
            f"'{capability.pricing_status}' (not 'verified'); an unverified price could "
            f"under-count cost and overshoot the ${budget.usd_limit:g} spend cap. Pass "
            f"--allow-unverified-pricing (LlmBudget.allow_unverified_pricing=true) to "
            f"proceed with a logged warning, or route to a verified-pricing model."
        )
    logger.warning(
        "Enforcing USD spend limit from UNVERIFIED pricing for model '%s' "
        "(pricing_status=%s); --allow-unverified-pricing is set.",
        model_id,
        capability.pricing_status,
    )
