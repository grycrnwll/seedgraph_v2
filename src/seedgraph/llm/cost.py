"""Cost estimation + DB-backed monthly spend (Track 1, plan §Cost).

Single home for the per-call cost arithmetic that was previously duplicated in
``extraction/runner.py``, ``extraction/chunked_runner.py`` and
``answer/compose.py``. ``estimate_cost`` is the authoritative formula; the runners
re-export it (``_estimate_cost``) so their imports stay stable.

No secrets, no network. ``monthly_spend`` reads ``llm_usage_events`` only.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from ..config.models import LlmBudget, ModelCapability


def estimate_cost(
    cap: "Optional[ModelCapability]", input_tokens: int, output_tokens: int
) -> float:
    """USD cost for ``input_tokens``/``output_tokens`` under capability ``cap``.

    Returns ``0.0`` when ``cap`` is ``None`` or carries no per-MTok pricing (the
    caller decides whether to record ``0.0`` or ``NULL``)."""
    if cap is None:
        return 0.0
    inp = cap.input_usd_per_mtok or 0.0
    out = cap.output_usd_per_mtok or 0.0
    return (input_tokens / 1_000_000) * inp + (output_tokens / 1_000_000) * out


def preflight_estimate(
    cap: "Optional[ModelCapability]",
    input_tokens: int,
    output_tokens: Optional[int] = None,
) -> float:
    """Projected cost BEFORE dispatch. When the output size is unknown we assume
    it mirrors the input (the existing dry-run convention in the extraction
    runner), giving a conservative upper-ish estimate for budget pre-flight."""
    if output_tokens is None:
        output_tokens = input_tokens
    return estimate_cost(cap, input_tokens, output_tokens)


def monthly_spend(conn: sqlite3.Connection, year_month: str) -> float:
    """Summed ``estimated_cost`` over ``llm_usage_events`` for calendar month
    ``year_month`` (``"YYYY-MM"``). ``created_at`` is an ISO-8601 UTC string, so a
    prefix match on the first 7 chars selects the month. NULL costs count as 0."""
    row = conn.execute(
        "SELECT COALESCE(SUM(estimated_cost), 0.0) FROM llm_usage_events "
        "WHERE substr(created_at, 1, 7) = ?",
        (year_month,),
    ).fetchone()
    return float(row[0] or 0.0)


def this_month() -> str:
    """The current calendar month as ``"YYYY-MM"`` (UTC) — the ``monthly_spend`` key."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


# --- Stage C cost controls --------------------------------------------------

@dataclass
class BudgetStatus:
    """A snapshot of the monthly soft-limit + confirmation gate for one prospective run.

    ``projected_usd`` is the plan's ``monthly_spend(conn, this_month) + this_run``:
    the DB-backed prior spend this calendar month plus the estimated cost of the run
    about to start. ``over_monthly_soft_limit`` / ``requires_confirmation`` are the
    two CLI gates (a soft-limit WARNING and a ``typer.confirm`` prompt)."""

    year_month: str
    prior_spend_usd: float
    this_run_usd: float
    projected_usd: float
    monthly_soft_limit_usd: Optional[float]
    over_monthly_soft_limit: bool
    confirmation_threshold_usd: Optional[float]
    requires_confirmation: bool


def budget_status(
    conn: sqlite3.Connection,
    budget: "LlmBudget",
    this_run_usd: float,
    *,
    year_month: Optional[str] = None,
) -> BudgetStatus:
    """Compose the monthly soft-limit + confirmation gate for a prospective run.

    The monthly soft limit is wired to ``monthly_spend(conn, this_month) +
    this_run`` (plan §Cost): prior DB-recorded spend this month plus the run
    estimate. ``require_confirmation_above_usd`` fires when the run estimate alone
    is at or above the configured threshold (the CLI prompts unless ``--yes``)."""
    ym = year_month or this_month()
    prior = monthly_spend(conn, ym)
    run = this_run_usd or 0.0
    projected = prior + run
    limit = budget.monthly_soft_limit_usd
    threshold = budget.require_confirmation_above_usd
    return BudgetStatus(
        year_month=ym,
        prior_spend_usd=prior,
        this_run_usd=run,
        projected_usd=projected,
        monthly_soft_limit_usd=limit,
        over_monthly_soft_limit=limit is not None and projected > limit,
        confirmation_threshold_usd=threshold,
        requires_confirmation=threshold is not None and run >= threshold,
    )
