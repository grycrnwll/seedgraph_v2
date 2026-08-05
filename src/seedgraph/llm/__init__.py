"""LLM access substrate: profiles, per-task routing, secrets, usage logging.

Phase 0 validates config and routes only — it invokes no model and ships no
backend dispatch layer.
"""

from .cost import (
    BudgetStatus,
    budget_status,
    estimate_cost,
    monthly_spend,
    preflight_estimate,
    this_month,
)
from .profiles import is_profile_available, resolve_profile
from .routing import NoLlmRoute, Route, resolve_route
from .secrets import resolve_secret
from .usage import UsageEvent, hash_text, log_usage

__all__ = [
    "BudgetStatus",
    "NoLlmRoute",
    "Route",
    "UsageEvent",
    "budget_status",
    "estimate_cost",
    "hash_text",
    "is_profile_available",
    "log_usage",
    "monthly_spend",
    "preflight_estimate",
    "resolve_profile",
    "resolve_route",
    "resolve_secret",
    "this_month",
]
