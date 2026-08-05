"""Policy-gated LLM executor (Track 1) — degradation is a typed result, never an exception.

The single layer every caller dispatches through. It composes the unchanged
``resolve_route`` content-access gate + ``check_pricing_allowed`` budget gate +
the token estimate (:func:`preflight`), the actual provider call with transport
retry and a one-hop runtime fallback (:func:`dispatch`), and the no-body usage row
(:func:`log_llm_usage`). :func:`run_llm` is the public composition; ``preflight`` is
public too. Every failure path returns a terminal :class:`LLMResult` (status from
the plan's taxonomy) so the four callers degrade honestly instead of crashing.

No prompt/response BODY ever reaches the DB — only full sha256 hashes
(``prompt_hash`` / ``response_hash``) plus token counts (cross-cutting #5).
"""

from __future__ import annotations

import random
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..budget import check_pricing_allowed
from ..config.models import GlobalConfig, LlmCapabilities, ModelCapability, ProjectConfig
from ..errors import ConfigError
from .backend import LLMBackend, LLMCompletion, default_backend
from .cost import estimate_cost
from .profiles import (
    is_local_profile,
    is_no_llm_profile,
    is_profile_available,
    resolve_profile,
)
from .providers import ProviderError
from .routing import NoLlmRoute, Route, local_fallback_route, resolve_route
from .secrets import resolve_profile_key
from .tokens import estimate_tokens
from .usage import UsageEvent, hash_text, log_usage

# --- error / result types ---------------------------------------------------

# The code set (plan §Track 1). ``policy_blocked`` / ``budget_exceeded`` /
# ``config_error`` are pre-dispatch blocks; the rest are dispatch failures.
ERROR_CODES = frozenset(
    {
        "config_error",
        "auth_error",
        "policy_blocked",
        "budget_exceeded",
        "rate_limited",
        "timeout",
        "provider_unavailable",
        "invalid_response",
        "schema_validation_failed",
    }
)

# Codes the transport retry will re-attempt (plan §Transport retry).
_RETRYABLE_CODES = frozenset({"rate_limited", "timeout", "provider_unavailable"})
# Codes that trigger the one-hop runtime fallback to ``route.fallback_profile``.
_FALLBACK_CODES = frozenset({"provider_unavailable", "invalid_response"})

# Appended to the repair user message when the failed completion was cut at
# max_tokens (``finish_reason == "length"``, Build C chunk 2 / design D3): the
# generic repair echoes the full prior response under the SAME max_tokens, so a
# budget-truncated output would just re-truncate. Asking for a terser COMPLETE
# regeneration is the fix; raising max_tokens on retry is deliberately rejected
# (runaway cost against the budget substrate).
_TERSER_REPAIR_SUFFIX = (
    "\n\nIMPORTANT: your previous answer was cut off at the output token limit. "
    "Regenerate the COMPLETE object much more tersely: shorter strings, fewer "
    "items, same JSON structure."
)


@dataclass
class LLMError:
    code: str
    message: str
    retryable: bool = False
    provider: Optional[str] = None
    status_code: Optional[int] = None
    request_id: Optional[str] = None


@dataclass
class LLMResult:
    """Outcome of one executor call — a terminal value on every path.

    ``status`` ∈ ``success | skipped_policy | skipped_no_llm | skipped_budget |
    extraction_failed`` (the plan's taxonomy). ``parsed`` is the task-level
    ``parse`` callback's output on success; ``completion`` carries the (possibly
    repair-merged) text + total token counts. Route metadata (``provider`` /
    ``model`` / ``access_mode`` / ``profile_id`` / ``external_full_text``) reflects
    the hop that actually produced the result (after any runtime fallback)."""

    status: str
    completion: Optional[LLMCompletion] = None
    parsed: object = None
    error: Optional[LLMError] = None
    input_tokens: int = 0
    output_tokens: int = 0
    retry_count: int = 0
    latency_ms: int = 0
    provider: Optional[str] = None
    model: Optional[str] = None
    access_mode: Optional[str] = None
    profile_id: Optional[str] = None
    external_full_text: bool = False
    source_access_class: Optional[str] = None
    prompt_hash: Optional[str] = None
    response_hash: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "success"


# code -> terminal LLMResult.status (status taxonomy table, plan §10).
def status_for_code(code: Optional[str]) -> str:
    if code is None:
        return "success"
    if code == "policy_blocked":
        return "skipped_policy"
    if code == "config_error":
        return "skipped_no_llm"
    if code == "budget_exceeded":
        return "skipped_budget"
    return "extraction_failed"


# LLMResult.status -> usage-row ``status`` value (reuses the extraction_runs vocab;
# a dispatch failure is logged as the generic ``error`` with the precise code).
def _usage_status(result_status: str) -> str:
    return "error" if result_status == "extraction_failed" else result_status


# --- runtime-fallback hop ---------------------------------------------------

@dataclass
class Hop:
    """A resolved next provider for the one-hop runtime fallback chain."""

    backend: LLMBackend
    model: Optional[str]
    provider: Optional[str] = None
    access_mode: Optional[str] = None
    profile_id: Optional[str] = None
    external_full_text: bool = False


# --- transport call + retry -------------------------------------------------

_RETRY_MAX = 3
_RETRY_BASE_DELAY = 0.5


def _sleep(seconds: float) -> None:  # pragma: no cover - patched/skipped in tests
    time.sleep(seconds)


def _call_with_retry(
    backend: LLMBackend,
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str],
    temperature: float,
    max_tokens: int,
) -> tuple[LLMCompletion, int]:
    """Call ``backend.complete`` with transport retry; return (completion, retries).

    Retries only ``rate_limited|timeout|provider_unavailable`` (exp backoff +
    jitter, honoring a 429 ``Retry-After``), capped at :data:`_RETRY_MAX`. Never
    retries auth/config/invalid_response — those re-raise immediately."""
    last: Optional[ProviderError] = None
    for attempt in range(_RETRY_MAX):
        try:
            completion = backend.complete(
                system_prompt,
                user_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion, attempt
        except ProviderError as exc:
            last = exc
            if not exc.retryable or exc.code not in _RETRYABLE_CODES or attempt == _RETRY_MAX - 1:
                raise
            delay = exc.retry_after if exc.retry_after is not None else (
                _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, _RETRY_BASE_DELAY)
            )
            _sleep(delay)
    raise last  # pragma: no cover - loop always returns or raises above


# --- dispatch (inner) -------------------------------------------------------

def dispatch(
    backend: LLMBackend,
    system_prompt: str,
    user_prompt: str,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    access_mode: Optional[str] = None,
    profile_id: Optional[str] = None,
    external_full_text: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    parse: Optional[Callable[[str], tuple]] = None,
    repair_prompt: Optional[Callable[[str, object], str]] = None,
    next_hop: Optional[Callable[[LLMError], Optional[Hop]]] = None,
) -> LLMResult:
    """One provider call (+ optional one repair re-prompt) → terminal LLMResult.

    Never raises: a ``ProviderError`` becomes a typed extraction_failed/auth result,
    a ``ConfigError`` (StubRealBackend) becomes skipped_no_llm, any other exception
    is caught as provider_unavailable. ``parse(text) -> (parsed|None, errors)`` is
    the task-level validator; if it yields ``None`` and ``repair_prompt`` is given,
    one repair call is made. On ``provider_unavailable`` / ``invalid_response`` (a
    down local provider / missing model) and a non-None ``next_hop``, the hop's
    backend is dispatched once more (content gate re-applied by the caller's hop
    factory) before the terminal degradation is returned."""
    start = time.monotonic()
    prompt_hash = hash_text(system_prompt + "\x00" + user_prompt)

    try:
        completion, retry_count = _call_with_retry(
            backend,
            system_prompt,
            user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ConfigError as exc:
        return _failed_result(
            LLMError("config_error", str(exc), False, provider),
            provider, model, access_mode, profile_id, external_full_text,
            prompt_hash, start,
        )
    except ProviderError as exc:
        error = LLMError(
            exc.code, exc.message, exc.retryable, exc.provider or provider,
            exc.status_code, exc.request_id,
        )
        if next_hop is not None and exc.code in _FALLBACK_CODES:
            hop = next_hop(error)
            if hop is not None:
                return dispatch(
                    hop.backend,
                    system_prompt,
                    user_prompt,
                    model=hop.model,
                    provider=hop.provider,
                    access_mode=hop.access_mode,
                    profile_id=hop.profile_id,
                    external_full_text=hop.external_full_text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    parse=parse,
                    repair_prompt=repair_prompt,
                    next_hop=None,  # cap at one hop
                )
        return _failed_result(
            error, provider, model, access_mode, profile_id, external_full_text, prompt_hash, start
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never traceback (no-crash regressions)
        return _failed_result(
            LLMError("provider_unavailable", str(exc), True, provider),
            provider, model, access_mode, profile_id, external_full_text, prompt_hash, start,
        )

    input_tokens = completion.input_tokens
    output_tokens = completion.output_tokens
    response_text = completion.text
    finish_reason = completion.finish_reason
    terser_applied = False
    parsed = response_text
    errors: object = None

    if parse is not None:
        parsed, errors = parse(response_text)
        if parsed is None and repair_prompt is not None:
            repair_user = repair_prompt(response_text, errors)
            if finish_reason == "length":
                # Truncated-at-max_tokens output: the generic repair would just
                # re-truncate; ask for a terser complete regeneration (D3). The
                # repair_prompt callback signature stays untouched.
                repair_user += _TERSER_REPAIR_SUFFIX
                terser_applied = True
            try:
                completion2, rc2 = _call_with_retry(
                    backend, system_prompt, repair_user,
                    model=model, temperature=temperature, max_tokens=max_tokens,
                )
            except (ProviderError, ConfigError, Exception):  # noqa: BLE001
                completion2, rc2 = None, 0
            if completion2 is not None:
                input_tokens += completion2.input_tokens
                output_tokens += completion2.output_tokens
                response_text = completion2.text
                finish_reason = completion2.finish_reason
                retry_count += rc2 + 1
                parsed, errors = parse(response_text)

    merged = LLMCompletion(
        text=response_text, input_tokens=input_tokens, output_tokens=output_tokens,
        finish_reason=finish_reason,
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    response_hash = hash_text(response_text)

    if parse is not None and parsed is None:
        message = "model output failed validation after repair"
        if finish_reason == "length":
            # Name the real cause in the audit trail without a new status code
            # (finish_reason is deliberately NOT persisted to llm_usage_events).
            message += " (output truncated at max_tokens{})".format(
                " after terser retry" if terser_applied else ""
            )
        return LLMResult(
            status="extraction_failed",
            completion=merged,
            parsed=None,
            error=LLMError("schema_validation_failed", message, False, provider),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            retry_count=retry_count,
            latency_ms=latency_ms,
            provider=provider,
            model=model,
            access_mode=access_mode,
            profile_id=profile_id,
            external_full_text=external_full_text,
            prompt_hash=prompt_hash,
            response_hash=response_hash,
        )

    return LLMResult(
        status="success",
        completion=merged,
        parsed=parsed,
        error=None,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retry_count=retry_count,
        latency_ms=latency_ms,
        provider=provider,
        model=model,
        access_mode=access_mode,
        profile_id=profile_id,
        external_full_text=external_full_text,
        prompt_hash=prompt_hash,
        response_hash=response_hash,
    )


def _failed_result(
    error: LLMError,
    provider, model, access_mode, profile_id, external_full_text,
    prompt_hash: str,
    start: float,
) -> LLMResult:
    return LLMResult(
        status=status_for_code(error.code),
        completion=None,
        parsed=None,
        error=error,
        input_tokens=0,
        output_tokens=0,
        retry_count=0,
        latency_ms=int((time.monotonic() - start) * 1000),
        provider=provider,
        model=model,
        access_mode=access_mode,
        profile_id=profile_id,
        external_full_text=external_full_text,
        prompt_hash=prompt_hash,
        response_hash=None,
    )


def terminal_block(code: str, message: str, *, provider=None, model=None, profile_id=None) -> LLMResult:
    """A pre-dispatch terminal result (policy / budget / no-llm block).

    Carries no tokens and no response hash (nothing was dispatched); the caller
    logs a usage row recording the block per the taxonomy."""
    return LLMResult(
        status=status_for_code(code),
        error=LLMError(code, message, False, provider),
        provider=provider,
        model=model,
        profile_id=profile_id,
    )


# --- secret-aware backend factory -------------------------------------------

def build_backend(profile, cfg: ProjectConfig, *, transport=None) -> LLMBackend:
    """Resolve a real backend for ``profile`` with secrets injected centrally.

    The api key is resolved here (never inside an adapter) via
    ``resolve_profile_key`` (keyring → env chain, ADR-0002); ``base_url`` comes
    from the profile. ``transport`` is the test seam (httpx.MockTransport)."""
    try:
        api_key = resolve_profile_key(profile)
    except ConfigError:
        api_key = None
    return default_backend(
        provider=profile.provider,
        model=profile.model,
        base_url=profile.base_url,
        api_key=api_key,
        transport=transport,
    )


# --- usage logging (inner) --------------------------------------------------

def log_llm_usage(
    conn: sqlite3.Connection,
    *,
    task_type: str,
    result: LLMResult,
    source_access_class: str,
    run_id: Optional[str] = None,
    cap: Optional[ModelCapability] = None,
    external_full_text: Optional[bool] = None,
    estimated_cost: Optional[float] = None,
) -> str:
    """Write one ``llm_usage_events`` row from a terminal :class:`LLMResult`.

    No bodies: only ``prompt_hash`` / ``response_hash`` (full sha256). ``status`` /
    ``error_code`` / ``latency_ms`` / ``request_id`` / ``retry_count`` come from the
    result; cost is computed from ``cap`` unless given."""
    ext = result.external_full_text if external_full_text is None else external_full_text
    if estimated_cost is None and cap is not None and (cap.input_usd_per_mtok or cap.output_usd_per_mtok):
        estimated_cost = estimate_cost(cap, result.input_tokens, result.output_tokens)
    request_id = result.error.request_id if result.error is not None else None
    return log_usage(
        conn,
        UsageEvent(
            task_type=task_type,
            provider=result.provider,
            model=result.model,
            access_mode=result.access_mode,
            run_id=run_id,
            input_tokens=result.input_tokens or None,
            output_tokens=result.output_tokens or None,
            estimated_cost=estimated_cost,
            source_access_class=source_access_class,
            external_full_text=bool(ext),
            status=_usage_status(result.status),
            error_code=result.error.code if result.error is not None else None,
            latency_ms=result.latency_ms or None,
            request_id=request_id,
            prompt_hash=result.prompt_hash,
            response_hash=result.response_hash,
            retry_count=result.retry_count or 0,
        ),
    )


# --- preflight (public) -----------------------------------------------------

@dataclass
class Plan:
    """A passed-preflight dispatch plan (route + resolved profile/model/cap)."""

    route: Route
    profile: object
    model: Optional[str]
    cap: Optional[ModelCapability]
    external_full_text: bool
    source_access_class: str


def preflight(
    task_type: str,
    access_class: str,
    cfg: ProjectConfig,
    *,
    capabilities: Optional[LlmCapabilities] = None,
    profile_id_override: Optional[str] = None,
    confirm_external: bool = False,
    prompt_tokens: Optional[int] = None,
) -> tuple[Optional[Plan], Optional[LLMResult]]:
    """Compose the unchanged content-access gate + pricing gate + token check.

    Returns ``(plan, None)`` on a go, or ``(None, terminal_result)`` for a block
    (the caller logs a usage row and degrades). The content gate is ``resolve_route``
    unchanged; a refusal falls back to a local profile (``local_fallback_route``) so
    restricted source text never leaves the machine, else skipped_policy."""
    effective = cfg
    if confirm_external or profile_id_override is not None:
        effective = cfg.model_copy(deep=True)
        if confirm_external:
            effective.content_policy.external_llm_for_private_full_text = True
        if profile_id_override is not None:
            route_cfg = effective.llm.routes.get(task_type)
            if route_cfg is not None:
                route_cfg.preferred_profile = profile_id_override

    try:
        route = resolve_route(task_type, access_class, effective)
    except ConfigError as exc:
        route = local_fallback_route(effective, task_type)
        if route is None:
            return None, terminal_block("policy_blocked", str(exc))

    if isinstance(route, NoLlmRoute):
        return None, terminal_block(
            "config_error",
            f"no usable LLM profile for task '{task_type}' (deterministic_fallback="
            f"{route.deterministic_fallback})",
        )

    profile = resolve_profile(route.profile_id, effective)
    model = profile.model
    cap = capabilities.models.get(model) if (capabilities is not None and model) else None

    # Budget gate (fail-closed unverified pricing only when a USD limit is armed).
    if cap is not None and effective.budget.usd_limit is not None:
        try:
            check_pricing_allowed(model, cap, effective.budget)
        except ConfigError as exc:
            return None, terminal_block(
                "budget_exceeded", str(exc), provider=route.provider, model=model,
                profile_id=route.profile_id,
            )

    return (
        Plan(
            route=route,
            profile=profile,
            model=model,
            cap=cap,
            external_full_text=bool(route.external_full_text),
            source_access_class=access_class,
        ),
        None,
    )


# --- run_llm (public) -------------------------------------------------------

def run_llm(
    task_type: str,
    system_prompt: str,
    user_prompt: str,
    *,
    access_class: str,
    config: Optional[ProjectConfig] = None,
    capabilities: Optional[LlmCapabilities] = None,
    conn: Optional[sqlite3.Connection] = None,
    run_id: Optional[str] = None,
    profile_id_override: Optional[str] = None,
    confirm_external: bool = False,
    parse: Optional[Callable[[str], tuple]] = None,
    repair_prompt: Optional[Callable[[str, object], str]] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    backend: Optional[LLMBackend] = None,
    backend_factory: Optional[Callable[[object, ProjectConfig], LLMBackend]] = None,
    transport=None,
    log: bool = True,
) -> LLMResult:
    """Public policy-gated call: preflight → dispatch (+ one-hop fallback) → usage row.

    The runtime fallback chain (plan §2): the preferred profile is tried; on
    ``provider_unavailable`` / ``invalid_response`` (a down local provider / missing
    local model) the route's ``fallback_profile`` is preflighted again (content gate
    re-applied so restricted full text can't leak to the hosted fallback) and
    dispatched once. ``backend`` pins one backend for tests; otherwise
    ``backend_factory`` / secret-aware :func:`build_backend` resolves it per hop."""
    if config is None:
        config = GlobalConfig()

    def _factory(profile, cfg):
        if backend is not None:
            return backend
        if backend_factory is not None:
            return backend_factory(profile, cfg)
        return build_backend(profile, cfg, transport=transport)

    plan, block = preflight(
        task_type, access_class, config,
        capabilities=capabilities, profile_id_override=profile_id_override,
        confirm_external=confirm_external,
    )
    if block is not None:
        if log and conn is not None:
            log_llm_usage(conn, task_type=task_type, result=block, source_access_class=access_class, run_id=run_id)
        return block

    client = _factory(plan.profile, config)
    next_hop = _make_next_hop(task_type, access_class, config, capabilities, plan, _factory)

    result = dispatch(
        client,
        system_prompt,
        user_prompt,
        model=plan.model,
        provider=plan.route.provider,
        access_mode=plan.route.access_mode,
        profile_id=plan.route.profile_id,
        external_full_text=plan.external_full_text,
        temperature=temperature,
        max_tokens=max_tokens,
        parse=parse,
        repair_prompt=repair_prompt,
        next_hop=next_hop,
    )
    result.source_access_class = access_class
    if log and conn is not None:
        log_llm_usage(conn, task_type=task_type, result=result, source_access_class=access_class, run_id=run_id, cap=plan.cap)
    return result


def _make_next_hop(task_type, access_class, config, capabilities, plan, factory):
    """Build the one-hop runtime fallback closure for ``run_llm``.

    Re-preflights on ``route.fallback_profile`` (content gate re-applied) and
    returns a :class:`Hop`, or ``None`` when no usable fallback exists / the gate
    forbids it."""
    fallback_id = plan.route.fallback
    used = {"done": False}

    def _next_hop(error: LLMError) -> Optional[Hop]:
        if used["done"] or not fallback_id:
            return None
        fb_plan, fb_block = preflight(
            task_type, access_class, config,
            capabilities=capabilities, profile_id_override=fallback_id,
        )
        if fb_block is not None or fb_plan is None:
            return None
        # Don't loop back onto the same provider/profile that just failed.
        if fb_plan.route.profile_id == plan.route.profile_id:
            return None
        used["done"] = True
        return Hop(
            backend=factory(fb_plan.profile, config),
            model=fb_plan.model,
            provider=fb_plan.route.provider,
            access_mode=fb_plan.route.access_mode,
            profile_id=fb_plan.route.profile_id,
            external_full_text=fb_plan.external_full_text,
        )

    return _next_hop


__all__ = [
    "ERROR_CODES",
    "Hop",
    "LLMError",
    "LLMResult",
    "Plan",
    "build_backend",
    "dispatch",
    "log_llm_usage",
    "preflight",
    "run_llm",
    "status_for_code",
    "terminal_block",
]
