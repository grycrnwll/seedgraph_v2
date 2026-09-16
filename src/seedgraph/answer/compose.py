"""Prompt build + token budgeting + single LLM call + citation materialization
(plan §5 / §6 / §8, 08 §5 steps 6/9/10; decisions 82/53/15/D9).

This module owns the one LLM call per answer (decision 82) and everything around it
that makes "cite by provided id" the *only representable* citation (should-fix #6):
a numbered-evidence prompt with an in-process ``evidence_index``
(``[E{n}] -> (work_id, [span_id...])``), strict-JSON parse with one repair retry, and
a number→id mapping that drops any out-of-range / invented marker (warned). The guard
then re-checks membership (belt-and-suspenders).

Content-access boundary (plan §7, decision 76/30/60): only bounded verbatim
fragments (``evidence_spans.exact_quote``, truncated) ever feed the model, never full
documents — so ``llm_usage_events.external_full_text = false`` is truthful.
``user_supplied_private`` fragments are not emitted to an external profile when
``content_policy.external_llm_for_answer_generation`` forbids it: route to a local
profile, else degrade to ``retrieval_only``.

No-LLM / degrade (decisions 58/38): ``--no-llm``, no resolvable profile, a
budget/policy block, or a second JSON-parse failure → ``retrieval_only`` with
``answer_text=""`` and the ranked cited evidence list. Honest degradation, never
rule-based prose.

Token budgeting (plan §8, D9): the model's ``context_window_tokens`` is read from the
bundled ``llm_capabilities.yaml`` snapshot; the effective evidence cap is
``min(max_evidence_tokens, context_window_tokens − prompt_overhead − reserved_output)``.
Over-budget candidates are dropped and are **absent** from the allowed set, so the
model cannot cite what it never saw.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from ..config.loader import load_llm_capabilities
from ..config.models import GlobalConfig, LlmCapabilities
from ..errors import ConfigError
from ..llm import executor as run_executor
from ..llm.backend import LLMBackend, default_backend
from ..llm.parse import parse_json_object
from ..llm.profiles import is_local_profile, is_no_llm_profile, is_profile_available, resolve_profile
from ..llm.routing import NoLlmRoute, Route, resolve_route
from ..llm.tokens import estimate_tokens
from ..vocab import AccessClass, AnswerCategory, AnswerMode, QueryType, is_shareable
from .trace import ShownEvidence
from .types import (
    AllowedSet,
    AnswerEnvelope,
    Citation,
    QuerySpec,
    RankedCandidate,
    Recommendation,
)

# Versioned prompt id stamped into llm_provenance (doc 13 §15).
PROMPT_VERSION = "answer_v1"
_PROMPT_FILE = Path(__file__).parent / "prompts" / "answer_v1.txt"

TASK_TYPE = "answer_generation"

# Scaffold defaults (mirrored by ``cfg.answer.*``; plan §8/§11). Kept so the
# budgeting/materialization seams are testable without a full config.
DEFAULT_MAX_EVIDENCE_TOKENS = 6000
DEFAULT_MAX_FRAGMENT_CHARS = 1200
DEFAULT_PROMPT_OVERHEAD_TOKENS = 800
DEFAULT_RESERVED_OUTPUT_TOKENS = 1024

# Offline test seam (mirrors ``extraction.runner._BACKEND_OVERRIDE``): tests
# monkeypatch this to inject a ``FakeLLMBackend`` so pytest never touches the
# network, a key, or a real provider. Production leaves it ``None``.
_BACKEND_OVERRIDE: LLMBackend | None = None

_EVIDENCE_OPEN = "<<<EVIDENCE"
_EVIDENCE_CLOSE = "EVIDENCE>>>"


# --- AnswerTrace carriers (00 §4.1) — compose's slice of the trace ----------

@dataclass
class ShownReport:
    """``build_prompt``'s shown-evidence report (design 00 §4.1).

    ``evidence`` is the ordered ``[E{n}]`` blocks actually rendered into the prompt (as
    :class:`~seedgraph.answer.trace.ShownEvidence` — the T3 structured shown-list, never
    the prompt body); ``cut_index`` is the count of shown blocks, i.e. the index at which
    the token-budget loop broke (``compose.py`` budget check). Because that loop *breaks*
    rather than skips, the shown candidates are always the prefix ``candidates[:cut_index]``
    — which is what makes ``cut_budget`` **exact** in the harness (``position > cut_index``)
    rather than inferred from :class:`AllowedSet` membership.
    """

    evidence: list[ShownEvidence]
    cut_index: int


@dataclass
class ComposeTrace:
    """Compose→harness carrier for the AnswerTrace fields ``generate`` owns (design 00
    §4.1): the shown-evidence list, the prompt hash + version (T3 — hash only, never the
    body), and which degrade site fired (``degrade_reasons`` mirrors the warning that
    site appended past the ones passed in).

    ``generate`` **always** returns one; ``shown_evidence`` / ``cut_index`` /
    ``prompt_version`` / ``prompt_sha256`` stay ``None`` on the degrades that return
    before a prompt is built (no_llm/NoLlmRoute, private-local-only, pricing, empty
    evidence index) and are populated on the LLM path AND the failed-dispatch degrade
    (a prompt *was* built and dispatched there — 00 §3). Not persisted: the harness lifts
    these into the persisted :class:`~seedgraph.answer.trace.AnswerTrace`.
    """

    shown_evidence: list[ShownEvidence] | None = None
    cut_index: int | None = None
    prompt_version: str | None = None
    prompt_sha256: str | None = None
    degrade_reasons: list[str] = field(default_factory=list)


# --- prompt building + token budgeting (plan §6.3 / §8) --------------------

def build_prompt(
    candidates: list[RankedCandidate],
    spec: QuerySpec,
    *,
    max_evidence_tokens: int = DEFAULT_MAX_EVIDENCE_TOKENS,
    context_window_tokens: int | None = None,
    max_fragment_chars: int = DEFAULT_MAX_FRAGMENT_CHARS,
    prompt_overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
    reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
) -> tuple[str, dict[int, tuple[str, list[str]]], AllowedSet, ShownReport]:
    """Render the numbered-evidence prompt + the load-bearing ``evidence_index``
    (plan §6.3 / §8).

    Each kept candidate is rendered as an ``[E{n}]`` block behind a fixed delimiter,
    its quote truncated to ``max_fragment_chars`` (keeping it a *fragment*). Candidates
    are added in rank order until the effective cap
    ``min(max_evidence_tokens, context_window − overhead − reserved)`` is reached; the
    rest are dropped (the first candidate is always shown so a non-empty set yields at
    least one block). The prompt states that evidence text is *data, not instructions*.

    Returns ``(prompt_text, evidence_index, allowed_set, shown_report)`` — ``allowed_set``
    is built from the **shown** candidates only, so dropped-for-budget items are
    uncitable; ``shown_report`` (00 §4.1) carries the per-block shown-evidence records
    plus ``cut_index`` (the budget-break index) so the AnswerTrace can tag those dropped
    items ``cut_budget`` **exactly** rather than infer them from allowed-set membership.
    """
    effective_cap = max_evidence_tokens
    if context_window_tokens is not None:
        effective_cap = min(
            effective_cap,
            context_window_tokens - prompt_overhead_tokens - reserved_output_tokens,
        )
    effective_cap = max(effective_cap, 1)

    blocks: list[str] = []
    evidence_index: dict[int, tuple[str, list[str]]] = {}
    shown: list[ShownEvidence] = []
    shown_work_ids: list[str] = []
    shown_span_ids: list[str] = []
    running = 0
    n = 1
    for candidate in candidates:
        item = candidate.item
        truncated = len(item.text) > max_fragment_chars
        quote = item.text[:max_fragment_chars] + "…" if truncated else item.text
        header = f"[E{n}] (work={item.work_id}"
        if item.section:
            header += f", section={item.section}"
        header += ")"
        block = f"{header}\n{_EVIDENCE_OPEN}\n{quote}\n{_EVIDENCE_CLOSE}"
        block_tokens = estimate_tokens(block)
        if blocks and running + block_tokens > effective_cap:
            break  # over budget -> dropped; NOT added to the allowed set (cut_budget)
        span_ids = [item.span_id] if item.span_id else []
        evidence_index[n] = (item.work_id, span_ids)
        blocks.append(block)
        running += block_tokens
        # T3 shown-list record: the [E{n}] block as rendered (bounded quote, never the
        # prompt body) — generate lifts this straight onto the AnswerTrace.
        shown.append(
            ShownEvidence(
                n=n,
                item_id=item.item_id,
                work_id=item.work_id,
                span_ids=list(span_ids),
                section=item.section,
                quote=quote,
                token_estimate=block_tokens,
                truncated=truncated,
            )
        )
        if item.work_id not in shown_work_ids:
            shown_work_ids.append(item.work_id)
        for span_id in span_ids:
            if span_id not in shown_span_ids:
                shown_span_ids.append(span_id)
        n += 1

    template = _PROMPT_FILE.read_text(encoding="utf-8")
    evidence_text = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    # NOT str.format — the template contains literal { } in its JSON example.
    prompt = template.replace("{evidence_blocks}", evidence_text).replace(
        "{question}", spec.question
    )

    allowed = AllowedSet(
        work_ids=frozenset(shown_work_ids),
        span_ids=frozenset(shown_span_ids),
        retrieved_item_ids=tuple(shown_work_ids) + tuple(shown_span_ids),
    )
    # cut_index == len(shown): the budget loop breaks (never skips), so the shown blocks
    # are exactly candidates[:cut_index] (00 §4.1).
    return prompt, evidence_index, allowed, ShownReport(evidence=shown, cut_index=len(shown))


# --- context expansion (08 §5 step 6) — best-effort, fail-closed ------------

def expand_context(
    project_conn: sqlite3.Connection,
    cache_root,
    *,
    span_id: str,
    window: int = 200,
) -> tuple[str, list[str]]:
    """08 §5 step 6 — best-effort surrounding-markdown context, fail-closed.

    Opens ``cache.db`` **read-only** and slices a ``window`` around the span. If the
    markdown is missing, unreadable, or its hash mismatches the span's anchor (stale),
    falls back to the stored ``evidence_spans.exact_quote`` and returns a
    ``stale_context`` warning — never a hard error (decisions 24/42/53). Returns
    ``(context_text, warnings)``.
    """
    row = project_conn.execute(
        "SELECT markdown_id, markdown_hash, start_char, end_char, exact_quote "
        "FROM evidence_spans WHERE span_id = ?",
        (span_id,),
    ).fetchone()
    if row is None:
        return "", ["stale_context"]
    markdown_id, md_hash, start_char, end_char, exact_quote = row
    try:
        from .. import cache_access

        cache_conn = cache_access.open_cache_ro(cache_root)
        try:
            md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
        finally:
            cache_conn.close()
        if md is None or md.markdown_hash != md_hash:
            return exact_quote, ["stale_context"]
        lo = max(0, int(start_char) - window)
        hi = min(len(md.text), int(end_char) + window)
        return md.text[lo:hi], []
    except Exception:  # noqa: BLE001 — any cache failure falls back to the verbatim quote
        return exact_quote, ["stale_context"]


# --- citation materialization (08 §5 step 10) ------------------------------

def _work_meta(project_conn: sqlite3.Connection, work_id: str) -> tuple[str | None, int | None]:
    row = project_conn.execute(
        "SELECT canonical_title, year FROM works WHERE work_id = ?", (work_id,)
    ).fetchone()
    if row is None:
        return None, None
    return row[0], row[1]


def _span_detail(
    project_conn: sqlite3.Connection, span_id: str
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return ``(exact_quote, section_kind, epistemic_type, assertion_status)`` for a
    span — quote is the verbatim authority (decision 53); the D2 provenance fields are
    copied from the highest-rank linked claim."""
    row = project_conn.execute(
        "SELECT e.exact_quote, d.section_kind "
        "FROM evidence_spans e LEFT JOIN document_sections d ON d.section_id = e.section_id "
        "WHERE e.span_id = ?",
        (span_id,),
    ).fetchone()
    if row is None:
        return None, None, None, None
    quote, section = row[0], row[1]
    from ..semantic.current import CURRENT_CLAIMS_CTE

    crow = project_conn.execute(
        CURRENT_CLAIMS_CTE + "SELECT c.epistemic_type, c.assertion_status FROM claim_spans cs "
        "JOIN current_claims c ON c.claim_id = cs.claim_id "
        "WHERE cs.span_id = ? ORDER BY cs.rank LIMIT 1",
        (span_id,),
    ).fetchone()
    epi = crow[0] if crow is not None else None
    assertion = crow[1] if crow is not None else None
    return quote, section, epi, assertion


def _build_citation(
    project_conn: sqlite3.Connection,
    work_id: str,
    span_ids: list[str],
    *,
    fallback_epistemic: str | None = None,
    fallback_assertion: str | None = None,
) -> Citation:
    title, year = _work_meta(project_conn, work_id)
    quote = None
    section = None
    epistemic = fallback_epistemic or "deterministic"
    assertion = fallback_assertion
    if span_ids:
        q, sec, epi, ass = _span_detail(project_conn, span_ids[0])
        quote = q
        section = sec
        if epi is not None:
            epistemic = epi
        assertion = ass if ass is not None else fallback_assertion
    return Citation(
        work_id=work_id,
        title=title,
        year=year,
        span_ids=list(span_ids),
        quote=quote,
        section=section,
        epistemic_type=epistemic,
        assertion_status=assertion,
    )


def materialize_citations(
    raw_markers: list,
    evidence_index: dict[int, tuple[str, list[str]]],
    project_conn: sqlite3.Connection,
) -> tuple[list[Citation], list[str]]:
    """08 §5 step 10 — map emitted ``[E{n}]`` markers back to concrete citations.

    Each in-range marker resolves through ``evidence_index`` to a ``work_id`` +
    ``span_id``s; ``Citation.quote`` is the verbatim ``evidence_spans.exact_quote``
    (decision 53) and the D2 ``epistemic_type`` / ``assertion_status`` are copied from
    ``extracted_claims``. Any marker not a key in ``evidence_index`` (out-of-range /
    invented) is **dropped** and recorded as a warning — making the membership
    guarantee structural, not prompt-trusted (plan §6.3). Returns ``(citations,
    warnings)``.
    """
    warnings: list[str] = []
    # Merge markers per work_id, preserving first-seen order, unioning span_ids.
    ordered_works: list[str] = []
    spans_by_work: dict[str, list[str]] = {}
    for marker in raw_markers:
        try:
            n = int(marker)
        except (TypeError, ValueError):
            warnings.append(f"dropped_invented_citation_marker:{marker}")
            continue
        if n not in evidence_index:
            warnings.append(f"dropped_invented_citation_marker:{n}")
            continue
        work_id, span_ids = evidence_index[n]
        if work_id not in spans_by_work:
            spans_by_work[work_id] = []
            ordered_works.append(work_id)
        for span_id in span_ids:
            if span_id not in spans_by_work[work_id]:
                spans_by_work[work_id].append(span_id)

    citations = [
        _build_citation(project_conn, work_id, spans_by_work[work_id])
        for work_id in ordered_works
    ]
    return citations, warnings


def build_recommendations(
    project_conn: sqlite3.Connection,
    spec: QuerySpec,
    candidates: list[RankedCandidate],
) -> list[Recommendation]:
    """08 §12 — surface metadata-only / unavailable works as recommendations.

    A work present in the corpus but lacking spans (e.g. ``metadata_only``) is
    returned as a :class:`Recommendation` (``status`` + ``action_hint``), **never** as
    a citation. Returns ``[]`` when nothing qualifies (or the table is absent).
    """
    cand_works = {c.item.work_id for c in candidates}
    try:
        rows = project_conn.execute(
            "SELECT d.work_id, w.canonical_title, w.year FROM project_documents d "
            "JOIN works w ON w.work_id = d.work_id "
            "WHERE d.inclusion_status = 'metadata_only' ORDER BY d.work_id"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[Recommendation] = []
    for work_id, title, year in rows:
        if work_id in cand_works:
            continue
        span_count = project_conn.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE work_id = ?", (work_id,)
        ).fetchone()[0]
        if span_count:
            continue
        out.append(
            Recommendation(
                work_id=work_id,
                title=title,
                year=year,
                reason="present in the corpus as metadata-only (no extracted full text)",
                status="metadata_only",
                action_hint="upload the PDF to extract evidence and cite this work",
            )
        )
        if len(out) >= 5:
            break
    return out


# --- JSON parse helpers -----------------------------------------------------

def _parse_json(text: str) -> dict | None:
    """Parse the model's strict-JSON reply; tolerates prose / code fences by
    extracting the first balanced ``{...}`` object (shared brace-depth parser,
    ``llm/parse.py``). Returns ``None`` on failure."""
    return parse_json_object(text)


# Task-level parse/repair callbacks for the executor (JSON repair stays here, not
# in the transport layer — plan §Track 1).
_ANSWER_REPAIR = (
    "Your previous reply was not valid JSON. Return ONLY a single strict JSON "
    "object with keys: query_type, answer_category, answer_text, cited_markers, "
    "insufficient_evidence."
)

# Executor error code -> answer warning string (status taxonomy, plan §10). A
# malformed/unrepairable response keeps the historical ``compose_failed`` warning.
_WARN_BY_CODE = {
    "schema_validation_failed": "compose_failed",
    "invalid_response": "compose_failed",
    "provider_unavailable": "model_unavailable",
    "rate_limited": "rate_limited",
    "timeout": "timed_out",
    "auth_error": "auth_failed",
    "config_error": "no_llm",
    "policy_blocked": "private_content_local_only",
    "budget_exceeded": "budget_exceeded",
}


def _parse_json_pair(text: str) -> tuple[dict | None, object]:
    """Executor ``parse`` adapter: ``(parsed_dict_or_None, errors)``."""
    return _parse_json(text), None


def _answer_repair_prompt(text: str, errors: object) -> str:
    return _ANSWER_REPAIR


def _warn_for(error) -> str:
    if error is None:
        return "compose_failed"
    return _WARN_BY_CODE.get(error.code, "compose_failed")


def _coerce_query_type(value) -> QueryType:
    try:
        return QueryType(value)
    except (ValueError, TypeError):
        return QueryType.factual  # closest safe default (decision 14)


def _coerce_answer_category(value) -> AnswerCategory:
    try:
        return AnswerCategory(value)
    except (ValueError, TypeError):
        return AnswerCategory.unresolved  # unknown -> unresolved (decision 14)


# --- retrieval-only envelope (no-LLM / degrade) ----------------------------

def _allowed_from_candidates(candidates: list[RankedCandidate]) -> AllowedSet:
    work_ids: list[str] = []
    span_ids: list[str] = []
    for candidate in candidates:
        if candidate.item.work_id not in work_ids:
            work_ids.append(candidate.item.work_id)
        if candidate.item.span_id and candidate.item.span_id not in span_ids:
            span_ids.append(candidate.item.span_id)
    return AllowedSet(
        work_ids=frozenset(work_ids),
        span_ids=frozenset(span_ids),
        retrieved_item_ids=tuple(work_ids) + tuple(span_ids),
    )


def _retrieval_only_envelope(
    candidates: list[RankedCandidate],
    spec: QuerySpec,
    project_conn: sqlite3.Connection,
    *,
    warnings: list[str],
    llm_provenance: dict | None = None,
) -> tuple[AnswerEnvelope, AllowedSet]:
    # One citation per work, unioning the candidates' spans (verbatim evidence list).
    ordered_works: list[str] = []
    spans_by_work: dict[str, list[str]] = {}
    fallback_by_work: dict[str, tuple[str | None, str | None]] = {}
    for candidate in candidates:
        item = candidate.item
        if item.work_id not in spans_by_work:
            spans_by_work[item.work_id] = []
            ordered_works.append(item.work_id)
            fallback_by_work[item.work_id] = (item.epistemic_type, item.assertion_status)
        if item.span_id and item.span_id not in spans_by_work[item.work_id]:
            spans_by_work[item.work_id].append(item.span_id)
    citations: list[Citation] = []
    for work_id in ordered_works:
        epi, ass = fallback_by_work[work_id]
        citations.append(
            _build_citation(
                project_conn,
                work_id,
                spans_by_work[work_id],
                fallback_epistemic=epi,
                fallback_assertion=ass,
            )
        )
    allowed = _allowed_from_candidates(candidates)
    has_span = any(c.span_ids for c in citations)
    category = AnswerCategory.source_grounded if has_span else AnswerCategory.unresolved
    envelope = AnswerEnvelope(
        answer_id="ans_" + uuid4().hex,
        question=spec.question,
        query_type=spec.protocol_hint,
        answer_category=category,
        answer_text="",
        citations=citations,
        recommendations=build_recommendations(project_conn, spec, candidates),
        cited_work_ids=[c.work_id for c in citations],
        cited_span_ids=[s for c in citations for s in c.span_ids],
        insufficient_evidence=False,
        retrieved_item_ids=list(allowed.retrieved_item_ids),
        warnings=warnings,
        mode=AnswerMode.RETRIEVAL_ONLY,
        llm_provenance=llm_provenance,
    )
    return envelope, allowed


def _local_fallback_route(cfg: GlobalConfig) -> Route | None:
    """First usable local, non-no-LLM profile authorized for answer generation
    (mirrors ``extraction.runner._local_fallback_route``). Used by the §7 access gate
    to keep private fragments on the machine."""
    route = cfg.llm.routes.get(TASK_TYPE)
    candidate_ids: list[str] = []
    if route is not None:
        candidate_ids = [route.fallback_profile, route.preferred_profile]
    candidate_ids += list(cfg.llm.profiles)
    seen: set[str] = set()
    for profile_id in candidate_ids:
        if profile_id is None or profile_id in seen:
            continue
        seen.add(profile_id)
        profile = cfg.llm.profiles.get(profile_id)
        if profile is None:
            continue
        if is_no_llm_profile(profile) or not is_local_profile(profile):
            continue
        if profile.allowed_tasks is not None and TASK_TYPE not in profile.allowed_tasks:
            continue
        if not is_profile_available(profile):
            continue
        return Route(
            profile_id=profile_id,
            provider=profile.provider,
            access_mode=profile.access_mode,
            external_full_text=False,
            fallback=None,
        )
    return None


# --- the single answer call (08 §5 step 9) ---------------------------------

def generate(
    candidates: list[RankedCandidate],
    spec: QuerySpec,
    cfg: GlobalConfig,
    *,
    project_conn: sqlite3.Connection,
    no_llm: bool = False,
    run_id: str | None = None,
    backend: LLMBackend | None = None,
    capabilities: LlmCapabilities | None = None,
    cache_root=None,
    warnings: list[str] | None = None,
) -> tuple[AnswerEnvelope, AllowedSet, ComposeTrace | None]:
    """08 §5 step 9 — produce the (pre-guard) answer envelope (single LLM call or
    degrade). Returns ``(envelope, allowed_set, compose_trace)`` so the harness can run
    ``guard.enforce`` against the exact shown-candidate allowlist and assemble the
    AnswerTrace (00 §4.1). ``compose_trace`` is always non-``None`` here: it carries the
    shown-evidence list + prompt hash/version (LLM path and failed-dispatch degrade) and
    ``degrade_reasons`` naming which of the six degrade sites fired.

    Resolves the ``answer_generation`` route, applies the private-fragment access gate
    (plan §7) and the fail-closed pricing gate (plan §8), builds the numbered-evidence
    prompt, makes **one** LLM call, parses strict JSON with one repair retry, then
    materializes citations. Writes one ``llm_usage_events`` row per call
    (``external_full_text=false``, ``source_access_class`` = max-restrictive over
    feeding spans, ``prompt_version``). Degrades to ``retrieval_only`` on ``no_llm`` /
    no profile / policy / budget block / second JSON failure (decisions 58/38).
    """
    if capabilities is None:
        capabilities = load_llm_capabilities()
    warnings = list(warnings or [])

    source_access_class = str(
        AccessClass.most_restrictive(*[c.item.access_class for c in candidates])
    )
    # Default-deny per the shareability lattice (vocab.is_shareable, decisions
    # 76/30/60): a fragment must be gated from an external profile unless its
    # access_class is on the shareable allowlist (open_access / metadata_only). This
    # covers user_supplied_private AND the other non-shareable classes (unknown /
    # licensed_future) so a restricted-but-not-"private" fragment cannot leak to an
    # external API when policy forbids it.
    has_restricted = any(not is_shareable(c.item.access_class) for c in candidates)

    route = resolve_route(TASK_TYPE, source_access_class, cfg)

    # No-LLM / no usable profile -> honest retrieval_only degrade (no call). No prompt
    # is built, and no_llm appends no warning, so degrade_reasons stays empty (the mode
    # is a deliberate choice, not a degrade *reason*).
    if no_llm or isinstance(route, NoLlmRoute):
        env, deg_allowed = _retrieval_only_envelope(candidates, spec, project_conn, warnings=warnings)
        return env, deg_allowed, ComposeTrace()

    profile = resolve_profile(route.profile_id, cfg)
    is_external = not is_local_profile(profile)

    # (§7) restricted-fragment access gate: a non-shareable fragment must not reach an
    # external API when policy forbids it — route local, else degrade (never send it
    # out). Keyed on the default-deny lattice, not just user_supplied_private.
    if is_external and has_restricted and not cfg.content_policy.external_llm_for_answer_generation:
        local = _local_fallback_route(cfg)
        if local is None:
            env, deg_allowed = _retrieval_only_envelope(
                candidates, spec, project_conn,
                warnings=warnings + ["private_content_local_only"],
            )
            return env, deg_allowed, ComposeTrace(degrade_reasons=["private_content_local_only"])
        route = local
        profile = resolve_profile(route.profile_id, cfg)
        is_external = False

    model = profile.model
    cap = capabilities.models.get(model) if model else None

    # (§8) fail-closed pricing gate for a USD-limited spend (phase_0 enforcer).
    if cfg.budget.usd_limit is not None and cap is not None:
        from ..budget import check_pricing_allowed

        try:
            check_pricing_allowed(model, cap, cfg.budget)
        except ConfigError:
            env, deg_allowed = _retrieval_only_envelope(
                candidates, spec, project_conn, warnings=warnings + ["budget_exceeded"]
            )
            return env, deg_allowed, ComposeTrace(degrade_reasons=["budget_exceeded"])

    context_window = cap.context_window_tokens if cap is not None else None
    prompt, evidence_index, allowed, shown_report = build_prompt(
        candidates,
        spec,
        max_evidence_tokens=cfg.answer.max_evidence_tokens,
        context_window_tokens=context_window,
        max_fragment_chars=cfg.answer.max_fragment_chars,
        prompt_overhead_tokens=cfg.answer.prompt_overhead_tokens,
        reserved_output_tokens=cfg.answer.reserved_output_tokens,
    )
    if not evidence_index:
        # The whole evidence set fell outside the budget -> nothing citable (in practice
        # only when candidates is empty; the first candidate is always shown otherwise).
        env, deg_allowed = _retrieval_only_envelope(
            candidates, spec, project_conn, warnings=warnings + ["budget_exceeded"]
        )
        return env, deg_allowed, ComposeTrace(degrade_reasons=["budget_exceeded"])

    # T3: record the shown-evidence list + the prompt's sha256 (hash only, never the
    # body) — both survive onto the trace on the LLM path AND the failed-dispatch degrade.
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    provenance = {
        "task_type": TASK_TYPE,
        "profile": route.profile_id,
        "model": model,
        "access_mode": route.access_mode,
        "source_text_left_machine": False,  # bounded fragments only (plan §7)
        "prompt_version": PROMPT_VERSION,
    }

    client = backend or _BACKEND_OVERRIDE or default_backend(provider=route.provider, model=model)

    # Dispatch through the executor (Track 1): the single LLM call + one repair
    # re-prompt + transport handling live in ``dispatch``; a provider/transport
    # error (e.g. a real key + a failing transport) returns a typed non-success
    # result instead of raising, so ``ask`` degrades to retrieval_only rather than
    # tracebacking. We KEEP the §7 fragment gate above (route may already be local).
    disp = run_executor.dispatch(
        client,
        prompt,
        "Return the strict JSON answer object now.",
        model=model,
        provider=route.provider,
        access_mode=route.access_mode,
        profile_id=route.profile_id,
        external_full_text=False,  # bounded fragments only (plan §7)
        temperature=0.0,
        parse=_parse_json_pair,
        repair_prompt=_answer_repair_prompt,
    )

    # One usage row per answer CALL (we reached dispatch); records status + full
    # sha256 hashes (no bodies). Pre-dispatch blocks above return earlier → 0 rows.
    run_executor.log_llm_usage(
        project_conn,
        task_type=TASK_TYPE,
        result=disp,
        source_access_class=source_access_class,
        run_id=run_id,
        cap=cap,
        external_full_text=False,
    )

    if disp.status != "success":
        # Provider/transport error or unrepairable JSON -> degrade, never fabricate
        # prose (risk table). Warning text derived from the executor error code.
        warn = _warn_for(disp.error)
        env, deg_allowed = _retrieval_only_envelope(
            candidates, spec, project_conn,
            warnings=warnings + [warn], llm_provenance=provenance,
        )
        # The delivered retrieval-only evidence has its own complete allowlist.
        # The trace still records only the prompt actually sent to the failed model.
        return env, deg_allowed, ComposeTrace(
            shown_evidence=shown_report.evidence,
            cut_index=shown_report.cut_index,
            prompt_version=PROMPT_VERSION,
            prompt_sha256=prompt_sha256,
            degrade_reasons=[warn],
        )

    parsed = disp.parsed

    query_type = _coerce_query_type(parsed.get("query_type"))
    answer_category = _coerce_answer_category(parsed.get("answer_category"))
    answer_text = parsed.get("answer_text") or ""
    insufficient = bool(parsed.get("insufficient_evidence", False))
    raw_markers = parsed.get("cited_markers") or []
    if not isinstance(raw_markers, list):
        raw_markers = []

    citations, cite_warnings = materialize_citations(raw_markers, evidence_index, project_conn)
    warnings += cite_warnings

    envelope = AnswerEnvelope(
        answer_id="ans_" + uuid4().hex,
        question=spec.question,
        query_type=query_type,
        answer_category=answer_category,
        answer_text=answer_text,
        citations=citations,
        recommendations=build_recommendations(project_conn, spec, candidates),
        cited_work_ids=[c.work_id for c in citations],
        cited_span_ids=[s for c in citations for s in c.span_ids],
        insufficient_evidence=insufficient,
        retrieved_item_ids=list(allowed.retrieved_item_ids),
        warnings=warnings,
        mode=spec.mode,
        llm_provenance=provenance,
    )
    return envelope, allowed, ComposeTrace(
        shown_evidence=shown_report.evidence,
        cut_index=shown_report.cut_index,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=prompt_sha256,
        degrade_reasons=[],
    )
