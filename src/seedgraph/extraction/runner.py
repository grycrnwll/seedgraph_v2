"""Default-note extraction orchestration (plan §5/§8/§10 step 8).

``extract_note`` is the per-work pipeline:

    staleness/idempotency -> resolve_source_access_class (inlined cache-walk)
    -> config + context-window validation -> router + content-access gate
    -> budget gate (BudgetState) -> LLM call -> validate (one repair re-prompt)
    -> normalize -> anchor each `found` quote via the phase_3 `ensure_span`
       creator over the db/adapter.raw_conn bridge (D7) -> write run/note/claims
       and claim_spans junction rows in ONE transaction -> FTS reindex
    -> usage log -> stamp provenance + access_class.

Every terminal path (``extraction_failed`` / ``skipped_no_llm`` /
``skipped_policy`` / ``skipped_oversize`` / ``skipped_budget``) writes exactly one
``extraction_runs`` row and NO ``structured_notes`` row (plan §1/§8, D4). Spans are
created ONLY through phase_3's ``ensure_span`` — this phase implements no fallback
creator (plan §4.6). ``access_class`` is stamped most-restrictive on the run, the
note, every claim, AND every span created this run (decision 76, plan §7).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .. import cache_access
from ..acquisition.bridge import resolve_work_markdown
from ..budget import check_pricing_allowed
from ..config.loader import load_llm_capabilities
from ..config.models import GlobalConfig
from ..db.adapter import raw_conn  # D7 bridge: Session -> live-txn DBAPI conn
from ..db.fts import ensure_fts_tables, reindex_claim_fts, reindex_note_fts
from ..errors import ConfigError
from ..ids import new_id
from ..llm import executor
from ..llm.backend import default_backend
from ..llm.profiles import is_local_profile, is_no_llm_profile, is_profile_available, resolve_profile
from ..llm.routing import (NoLlmRoute, Route, local_fallback_route, resolve_route,
                           assert_external_content_allowed)
from ..llm.tokens import estimate_tokens
from ..llm.usage import UsageEvent, log_usage
from ..sections.store import load_sections
from ..spans.store import ensure_span
from ..vocab import AccessClass, StatusValue
from .normalize import normalize_note  # re-exported (plan §6)
from .prompt import build_prompt, build_repair_prompt
from .schema import PROMPT_VERSION, SCHEMA_ID, SCHEMA_VERSION
from .validator import validate_note  # re-exported (plan §6)

if TYPE_CHECKING:
    from sqlmodel import Session

    from ..config.models import LlmCapabilities, ProjectConfig
    from ..db.models_project import StructuredNote
    from ..llm.backend import LLMBackend

__all__ = [
    "BudgetState",
    "ExtractionResult",
    "extract_note",
    "validate_note",
    "normalize_note",
    "resolve_source_access_class",
    "current_note",
]

TASK_CLASS = "note_extraction"

_DEFAULT_PRIVATE = AccessClass.user_supplied_private.value

# Offline test seam (analogous to FakeMarkerBackend injection): when set, the CLI
# path — which cannot pass ``backend=`` through Typer — dispatches through this
# backend instead of building the lazy stub real client. Tests monkeypatch it;
# production leaves it ``None`` so the stub real backend's actionable error fires.
_BACKEND_OVERRIDE = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BudgetState:
    """Running cost accumulator threaded across the works loop (plan §8, should_fix #3).

    Owns the cumulative USD spend and the stop latch. The CLI creates one and
    passes it into every ``extract_note`` call so per-run + cumulative limits and
    the stop-on-exceeded gate are enforced across the whole corpus rather than
    per-work. (Phase-private: phase_0 ``budget.check_pricing_allowed`` is the
    fail-closed pricing enforcer this state defers to; it is NOT BudgetState.)

    Thread-safe (Build C chunk 6 / D6): spend accumulation and the stop latch go
    through :meth:`add_spend` / :meth:`is_stopped`, which serialize on an internal
    ``threading.Lock``, so one BudgetState can be shared across the ``extract
    notes --concurrency`` worker pool with no lost updates and a race-free latch.
    """

    spent_usd: float = 0.0
    runs: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    # Stage C: prior spend this calendar month, read once from llm_usage_events
    # (``monthly_spend(conn, this_month)``) and seeded by the CLI so the monthly
    # soft limit is enforced against ``prior_month_spent_usd + spent_usd`` (DB +
    # this run), not this run alone. Defaults to 0.0 (behavior unchanged).
    prior_month_spent_usd: float = 0.0
    # Internal lock — excluded from init/repr/compare so the dataclass surface
    # (construction, equality on the accounting fields) is unchanged.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def add_spend(
        self, cost: float | None, budget, *, per_run_cost: float | None = None
    ) -> None:
        """Accumulate ``cost`` and trip the stop latch, atomically under the lock.

        ``budget`` is the project :class:`~seedgraph.config.models.LlmBudget`.
        ``per_run_cost`` is what ``per_run_soft_limit_usd`` is compared against —
        it defaults to ``cost`` (the whole-note runner's per-work cost); the
        chunked runner passes its running per-PAPER total (Σ chunks) instead.
        Semantics are byte-identical to the pre-lock accounting for a single
        caller; the lock only removes the read-modify-write race under
        ``--concurrency`` (D6).
        """
        inc = cost or 0.0
        check = inc if per_run_cost is None else per_run_cost
        with self._lock:
            self.spent_usd += inc
            self.runs += 1
            if not budget.stop_on_budget_exceeded:
                return
            per_run = budget.per_run_soft_limit_usd
            monthly = budget.monthly_soft_limit_usd
            if per_run is not None and check >= per_run:
                self.stopped = True
                self.stop_reason = "per_run_soft_limit_usd exceeded"
            # Monthly soft limit (Stage C): prior DB spend this month + accrual.
            if monthly is not None and (self.prior_month_spent_usd + self.spent_usd) >= monthly:
                self.stopped = True
                self.stop_reason = "monthly_soft_limit_usd exceeded"

    def is_stopped(self) -> bool:
        """Race-free read of the stop latch (pairs with :meth:`add_spend`)."""
        with self._lock:
            return self.stopped


@dataclass
class ExtractionResult:
    """Outcome of one ``extract_note`` call (the CLI per-work summary line)."""

    work_id: str
    run_status: str
    extraction_run_id: str | None = None
    note_id: str | None = None
    claim_count: int = 0
    span_count: int = 0
    estimated_cost: float | None = None
    message: str | None = None
    errors: list[str] = field(default_factory=list)


# --- run-row writer (every terminal path writes exactly one) ----------------

_RUN_COLUMNS = (
    "extraction_run_id, work_id, markdown_id, markdown_hash, schema_id, lens_id, "
    "schema_version, prompt_version, model_name, provider, access_mode, temperature, "
    "access_class, external_full_text, run_status, input_tokens, output_tokens, "
    "estimated_cost, run_id, created_at"
)
_RUN_PLACEHOLDERS = ", ".join(["?"] * 20)


def _insert_run(
    conn,
    *,
    run_id: str,
    work_id: str,
    markdown_id: str,
    markdown_hash: str,
    schema_id: str,
    run_status: str,
    access_class: str,
    model_name: str | None = None,
    provider: str | None = None,
    access_mode: str | None = None,
    temperature: float | None = None,
    external_full_text: int = 0,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    estimated_cost: float | None = None,
    build_run_id: str | None = None,
) -> None:
    conn.execute(
        f"INSERT INTO extraction_runs ({_RUN_COLUMNS}) VALUES ({_RUN_PLACEHOLDERS})",
        (
            run_id,
            work_id,
            markdown_id,
            markdown_hash,
            schema_id,
            None,  # lens_id (phase_6)
            SCHEMA_VERSION,
            PROMPT_VERSION,
            model_name,
            provider,
            access_mode,
            temperature,
            access_class,
            external_full_text,
            run_status,
            input_tokens,
            output_tokens,
            estimated_cost,
            build_run_id,
            _now(),
        ),
    )


def _terminal_run(
    session,
    *,
    work_id,
    markdown_id,
    markdown_hash,
    schema_id,
    run_status,
    access_class,
    message,
    build_run_id=None,
    model_name=None,
    provider=None,
    access_mode=None,
    temperature=None,
    external_full_text=0,
    input_tokens=None,
    output_tokens=None,
    estimated_cost=None,
    errors=None,
    usage: bool = False,
) -> ExtractionResult:
    """Write one ``extraction_runs`` row (NO note) and return the result."""
    conn = raw_conn(session)
    run_id = new_id("extr")
    _insert_run(
        conn,
        run_id=run_id,
        work_id=work_id,
        markdown_id=markdown_id,
        markdown_hash=markdown_hash,
        schema_id=schema_id,
        run_status=run_status,
        access_class=access_class,
        model_name=model_name,
        provider=provider,
        access_mode=access_mode,
        temperature=temperature,
        external_full_text=external_full_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost=estimated_cost,
        build_run_id=build_run_id,
    )
    session.commit()
    if usage:
        log_usage(
            raw_conn(session),
            UsageEvent(
                task_type=TASK_CLASS,
                provider=provider,
                model=model_name,
                access_mode=access_mode,
                run_id=build_run_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost=estimated_cost,
                source_access_class=access_class,
                external_full_text=bool(external_full_text),
            ),
        )
    return ExtractionResult(
        work_id=work_id,
        run_status=run_status,
        extraction_run_id=run_id,
        estimated_cost=estimated_cost,
        message=message,
        errors=errors or [],
    )


# Consolidated into llm/cost.py (Track 1). Re-exported under the historical name
# so chunked_runner's ``from .runner import _estimate_cost`` keeps working.
from ..llm.cost import estimate_cost as _estimate_cost  # noqa: E402


def _local_fallback_route(config: "ProjectConfig") -> Route | None:
    """Content-gate local fallback for note extraction.

    Thin wrapper over the consolidated ``routing.local_fallback_route`` (Track 1)
    so the chunked runner's ``from .runner import _local_fallback_route`` stays
    stable. Returns a local :class:`Route` or ``None`` (caller → skipped_policy)."""
    return local_fallback_route(config, TASK_CLASS)


def _runtime_fallback(config, route, access_class, injected_backend):
    """Build the one-hop runtime-fallback closure for ``dispatch`` (plan §2).

    When the preferred dispatch fails with ``provider_unavailable`` /
    ``invalid_response`` (Ollama down / model missing), re-route ONCE to
    ``route.fallback_profile`` — re-applying the content-access gate so restricted
    full text never leaks to a hosted fallback. In tests an injected backend is
    reused for the hop (no network); in production a secret-aware backend is built.
    Returns ``None`` (no hop) when there is no fallback profile, it is unavailable,
    or the content gate forbids it."""
    fb_id = route.fallback
    used = {"done": False}

    def _next_hop(error):
        if used["done"] or not fb_id:
            return None
        try:
            fb_profile = resolve_profile(fb_id, config)
        except ConfigError:
            return None
        if is_no_llm_profile(fb_profile):
            return None
        fb_external = not is_local_profile(fb_profile)
        rt = config.llm.routes.get(TASK_CLASS)
        requires_src = rt.requires_source_text if rt is not None else True
        # Content-access gate (re-applied on the hop): restricted full text must
        # not leave the machine for an external fallback when policy forbids it.
        if fb_external and requires_src:
            try:
                assert_external_content_allowed(access_class, config, task_type=TASK_CLASS,
                                                profile_id=fb_id)
            except ConfigError:
                return None
        if not is_profile_available(fb_profile):
            return None
        used["done"] = True
        fb_backend = injected_backend or executor.build_backend(fb_profile, config)
        return executor.Hop(
            backend=fb_backend,
            model=fb_profile.model,
            provider=fb_profile.provider,
            access_mode=fb_profile.access_mode,
            profile_id=fb_id,
            external_full_text=bool(fb_external and requires_src),
        )

    return _next_hop


# Visible splice marker left where a references section was removed from the
# PROMPT text (Build C chunk 3, design D5) — makes the trim auditable in any
# recorded prompt.
REFERENCES_OMITTED_MARKER = "[references omitted]"


def _strip_references(md_text: str, sections) -> str:
    """Return ``md_text`` with ``section_kind='references'`` char ranges spliced out.

    PROMPT-ONLY trimming (design D5): the returned text is embedded in the LLM
    prompt in place of the full markdown; the anchoring source is untouched —
    ``ensure_span`` still anchors quotes against the stored FULL markdown by
    ``markdown_id``. Each removed range is replaced by a visible
    :data:`REFERENCES_OMITTED_MARKER`. Fail-open: the full text is returned
    unchanged when there are no ``sections`` rows, no references-kind ranges, or
    when trimming would leave the prompt empty (an all-references document —
    e.g. an annotated bibliography — extracts as before).
    """
    if not sections:
        return md_text
    ranges = sorted(
        (int(s.start_char), int(s.end_char))
        for s in sections
        if getattr(s, "section_kind", None) == "references"
        and int(s.end_char) > int(s.start_char)
    )
    if not ranges:
        return md_text
    parts: list[str] = []
    cursor = 0
    for start, end in ranges:
        start = max(start, cursor)  # sections tile without overlap; defensive clamp
        if start >= end:
            continue
        parts.append(md_text[cursor:start])
        parts.append(f"{REFERENCES_OMITTED_MARKER}\n")
        cursor = max(cursor, end)
    parts.append(md_text[cursor:])
    trimmed = "".join(parts)
    if not trimmed.replace(REFERENCES_OMITTED_MARKER, "").strip():
        return md_text  # trimming would empty the prompt -> fail open
    return trimmed


def resolve_source_access_class(cache_session, markdown_id: str) -> str:
    """Resolve the source ``access_class`` for ``markdown_id`` (inlined cache-walk).

    Walks ``markdown_id -> markdown_documents.source_file_id ->
    source_files.access_class`` in ``cache.db``. Single contributing source ⇒ that
    class; unresolved/missing ⇒ ``'user_supplied_private'`` (fail-closed default-
    deny, decisions 30/60/76). Inlined here (no separate ``extraction/access.py``).
    """
    row = cache_session.execute(
        "SELECT s.access_class AS access_class "
        "FROM markdown_documents m "
        "JOIN source_files s ON s.source_file_id = m.source_file_id "
        "WHERE m.markdown_id = ?",
        (markdown_id,),
    ).fetchone()
    if row is None:
        return _DEFAULT_PRIVATE
    try:
        access_class = row["access_class"]
    except (TypeError, IndexError):
        access_class = row[0]
    # Single source: that class, but fail-closed-normalize an unknown/NULL value.
    return str(AccessClass.most_restrictive(access_class))


def current_note(
    project_session: "Session",
    work_id: str,
    schema_id: str,
    markdown_hash: str,
) -> "StructuredNote | None":
    """Return the current ``structured_notes`` row for ``(work_id, schema_id)`` or None.

    Current = latest note whose ``markdown_hash == markdown_hash`` AND
    ``schema_version == SCHEMA_VERSION`` AND ``prompt_version == PROMPT_VERSION``
    (append-only, pull-based staleness — plan §7). A changed markdown hash or a
    schema/prompt bump returns None (re-extract); model identity is NOT an
    invalidation key by default.
    """
    from sqlmodel import select

    from ..db.models_project import StructuredNote

    stmt = (
        select(StructuredNote)
        .where(StructuredNote.work_id == work_id)
        .where(StructuredNote.schema_id == schema_id)
        .where(StructuredNote.markdown_hash == markdown_hash)
        .where(StructuredNote.schema_version == SCHEMA_VERSION)
        .where(StructuredNote.prompt_version == PROMPT_VERSION)
        .order_by(StructuredNote.created_at.desc())
    )
    return project_session.exec(stmt).first()


_UNCHECKED = object()


class NoteConflict(ValueError):
    """Activation would replace an interpretation created after preparation."""


def latest_note_id(conn, work_id: str, schema_id: str = SCHEMA_ID) -> str | None:
    row = conn.execute(
        "SELECT note_id FROM structured_notes WHERE work_id=? AND schema_id=? "
        "ORDER BY created_at DESC, rowid DESC LIMIT 1", (work_id, schema_id)
    ).fetchone()
    return row[0] if row else None


def save_normalized_note(
    project_session, cache_session, *, work_id, markdown_id, markdown_hash,
    drafts, note_text, archetype, raw_note_json, resolved_class,
    schema_id=SCHEMA_ID, model_name=None, provider=None, access_mode=None,
    temperature=None, external_full_text=0, input_tokens=None, output_tokens=None,
    estimated_cost=None, build_run_id=None, cache_root=None, extraction_run_id=None,
    extraction_mode="whole", chunk_count=None, expected_current=_UNCHECKED,
    before_activation=None,
) -> ExtractionResult:
    """One shared atomic activation path for API and host-generated notes.

    Callers validate and normalize first. This boundary anchors all found claims
    and retains old interpretations. Agent callers supply expected_current for
    a compare-and-activate under SQLite's writer lock.
    """
    try:
        conn = raw_conn(project_session)
        run_id = extraction_run_id or new_id("extr")
        if expected_current is not _UNCHECKED:
            # Serialize compare-and-activate with other writers, including API runs.
            conn.execute("BEGIN IMMEDIATE")
            if latest_note_id(conn, work_id, schema_id) != expected_current:
                project_session.rollback()
                raise NoteConflict("current note changed after preparation; prepare again")
        if before_activation is not None:
            before_activation()
        note_id = new_id("note")
        now = _now()

        _insert_run(
            conn,
            run_id=run_id,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="success",
            access_class=resolved_class,
            model_name=model_name,
            provider=provider,
            access_mode=access_mode,
            temperature=temperature,
            external_full_text=external_full_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            build_run_id=build_run_id,
        )
        conn.execute(
            "UPDATE extraction_runs SET extraction_mode=?, chunk_count=? WHERE extraction_run_id=?",
            (extraction_mode, chunk_count, run_id),
        )
        conn.execute(
            "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
            "markdown_hash, schema_id, schema_version, prompt_version, archetype, access_class, "
            "raw_note_json, note_text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                note_id,
                run_id,
                work_id,
                markdown_id,
                markdown_hash,
                schema_id,
                SCHEMA_VERSION,
                PROMPT_VERSION,
                archetype,
                resolved_class,
                raw_note_json,
                note_text,
                now,
            ),
        )

        ensure_fts_tables(project_session)

        span_count = 0
        for draft in drafts:
            claim_id = new_id("claim")
            status = draft.status
            inferred_explanation = draft.inferred_explanation
            span_ids = []
            if status == StatusValue.found.value:
                quotes = getattr(draft, "exact_quotes", None) or (
                    [draft.exact_quote] if draft.exact_quote else []
                )
                for quote in quotes:
                    span_id = ensure_span(
                        conn, cache_session, cache_root, markdown_id=markdown_id,
                        work_id=work_id, exact_quote=quote, access_class=resolved_class,
                        span_kind="manual",
                    )
                    if span_id is not None and span_id not in span_ids:
                        span_ids.append(span_id)
                if not span_ids:
                    status = StatusValue.ambiguous.value
                    note_msg = "quote could not be anchored verbatim; downgraded to ambiguous"
                    inferred_explanation = (
                        f"{inferred_explanation} | {note_msg}" if inferred_explanation else note_msg
                    )

            conn.execute(
                "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
                "work_id, claim_type, claim_subtype, field_key, normalized_label, claim_text, "
                "status, epistemic_type, assertion_status, inferred_explanation, confidence, "
                "access_class, created_at, source_chunk_index, source_extraction_run_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    note_id,
                    run_id,
                    work_id,
                    draft.claim_type,
                    draft.claim_subtype,
                    draft.field_key,
                    draft.normalized_label,
                    draft.claim_text,
                    status,
                    draft.epistemic_type,
                    draft.assertion_status,
                    inferred_explanation,
                    draft.confidence,
                    resolved_class,
                    now,
                    getattr(draft, "source_chunk_index", None),
                    getattr(draft, "source_extraction_run_id", None),
                ),
            )
            for rank, span_id in enumerate(span_ids):
                conn.execute(
                    "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (claim_id, span_id, rank, now),
                )
                span_count += 1

        reindex_claim_fts(project_session, note_id)
        reindex_note_fts(project_session, note_id)

        project_session.commit()

        return ExtractionResult(
            work_id=work_id, run_status="success", extraction_run_id=run_id,
            note_id=note_id, claim_count=len(drafts), span_count=span_count,
            estimated_cost=estimated_cost,
            message=f"note for {work_id}: {len(drafts)} claims, {span_count} spans",
        )
    except BaseException:
        project_session.rollback()
        raise



def extract_note(
    project_session: "Session",
    cache_session,
    *,
    work_id: str,
    schema_id: str = SCHEMA_ID,
    profile_id: str | None = None,
    force: bool = False,
    confirm_external: bool = False,
    dry_run: bool = False,
    build_run_id: str | None = None,
    budget_state: "BudgetState | None" = None,
    backend: "LLMBackend | None" = None,
    config: "ProjectConfig | None" = None,
    cache_root: "Path | str | None" = None,
    capabilities: "LlmCapabilities | None" = None,
) -> ExtractionResult:
    """Extract one default note for ``work_id`` (the plan §6 orchestration entry).

    ``cache_session`` is the read-only cache connection from
    ``cache_access.open_cache_ro`` (raw sqlite3). ``config`` is the routing
    :class:`~seedgraph.config.models.ProjectConfig` (defaults to a keyless
    :class:`GlobalConfig`); ``capabilities`` defaults to the bundled D9 snapshot;
    ``backend`` is injected by tests (a real client is never built in pytest).

    Idempotent: a current note for ``(work_id, schema_id, markdown_hash,
    schema_version, prompt_version)`` short-circuits unless ``force``. Performs
    config + context-window gating, content-access gating, and budget gating via
    ``budget_state``. On success writes the run/note/claims/claim_spans atomically,
    reindexes ``claim_fts``/``note_fts``, logs usage, and stamps ``access_class``.
    ``dry_run`` estimates cost and writes nothing.
    """
    if config is None:
        config = GlobalConfig()
    if capabilities is None:
        capabilities = load_llm_capabilities()
    if budget_state is None:
        budget_state = BudgetState()

    # 1. Resolve work -> current markdown (none -> skip, no run row).
    resolved = resolve_work_markdown(project_session, work_id=work_id)
    if resolved is None:
        return ExtractionResult(
            work_id=work_id,
            run_status="skipped_no_markdown",
            message=f"work {work_id} has no resolvable markdown (skipped)",
        )
    markdown_id, markdown_hash = resolved

    md = cache_access.read_markdown(cache_session, cache_root, markdown_id)
    if md is None:
        return ExtractionResult(
            work_id=work_id,
            run_status="skipped_no_markdown",
            message=f"markdown {markdown_id} unresolvable in cache (skipped)",
        )

    resolved_class = resolve_source_access_class(cache_session, markdown_id)

    # 2. Idempotency / staleness (append-only, pull-based — plan §7).
    if not force:
        existing = current_note(project_session, work_id, schema_id, markdown_hash)
        if existing is not None:
            return ExtractionResult(
                work_id=work_id,
                run_status="skipped_idempotent",
                note_id=existing.note_id,
                message=f"current note already exists for {work_id} (use --force)",
            )

    # 3. Budget stop latch (set by a prior work in the loop, possibly on another
    #    --concurrency worker thread) -> skipped_budget.
    if budget_state.is_stopped():
        return _terminal_run(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="skipped_budget",
            access_class=resolved_class,
            message=(
                f"budget stop ({budget_state.stop_reason}); {work_id} not processed "
                f"(skipped_budget)"
            ),
            build_run_id=build_run_id,
        )

    # 4. Build the prompt + resolve the route (with --profile / --confirm-external).
    #    References-tail exclusion (D5): trim references-kind section ranges from
    #    the PROMPT text only, fail-open. The oversize gate below estimates the
    #    built prompt, so gate and dispatch agree automatically; ensure_span still
    #    anchors against the stored FULL markdown (never the trimmed text).
    prompt_text = _strip_references(
        md.text, load_sections(raw_conn(project_session), markdown_id)
    )
    system_prompt, user_prompt = build_prompt(prompt_text, schema_id=schema_id)

    effective_config = config
    if confirm_external or profile_id is not None:
        effective_config = config.model_copy(deep=True)
        if confirm_external:
            effective_config.content_policy.external_llm_for_private_full_text = True
        if profile_id is not None:
            route_cfg = effective_config.llm.routes.get(TASK_CLASS)
            if route_cfg is not None:
                route_cfg.preferred_profile = profile_id

    try:
        route = resolve_route(TASK_CLASS, resolved_class, effective_config)
    except ConfigError:
        # Content-access gate blocked the external route: try a local fallback,
        # else refuse this work (skipped_policy) — source text never leaves.
        route = _local_fallback_route(effective_config)
        if route is None:
            return _terminal_run(
                project_session,
                work_id=work_id,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                schema_id=schema_id,
                run_status="skipped_policy",
                access_class=resolved_class,
                message=(
                    f"private source ({resolved_class}) cannot leave the machine for "
                    f"an external profile and no local profile is available; pass "
                    f"--confirm-external or configure a local profile (skipped_policy)"
                ),
                build_run_id=build_run_id,
            )

    # 5. No-LLM honest degradation (no rule-based substitute).
    if isinstance(route, NoLlmRoute):
        return _terminal_run(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="skipped_no_llm",
            access_class=resolved_class,
            message=f"no usable LLM backend for note extraction on {work_id} (skipped_no_llm)",
            build_run_id=build_run_id,
        )

    profile = resolve_profile(route.profile_id, effective_config)
    model = profile.model
    cap = capabilities.models.get(model) if model else None
    external_full_text = 1 if route.external_full_text else 0

    # 6. Context-window gate (D4/D9): estimate prompt tokens vs context window.
    #    FAIL-CLOSED on an unknown model: if the routed model is absent from the
    #    llm_capabilities.yaml snapshot its window cannot be verified, so we refuse
    #    to dispatch rather than risk silent truncation — consistent with the
    #    fail-closed pricing posture (plan §8: "prevents silent truncation; over-long
    #    papers handed to phase_4b, never silently dropped").
    prompt_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    context_window = cap.context_window_tokens if cap is not None else None
    if context_window is None or prompt_tokens > context_window:
        if context_window is None:
            message = (
                f"model {model} is absent from the llm_capabilities.yaml snapshot, so its "
                f"context window cannot be verified; refusing to dispatch paper {work_id} "
                f"(~{prompt_tokens} tok) to avoid silent truncation — add {model} to "
                f"llm_capabilities.yaml or route to phase_4b chunked/section extraction "
                f"(skipped_oversize)"
            )
        else:
            message = (
                f"paper {work_id} (~{prompt_tokens} tok) exceeds model {model} context "
                f"window ({context_window} tok); route to phase_4b chunked/section "
                f"extraction or use a larger-context profile (skipped_oversize)"
            )
        return _terminal_run(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="skipped_oversize",
            access_class=resolved_class,
            model_name=model,
            provider=route.provider,
            access_mode=route.access_mode,
            # The context-window gate returns BEFORE any dispatch, so NO source
            # text leaves the machine here — stamp 0 even for an external route
            # (plan §8: external_full_text=1 only when source text actually leaves
            # the machine; keeps the doc-09 content-access audit truthful).
            external_full_text=0,
            message=message,
            build_run_id=build_run_id,
        )

    # 7. Budget pre-flight: fail-closed unverified pricing only when a hard USD
    #    limit is configured (phase_0 enforcer; soft limits handled post-dispatch).
    if cap is not None and config.budget.usd_limit is not None:
        check_pricing_allowed(model, cap, config.budget)

    # 8. dry_run: estimate + report, write nothing.
    if dry_run:
        projected = _estimate_cost(cap, prompt_tokens, prompt_tokens)
        return ExtractionResult(
            work_id=work_id,
            run_status="dry_run",
            estimated_cost=projected,
            message=f"dry-run: {work_id} ~{prompt_tokens} prompt tok, ~${projected:.6f}",
        )

    # 9. Dispatch through the executor (Track 1): the single call + one repair
    #    re-prompt + transport retry + the runtime fallback chain all live in
    #    ``dispatch``. ``parse=validate_note`` keeps JSON repair task-level; a
    #    provider/transport error returns a typed non-success result (no crash).
    #    A down local provider / missing model on the preferred profile re-routes
    #    once to ``route.fallback_profile`` (content gate re-applied) — this makes
    #    local-first note extraction fall back to Anthropic when Ollama is down.
    if backend is not None:
        client = backend
    elif _BACKEND_OVERRIDE is not None:
        client = _BACKEND_OVERRIDE
    else:
        client = executor.build_backend(profile, effective_config)
    temperature = profile_temperature(profile)
    # Output budget: a claim-rich note overflows the executor default (4096) and
    # truncates mid-JSON -> schema_validation_failed. Size from the route's
    # configured max_output_tokens, else the model capability (capped at 16k --
    # ample for ~80+ claims without inviting runaway cost). Mirrors metadata.py.
    max_out = 8192
    try:
        route_cfg = effective_config.llm.routes.get(TASK_CLASS)
        if route_cfg is not None and route_cfg.max_output_tokens:
            max_out = int(route_cfg.max_output_tokens)
        elif cap is not None and cap.max_output_tokens:
            max_out = min(int(cap.max_output_tokens), 16384)
    except Exception:  # noqa: BLE001 - never let a config quirk break extraction
        max_out = 8192
    disp = executor.dispatch(
        client,
        system_prompt,
        user_prompt,
        model=model,
        provider=route.provider,
        access_mode=route.access_mode,
        profile_id=route.profile_id,
        external_full_text=bool(route.external_full_text),
        temperature=temperature,
        max_tokens=max_out,
        parse=validate_note,
        repair_prompt=build_repair_prompt,
        next_hop=_runtime_fallback(
            effective_config, route, resolved_class, backend or _BACKEND_OVERRIDE
        ),
    )
    # A runtime fallback hop may have changed the model/provider — recompute cost.
    model = disp.model or model
    provider_name = disp.provider or route.provider
    access_mode_name = disp.access_mode or route.access_mode
    cap = capabilities.models.get(model) if model else cap
    input_tokens = disp.input_tokens
    output_tokens = disp.output_tokens
    external_full_text = 1 if disp.external_full_text else 0
    estimated_cost = _estimate_cost(cap, input_tokens, output_tokens)

    if disp.status != "success" or disp.parsed is None:
        # Provider/transport error or unrepairable JSON -> extraction_failed, no
        # note, one usage row recording the status/error_code + sha256 hashes.
        code = disp.error.code if disp.error is not None else "invalid_response"
        result = _terminal_run(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="extraction_failed",
            access_class=resolved_class,
            model_name=model,
            provider=provider_name,
            access_mode=access_mode_name,
            temperature=temperature,
            external_full_text=external_full_text,
            input_tokens=input_tokens or None,
            output_tokens=output_tokens or None,
            estimated_cost=estimated_cost,
            message=f"extraction failed for {work_id}: {code}",
            errors=[disp.error.message] if disp.error is not None else [],
            usage=False,
        )
        executor.log_llm_usage(
            raw_conn(project_session),
            task_type=TASK_CLASS,
            result=disp,
            source_access_class=resolved_class,
            run_id=build_run_id,
            external_full_text=bool(external_full_text),
            estimated_cost=estimated_cost,
        )
        _account_budget(budget_state, config, estimated_cost)
        return result

    note = disp.parsed

    try:
        drafts, note_text, archetype = normalize_note(note)
    except ValueError as exc:
        result = _terminal_run(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="extraction_failed",
            access_class=resolved_class,
            model_name=model,
            provider=provider_name,
            access_mode=access_mode_name,
            temperature=temperature,
            external_full_text=external_full_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=estimated_cost,
            message=f"extraction failed for {work_id}: {exc}",
            errors=[str(exc)],
            usage=True,
        )
        _account_budget(budget_state, config, estimated_cost)
        return result

    saved = save_normalized_note(
        project_session, cache_session, work_id=work_id, markdown_id=markdown_id,
        markdown_hash=markdown_hash, schema_id=schema_id, drafts=drafts,
        note_text=note_text, archetype=archetype, raw_note_json=note.model_dump_json(),
        resolved_class=resolved_class, model_name=model, provider=provider_name,
        access_mode=access_mode_name, temperature=temperature,
        external_full_text=external_full_text, input_tokens=input_tokens,
        output_tokens=output_tokens, estimated_cost=estimated_cost,
        build_run_id=build_run_id, cache_root=cache_root,
    )

    # 11. Usage log (append-only audit; status + sha256 hashes, NO bodies) + budget.
    executor.log_llm_usage(
        raw_conn(project_session),
        task_type=TASK_CLASS,
        result=disp,
        source_access_class=resolved_class,
        run_id=build_run_id,
        external_full_text=bool(external_full_text),
        estimated_cost=estimated_cost,
    )
    _account_budget(budget_state, config, estimated_cost)

    return saved



def profile_temperature(profile) -> float:
    """Deterministic extraction temperature (0.0) — kept as one call site."""
    return 0.0


def _account_budget(budget_state: BudgetState, config, cost: float | None) -> None:
    """Add ``cost`` to the running total and trip the stop latch on a soft limit.

    Thin delegate to the locked :meth:`BudgetState.add_spend` (D6) — kept as the
    historical single call site (tests import it by name).
    """
    budget_state.add_spend(cost, config.budget)
