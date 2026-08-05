"""Phase 4b — chunked map-reduce extraction orchestration (plan §5 / §6 / §10).

``extract_note_chunked`` is the phase_4b driver: it takes a work that phase_4
recorded as ``run_status='skipped_oversize'`` (the input worklist — decision D4)
and produces the SAME evidence-linked ``structured_notes`` artifact phase_4
produces for fitting papers, via a section map-reduce:

  staleness / idempotency  →  resolve source ``access_class`` (phase-4
  ``resolve_source_access_class`` cache-walk, fail-closed most-restrictive)  →
  ``plan_chunks`` over ``document_sections`` (D9 capability-snapshot sizing)  →
  per-chunk gates [config / per-chunk context-window / content-access / budget]
  + MAP extraction reusing phase-4 ``validate_note`` / ``normalize_note`` (one
  ``chunked_map`` ``extraction_runs`` row each, with ``chunk_index`` /
  ``chunk_count`` / ``chunk_section_ids``)  →  ``merge_chunk_drafts`` REDUCE
  (deterministic, no second LLM pass)  →  anchor every merged quote via the
  phase-3 ``ensure_span`` over the ``db.adapter.raw_conn(project_session)`` bridge
  (decision D7) against the FULL markdown (plan §4.5)  →  write the
  ``chunked_reduce`` run + one ``structured_notes`` row + merged ``extracted_claims``
  + ``claim_spans`` + ``claim_fts`` / ``note_fts`` in ONE transaction  →
  ``run.update_manifest(slug, run_id, {"chunked_extraction": ...})`` (decision D5,
  a disjoint top-level section, never phase-4's ``note_extraction``).

Honest, never-silently-dropped failure (D4): explicit ``skipped_*`` /
``extraction_failed`` paths each write run rows and no note —
``skipped_no_llm`` (no usable backend), ``skipped_policy`` (content gate refuses
external + no local), ``skipped_context`` (a chunk overflows the window —
work continues), ``skipped_oversize_section`` (an irreducible paragraph, or every
chunk overflows — no note), ``skipped_budget`` (budget stop). Per-chunk JSON that
stays unrepairable marks that map run ``extraction_failed`` and the reduce
proceeds over the succeeded chunks; only an all-fail work gets no note.

Reuses (does not re-author): phase-4 ``extraction/{schema,prompt,validator,
normalize}`` + ``resolve_source_access_class`` + ``db/fts``; phase-3
``spans.ensure_span`` + ``cache_access.open_cache_ro`` + ``segment.paragraphs`` +
``document_sections``; foundation ``ids.new_id``, ``llm/router``, ``llm/usage``,
``llm/capabilities`` (D9), ``db.adapter.raw_conn`` (D7), ``run.update_manifest``
(D5). Adds NO new artifact table — chunk provenance is the additive columns of
``db/schema/project/0008_chunked.sql`` (decision D6).

Decisions implemented: D4, D2, D5, D6, D7, D9.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .. import cache_access
from ..acquisition.bridge import resolve_work_markdown
from ..config.loader import load_llm_capabilities
from ..config.models import GlobalConfig
from ..db.adapter import raw_conn  # D7 bridge: Session -> live-txn DBAPI conn
from ..db.fts import ensure_fts_tables, reindex_claim_fts, reindex_note_fts
from ..errors import ConfigError
from ..ids import new_id
from ..llm import executor
from ..llm.backend import default_backend
from ..llm.profiles import resolve_profile
from ..llm.routing import NoLlmRoute, resolve_route
from ..llm.tokens import estimate_tokens
from ..llm.usage import UsageEvent, log_usage
from ..sections.store import load_sections
from ..spans.store import ensure_span
from ..vocab import StatusValue
from .chunker import plan_chunks
from .normalize import normalize_note
from .prompt import build_prompt, build_repair_prompt
from .reduce import merge_chunk_drafts
from .runner import (  # heavy reuse — phase_4 owns these (plan §5)
    BudgetState,
    _estimate_cost,
    _local_fallback_route,
    current_note,
    profile_temperature,
    resolve_source_access_class,
)
from .schema import PROMPT_VERSION, SCHEMA_ID, SCHEMA_VERSION
from .validator import validate_note

if TYPE_CHECKING:  # type-only; never imported at runtime, so import-clean today
    from sqlmodel import Session


# Reuses the existing ``note_extraction`` task class (plan §8) — same routing key,
# content-access gate, and budget path as phase_4's whole-doc extractor.
TASK_CLASS = "note_extraction"

_DEFAULT_SCHEMA_ID = SCHEMA_ID
_FOUND = StatusValue.found.value
_AMBIGUOUS = StatusValue.ambiguous.value

# Offline test seam (mirrors ``extraction.runner._BACKEND_OVERRIDE``): the CLI path
# cannot pass ``backend=`` through Typer, so tests monkeypatch this to inject a
# deterministic FakeLLMBackend; production leaves it ``None`` (the stub real backend
# then raises an actionable "configure a profile/key" error). pytest NEVER dispatches
# a real client.
_BACKEND_OVERRIDE = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Full extraction_runs column set INCLUDING the phase_4b chunk-provenance columns
# (0008_chunked.sql). One writer for both ``chunked_map`` and ``chunked_reduce``
# rows (raw SQL, like phase_4 — the ORM is mapping-only, D6).
_RUN_COLUMNS = (
    "extraction_run_id, work_id, markdown_id, markdown_hash, schema_id, lens_id, "
    "schema_version, prompt_version, model_name, provider, access_mode, temperature, "
    "access_class, external_full_text, run_status, input_tokens, output_tokens, "
    "estimated_cost, run_id, created_at, extraction_mode, parent_extraction_run_id, "
    "chunk_index, chunk_count, chunk_section_ids"
)
_RUN_PLACEHOLDERS = ", ".join(["?"] * 25)


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
    extraction_mode: str,
    model_name: str | None = None,
    provider: str | None = None,
    access_mode: str | None = None,
    temperature: float | None = None,
    external_full_text: int = 0,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    estimated_cost: float | None = None,
    build_run_id: str | None = None,
    parent_extraction_run_id: str | None = None,
    chunk_index: int | None = None,
    chunk_count: int | None = None,
    chunk_section_ids: str | None = None,
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
            extraction_mode,
            parent_extraction_run_id,
            chunk_index,
            chunk_count,
            chunk_section_ids,
        ),
    )


def _resolve_reserved_output(reserve_output, cap) -> int:
    """Resolve the output budget reserved from the window (plan §8 / D9).

    ``None`` -> the model's ``max_output_tokens`` (or a 25% fallback); a fraction in
    ``(0, 1]`` -> that fraction of the context window; any other number -> that many
    tokens.
    """
    window = int(cap.context_window_tokens)
    if reserve_output is None:
        # An explicit max_output_tokens (incl. 0) is honored; only an UNKNOWN
        # (None) output cap falls back to a conservative 25% reservation.
        if cap.max_output_tokens is not None:
            return int(cap.max_output_tokens)
        return max(1, window // 4)
    if isinstance(reserve_output, float) and 0.0 < reserve_output <= 1.0:
        return int(reserve_output * window)
    return int(reserve_output)


def _pseudo_sections(markdown_id: str, text: str):
    """Single whole-document section fallback when phase-3 sections are absent."""
    return [
        SimpleNamespace(
            section_id=None, start_char=0, end_char=len(text), ordinal=0
        )
    ]


def _write_reduce_skip(
    project_session,
    *,
    work_id,
    markdown_id,
    markdown_hash,
    schema_id,
    run_status,
    access_class,
    build_run_id,
    model_name=None,
    provider=None,
    access_mode=None,
    chunk_count=None,
) -> str:
    """Write ONE ``chunked_reduce`` run with a terminal skip/failure status, no note."""
    conn = raw_conn(project_session)
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
        extraction_mode="chunked_reduce",
        model_name=model_name,
        provider=provider,
        access_mode=access_mode,
        external_full_text=0,
        chunk_count=chunk_count,
        build_run_id=build_run_id,
    )
    project_session.commit()
    return run_id


def _account_chunk(budget_state, config, chunk_cost: float, paper_cost: float) -> None:
    """Accumulate one chunk's cost and trip the stop latch (per-paper + cumulative).

    ``per_run_soft_limit_usd`` is interpreted per PAPER (= Σ its chunks, plan §8);
    ``monthly_soft_limit_usd`` is cumulative across chunks AND works (the shared
    ``budget_state.spent_usd``). Delegates to the locked
    :meth:`BudgetState.add_spend` so accounting is race-free when the whole-note
    ``--concurrency`` pool re-routes an oversize work here (Build C chunk 6 / D6).
    """
    budget_state.add_spend(chunk_cost, config.budget, per_run_cost=paper_cost)


@dataclass(frozen=True)
class ChunkedExtractionResult:
    """Per-work outcome of ``extract_note_chunked`` (plan §6 CLI/summary surface).

    Carries everything the CLI prints per work — chunk count, per-chunk/total
    token+cost, merged-claim count, and any ``skipped_*`` reason — plus the
    provenance back-links a caller/eval (phase 9) needs to audit chunk-merge
    integrity: the ``chunked_reduce`` run id bound 1:1 to the note and the list of
    ``chunked_map`` run ids enumerating the plan.
    """

    work_id: str
    status: str
    """Overall outcome = the reduce run's ``run_status``
    (``success`` | ``extraction_failed`` | ``skipped_no_llm`` | ``skipped_policy``
    | ``skipped_oversize_section`` | ``skipped_budget`` | ``skipped`` (idempotent
    no-op when a current note exists and ``force`` is False))."""
    note_id: str | None = None
    """The single current ``structured_notes`` row id on success, else ``None``."""
    reduce_run_id: str | None = None
    """The ``chunked_reduce`` ``extraction_runs`` row id (bound 1:1 to the note)."""
    map_run_ids: list[str] = field(default_factory=list)
    """The ``chunked_map`` run ids, in ``chunk_index`` order."""
    chunk_count: int = 0
    chunks_succeeded: int = 0
    chunks_failed: int = 0
    merged_claim_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    external_full_text: bool = False
    """True iff any chunk's source text actually left the machine (content gate)."""
    skipped_reason: str | None = None
    """Human-readable reason when ``status`` is a ``skipped_*`` / failed value."""


def extract_note_chunked(
    project_session: "Session",
    cache_session: "Session",
    *,
    work_id: str,
    schema_id: str = _DEFAULT_SCHEMA_ID,
    profile_id: str | None = None,
    overlap_tokens: int = 128,
    reserve_output: float | int | None = None,
    keep_chunk_json: bool = False,
    force: bool = False,
    confirm_external: bool = False,
    dry_run: bool = False,
    build_run_id: str | None = None,
    budget_state: "BudgetState | None" = None,
    backend=None,
    config=None,
    capabilities=None,
    cache_root=None,
) -> ChunkedExtractionResult:
    """Extract one oversize work via section map-reduce into a single merged note.

    Orchestration per plan §6/§10 (summarized in the module docstring). The output
    is byte-for-schema-identical to a phase-4 note: exactly one *current*
    ``structured_notes`` row for ``(work_id, schema_id)`` (same staleness rule as
    phase_4 — latest row whose ``markdown_hash``/``schema_version``/``prompt_version``
    match the live cache), merged typed ``extracted_claims`` each carrying both D2
    provenance fields, and ``claim_spans → evidence_spans`` for every ``found``
    substantive claim (anchored against the FULL markdown via phase-3
    ``ensure_span`` over the D7 ``raw_conn`` bridge). All runs/note/claims/created
    spans are stamped most-restrictive ``access_class``; the run set (N
    ``chunked_map`` + 1 ``chunked_reduce``) is the chunk plan's persisted shadow.

    Args:
        project_session: SQLModel session on ``project.db`` (one open transaction;
            the reduce write — run + note + claims + claim_spans + FTS + the
            ``ensure_span`` inserts via ``raw_conn`` — commits atomically, D7).
        cache_session: SQLModel session on ``cache.db``; used read-only to resolve
            current markdown bytes/hash + source ``access_class`` (fail-closed).
        work_id: the work to extract (must be a phase-4 ``skipped_oversize`` work
            with no current note unless ``force``).
        schema_id: extraction schema; defaults to ``default_research_note_v1``.
        profile_id: override the routed preferred LLM profile.
        overlap_tokens: fixed inter-chunk overlap passed to ``plan_chunks``.
        reserve_output: fraction of the context window reserved for output (else
            the model's ``max_output_tokens`` when ``None``) — sizing input, D9.
        keep_chunk_json: persist per-chunk JSON on map runs for audit (default off,
            bounds size — plan §4.7).
        force: re-extract even if a current note exists (re-plan + re-extract).
        confirm_external: per-project confirmation to send private full text to an
            external backend (content gate, evaluated per chunk).
        dry_run: plan chunks + project total cost only; write nothing, dispatch
            nothing (plan §8).
        build_run_id: optional manifest/build run id threaded onto run rows + used
            for the ``chunked_extraction`` manifest section (D5).
        budget_state: USD accumulator threaded across BOTH the works loop and the
            inner chunk loop; enforces per-paper (= Σ its chunks) and cumulative
            limits, halts on ``stop_on_budget_exceeded`` (marks remaining
            ``skipped_budget``). Routes through the phase_0 budget enforcer's
            fail-closed-on-unverified-pricing rule (plan §8).

    Returns:
        A :class:`ChunkedExtractionResult` for this work.

    Implements: D4 (oversize → merged note), D2, D5, D6, D7, D9.

    ``backend`` injects a deterministic test backend (tests pass a FakeLLMBackend);
    ``config`` / ``capabilities`` / ``cache_root`` default to a keyless
    :class:`GlobalConfig`, the bundled D9 snapshot, and ``$SEEDGRAPH_HOME`` — these
    additive params mirror phase-4 ``extract_note`` so the phase is fully testable
    offline.
    """
    if config is None:
        config = GlobalConfig()
    if capabilities is None:
        capabilities = load_llm_capabilities()
    if budget_state is None:
        budget_state = BudgetState()

    # 1. Resolve work -> current markdown (none -> skip, no run row — phase_4 parity).
    resolved = resolve_work_markdown(project_session, work_id=work_id)
    if resolved is None:
        return ChunkedExtractionResult(
            work_id=work_id,
            status="skipped_no_markdown",
            skipped_reason=f"work {work_id} has no resolvable markdown",
        )
    markdown_id, markdown_hash = resolved
    md = cache_access.read_markdown(cache_session, cache_root, markdown_id)
    if md is None:
        return ChunkedExtractionResult(
            work_id=work_id,
            status="skipped_no_markdown",
            skipped_reason=f"markdown {markdown_id} unresolvable in cache",
        )

    resolved_class = resolve_source_access_class(cache_session, markdown_id)

    # 2. Idempotency / staleness (append-only, pull-based — same rule as phase_4).
    if not force:
        existing = current_note(project_session, work_id, schema_id, markdown_hash)
        if existing is not None:
            return ChunkedExtractionResult(
                work_id=work_id,
                status="skipped",
                note_id=existing.note_id,
                skipped_reason=f"current note already exists for {work_id} (use --force)",
            )

    # 3. Budget stop latch (tripped by a prior work/chunk) -> skipped_budget reduce run.
    if budget_state.is_stopped():
        rid = _write_reduce_skip(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="skipped_budget",
            access_class=resolved_class,
            build_run_id=build_run_id,
        )
        return ChunkedExtractionResult(
            work_id=work_id,
            status="skipped_budget",
            reduce_run_id=rid,
            skipped_reason=f"budget stop ({budget_state.stop_reason}); {work_id} not processed",
        )

    # 4. Route resolution + content-access gate (resolved ONCE per work; access_class
    #    is constant across a work's chunks, so the per-chunk content gate is the same
    #    route — text only leaves when route.external_full_text).
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
        # Content-access gate blocked the external route: try a local fallback, else
        # refuse this work (skipped_policy) — source text never leaves the machine.
        route = _local_fallback_route(effective_config)
        if route is None:
            rid = _write_reduce_skip(
                project_session,
                work_id=work_id,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                schema_id=schema_id,
                run_status="skipped_policy",
                access_class=resolved_class,
                build_run_id=build_run_id,
            )
            return ChunkedExtractionResult(
                work_id=work_id,
                status="skipped_policy",
                reduce_run_id=rid,
                skipped_reason=(
                    f"private source ({resolved_class}) cannot leave the machine for an "
                    f"external profile and no local profile is available; pass "
                    f"--confirm-external or configure a local profile"
                ),
            )

    # 5. No-LLM honest degradation (no rule-based substitute) -> skipped_no_llm.
    if isinstance(route, NoLlmRoute):
        rid = _write_reduce_skip(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="skipped_no_llm",
            access_class=resolved_class,
            build_run_id=build_run_id,
        )
        return ChunkedExtractionResult(
            work_id=work_id,
            status="skipped_no_llm",
            reduce_run_id=rid,
            skipped_reason=f"no usable LLM backend for chunked extraction on {work_id}",
        )

    profile = resolve_profile(route.profile_id, effective_config)
    model = profile.model
    cap = capabilities.models.get(model) if model else None
    external_full_text = 1 if route.external_full_text else 0

    # 6. Fail-closed on an unknown model (cannot verify the window -> cannot size
    #    chunks safely; consistent with phase_4's fail-closed oversize posture).
    if cap is None:
        rid = _write_reduce_skip(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="skipped_oversize_section",
            access_class=resolved_class,
            build_run_id=build_run_id,
            model_name=model,
            provider=route.provider,
            access_mode=route.access_mode,
        )
        return ChunkedExtractionResult(
            work_id=work_id,
            status="skipped_oversize_section",
            reduce_run_id=rid,
            skipped_reason=(
                f"model {model} is absent from the llm_capabilities.yaml snapshot; cannot "
                f"size chunks without a verified context window"
            ),
        )

    context_window = int(cap.context_window_tokens)

    # 7. Deterministic chunk plan (D9-sized from the capability snapshot).
    sys_empty, user_empty = build_prompt("", schema_id=schema_id)
    prompt_overhead = estimate_tokens(sys_empty) + estimate_tokens(user_empty)
    reserved_output = _resolve_reserved_output(reserve_output, cap)
    sections = load_sections(raw_conn(project_session), markdown_id)
    if sections:
        # References-tail exclusion (Build C chunk 3, D5): drop references-kind
        # sections from the chunk plan so no chunk reads the references body.
        # Fail-open: an all-references document (e.g. an annotated bibliography)
        # keeps its full section set; the _pseudo_sections whole-doc fallback
        # below is NEVER filtered (no section kinds exist without sections rows).
        # ponytail: plan_chunks' fixed inter-chunk overlap reaches backward, so a
        # MID-document references section can still bleed <= overlap_tokens*4
        # chars into the next chunk's read scope — acceptable ceiling (references
        # are a document tail in practice; the trim is a cost optimization, not a
        # correctness gate).
        plan_sections = [
            s for s in sections if getattr(s, "section_kind", None) != "references"
        ]
        if not plan_sections:
            plan_sections = sections
    else:
        plan_sections = _pseudo_sections(markdown_id, md.text)
    chunks = plan_chunks(
        plan_sections,
        md.text,
        model_caps=cap,
        prompt_overhead_tokens=prompt_overhead,
        reserved_output_tokens=reserved_output,
        overlap_tokens=overlap_tokens,
    )
    chunk_count = len(chunks)
    if chunk_count == 0:
        rid = _write_reduce_skip(
            project_session,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status="skipped_oversize_section",
            access_class=resolved_class,
            build_run_id=build_run_id,
            model_name=model,
            provider=route.provider,
            access_mode=route.access_mode,
        )
        return ChunkedExtractionResult(
            work_id=work_id,
            status="skipped_oversize_section",
            reduce_run_id=rid,
            chunk_count=0,
            skipped_reason=f"no usable chunks planned for {work_id} (empty document)",
        )

    # 8. dry-run: project the full cost (Σ chunks), write/dispatch NOTHING (plan §8).
    if dry_run:
        projected = sum(
            _estimate_cost(cap, c.est_prompt_tokens, reserved_output) for c in chunks
        )
        return ChunkedExtractionResult(
            work_id=work_id,
            status="dry_run",
            chunk_count=chunk_count,
            estimated_cost=projected,
            skipped_reason=(
                f"dry-run: {work_id} {chunk_count} chunk(s), ~${projected:.6f} projected"
            ),
        )

    # The reduce run id is pre-minted so every map run can carry it as parent (the
    # chunked_map rows are the chunk plan's persisted shadow). No FK on parent, so
    # the reduce row is written last with aggregate token totals.
    reduce_run_id = new_id("extr")
    dispatch_budget = max(1, context_window - reserved_output)

    if backend is not None:
        client = backend
    elif _BACKEND_OVERRIDE is not None:
        client = _BACKEND_OVERRIDE
    else:
        client = executor.build_backend(profile, effective_config)

    per_chunk_drafts: list[list] = []
    map_run_ids: list[str | None] = []
    succeeded = 0
    failed = 0
    oversize_skips = 0
    budget_skips = 0
    total_input = 0
    total_output = 0
    total_cost = 0.0
    paper_cost = 0.0
    any_external = False

    temperature = profile_temperature(profile)

    # 9. MAP: one chunked_map run per chunk (gate -> dispatch -> validate/normalize).
    for chunk in chunks:
        section_ids_json = json.dumps(chunk.section_ids)
        conn = raw_conn(project_session)

        # 9a. Per-chunk context gate (defense-in-depth vs estimator drift). A chunk
        #     over the whole window -> skipped_context; a chunk that fits the window
        #     but exceeds the input budget (an irreducible oversized paragraph, no
        #     room for reserved output) -> skipped_oversize_section. Either way no
        #     dispatch; the work continues over the other chunks (D4).
        if chunk.est_prompt_tokens > dispatch_budget:
            status = (
                "skipped_context"
                if chunk.est_prompt_tokens > context_window
                else "skipped_oversize_section"
            )
            mid = new_id("extr")
            _insert_run(
                conn,
                run_id=mid,
                work_id=work_id,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                schema_id=schema_id,
                run_status=status,
                access_class=resolved_class,
                extraction_mode="chunked_map",
                model_name=model,
                provider=route.provider,
                access_mode=route.access_mode,
                external_full_text=0,  # nothing dispatched -> nothing left the machine
                build_run_id=build_run_id,
                parent_extraction_run_id=reduce_run_id,
                chunk_index=chunk.index,
                chunk_count=chunk_count,
                chunk_section_ids=section_ids_json,
            )
            project_session.commit()
            per_chunk_drafts.append([])
            map_run_ids.append(mid)
            oversize_skips += 1
            continue

        # 9b. Budget gate: stop latch tripped by a prior chunk/work -> skipped_budget.
        if budget_state.is_stopped():
            mid = new_id("extr")
            _insert_run(
                conn,
                run_id=mid,
                work_id=work_id,
                markdown_id=markdown_id,
                markdown_hash=markdown_hash,
                schema_id=schema_id,
                run_status="skipped_budget",
                access_class=resolved_class,
                extraction_mode="chunked_map",
                model_name=model,
                provider=route.provider,
                access_mode=route.access_mode,
                external_full_text=0,
                build_run_id=build_run_id,
                parent_extraction_run_id=reduce_run_id,
                chunk_index=chunk.index,
                chunk_count=chunk_count,
                chunk_section_ids=section_ids_json,
            )
            project_session.commit()
            per_chunk_drafts.append([])
            map_run_ids.append(mid)
            budget_skips += 1
            continue

        # 9c. Dispatch the chunk through the executor (parse=validate_note, one
        #     repair re-prompt). A provider/transport error returns a typed
        #     non-success result (no crash) → the chunk is marked extraction_failed
        #     and the reduce proceeds over the succeeded chunks (D4).
        chunk_text = md.text[chunk.start_char:chunk.end_char]
        system_prompt, user_prompt = build_prompt(chunk_text, schema_id=schema_id)
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
            parse=validate_note,
            repair_prompt=build_repair_prompt,
        )
        in_tok = disp.input_tokens
        out_tok = disp.output_tokens
        note = disp.parsed if disp.status == "success" else None

        chunk_cost = _estimate_cost(cap, in_tok, out_tok)
        total_input += in_tok
        total_output += out_tok
        total_cost += chunk_cost
        paper_cost += chunk_cost
        if route.external_full_text:
            any_external = True

        drafts = None
        if note is not None:
            try:
                drafts, _nt, _arch = normalize_note(note)
            except ValueError:
                drafts = None

        mid = new_id("extr")
        if drafts is None:
            run_status = "extraction_failed"
            per_chunk_drafts.append([])
            failed += 1
        else:
            run_status = "success"
            per_chunk_drafts.append(drafts)
            succeeded += 1
        _insert_run(
            conn,
            run_id=mid,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status=run_status,
            access_class=resolved_class,
            extraction_mode="chunked_map",
            model_name=model,
            provider=route.provider,
            access_mode=route.access_mode,
            temperature=temperature,
            external_full_text=external_full_text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            estimated_cost=chunk_cost,
            build_run_id=build_run_id,
            parent_extraction_run_id=reduce_run_id,
            chunk_index=chunk.index,
            chunk_count=chunk_count,
            chunk_section_ids=section_ids_json,
        )
        project_session.commit()
        map_run_ids.append(mid)
        _account_chunk(budget_state, config, chunk_cost, paper_cost)

    # One aggregate usage row per dispatched work (Track 1: per-chunk dispatch,
    # ONE log_llm_usage at reduce — totals + status, no bodies). Only when at
    # least one chunk actually dispatched (succeeded or failed); pure skip paths
    # log nothing. Multi-chunk → no single prompt/response body, so the sha256
    # hash columns stay NULL (still no bodies — cross-cutting #5).
    if succeeded + failed > 0:
        executor.log_llm_usage(
            raw_conn(project_session),
            task_type=TASK_CLASS,
            result=executor.LLMResult(
                status="success" if succeeded > 0 else "extraction_failed",
                input_tokens=total_input,
                output_tokens=total_output,
                provider=route.provider,
                model=model,
                access_mode=route.access_mode,
                external_full_text=bool(any_external),
            ),
            source_access_class=resolved_class,
            run_id=build_run_id,
            external_full_text=bool(any_external),
            estimated_cost=total_cost,
        )
        project_session.commit()

    # 10. No chunk produced drafts -> no note; record the work-level reason (D4 —
    #     explicit, never silently dropped).
    if succeeded == 0:
        if oversize_skips > 0:
            work_status = "skipped_oversize_section"
            reason = f"every chunk overflowed the model window for {work_id}"
        elif budget_skips > 0 and failed == 0:
            work_status = "skipped_budget"
            reason = f"budget stop before any chunk of {work_id} was extracted"
        else:
            work_status = "extraction_failed"
            reason = f"all {chunk_count} chunk(s) of {work_id} failed JSON extraction"
        conn = raw_conn(project_session)
        _insert_run(
            conn,
            run_id=reduce_run_id,
            work_id=work_id,
            markdown_id=markdown_id,
            markdown_hash=markdown_hash,
            schema_id=schema_id,
            run_status=work_status,
            access_class=resolved_class,
            extraction_mode="chunked_reduce",
            model_name=model,
            provider=route.provider,
            access_mode=route.access_mode,
            external_full_text=1 if any_external else 0,
            input_tokens=total_input or None,
            output_tokens=total_output or None,
            estimated_cost=total_cost or None,
            build_run_id=build_run_id,
            chunk_count=chunk_count,
        )
        project_session.commit()
        return ChunkedExtractionResult(
            work_id=work_id,
            status=work_status,
            reduce_run_id=reduce_run_id,
            map_run_ids=[m for m in map_run_ids if m is not None],
            chunk_count=chunk_count,
            chunks_succeeded=0,
            chunks_failed=failed,
            input_tokens=total_input,
            output_tokens=total_output,
            estimated_cost=total_cost,
            external_full_text=any_external,
            skipped_reason=reason,
        )

    # 11. REDUCE (pure/deterministic, NO second LLM call) — merge per-chunk drafts.
    merged_claims, note_text, archetype = merge_chunk_drafts(
        per_chunk_drafts, source_run_ids=map_run_ids
    )

    # 12. ONE transaction (D7): reduce run + note + claims + claim_spans + FTS, with
    #     every found quote anchored against the FULL markdown via phase-3 ensure_span
    #     over the raw_conn bridge (plan §4.5 / §6).
    conn = raw_conn(project_session)
    note_id = new_id("note")
    now = _now()
    _insert_run(
        conn,
        run_id=reduce_run_id,
        work_id=work_id,
        markdown_id=markdown_id,
        markdown_hash=markdown_hash,
        schema_id=schema_id,
        run_status="success",
        access_class=resolved_class,
        extraction_mode="chunked_reduce",
        model_name=model,
        provider=route.provider,
        access_mode=route.access_mode,
        temperature=temperature,
        external_full_text=1 if any_external else 0,
        input_tokens=total_input,
        output_tokens=total_output,
        estimated_cost=total_cost,
        build_run_id=build_run_id,
        chunk_count=chunk_count,
    )
    raw_note_json = json.dumps(
        {
            "schema_id": schema_id,
            "archetype": archetype,
            "note_text": note_text,
            "merged_claims": [
                {
                    "field_key": d.field_key,
                    "claim_type": d.claim_type,
                    "status": d.status,
                    "normalized_label": d.normalized_label,
                    "claim_text": d.claim_text,
                    "epistemic_type": d.epistemic_type,
                    "assertion_status": d.assertion_status,
                    "confidence": d.confidence,
                    "source_chunk_index": getattr(d, "source_chunk_index", None),
                    "exact_quotes": getattr(d, "exact_quotes", []),
                }
                for d in merged_claims
            ],
        }
    )
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_id, schema_version, prompt_version, archetype, access_class, "
        "raw_note_json, note_text, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            note_id,
            reduce_run_id,
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
    for draft in merged_claims:
        claim_id = new_id("claim")
        status = draft.status
        inferred_explanation = draft.inferred_explanation
        span_ids: list[str] = []
        if status == _FOUND:
            quotes = getattr(draft, "exact_quotes", None) or (
                [draft.exact_quote] if draft.exact_quote else []
            )
            for quote in quotes:
                if not quote:
                    continue
                # Anchor against the FULL markdown (the chunk only scoped the read):
                # a document-wide-unique quote anchors with correct full-doc offsets
                # regardless of which chunk surfaced it; non-unique -> None (no span).
                sid = ensure_span(
                    conn,
                    cache_session,
                    cache_root,
                    markdown_id=markdown_id,
                    work_id=work_id,
                    exact_quote=quote,
                    access_class=resolved_class,
                    span_kind="manual",
                )
                if sid is not None and sid not in span_ids:
                    span_ids.append(sid)
            if not span_ids:
                # No verbatim anchor -> downgrade to ambiguous, NEVER fabricate a span
                # (phase-4 rule, unchanged).
                status = _AMBIGUOUS
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
                reduce_run_id,
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
        for rank, sid in enumerate(span_ids):
            conn.execute(
                "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) "
                "VALUES (?, ?, ?, ?)",
                (claim_id, sid, rank, now),
            )
            span_count += 1

    reindex_claim_fts(project_session, note_id)
    reindex_note_fts(project_session, note_id)
    project_session.commit()
    # NOTE: the single aggregate usage row was already written right after the MAP
    # loop (Track 1: one log_llm_usage per dispatched work), so the reduce step
    # adds no further usage row.

    return ChunkedExtractionResult(
        work_id=work_id,
        status="success",
        note_id=note_id,
        reduce_run_id=reduce_run_id,
        map_run_ids=[m for m in map_run_ids if m is not None],
        chunk_count=chunk_count,
        chunks_succeeded=succeeded,
        chunks_failed=failed,
        merged_claim_count=len(merged_claims),
        input_tokens=total_input,
        output_tokens=total_output,
        estimated_cost=total_cost,
        external_full_text=any_external,
        skipped_reason=None,
    )
