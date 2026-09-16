"""Chunk-1 read/query tools for the MCP server (design 00 §4.1, plan 01 chunk 1).

Thin adapters over the service layer (decision 81): each tool body resolves a
cached :class:`ProjectHandle` via ``ctx.get_handle`` (which owns the M9
``project_not_found`` error), calls one service function, wraps its return, and
returns it. No reshaping beyond ``{"key": ...}`` envelopes and Pydantic
``model_dump()`` — the service functions are the single source of truth.

This module imports **no** ``mcp`` SDK symbol: the SDK stays confined to
``server.py`` (and the deferred ``_tool_error`` import in ``context.py``). Every
name here is a base-package service seam, so importing this module never pulls
the SDK. The chunk-1 tools never raise a structured error directly — ``get_handle``
raises ``project_not_found`` on a bad slug, and the run/lens tools degrade to
empties on a missing run/lens rather than erroring — so no ``_tool_error`` import
is needed here.

The §5.1 redaction filter (:func:`redact_response`) is implemented ONCE and
applied by the single registration wrapper (:func:`register_read_tools`) around
every tool's return, never per-tool.
"""

from __future__ import annotations

import functools
import inspect
import sqlite3
from dataclasses import asdict
from typing import Any

from sqlmodel import Session

from .. import run as run_mod
from ..acquisition import service as acquisition_service
from ..answer import harness as answer_harness
from ..answer.types import AnswerMode
from ..errors import ConfigError, SeedgraphError
from ..fts import search as fts_search
from ..graph import analyze as graph_analyze_mod
from ..lenses import registry as lens_registry
from ..lenses import runner as lens_runner
from ..project import review as project_review
from ..project import service as project_service
from ..semantic import query as semantic_query

# The redaction verdict routes through the EXISTING single source (design 00
# §5.1): ``vocab.is_shareable`` as re-exported by ``semantic/access.py``. No new
# shareability logic lives in ``seedgraph.mcp`` — fail-closed on unknown/NULL.
from ..semantic.access import is_shareable

# ---------------------------------------------------------------------------
# §5.1 redaction filter — field-name registry + response walk (co-located with
# the tool wrapper per plan risk 5).
# ---------------------------------------------------------------------------

# Fields carrying verbatim span-quote / claim full text, keyed off a co-located
# ``access_class`` sibling. Grounded in the ACTUAL return shapes, not guessed:
#   * ``claim_text``  — ``lenses.runner.lens_results`` rows (results.py:156) AND
#     ``semantic.query.concept_detail`` claim rows (chunk-2 ``concept_show``).
#   * ``exact_quote`` — ``semantic.query.concept_detail`` span rows (chunk-2
#     ``concept_show``): the verbatim evidence-span quote (query.py:255).
#   * ``quote_text``  — ``fts.search.SpanHit`` (chunk-2 ``search_spans``): the
#     verbatim span text of a search hit (search.py:24).
#   * ``text``        — ``answer.types`` Span verbatim text (types.py:96), the
#     chunk-3 ``ask`` envelope; listed now so chunk 3 plugs in for free.
#   * ``quote``       — ``answer.types`` Citation verbatim quote (types.py:134).
# ``normalized_label`` / ``canonical_label`` are deliberately absent: a label is
# the extractor's synthesis across papers, not a verbatim phrase from any one of
# them (§4.3 rule 8), so labels are never blanked.
_REDACTABLE_TEXT_FIELDS: frozenset[str] = frozenset(
    {"claim_text", "exact_quote", "quote_text", "text", "quote"}
)


def redact_response(obj: Any, *, redact_private: bool) -> Any:
    """Blank verbatim full-text fields on works whose access class is not shareable.

    Design 00 §5.1. When ``redact_private`` is on, walk ``obj`` and, for any dict
    that carries an ``access_class`` key whose value is NOT shareable (decided via
    ``semantic.access.is_shareable`` — the existing single source, fail-closed on
    unknown/NULL), replace each co-located field in :data:`_REDACTABLE_TEXT_FIELDS`
    with ``{"redacted": True, "access_class": <class>}``. The marker is never
    dropped: the model must see that withheld evidence *exists*. Shareable rows and
    every non-text field pass through untouched.

    When ``redact_private`` is off (the default, CLI parity) this is a
    pass-through no-op — ``obj`` is returned unchanged. The walk never mutates its
    input (it builds fresh dicts/lists), so a caller holding the original is safe.

    Most chunk-1 tools return metadata rows with no ``access_class``/text pair and
    so are untouched even with the flag on; ``lens_results`` is the one carrier
    (``claim_text`` + ``access_class``). Chunk 3's ``ask`` envelope exercises the
    ``text``/``quote`` fields.
    """
    if not redact_private:
        return obj
    return _walk(obj)


def _walk(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_walk(item) for item in obj]
    if isinstance(obj, dict):
        access_class = obj.get("access_class")
        redact_here = "access_class" in obj and not is_shareable(access_class)
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if redact_here and key in _REDACTABLE_TEXT_FIELDS and value is not None:
                out[key] = {"redacted": True, "access_class": access_class}
            else:
                out[key] = _walk(value)
        return out
    return obj


def _stamp_access_class(
    conn: sqlite3.Connection,
    rows: list[dict],
    *,
    id_key: str,
    table: str,
) -> None:
    """Stamp each row's denormalized ``access_class`` so the §5.1 filter can see it.

    Redaction-coverage fix (design 00 §5.1, plan risk 5). Two chunk-2 tools return
    verbatim full text on rows that do NOT co-locate an ``access_class``:
    ``concept_detail``'s ``claims`` carry ``claim_text`` but no ``access_class``, and
    ``search_spans``' hits carry ``quote_text`` and no ``access_class`` — so the
    :func:`redact_response` walk could not tell they belong to a non-shareable work
    and private text would leak with ``--redact-private`` on. This joins the EXISTING
    denormalized ``access_class`` already stored on each row's own ``{table}`` row
    (``extracted_claims`` for claims, ``evidence_spans`` for spans — the same
    per-row denorm ``semantic.access.resolve_access_class`` trusts) keyed by
    ``id_key``, and stamps it onto ``rows`` in place. It does NOT compute
    shareability — that verdict stays :func:`semantic.access.is_shareable`. ``rows``
    is mutated in place; ``[]`` is a no-op. ``id_key`` / ``table`` are code
    constants (never user input), so their interpolation is safe.
    """
    ids = [r[id_key] for r in rows if r.get(id_key) is not None]
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    ac_map = {
        row[0]: row[1]
        for row in conn.execute(
            f"SELECT {id_key}, access_class FROM {table} "  # noqa: S608 (code constants)
            f"WHERE {id_key} IN ({placeholders})",
            ids,
        ).fetchall()
    }
    for r in rows:
        r["access_class"] = ac_map.get(r.get(id_key))


def _stamp_citation_access_class(
    conn: sqlite3.Connection, citations: list[dict]
) -> None:
    """Stamp each serialized ``Citation``'s ``access_class`` so §5.1 can see it.

    Redaction-coverage fix for the chunk-3 ``ask`` envelope — the same class of gap
    chunk 2 closed for ``concept_show`` / ``search_spans``. ``AnswerEnvelope``
    exposes exactly one verbatim-text field, ``Citation.quote`` (the stored
    ``evidence_spans.exact_quote``, decision 53), and :class:`answer.types.Citation`
    is ``extra="forbid"`` with NO ``access_class`` — so the :func:`redact_response`
    walk could not tell a quote belongs to a non-shareable work and private span
    text would leak with ``--redact-private`` on.

    ``quote`` is built from ``span_ids[0]`` (``compose._build_citation``), so the
    stamp reads the EXISTING denormalized ``evidence_spans.access_class`` of that
    exact span — not a fold over the work, which would over-redact a shareable quote
    merely because the same work also holds an unrelated private span. No
    shareability logic is added: the verdict stays
    :func:`semantic.access.is_shareable`. A citation with no spans carries no quote
    and is stamped ``None`` (fail-closed). ``citations`` is mutated in place.
    """
    ids = sorted({c["span_ids"][0] for c in citations if c.get("span_ids")})
    ac_map: dict[str, str | None] = {}
    if ids:
        placeholders = ",".join("?" * len(ids))
        ac_map = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT span_id, access_class FROM evidence_spans "
                f"WHERE span_id IN ({placeholders})",  # noqa: S608 (bound params)
                ids,
            ).fetchall()
        }
    for c in citations:
        spans = c.get("span_ids") or []
        c["access_class"] = ac_map.get(spans[0]) if spans else None


# ---------------------------------------------------------------------------
# §5.3 budget handshake (design 00 §5.3, plan 01 chunk 3)
# ---------------------------------------------------------------------------

#: The routing task key the answer composer resolves (``answer.compose.TASK_TYPE``).
_ANSWER_TASK_TYPE = "answer_generation"


def _budget_handshake(ctx: Any, handle: Any, project: str, *, confirm_spend: bool) -> None:
    """Fail closed before any paid ``ask`` dispatch unless the client confirmed (§5.3).

    Mirrors ``web.planner.plan_job``'s pre-flight exactly — resolve the task route,
    price the prospective call with ``llm.cost.preflight_estimate`` against the
    routed model's ``ModelCapability``, then fold it into
    ``llm.cost.budget_status(conn, cfg.budget, estimate)`` (prior DB-recorded
    monthly spend + this call). If that snapshot says the project's policy
    ``requires_confirmation`` (``budget.require_confirmation_above_usd``) or the
    projection crosses ``budget.monthly_soft_limit_usd``, and ``confirm_spend`` is
    False, raise the M9 ``budget_confirmation_required`` error carrying the numbers
    the client needs to decide; the client re-calls with ``confirm_spend=true``.

    This is a CONSENT gate, not a cap: the project spend cap is advisory (dogfooding
    found a run exceed it), so a "the cap will catch it" shortcut would leave the
    MCP paid path ungated. ``confirm_spend=True`` when nothing needed confirming is
    a no-op, never an error. Called ONLY when ``no_llm=False`` — the free default
    path loads no config and resolves no route.

    Input tokens are the config-declared prompt ceiling
    (``answer.max_evidence_tokens + answer.prompt_overhead_tokens``, the same budget
    ``compose.build_prompt`` fills) and output is ``answer.reserved_output_tokens``:
    a pre-retrieval upper bound, deliberately conservative — a consent gate must not
    under-quote.
    """
    from ..config.loader import load_llm_capabilities, load_project_config
    from ..llm import cost as llm_cost
    from ..llm.routing import NoLlmRoute, resolve_route
    from ..vocab import AccessClass

    # Deferred import breaks the server<->tools_read load cycle (mirrors context.py).
    from .server import _tool_error

    cfg = load_project_config(handle.slug, ctx.root)
    try:
        caps = load_llm_capabilities()
    except SeedgraphError:
        caps = None

    # Task-level route for PRICING only (planner.plan_job passes "open_access" too).
    # The real per-answer content gate keys on the retrieved spans' access class and
    # fires inside compose (§5.2) — this is not a second gate.
    try:
        route = resolve_route(_ANSWER_TASK_TYPE, AccessClass.open_access.value, cfg)
    except ConfigError as exc:
        _tool_error("gate_refused", str(exc), project=project)

    profile_id: str | None = None
    cap = None
    if not isinstance(route, NoLlmRoute):
        profile_id = route.profile_id
        profile = cfg.llm.profiles.get(profile_id)
        model = profile.model if profile is not None else None
        cap = caps.models.get(model) if (caps is not None and model) else None

    in_tokens = cfg.answer.max_evidence_tokens + cfg.answer.prompt_overhead_tokens
    estimate = llm_cost.preflight_estimate(
        cap, in_tokens, cfg.answer.reserved_output_tokens
    )
    with ctx.project_conn(handle) as conn:
        status = llm_cost.budget_status(conn, cfg.budget, estimate)

    if confirm_spend:
        return  # explicit consent — a no-op when nothing needed confirming.
    if status.requires_confirmation or status.over_monthly_soft_limit:
        _tool_error(
            "budget_confirmation_required",
            f"A paid answer for project {project!r} needs explicit confirmation "
            f"(estimate ${estimate:.4f}; month-to-date ${status.prior_spend_usd:.4f}). "
            "Re-call `ask` with confirm_spend=true to proceed, or leave no_llm=true "
            "for the free retrieval-only answer.",
            estimate_usd=estimate,
            monthly_spend_usd=status.prior_spend_usd,
            monthly_limit_usd=status.monthly_soft_limit_usd,
            profile=profile_id,
            requires_confirmation=status.requires_confirmation,
            over_monthly_soft_limit=status.over_monthly_soft_limit,
            confirmation_threshold_usd=status.confirmation_threshold_usd,
        )


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_read_tools(mcp: Any, ctx: Any) -> None:
    """Register the 9 chunk-1 read tools on ``mcp``, closing over ``ctx`` lexically.

    Every tool's return is passed through the §5.1 redaction filter by the single
    ``_redacting`` registration wrapper (not per-tool code). The wrapper preserves
    each tool's signature so FastMCP derives the identical JSON schema (``wraps``
    sets ``__wrapped__``, which ``inspect.signature`` follows; setting
    ``__signature__`` is belt-and-suspenders).
    """

    def _redacting(fn):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            return redact_response(
                fn(*args, **kwargs), redact_private=ctx.redact_private
            )

        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    def _tool(fn):
        """Register ``fn`` as a redacting MCP tool (name + description from ``fn``)."""
        mcp.tool()(_redacting(fn))
        return fn

    @_tool
    def project_list() -> dict:
        """List every project slug under the server's home root.

        The only tool with no ``project`` parameter — it enumerates the home the
        server was started against. An empty list is a valid answer.
        """
        return {"projects": project_service.list_projects(root=ctx.root)}

    @_tool
    def project_dashboard(project: str) -> dict:
        """Project dashboard read-model (verbatim).

        Corpus membership counts, extraction coverage, concept/claim/lens totals,
        open reviews, run summaries, per-task model availability, monthly budget,
        and a doctor badge. Pure read; safe offline/keyless.
        """
        h = ctx.get_handle(project)
        return project_service.project_dashboard(h)

    @_tool
    def list_documents(project: str, statuses: list[str] | None = None) -> dict:
        """Corpus documents (work + membership) for the given inclusion statuses.

        ``statuses`` filters ``inclusion_status``; omit it for the service default
        (included + metadata_only + excluded). Rows carry bibliographic metadata
        (title/label/year) and membership, never full text.
        """
        h = ctx.get_handle(project)
        if statuses is None:
            rows = project_service.list_documents(h)
        else:
            rows = project_service.list_documents(h, statuses=tuple(statuses))
        return {"documents": rows}

    @_tool
    def corpus_status(project: str, statuses: list[str] | None = None) -> dict:
        """Dense corpus table with acquisition + enrichment flags.

        Per work: ``inclusion_status`` / ``access_status`` / ``acquisition_state`` /
        ``has_markdown`` plus ``has_citations`` / ``has_extraction`` /
        ``has_concepts`` / ``has_review``. No full text is returned.

        Mental model: **seeds are the spine** of the corpus; **provided papers are
        the local literature corpus** (their full text is held and read);
        **``metadata_only`` works are the citation frontier + access-blocked
        papers** — known by metadata only, never locally read.
        """
        h = ctx.get_handle(project)
        rows = acquisition_service.corpus_rows(
            h, statuses=tuple(statuses) if statuses else None
        )
        return {"rows": rows}

    @_tool
    def review_list(project: str) -> dict:
        """Open review-queue items as display rows.

        Each row: ``item_id`` / ``item_type`` / ``target_id`` / ``status``, a
        human ``decision`` line (WHAT is being decided), and the item type's action
        ``menu``. The decision line carries concept/work labels and bibliographic
        metadata — not verbatim span or claim full text.
        """
        h = ctx.get_handle(project)
        items = project_review.list_open(h)
        return {"items": project_review.review_rows(h, items)}

    @_tool
    def run_list(project: str) -> dict:
        """Every run of the project, most-recent first.

        Each row: ``run_id`` / ``created_at`` / ``sections`` / ``last_event`` /
        derived ``status`` (running / finished / failed / interrupted / None).
        """
        h = ctx.get_handle(project)
        return {"runs": run_mod.list_runs(h.slug, root=ctx.root)}

    @_tool
    def run_events(project: str, run_id: str, after: int = -1) -> dict:
        """Poll a run's progress events (the web UI's polling contract).

        Returns ``events`` with ``seq > after``, the run's coarse ``status``
        derived over the FULL log (``running`` / ``finished`` / ``failed`` /
        ``interrupted``; ``None`` when there are no events yet), and ``next_after``
        — the highest ``seq`` returned, to pass as ``after`` on the next poll.
        Watches CLI/``serve``-launched jobs; job launch itself is a v2 tool.
        """
        h = ctx.get_handle(project)
        events = run_mod.read_events(h.slug, run_id, after=-1, root=ctx.root)
        status = run_mod._status_from_events(events)
        visible = [e for e in events if int(e.get("seq", -1)) > after]
        next_after = visible[-1]["seq"] if visible else after
        return {"events": visible, "status": status, "next_after": next_after}

    @_tool
    def lens_results(project: str, lens_id: str, status: str = "found") -> dict:
        """Result rows for one lens (default ``status='found'``).

        Each row: ``lens_output_id`` / ``work_id`` / ``status`` / ``claim_id`` /
        ``claim_text`` / ``normalized_label`` / ``section`` / ``confidence`` /
        ``span_ids`` / ``access_class``. ``claim_text`` is verbatim claim full text
        and is blanked for non-shareable works under ``--redact-private``.
        """
        h = ctx.get_handle(project)
        with Session(h.engine) as session:
            rows = lens_runner.lens_results(session, lens_id, status=status)
        return {"results": rows}

    @_tool
    def lens_staleness(project: str, lens_id: str) -> dict:
        """Coverage + staleness read-model for one lens (verbatim).

        Registry status, per-status coverage counts, works covered/total, and the
        stale run ids. An unregistered lens yields a clean zeroed state rather than
        an error.
        """
        h = ctx.get_handle(project)
        project_dir = h.root / "projects" / h.slug
        with Session(h.engine) as session:
            return lens_registry.lens_staleness(session, project_dir, lens_id)


def register_concept_graph_tools(mcp: Any, ctx: Any) -> None:
    """Register the 5 chunk-2 concept / search / graph tools on ``mcp`` (00 §4.1).

    Same shape as :func:`register_read_tools`: each body resolves a cached handle
    (``ctx.get_handle`` owns the M9 ``project_not_found`` error), opens an M7
    connection via ``ctx.project_conn`` for the SQL-backed seams, calls one service
    function, and returns its (wrapped) result — passed through the same §5.1
    :func:`redact_response` filter by a local ``_redacting`` wrapper (chunk 1's is
    left untouched; both single-source the actual :func:`redact_response` logic).

    Two tools return VERBATIM span/claim full text whose service returns do not
    co-locate an ``access_class`` on the text-bearing row (``concept_detail``'s
    ``claims`` carry ``claim_text`` but no ``access_class``; ``search_spans``' hits
    carry ``quote_text`` and no ``access_class``). When ``--redact-private`` is on,
    the tool body stamps each row's denormalized ``access_class`` (via
    :func:`_stamp_access_class`) so the filter can withhold non-shareable text, and
    the field names ``exact_quote`` / ``quote_text`` are registered in
    :data:`_REDACTABLE_TEXT_FIELDS`. The shareability verdict itself stays
    ``semantic.access.is_shareable`` — no policy logic is added here. ``concept_show``
    and ``graph_analyze`` raise the M9 structured errors ``concept_not_found`` /
    ``no_citation_run`` via the deferred ``_tool_error`` import (the
    ``server`` <-> ``tools_read`` load-cycle break, mirroring ``context.py``).
    """

    def _redacting(fn):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            return redact_response(
                fn(*args, **kwargs), redact_private=ctx.redact_private
            )

        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    def _tool(fn):
        """Register ``fn`` as a redacting MCP tool (name + description from ``fn``)."""
        mcp.tool()(_redacting(fn))
        return fn

    @_tool
    def concepts_overview(project: str) -> dict:
        """Orientation sheet over the FULL concept overlay — step (a) of the loop.

        Ranks concepts by distinct-paper RECURRENCE (``paper_frequency`` — how many
        papers a concept recurs across). This is a DIFFERENT axis from
        ``concepts_list``'s IDF ``weight``: do NOT reuse the weight rule here. It is
        ORIENTATION, not retrieval — read the corpus's *shape* (which concept_types
        are populated, what recurs, what background is assumed) before touching the
        concept table. It is a rank-and-page view over the FULL overlay, NEVER a
        degree-pruned view (the singleton tail is real hyper-specific concepts, not
        noise). Returns ``{scale, background_frame, by_type, tail_hint}``; pure
        on-the-fly aggregation, no new table/column.
        """
        h = ctx.get_handle(project)
        with ctx.project_conn(h) as conn:
            return semantic_query.concept_overview(conn)

    @_tool
    def concepts_list(
        project: str,
        concept_type: str | None = None,
        status: str | None = None,
    ) -> dict:
        """List concepts (optionally scoped by ``concept_type`` / ``status``).

        Ordered weight-descending as the service emits. ``weight`` is IDF-style
        DISCRIMINATIVENESS, NOT importance: high = distinctive (few papers), low =
        ubiquitous — never a relevance or quality score. Select by MEANING: read the
        whole (typed) slice and match near-synonyms by meaning, never by a
        lexical/substring pre-filter (near-synonyms share no words with your term).
        ``concept_type`` is a STRUCTURAL filter (method / data /
        identification_assumption / …), not a lexical one. After orienting with
        ``concepts_overview``, scope by ``concept_type`` and read that typed slice
        whole. Returns ``{"concepts": [...]}``.
        """
        h = ctx.get_handle(project)
        with ctx.project_conn(h) as conn:
            rows = semantic_query.list_concepts(
                conn, concept_type=concept_type, status=status
            )
        return {"concepts": rows}

    @_tool
    def concept_show(project: str, concept_id: str) -> dict:
        """Full detail for one concept: its papers, claims, and evidence spans.

        ``concept_id`` is the FULL ``concept::…`` id (as emitted by
        ``concepts_list`` / ``concepts_overview``), NOT the human label. Returns
        ``{concept, papers[], claims[], spans[]}``. An unknown id is a real error
        (``concept_not_found``) — never the CLI's exit-0 "no concept" prose. Span
        ``exact_quote`` and claim ``claim_text`` are verbatim full text, withheld
        for non-shareable works under ``--redact-private``.
        """
        h = ctx.get_handle(project)
        with ctx.project_conn(h) as conn:
            detail = semantic_query.concept_detail(conn, concept_id)
            if detail is None:
                # Deferred import breaks the server<->tools_read load cycle
                # (mirrors context.py); never called at import time.
                from .server import _tool_error

                _tool_error(
                    "concept_not_found",
                    f"No concept {concept_id!r} in project {project!r}.",
                    concept_id=concept_id,
                    project=project,
                )
            # Redaction coverage (§5.1): concept_detail's claim rows carry
            # claim_text but no access_class of their own; stamp it so the filter
            # fires. Spans already carry access_class (exact_quote is registered).
            if ctx.redact_private:
                _stamp_access_class(
                    conn, detail["claims"], id_key="claim_id", table="extracted_claims"
                )
        return detail

    @_tool
    def search_spans(
        project: str,
        query: str,
        work_id: str | None = None,
        section_kind: str | None = None,
        limit: int = 20,
    ) -> dict:
        """Exact-term FTS5 search over evidence spans (verbatim, NO stemming).

        Terms match verbatim tokens — ``mixing`` does NOT match ``mix`` /
        ``mixture``; a multi-word ``query`` matches as an adjacent phrase. Optional
        ``work_id`` / ``section_kind`` filters and a result ``limit`` (default 20).
        Returns ``{"hits": [...]}`` ranked by bm25 (lower ``rank`` is better).
        ``quote_text`` is verbatim span text, withheld for non-shareable works under
        ``--redact-private``.
        """
        h = ctx.get_handle(project)
        with ctx.project_conn(h) as conn:
            hits = fts_search.search_spans(
                conn, query, work_id=work_id, section_kind=section_kind, limit=limit
            )
            rows = [asdict(hit) for hit in hits]
            # Redaction coverage (§5.1): SpanHit carries quote_text but no
            # access_class; stamp each span's from evidence_spans so the filter
            # can withhold non-shareable quotes.
            if ctx.redact_private:
                _stamp_access_class(
                    conn, rows, id_key="span_id", table="evidence_spans"
                )
        return {"hits": rows}

    @_tool
    def graph_analyze(project: str, run_id: str | None = None) -> dict:
        """Whole-graph citation-structure summary for a run.

        ``run_id`` defaults to the latest run (``run.latest_run_id``). With NO
        citation run at all this is a real error (``no_citation_run``) — the MCP
        tool is stricter than the CLI's silent ``adhoc`` fallback. Returns the
        deterministic summary dict: run header counts, ``communities`` (sizes),
        ``god_nodes`` (highest-centrality works), ``bridges`` (cross-community
        citation edges, flagged-vs-derived), and ``read_next``. Semantic reach (a
        shared concept) is NOT citation reach — these communities are built from
        real ``cites`` edges only.
        """
        h = ctx.get_handle(project)
        rid = run_id or run_mod.latest_run_id(h.slug, root=ctx.root)
        if rid is None:
            from .server import _tool_error

            _tool_error(
                "no_citation_run",
                f"Project {project!r} has no citation run to analyze "
                "(run `cite build` first).",
                project=project,
            )
        with ctx.project_conn(h) as conn:
            return graph_analyze_mod.analyze_summary(h, run_id=rid, conn=conn)


def register_ask_tool(mcp: Any, ctx: Any) -> None:
    """Register the chunk-3 ``ask`` tool on ``mcp`` (00 §4.1 / §5.2 / §5.3).

    Same registration shape as the other chunks: one thin body over
    ``answer.harness.answer`` (decision 81), passed through the same §5.1
    :func:`redact_response` filter by a local ``_redacting`` wrapper.

    Three things are load-bearing here.

    **Cost posture.** ``no_llm`` defaults to **True** — the free, local,
    deterministic retrieval floor. Reaching a paid model requires BOTH an explicit
    ``no_llm=false`` AND a passed gate: no other argument, and no default, can get
    there. When ``no_llm`` is true the body loads no config, resolves no route, and
    runs no budget query.

    **Consent (§5.3).** With ``no_llm=false`` the §5.3 handshake
    (:func:`_budget_handshake`) runs BEFORE dispatch. It is CONDITIONAL, not a
    blanket confirm-every-paid-call: it fails closed with
    ``budget_confirmation_required`` only when the priced snapshot reports
    ``requires_confirmation`` (the estimate clears
    ``budget.require_confirmation_above_usd``) OR ``over_monthly_soft_limit`` (the
    projection crosses ``budget.monthly_soft_limit_usd``) — and then only while
    ``confirm_spend`` is False. A cheap call under a project's thresholds proceeds
    without a handshake; ``confirm_spend=True`` when nothing needed confirming is a
    no-op, never an error.

    **Honesty (38/58, §5.2).** The real content/pricing gates live in the callees
    (``llm.routing.resolve_route`` inside ``answer.compose.generate``) and are NOT
    reimplemented here; a :class:`errors.ConfigError` escaping the harness is
    re-raised as ``gate_refused`` carrying the router's ORIGINAL message verbatim —
    never re-worded, never softened, never swallowed into a silent degrade.

    **Persistence.** Saving both answer and trace is best-effort by default.
    Delivery adds an explicit saved/skipped/not_saved status. Combined no-LLM and
    no-save requests validate an existing checkpointed corpus without mutations.
    """

    def _redacting(fn):
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            return redact_response(
                fn(*args, **kwargs), redact_private=ctx.redact_private
            )

        wrapper.__signature__ = inspect.signature(fn)  # type: ignore[attr-defined]
        return wrapper

    def _tool(fn):
        """Register ``fn`` as a redacting MCP tool (name + description from ``fn``)."""
        mcp.tool()(_redacting(fn))
        return fn

    @_tool
    def ask(
        project: str,
        question: str,
        no_llm: bool = True,
        mode: str = "project_only",
        graph_depth: int = 1,
        max_candidates: int = 40,
        confirm_spend: bool = False,
        no_save: bool = False,
    ) -> dict:
        """Answer a question over the project corpus with traceable citations.

        ``no_llm=true`` is the DEFAULT and the FREE path: deterministic FTS5
        retrieval + ranking returning a cited evidence list with NO prose and NO
        model call. Set ``no_llm=false`` only when synthesis is actually wanted — it
        may cost money, and if the project's budget policy requires confirmation the
        call fails closed with ``budget_confirmation_required`` (carrying
        ``estimate_usd`` / ``monthly_spend_usd`` / ``monthly_limit_usd`` /
        ``profile``); re-call with ``confirm_spend=true`` to proceed.

        A retrieval-only envelope is an HONEST answer, not a failure: empty
        ``answer_text`` with ``mode='retrieval_only'`` means "here is the evidence,
        no model summarized it". ``insufficient_evidence=true`` is a real negative
        answer — the corpus does not support one — never a reason to answer from
        background knowledge. Citations are in-corpus only; ``recommendations`` are
        works known by METADATA ONLY (never read), surfaced as leads, never as
        evidence.

        ``mode``: ``project_only`` (default) or ``allow_outside``. ``graph_depth``
        bounds citation-neighborhood traversal for citation-shaped questions;
        ``max_candidates`` caps ranked candidates before token budgeting. Citation
        ``quote`` is verbatim span text, withheld for non-shareable works under
        ``--redact-private``. Saving both artifacts is attempted by default;
        ``persistence`` reports saved, skipped or not_saved. ``no_save=true`` skips
        both files; combined with no_llm it reads a checkpointed, quiescent corpus
        without initialization or writes. Saved files remain unredacted locally.
        """
        from .server import _tool_error

        h = ctx.get_handle(project, read_only=no_llm and no_save)
        if mode not in (AnswerMode.PROJECT_ONLY.value, AnswerMode.ALLOW_OUTSIDE.value):
            # `retrieval_only` is an OUTPUT mode the harness sets on a degrade; it is
            # not a valid input, so this checks the two-member input set explicitly
            # rather than round-tripping the whole enum.
            _tool_error(
                "invalid_mode",
                f"mode must be 'project_only' or 'allow_outside', got {mode!r}.",
                mode=mode,
            )

        # §5.3: the consent gate runs BEFORE dispatch, and only on the paid path.
        if not no_llm:
            _budget_handshake(ctx, h, project, confirm_spend=confirm_spend)

        try:
            envelope, trace = answer_harness.answer(
                question,
                h,
                mode=AnswerMode(mode),
                max_candidates=max_candidates,
                no_llm=no_llm,
                graph_depth=graph_depth,
            )
        except ConfigError as exc:
            # §5.2 / 38/58: the router's (or pricing gate's) own words, verbatim.
            _tool_error("gate_refused", str(exc), project=project)
        except (SeedgraphError, OSError, sqlite3.Error, ValueError) as exc:
            _tool_error(getattr(exc, "code", "retrieval_failed"), str(exc), project=project)

        from ..answer.delivery import deliver

        payload = deliver(envelope, trace, slug=project, root=h.root, no_save=no_save)
        if ctx.redact_private:
            # Redaction coverage (§5.1): Citation carries `quote` but no
            # access_class; stamp it from the span the quote came from so the
            # filter can withhold non-shareable verbatim text.
            try:
                with ctx.project_conn(h) as conn:
                    _stamp_citation_access_class(conn, payload.get("citations") or [])
            except (SeedgraphError, OSError, sqlite3.Error, ValueError) as exc:
                _tool_error(getattr(exc, "code", "retrieval_failed"), str(exc), project=project)
        return payload
