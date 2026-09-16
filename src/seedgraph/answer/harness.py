"""Answer-harness orchestration — the public ``answer()`` (plan §5 / §6 / §6.1).

``harness.answer()`` owns the 08 §5 eleven-step pipeline ordering and the
deterministic pre-classifier (steps 1-2). It wires retrieval → rank → (optional)
traversal → compose → guard end to end and returns a guarded
:class:`AnswerEnvelope`. Per-query-type branching happens only at steps 7-9; every
other step is shared.

Decisions implemented:
* 82 — deterministic retrieval + a single self-declaring LLM call.
* 58/38 — ``no_llm`` (or any policy/budget/profile block) yields ``retrieval_only``
  honest degradation, never rule-based prose.
* 81 — this core is the complete surface; CLI ``ask`` + the FastAPI read view are
  thin wrappers over this function.
* 16/35 — an answer is a query, not a build run: ``run_id`` controls only the
  ``--save`` nesting path.
* 14 — an unknown LLM-declared ``query_type`` / ``answer_category`` normalizes to the
  closest closed-vocabulary member (in ``compose``).

Empty-corpus clean exit (plan §4): when retrieval returns nothing (empty or missing
FTS) the harness short-circuits to ``insufficient_evidence=True`` /
``answer_category=unresolved`` with an ``empty_corpus`` warning and makes no LLM call.
"""

from __future__ import annotations

import re
import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .. import __version__
from ..config.loader import load_project_config as _load_runtime_config
from ..config.models import GlobalConfig
from ..db.connection import connect_readonly
from ..project import layout
from . import compose, guard, rank, retrieve, traverse
from .trace import (
    TEXT_PREVIEW_CHARS,
    AnswerTrace,
    NeighborhoodTrace,
    TraceCandidate,
)
from .types import (
    AnswerCategory,
    AnswerEnvelope,
    AnswerMode,
    QuerySpec,
    QueryType,
    Recommendation,
    RetrievedItem,
)

# Phase 5 ``ProjectHandle`` (slug + db_path + root + config). Annotated loosely.
Project = Any

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*")
_STOPWORDS = retrieve._STOPWORDS


def _has(low: str, *subs: str) -> bool:
    return any(sub in low for sub in subs)


def _preclassify(question: str) -> QuerySpec:
    """08 §5 steps 1-2 — deterministic pre-classification (no LLM).

    stdlib regex/keyword rules map the question to a ``protocol_hint`` and extract
    exact phrases (quoted substrings → FTS5 phrase queries), named work tokens (drives
    ``comparative`` partitioning, §6.4 — a single named work degrades to ``factual``),
    and candidate concept tokens. The final ``query_type`` is still LLM-self-declared
    (decision 82); this only selects the protocol.
    """
    q = question.strip()
    low = q.lower()
    phrases = re.findall(r'"([^"]+)"', q)
    work_tokens = list(phrases)

    if _has(low, "compare", "versus", " vs ", "difference between", "differ between", "contrast"):
        hint = QueryType.comparative
    elif _has(low, "cited by", "who cites", "citation", "references", "cites "):
        hint = QueryType.citation_search
    elif _has(low, "gap", "missing", "understud", "not been studied", "hasn't been",
              "has not been", "unanswered", "open question", "open problem", "overlooked"):
        hint = QueryType.gap_finding
    elif "assumption" in low:
        hint = QueryType.assumption_search
    elif _has(low, "evidence", "quote", "passage", "show me where", "where does",
              "support for", "where in"):
        hint = QueryType.evidence_request
    elif _has(low, "what is", "what are", "define", "definition", "explain", "what does",
              " mean", "concept of"):
        hint = QueryType.concept_explanation
    elif _has(low, "synthes", "across", "overall", "common theme", "summar",
              "in the literature", "consensus", "themes"):
        hint = QueryType.synthesis
    else:
        hint = QueryType.factual

    # §6.4 comparative degrades to factual unless >= 2 named works resolve.
    if hint == QueryType.comparative and len(work_tokens) < 2:
        hint = QueryType.factual

    concept_tokens: list[str] = []
    seen: set[str] = set()
    for token in _WORD_RE.findall(low):
        if len(token) < 4 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        concept_tokens.append(token)

    return QuerySpec(
        question=q,
        protocol_hint=hint,
        phrases=phrases,
        work_tokens=work_tokens,
        concept_tokens=concept_tokens[:12],
    )


def _resolve_work_id(conn: sqlite3.Connection, token: str) -> str | None:
    """Resolve a named-work token (e.g. ``"Paper A"``) to a ``work_id`` by a
    case-insensitive title match. Returns ``None`` when nothing resolves."""
    token = token.strip()
    if not token:
        return None
    row = conn.execute(
        "SELECT work_id FROM works WHERE lower(canonical_title) LIKE ? ORDER BY work_id LIMIT 1",
        (f"%{token.lower()}%",),
    ).fetchone()
    return row[0] if row is not None else None


def _comparative_retrieve(
    conn: sqlite3.Connection,
    project_slug: str,
    spec: QuerySpec,
    limit: int,
) -> list[RetrievedItem]:
    """§6.4 — run the factual protocol once per named work, candidates kept
    partitioned by ``work_id`` (side-by-side evidence).

    Each named work token is resolved to a ``work_id``; retrieval then runs once per
    resolved work, scoped to that work and post-filtered to it so the partitions are
    pure (notes/concept lookup ignore the ``work_id`` filter). Candidates are returned
    in contiguous per-work blocks (work A's evidence, then work B's). Work-name phrases
    are dropped from the per-partition FTS query (they identify the works, not the
    comparison dimension); the remaining phrases / question tokens drive recall within
    each work. Degrades to a single un-partitioned retrieval when fewer than two named
    works resolve (a single-work comparative is already pre-classified as factual)."""
    work_ids: list[str] = []
    for token in spec.work_tokens:
        wid = _resolve_work_id(conn, token)
        if wid is not None and wid not in work_ids:
            work_ids.append(wid)
    if len(work_ids) < 2:
        return retrieve.retrieve(conn, project_slug, spec, limit)

    dim_phrases = [p for p in spec.phrases if _resolve_work_id(conn, p) is None]
    per = max(1, limit // len(work_ids))
    partitioned: list[RetrievedItem] = []
    seen: set[str] = set()
    for wid in work_ids:
        sub = spec.model_copy(update={"work_id": wid, "phrases": dim_phrases})
        items = [
            it
            for it in retrieve.retrieve(conn, project_slug, sub, limit)
            if it.work_id == wid
        ]
        for item in items[:per]:
            if item.item_id in seen:
                continue
            seen.add(item.item_id)
            partitioned.append(item)
    return partitioned


def _citation_search_recommendations(
    conn: sqlite3.Connection,
    ranked: list,
    *,
    top: int,
    graph_depth: int = 1,
) -> tuple[list[Recommendation], list[str], dict | None]:
    """08 §5 step 7 — citation-graph traversal for ``citation_search``.

    Builds the phase-7 NetworkX project graph and takes the depth-bounded
    undirected :func:`traverse.neighborhood` of the retrieved (seed) works —
    at ``graph_depth=1`` (the default) the neighbour set and order are identical
    to the pre-Build-B per-seed ``cited_by + cites_of`` union. Span-less
    neighbour works (no extracted full text) are surfaced as graph-derived
    :class:`Recommendation`s — never citations. Returns ``(recs, warnings, nbhd)``
    where ``warnings`` carries the engulf notice when the neighborhood covers
    >= 80% of the citation graph and ``nbhd`` is the ``traverse.neighborhood``
    membership dict escaping for AnswerTrace assembly (00 §4.1) instead of dying
    as a local. ``([], [], None)`` when NetworkX or the graph is unavailable
    (best-effort enrichment, never raises)."""
    try:
        from ..semantic.graph_build import build_graph

        graph = build_graph(conn, run_id="ask")
    except Exception:  # noqa: BLE001 — graph build is best-effort enrichment
        return [], [], None
    seeds: list[str] = []
    for candidate in ranked:
        wid = candidate.item.work_id
        if wid not in seeds:
            seeds.append(wid)
    nbhd = traverse.neighborhood(graph, seeds, depth=graph_depth)
    neighbours = [n for n in nbhd["nodes"] if n not in seeds]
    warnings: list[str] = []
    if nbhd["engulfed"]:
        warnings.append(
            f"depth-{graph_depth} neighborhood covers >=80% of the citation "
            "graph; falling back to per-seed depth-1 for bridge claims is advised"
        )
    recs: list[Recommendation] = []
    for neighbour in neighbours:
        span_count = conn.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE work_id = ?", (neighbour,)
        ).fetchone()[0]
        if span_count:
            continue  # has spans -> citable elsewhere, not a graph recommendation
        title, year = compose._work_meta(conn, neighbour)
        status_row = conn.execute(
            "SELECT inclusion_status FROM project_documents WHERE work_id = ?", (neighbour,)
        ).fetchone()
        if status_row is not None and status_row[0] == "excluded":
            continue
        status = "metadata_only" if status_row is not None and status_row[0] == "metadata_only" else "unavailable"
        recs.append(
            Recommendation(
                work_id=neighbour,
                title=title,
                year=year,
                reason="connected to a retrieved work via the project citation graph; no extracted full text",
                status=status,
                action_hint="upload the PDF to extract evidence and cite this work",
            )
        )
        if len(recs) >= top:
            break
    return recs, warnings, nbhd


def _merge_recommendations(
    base: list[Recommendation], extra: list[Recommendation]
) -> list[Recommendation]:
    """Concatenate two recommendation lists, deduplicating by ``work_id`` (first wins)."""
    seen = {r.work_id for r in base}
    out = list(base)
    for rec in extra:
        if rec.work_id not in seen:
            seen.add(rec.work_id)
            out.append(rec)
    return out


def _abstain_envelope(
    spec: QuerySpec,
    mode: AnswerMode,
    warnings: list[str],
    recommendations,
) -> AnswerEnvelope:
    """The clean empty/weak short-circuit envelope — no prose, no LLM call (plan §4)."""
    return AnswerEnvelope(
        answer_id="ans_" + uuid4().hex,
        question=spec.question,
        query_type=spec.protocol_hint,
        answer_category=AnswerCategory.unresolved,
        answer_text="",
        citations=[],
        recommendations=recommendations,
        cited_work_ids=[],
        cited_span_ids=[],
        insufficient_evidence=True,
        retrieved_item_ids=[],
        warnings=warnings,
        mode=mode,
        llm_provenance=None,
    )


# --- AnswerTrace assembly (00 §4.1) — pure observation, at answer()'s returns -----

def _path_label(spec: QuerySpec) -> str:
    """00 §3 ``path`` — which retrieval branch ran. ``gap_finding`` returns before this
    is asked; ``citation_search`` runs the shared retrieval (``standard``) plus the
    neighborhood pass, so only ``comparative`` is distinct here."""
    if spec.protocol_hint == QueryType.comparative:
        return "comparative"
    return "standard"


def _disposition(position: int, cut_index: int | None, max_candidates: int) -> str:
    """Exact disposition (C2) for a candidate at 1-based ``position`` in the FULL ranked
    list (before ``harness.answer``'s ``ranked_full[:max_candidates]`` slice). Exactly
    three values (C2 pins three — no fourth), each naming which truncation dropped the row:

    * ``cut_rank`` — ``position > max_candidates``: dropped by the rank slice. This is the
      *only* meaning of ``cut_rank``; it is never reused for "abstained / nothing shown".
    * ``cut_budget`` — survived the slice, a prompt was built (``cut_index is not None``),
      and ``position`` fell past the token-budget break index. Because ``build_prompt``'s
      loop *breaks* rather than skips, the shown blocks are exactly the prefix, so a 1-based
      ``position <= cut_index`` is ``shown`` and everything after is ``cut_budget`` — read
      straight from ``build_prompt``'s report, never inferred from ``AllowedSet`` membership.
    * ``shown`` — survived the slice and was surfaced downstream. On the LLM / failed-
      dispatch path that means ``position <= cut_index`` (it went into the prompt). Off the
      prompt path (``cut_index is None`` — a retrieval-only degrade or an empty/weak
      abstain) no budget loop ran: the retrieval-only envelope surfaces the whole sliced
      list as its candidate set, so every survivor is ``shown``; the abstain envelope emits
      no candidate surface, but a survivor there still passed *both* truncation mechanisms —
      "nothing reached a model" is recorded by ``shown_evidence=None`` +
      ``outcome="abstain_empty"``, not by the disposition — so it is likewise ``shown``
      rather than mislabeled ``cut_rank``.
    """
    if position > max_candidates:
        return "cut_rank"
    if cut_index is not None:
        return "shown" if position <= cut_index else "cut_budget"
    return "shown"


def _trace_candidates(
    ranked: list, cut_index: int | None, max_candidates: int
) -> list[TraceCandidate]:
    """The one disposition-tagged candidate table (C2): every ranked candidate once — the
    FULL pre-slice ranked list, so a rank-truncated candidate (``position >
    max_candidates``) appears tagged ``cut_rank`` with all scores/boosts verbatim and text
    truncated to :data:`TEXT_PREVIEW_CHARS`. ``ranked`` is iterated in the SAME order
    compose fed ``build_prompt``, so ``cut_index`` splits ``shown`` from ``cut_budget``
    positionally among the survivors (see :func:`_disposition`)."""
    out: list[TraceCandidate] = []
    for position, candidate in enumerate(ranked, start=1):
        item = candidate.item
        preview = item.text
        if len(preview) > TEXT_PREVIEW_CHARS:
            preview = preview[:TEXT_PREVIEW_CHARS] + "…"
        out.append(
            TraceCandidate(
                item_id=item.item_id,
                kind=item.kind,
                work_id=item.work_id,
                span_id=item.span_id,
                concept_id=item.concept_id,
                section=item.section,
                disposition=_disposition(position, cut_index, max_candidates),
                rank_position=position,
                rank_score=candidate.rank_score,
                bm25_score=item.bm25_score,
                boosts=dict(candidate.boosts),
                access_class=item.access_class,
                epistemic_type=item.epistemic_type,
                concept_weight=item.concept_weight,
                concept_paper_frequency=item.concept_paper_frequency,
                text_preview=preview,
            )
        )
    return out


def _neighborhood_trace(nbhd: dict | None) -> NeighborhoodTrace | None:
    """Map the ``traverse.neighborhood`` dict (C4) onto its trace model verbatim — None
    on any non-``citation_search`` path (traversal didn't run)."""
    if nbhd is None:
        return None
    return NeighborhoodTrace(
        seeds=list(nbhd["seeds"]),
        unknown_seeds=list(nbhd["unknown_seeds"]),
        depth=nbhd["depth"],
        nodes=list(nbhd["nodes"]),
        edges=[list(e) for e in nbhd["edges"]],
        coverage_ratio=nbhd["coverage_ratio"],
        engulfed=nbhd["engulfed"],
    )


def _build_trace(
    *,
    answer_id: str,
    spec: QuerySpec,
    path: str,
    outcome: str,
    degrade_reasons: list[str],
    retrieved_count: int,
    ranked: list,
    max_candidates: int,
    nbhd: dict | None,
    compose_trace: compose.ComposeTrace | None = None,
) -> AnswerTrace:
    """Assemble the :class:`AnswerTrace` from ``answer()``'s locals (00 §4.1 — assembly
    happens ONLY here, at the return points; no collector object, no global state).

    ``ranked`` is the FULL pre-slice ranked list (``ranked_full``); ``max_candidates`` is
    the ``[:max_candidates]`` slice boundary, so :func:`_trace_candidates` can tag the
    rank-truncated overflow ``cut_rank`` (C2 / 00 §1 — dropped candidates were doubly lost
    before this). ``compose_trace`` (present only on the compose path) carries the exact
    shown-evidence list, prompt hash/version, and the token-budget break index — so
    ``shown``/``cut_budget`` are split exactly and ``dispositions_exact=True`` on every path
    (off the compose path no budget loop ran, so the labeling is unambiguous too). ``None``
    on the gap_finding / short-circuit-abstain paths (compose never ran): ``shown_evidence``
    / ``prompt_*`` stay ``None`` and the survivors are ``shown`` (see :func:`_disposition`)."""
    shown_evidence = compose_trace.shown_evidence if compose_trace is not None else None
    cut_index = compose_trace.cut_index if compose_trace is not None else None
    prompt_version = compose_trace.prompt_version if compose_trace is not None else None
    prompt_sha256 = compose_trace.prompt_sha256 if compose_trace is not None else None
    return AnswerTrace(
        answer_id=answer_id,
        question=spec.question,
        created_at=datetime.now(timezone.utc).isoformat(),
        seedgraph_version=__version__,
        boost_constants=rank.boost_constants(),
        spec=spec.model_dump(mode="json"),
        path=path,
        outcome=outcome,
        degrade_reasons=list(degrade_reasons),
        retrieved_count=retrieved_count,
        dispositions_exact=True,
        candidates=_trace_candidates(ranked, cut_index, max_candidates),
        shown_evidence=shown_evidence,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        neighborhood=_neighborhood_trace(nbhd),
    )


def answer(
    question: str,
    project: Project,
    mode: AnswerMode = AnswerMode.PROJECT_ONLY,
    max_candidates: int = 40,
    no_llm: bool = False,
    run_id: str | None = None,
    *,
    graph_depth: int = 1,
    config: GlobalConfig | None = None,
    capabilities=None,
    cache_root=None,
) -> tuple[AnswerEnvelope, AnswerTrace]:
    """Answer a project question through Seedgraph retrieval (plan §6, 08 §5).

    Runs the 08 §5 pipeline per the §6.1 trace: pre-classify → concept/claim/span
    retrieval → deterministic rank + ``rank_fusion`` → single-call compose or
    ``retrieval_only`` degrade → guard. The harness never opens another project's DB
    (private-by-default, decisions 30/60/76). Returns ``(envelope, trace)`` — the final
    guarded :class:`AnswerEnvelope` and the always-on :class:`~seedgraph.answer.trace.
    AnswerTrace` capturing the run's intermediate state (00 §4.1); on an empty/unextracted
    corpus a clean ``insufficient_evidence`` envelope with no LLM call (plan §4). The
    trace is pure observation — envelope output is byte-identical to the pre-trace
    pipeline on every path (C9). Callers persist both (00 §5) or drop the trace visibly
    (``env, _ = answer(...)``).

    Args:
        question: the user's natural-language question.
        project: resolved phase-5 project handle (``slug`` + ``db_path`` + ``root``).
        mode: ``PROJECT_ONLY`` (default) / ``ALLOW_OUTSIDE``.
        max_candidates: cap on ranked candidates before token budgeting (§8).
        no_llm: force ``retrieval_only`` honest degradation (decisions 58/38).
        run_id: present only inside a build run; controls the ``--save`` nesting path.
        graph_depth: citation-neighborhood hop bound for ``citation_search``
            (default 1 pins the pre-Build-B one-hop behavior; deeper widens the
            recommendation surface, with an engulf warning at >= 80% coverage).
        config: runtime routing config (loaded from disk when ``None``).
        capabilities / cache_root: D9 snapshot + cache root (defaults resolved).
    """
    spec = _preclassify(question)
    spec.mode = mode

    if config is None:
        config = _load_runtime_config(project.slug, root=getattr(project, "root", None))
    if cache_root is None:
        cache_root = getattr(project, "root", None)

    conn = (connect_readonly(project.db_path) if getattr(project, "read_only", False)
            else sqlite3.connect(str(project.db_path)))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        # 08 §5 step 7 (gap_finding): on-demand co-citation traversal over existing
        # rows -> recommendations only; no spans/citations, no LLM call (§6.1).
        if spec.protocol_hint == QueryType.gap_finding:
            recs = traverse.cocitation_candidates(conn, project.slug, max_candidates)
            # Build B chunk 4: read-next (high-centrality unread) recommendations
            # merged AFTER co-citation, first-wins dedup. The graph is obtained
            # best-effort here (mirrors _citation_search_recommendations) — the
            # gap_finding branch never built the semantic graph before, and
            # answer() deliberately does not thread a graph parameter.
            try:
                from ..semantic.graph_build import build_graph

                graph = build_graph(conn, run_id="ask")
                read_next_recs = traverse.read_next_candidates(
                    conn, graph, max_candidates
                )
            except Exception:  # noqa: BLE001 — graph analysis is best-effort enrichment
                read_next_recs = []
            recs = _merge_recommendations(recs, read_next_recs)
            env = _abstain_envelope(spec, mode, ["gap_finding"], recs)
            # gap_finding returns before retrieval: no candidates, no neighborhood (C6).
            trace = _build_trace(
                answer_id=env.answer_id,
                spec=spec,
                path="gap_finding",
                outcome="abstain_gap_finding",
                degrade_reasons=[],
                retrieved_count=0,
                ranked=[],
                max_candidates=max_candidates,
                nbhd=None,
            )
            return env, trace

        # comparative (§6.4): factual retrieval run once per named work, kept
        # work-partitioned for side-by-side evidence; otherwise the shared pipeline.
        if spec.protocol_hint == QueryType.comparative:
            items = _comparative_retrieve(conn, project.slug, spec, max_candidates)
        else:
            items = retrieve.retrieve(conn, project.slug, spec, max_candidates)
        # Keep the FULL ranked list for trace assembly (rank-truncated candidates must
        # appear as cut_rank, 00 §1/C2); everything downstream receives the sliced list so
        # the envelope stays byte-identical to the pre-trace pipeline (C9, parity fixtures).
        ranked_full = rank.rank_fusion(rank.deterministic_rank(items, spec))
        ranked = ranked_full[:max_candidates]

        # 08 §5 step 7 (citation_search): depth-bounded neighborhood traversal over
        # the phase-7 citation graph -> graph-derived recommendations for span-less
        # works (+ the engulf warning when the neighborhood swallows the graph).
        graph_recs: list[Recommendation] = []
        graph_warnings: list[str] = []
        nbhd: dict | None = None
        if spec.protocol_hint == QueryType.citation_search:
            graph_recs, graph_warnings, nbhd = _citation_search_recommendations(
                conn, ranked, top=max_candidates, graph_depth=graph_depth
            )

        short, sc_warnings = guard.short_circuit_if_empty(
            ranked,
            weak_evidence_floor=config.answer.weak_evidence_floor,
            absent_floor=config.answer.absent_floor,
        )
        if short:
            recs = _merge_recommendations(
                graph_recs, compose.build_recommendations(conn, spec, ranked)
            )
            env = _abstain_envelope(spec, mode, sc_warnings + graph_warnings, recs)
            # Weak/empty short-circuit: no prompt was built, so a candidate that survived
            # the rank slice is ``shown`` and only the rank-truncated overflow is
            # ``cut_rank`` (00 §1/C2 — the abstain path no longer blanket-labels cut_rank).
            # Neighborhood escapes for a citation_search that short-circuited (00 §3, C6).
            trace = _build_trace(
                answer_id=env.answer_id,
                spec=spec,
                path=_path_label(spec),
                outcome="abstain_empty",
                degrade_reasons=[],
                retrieved_count=len(items),
                ranked=ranked_full,
                max_candidates=max_candidates,
                nbhd=nbhd,
            )
            return env, trace

        passed_warnings = sc_warnings + graph_warnings
        envelope, allowed, compose_trace = compose.generate(
            ranked,
            spec,
            config,
            project_conn=conn,
            no_llm=no_llm,
            run_id=run_id,
            capabilities=capabilities,
            cache_root=cache_root,
            warnings=passed_warnings,
        )
        # outcome + degrade_reasons come straight from the ComposeTrace (00 §4.1): a
        # RETRIEVAL_ONLY envelope means one of the six degrade sites fired, and that site
        # set degrade_reasons to exactly the warning(s) it appended — no longer inferred
        # by slicing envelope.warnings. An LLM call that produced prose has empty reasons.
        outcome = "retrieval_only" if envelope.mode == AnswerMode.RETRIEVAL_ONLY else "llm"
        degrade_reasons = list(compose_trace.degrade_reasons) if compose_trace else []
        # Step 7 graph recommendations lead the list for citation_search (the graph
        # neighbours ARE the answer surface), then compose's metadata-only recs.
        if graph_recs:
            envelope.recommendations = _merge_recommendations(
                graph_recs, envelope.recommendations
            )
        envelope = guard.enforce(
            envelope, allowed, envelope.mode,
            support_floor=config.answer.support_floor, conn=conn,
        )
        trace = _build_trace(
            answer_id=envelope.answer_id,
            spec=spec,
            path=_path_label(spec),
            outcome=outcome,
            degrade_reasons=degrade_reasons,
            retrieved_count=len(items),
            ranked=ranked_full,
            max_candidates=max_candidates,
            nbhd=nbhd,
            compose_trace=compose_trace,
        )
        return envelope, trace
    finally:
        conn.close()


def save_answer(
    envelope: AnswerEnvelope,
    *,
    slug: str,
    root: Path | str | None = None,
    run_id: str | None = None,
) -> Path:
    """Persist ``envelope`` to the context-correct path (plan §4).

    Ad-hoc ``ask`` → ``projects/{slug}/answers/{answer_id}.json``; inside a build run
    (``run_id`` set) → ``projects/{slug}/runs/{run_id}/answers/{answer_id}.json``.
    """
    if run_id:
        base = layout.project_runs_dir(slug, root) / run_id / "answers"
    else:
        base = layout.project_dir(slug, root) / "answers"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{envelope.answer_id}.json"
    fd, tmp = tempfile.mkstemp(dir=str(base), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(envelope.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path
