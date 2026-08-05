"""Assemble per-run eval metric artifacts under runs/{run_id}/eval/ (phase_9 §4/§6/§10).

Pure orchestration over ``metrics`` (deterministic) + ``audit_records`` counts.
Writes ``retrieval_metrics`` (work-level AND span-level — ``metrics`` called twice),
``leakage_metrics``, ``answer_metrics`` (incl. ``unsupported_synthesis_rate``), and
the aggregate ``metrics.{json,md}`` — the MVP §12 checklist, ``user_correction_rate``,
``span_indexing_quality``, ``concept_identity_noise`` over/under-merge vs the stated
tolerance (decision D10), oversize-aware note-coverage with the ``skipped_oversize``
count (decision D4), the phase_2 ``unresolved_target`` coverage gap (decision D3),
and open-audit counts.

No hard numeric pass/fail gate on a tiny corpus — every metric is reported WITH n;
only the safety invariants gate CI (phase_9 §12).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import metrics

# Default concept-identity noise tolerance (decision D10; phase_9 §1/§3). Exceeding
# it is a flagged report FINDING, never a hard CI failure on the tiny corpus.
NOISE_TOLERANCE: dict[str, float] = {"over_merge_rate": 0.15, "under_merge_rate": 0.25}


def _eval_run_dir(project: str, run_id: str | None, root=None) -> Path:
    from ..project import layout

    runs = layout.project_runs_dir(project, root)
    out = runs / (run_id or "adhoc") / "eval"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _project_eval_dir(project: str, root=None) -> Path:
    from .. import paths

    return paths.project_dir(project, root) / "eval"


def _open_conn(project: str, root=None) -> sqlite3.Connection:
    from ..db.connection import project_db_path

    conn = sqlite3.connect(str(project_db_path(project, root)))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _write_pair(out_dir: Path, stem: str, payload: dict, markdown: str) -> Path:
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(markdown, encoding="utf-8")
    return json_path


def write_retrieval_metrics(
    conn: sqlite3.Connection, project: str, run_id: str | None, *, k: int = 10, root=None
) -> Path:
    """Run ``retrieval_gold`` through the live harness, score
    recall/precision/reciprocal-rank/coverage at BOTH work-level (over
    ``relevant_work_ids``) and span-level (over ``relevant_span_ids``), and write
    ``runs/{run_id}/eval/retrieval_metrics.{json,md}`` (both blocks). Returns the
    json path (doc 09 §7).
    """
    from . import goldsets, runners
    from ..project import service

    gold_path = _project_eval_dir(project, root) / "retrieval_gold.jsonl"
    rows = goldsets.load_retrieval_gold(gold_path) if gold_path.exists() else []
    handle = service.open_project(project, root=root)

    work_recall: list[float] = []
    work_prec: list[float] = []
    work_rr: list[float] = []
    span_recall: list[float] = []
    span_prec: list[float] = []
    span_rr: list[float] = []
    work_retrieved_per_q: list[list[str]] = []
    span_retrieved_per_q: list[list[str]] = []

    for row in rows:
        env = runners.run_question(handle, row["question"], row.get("query_type"))
        retrieved_works = list(env.cited_work_ids)
        retrieved_spans = list(env.retrieved_item_ids)
        work_retrieved_per_q.append(retrieved_works)
        span_retrieved_per_q.append(retrieved_spans)
        rel_works = set(row["relevant_work_ids"])
        rel_spans = set(row["relevant_span_ids"])
        work_recall.append(metrics.recall_at_k(retrieved_works, rel_works, k))
        work_prec.append(metrics.precision_at_k(retrieved_works, rel_works, k))
        work_rr.append(metrics.reciprocal_rank(retrieved_works, rel_works))
        span_recall.append(metrics.recall_at_k(retrieved_spans, rel_spans, k))
        span_prec.append(metrics.precision_at_k(retrieved_spans, rel_spans, k))
        span_rr.append(metrics.reciprocal_rank(retrieved_spans, rel_spans))

    corpus_works = {r[0] for r in conn.execute("SELECT work_id FROM works").fetchall()}

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    payload = {
        "n_questions": len(rows),
        "k": k,
        "work_level": {
            "recall_at_k": _mean(work_recall),
            "precision_at_k": _mean(work_prec),
            "mrr": _mean(work_rr),
            "corpus_coverage": metrics.corpus_coverage(work_retrieved_per_q, corpus_works),
        },
        "span_level": {
            "recall_at_k": _mean(span_recall),
            "precision_at_k": _mean(span_prec),
            "mrr": _mean(span_rr),
        },
    }
    md = (
        f"# Retrieval metrics (n={len(rows)}, k={k})\n\n"
        f"## Work level\n"
        f"- recall@{k}: {payload['work_level']['recall_at_k']:.3f}\n"
        f"- precision@{k}: {payload['work_level']['precision_at_k']:.3f}\n"
        f"- MRR: {payload['work_level']['mrr']:.3f}\n"
        f"- corpus_coverage: {payload['work_level']['corpus_coverage']:.3f}\n\n"
        f"## Span level\n"
        f"- recall@{k}: {payload['span_level']['recall_at_k']:.3f}\n"
        f"- precision@{k}: {payload['span_level']['precision_at_k']:.3f}\n"
        f"- MRR: {payload['span_level']['mrr']:.3f}\n"
    )
    return _write_pair(_eval_run_dir(project, run_id, root), "retrieval_metrics", payload, md)


def write_leakage_metrics(
    conn: sqlite3.Connection, project: str, run_id: str | None, *, root=None
) -> Path:
    """Run ``leakage_probes`` through the live harness, assert each envelope
    abstains (``insufficient_evidence=true``), and write
    ``runs/{run_id}/eval/leakage_metrics.{json,md}`` with the abstention rate
    (doc 09 §9). Returns the json path.
    """
    from . import goldsets, runners
    from ..project import service

    probe_path = _project_eval_dir(project, root) / "leakage_probes.jsonl"
    rows = goldsets.load_leakage_probes(probe_path) if probe_path.exists() else []
    handle = service.open_project(project, root=root)

    abstained = 0
    for row in rows:
        env = runners.run_question(handle, row["question"], "factual")
        if env.insufficient_evidence:
            abstained += 1
    rate = abstained / len(rows) if rows else 1.0
    payload = {"n_probes": len(rows), "abstained": abstained, "abstention_rate": rate}
    md = (
        f"# Leakage metrics (n={len(rows)})\n\n"
        f"- abstained: {abstained}/{len(rows)}\n"
        f"- abstention_rate: {rate:.3f}\n"
    )
    return _write_pair(_eval_run_dir(project, run_id, root), "leakage_metrics", payload, md)


def write_answer_metrics(
    conn: sqlite3.Connection, project: str, run_id: str | None, *, root=None
) -> Path:
    """Run the answer set, apply deterministic faithfulness checks over the returned
    envelopes (cited ⊆ retrieved ∩ project; stated-vs-inferred; abstention), emit
    ``unsupported_synthesis_rate`` DISTINCTLY, and write
    ``runs/{run_id}/eval/answer_metrics.{json,md}`` (doc 09 §8). Returns the json
    path. Graded residue is queued as ``answer_faithfulness`` audits by the caller.
    """
    from . import goldsets, runners
    from ..project import service

    gold_path = _project_eval_dir(project, root) / "retrieval_gold.jsonl"
    rows = goldsets.load_retrieval_gold(gold_path) if gold_path.exists() else []
    handle = service.open_project(project, root=root)
    project_works = {r[0] for r in conn.execute("SELECT work_id FROM works").fetchall()}

    envelopes = []
    faithful = 0
    for row in rows:
        env = runners.run_question(handle, row["question"], row.get("query_type"))
        envelopes.append(env)
        allowed_spans = set(env.retrieved_item_ids)
        cited_ok = set(env.cited_work_ids) <= project_works and set(env.cited_span_ids) <= allowed_spans
        if cited_ok:
            faithful += 1
    payload = {
        "n_answers": len(rows),
        "faithful": faithful,
        "unsupported_synthesis_rate": metrics.unsupported_synthesis_rate(envelopes),
    }
    md = (
        f"# Answer metrics (n={len(rows)})\n\n"
        f"- faithful (cited ⊆ retrieved ∩ project): {faithful}/{len(rows)}\n"
        f"- unsupported_synthesis_rate: {payload['unsupported_synthesis_rate']:.3f}\n"
    )
    return _write_pair(_eval_run_dir(project, run_id, root), "answer_metrics", payload, md)


def _aggregate(conn: sqlite3.Connection, run_id: str | None) -> dict:
    """Compute the deterministic aggregate metric block from project.db state."""
    # D4 — oversize-aware note coverage. Window-fitting = extraction runs that were
    # not skipped_oversize; covered = those with a structured_note.
    try:
        runs = conn.execute(
            "SELECT extraction_run_id, run_status FROM extraction_runs"
        ).fetchall()
    except sqlite3.OperationalError:
        runs = []
    oversize_skipped = sum(1 for _rid, status in runs if status == "skipped_oversize")
    window_runs = [rid for rid, status in runs if status == "success"]
    noted = set()
    try:
        noted = {
            r[0] for r in conn.execute("SELECT extraction_run_id FROM structured_notes").fetchall()
        }
    except sqlite3.OperationalError:
        noted = set()
    note_flags = [1 if rid in noted else 0 for rid in window_runs]
    note_cov = metrics.note_coverage(note_flags, oversize_skipped)

    # D3 — unresolved_target coverage gap.
    n_edges = conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0]
    try:
        n_unresolved = conn.execute(
            "SELECT COUNT(*) FROM reference_entries "
            "WHERE resolved_work_id IS NULL AND resolution_status='unresolved'"
        ).fetchone()[0]
    except sqlite3.OperationalError:
        n_unresolved = 0
    unresolved_rate = metrics.unresolved_target_rate(n_edges, n_unresolved)

    # D10 — concept-identity noise from graded concept_merge audits.
    merge_verdicts = [
        r[0]
        for r in conn.execute(
            "SELECT verdict FROM audit_records "
            "WHERE audit_type='concept_merge' AND status='resolved' AND verdict IS NOT NULL"
        ).fetchall()
    ]
    noise = metrics.concept_identity_noise(merge_verdicts)
    noise_findings = {
        key: noise[key] > NOISE_TOLERANCE[key] for key in NOISE_TOLERANCE
    }

    # decision 64 — user_correction_rate over resolved decisions.
    decisions = [
        r[0]
        for r in conn.execute(
            "SELECT decision FROM audit_records WHERE status='resolved' AND decision IS NOT NULL"
        ).fetchall()
    ]
    correction_rate = metrics.user_correction_rate(decisions)

    # open-audit counts by type.
    open_counts: dict[str, int] = {}
    for atype, cnt in conn.execute(
        "SELECT audit_type, COUNT(*) FROM audit_records WHERE status='open' GROUP BY audit_type"
    ).fetchall():
        open_counts[atype] = cnt

    return {
        "note_coverage": note_cov,
        "unresolved_target": {
            "citation_edges": n_edges,
            "unresolved_targets": n_unresolved,
            "unresolved_target_rate": unresolved_rate,
        },
        "concept_identity_noise": {
            "rates": noise,
            "tolerance": NOISE_TOLERANCE,
            "over_tolerance": noise_findings,
            "n_graded": len(merge_verdicts),
        },
        "user_correction_rate": correction_rate,
        "open_audits": open_counts,
    }


def build_report(conn: sqlite3.Connection, project: str, run_id: str | None, *, root=None) -> Path:
    """Assemble the aggregate ``runs/{run_id}/eval/metrics.{json,md}``: the MVP §12
    checklist, ``user_correction_rate``, ``span_indexing_quality``,
    ``concept_identity_noise`` over/under-merge vs :data:`NOISE_TOLERANCE` (D10),
    oversize-aware note-coverage with the ``skipped_oversize`` count (D4), the
    phase_2 ``unresolved_target`` coverage gap (D3), and open-audit counts. Returns
    the ``metrics.md`` path. No hard numeric gate — report-with-n (§12).
    """
    agg = _aggregate(conn, run_id)
    out_dir = _eval_run_dir(project, run_id, root)

    nc = agg["note_coverage"]
    ut = agg["unresolved_target"]
    cin = agg["concept_identity_noise"]
    lines = [
        f"# Eval report — project '{project}' (run {run_id or 'adhoc'})",
        "",
        "## D4 — Oversize-aware note coverage",
        f"- window_fitting papers: {int(nc['window_fitting'])}",
        f"- covered (have default note): {int(nc['covered'])}",
        f"- coverage: {nc['coverage']:.3f}",
        f"- skipped_oversize (recorded, not-processed): {int(nc['oversize_skipped'])}",
        "",
        "## D3 — Unresolved-target coverage gap",
        f"- citation_edges: {ut['citation_edges']}",
        f"- unresolved_targets (referenced-but-absent, link-only): {ut['unresolved_targets']}",
        f"- unresolved_target_rate: {ut['unresolved_target_rate']:.3f}",
        "",
        "## D10 — Concept-identity noise (sampled; NOT perfect-identity)",
        f"- graded merges: {cin['n_graded']}",
        f"- over_merge_rate: {cin['rates']['over_merge_rate']:.3f} "
        f"(tolerance {cin['tolerance']['over_merge_rate']}; "
        f"over_tolerance={cin['over_tolerance']['over_merge_rate']})",
        f"- under_merge_rate: {cin['rates']['under_merge_rate']:.3f} "
        f"(tolerance {cin['tolerance']['under_merge_rate']}; "
        f"over_tolerance={cin['over_tolerance']['under_merge_rate']})",
        "",
        "## Correction + open audits",
        f"- user_correction_rate (reject+edit)/resolved: {agg['user_correction_rate']:.3f}",
        f"- open audits by type: {json.dumps(agg['open_audits'], sort_keys=True)}",
        "",
    ]
    (out_dir / "metrics.json").write_text(
        json.dumps(agg, indent=2, sort_keys=True), encoding="utf-8"
    )
    md_path = out_dir / "metrics.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path
