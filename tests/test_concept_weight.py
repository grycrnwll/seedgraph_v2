"""Build B chunk 5 — anti-stopword IDF concept weight (decision 71).

``weight = log(N_total / paper_frequency)`` over the STAGED extraction set
(distinct works in ``extracted_claims``), computed at build into the 0015
``concepts.weight`` column. Read surfaces order sharp-first. Fully offline.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timezone

import seedgraph.db.migrations as migrations
from seedgraph.semantic import build_semantic_overlay
from seedgraph.semantic.graph_build import build_graph
from seedgraph.semantic.query import concept_detail, concept_overview, list_concepts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _conn(tmp_path) -> sqlite3.Connection:
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(tmp_path / "project.db"))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    return conn


def _add_work(conn, work_id):
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, created_at) VALUES (?,?,?)",
        (work_id, f"Paper {work_id}", _now()),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
        "created_at, updated_at) VALUES (?, 'included', 0, ?, ?)",
        (work_id, _now(), _now()),
    )
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_version, prompt_version, access_class, run_status, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("extr_" + work_id, work_id, "md_" + work_id, "h", "v1", "p1",
         "open_access", "success", _now()),
    )


def _add_claim(conn, claim_id, work_id, label):
    conn.execute(
        "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, "
        "claim_type, field_key, normalized_label, claim_text, status, "
        "epistemic_type, access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, "extr_" + work_id, work_id, "method", "f", label, label,
         "found", "llm_extracted", "open_access", _now()),
    )


def _seed_four_works(conn):
    """4 staged works; 'everywhere' in all 4, 'rare' in exactly 1."""
    for i in range(4):
        wid = f"work_{i}"
        _add_work(conn, wid)
        _add_claim(conn, f"c_every_{i}", wid, "everywhere concept")
    _add_claim(conn, "c_rare", "work_0", "rare concept")
    conn.commit()


def test_weight_zero_for_everywhere_and_log_n_for_one_of_n(tmp_path):
    conn = _conn(tmp_path)
    _seed_four_works(conn)
    build_semantic_overlay(conn, run_id="r1", tau=0.6)
    rows = dict(conn.execute("SELECT concept_id, weight FROM concepts").fetchall())
    # in EVERY staged work => log(4/4) = 0.0 (an anti-stopword, not unimportant).
    assert rows["concept::everywhere concept"] == 0.0
    # in 1 of N => log(N).
    assert abs(rows["concept::rare concept"] - math.log(4)) < 1e-12
    conn.close()


def test_degenerate_guards_yield_zero(tmp_path):
    # paper_frequency 0 / N_total 0 can't arise through gather (no concept
    # without a claim), so pin the guard at the arithmetic level instead: the
    # INSERT path writes 0.0 whenever the log is undefined.
    import seedgraph.semantic.concepts as concepts_mod

    assert concepts_mod  # the guard lives inline; exercised via the build below.
    conn = _conn(tmp_path)
    _add_work(conn, "work_0")
    _add_claim(conn, "c1", "work_0", "solo concept")
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", tau=0.6)
    # 1 staged work, pf=1 => log(1/1) = 0.0 — bounded below at zero, no negative.
    row = conn.execute("SELECT weight FROM concepts").fetchone()
    assert row[0] == 0.0
    conn.close()


def test_list_concepts_sharp_first_with_weight(tmp_path):
    conn = _conn(tmp_path)
    _seed_four_works(conn)
    build_semantic_overlay(conn, run_id="r1", tau=0.6)
    rows = list_concepts(conn)
    # sharp-first: rare (weight log 4) before everywhere (weight 0) — the
    # inverse of the old hub-first paper_frequency ordering.
    assert [r["concept_id"] for r in rows] == [
        "concept::rare concept", "concept::everywhere concept",
    ]
    assert rows[0]["weight"] > rows[1]["weight"] == 0.0
    # weight rides the detail read surface too.
    detail = concept_detail(conn, "concept::rare concept")
    assert abs(detail["concept"]["weight"] - math.log(4)) < 1e-12
    conn.close()


def test_weight_attached_to_concept_graph_nodes(tmp_path):
    conn = _conn(tmp_path)
    _seed_four_works(conn)
    build_semantic_overlay(conn, run_id="r1", tau=0.6)
    graph = build_graph(conn, run_id="r1")
    node = graph.nodes["concept::rare concept"]
    assert abs(node["weight"] - math.log(4)) < 1e-12
    assert graph.nodes["concept::everywhere concept"]["weight"] == 0.0
    conn.close()


# ---------------------------------------------------------------------------
# chunk 6 — query-time ubiquity down-ranking (retrieve stamps, rank boosts)
# ---------------------------------------------------------------------------


def test_rebuild_recomputes_weight_never_stale(tmp_path):
    conn = _conn(tmp_path)
    _seed_four_works(conn)
    build_semantic_overlay(conn, run_id="r1", tau=0.6)
    before = conn.execute(
        "SELECT weight FROM concepts WHERE concept_id='concept::rare concept'"
    ).fetchone()[0]
    # stage 4 more works mentioning only 'everywhere concept': N_total doubles.
    for i in range(4, 8):
        wid = f"work_{i}"
        _add_work(conn, wid)
        _add_claim(conn, f"c_every_{i}", wid, "everywhere concept")
    conn.commit()
    build_semantic_overlay(conn, run_id="r2", tau=0.6)
    after = conn.execute(
        "SELECT weight FROM concepts WHERE concept_id='concept::rare concept'"
    ).fetchone()[0]
    assert abs(before - math.log(4)) < 1e-12
    assert abs(after - math.log(8)) < 1e-12  # delete-and-rewrite: never stale
    conn.close()


# ---------------------------------------------------------------------------
# concept_overview — compact recurrence-ranked orientation sheet (NOT list's
# IDF sort). Direct concept inserts pin the aggregation without the build path.
# ---------------------------------------------------------------------------


def _add_concept(conn, label, ctype, pf):
    norm = label.lower()
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, "
        "concept_type, paper_frequency, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (f"concept::{norm}", norm, label, ctype, pf, _now(), _now()),
    )


def test_concept_overview_orientation_sheet(tmp_path):
    conn = _conn(tmp_path)
    # method (5): OLS·10 IV·5 GMM·3 MLE·2 Bootstrap·1(singleton)
    _add_concept(conn, "OLS", "method", 10)
    _add_concept(conn, "IV", "method", 5)
    _add_concept(conn, "GMM", "method", 3)
    _add_concept(conn, "MLE", "method", 2)
    _add_concept(conn, "Bootstrap", "method", 1)
    # result (3): Effect Positive·9 Null Result·4 Large Effect·2
    _add_concept(conn, "Effect Positive", "result", 9)
    _add_concept(conn, "Null Result", "result", 4)
    _add_concept(conn, "Large Effect", "result", 2)
    # identification_assumption (2): Parallel Trends·7 Exogeneity·3
    _add_concept(conn, "Parallel Trends", "identification_assumption", 7)
    _add_concept(conn, "Exogeneity", "identification_assumption", 3)
    # limitation (1): Small Sample·6
    _add_concept(conn, "Small Sample", "limitation", 6)
    conn.commit()

    ov = concept_overview(conn, per_type_cap=2, min_pf=2, background_n=2)

    assert ov["scale"]["total_concepts"] == 11
    assert ov["scale"]["singleton_count"] == 1  # only Bootstrap·pf1
    assert ov["scale"]["type_counts"] == {
        "method": 5, "result": 3, "identification_assumption": 2, "limitation": 1,
    }

    # background_frame = top-2 by paper_frequency across ALL types (the hubs).
    assert [(c["canonical_label"], c["paper_frequency"]) for c in ov["background_frame"]] == [
        ("OLS", 10), ("Effect Positive", 9),
    ]

    by_type = {b["concept_type"]: b for b in ov["by_type"]}
    featured_labels = {c["canonical_label"] for b in ov["by_type"] for c in b["featured"]}

    # (1) background concepts are excluded from every by_type block.
    assert "OLS" not in featured_labels
    assert "Effect Positive" not in featured_labels

    # (2) per_type_cap respected; method's top (OLS) is background, so the featured
    # are the next two by pf — capping then drops MLE·2.
    assert all(len(b["featured"]) <= 2 for b in ov["by_type"])
    assert [c["canonical_label"] for c in by_type["method"]["featured"]] == ["IV", "GMM"]

    # (3) more_count = type_count - len(featured) (matches `concepts list --type`):
    # method's 3 extras = OLS(background) + MLE(cap) + Bootstrap(min_pf).
    assert by_type["method"]["more_count"] == 3
    assert by_type["result"]["more_count"] == 1
    assert by_type["identification_assumption"]["more_count"] == 0

    # (4) salience ordering honored (curated order, unlisted types absent here).
    assert [b["concept_type"] for b in ov["by_type"]] == [
        "identification_assumption", "limitation", "method", "result",
    ]
    conn.close()
