"""Phase 7 — Semantic graph overlay acceptance tests (plan §11).

Every test mirrors a named acceptance/unit test from phase_7 plan §11 (plus the
§10-step-2 ORM-parity test and the §11 eval-hook). Fully offline: the LLM seam is
a NullProposer / in-test mock proposer — no network, no key, no cache.db read from
the concept layer.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect

import seedgraph.db.migrations as migrations
from seedgraph import semantic
from seedgraph.db.models_project import (
    Concept,
    ConceptAlias,
    ConceptConstraint,
    ClaimConcept,
    EdgeSpan,
    ProjectGraphEdge,
)
from seedgraph.errors import ValidationError
from seedgraph.semantic import (
    access,
    canon,
    concepts,
    edges,
    export,
    graph_build,
    llm_propose,
    merge,
)
from seedgraph.semantic import build_semantic_overlay
from seedgraph.semantic.query import concept_detail, list_concepts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Offline seed helpers
# --------------------------------------------------------------------------


def _migrated_conn(tmp_path) -> sqlite3.Connection:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "project.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    return conn


def _add_work(conn, work_id, title="Paper"):
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, created_at) VALUES (?,?,?)",
        (work_id, title, _now()),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, created_at, "
        "updated_at) VALUES (?,?,?,?,?)",
        (work_id, "included", 0, _now(), _now()),
    )


def _add_run(conn, run_id, work_id, access_class="open_access"):
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_version, prompt_version, access_class, run_status, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (run_id, work_id, "md_" + work_id, "h", "v1", "p1", access_class, "success", _now()),
    )


def _add_claim(conn, claim_id, extr_id, work_id, claim_type, label, text=None,
               access_class="open_access"):
    conn.execute(
        "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, claim_type, "
        "field_key, normalized_label, claim_text, status, epistemic_type, access_class, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, extr_id, work_id, claim_type, "f", label, text, "found",
         "llm_extracted", access_class, _now()),
    )


def _add_span(conn, span_id, work_id, quote, access_class="open_access"):
    conn.execute(
        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, work_id, start_char, end_char, exact_quote, quote_hash, "
        "access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (span_id, "md_" + work_id, "h", "sf", "sfh", work_id, 0, len(quote), quote,
         "qh_" + span_id, access_class, _now()),
    )


def _link_claim_span(conn, claim_id, span_id):
    conn.execute(
        "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?,?,0,?)",
        (claim_id, span_id, _now()),
    )


def _seed_basic(conn):
    """Two works; acronym fold + an over-merge pair; one span on the MLM claim."""
    _add_work(conn, "work_a", "Paper A")
    _add_work(conn, "work_b", "Paper B")
    _add_run(conn, "extr_a", "work_a")
    _add_run(conn, "extr_b", "work_b")
    _add_claim(conn, "claim_1", "extr_a", "work_a", "method", "MLM", "We pretrain with MLM.")
    _add_claim(conn, "claim_2", "extr_b", "work_b", "method", "Masked Language Modeling",
               "Masked Language Modeling objective.")
    _add_claim(conn, "claim_3", "extr_a", "work_a", "method", "BERT-base", "BERT-base.")
    _add_claim(conn, "claim_4", "extr_b", "work_b", "method", "BERT-large", "BERT-large.")
    _add_span(conn, "span_1", "work_a", "We pretrain with MLM.")
    _link_claim_span(conn, "claim_1", "span_1")
    conn.commit()


class _FoldProposer:
    """In-test mock proposer: returns the given normalized-key subsets when their
    members are all present in the offered cluster. Never touches a network/SDK."""

    def __init__(self, subsets):
        self._subsets = subsets

    def propose(self, cluster):
        cl = set(cluster)
        return [list(s) for s in self._subsets if set(s) <= cl]


# --------------------------------------------------------------------------
# Unit (plan §11)
# --------------------------------------------------------------------------


def test_concept_normalize():
    """normalize/concept_id determinism + NFC stability + acronym detection."""
    assert merge.normalize("BERT-base") == "bert base"
    assert merge.normalize("  SQuAD   v2.0 ") == "squad v2 0"
    # determinism: same surface form -> same merge_key -> same deterministic id.
    assert merge.normalize("Transformer") == merge.normalize("transformer")
    cid = merge.concept_id(merge.normalize("Transformer"))
    assert cid == "concept::transformer"
    assert merge.concept_id(merge.normalize("Transformer")) == cid
    # NFC stability: composed vs decomposed accents normalize identically.
    import unicodedata

    composed = "résumé"
    decomposed = unicodedata.normalize("NFD", composed)
    assert merge.normalize(composed) == merge.normalize(decomposed)
    # acronym detection.
    assert merge.is_acronym_expansion("MLM", "Masked Language Modeling")
    assert merge.is_acronym_expansion("mlm", "masked language modeling")
    assert not merge.is_acronym_expansion("BERT-base", "BERT-large")


def test_concept_canon():
    """candidate_clusters groups by token-Jaccard >= tau OR acronym; the closure
    guardrail rejects invented umbrella labels not present in the corpus."""
    labels = ["mlm", "masked language modeling", "graph neural network",
              "graph neural networks"]
    clusters = canon.candidate_clusters(labels, 0.5)
    as_sets = {frozenset(c) for c in clusters}
    assert frozenset({"mlm", "masked language modeling"}) in as_sets  # acronym join
    assert frozenset({"graph neural network", "graph neural networks"}) in as_sets  # jaccard

    present = set(labels)
    # closure: an invented umbrella not present in the corpus is rejected.
    assert not canon.passes_guardrails(
        "neural networks", frozenset({"graph neural network", "graph neural networks"}),
        present,
    )
    # a present canonical with a real (acronym) relationship passes the guardrails.
    assert canon.passes_guardrails(
        "masked language modeling",
        frozenset({"mlm", "masked language modeling"}),
        present,
    )


def test_concept_overmerge_guard(tmp_path):
    """Anti-overmerge guard HOLDS (D10): BERT-base != BERT-large, SQuAD v1.1 !=
    SQuAD v2.0; acronym MLM <-> Masked Language Modeling folds; absent-LLM => split,
    never over-merge."""
    present = {"bert base", "bert large", "squad v1 1", "squad v2 0",
               "mlm", "masked language modeling",
               "pre trained language model", "pre trained language models"}
    # discriminating-token veto ALWAYS wins.
    assert not canon.passes_guardrails("bert base", frozenset({"bert base", "bert large"}), present)
    assert not canon.passes_guardrails("squad v1 1", frozenset({"squad v1 1", "squad v2 0"}), present)

    # Bare-numeric veto (D8): dot-free version pairs are vetoed post-normalization
    # — merge.normalize strips dots, so "SQuAD 1.1"/"SQuAD 2.0" become bare-digit
    # token labels and the bare numerics in DISCRIMINATING_TOKENS must catch them.
    squad_11 = merge.normalize("SQuAD 1.1")
    squad_20 = merge.normalize("SQuAD 2.0")
    assert squad_11 == "squad 1 1" and squad_20 == "squad 2 0"
    present_dotfree = present | {squad_11, squad_20}
    assert not canon.passes_guardrails(
        squad_11, frozenset({squad_11, squad_20}), present_dotfree
    )
    # A high-overlap pair (>=5 shared tokens, Jaccard 6/8 = 0.75 >= tau) differing
    # ONLY by a bare digit: it co-clusters at tau=0.6, so the veto — not the
    # token-overlap precondition — must be what keeps the pair split.
    task1 = "general language understanding evaluation benchmark task 1"
    task2 = "general language understanding evaluation benchmark task 2"
    assert merge.token_jaccard(task1, task2) >= 0.6
    digit_clusters = {frozenset(c) for c in canon.candidate_clusters([task1, task2], 0.6)}
    assert frozenset({task1, task2}) in digit_clusters
    assert not canon.passes_guardrails(
        task1, frozenset({task1, task2}), present | {task1, task2}
    )
    # they do not even cluster (jaccard below tau, no acronym).
    clusters = {frozenset(c) for c in canon.candidate_clusters(sorted(present), 0.6)}
    assert frozenset({"bert base"}) in clusters
    assert frozenset({"bert large"}) in clusters
    # acronym fold is permitted.
    assert canon.passes_guardrails(
        "masked language modeling", frozenset({"mlm", "masked language modeling"}), present
    )

    # End-to-end with NO LLM: the over-merge pair stays two concepts, and a
    # token-overlapping non-acronym pair is left SPLIT (under-merge, never over).
    conn = _migrated_conn(tmp_path)
    _add_work(conn, "work_a")
    _add_run(conn, "extr_a", "work_a")
    _add_claim(conn, "c1", "extr_a", "work_a", "method", "BERT-base")
    _add_claim(conn, "c2", "extr_a", "work_a", "method", "BERT-large")
    _add_claim(conn, "c3", "extr_a", "work_a", "method", "pre trained language model")
    _add_claim(conn, "c4", "extr_a", "work_a", "method", "pre trained language models")
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    ids = {r[0] for r in conn.execute("SELECT concept_id FROM concepts")}
    assert "concept::bert base" in ids and "concept::bert large" in ids
    assert "concept::pre trained language model" in ids
    assert "concept::pre trained language models" in ids
    conn.close()


def test_concept_merge(tmp_path):
    """relabel-then-one-merge; field-typed-label-wins; merged_from -> concept_aliases;
    paper_frequency = distinct works; concept_type drawn from ClaimType."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    # field-typed: a generic mention loses to the field-typed surface form.
    _add_claim(conn, "claim_5", "extr_a", "work_a", "method", "BERT")
    _add_claim(conn, "claim_6", "extr_a", "work_a", "background", "bert")
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)

    # relabel-then-one-merge: the acronym folds under the expanded canonical id.
    row = conn.execute(
        "SELECT canonical_label, concept_type, paper_frequency FROM concepts "
        "WHERE concept_id='concept::masked language modeling'"
    ).fetchone()
    assert row is not None
    assert row[0] == "Masked Language Modeling"  # field-typed/longest surface wins
    assert row[1] == "method"
    from seedgraph.vocab import CLAIM_TYPES

    assert row[1] in CLAIM_TYPES  # concept_type drawn from ClaimType vocab
    assert row[2] == 2  # distinct works
    assert conn.execute(
        "SELECT 1 FROM concepts WHERE concept_id='concept::mlm'"
    ).fetchone() is None
    # merged_from surface lands in concept_aliases.
    aliases = {
        r[0]
        for r in conn.execute(
            "SELECT alias_label FROM concept_aliases "
            "WHERE concept_id='concept::masked language modeling'"
        )
    }
    assert "MLM" in aliases

    # field-typed-label-wins: 'BERT' (method) beats generic 'bert' (background).
    bert = conn.execute(
        "SELECT canonical_label FROM concepts WHERE concept_id='concept::bert'"
    ).fetchone()
    assert bert is not None and bert[0] == "BERT"
    conn.close()


def _seed_acronym_pair(conn, type_a, type_b):
    """Two works; MLM / Masked Language Modeling acronym pair with given types."""
    _add_work(conn, "work_a")
    _add_work(conn, "work_b")
    _add_run(conn, "extr_a", "work_a")
    _add_run(conn, "extr_b", "work_b")
    _add_claim(conn, "c1", "extr_a", "work_a", type_a, "MLM")
    _add_claim(conn, "c2", "extr_b", "work_b", type_b, "Masked Language Modeling")
    conn.commit()


def _fence_snapshot(conn):
    """run_id-free dump of the concept overlay for determinism comparison."""
    return (
        conn.execute(
            "SELECT concept_id, normalized_label, canonical_label, concept_type, "
            "paper_frequency, status, epistemic_type, access_class "
            "FROM concepts ORDER BY concept_id"
        ).fetchall(),
        conn.execute(
            "SELECT concept_id, alias_label, fold_reason FROM concept_aliases "
            "ORDER BY concept_id, alias_label"
        ).fetchall(),
        conn.execute(
            "SELECT claim_id, concept_id, work_id FROM claim_concepts "
            "ORDER BY claim_id, concept_id"
        ).fetchall(),
    )


def test_concept_type_fence(tmp_path):
    """Type-scoped canonicalization fence (D9, Build C chunk 8): an acronym pair
    whose majority concept_types DIFFER does not auto-fold (the fixed regression
    — pre-fence it silently folded cross-type with no review); the same pair with
    matching types still folds; a sticky cross-type must_link still merges (human
    decision outranks the fence); output is deterministic across two builds."""
    # --- cross-type acronym pair stays SPLIT (the regression pin) ----------
    conn = _migrated_conn(tmp_path / "fence")
    _seed_acronym_pair(conn, "method", "background")
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    ids = {r[0] for r in conn.execute("SELECT concept_id FROM concepts")}
    assert ids == {"concept::mlm", "concept::masked language modeling"}
    types = dict(
        conn.execute("SELECT concept_id, concept_type FROM concepts").fetchall()
    )
    assert types["concept::mlm"] == "method"
    assert types["concept::masked language modeling"] == "background"
    # deterministic across a rebuild (per-type cluster order is stable).
    snap1 = _fence_snapshot(conn)
    concepts.build_concepts(
        conn, run_id="r2", proposer=llm_propose.NullProposer(), tau=0.6
    )
    assert _fence_snapshot(conn) == snap1
    conn.close()

    # --- matching types: the acronym auto-fold still applies ----------------
    conn = _migrated_conn(tmp_path / "fold")
    _seed_acronym_pair(conn, "method", "method")
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    ids = {r[0] for r in conn.execute("SELECT concept_id FROM concepts")}
    assert ids == {"concept::masked language modeling"}
    aliases = {
        r[0]
        for r in conn.execute(
            "SELECT alias_label FROM concept_aliases "
            "WHERE concept_id='concept::masked language modeling'"
        )
    }
    assert "MLM" in aliases
    conn.close()

    # --- cross-type must_link still merges (human decision crosses the fence)
    conn = _migrated_conn(tmp_path / "mustlink")
    _seed_acronym_pair(conn, "method", "background")
    conn.execute(
        "INSERT INTO concept_constraints (kind, label_a, label_b, source, created_at) "
        "VALUES ('must_link', 'masked language modeling', 'mlm', 'user', ?)",
        (_now(),),
    )
    conn.commit()
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    ids = {r[0] for r in conn.execute("SELECT concept_id FROM concepts")}
    assert len(ids) == 1
    assert conn.execute(
        "SELECT COUNT(DISTINCT claim_id) FROM claim_concepts"
    ).fetchone()[0] == 2  # both claims fold into the one merged concept

    # --- deterministic output across two builds ------------------------------
    snap1 = _fence_snapshot(conn)
    concepts.build_concepts(
        conn, run_id="r2", proposer=llm_propose.NullProposer(), tau=0.6
    )
    assert _fence_snapshot(conn) == snap1
    conn.close()


def test_claim_concepts(tmp_path):
    """Every concept has >=1 claim_concepts mention; no orphan concept is created."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    orphans = conn.execute(
        "SELECT c.concept_id FROM concepts c "
        "LEFT JOIN claim_concepts cc ON cc.concept_id=c.concept_id "
        "WHERE cc.concept_id IS NULL"
    ).fetchall()
    assert orphans == []
    # and every claim_concepts row points at a real concept.
    dangling = conn.execute(
        "SELECT cc.concept_id FROM claim_concepts cc "
        "LEFT JOIN concepts c ON c.concept_id=cc.concept_id WHERE c.concept_id IS NULL"
    ).fetchall()
    assert dangling == []
    conn.close()


def test_project_graph_edges(tmp_path):
    """Polymorphic insert rejects dangling endpoint; DB CHECK rejects bad node_type /
    epistemic_type; run_id scopes the UNIQUE; co_occurs_with carries deterministic."""
    conn = _migrated_conn(tmp_path)
    _add_work(conn, "work_a")
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, concept_type, "
        "paper_frequency, status, epistemic_type, access_class, created_at, updated_at) "
        "VALUES ('concept::x','x','X','method',1,'auto','deterministic','open_access',?,?)",
        (_now(), _now()),
    )
    conn.commit()

    # dangling endpoint rejected at the single insert path.
    with pytest.raises(ValidationError):
        edges.insert_edge(conn, src=("Work", "work_missing"), tgt=("Concept", "concept::x"),
                          edge_type="discusses", epistemic_type="deterministic",
                          access_class="open_access", run_id="r1")
    with pytest.raises(ValidationError):
        edges.insert_edge(conn, src=("Work", "work_a"), tgt=("Concept", "concept::nope"),
                          edge_type="discusses", epistemic_type="deterministic",
                          access_class="open_access", run_id="r1")

    # DB CHECK rejects a bad node_type and a bad epistemic_type (closed vocab).
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO project_graph_edges (edge_id, source_node_type, source_node_id, "
            "target_node_type, target_node_id, edge_type, epistemic_type, access_class, "
            "created_at) VALUES ('pge_bad','Banana','x','Concept','concept::x','discusses',"
            "'deterministic','open_access',?)",
            (_now(),),
        )
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO project_graph_edges (edge_id, source_node_type, source_node_id, "
            "target_node_type, target_node_id, edge_type, epistemic_type, access_class, "
            "created_at) VALUES ('pge_bad2','Work','work_a','Concept','concept::x','discusses',"
            "'bogus','open_access',?)",
            (_now(),),
        )
    conn.rollback()

    # run_id scopes the UNIQUE: same tuple+run_id is idempotent; a new run_id is a new row.
    e1 = edges.insert_edge(conn, src=("Work", "work_a"), tgt=("Concept", "concept::x"),
                           edge_type="discusses", epistemic_type="deterministic",
                           access_class="open_access", run_id="r1")
    e1b = edges.insert_edge(conn, src=("Work", "work_a"), tgt=("Concept", "concept::x"),
                            edge_type="discusses", epistemic_type="deterministic",
                            access_class="open_access", run_id="r1")
    assert e1 == e1b
    e2 = edges.insert_edge(conn, src=("Work", "work_a"), tgt=("Concept", "concept::x"),
                           edge_type="discusses", epistemic_type="deterministic",
                           access_class="open_access", run_id="r2")
    assert e2 != e1
    n = conn.execute(
        "SELECT COUNT(*) FROM project_graph_edges WHERE edge_type='discusses'"
    ).fetchone()[0]
    assert n == 2
    conn.commit()

    # co_occurs_with carries deterministic on its distinct edge_type.
    conn.close()
    conn2 = _migrated_conn(tmp_path / "co")
    _add_work(conn2, "work_a")
    _add_work(conn2, "work_b")
    _add_run(conn2, "extr_a", "work_a")
    _add_run(conn2, "extr_b", "work_b")
    for wid, extr in (("work_a", "extr_a"), ("work_b", "extr_b")):
        _add_claim(conn2, f"al_{wid}", extr, wid, "method", "alpha concept")
        _add_claim(conn2, f"be_{wid}", extr, wid, "method", "beta concept")
    conn2.commit()
    build_semantic_overlay(conn2, run_id="r1", profile=None, tau=0.6)
    co = conn2.execute(
        "SELECT epistemic_type FROM project_graph_edges WHERE edge_type='co_occurs_with'"
    ).fetchall()
    assert co and all(r[0] == "deterministic" for r in co)
    conn2.close()


def test_co_occurs_strength_provenance(tmp_path):
    """Build B chunk 7 — co_occurs_with carries Jaccard confidence + shared_count;
    --min-shared thresholds; exported edges.csv columns ride along."""
    conn = _migrated_conn(tmp_path)
    # alpha in 3 works; beta in 2 of them => shared 2 / union 3 => jaccard 0.666…
    for i in range(3):
        wid = f"work_{i}"
        _add_work(conn, wid)
        _add_run(conn, f"extr_{wid}", wid)
        _add_claim(conn, f"al_{wid}", f"extr_{wid}", wid, "method", "alpha concept")
    for i in range(2):
        wid = f"work_{i}"
        _add_claim(conn, f"be_{wid}", f"extr_{wid}", wid, "method", "beta concept")
    conn.commit()
    report = build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    assert report.co_occurs_edges == 1
    conf, shared = conn.execute(
        "SELECT confidence, shared_count FROM project_graph_edges "
        "WHERE edge_type='co_occurs_with'"
    ).fetchone()
    assert shared == 2
    assert abs(conf - 2 / 3) < 1e-12

    # the strength rides the computed view + the CSV export columns.
    g = graph_build.build_graph(conn, run_id="r1")
    co = next(
        d for _u, _v, d in g.edges(data=True) if d.get("edge_type") == "co_occurs_with"
    )
    assert co["shared_count"] == 2 and abs(co["confidence"] - 2 / 3) < 1e-12
    written = export.export_graph(conn, slug="x", run_id="r1", fmt="csv",
                                  root=tmp_path / "h_csv")
    edges_csv = [p for p in written if p.name == "edges.csv"][0].read_text()
    header = edges_csv.splitlines()[0]
    assert header == "source,target,edge_type,epistemic_type,access_class,confidence,shared_count"
    co_row = next(l for l in edges_csv.splitlines() if "co_occurs_with" in l)
    assert ",2" in co_row and "0.666666" in co_row

    # min_shared=3 drops the 2-shared pair (idempotent rebuild clears first).
    report3 = build_semantic_overlay(conn, run_id="r2", profile=None, tau=0.6,
                                     min_shared=3)
    assert report3.co_occurs_edges == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM project_graph_edges WHERE edge_type='co_occurs_with'"
    ).fetchone()[0] == 0
    conn.close()


def test_concepts_build_min_shared_cli_knob(tmp_path):
    """`concepts build --min-shared 3` threads through to build_co_occurs_edges."""
    from typer.testing import CliRunner

    from seedgraph.cli import app as cli_app
    from seedgraph.project import service

    h = service.create_project("minshared")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    for i in range(2):
        wid = f"work_{i}"
        _add_work(conn, wid)
        _add_run(conn, f"extr_{wid}", wid)
        _add_claim(conn, f"al_{wid}", f"extr_{wid}", wid, "method", "alpha concept")
        _add_claim(conn, f"be_{wid}", f"extr_{wid}", wid, "method", "beta concept")
    conn.commit()
    conn.close()

    runner = CliRunner()
    # default threshold (2): the 2-shared pair survives.
    res = runner.invoke(cli_app, ["concepts", "build", "minshared", "--no-llm"])
    assert res.exit_code == 0, res.output
    assert "co_occurs=1" in res.output
    # --min-shared 3 drops it on the rebuild.
    res3 = runner.invoke(
        cli_app, ["concepts", "build", "minshared", "--no-llm", "--min-shared", "3"]
    )
    assert res3.exit_code == 0, res3.output
    assert "co_occurs=0" in res3.output


def _seed_borderline(conn):
    """Two near-synonyms that cluster (jaccard 0.6) but are not acronym-related."""
    _add_work(conn, "work_a")
    _add_work(conn, "work_b")
    _add_run(conn, "extr_a", "work_a")
    _add_run(conn, "extr_b", "work_b")
    _add_claim(conn, "c1", "extr_a", "work_a", "method", "pre trained language model")
    _add_claim(conn, "c2", "extr_b", "work_b", "method", "pre trained language models")
    conn.commit()


def test_concept_review(tmp_path):
    """Borderline LLM fold enqueues concept_merge_candidate (concepts stay SPLIT);
    approve writes alias + must_link + user_confirmed; reject writes cannot_link +
    user_split; both constraints are sticky (no flip) on the next build."""
    from seedgraph.project import review as review_mod
    from seedgraph.project import service

    # --- approve path -----------------------------------------------------
    h = service.create_project("rev-approve")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    _seed_borderline(conn)
    proposer = _FoldProposer([["pre trained language model", "pre trained language models"]])
    concepts.build_concepts(conn, run_id="r1", proposer=proposer, tau=0.6)
    conn.commit()
    # borderline fold enqueued; the two concepts stayed SPLIT.
    assert conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0] == 2
    item = conn.execute(
        "SELECT item_id, item_type FROM review_queue WHERE status='open'"
    ).fetchone()
    assert item is not None and item[1] == "concept_merge_candidate"
    conn.close()

    review_mod.resolve(h, item[0], "approve")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    must = conn.execute(
        "SELECT label_a, label_b FROM concept_constraints WHERE kind='must_link'"
    ).fetchall()
    assert must == [("pre trained language model", "pre trained language models")]
    confirmed = conn.execute(
        "SELECT concept_id FROM concepts WHERE status='user_confirmed'"
    ).fetchall()
    assert confirmed == [("concept::pre trained language models",)]
    alias = conn.execute(
        "SELECT alias_label FROM concept_aliases "
        "WHERE concept_id='concept::pre trained language models' "
        "AND alias_label='pre trained language model'"
    ).fetchone()
    assert alias is not None

    # sticky: the next build folds the pair into ONE concept and does not flip.
    concepts.build_concepts(conn, run_id="r2", proposer=llm_propose.NullProposer(), tau=0.6)
    conn.commit()
    rows = conn.execute("SELECT concept_id, status FROM concepts").fetchall()
    assert rows == [("concept::pre trained language models", "user_confirmed")]
    assert conn.execute(
        "SELECT COUNT(DISTINCT claim_id) FROM claim_concepts"
    ).fetchone()[0] == 2  # both claims fold into the one confirmed concept
    conn.close()

    # --- reject path ------------------------------------------------------
    h2 = service.create_project("rev-reject")
    conn = sqlite3.connect(str(h2.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    _seed_borderline(conn)
    concepts.build_concepts(conn, run_id="r1", proposer=proposer, tau=0.6)
    conn.commit()
    item2 = conn.execute("SELECT item_id FROM review_queue WHERE status='open'").fetchone()
    conn.close()
    review_mod.resolve(h2, item2[0], "reject")
    conn = sqlite3.connect(str(h2.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    cannot = conn.execute(
        "SELECT label_a, label_b FROM concept_constraints WHERE kind='cannot_link'"
    ).fetchall()
    assert cannot == [("pre trained language model", "pre trained language models")]
    split = {r[0] for r in conn.execute("SELECT concept_id FROM concepts WHERE status='user_split'")}
    assert split == {"concept::pre trained language model", "concept::pre trained language models"}
    # sticky: next build keeps them SPLIT (cannot_link, no flip), status preserved.
    concepts.build_concepts(conn, run_id="r2", proposer=proposer, tau=0.6)
    conn.commit()
    rows = conn.execute("SELECT concept_id, status FROM concepts ORDER BY concept_id").fetchall()
    assert rows == [
        ("concept::pre trained language model", "user_split"),
        ("concept::pre trained language models", "user_split"),
    ]
    conn.close()


def test_concept_edge_review(tmp_path):
    """An interpretive related_to edge enqueues concept_edge_candidate; approve flips
    the edge to user_validated; reject deletes the edge (UserValidation->Edge)."""
    from seedgraph.project import review as review_mod
    from seedgraph.project import service

    # approve -> user_validated
    h = service.create_project("edge-approve")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    _seed_basic(conn)
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    edges.build_interpretive_edges(
        conn, run_id="r1",
        proposals=[("concept::masked language modeling", "concept::bert base",
                    "related_to", 0.5)],
    )
    conn.commit()
    edge = conn.execute(
        "SELECT edge_id, epistemic_type FROM project_graph_edges WHERE edge_type='related_to'"
    ).fetchone()
    assert edge is not None and edge[1] == "llm_inferred"
    item = conn.execute(
        "SELECT item_id, item_type FROM review_queue WHERE status='open'"
    ).fetchone()
    assert item[1] == "concept_edge_candidate"
    conn.close()
    review_mod.resolve(h, item[0], "approve")
    conn = sqlite3.connect(str(h.db_path))
    assert conn.execute(
        "SELECT epistemic_type FROM project_graph_edges WHERE edge_id=?", (edge[0],)
    ).fetchone()[0] == "user_validated"
    conn.close()

    # reject -> edge deleted
    h2 = service.create_project("edge-reject")
    conn = sqlite3.connect(str(h2.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    _seed_basic(conn)
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    edges.build_interpretive_edges(
        conn, run_id="r1",
        proposals=[("concept::masked language modeling", "concept::bert base",
                    "related_to", 0.5)],
    )
    conn.commit()
    item2 = conn.execute("SELECT item_id FROM review_queue WHERE status='open'").fetchone()
    edge2 = conn.execute(
        "SELECT edge_id FROM project_graph_edges WHERE edge_type='related_to'"
    ).fetchone()[0]
    conn.close()
    review_mod.resolve(h2, item2[0], "reject")
    conn = sqlite3.connect(str(h2.db_path))
    assert conn.execute(
        "SELECT 1 FROM project_graph_edges WHERE edge_id=?", (edge2,)
    ).fetchone() is None
    conn.close()


def test_access_export_guard(tmp_path):
    """resolve_access_class reads ONLY extracted_claims.access_class (a mock DB with
    no source_files/cache.db proves no cross-db read); private concept WITHHELD by
    default export; unknown -> withheld; MISSING access_class column -> abort;
    --allow-private includes it; definition dropped on non-shareable concepts."""
    # 1) reads ONLY extracted_claims.access_class — a minimal DB with nothing else.
    mock = sqlite3.connect(":memory:")
    mock.execute("CREATE TABLE extracted_claims (claim_id TEXT, access_class TEXT)")
    mock.execute("INSERT INTO extracted_claims VALUES ('c1','open_access'),('c2','metadata_only')")
    assert access.resolve_access_class(mock, ["c1", "c2"]) == "metadata_only"
    # unknown / missing value -> fail-closed private.
    mock.execute("INSERT INTO extracted_claims VALUES ('c3','unknown')")
    assert access.resolve_access_class(mock, ["c1", "c3"]) == "user_supplied_private"
    mock.close()

    # 2) MISSING access_class column -> build aborts (no default-open).
    bad = sqlite3.connect(":memory:")
    bad.execute("CREATE TABLE extracted_claims (claim_id TEXT)")
    with pytest.raises(ValidationError):
        access.resolve_access_class(bad, ["c1"])
    bad.close()

    # 3) private concept withheld by default export; --allow-private includes it;
    #    definition dropped for the non-shareable concept even under allow-private.
    conn = _migrated_conn(tmp_path)
    _add_work(conn, "work_a")
    _add_run(conn, "extr_a", "work_a", access_class="user_supplied_private")
    _add_claim(conn, "c1", "extr_a", "work_a", "method", "private method",
               text="a private definition sentence", access_class="user_supplied_private")
    _add_run(conn, "extr_b", "work_a", access_class="open_access")
    _add_claim(conn, "c2", "extr_b", "work_a", "method", "open method",
               text="an open sentence", access_class="open_access")
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    # the private concept's denormalized access_class is private.
    assert conn.execute(
        "SELECT access_class FROM concepts WHERE concept_id='concept::private method'"
    ).fetchone()[0] == "user_supplied_private"

    written = export.export_graph(conn, slug="x", run_id="r1", fmt="json", allow_private=False,
                                  root=tmp_path / "home_default")
    default_doc = json.loads([p for p in written if p.name == "graph.json"][0].read_text())
    default_nodes = {n["id"] for n in default_doc["nodes"]}
    assert "concept::open method" in default_nodes
    assert "concept::private method" not in default_nodes  # withheld by default-deny

    written2 = export.export_graph(conn, slug="x", run_id="r1", fmt="json", allow_private=True,
                                   root=tmp_path / "home_private")
    # cross-cutting #2: graph.json stays PUBLIC-SAFE even under --allow-private — the
    # private concept is withheld from the run-dir graph.json the HTTP route serves.
    pub_doc = json.loads([p for p in written2 if p.name == "graph.json"][0].read_text())
    assert "concept::private method" not in {n["id"] for n in pub_doc["nodes"]}
    # --allow-private instead writes a SEPARATE exports/graph.private.json.
    priv_path = [p for p in written2 if p.name == "graph.private.json"][0]
    assert priv_path.parent.name == "exports"
    priv_doc = json.loads(priv_path.read_text())
    priv_nodes = {n["id"]: n for n in priv_doc["nodes"]}
    assert "concept::private method" in priv_nodes  # included under --allow-private
    # definition is full-text-derived: dropped for the non-shareable concept.
    assert priv_nodes["concept::private method"].get("definition") in (None, "")
    conn.close()


def test_graph_export(tmp_path):
    """graph.json/GraphML rebuild deterministically; epistemic_type present on EVERY
    edge; llm_inferred absent from citation edges; deterministic citation edges
    shareable; ConceptAlias nodes + has_alias edges present."""
    import networkx as nx

    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    # a shareable deterministic citation edge between the two works.
    conn.execute(
        "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, provenance, "
        "confidence, run_id, created_at) VALUES ('work_a','work_b','cites',"
        "'provider_reference',1.0,'cite1',?)",
        (_now(),),
    )
    conn.commit()
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)

    g = graph_build.build_graph(conn, run_id="r1")
    # ConceptAlias nodes + has_alias edges present.
    alias_nodes = [n for n, d in g.nodes(data=True) if d.get("node_type") == "ConceptAlias"]
    assert alias_nodes
    has_alias = [(u, v) for u, v, d in g.edges(data=True) if d.get("edge_type") == "has_alias"]
    assert has_alias
    # every edge carries an epistemic_type; citation edges never carry llm_inferred.
    for _u, _v, d in g.edges(data=True):
        assert d.get("epistemic_type")
        if d.get("is_citation"):
            assert d["epistemic_type"] == "deterministic"
    # the citation edge is shareable (deterministic/metadata).
    cite_edges = [(u, v, d) for u, v, d in g.edges(data=True) if d.get("is_citation")]
    assert cite_edges and access.is_shareable_citation_edge(cite_edges[0][2]["epistemic_type"])

    # deterministic rebuild: two json exports are byte-identical.
    w1 = export.export_graph(conn, slug="x", run_id="r1", fmt="json", root=tmp_path / "h1")
    w2 = export.export_graph(conn, slug="x", run_id="r1", fmt="json", root=tmp_path / "h2")
    a = [p for p in w1 if p.name == "graph.json"][0].read_bytes()
    b = [p for p in w2 if p.name == "graph.json"][0].read_bytes()
    assert a == b

    # GraphML round-trips into networkx with the same node/edge counts.
    doc = json.loads(a.decode())
    wg = export.export_graph(conn, slug="x", run_id="r1", fmt="graphml", root=tmp_path / "h3")
    gml_path = [p for p in wg if p.name == "graph.graphml"][0]
    reloaded = nx.read_graphml(str(gml_path))
    assert reloaded.number_of_nodes() == len(doc["nodes"])
    assert reloaded.number_of_edges() == len(doc["links"])
    conn.close()


def test_no_llm_overlay(tmp_path):
    """--no-llm over extracted claims builds deterministic concepts + co_occurs edges,
    no provisional rows, manifest concept_mode=deterministic; zero-claim corpus ->
    empty overlay, manifest concept_mode=none (decision 38)."""
    conn = _migrated_conn(tmp_path)
    _add_work(conn, "work_a")
    _add_work(conn, "work_b")
    _add_run(conn, "extr_a", "work_a")
    _add_run(conn, "extr_b", "work_b")
    for wid, extr in (("work_a", "extr_a"), ("work_b", "extr_b")):
        _add_claim(conn, f"al_{wid}", extr, wid, "method", "alpha concept")
        _add_claim(conn, f"be_{wid}", extr, wid, "method", "beta concept")
    conn.commit()
    report = build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    assert report.concept_mode == "deterministic"
    assert report.co_occurs_edges >= 1  # alpha + beta co-occur in 2 works
    assert conn.execute(
        "SELECT COUNT(*) FROM concepts WHERE status='provisional'"
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM review_queue").fetchone()[0] == 0
    conn.close()

    # zero-claim corpus -> honestly empty overlay, concept_mode=none.
    conn2 = _migrated_conn(tmp_path / "empty")
    _add_work(conn2, "work_a")
    conn2.commit()
    report2 = build_semantic_overlay(conn2, run_id="r1", profile=None, tau=0.6)
    assert report2.concept_mode == "none"
    assert report2.concepts_written == 0
    assert conn2.execute("SELECT COUNT(*) FROM concepts").fetchone()[0] == 0
    conn2.close()


# --------------------------------------------------------------------------
# Milestone (doc 10 §11)
# --------------------------------------------------------------------------


def test_concept_select_milestone(tmp_path):
    """concepts build then concepts show <id> returns the concept's linked papers,
    claims, and evidence spans; the FastAPI route returns the same triple."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from seedgraph.cli import app as cli_app
    from seedgraph.project import service
    from seedgraph.web.routes import router

    h = service.create_project("milestone")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    _seed_basic(conn)
    conn.close()

    runner = CliRunner()
    res = runner.invoke(cli_app, ["concepts", "build", "milestone"])
    assert res.exit_code == 0, res.output

    cid = "concept::masked language modeling"
    conn = sqlite3.connect(str(h.db_path))
    detail = concept_detail(conn, cid)
    conn.close()
    assert {p["work_id"] for p in detail["papers"]} == {"work_a", "work_b"}
    assert {c["claim_id"] for c in detail["claims"]} == {"claim_1", "claim_2"}
    assert {s["span_id"] for s in detail["spans"]} == {"span_1"}

    show = runner.invoke(cli_app, ["concepts", "show", "milestone", cid])
    assert show.exit_code == 0, show.output
    assert "work_a" in show.output and "claim_1" in show.output and "span_1" in show.output

    # equivalent FastAPI read route returns the same triple. The concept route is a
    # private-evidence GET behind require_local_session (cross-cutting #1); the test
    # uses the dependency-override seam rather than TestClient loopback quirks.
    from seedgraph.web.serve import require_local_session

    api = FastAPI()
    api.include_router(router)
    api.dependency_overrides[require_local_session] = lambda: None
    client = TestClient(api)
    resp = client.get(f"/projects/milestone/concepts/{cid}")
    assert resp.status_code == 200
    body = resp.json()
    assert {p["work_id"] for p in body["papers"]} == {"work_a", "work_b"}
    assert {c["claim_id"] for c in body["claims"]} == {"claim_1", "claim_2"}
    assert {s["span_id"] for s in body["spans"]} == {"span_1"}


# --------------------------------------------------------------------------
# Schema authority parity (plan §10 step 2; D6)
# --------------------------------------------------------------------------


def _affinity(type_text: str) -> str:
    t = type_text.upper()
    if "INT" in t:
        return "INTEGER"
    if any(k in t for k in ("CHAR", "CLOB", "TEXT")):
        return "TEXT"
    if any(k in t for k in ("REAL", "FLOA", "DOUB")):
        return "REAL"
    return "NUMERIC"


def _assert_table_parity(insp, table_name, model):
    orm_cols = {c.name: c for c in model.__table__.columns}
    mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
    assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"

    mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
    orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
    assert mig_pk == orm_pk, f"{table_name}: PK drift"

    orm_dialect = sqlite_dialect.dialect()
    for name, orm_col in orm_cols.items():
        mig_col = mig_cols[name]
        orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
        mig_aff = _affinity(str(mig_col["type"]))
        assert orm_aff == mig_aff, f"{table_name}.{name}: type {orm_aff} != {mig_aff}"
        if name not in mig_pk:
            assert orm_col.nullable == mig_col["nullable"], f"{table_name}.{name}: nullability drift"

    mig_ix = {tuple(ix["column_names"]) for ix in insp.get_indexes(table_name) if not ix.get("unique")}
    orm_ix = {tuple(c.name for c in ix.columns) for ix in model.__table__.indexes}
    assert mig_ix == orm_ix, f"{table_name}: index column-sets drift ({mig_ix} != {orm_ix})"

    mig_uq = {frozenset(u["column_names"]) for u in insp.get_unique_constraints(table_name)}
    orm_uq = {
        frozenset(c.name for c in u.columns)
        for u in model.__table__.constraints
        if u.__class__.__name__ == "UniqueConstraint"
    }
    assert mig_uq == orm_uq, f"{table_name}: unique-constraint drift ({mig_uq} != {orm_uq})"


def test_orm_migration_parity(tmp_path):
    """The six §4 ORM classes mirror the migrated 0010 schema exactly (no
    create_all authors schema; D6)."""
    db = tmp_path / "parity.db"
    conn = sqlite3.connect(str(db))
    migrations.run_migrations(conn, "project")
    conn.close()

    insp = inspect(create_engine(f"sqlite:///{db.as_posix()}"))
    for table in ("concepts", "concept_aliases", "claim_concepts", "edge_spans",
                  "concept_constraints", "project_graph_edges"):
        assert table in insp.get_table_names(), f"{table} missing from migrated schema"

    _assert_table_parity(insp, "concepts", Concept)
    _assert_table_parity(insp, "concept_aliases", ConceptAlias)
    _assert_table_parity(insp, "claim_concepts", ClaimConcept)
    _assert_table_parity(insp, "edge_spans", EdgeSpan)
    _assert_table_parity(insp, "concept_constraints", ConceptConstraint)
    _assert_table_parity(insp, "project_graph_edges", ProjectGraphEdge)


# --------------------------------------------------------------------------
# Eval hooks (plan §11)
# --------------------------------------------------------------------------


def test_concept_merge_audit_hook(tmp_path):
    """The build emits sampled concept-merge audit candidates into audit_records for
    manual grading, and surfaces cross-run constraint-flip detection (doc 09 §6)."""
    from seedgraph.semantic import audit

    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    build_semantic_overlay(conn, run_id="r1", profile=None, tau=0.6)
    written = audit.emit_merge_audit_sample(conn, run_id="r1", sample_size=10, seed=7)
    conn.commit()
    assert written >= 1  # the MLM/Masked-Language-Modeling fold is a merge candidate
    rows = conn.execute(
        "SELECT audit_type, subject_type FROM audit_records WHERE audit_type='concept_merge'"
    ).fetchall()
    assert rows and all(r == ("concept_merge", "concept_merge") for r in rows)

    # cross-run constraint-flip detection: a pair in BOTH must_link and cannot_link.
    now = _now()
    conn.execute(
        "INSERT INTO concept_constraints (kind, label_a, label_b, source, created_at) "
        "VALUES ('must_link','a','b','user',?)", (now,)
    )
    conn.execute(
        "INSERT INTO concept_constraints (kind, label_a, label_b, source, created_at) "
        "VALUES ('cannot_link','a','b','user',?)", (now,)
    )
    conn.commit()
    assert audit.detect_constraint_flips(conn) == [("a", "b")]
    conn.close()
