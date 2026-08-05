"""Build C chunk 9 — concept_set_hash + canon decision log + flip comparator.

Decision 56's mandate (D10): every non-empty concepts build writes exactly ONE
``audit_records`` row (``audit_type='canon_decision_log'``) keyed by the
corpus-set hash, and ``compare_canon_decision_logs`` diffs the two most recent
rows — same hash + decision delta ⇒ machine-side flip; different hash ⇒ honest
"corpus grew" report. Fully offline (NullProposer / in-test mock proposer).
"""

from __future__ import annotations

import json
import sqlite3
import string
from datetime import datetime, timezone

import seedgraph.db.migrations as migrations
from seedgraph.eval import audit as eval_audit
from seedgraph.semantic import audit, concepts, llm_propose


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Offline seed helpers (mirrors test_phase_7's minimal corpus)
# --------------------------------------------------------------------------


def _migrated_conn(tmp_path) -> sqlite3.Connection:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "project.db"
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA foreign_keys=ON")
    migrations.run_migrations(conn, "project")
    return conn


def _add_work(conn, work_id):
    conn.execute(
        "INSERT INTO works (work_id, canonical_title, created_at) VALUES (?,?,?)",
        (work_id, "Paper " + work_id, _now()),
    )
    conn.execute(
        "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
        "created_at, updated_at) VALUES (?,?,?,?,?)",
        (work_id, "included", 0, _now(), _now()),
    )
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_version, prompt_version, access_class, run_status, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("extr_" + work_id, work_id, "md_" + work_id, "h", "v1", "p1",
         "open_access", "success", _now()),
    )


def _add_claim(conn, claim_id, work_id, claim_type, label):
    conn.execute(
        "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, "
        "claim_type, field_key, normalized_label, claim_text, status, "
        "epistemic_type, access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (claim_id, "extr_" + work_id, work_id, claim_type, "f", label,
         label + " text.", "found", "llm_extracted", "open_access", _now()),
    )


def _seed_basic(conn):
    """Acronym fold pair (MLM) + a veto pair (BERT-base/large), all 'method'."""
    _add_work(conn, "work_a")
    _add_work(conn, "work_b")
    _add_claim(conn, "claim_1", "work_a", "method", "MLM")
    _add_claim(conn, "claim_2", "work_b", "method", "Masked Language Modeling")
    _add_claim(conn, "claim_3", "work_a", "method", "BERT-base")
    _add_claim(conn, "claim_4", "work_b", "method", "BERT-large")
    conn.commit()


class _FoldProposer:
    """Mock proposer: returns given subsets when all members are in the cluster."""

    def __init__(self, subsets):
        self._subsets = subsets

    def propose(self, cluster):
        cl = set(cluster)
        return [list(s) for s in self._subsets if set(s) <= cl]


def _log_rows(conn):
    return conn.execute(
        "SELECT subject_id, run_id, status, payload FROM audit_records "
        "WHERE audit_type='canon_decision_log' ORDER BY rowid"
    ).fetchall()


# --------------------------------------------------------------------------
# concept_set_hash — invariance
# --------------------------------------------------------------------------


def test_concept_set_hash_order_and_duplicate_insensitive():
    """Same de-duped set ⇒ same 64-hex digest, regardless of order/repeats."""
    h1 = audit.concept_set_hash([("method", "mlm"), ("dataset", "squad")])
    h2 = audit.concept_set_hash(
        [("dataset", "squad"), ("method", "mlm"), ("method", "mlm")]
    )
    assert h1 == h2
    assert len(h1) == 64
    assert set(h1) <= set(string.hexdigits.lower())


def test_concept_set_hash_changes_on_new_label_or_type():
    """Adding a label OR re-typing an existing label changes the hash."""
    base = audit.concept_set_hash([("method", "mlm"), ("dataset", "squad")])
    grown = audit.concept_set_hash(
        [("method", "mlm"), ("dataset", "squad"), ("method", "bert base")]
    )
    retyped = audit.concept_set_hash([("finding", "mlm"), ("dataset", "squad")])
    assert grown != base
    assert retyped != base
    assert grown != retyped


# --------------------------------------------------------------------------
# One build ⇒ exactly one decision-log row naming every cluster
# --------------------------------------------------------------------------


def test_one_build_writes_exactly_one_decision_log_row(tmp_path):
    """One build ⇒ ONE canon_decision_log row: 64-hex subject_id equal to the
    concept_set_hash of the (concept_type, normalized_label) set, status
    'resolved', non-null JSON payload naming every cluster's members."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    rows = _log_rows(conn)
    assert len(rows) == 1
    subject_id, run_id, status, payload = rows[0]
    assert run_id == "r1"
    assert status == "resolved"
    expected_hash = audit.concept_set_hash(
        [
            ("method", "mlm"),
            ("method", "masked language modeling"),
            ("method", "bert base"),
            ("method", "bert large"),
        ]
    )
    assert subject_id == expected_hash
    decisions = json.loads(payload)
    # every normalized label appears in exactly one named cluster
    all_members = sorted(
        m for cluster in decisions["clusters"] for m in cluster["members"]
    )
    assert all_members == sorted(
        ["mlm", "masked language modeling", "bert base", "bert large"]
    )
    # clusters are type-scoped (chunk 8 ordering guarantee)
    assert {c["concept_type"] for c in decisions["clusters"]} == {"method"}
    # the deterministic acronym fold is recorded on its cluster
    mlm_cluster = next(
        c for c in decisions["clusters"] if "mlm" in c["members"]
    )
    assert ["masked language modeling", "mlm"] in mlm_cluster["deterministic_folds"]
    conn.close()


def test_vetoed_proposal_recorded_with_guardrail_reason(tmp_path):
    """A guardrail-vetoed proposer subset appears in the payload with WHICH
    guardrail fired (the old bare-continue path now captures the veto reason);
    an already-deterministically-folded proposal is recorded as skipped."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    proposer = _FoldProposer(
        [("bert base", "bert large"), ("mlm", "masked language modeling")]
    )
    # tau=0.3 clusters bert base/large (jaccard 1/3) so the proposal is offered,
    # then the discriminating-token veto (base vs large) fires.
    concepts.build_concepts(conn, run_id="r1", proposer=proposer, tau=0.3)
    rows = _log_rows(conn)
    assert len(rows) == 1
    decisions = json.loads(rows[0][3])
    bert_cluster = next(
        c for c in decisions["clusters"] if "bert base" in c["members"]
    )
    veto = next(
        p for p in bert_cluster["proposals"]
        if p["subset"] == ["bert base", "bert large"]
    )
    assert veto["outcome"] == "guardrail_veto"
    assert veto["guardrail"] == "discriminating_token_veto"
    # the MLM pair was already deterministically folded → skip, not enqueue
    mlm_cluster = next(
        c for c in decisions["clusters"] if "mlm" in c["members"]
    )
    mlm_prop = next(
        p for p in mlm_cluster["proposals"]
        if p["subset"] == ["masked language modeling", "mlm"]
    )
    assert mlm_prop["outcome"] == "no_new_pairs"
    assert mlm_prop["skipped_pairs"] == [
        {"pair": ["masked language modeling", "mlm"], "reason": "already_folded"}
    ]
    # no fold survived the guardrails → zero enqueued review pairs anywhere
    assert all(
        p.get("enqueued_pairs", []) == []
        for c in decisions["clusters"]
        for p in c["proposals"]
    )
    conn.close()


# --------------------------------------------------------------------------
# Comparator — zero flips / forced flip / honest set-change
# --------------------------------------------------------------------------


def test_two_identical_builds_compare_to_zero_flips(tmp_path):
    """Two builds over the SAME corpus ⇒ same hash, byte-identical decisions,
    comparator reports status='same_set' with zero flips."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    concepts.build_concepts(
        conn, run_id="r2", proposer=llm_propose.NullProposer(), tau=0.6
    )
    rows = _log_rows(conn)
    assert len(rows) == 2  # one row PER build
    report = audit.compare_canon_decision_logs(conn)
    assert report.status == "same_set"
    assert report.earlier_hash == report.later_hash
    assert report.flips == ()
    assert "zero flips" in report.summary
    conn.close()


def test_forced_flip_detected_on_unchanged_set(tmp_path):
    """A cannot_link constraint added BETWEEN builds flips the deterministic
    fold decision while the concept set (hash) is unchanged ⇒ the comparator
    reports the machine-side delta."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    conn.execute(
        "INSERT INTO concept_constraints (kind, label_a, label_b, source, "
        "created_at) VALUES (?,?,?,?,?)",
        ("cannot_link", "masked language modeling", "mlm", "user", _now()),
    )
    concepts.build_concepts(
        conn, run_id="r2", proposer=llm_propose.NullProposer(), tau=0.6
    )
    report = audit.compare_canon_decision_logs(conn)
    assert report.status == "same_set"
    assert len(report.flips) >= 1
    changed_fields = {f["field"] for f in report.flips if f["kind"] == "decision_changed"}
    # the fold flipped: gone from deterministic_folds, present in cannot_link_skips
    assert {"deterministic_folds", "cannot_link_skips"} <= changed_fields
    fold_flip = next(f for f in report.flips if f["field"] == "deterministic_folds")
    assert fold_flip["before"] == [["masked language modeling", "mlm"]]
    assert fold_flip["after"] == []
    conn.close()


def test_corpus_growth_reported_honestly_not_as_flip(tmp_path):
    """A new label between builds changes the hash ⇒ status='set_changed' with
    zero flips and the honest re-roll summary (never a false flip alarm)."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    _add_claim(conn, "claim_5", "work_a", "dataset", "SQuAD")
    conn.commit()
    concepts.build_concepts(
        conn, run_id="r2", proposer=llm_propose.NullProposer(), tau=0.6
    )
    report = audit.compare_canon_decision_logs(conn)
    assert report.status == "set_changed"
    assert report.earlier_hash != report.later_hash
    assert report.flips == ()
    assert "corpus grew" in report.summary
    conn.close()


def test_comparator_needs_two_logs(tmp_path):
    """Fewer than two decision logs ⇒ status='insufficient_logs' (no diff)."""
    conn = _migrated_conn(tmp_path)
    report = audit.compare_canon_decision_logs(conn)
    assert report.status == "insufficient_logs"
    assert report.flips == ()
    conn.close()


def test_empty_build_writes_no_decision_log(tmp_path):
    """A zero-claim ('none') build performs no canonicalization ⇒ no log row."""
    conn = _migrated_conn(tmp_path)
    report = concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    assert report.concept_mode == "none"
    assert _log_rows(conn) == []
    conn.close()


# --------------------------------------------------------------------------
# eval/audit.py closed-set extension
# --------------------------------------------------------------------------


def test_eval_soft_vocab_admits_canon_decision_log(tmp_path):
    """'canon_decision_log'/'concept_set' joined the closed soft-vocab sets, so
    the decision-log rows are in-vocab (open_audit validation accepts them)."""
    assert "canon_decision_log" in eval_audit.AUDIT_TYPES
    assert "concept_set" in eval_audit.SUBJECT_TYPES
    conn = _migrated_conn(tmp_path)
    audit_id = eval_audit.open_audit(
        conn, "canon_decision_log", "concept_set", "deadbeef" * 8, None,
        "user_supplied_private",
    )
    assert audit_id.startswith("audit_")
    conn.close()


def test_payload_column_migrated_and_null_for_other_rows(tmp_path):
    """0016 adds a nullable payload column: decision-log rows fill it, every
    other producer leaves it NULL (additive, no backfill)."""
    conn = _migrated_conn(tmp_path)
    _seed_basic(conn)
    concepts.build_concepts(
        conn, run_id="r1", proposer=llm_propose.NullProposer(), tau=0.6
    )
    eval_audit.open_audit(
        conn, "concept_merge", "concept_merge", "alias_1", "r1", "open_access"
    )
    assert conn.execute(
        "SELECT payload IS NOT NULL FROM audit_records "
        "WHERE audit_type='canon_decision_log'"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT payload IS NULL FROM audit_records WHERE audit_type='concept_merge'"
    ).fetchone()[0] == 1
    conn.close()
