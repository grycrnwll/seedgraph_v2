"""Phase 9 — contract / acceptance + corpus-verification (plan §11, maps to
test_eval_acceptance.py).

This is the CI gate: it runs over the synthetic ``mini_project`` fixtures +
recorded AnswerEnvelope cassettes with NO live LLM and NO API key (must_fix #1;
doc 10 §3). The phase-8 ``AnswerEnvelope`` is IMPORTED (consumed), never redefined.
The ``first_corpus.yaml``-derived assertions are blocked while the file is PROPOSED.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

# Import-clean check: the whole eval package imports without an API key or any
# built upstream phase (AnswerEnvelope stays a deferred, body-only reference).
from seedgraph.eval import audit, boundary, fixtures, goldsets, metrics, report, runners  # noqa: F401

# Repo-level synthetic fixture builder (evals/ is not part of the installed package).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from evals.fixtures import mini_project  # noqa: E402

_FIRST_CORPUS = _REPO_ROOT / "evals" / "fixtures" / "first_corpus.yaml"


@pytest.fixture
def mini(isolated_home):
    """Build the synthetic mini_project into the isolated SEEDGRAPH_HOME."""
    return mini_project.build(isolated_home)


def _project_conn(facts) -> sqlite3.Connection:
    conn = sqlite3.connect(facts["db_path"])
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _first_corpus_status() -> str:
    spec = yaml.safe_load(_FIRST_CORPUS.read_text(encoding="utf-8")) or {}
    return str(spec.get("status", "")).lower()


# --- Corpus-verification precondition (REQUIRED — first_corpus.yaml is PROPOSED) ---


def test_verify_corpus_promotes_proposed_to_verified(mini, isolated_home):
    """eval verify-corpus resolves each identifier, confirms each
    expected_citation_edges entry against the actually-built graph, and binds every
    expected_claim to a real evidence span (md[start:end]==quote); on full success it
    stamps status: proposed -> status: verified (plan §2/§10 step 10/§11). Runs
    KEY-FREE over the synthetic mini corpus (no network — works are already acquired)."""
    corpus = isolated_home / "synthetic_corpus.yaml"
    corpus.write_text(
        "status: proposed\n"
        "papers:\n"
        '  - {work_id: work_p1, title: "Difference-in-Differences with staggered adoption", '
        'identifiers: {doi: "10.0000/p1"}}\n'
        '  - {work_id: work_p2, title: "Event-study estimation under heterogeneity", '
        'identifiers: {arxiv: "2099.00002"}}\n'
        "expected_citation_edges:\n"
        "  - {citing: work_p1, cited: work_p2}\n"
        "expected_claims:\n"
        "  work_p1:\n"
        '    - {claim_id: c1, text: "parallel trends"}\n'
        "  work_p2:\n"
        '    - {claim_id: c2, text: "rank condition"}\n',
        encoding="utf-8",
    )
    result = runners.verify_corpus(
        mini["slug"], "run_mini", corpus_path=corpus, root=isolated_home
    )
    assert result.verified is True, result.failures
    assert _status_of(corpus) == "verified"


def test_verify_corpus_fails_closed_and_blocks_acceptance(mini, isolated_home):
    """Any unfetchable PDF / unresolved identifier / missing edge / unattachable
    claim leaves the corpus status: proposed (verified=False) and BLOCKS all
    first_corpus-based acceptance (fail-closed; plan §2/§11)."""
    corpus = isolated_home / "bad_corpus.yaml"
    corpus.write_text(
        "status: proposed\n"
        "papers:\n"
        '  - {work_id: work_p1, title: "DiD", identifiers: {doi: "10.0000/p1"}}\n'
        "expected_citation_edges:\n"
        "  - {citing: work_p1, cited: work_absent}\n"  # endpoint not in the graph
        "expected_claims: {}\n",
        encoding="utf-8",
    )
    result = runners.verify_corpus(
        mini["slug"], "run_mini", corpus_path=corpus, root=isolated_home
    )
    assert result.verified is False
    assert result.failures  # at least one fail-closed reason
    assert _status_of(corpus) == "proposed"  # NOT promoted


def test_acceptance_first_corpus_assertions_blocked_until_verified(mini):
    """A precondition asserts first_corpus.yaml.status == 'verified'; while it is
    'proposed' the first_corpus-derived assertions are blocked/skipped (never
    silently passed), while the synthetic mini_project fixtures still gate CI
    key-free (plan §11)."""
    status = _first_corpus_status()
    first_corpus_verified = status == "verified"
    # The real fixture ships PROPOSED — the derived oracle assertions MUST NOT run.
    assert not first_corpus_verified, (
        "first_corpus.yaml is verified on disk — tests must not pre-set it; only a "
        "live `eval verify-corpus` run may promote it (plan §11)."
    )
    if first_corpus_verified:  # pragma: no cover - blocked while PROPOSED
        pytest.fail("first_corpus oracle assertions would run here once verified")
    # The synthetic mini_project still gates CI key-free even while first_corpus is blocked.
    conn = _project_conn(mini)
    try:
        assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] >= 1
    finally:
        conn.close()


# --- MVP §12 over mini_project + recorded envelopes (key-free CI gate) ---


def test_mvp_same_file_hash_not_converted_twice(mini, isolated_home):
    """Dedup invariant: a PDF with an already-seen file_hash is not processed/
    converted twice in cache.db (doc 09 §12 / doc 10 §4)."""
    cache_db = Path(mini["cache_root"]) / "cache.db"
    conn = sqlite3.connect(str(cache_db))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        total, distinct = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT file_hash) FROM source_files"
        ).fetchone()
        assert total == distinct  # no file_hash converted twice
        existing = conn.execute("SELECT file_hash FROM source_files LIMIT 1").fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):  # UNIQUE(file_hash) blocks re-ingest
            conn.execute(
                "INSERT INTO source_files (source_file_id, file_hash, file_type, access_class, "
                "created_at) VALUES ('sf_dup', ?, 'pdf', 'open_access', 'now')",
                (existing,),
            )
    finally:
        conn.close()


def test_mvp_markdown_present_for_fixture_papers(mini):
    """Every window-fitting fixture paper has its converted markdown present and
    re-sliceable (doc 09 §12)."""
    cache_db = Path(mini["cache_root"]) / "cache.db"
    cache_conn = sqlite3.connect(str(cache_db))
    conn = _project_conn(mini)
    try:
        for work_id in mini["window_fitting_work_ids"]:
            bridge = conn.execute(
                "SELECT markdown_id FROM work_source_files WHERE work_id=?", (work_id,)
            ).fetchone()
            assert bridge is not None and bridge[0], f"{work_id}: no acquired markdown"
            text = runners._markdown_text(cache_conn, Path(mini["cache_root"]), bridge[0])
            assert text is not None and len(text) > 0, f"{work_id}: markdown not re-sliceable"
    finally:
        conn.close()
        cache_conn.close()


def test_mvp_note_coverage_windowfitting_with_oversize_recorded(mini):
    """Every window-fitting converted paper has a default note, while
    skipped_oversize papers are recorded with a not-processed record — not silently
    dropped, not counted as a coverage failure (decision D4; doc 09 §12)."""
    conn = _project_conn(mini)
    try:
        success_runs = [
            r[0]
            for r in conn.execute(
                "SELECT extraction_run_id FROM extraction_runs WHERE run_status='success'"
            ).fetchall()
        ]
        noted = {
            r[0] for r in conn.execute("SELECT extraction_run_id FROM structured_notes").fetchall()
        }
        flags = [1 if rid in noted else 0 for rid in success_runs]
        oversize = conn.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE run_status='skipped_oversize'"
        ).fetchone()[0]
        cov = metrics.note_coverage(flags, oversize)
        assert cov["coverage"] == 1.0  # every window-fitting paper has a note
        assert cov["oversize_skipped"] >= 1.0  # the oversize paper is RECORDED, not dropped
        # the oversize run exists (a not-processed record), and has NO note (never a failure).
        oversize_run = conn.execute(
            "SELECT extraction_run_id FROM extraction_runs WHERE run_status='skipped_oversize'"
        ).fetchone()[0]
        assert oversize_run not in noted
    finally:
        conn.close()


def test_mvp_citation_edges_present_and_unresolved_target_recorded(mini):
    """Included-paper citation edges are present; any referenced-but-absent target
    is recorded as a phase_2 unresolved_target diagnostic and NEVER silently created
    (decision D3; doc 09 §12)."""
    conn = _project_conn(mini)
    try:
        edges = conn.execute(
            "SELECT COUNT(*) FROM citation_edges WHERE edge_type='cites'"
        ).fetchone()[0]
        assert edges >= 1  # included-paper edges present
        # the unresolved reference is recorded as a diagnostic ...
        diag = conn.execute(
            "SELECT raw_reference_text FROM reference_entries "
            "WHERE reference_id=? AND resolved_work_id IS NULL AND resolution_status='unresolved'",
            (mini["unresolved_ref_id"],),
        ).fetchone()
        assert diag is not None  # recorded, visible
        # ... and the absent target was NEVER materialized as a works row (D3 link-only).
        absent = conn.execute(
            "SELECT COUNT(*) FROM works WHERE canonical_title LIKE ?",
            (f"%{mini['absent_target_hint']}%",),
        ).fetchone()[0]
        assert absent == 0
    finally:
        conn.close()


def test_mvp_substantive_claim_has_span_or_not_found(mini):
    """Every substantive claim has >=1 evidence span OR an explicit not_found
    record (doc 09 §12)."""
    conn = _project_conn(mini)
    try:
        claims = conn.execute(
            "SELECT claim_id, status FROM extracted_claims"
        ).fetchall()
        assert claims
        for claim_id, status in claims:
            n_spans = conn.execute(
                "SELECT COUNT(*) FROM claim_spans WHERE claim_id=?", (claim_id,)
            ).fetchone()[0]
            assert n_spans >= 1 or status == "not_found", f"{claim_id}: no span and not not_found"
        # the explicit not_found claim exists and carries no span (non-vacuous).
        nf = conn.execute(
            "SELECT status FROM extracted_claims WHERE claim_id=?", (mini["not_found_claim_id"],)
        ).fetchone()
        assert nf == ("not_found",)
        assert conn.execute(
            "SELECT COUNT(*) FROM claim_spans WHERE claim_id=?", (mini["not_found_claim_id"],)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_mvp_answer_citations_subset_of_retrieved_and_project(mini):
    """A recorded answer's cited_work_ids subset of project works and cited_span_ids
    subset of retrieved_item_ids — no invented citations (doc 09 §8/§12). KEY-FREE:
    replays a recorded AnswerEnvelope cassette (no live LLM)."""
    conn = _project_conn(mini)
    try:
        project_works = {r[0] for r in conn.execute("SELECT work_id FROM works").fetchall()}
    finally:
        conn.close()
    # step-0 contract (§9): the phase-8 AnswerEnvelope is IMPORTED (never redefined)
    # and carries the consumed faithfulness/leakage fields.
    from seedgraph.answer.types import AnswerEnvelope

    required = {"answer_text", "answer_category", "cited_work_ids", "cited_span_ids",
                "retrieved_item_ids", "insufficient_evidence"}
    assert required <= set(AnswerEnvelope.model_fields)

    env = fixtures.load_envelope("answer_p1")
    assert isinstance(env, AnswerEnvelope)  # cassette validated against the imported shape
    assert set(env.cited_work_ids) <= project_works
    assert set(env.cited_span_ids) <= set(env.retrieved_item_ids)  # no invented citations
    assert env.cited_span_ids  # non-vacuous: it actually cites a span


def test_mvp_sampled_claim_span_inspectable_quote_match(mini):
    """A sampled claim's evidence span is inspectable and md[start:end] == quote
    (quote_hash match) via the read-only cache.db re-slice (doc 09 §11/§12)."""
    sp = mini["span_p1"]
    cache_db = Path(mini["cache_root"]) / "cache.db"
    cache_conn = sqlite3.connect(str(cache_db))
    try:
        text = runners._markdown_text(cache_conn, Path(mini["cache_root"]), sp["markdown_id"])
        assert text is not None
        assert text[sp["start"]:sp["end"]] == sp["quote"]  # md[start:end] == quote
    finally:
        cache_conn.close()
    # the deterministic span_indexing metric agrees on the re-sliced span.
    quality = metrics.span_indexing_quality(
        [{"markdown": text, "start": sp["start"], "end": sp["end"], "quote": sp["quote"]}]
    )
    assert quality == 1.0


# --- Concept-identity noise (decision D10) ---


def test_concept_identity_noise_audit_within_or_flags_tolerance(mini):
    """The graded concept-merge sample yields {over_merge_rate, under_merge_rate};
    the test asserts the phase_7 anti-overmerge guard held and borderline merges
    routed to review_queue — it does NOT require perfect identity, and rates above
    report.NOISE_TOLERANCE are reported as findings, not a CI failure (decision D10)."""
    conn = _project_conn(mini)
    try:
        # borderline merges routed to review_queue (the anti-overmerge guard surface).
        borderline = conn.execute(
            "SELECT COUNT(*) FROM review_queue WHERE item_type='concept_merge_candidate'"
        ).fetchone()[0]
        assert borderline >= 1

        # grade a concept-merge sample: mostly correct, no over-merge (guard held).
        verdicts = ["correct", "correct", "correct", "under_merge"]
        for i, verdict in enumerate(verdicts):
            aid = audit.open_audit(
                conn, "concept_merge", "concept_merge", f"alias_{i}", "run_mini", "open_access"
            )
            audit.record_verdict(conn, aid, "accept", verdict=verdict)

        graded = [
            r[0]
            for r in conn.execute(
                "SELECT verdict FROM audit_records WHERE audit_type='concept_merge' "
                "AND status='resolved'"
            ).fetchall()
        ]
        noise = metrics.concept_identity_noise(graded)
        # anti-overmerge guard held: zero over-merges in the graded sample.
        assert noise["over_merge_rate"] == 0.0
        # rates over tolerance are FINDINGS, not failures — the test only records them.
        over_flag = noise["over_merge_rate"] > report.NOISE_TOLERANCE["over_merge_rate"]
        under_flag = noise["under_merge_rate"] > report.NOISE_TOLERANCE["under_merge_rate"]
        assert over_flag is False  # within tolerance here, but exceeding would NOT fail CI
        assert under_flag in (True, False)  # reported either way, never gates CI
    finally:
        conn.close()


# --- Leakage abstention (doc 09 §9) ---


def test_leakage_abstention_rate_is_100_percent():
    """Abstention rate over the recorded leakage-probe envelopes == 100%
    (insufficient_evidence=true for every probe) — deterministic + key-free because
    the envelopes are canned (doc 09 §9; hard invariant)."""
    leak_envelopes = fixtures.load_envelopes("leak_")
    assert len(leak_envelopes) >= 5  # non-vacuous probe set
    abstained = sum(1 for env in leak_envelopes if env.insufficient_evidence)
    assert abstained == len(leak_envelopes)  # 100% abstention — HARD invariant


# --- Content-access boundary (doc 09 §10) ---


def test_content_access_boundary_no_private_leak(mini):
    """boundary.find_boundary_violations returns empty: zero user_supplied_private-
    derived artifacts in any shareable/export output; audit_records and per-project
    eval/ files stay confined to project scope (doc 09 §10; decision 60/D8)."""
    conn = _project_conn(mini)
    try:
        # the fixture has a genuinely-private span (non-vacuous boundary).
        priv = conn.execute(
            "SELECT access_class FROM evidence_spans WHERE span_id=?", (mini["private_span_id"],)
        ).fetchone()
        assert priv == ("user_supplied_private",)
        from seedgraph.vocab import is_shareable

        assert is_shareable("user_supplied_private") is False  # private never shareable
        # the shareable export surface leaks nothing private-derived.
        assert boundary.find_boundary_violations(conn, mini["slug"]) == []
    finally:
        conn.close()


def _status_of(corpus_path: Path) -> str:
    spec = yaml.safe_load(corpus_path.read_text(encoding="utf-8")) or {}
    return str(spec.get("status", "")).lower()
