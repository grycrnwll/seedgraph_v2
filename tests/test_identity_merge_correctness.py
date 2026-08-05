"""Build A (identity + merge correctness) acceptance tests.

Chunk 1 — identifier surface-form normalization: the v1-ported pure-regex
normalizers in ``project/identity.py:normalize_id`` must map every surface form
v1 accepted to one canonical form, idempotently, so the same id always compares
equal at the identity layer.

Chunk 2 — arXiv-DOI identity fold: ``{doi: 10.48550/arxiv.X}`` and
``{arxiv: X}`` resolve to ONE work (``strong_ids`` folds the alias family to an
arxiv pair; ``canonical_key`` is ``strong_ids[0]`` by construction).

Chunk 3 — conflicting-strong-id merge blocker (v1 Fix 1): a single-work match
whose incoming record carries a DIFFERENT value for a strong id the matched work
already owns never silently merges — it demotes to a ``duplicate_candidate``
review item (``reason="id_disagreement"``). Canonical fixture: preprint/published
pair sharing an arXiv id with two DOIs.

Chunk 4 — weak-id title-corroboration guard (v1 Fix 5): a single-work match
shared ONLY on weak ids (arxiv/s2/ssrn) with both title hashes present and
conflicting demotes to review (``reason="weak_id_title_conflict"``); a shared
doi/openalex id always merges regardless of title.

All tests offline / keyless.
"""

from __future__ import annotations

import json

import pytest
from sqlmodel import Session, select

from seedgraph.db.project_models import Identifier, ReviewQueueItem, Work
from seedgraph.project import identity, service
from seedgraph.project.identity import canonical_key, normalize_id, strong_ids


# --------------------------------------------------------------------------
# Chunk 1 — normalizer table (mirrors v1 identity.py:163-240 docstring examples)
# --------------------------------------------------------------------------

NORMALIZE_CASES = [
    # doi: strip doi.org / dx.doi.org URL prefixes (scheme optional) + doi: label; lowercase
    ("doi", "https://doi.org/10.1234/ABC", "10.1234/abc"),
    ("doi", "https://dx.doi.org/10.1234/ABC", "10.1234/abc"),
    ("doi", "dx.doi.org/10.1/X", "10.1/x"),
    ("doi", "doi.org/10.1/X", "10.1/x"),
    ("doi", "doi:10.1234/X", "10.1234/x"),
    ("doi", "  10.1234/AbC ", "10.1234/abc"),
    # arxiv: strip abs/pdf URL prefixes + arXiv: label; DROP the v\d+ version
    # suffix (load-bearing: v1/v2 of a preprint are the same work)
    ("arxiv", "https://arxiv.org/abs/2401.01234v2", "2401.01234"),
    ("arxiv", "http://arxiv.org/pdf/2401.01234", "2401.01234"),
    ("arxiv", "arxiv.org/abs/2401.01234", "2401.01234"),
    ("arxiv", "arXiv:2401.01234v2", "2401.01234"),
    ("arxiv", "ARXIV:2401.01234", "2401.01234"),
    ("arxiv", "2401.01234v3", "2401.01234"),
    ("arxiv", "2401.01234", "2401.01234"),
    # openalex: strip openalex.org/ URL prefix; uppercase the leading W
    ("openalex", "https://openalex.org/W2044", "W2044"),
    ("openalex", "openalex.org/w2044", "W2044"),
    ("openalex", "w2044", "W2044"),
    ("openalex", "W2044", "W2044"),
    # s2 / ssrn: opaque provider ids — trimmed only, case preserved
    ("s2", " AbC123 ", "AbC123"),
    ("ssrn", " 4123456 ", "4123456"),
]


@pytest.mark.parametrize("id_type,raw,expected", NORMALIZE_CASES)
def test_normalize_id_surface_forms(id_type, raw, expected):
    assert normalize_id(id_type, raw) == expected


@pytest.mark.parametrize("id_type,raw,expected", NORMALIZE_CASES)
def test_normalize_id_idempotent(id_type, raw, expected):
    canonical = normalize_id(id_type, raw)
    assert normalize_id(id_type, canonical) == canonical


def test_normalize_id_empty_inputs_are_none():
    assert normalize_id("doi", None) is None
    assert normalize_id("arxiv", "   ") is None
    assert normalize_id("openalex", "") is None
    # a bare arXiv label with no id collapses to nothing, not an empty string
    assert normalize_id("arxiv", "arXiv:") is None


# --------------------------------------------------------------------------
# Chunk 1 — round trip: two arXiv surface forms resolve to ONE work
# --------------------------------------------------------------------------

def test_arxiv_surface_forms_round_trip_one_work():
    h = service.create_project("norm_roundtrip")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(
            s, {"arxiv": "arXiv:2401.01234v2", "title": "Same Preprint"}
        )
        w2, o2 = identity.upsert_work(
            s, {"arxiv": "https://arxiv.org/abs/2401.01234", "title": "Same Preprint"}
        )
        s.commit()
        assert o1 == "created"
        assert o2 == "merged"
        assert w1.work_id == w2.work_id
        assert len(s.exec(select(Work)).all()) == 1
        # one canonical identifier row, version suffix and URL prefix both gone
        idents = s.exec(select(Identifier)).all()
        assert {(i.id_type, i.id_value) for i in idents} == {("arxiv", "2401.01234")}
        # scalar mirror column carries the canonical form too
        assert w1.arxiv_id == "2401.01234"


# --------------------------------------------------------------------------
# Chunk 2 — arXiv-DOI alias fold: {doi: 10.48550/arxiv.X} == {arxiv: X}
# --------------------------------------------------------------------------

def test_arxiv_doi_alias_folds_to_arxiv_pair():
    # The alias doi yields the arxiv pair INSTEAD OF a doi pair, with the
    # chunk-1 normalizer applied to the captured id (version suffix dropped).
    assert strong_ids({"doi": "10.48550/arXiv.2401.01234v2"}) == [("arxiv", "2401.01234")]
    assert (
        canonical_key({"doi": "10.48550/arXiv.2401.01234"})
        == ("arxiv", "2401.01234")
        == canonical_key({"arxiv": "2401.01234"})
    )
    # The fold moves ONLY the 10.48550 family — any other DOI keys as doi.
    assert canonical_key({"doi": "10.1234/X"}) == ("doi", "10.1234/x")


def test_canonical_key_is_first_strong_id():
    # canonical_key is defined as strong_ids[0], so key and lookup can never
    # disagree; the folded arxiv pair outranks the weaker s2 id.
    rec = {"doi": "10.48550/arxiv.2401.01234", "s2": "S2X"}
    assert canonical_key(rec) == strong_ids(rec)[0] == ("arxiv", "2401.01234")
    assert canonical_key({}) is None
    assert strong_ids({}) == []


def test_arxiv_doi_alias_round_trip_one_work():
    # alias-doi record first: _create_work's scalar mirrors follow the FOLDED
    # pairs automatically — arxiv_id set, doi NULL for the pure-alias record.
    h = service.create_project("alias_roundtrip")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(
            s, {"doi": "10.48550/arXiv.2401.01234", "title": "Same Preprint"}
        )
        w2, o2 = identity.upsert_work(s, {"arxiv": "2401.01234", "title": "Same Preprint"})
        s.commit()
        assert (o1, o2) == ("created", "merged")
        assert w1.work_id == w2.work_id
        assert len(s.exec(select(Work)).all()) == 1
        idents = s.exec(select(Identifier)).all()
        assert {(i.id_type, i.id_value) for i in idents} == {("arxiv", "2401.01234")}
        assert w1.arxiv_id == "2401.01234"
        assert w1.doi is None


def test_arxiv_doi_alias_disagreeing_arxiv_id_drops_the_alias_doi():
    # Senior-review R3 pin (behavior, not endorsement): a record carrying BOTH
    # the alias doi 10.48550/arxiv.X and a DIFFERENT arxiv id Y keeps Y and
    # silently DROPS X — the fold's setdefault never overwrites an existing
    # arxiv pair, and the doi pair was already deleted.
    # ponytail: deliberate tradeoff — the X-vs-Y disagreement is DISCARDED in
    # this pure normalizer (no session to enqueue from), NOT routed to review
    # as an id_disagreement; ceiling: such a record never surfaces X anywhere
    # downstream (lookup, scalar mirrors, review payloads).
    rec = {"doi": "10.48550/arXiv.2401.01234", "arxiv": "9999.88888"}
    assert strong_ids(rec) == [("arxiv", "9999.88888")]
    # canonical_key is strong_ids[0] by construction — X is gone there too.
    assert canonical_key(rec) == ("arxiv", "9999.88888")


def test_arxiv_doi_alias_round_trip_reverse_order():
    # arxiv record first: the alias doi FOLDS at lookup time and merges into the
    # existing work; the merge backfill never smuggles the alias doi onto it.
    h = service.create_project("alias_roundtrip_rev")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(s, {"arxiv": "2401.01234", "title": "Same Preprint"})
        w2, o2 = identity.upsert_work(
            s, {"doi": "10.48550/arxiv.2401.01234v2", "title": "Same Preprint"}
        )
        s.commit()
        assert (o1, o2) == ("created", "merged")
        assert w1.work_id == w2.work_id
        assert w2.doi is None
        idents = s.exec(select(Identifier)).all()
        assert {(i.id_type, i.id_value) for i in idents} == {("arxiv", "2401.01234")}


# --------------------------------------------------------------------------
# Chunk 3 — conflicting-strong-id merge blocker (v1 Fix 1)
# --------------------------------------------------------------------------

def test_same_arxiv_diff_doi_does_not_merge():
    # Canonical preprint/published fixture (v1 test_same_arxiv_diff_doi_does_not_merge):
    # same arXiv id, two DOIs. The single-work match says "merge", but the
    # present-and-DIFFERING doi blocks it — two distinct works, one review item.
    h = service.create_project("blocker_diff_doi")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(
            s, {"arxiv": "2401.01234", "doi": "10.1/preprint", "title": "Same Preprint"}
        )
        w2, o2 = identity.upsert_work(
            s, {"arxiv": "2401.01234", "doi": "10.2/published", "title": "Same Preprint"}
        )
        s.commit()
        assert (o1, o2) == ("created", "duplicate_candidate")
        assert w1.work_id != w2.work_id
        assert len(s.exec(select(Work)).all()) == 2
        # neither work carries the other's DOI — identifier rows stay disjoint
        w1_ids = {
            (i.id_type, i.id_value)
            for i in s.exec(select(Identifier).where(Identifier.work_id == w1.work_id)).all()
        }
        w2_ids = {
            (i.id_type, i.id_value)
            for i in s.exec(select(Identifier).where(Identifier.work_id == w2.work_id)).all()
        }
        assert w1_ids == {("arxiv", "2401.01234"), ("doi", "10.1/preprint")}
        assert w2_ids == {("doi", "10.2/published")}
        assert (w1.doi, w2.doi) == ("10.1/preprint", "10.2/published")
        # exactly one review item: id_disagreement, targeting the blocked
        # newcomer, naming the matched work
        items = s.exec(select(ReviewQueueItem)).all()
        assert len(items) == 1
        assert items[0].target_id == w2.work_id
        payload = json.loads(items[0].payload)
        assert payload["reason"] == "id_disagreement"
        assert payload["conflicting_work_ids"] == [w1.work_id]


def test_blocked_merge_shared_id_ownership_asymmetry():
    # Pin (not forbid) the SHARED-id asymmetry on the blocked pair: _create_work
    # mirrors ALL incoming strong ids onto the new work's scalar columns, while
    # _attach_unclaimed_ids skips the already-claimed arXiv id — so the blocked
    # work's arxiv_id column carries the shared value but the identifiers row
    # stays owned by the matched work. Same asymmetry as the >=2-match
    # cross_id_collision path; this test documents it.
    h = service.create_project("blocker_asymmetry")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, _ = identity.upsert_work(
            s, {"arxiv": "2401.01234", "doi": "10.1/preprint", "title": "Same Preprint"}
        )
        w2, outcome = identity.upsert_work(
            s, {"arxiv": "2401.01234", "doi": "10.2/published", "title": "Same Preprint"}
        )
        s.commit()
        assert outcome == "duplicate_candidate"
        assert w2.arxiv_id == "2401.01234"  # scalar mirror set on the blocked work
        rows = s.exec(
            select(Identifier).where(
                Identifier.id_type == "arxiv", Identifier.id_value == "2401.01234"
            )
        ).all()
        assert [r.work_id for r in rows] == [w1.work_id]


def test_same_arxiv_same_doi_still_merges():
    # No disagreement, no demotion: identical strong ids merge exactly as before.
    h = service.create_project("blocker_same_doi")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(
            s, {"arxiv": "2401.01234", "doi": "10.1/final", "title": "Same Paper"}
        )
        w2, o2 = identity.upsert_work(
            s, {"arxiv": "2401.01234", "doi": "10.1/final", "title": "Same Paper"}
        )
        s.commit()
        assert (o1, o2) == ("created", "merged")
        assert w1.work_id == w2.work_id
        assert len(s.exec(select(Work)).all()) == 1
        assert s.exec(select(ReviewQueueItem)).all() == []


# --------------------------------------------------------------------------
# Chunk 4 — weak-id title-corroboration guard (v1 Fix 5)
# --------------------------------------------------------------------------

def test_weak_id_title_conflict_same_s2_different_title_demotes():
    # Port of v1 test_same_s2_conflicting_title_does_not_merge (prototype
    # tests/test_truth_layer_fixes.py:84-88): a single-work match shared ONLY on
    # a weak id (s2) with conflicting titles never silently merges — two works,
    # one review item with the new reason.
    h = service.create_project("weak_guard_conflict")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(s, {"s2": "abc", "title": "Paper One"})
        w2, o2 = identity.upsert_work(
            s, {"s2": "abc", "title": "Totally Different Paper"}
        )
        s.commit()
        assert (o1, o2) == ("created", "duplicate_candidate")
        assert w1.work_id != w2.work_id
        assert len(s.exec(select(Work)).all()) == 2
        # the s2 identifier row stays owned by the matched work (never re-attached)
        rows = s.exec(
            select(Identifier).where(
                Identifier.id_type == "s2", Identifier.id_value == "abc"
            )
        ).all()
        assert [r.work_id for r in rows] == [w1.work_id]
        items = s.exec(select(ReviewQueueItem)).all()
        assert len(items) == 1
        assert items[0].target_id == w2.work_id
        payload = json.loads(items[0].payload)
        assert payload["reason"] == "weak_id_title_conflict"
        assert payload["conflicting_work_ids"] == [w1.work_id]


def test_same_s2_same_title_still_merges():
    # v1 test_same_s2_same_title_merges: corroborating title, weak id merges.
    h = service.create_project("weak_guard_same_title")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(s, {"s2": "abc", "title": "Same Paper"})
        w2, o2 = identity.upsert_work(s, {"s2": "abc", "title": "Same Paper"})
        s.commit()
        assert (o1, o2) == ("created", "merged")
        assert w1.work_id == w2.work_id
        assert len(s.exec(select(Work)).all()) == 1
        assert s.exec(select(ReviewQueueItem)).all() == []


def test_same_doi_different_title_still_merges():
    # Strong id wins (v1 parity): title evidence never blocks a doi/openalex
    # merge; the existing title survives (backfill is empty-only).
    h = service.create_project("weak_guard_doi_wins")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(s, {"doi": "10.1/x", "title": "Paper One"})
        w2, o2 = identity.upsert_work(
            s, {"doi": "10.1/x", "title": "Totally Different Paper"}
        )
        s.commit()
        assert (o1, o2) == ("created", "merged")
        assert w1.work_id == w2.work_id
        assert len(s.exec(select(Work)).all()) == 1
        assert w1.canonical_title == "Paper One"
        assert s.exec(select(ReviewQueueItem)).all() == []


def test_weak_id_match_missing_title_hash_still_merges():
    # The guard needs BOTH title hashes present; a title-less side cannot
    # conflict, so the weak-id merge (and its title backfill) proceeds.
    h = service.create_project("weak_guard_no_title")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(s, {"s2": "abc"})
        w2, o2 = identity.upsert_work(s, {"s2": "abc", "title": "Now Titled"})
        s.commit()
        assert (o1, o2) == ("created", "merged")
        assert w1.work_id == w2.work_id
        assert w1.canonical_title == "Now Titled"
        assert s.exec(select(ReviewQueueItem)).all() == []


# --------------------------------------------------------------------------
# Chunks 6-8 — merge executor, review appliers, manual override, doctor scan
# --------------------------------------------------------------------------

import os  # noqa: E402
import sqlite3  # noqa: E402
from pathlib import Path  # noqa: E402

from seedgraph.citation.edges import authoritative_edges, write_edge  # noqa: E402
from seedgraph.errors import ValidationError  # noqa: E402
from seedgraph.project import merge, review as review_mod  # noqa: E402
from seedgraph.project.doctor_reconcile import work_refs_findings  # noqa: E402
from seedgraph.project.merge import FK_REMAP_MANIFEST, merge_works  # noqa: E402


def _conn(h) -> sqlite3.Connection:
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _open_items(h, item_type=None):
    items = review_mod.list_open(h)
    if item_type:
        items = [i for i in items if i.item_type == item_type]
    return items


_NOW = "2026-07-02T00:00:00+00:00"


def _seed_merge_fixture(h):
    """Full fixture: survivor S, victim V, bystanders X/Y, rows in EVERY table
    referencing works (hard FKs + all three soft refs)."""
    with Session(h.engine, expire_on_commit=False) as s:
        S, _ = identity.upsert_work(s, {"doi": "10.1/s"})
        V, _ = identity.upsert_work(
            s, {"doi": "10.1/v", "arxiv": "1111.2222", "title": "Victim Title",
                "authors": ["A. Uthor"], "venue": "VenueV", "year": 2020},
        )
        X, _ = identity.upsert_work(s, {"doi": "10.1/x", "title": "X"})
        Y, _ = identity.upsert_work(s, {"doi": "10.1/y", "title": "Y"})
        # an OPEN review item targeting the victim (soft ref -> retargeted)
        item_id = review_mod.enqueue_in_session(
            s, "duplicate_candidate", target_type="work", target_id=V.work_id,
            payload={"kind": "duplicate_candidate", "reason": "cross_id_collision",
                     "conflicting_work_ids": [S.work_id], "incoming": {}},
        )
        s.commit()
    sid, vid, xid, yid = S.work_id, V.work_id, X.work_id, Y.work_id

    conn = _conn(h)
    try:
        now = _NOW
        # citation_edges: X->V (remapped X->S COLLIDES with pre-existing X->S),
        # V->S (SELF-LOOP on remap), V->Y (remaps cleanly to S->Y).
        write_edge(conn, source=xid, target=vid, provenance="provider_reference",
                   confidence=1.0, run_id="R")
        write_edge(conn, source=xid, target=sid, provenance="provider_reference",
                   confidence=1.0, run_id="R")
        write_edge(conn, source=vid, target=sid, provenance="provider_reference",
                   confidence=1.0, run_id="R")
        write_edge(conn, source=vid, target=yid, provenance="provider_reference",
                   confidence=1.0, run_id="R")
        # reference_entries: V cites Y; X's entry resolved to V.
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, "
            "raw_reference_text, resolved_work_id, resolution_status, created_at) "
            "VALUES ('ref_v', ?, 'raw v', ?, 'resolved', ?)", (vid, yid, now))
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, "
            "raw_reference_text, resolved_work_id, resolution_status, created_at) "
            "VALUES ('ref_x', ?, 'raw x', ?, 'resolved', ?)", (xid, vid, now))
        # bridges on BOTH (survivor's kept; victim's dropped — UNIQUE(work_id)).
        conn.execute(
            "INSERT INTO work_source_files (work_id, source_file_id, file_hash, "
            "acquisition_method, created_at, updated_at) "
            "VALUES (?, 'sf_s', 'hash_s', 'manual_upload', ?, ?)", (sid, now, now))
        conn.execute(
            "INSERT INTO work_source_files (work_id, source_file_id, file_hash, "
            "acquisition_method, created_at, updated_at) "
            "VALUES (?, 'sf_v', 'hash_v', 'manual_upload', ?, ?)", (vid, now, now))
        # membership: S metadata_only, V included w/ access -> S adopts included.
        conn.execute(
            "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
            "created_at, updated_at) VALUES (?, 'metadata_only', 0, ?, ?)",
            (sid, now, now))
        conn.execute(
            "INSERT INTO project_documents (work_id, inclusion_status, is_seed, "
            "access_status, created_at, updated_at) "
            "VALUES (?, 'included', 0, 'open_access', ?, ?)", (vid, now, now))
        # plain work_id tables.
        conn.execute(
            "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, "
            "source_file_id, source_file_hash, work_id, level, ordinal, start_char, "
            "end_char, section_parser_version, created_at) "
            "VALUES ('sec_1', 'md_1', 'mh', 'sf_v', 'hash_v', ?, 1, 0, 0, 10, 'v1', ?)",
            (vid, now))
        conn.execute(
            "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, "
            "source_file_id, source_file_hash, work_id, start_char, end_char, "
            "exact_quote, quote_hash, created_at) "
            "VALUES ('span_1', 'md_1', 'mh', 'sf_v', 'hash_v', ?, 0, 5, 'quote', 'qh', ?)",
            (vid, now))
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
            "markdown_hash, schema_version, prompt_version, run_status, created_at) "
            "VALUES ('extr_1', ?, 'md_1', 'mh', '1', '1', 'success', ?)", (vid, now))
        conn.execute(
            "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, "
            "markdown_id, markdown_hash, schema_version, prompt_version, "
            "raw_note_json, note_text, created_at) "
            "VALUES ('note_1', 'extr_1', ?, 'md_1', 'mh', '1', '1', '{}', 'nt', ?)",
            (vid, now))
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, extraction_run_id, work_id, "
            "claim_type, field_key, status, epistemic_type, created_at) "
            "VALUES ('claim_1', 'extr_1', ?, 'method', 'f', 'found', 'llm_extracted', ?)",
            (vid, now))
        conn.execute(
            "INSERT INTO lenses (lens_id, name, object_type, yaml_path, "
            "definition_hash, created_at, updated_at) "
            "VALUES ('lens_1', 'L', 'assumption', 'l.yaml', 'dh', ?, ?)", (now, now))
        conn.execute(
            "INSERT INTO lens_outputs (lens_output_id, extraction_run_id, lens_id, "
            "work_id, status, fields_json, created_at) "
            "VALUES ('lensout_1', 'extr_1', 'lens_1', ?, 'found', '{}', ?)", (vid, now))
        for c in ("c1", "c2", "c3"):
            conn.execute(
                "INSERT INTO concepts (concept_id, normalized_label, canonical_label, "
                "concept_type, created_at, updated_at) VALUES (?, ?, ?, 'method', ?, ?)",
                (f"concept::{c}", c, c.upper(), now, now))
        conn.execute(
            "INSERT INTO claim_concepts (claim_id, concept_id, work_id, created_at) "
            "VALUES ('claim_1', 'concept::c1', ?, ?)", (vid, now))
        # soft ref: span_fts row on the victim.
        conn.execute(
            "INSERT INTO span_fts (quote_text, span_id, markdown_id, work_id, section_id) "
            "VALUES ('quote', 'span_1', 'md_1', ?, 'sec_1')", (vid,))
        # soft ref: project_graph_edges — a machine discusses row (remaps), a
        # user_validated row (remaps; would dangle PERMANENTLY if missed), and a
        # would-collide pair (V->c1 vs pre-existing S->c1, same key -> dropped).
        pge = ("INSERT INTO project_graph_edges (edge_id, source_node_type, "
               "source_node_id, target_node_type, target_node_id, edge_type, "
               "epistemic_type, run_id, created_at) VALUES (?, 'Work', ?, "
               "'Concept', ?, 'discusses', ?, 'R', ?)")
        conn.execute(pge, ("pge_m", vid, "concept::c3", "llm_extracted", now))
        conn.execute(pge, ("pge_u", vid, "concept::c2", "user_validated", now))
        conn.execute(pge, ("pge_v1", vid, "concept::c1", "llm_extracted", now))
        conn.execute(pge, ("pge_s1", sid, "concept::c1", "llm_extracted", now))
        conn.commit()
    finally:
        conn.close()
    return sid, vid, xid, yid, item_id


def test_merge_works_full_fixture_remaps_everything():
    h = service.create_project("mergefull")
    sid, vid, xid, yid, item_id = _seed_merge_fixture(h)

    with Session(h.engine, expire_on_commit=False) as s:
        survivor = merge_works(s, sid, vid)
        assert survivor.work_id == sid
        s.commit()

    conn = _conn(h)
    try:
        # victim gone; survivor scalars backfilled EMPTY-ONLY (own doi kept).
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (vid,)).fetchone() is None
        row = conn.execute(
            "SELECT canonical_title, arxiv_id, doi, venue, year FROM works WHERE work_id=?",
            (sid,)).fetchone()
        assert row == ("Victim Title", "1111.2222", "10.1/s", "VenueV", 2020)

        # identifiers: victim's doi+arxiv remapped to the survivor (CASCADE never
        # the mechanism — the rows SURVIVE).
        idrows = {tuple(r) for r in conn.execute(
            "SELECT id_type, id_value FROM identifiers WHERE work_id=?", (sid,)).fetchall()}
        assert {("doi", "10.1/s"), ("doi", "10.1/v"), ("arxiv", "1111.2222")} <= idrows

        # citation_edges: X->V collided with X->S (dropped), V->S self-looped
        # (dropped), V->Y remapped to S->Y. No victim endpoint anywhere.
        edges = {tuple(r) for r in conn.execute(
            "SELECT source_work_id, target_work_id FROM citation_edges").fetchall()}
        assert edges == {(xid, sid), (sid, yid)}

        # reference_entries remapped on BOTH columns.
        assert conn.execute(
            "SELECT citing_work_id FROM reference_entries WHERE reference_id='ref_v'"
        ).fetchone()[0] == sid
        assert conn.execute(
            "SELECT resolved_work_id FROM reference_entries WHERE reference_id='ref_x'"
        ).fetchone()[0] == sid

        # bridge: survivor's row kept, victim's dropped (UNIQUE(work_id) policy).
        assert conn.execute(
            "SELECT work_id, file_hash FROM work_source_files").fetchall() == [(sid, "hash_s")]

        # membership: one row; survivor adopted included + the victim's access.
        assert conn.execute(
            "SELECT work_id, inclusion_status, access_status FROM project_documents"
        ).fetchall() == [(sid, "included", "open_access")]

        # plain work_id tables all remapped.
        for table in ("document_sections", "evidence_spans", "extraction_runs",
                      "structured_notes", "extracted_claims", "lens_outputs",
                      "claim_concepts"):
            vals = {r[0] for r in conn.execute(f"SELECT work_id FROM {table}").fetchall()}
            assert vals == {sid}, f"{table} not fully remapped: {vals}"

        # soft refs: span_fts + the open review item retargeted (still open).
        assert {r[0] for r in conn.execute("SELECT work_id FROM span_fts").fetchall()} == {sid}
        assert conn.execute(
            "SELECT target_id, status FROM review_queue WHERE item_id=?", (item_id,)
        ).fetchone() == (sid, "open")

        # project_graph_edges: machine + user_validated rows remapped; the
        # would-collide pair collapsed to ONE row; zero Work-victim endpoints.
        pge = conn.execute(
            "SELECT edge_id, source_node_id, target_node_id FROM project_graph_edges"
        ).fetchall()
        edge_ids = {e[0] for e in pge}
        assert "pge_m" in edge_ids and "pge_u" in edge_ids
        c1_rows = [e for e in pge if e[2] == "concept::c1"]
        assert len(c1_rows) == 1 and c1_rows[0][1] == sid
        assert all(e[1] != vid and e[2] != vid for e in pge)

        # integrity: FK check clean; zero dangling polymorphic endpoints; zero
        # dangling soft work refs.
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        from seedgraph.doctor import semantic_graph_findings
        dangling, _orphans = semantic_graph_findings(conn)
        assert dangling == []
        soft_dangling, _drift = work_refs_findings(conn)
        assert soft_dangling == []
    finally:
        conn.close()


def test_merge_works_noop_and_missing_raise():
    h = service.create_project("mergeguards")
    with Session(h.engine, expire_on_commit=False) as s:
        w, _ = identity.upsert_work(s, {"doi": "10.1/only", "title": "Only"})
        s.commit()
        # no-op: survivor == victim
        out = merge_works(s, w.work_id, w.work_id)
        assert out.work_id == w.work_id
        assert s.get(Work, w.work_id) is not None
        # missing victim / survivor raise (the executor never guesses)
        with pytest.raises(ValidationError):
            merge_works(s, w.work_id, "work_missing")
        with pytest.raises(ValidationError):
            merge_works(s, "work_missing", w.work_id)


def test_fk_coverage_invariant_every_works_fk_in_manifest():
    """Any FUTURE FK into works(work_id) must be added to merge.FK_REMAP_MANIFEST —
    this test fails the suite instead of the data silently dangling."""
    h = service.create_project("fkcov")
    conn = _conn(h)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        found: set[tuple[str, str]] = set()
        for table in tables:
            for row in conn.execute(f"PRAGMA foreign_key_list({table})").fetchall():
                # row: (id, seq, ref_table, from_col, to_col, on_update, on_delete, match)
                if row[2] == "works":
                    found.add((table, row[3]))
    finally:
        conn.close()
    manifest = set(FK_REMAP_MANIFEST)
    missing = found - manifest
    assert not missing, (
        f"FK(s) into works not covered by merge.FK_REMAP_MANIFEST: {sorted(missing)} "
        f"— extend merge.py's remap list (and its soft-ref manifest if FK-less)"
    )
    stale = manifest - found
    assert not stale, f"manifest names non-existent FK(s): {sorted(stale)}"


# --------------------------------------------------------------------------
# Chunk 7 — duplicate_candidate appliers (merge / exclude / reject)
# --------------------------------------------------------------------------

def _blocker_pair(h):
    """The canonical preprint/published pair: same arXiv, different DOIs."""
    with Session(h.engine, expire_on_commit=False) as s:
        w1, _ = identity.upsert_work(s, {"arxiv": "2001.00001", "doi": "10.1/aaa", "title": "T"})
        w2, _ = identity.upsert_work(s, {"arxiv": "2001.00001", "doi": "10.1/bbb", "title": "T"})
        s.commit()
    [item] = _open_items(h, "duplicate_candidate")
    return w1.work_id, w2.work_id, item.item_id


def test_resolve_merge_default_survivor_collapses_and_retargets():
    h = service.create_project("resolvemerge")
    w1, w2, item_id = _blocker_pair(h)
    review_mod.resolve(h, item_id, "merge")

    # post-crash-style verification: a FRESH connection sees the collapse AND the
    # status flip together (one commit — decision 80).
    conn = _conn(h)
    try:
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (w2,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (w1,)).fetchone() is not None
        status, action, target = conn.execute(
            "SELECT status, action, target_id FROM review_queue WHERE item_id=?",
            (item_id,)).fetchone()
        assert (status, action) == ("resolved", "merge")
        assert target == w1  # item retargeted to the survivor (v1 detail)
        # both DOIs now owned by the survivor
        ids = {r[0] for r in conn.execute(
            "SELECT id_value FROM identifiers WHERE work_id=? AND id_type='doi'", (w1,))}
        assert ids == {"10.1/aaa", "10.1/bbb"}
    finally:
        conn.close()

    # idempotent re-resolve is a no-op (no error, nothing changes)
    review_mod.resolve(h, item_id, "merge")


def test_resolve_merge_survivor_override():
    h = service.create_project("resolveovr")
    w1, w2, item_id = _blocker_pair(h)
    review_mod.resolve(h, item_id, "merge", survivor_id=w2)
    conn = _conn(h)
    try:
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (w1,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (w2,)).fetchone() is not None
    finally:
        conn.close()
    # a survivor outside the pair is refused
    h2 = service.create_project("resolveovr2")
    _a1, _a2, item2 = _blocker_pair(h2)
    with pytest.raises(ValidationError):
        review_mod.resolve(h2, item2, "merge", survivor_id="work_outsider")


def test_resolve_merge_stale_sibling_refused():
    h = service.create_project("resolvestale")
    w1, w2, item_id = _blocker_pair(h)
    # a sibling item names w2 as its conflicting (would-be survivor) work...
    with Session(h.engine, expire_on_commit=False) as s:
        w3, _ = identity.upsert_work(s, {"doi": "10.9/zzz", "title": "Z"})
        sibling = review_mod.enqueue_in_session(
            s, "duplicate_candidate", target_type="work", target_id=w3.work_id,
            payload={"kind": "duplicate_candidate", "reason": "cross_id_collision",
                     "conflicting_work_ids": [w2], "incoming": {}},
        )
        s.commit()
        w3_id = w3.work_id
    # ...then w2 is collapsed away by resolving the first item.
    review_mod.resolve(h, item_id, "merge")
    # resolving the sibling now names a vanished work: REFUSED with a message
    # naming it; item stays open; no action='merge' row recorded.
    with pytest.raises(ValidationError, match=w2):
        review_mod.resolve(h, sibling, "merge")
    conn = _conn(h)
    try:
        assert conn.execute(
            "SELECT status, action FROM review_queue WHERE item_id=?", (sibling,)
        ).fetchone() == ("open", None)
        assert conn.execute(
            "SELECT 1 FROM works WHERE work_id=?", (w3_id,)).fetchone() is not None
    finally:
        conn.close()


def test_resolve_merge_atomic_rollback_on_failure(monkeypatch):
    h = service.create_project("resolveatomic")
    _w1, w2, item_id = _blocker_pair(h)

    def _boom(session, survivor, victim):
        raise RuntimeError("injected merge failure")

    monkeypatch.setattr(merge, "merge_works", _boom)
    with pytest.raises(RuntimeError):
        review_mod.resolve(h, item_id, "merge")
    conn = _conn(h)
    try:
        # nothing committed: both works intact, item still open (atomicity).
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (w2,)).fetchone() is not None
        assert conn.execute(
            "SELECT status FROM review_queue WHERE item_id=?", (item_id,)
        ).fetchone()[0] == "open"
    finally:
        conn.close()


def test_resolve_exclude_sets_membership_and_graph_drops_node():
    from seedgraph.graph.build import build_citation_graph

    h = service.create_project("resolveexcl")
    a = service.add_work(h, ids={"doi": "10.2/a"}, title="A").work_id
    b = service.add_work(h, ids={"doi": "10.2/b"}, title="B").work_id
    conn = _conn(h)
    try:
        write_edge(conn, source=a, target=b, provenance="provider_reference",
                   confidence=1.0, run_id="R")
        conn.commit()
        assert set(build_citation_graph(conn, run_id="R").nodes) == {a, b}
    finally:
        conn.close()
    item_id = review_mod.enqueue(
        h, "duplicate_candidate", target_type="work", target_id=b,
        payload={"kind": "duplicate_candidate", "reason": "title_collision",
                 "conflicting_work_ids": [a], "incoming": {}},
    )
    review_mod.resolve(h, item_id, "exclude")
    conn = _conn(h)
    try:
        assert conn.execute(
            "SELECT inclusion_status, inclusion_reason FROM project_documents "
            "WHERE work_id=?", (b,)).fetchone() == ("excluded", "user_excluded")
        g = build_citation_graph(conn, run_id="R")
        assert b not in set(g.nodes)  # excluded node dropped from the graph
        assert g.number_of_edges() == 0
    finally:
        conn.close()


def test_resolve_reject_changes_nothing():
    h = service.create_project("resolverej")
    _w1, _w2, item_id = _blocker_pair(h)
    review_mod.resolve(h, item_id, "reject")
    conn = _conn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM works").fetchone()[0] == 2
        assert conn.execute(
            "SELECT status, action FROM review_queue WHERE item_id=?", (item_id,)
        ).fetchone() == ("resolved", "reject")
        assert conn.execute("SELECT COUNT(*) FROM project_documents").fetchone()[0] == 0
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Chunk 8 — manual-override applier for citation_resolution (+ migration 0014)
# --------------------------------------------------------------------------

MO_MD = (
    "# Citing Paper\n\nBody text.\n\n## References\n\n"
    "Doe, J. (2020). A quite unfindable manuscript title. Some Journal. "
    "https://doi.org/10.7777/miss\n"
)


def _bridged_citing_work(slug):
    """Project + citing work with a bridged, sectioned references markdown whose
    single entry carries a DOI the (empty) chain will MISS."""
    from seedgraph import cache_access
    from seedgraph.acquisition.bridge import write_bridge
    from seedgraph.cache.convert import convert_source_file
    from seedgraph.cache.ingest import ingest_file
    from seedgraph.cache.marker_backend import FakeMarkerBackend
    from seedgraph.db.adapter import raw_conn
    from seedgraph.sections.parser import parse_sections
    from seedgraph.sections.store import replace_sections
    from seedgraph.vocab import AccessClass, AcquisitionMethod

    h = service.create_project(slug)
    a = service.add_work(h, ids={"doi": "10.1000/citing"}, title="Citing Paper").work_id
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{slug}.pdf"
    p.write_bytes(b"%PDF-1.4 " + slug.encode() + b" body words here")
    src = ingest_file(p, access_class=AccessClass.open_access,
                      acquisition_method=AcquisitionMethod.open_access_fetch, root=None)
    md = convert_source_file(src.source_file_id,
                             backend=FakeMarkerBackend(markdown=MO_MD), root=None)
    with Session(h.engine) as s:
        write_bridge(s, work_id=a, source_file_id=src.source_file_id,
                     file_hash=src.file_hash, markdown_id=md.markdown_id,
                     markdown_hash=md.markdown_hash,
                     acquisition_method="open_access_fetch")
        conn = raw_conn(s)
        cache_conn = cache_access.open_cache_ro(None)
        try:
            row = cache_access.read_markdown(cache_conn, None, md.markdown_id)
        finally:
            cache_conn.close()
        secs = parse_sections(row.text, markdown_id=md.markdown_id,
                              markdown_hash=row.markdown_hash,
                              source_file_id=row.source_file_id,
                              source_file_hash=row.source_file_hash, work_id=a)
        replace_sections(conn, md.markdown_id, secs)
        s.commit()
    return h, a


class _EmptyProviders:
    """Provider chain double that misses everything (offline)."""

    async def by_doi(self, doi):
        return None

    async def by_title(self, title, year=None):
        return None


def test_manual_override_approve_writes_edge_beats_parsed_and_persists():
    from seedgraph.citation.parsed_bib import build_parsed_edges
    from seedgraph.project import identity as identity_mod

    h, a = _bridged_citing_work("manoverride")
    # Stage A: the parsed DOI misses the chain -> AMBIGUOUS (ch5), review item,
    # no edge, no self-minted confident identity.
    res = build_parsed_edges(h, None, _EmptyProviders(), identity_mod,
                             work_id=a, run_id="R1")
    assert res.ambiguous == 1 and res.resolved == 0 and res.edges_written == 0
    [item] = _open_items(h, "citation_resolution")
    payload = json.loads(item.payload)
    assert payload["doi"] == "10.7777/miss"
    assert payload["candidates"] and payload["candidates"][0]["doi"] == "10.7777/miss"
    ref_id = payload["reference_id"]

    # approve: single candidate default -> upsert metadata_only + manual_override
    # edge under the payload's run, reference row durably corrected — atomically.
    review_mod.resolve(h, item.item_id, "approve")

    conn = _conn(h)
    try:
        chosen = conn.execute(
            "SELECT work_id FROM works WHERE doi='10.7777/miss'").fetchone()[0]
        # not-yet-a-work candidate was upserted metadata_only
        assert conn.execute(
            "SELECT inclusion_status FROM project_documents WHERE work_id=?",
            (chosen,)).fetchone()[0] == "metadata_only"
        # the manual_override edge exists under the payload's run
        assert conn.execute(
            "SELECT provenance, confidence, reference_id FROM citation_edges "
            "WHERE source_work_id=? AND target_work_id=? AND run_id='R1'",
            (a, chosen)).fetchall() == [("manual_override", 1.0, ref_id)]
        # seed the parsed tier for the same pair: the authority ladder picks the
        # override (extends the phase_2 forward-seam assertion)
        write_edge(conn, source=a, target=chosen, provenance="parsed_bibliography",
                   confidence=0.8, run_id="R1", reference_id=ref_id)
        conn.commit()
        auth = [e for e in authoritative_edges(conn, "R1")
                if e["source_work_id"] == a and e["target_work_id"] == chosen]
        assert len(auth) == 1 and auth[0]["provenance"] == "manual_override"
        # the reference row is durably corrected (migration 0014 admits the value)
        assert conn.execute(
            "SELECT resolved_work_id, resolution_status, resolution_source, confidence "
            "FROM reference_entries WHERE reference_id=?", (ref_id,)
        ).fetchone() == (chosen, "resolved", "manual_override", 1.0)
        # item flipped in the same transaction
        assert conn.execute(
            "SELECT status, action FROM review_queue WHERE item_id=?",
            (item.item_id,)).fetchone() == ("resolved", "approve")
    finally:
        conn.close()

    # a FUTURE Stage-B projection under a NEW run targets the human-chosen work
    res2 = build_parsed_edges(h, None, _EmptyProviders(), identity_mod,
                              work_id=a, run_id="R2")
    assert res2.reparsed is False and res2.edges_written == 1
    conn = _conn(h)
    try:
        tgt = conn.execute(
            "SELECT target_work_id FROM citation_edges "
            "WHERE run_id='R2' AND provenance='parsed_bibliography'").fetchone()[0]
        assert tgt == chosen
    finally:
        conn.close()


def test_manual_override_empty_run_id_refused_and_reject_writes_nothing():
    h = service.create_project("manoverridebad")
    a = service.add_work(h, ids={"doi": "10.3/citing"}, title="Citing").work_id
    item_id = review_mod.enqueue(
        h, "citation_resolution", target_type="reference_entry", target_id="ref_x",
        payload={"kind": "citation_resolution", "status": "ambiguous",
                 "citing_work_id": a, "reference_id": "ref_x", "raw": "raw",
                 "candidates": [{"doi": "10.4/cand"}], "run_id": ""},
    )
    with pytest.raises(ValidationError, match="run_id"):
        review_mod.resolve(h, item_id, "approve")
    conn = _conn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 0
        assert conn.execute(
            "SELECT status FROM review_queue WHERE item_id=?", (item_id,)
        ).fetchone()[0] == "open"
    finally:
        conn.close()
    # reject is a pure status flip: no edge, no reference update, raw kept
    review_mod.resolve(h, item_id, "reject")
    conn = _conn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0] == 0
        assert conn.execute(
            "SELECT status, action FROM review_queue WHERE item_id=?", (item_id,)
        ).fetchone() == ("resolved", "reject")
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Chunk 6 — work_refs_reconcile doctor scan
# --------------------------------------------------------------------------

def test_work_refs_reconcile_reports_dangling_and_drift():
    h = service.create_project("wrfdoctor")
    a = service.add_work(h, ids={"doi": "10.6/a"}, title="A").work_id
    conn = _conn(h)
    try:
        dangling, drift = work_refs_findings(conn)
        assert dangling == [] and drift == 0
        # a dangling span_fts soft ref + a non-normalized stored identifier
        conn.execute(
            "INSERT INTO span_fts (quote_text, span_id, markdown_id, work_id, section_id) "
            "VALUES ('q', 'span_x', 'md_x', 'work_gone', NULL)")
        conn.execute(
            "INSERT INTO identifiers (work_id, id_type, id_value) "
            "VALUES (?, 'arxiv', 'arXiv:2401.99999v2')", (a,))
        conn.commit()
        dangling, drift = work_refs_findings(conn)
        assert ("span_fts", "work_gone") in dangling
        assert drift == 1
    finally:
        conn.close()
