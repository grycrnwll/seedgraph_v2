"""Build C chunk 5 — centrality-ordered extraction worklist + ``--cap``, fully OFFLINE.

``extract notes`` / ``extract notes-chunked`` without ``--work`` order their
worklists by citation in-degree descending (deterministic ``work_id`` tie-break;
zero-in-edge works last) via ``cli._worklist_by_centrality`` (design D7: one
``LEFT JOIN citation_edges ON target_work_id`` + ``GROUP BY`` SQL read), and
``--cap N`` truncates the list so a mid-run budget stop strands the
least-central tail. The ordering choice is echoed in the CLI header AND
recorded durably in the run manifest (gap scan §5.2 build-entails / critic C-4).

Fixtures pin ``works.created_at`` to work_id-ascending order and put the
highest in-degree on the lexicographically LAST work, so the centrality order
is the exact REVERSE of both the pre-chunk-5 ``created_at`` order and a plain
``work_id`` order — a regression to either fails these tests.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from typer.testing import CliRunner

from seedgraph.cli import _worklist_by_centrality, app
from seedgraph.extraction import runner as runner_mod
from seedgraph.extraction.schema import PROMPT_VERSION, SCHEMA_ID, SCHEMA_VERSION
from seedgraph.ids import new_id
from seedgraph.llm.backend import FakeLLMBackend
from seedgraph.project import service
from seedgraph.run import latest_run_id, read_manifest

from test_phase_4 import MD, _add_doc, good_note_dict

cli = CliRunner()


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_project(slug: str, n: int = 3):
    """Project with ``n`` included, bridged-markdown works; return (h, [work_ids], {wid: md})."""
    h = service.create_project(slug)
    wids = []
    mds = {}
    for i in range(n):
        wid, md = _add_doc(h, f"{slug}_{i}", MD, doi=f"10.1/{slug}{i}")
        wids.append(wid)
        mds[wid] = md
    return h, wids, mds


def _set_indegrees(h, degree_by_wid: dict) -> None:
    """Give each work its citation in-degree (distinct in-project sources) and pin
    ``works.created_at`` to work_id-ASC order so the old created_at ordering
    provably disagrees with centrality order in these fixtures."""
    conn = sqlite3.connect(str(h.db_path))
    try:
        all_wids = sorted(degree_by_wid)
        for i, wid in enumerate(all_wids):
            conn.execute(
                "UPDATE works SET created_at=? WHERE work_id=?",
                (f"2020-01-01T00:00:{i:02d}+00:00", wid),
            )
        for wid, deg in degree_by_wid.items():
            sources = [w for w in all_wids if w != wid][:deg]
            assert len(sources) == deg, "fixture needs deg <= n-1 distinct sources"
            for src in sources:
                conn.execute(
                    "INSERT INTO citation_edges (source_work_id, target_work_id, "
                    "edge_type, provenance, confidence, run_id, created_at) "
                    "VALUES (?,?,'cites','provider_reference',1.0,'r1',?)",
                    (src, wid, _now()),
                )
        conn.commit()
    finally:
        conn.close()


def _order_of(output: str, wids: list) -> list:
    """The given work_ids sorted by first appearance in the CLI output."""
    return sorted(wids, key=lambda w: output.index(w))


def _fake_llm(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setattr(
        runner_mod, "_BACKEND_OVERRIDE", FakeLLMBackend(response=good_note_dict())
    )


# --------------------------------------------------------------------------
# extract notes — dry-run order, --cap, tie-break
# --------------------------------------------------------------------------

def test_extract_notes_dry_run_orders_by_in_degree(monkeypatch):
    """0/1/2 in-edge works: dry-run lists them in-degree DESC (zero last), and the
    header names the ordering choice."""
    _fake_llm(monkeypatch)
    h, wids, _mds = _seed_project("centord")
    s = sorted(wids)
    _set_indegrees(h, {s[2]: 2, s[1]: 1, s[0]: 0})

    r = cli.invoke(app, ["extract", "notes", "centord", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "ordered by citation in-degree; cap=none" in r.output
    assert _order_of(r.output, wids) == [s[2], s[1], s[0]]


def test_extract_notes_cap_drops_least_central(monkeypatch):
    """--cap 2 keeps the two most-cited works in order and drops exactly the
    zero-in-degree one."""
    _fake_llm(monkeypatch)
    h, wids, _mds = _seed_project("centcap")
    s = sorted(wids)
    _set_indegrees(h, {s[2]: 2, s[1]: 1, s[0]: 0})

    r = cli.invoke(app, ["extract", "notes", "centcap", "--dry-run", "--cap", "2"])
    assert r.exit_code == 0, r.output
    assert "ordered by citation in-degree; cap=2" in r.output
    assert s[0] not in r.output  # the least-central work is the stranded tail
    assert _order_of(r.output, [s[2], s[1]]) == [s[2], s[1]]


def test_extract_notes_tiebreak_deterministic(monkeypatch):
    """Equal in-degrees break on work_id ASC; two invocations agree exactly."""
    _fake_llm(monkeypatch)
    h, wids, _mds = _seed_project("centtie")
    s = sorted(wids)
    _set_indegrees(h, {s[0]: 1, s[1]: 1, s[2]: 0})

    r1 = cli.invoke(app, ["extract", "notes", "centtie", "--dry-run"])
    r2 = cli.invoke(app, ["extract", "notes", "centtie", "--dry-run"])
    assert r1.exit_code == 0, r1.output
    assert r2.exit_code == 0, r2.output
    # tied works in work_id order, zero-in-degree work last — both runs.
    assert _order_of(r1.output, wids) == [s[0], s[1], s[2]]
    assert _order_of(r2.output, wids) == [s[0], s[1], s[2]]


# --------------------------------------------------------------------------
# the helper's explicit-candidate branch (the notes-chunked path) + dedup
# --------------------------------------------------------------------------

def test_worklist_by_centrality_candidate_set_cap_and_dedup():
    """An explicit candidate set is reordered/capped; duplicate (source, target)
    edges under a second provenance never double-count (COUNT DISTINCT source)."""
    h, wids, _mds = _seed_project("centhelp")
    s = sorted(wids)
    _set_indegrees(h, {s[2]: 2, s[1]: 1, s[0]: 0})
    # s[0] -> s[1] already exists as provider_reference; a parsed_bibliography
    # duplicate must NOT lift s[1] to a tie with s[2].
    conn = sqlite3.connect(str(h.db_path))
    try:
        conn.execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, "
            "provenance, confidence, run_id, created_at) "
            "VALUES (?,?,'cites','parsed_bibliography',1.0,'r1',?)",
            (s[0], s[1], _now()),
        )
        conn.commit()
    finally:
        conn.close()

    assert _worklist_by_centrality(h, None, work_ids=[s[0], s[1], s[2]]) == [s[2], s[1], s[0]]
    assert _worklist_by_centrality(h, 2, work_ids=[s[0], s[1], s[2]]) == [s[2], s[1]]
    assert _worklist_by_centrality(h, None, work_ids=[s[0], s[1]]) == [s[1], s[0]]
    assert _worklist_by_centrality(h, None, work_ids=[]) == []
    # default membership (included + bridged markdown) agrees.
    assert _worklist_by_centrality(h, None) == [s[2], s[1], s[0]]
    assert _worklist_by_centrality(h, 1) == [s[2]]


# --------------------------------------------------------------------------
# durable run-level reproducibility record (manifest)
# --------------------------------------------------------------------------

def test_extract_notes_records_ordering_in_manifest(monkeypatch):
    """A real (non-dry) default-worklist run mints a run and records
    ordering/cap in its manifest; a dry run records nothing."""
    _fake_llm(monkeypatch)
    h, wids, _mds = _seed_project("centman")
    s = sorted(wids)
    _set_indegrees(h, {s[2]: 2, s[1]: 1, s[0]: 0})

    r0 = cli.invoke(app, ["extract", "notes", "centman", "--dry-run", "--cap", "2"])
    assert r0.exit_code == 0, r0.output
    assert latest_run_id("centman") is None  # dry run writes nothing durable

    r = cli.invoke(app, ["extract", "notes", "centman", "--cap", "2"])
    assert r.exit_code == 0, r.output
    rid = latest_run_id("centman")
    assert rid is not None
    section = read_manifest("centman", rid).sections["note_extraction"]
    assert section == {
        "run_id": rid,
        "ordering": "citation_in_degree",
        "cap": 2,
        "works": 2,
    }


# --------------------------------------------------------------------------
# extract notes-chunked — same ordering + cap over the skipped_oversize set
# --------------------------------------------------------------------------

def _seed_skipped_oversize(h, wids: list, mds: dict) -> None:
    """Raw-insert a ``skipped_oversize`` extraction_runs row per work (no note),
    putting each on the phase_4b default worklist."""
    conn = sqlite3.connect(str(h.db_path))
    try:
        for wid in wids:
            md = mds[wid]
            conn.execute(
                "INSERT INTO extraction_runs (extraction_run_id, work_id, "
                "markdown_id, markdown_hash, schema_id, schema_version, "
                "prompt_version, access_class, external_full_text, run_status, "
                "created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'open_access', 0, 'skipped_oversize', ?)",
                (new_id("extr"), wid, md.markdown_id, md.markdown_hash,
                 SCHEMA_ID, SCHEMA_VERSION, PROMPT_VERSION, _now()),
            )
        conn.commit()
    finally:
        conn.close()


def test_extract_notes_chunked_orders_and_caps(monkeypatch):
    """notes-chunked default worklist is in-degree DESC; --cap drops the
    zero-in-degree tail (dry-run: order + header only, no manifest)."""
    _fake_llm(monkeypatch)
    h, wids, mds = _seed_project("centchunk")
    s = sorted(wids)
    _set_indegrees(h, {s[2]: 2, s[1]: 1, s[0]: 0})
    _seed_skipped_oversize(h, wids, mds)

    r = cli.invoke(app, ["extract", "notes-chunked", "centchunk", "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "ordered by citation in-degree; cap=none" in r.output
    assert _order_of(r.output, wids) == [s[2], s[1], s[0]]

    r2 = cli.invoke(
        app,
        ["extract", "notes-chunked", "centchunk", "--dry-run", "--cap", "2"],
    )
    assert r2.exit_code == 0, r2.output
    assert "ordered by citation in-degree; cap=2" in r2.output
    assert s[0] not in r2.output
    assert _order_of(r2.output, [s[2], s[1]]) == [s[2], s[1]]
