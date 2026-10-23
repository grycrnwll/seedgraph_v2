"""run_migration end to end on the synthetic v2 home (design §1.3, §4.5, §6, §11 S6)."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest

from fixtures import build_v2_home
from seedgraph.library.store import open_library_at
from seedgraph.migrate import MigrationRefused, run_migration
from seedgraph.migrate import dump as dumpmod
from seedgraph.migrate.v2guard import ProcessInfo

T1 = "2026-10-23T01:00:00+00:00"
T2 = "2026-10-23T02:30:00+00:00"
NO_PROCS = lambda: []  # noqa: E731
BIG_DISK = lambda p: 10 ** 15  # noqa: E731


def tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def run(v2, v3, **kw):
    kw.setdefault("process_lister", NO_PROCS)
    kw.setdefault("disk_free", BIG_DISK)
    return run_migration(v2, v3, **kw)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_v2_home(tmp_path_factory.mktemp("v2") / "home")


@pytest.fixture(scope="module")
def twice(fx, tmp_path_factory):
    """Two from-scratch runs into one v3 home, different snapshot times and run ids."""
    v3 = tmp_path_factory.mktemp("v3") / "home"
    (v3 / "golden_reports").mkdir(parents=True)
    (v3 / "golden_set.json").write_text('{"golden": true}', encoding="utf-8")
    (v3 / "golden_reports" / "r1.md").write_text("report", encoding="utf-8")
    (v3 / "notes-of-mine.txt").write_text("unknown file", encoding="utf-8")
    (v3 / "scratch").mkdir()
    (v3 / "scratch" / "x.bin").write_bytes(b"\x00\x01")
    before_v2 = tree_digest(fx.home)
    outside = {k: v for k, v in tree_digest(v3).items()}
    r1 = run(fx.home, v3, run_id="mig-test-1", now=T1, graph_names={"did": "Difference-in-Differences"})
    r2 = run(fx.home, v3, run_id="mig-test-2", now=T2, graph_names={"did": "Difference-in-Differences"})
    return {"v3": v3, "r1": r1, "r2": r2, "before_v2": before_v2, "outside": outside}


def test_two_runs_give_byte_identical_dumps(twice):
    r1, r2 = twice["r1"], twice["r2"]
    assert r1.snapshot_time != r2.snapshot_time
    assert r1.dump_sha256 == r2.dump_sha256
    assert dumpmod.first_difference(r1.dump_path, r2.dump_path) is None
    assert r1.dump_path.read_bytes() == r2.dump_path.read_bytes()
    assert (r1.run_dir / "dump.sha256").read_text().strip() == r1.dump_sha256


def test_runs_are_clean(twice):
    for r in (twice["r1"], twice["r2"]):
        assert r.errors == []
        assert r.ok and r.swapped
        assert r.guard.status == "pass"
        assert r.plan.ledger.unaccounted() == []


def test_v2_home_is_byte_identical_after_both_runs(fx, twice):
    assert tree_digest(fx.home) == twice["before_v2"]


def test_swap_never_touches_golden_or_unknown_files(twice):
    v3 = twice["v3"]
    now = tree_digest(v3)
    for rel, sha in twice["outside"].items():
        assert now[rel] == sha
    top = sorted(p.name for p in v3.iterdir())
    assert top == ["golden_reports", "golden_set.json", "graphs", "library", "migration", "notes-of-mine.txt",
                   "scratch"]


def test_second_run_moves_the_first_library_aside(twice):
    v3, r1, r2 = twice["v3"], twice["r1"], twice["r2"]
    assert r1.replaced_dir is None
    assert r2.replaced_dir is not None and r2.replaced_dir.parent == v3 / "migration" / "replaced"
    assert (r2.replaced_dir / "library" / "library.db").is_file()
    assert (r2.replaced_dir / "graphs" / "did" / "graph.db").is_file()
    conn = sqlite3.connect(f"file:{(r2.replaced_dir / 'library' / 'library.db').as_posix()}?mode=ro", uri=True)
    assert dict(conn.execute("SELECT key, value FROM library_meta"))["migration_run_id"] == "mig-test-1"
    conn.close()
    live = sqlite3.connect(f"file:{(v3 / 'library' / 'library.db').as_posix()}?mode=ro", uri=True)
    meta = dict(live.execute("SELECT key, value FROM library_meta"))
    live.close()
    assert meta["migration_run_id"] == "mig-test-2" and meta["origin"] == "migration" and meta["dirty"] == "0"
    assert not (v3 / "migration" / "swap_in_progress.json").exists()
    assert not (v3 / "migration" / "staging" / "mig-test-2").exists()


def test_run_folder_holds_dump_and_guards(twice):
    rd = twice["r2"].run_dir
    for name in ("dump.jsonl.gz", "dump.sha256", "v2_guard_before.json", "v2_guard_after.json"):
        assert (rd / name).is_file(), name


def test_blobs_are_copies_not_hardlinks(fx, twice):
    lib = twice["v3"] / "library"
    pdfs = sorted((lib / "pdfs").iterdir())
    mds = sorted((lib / "markdown").iterdir())
    assert len(pdfs) == 24 and len(mds) == 22
    for p in [*pdfs, *mds]:
        assert os.stat(p).st_nlink == 1
        assert hashlib.sha256(p.read_bytes()).hexdigest() == p.stem


def test_display_name_is_stored_slug_fixed(twice):
    conn = sqlite3.connect(f"file:{(twice['v3'] / 'graphs' / 'did' / 'graph.db').as_posix()}?mode=ro", uri=True)
    meta = dict(conn.execute("SELECT key, value FROM graph_meta"))
    conn.close()
    assert meta["slug"] == "did" and meta["display_name"] == "Difference-in-Differences"
    assert meta["created_at"] == "2026-07-03T22:40:00+00:00"


def test_dump_excludes_exactly_the_listed_run_specific_values(twice):
    assert dumpmod.EXCLUDED == (
        "migration_runs", "schema_migrations.applied_at", "library_meta[key=migration_run_id]",
        "attention_items.created_at", "conversion_queue.queued_at")
    lines = [json.loads(x) for x in gzip.open(twice["r2"].dump_path, "rt", encoding="utf-8")]
    tables = {(x["db"], x["table"]) for x in lines if "db" in x}
    assert ("library", "migration_runs") not in tables
    assert ("library", "papers") in tables and ("graph:did", "graph_papers") in tables
    assert ("library", "span_fts") in tables and not any(t.endswith("_data") for _, t in tables)
    for x in lines:
        if x.get("table") == "schema_migrations":
            assert "applied_at" not in x["row"]
        if x.get("table") == "attention_items":
            assert "created_at" not in x["row"]
        if x.get("table") == "library_meta":
            assert x["row"]["key"] != "migration_run_id"
    manifest = [x for x in lines if "manifest" in x]
    assert {m["manifest"] for m in manifest} == {"library/pdfs", "library/markdown"}
    for group in ("library/pdfs", "library/markdown"):
        paths = [m["path"] for m in manifest if m["manifest"] == group]
        assert paths == sorted(paths) and paths
    # the excluded values really vary between runs: the snapshot time is in the live store
    conn = sqlite3.connect(f"file:{(twice['v3'] / 'library' / 'library.db').as_posix()}?mode=ro", uri=True)
    assert {r[0] for r in conn.execute("SELECT created_at FROM attention_items")} == {T2}
    assert conn.execute("SELECT dump_sha256 FROM migration_runs").fetchone()[0] == twice["r2"].dump_sha256
    conn.close()


def test_staging_only_run_leaves_the_live_library(fx, twice):
    v3 = twice["v3"]
    r = run(fx.home, v3, run_id="mig-test-staging", now=T1, swap=False,
            graph_names={"did": "Difference-in-Differences"})
    assert not r.swapped and r.library_dir == v3 / "migration" / "staging" / "mig-test-staging" / "library"
    assert r.dump_sha256 == twice["r1"].dump_sha256
    live = sqlite3.connect(f"file:{(v3 / 'library' / 'library.db').as_posix()}?mode=ro", uri=True)
    assert dict(live.execute("SELECT key, value FROM library_meta"))["migration_run_id"] == "mig-test-2"
    live.close()


def test_a_stage_hook_adds_files_that_are_dumped_and_swapped(fx, tmp_path):
    def hook(ctx):
        d = ctx.staging_dir / "v2_answers" / "did"
        d.mkdir(parents=True)
        (d / "index.md").write_text("# answers\n", encoding="utf-8")

    v3 = tmp_path / "v3"
    r = run(fx.home, v3, run_id="mig-hook", now=T1, stage_hook=hook)
    assert (v3 / "v2_answers" / "did" / "index.md").is_file()
    lines = [json.loads(x) for x in gzip.open(r.dump_path, "rt", encoding="utf-8")]
    assert any(x.get("manifest") == "v2_answers" for x in lines)


# --- refusals --------------------------------------------------------------------------

def test_dirty_library_is_refused_then_replaced_with_the_flag(fx, tmp_path):
    v3 = tmp_path / "v3"
    run(fx.home, v3, run_id="mig-a", now=T1)
    lib = open_library_at(v3 / "library")  # a normal (non-migration) write marks it dirty
    with lib.write() as conn:
        conn.execute("UPDATE papers SET venue = venue WHERE 0")
    lib.close()
    with pytest.raises(MigrationRefused, match="replace-dirty"):
        run(fx.home, v3, run_id="mig-b", now=T1)
    assert not (v3 / "migration" / "staging" / "mig-b").exists()
    r = run(fx.home, v3, run_id="mig-c", now=T1, replace_dirty=True)
    assert r.swapped and (r.replaced_dir / "library" / "library.db").is_file()


def test_a_non_migration_library_is_refused(fx, tmp_path):
    v3 = tmp_path / "v3"
    open_library_at(v3 / "library").close()  # origin = 'local'
    with pytest.raises(MigrationRefused):
        run(fx.home, v3, run_id="mig-x", now=T1)


def test_disk_space_refusal_happens_before_anything_is_written(fx, tmp_path):
    v3 = tmp_path / "v3"
    with pytest.raises(MigrationRefused, match="disk space"):
        run(fx.home, v3, run_id="mig-disk", now=T1, disk_free=lambda p: 1024)
    assert not (v3 / "migration").exists()


def test_a_running_v2_writer_is_refused(fx, tmp_path):
    v3 = tmp_path / "v3"
    procs = lambda: [ProcessInfo(4242, "python -m seedgraph serve --project second_test")]  # noqa: E731
    with pytest.raises(MigrationRefused, match="v2 writer"):
        run(fx.home, v3, run_id="mig-proc", now=T1, process_lister=procs)
    assert not (v3 / "migration").exists()


def test_a_v2_mcp_reader_only_warns(fx, tmp_path):
    procs = lambda: [ProcessInfo(4243, "C:/v2/.venv/Scripts/seedgraph.exe mcp serve")]  # noqa: E731
    r = run(fx.home, tmp_path / "v3", run_id="mig-reader", now=T1, process_lister=procs)
    assert r.swapped and any("read-only v2 reader" in w for w in r.warnings)


def test_a_half_finished_swap_is_reported_on_the_next_run(fx, tmp_path):
    v3 = tmp_path / "v3"
    (v3 / "migration").mkdir(parents=True)
    (v3 / "migration" / "swap_in_progress.json").write_text('{"run_id": "mig-crashed"}', encoding="utf-8")
    r = run(fx.home, v3, run_id="mig-after-crash", now=T1)
    assert any("mig-crashed" in w for w in r.warnings)
