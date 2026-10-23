"""Migration core: read-only v2 access, the snapshot and the v2 guard (design §6.1, §6.4)."""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from fixtures import build_v2_home
from seedgraph.migrate import snapshot as snap
from seedgraph.migrate import v2guard

MIGRATE_SRC = Path(snap.__file__).resolve().parent
#: Every migration module (S6's and any later slice's), except the frozen v2 schema loader,
#: which builds v2-shaped databases for the fixtures and is never used by the migration.
MODULES = sorted(p for p in MIGRATE_SRC.rglob("*.py") if "v2_schema" not in p.relative_to(MIGRATE_SRC).parts)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_v2_home(tmp_path_factory.mktemp("v2") / "home")


def tree_digest(root: Path) -> dict[str, str]:
    """An independent sha256 of every file under ``root`` (not via v2guard)."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# --- static: every sqlite connect in migrate/ that can reach v2 is read-only ---------

def _connect_calls(tree: ast.AST):
    """``(enclosing function name, call)`` for every ``sqlite3.connect(...)`` / bare
    ``connect(...)`` imported from sqlite3, anywhere in the module."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    from_sqlite = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                   and n.module == "sqlite3" for a in n.names}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        hit = (isinstance(f, ast.Attribute) and f.attr == "connect" and isinstance(f.value, ast.Name)
               and f.value.id == "sqlite3") or (isinstance(f, ast.Name) and f.id in from_sqlite)
        if hit:
            p = node
            while p in parents and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
                p = parents[p]
            yield (p.name if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<module>"), node


def test_only_the_ro_helper_connects_to_sqlite_directly():
    """``sqlite3.connect`` appears only inside ``snapshot.connect_v2_ro`` and that call
    is a ``mode=ro`` URI. v3 stores are opened through ``library.store`` only."""
    assert len(MODULES) >= 8
    found = []
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found += [(path.stem, fn, call) for fn, call in _connect_calls(tree)]
    assert [(m, fn) for m, fn, _ in found] == [("snapshot", "connect_v2_ro")]
    call = found[0][2]
    kw = {k.arg: k.value for k in call.keywords}
    assert isinstance(kw.get("uri"), ast.Constant) and kw["uri"].value is True
    src = (MIGRATE_SRC / "snapshot.py").read_text(encoding="utf-8")
    assert '"?mode=ro"' in src


def test_migrate_modules_never_build_v2_databases():
    for path in MODULES:
        text = path.read_text(encoding="utf-8")
        assert "create_v2_db" not in text and "apply_v2_schema" not in text, path.name


def test_a_closed_wal_database_is_read_without_creating_sidecar_files(tmp_path, fx):
    home = tmp_path / "home"
    shutil.copytree(fx.home, home)
    db = home / "projects" / "oct2026test" / "project.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    listing = sorted(p.name for p in db.parent.iterdir())
    assert "project.db-wal" not in listing and "project.db-shm" not in listing
    before = tree_digest(home)
    s = snap.take_snapshot(home, now="t")
    v2guard.guard(home)
    assert s.projects["oct2026test"].rows("works")
    assert sorted(p.name for p in db.parent.iterdir()) == listing
    assert tree_digest(home) == before


def test_ro_helper_refuses_writes(fx):
    conn = snap.connect_v2_ro(fx.home / "cache" / "cache.db")
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE x (a)")
    finally:
        conn.close()


# --- snapshot ------------------------------------------------------------------------

def test_snapshot_reads_every_db_and_lists_files_sorted(fx):
    s = snap.take_snapshot(fx.home, now="2026-10-23T00:00:00+00:00")
    assert s.taken_at == "2026-10-23T00:00:00+00:00"
    assert list(s.projects) == ["second_test", "land_value", "test_proj", "oct2026test",
                                "dbg_probe", "dbg_probe2"]
    assert s.cache.schema_version == 3
    assert all(p.schema_version == 17 for p in s.projects.values())
    assert len(s.cache.rows("source_files")) == 24
    assert isinstance(s.cache.data_version, int)
    paths = [f.relpath for f in s.pdf_files]
    assert paths == sorted(paths) and len(paths) == 24
    assert all(f.sha256 == Path(f.relpath).stem for f in s.pdf_files)
    md = s.markdown_files
    assert [f.relpath for f in md] == sorted(f.relpath for f in md)
    # answers: top-level saved answers only, run-scoped answers listed separately
    assert len(s.answer_files["second_test"]) == 2
    assert len(s.answer_files["test_proj"]) == 1
    assert all("/runs/" not in a for a in s.answer_files["second_test"])
    assert s.run_answer_files["second_test"]
    assert "plot_layout_v1.yaml" in "".join(s.lens_files["second_test"])
    assert s.project_yaml["second_test"]["created_at"] == "2026-07-03T22:40:00+00:00"


def test_snapshot_is_equal_on_unchanged_data_and_leaves_v2_untouched(fx):
    before = tree_digest(fx.home)
    a = snap.take_snapshot(fx.home, now="t1")
    b = snap.take_snapshot(fx.home, now="t2")
    assert a.cache.logical_hash == b.cache.logical_hash
    assert {k: v.logical_hash for k, v in a.projects.items()} == \
        {k: v.logical_hash for k, v in b.projects.items()}
    assert tree_digest(fx.home) == before


def test_snapshot_reads_a_wal_database_with_a_live_wal(tmp_path, fx):
    home = tmp_path / "home"
    shutil.copytree(fx.home, home)
    db = home / "projects" / "land_value" / "project.db"
    writer = sqlite3.connect(db)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")
    writer.execute("UPDATE works SET venue = 'edited in the wal' WHERE rowid = 1")
    writer.commit()  # the change now lives in the -wal; keep the connection open
    try:
        assert (db.parent / "project.db-wal").stat().st_size > 0
        g1 = v2guard.guard(home)
        s = snap.take_snapshot(home, now="t")
        assert len(s.projects["land_value"].rows("works")) > 0
        g2 = v2guard.guard(home)
        assert "projects/land_value/project.db-wal" in g1.byte
        assert v2guard.compare(g1, g2).status == "pass"
    finally:
        writer.close()


# --- v2 guard ------------------------------------------------------------------------

def test_guard_passes_on_unchanged_home(fx):
    assert v2guard.compare(v2guard.guard(fx.home), v2guard.guard(fx.home)).status == "pass"


def test_guard_fails_on_a_logical_change(tmp_path, fx):
    home = tmp_path / "home"
    shutil.copytree(fx.home, home)
    before = v2guard.guard(home)
    conn = sqlite3.connect(home / "projects" / "test_proj" / "project.db")
    conn.execute("UPDATE works SET venue = 'changed' WHERE rowid = 1")
    conn.commit()
    conn.close()
    res = v2guard.compare(before, v2guard.guard(home))
    assert res.status == "fail"
    assert any("test_proj" in d for d in res.diffs)


def test_guard_fails_on_a_tree_change(tmp_path, fx):
    home = tmp_path / "home"
    shutil.copytree(fx.home, home)
    before = v2guard.guard(home)
    md = sorted((home / "cache" / "markdown").iterdir())[0]
    md.write_bytes(md.read_bytes() + b"x")
    assert v2guard.compare(before, v2guard.guard(home)).status == "fail"


def test_guard_byte_only_difference_is_a_checkpoint_warning(fx):
    before = v2guard.guard(fx.home)
    after = v2guard.guard(fx.home)
    key = "projects/second_test/project.db"
    after.byte[key] = "0" * 64
    res = v2guard.compare(before, after)
    assert res.status == "warn_checkpoint"


def test_guard_record_round_trips_json(fx):
    g = v2guard.guard(fx.home)
    assert v2guard.GuardRecord.from_json(g.to_json()) == g


# --- process check -------------------------------------------------------------------

def P(pid, cmd, exe="C:/v2/.venv/Scripts/python.exe"):
    return v2guard.ProcessInfo(pid=pid, cmdline=cmd, exe=exe)


def test_process_check_refuses_a_v2_writer_and_warns_on_the_mcp_reader(tmp_path):
    procs = [P(10, "python -m seedgraph serve --project second_test"),
             P(11, "C:/v2/.venv/Scripts/seedgraph.exe mcp serve"),
             P(12, "python unrelated.py"),
             P(13, "seedgraph extract run --project land_value")]
    chk = v2guard.check_processes(tmp_path / "v2", tmp_path / "v3", lister=lambda: procs)
    assert [p.pid for p in chk.v2_writers] == [10, 13]
    assert [p.pid for p in chk.v2_readers] == [11]
    assert not chk.ok


def test_process_check_ok_with_only_readers(tmp_path):
    procs = [P(11, "seedgraph mcp serve")]
    chk = v2guard.check_processes(tmp_path / "v2", tmp_path / "v3", lister=lambda: procs)
    assert chk.ok and chk.warnings


def test_process_check_refuses_a_v3_process_holding_the_library(tmp_path):
    v3 = tmp_path / "v3"
    procs = [P(20, f"seedgraph-v3 serve --root {v3}", exe="C:/v3/.venv/Scripts/python.exe")]
    chk = v2guard.check_processes(tmp_path / "v2", v3, lister=lambda: procs)
    assert [p.pid for p in chk.v3_holders] == [20]
    assert not chk.ok


def test_process_check_refuses_a_hot_rollback_journal(tmp_path, fx):
    home = tmp_path / "home"
    shutil.copytree(fx.home, home)
    (home / "projects" / "test_proj" / "project.db-journal").write_bytes(b"\0" * 512)
    chk = v2guard.check_processes(home, tmp_path / "v3", lister=lambda: [])
    assert not chk.ok
    assert any("journal" in r for r in chk.reasons)


def test_process_check_skips_the_migration_itself_and_its_launchers(tmp_path):
    """`seedgraph migrate run` runs under a launcher .exe (and maybe uv): those ancestors
    are the migration, not a v2 writer."""
    me, parent, grand = os.getpid(), os.getppid(), 999_001
    procs = [v2guard.ProcessInfo(me, "python -m seedgraph migrate run", None, parent),
             v2guard.ProcessInfo(parent, "C:/v3/.venv/Scripts/seedgraph.exe migrate run --v2-home x", None, grand),
             v2guard.ProcessInfo(grand, "uv run seedgraph migrate run", None, 4),
             v2guard.ProcessInfo(55, "seedgraph migrate run", None, 7)]  # someone else's: refused
    chk = v2guard.check_processes(tmp_path / "v2", tmp_path / "v3", lister=lambda: procs)
    assert [p.pid for p in chk.v2_writers] == [55]


def test_an_unavailable_process_list_warns(tmp_path):
    chk = v2guard.check_processes(tmp_path / "v2", tmp_path / "v3", lister=lambda: [])
    assert chk.ok and any("unavailable" in w for w in chk.warnings)
