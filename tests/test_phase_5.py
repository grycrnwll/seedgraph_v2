"""Phase 5 (Project Model MVP) acceptance tests.

Each test maps to a plan §11 acceptance item and runs fully offline / keyless
(no LLM, no network). The module-level imports double as an import-cleanliness
check: the whole phase_5 surface must import without error.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlmodel import Session, select

# Import-cleanliness: these must all import without error at collection time.
from seedgraph.db import engine as db_engine
from seedgraph.db import project_models
from seedgraph.db.project_models import Identifier, ProjectDocument, ReviewQueueItem, Work
from seedgraph.errors import ValidationError
from seedgraph.project import config as project_config
from seedgraph.project import identity, layout, review, service

_SCHEMA_DIR = Path(db_engine.__file__).resolve().parent / "schema"


# --------------------------------------------------------------------------
# test_ids
# --------------------------------------------------------------------------

def test_phase5_ids_shape_and_normalization():
    from seedgraph.ids import new_id

    a, b = new_id("work"), new_id("work")
    assert re.fullmatch(r"work_[0-9a-f]{32}", a)
    assert a != b  # unique across calls

    # case + diacritic + punctuation + whitespace folding, idempotent
    raw = "  Héllo,   WORLD!!  "
    norm = identity.normalize_title(raw)
    assert norm == "hello world"
    assert identity.normalize_title(norm) == norm  # idempotent
    assert identity.normalize_title("Café") == identity.normalize_title("cafe")
    # title_hash is sha1 over the normalized form
    assert identity.title_hash(raw) == identity.sha1_hex(identity.normalize_title(raw))


# --------------------------------------------------------------------------
# test_schema_parity (D6 — ORM-vs-migration parity)
# --------------------------------------------------------------------------

def _affinity(type_str: str) -> str:
    """SQLite type affinity (with the SQLModel JSON column treated as TEXT, since the
    0002 migration declares ``authors`` TEXT and SQLModel serializes JSON to it)."""
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"  # TEXT/VARCHAR/CHAR/CLOB/JSON


_PHASE5_TABLES = {
    "works": Work,
    "identifiers": Identifier,
    "project_documents": ProjectDocument,
}


def test_phase5_schema_parity(tmp_path):
    # Apply ONLY the migrations that shape the phase_5 tables to a fresh tmp db:
    # 0002 (creates works/identifiers/project_documents) and 0017 (Build D ch10 —
    # ALTERs works, adding nullable abstract/oa_status, mirrored on the Work ORM).
    mig_db = tmp_path / "mig.db"
    conn = sqlite3.connect(str(mig_db))
    for name in ("0002_project_model.sql", "0017_works_abstract_oa_status.sql"):
        conn.executescript((_SCHEMA_DIR / "project" / name).read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    insp = inspect(create_engine(f"sqlite:///{mig_db.as_posix()}"))

    # Same table set.
    assert set(insp.get_table_names()) == set(_PHASE5_TABLES)

    orm_dialect = sqlite_dialect.dialect()
    for table_name, model in _PHASE5_TABLES.items():
        orm_cols = {c.name: c for c in model.__table__.columns}
        mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
        assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"

        mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
        orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
        assert mig_pk == orm_pk, f"{table_name}: PK drift"

        for name, orm_col in orm_cols.items():
            mig_col = mig_cols[name]
            orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
            mig_aff = _affinity(str(mig_col["type"]))
            assert orm_aff == mig_aff, f"{table_name}.{name}: type {orm_aff} != {mig_aff}"
            # Nullability matches for non-PK columns (SQLite reflects TEXT PKs as
            # nullable; SQLAlchemy marks PK columns NOT NULL — a known divergence).
            if name not in mig_pk:
                assert orm_col.nullable == mig_col["nullable"], (
                    f"{table_name}.{name}: nullability drift"
                )

        # Indexes (non-unique) match by column-set.
        mig_ix = {tuple(ix["column_names"]) for ix in insp.get_indexes(table_name)}
        orm_ix = {
            tuple(c.name for c in ix.columns)
            for ix in model.__table__.indexes
        }
        assert mig_ix == orm_ix, f"{table_name}: index column-sets drift"

    # identifiers: UNIQUE(id_type, id_value) + FK -> works.work_id
    uq = [set(u["column_names"]) for u in insp.get_unique_constraints("identifiers")]
    assert {"id_type", "id_value"} in uq
    id_fks = insp.get_foreign_keys("identifiers")
    assert any(
        fk["referred_table"] == "works" and fk["constrained_columns"] == ["work_id"]
        for fk in id_fks
    )

    # project_documents: FK -> works.work_id + the inclusion_status CHECK
    pd_fks = insp.get_foreign_keys("project_documents")
    assert any(
        fk["referred_table"] == "works" and fk["constrained_columns"] == ["work_id"]
        for fk in pd_fks
    )
    checks = " ".join(ck["sqltext"] for ck in insp.get_check_constraints("project_documents"))
    assert "inclusion_status" in checks

    # No project_id column crept into any phase_5 table (decision 12/79).
    for table_name in _PHASE5_TABLES:
        assert "project_id" not in {c["name"] for c in insp.get_columns(table_name)}

    # D6: no phase_5 production module authors schema via create_all.
    src_root = Path(db_engine.__file__).resolve().parent.parent  # src/seedgraph
    targets = [
        src_root / "db" / "engine.py",
        src_root / "db" / "project_models.py",
        src_root / "web" / "routes.py",
        *(src_root / "project").glob("*.py"),
    ]
    call_re = re.compile(r"\.create_all\s*\(")
    for path in targets:
        assert not call_re.search(path.read_text(encoding="utf-8")), (
            f"{path.name} calls create_all — violates D6"
        )


def test_phase5_init_project_db_never_uses_create_all(tmp_path, monkeypatch):
    """init_project_db must build the schema via the .sql migrate step, even if
    SQLModel.metadata.create_all is sabotaged (D6 behavioral guard)."""
    import sqlmodel

    def _boom(*_a, **_k):  # pragma: no cover - must never be hit
        raise AssertionError("create_all must not author project.db schema (D6)")

    monkeypatch.setattr(sqlmodel.SQLModel.metadata, "create_all", _boom)

    db_path = tmp_path / "p.db"
    engine = db_engine.make_project_engine(db_path)
    db_engine.init_project_db(engine)  # must not raise
    names = set(inspect(engine).get_table_names())
    assert {"works", "identifiers", "project_documents", "review_queue"} <= names


# --------------------------------------------------------------------------
# test_slug_layout
# --------------------------------------------------------------------------

def test_phase5_slug_layout(tmp_path):
    assert layout.validate_slug("did-demo.v2_1") == "did-demo.v2_1"
    for bad in ("..", ".", "a/b", "a b", "ABC", ""):
        with pytest.raises(ValidationError):
            layout.validate_slug(bad)

    root = tmp_path / "home"
    pdir = layout.project_dir("p", root)
    assert pdir == root / "projects" / "p"
    assert layout.project_db_path("p", root) == pdir / "project.db"
    assert layout.project_yaml_path("p", root) == pdir / "project.yaml"
    assert layout.project_runs_dir("p", root) == pdir / "runs"


# --------------------------------------------------------------------------
# test_identity_merge
# --------------------------------------------------------------------------

def test_phase5_identity_merge_same_doi():
    h = service.create_project("merge_doi")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, o1 = identity.upsert_work(s, {"doi": "10.1/x", "title": "Paper One"})
        w2, o2 = identity.upsert_work(
            s, {"doi": "10.1/x", "arxiv": "2202.0001", "title": "Paper One"}
        )
        s.commit()
        assert o1 == "created"
        assert o2 == "merged"
        assert w1.work_id == w2.work_id
        assert len(s.exec(select(Work)).all()) == 1
        # identifiers remapped onto the one work, no duplicate id rows
        idents = s.exec(select(Identifier)).all()
        assert {(i.id_type, i.id_value) for i in idents} == {("doi", "10.1/x"), ("arxiv", "2202.0001")}
        assert all(i.work_id == w1.work_id for i in idents)


def test_phase5_identity_merge_cross_work_collision():
    h = service.create_project("merge_cross")
    with Session(h.engine, expire_on_commit=False) as s:
        wa, _ = identity.upsert_work(s, {"doi": "10.2/a", "title": "A"})
        wb, _ = identity.upsert_work(s, {"arxiv": "2101.9", "title": "B"})
        s.commit()
        a_id, b_id = wa.work_id, wb.work_id

    with Session(h.engine, expire_on_commit=False) as s:
        wc, outcome = identity.upsert_work(
            s, {"doi": "10.2/a", "arxiv": "2101.9", "title": "C"}
        )  # NO UNIQUE crash
        s.commit()
        assert outcome == "duplicate_candidate"
        # A and B untouched: still exactly one identifier each, owned as before
        assert len(s.exec(select(Work)).all()) == 3
        a_ids = s.exec(select(Identifier).where(Identifier.work_id == a_id)).all()
        b_ids = s.exec(select(Identifier).where(Identifier.work_id == b_id)).all()
        assert {(i.id_type, i.id_value) for i in a_ids} == {("doi", "10.2/a")}
        assert {(i.id_type, i.id_value) for i in b_ids} == {("arxiv", "2101.9")}
        # new work carries only the unclaimed ids (here: none)
        c_ids = s.exec(select(Identifier).where(Identifier.work_id == wc.work_id)).all()
        assert c_ids == []
        # exactly one review item, reason cross_id_collision, conflicting=[A,B]
        items = s.exec(select(ReviewQueueItem)).all()
        assert len(items) == 1
        payload = json.loads(items[0].payload)
        assert payload["reason"] == "cross_id_collision"
        assert payload["conflicting_work_ids"] == [a_id, b_id]
        assert items[0].target_id == wc.work_id


def test_phase5_identity_merge_title_collision():
    h = service.create_project("merge_title")
    with Session(h.engine, expire_on_commit=False) as s:
        w1, _ = identity.upsert_work(s, {"doi": "10.3/a", "title": "Same Title"})
        # same normalized title, disjoint strong ids
        w2, outcome = identity.upsert_work(s, {"arxiv": "2103.1", "title": "same   title"})
        s.commit()
        assert outcome == "created"  # titles NEVER auto-merge
        assert w1.work_id != w2.work_id
        assert len(s.exec(select(Work)).all()) == 2
        items = s.exec(select(ReviewQueueItem)).all()
        assert len(items) == 1
        assert json.loads(items[0].payload)["reason"] == "title_collision"


# --------------------------------------------------------------------------
# test_project_service
# --------------------------------------------------------------------------

def test_phase5_project_service_create_open(tmp_path):
    root = tmp_path / "home"
    h = service.create_project("svc", name="Service Demo", description="d", root=root)
    assert h.db_path.exists()
    assert layout.project_yaml_path("svc", root).exists()
    assert layout.project_runs_dir("svc", root).is_dir()

    # hard-error if the project dir already exists
    with pytest.raises(ValidationError):
        service.create_project("svc", root=root)

    # open round-trips ProjectConfig (project_id == slug)
    reopened = service.open_project("svc", root=root)
    assert reopened.config.project_id == "svc"
    assert reopened.config.project_name == "Service Demo"
    assert reopened.config.answer_policy.require_project_sources is True

    with pytest.raises(ValidationError):
        service.open_project("missing", root=root)


# --------------------------------------------------------------------------
# test_inclusion_status
# --------------------------------------------------------------------------

def test_phase5_inclusion_status_transitions(monkeypatch):
    h = service.create_project("incl")

    ticks = iter([f"2026-01-01T00:00:{i:02d}+00:00" for i in range(20)])
    monkeypatch.setattr(service, "_now", lambda: next(ticks))

    work = service.add_work(h, ids={"doi": "10.9/z"}, title="Seed", is_seed=True)

    def _doc():
        with Session(h.engine) as s:
            return s.get(ProjectDocument, work.work_id)

    doc = _doc()
    assert doc.is_seed == 1
    assert doc.inclusion_reason == "seed_document"
    assert doc.access_status is None
    created_at = doc.created_at

    for status in ("excluded", "metadata_only", "included"):
        service.set_inclusion_status(h, work.work_id, status)
        doc = _doc()
        assert doc.inclusion_status == status
        assert doc.access_status is None  # decision 30 — NEVER derived from membership
        assert doc.updated_at > created_at  # bumped on every transition


# --------------------------------------------------------------------------
# test_corpus_filter_isolation
# --------------------------------------------------------------------------

def test_phase5_corpus_filter_isolation():
    h = service.create_project("proj_a")
    service.add_work(h, ids={"doi": "10.1/inc"}, title="Included")
    service.add_work(h, ids={"doi": "10.1/meta"}, title="Meta", inclusion_status="metadata_only")
    service.add_work(h, ids={"doi": "10.1/exc"}, title="Excluded", inclusion_status="excluded")

    included = service.corpus_works(h)  # default ("included",)
    assert {w.canonical_title for w in included} == {"Included"}
    every = service.corpus_works(h, ("included", "metadata_only", "excluded"))
    assert len(every) == 3

    # two projects under one root never see each other's works (structural isolation)
    h2 = service.create_project("proj_b")
    service.add_work(h2, ids={"doi": "10.2/only"}, title="OnlyB")
    a_ids = {w.work_id for w in service.corpus_works(h)}
    b_ids = {w.work_id for w in service.corpus_works(h2)}
    assert a_ids and b_ids
    assert a_ids.isdisjoint(b_ids)


# --------------------------------------------------------------------------
# test_review_queue
# --------------------------------------------------------------------------

def test_phase5_review_queue_payload_validation():
    h = service.create_project("rq")

    # enqueue REJECTS a payload that fails the discriminated-union validation
    with pytest.raises(Exception):
        review.enqueue(h, "duplicate_candidate", payload={"kind": "not_a_variant"})
    assert review.list_open(h) == []  # nothing persisted

    # Build A ch7: resolve('merge') now COLLAPSES the pair in-transaction (a
    # fake/stale pair is refused), so the flip/idempotency assertions below need
    # a real duplicate pair.
    wx = service.add_work(h, ids={"doi": "10.rq/x"}, title="X").work_id
    wy = service.add_work(h, ids={"doi": "10.rq/y"}, title="Y").work_id
    item_id = review.enqueue(
        h,
        "duplicate_candidate",
        target_type="work",
        target_id=wx,
        payload={
            "kind": "duplicate_candidate",
            "reason": "title_collision",
            "conflicting_work_ids": [wy],
            "incoming": {"title": "t"},
        },
    )
    assert len(review.list_open(h)) == 1

    review.resolve(h, item_id, "merge")
    assert review.list_open(h) == []
    review.resolve(h, item_id, "exclude")  # idempotent no-op (already resolved)
    with Session(h.engine) as s:
        row = s.get(ReviewQueueItem, item_id)
    assert row.status == "resolved"
    assert row.action == "merge"  # unchanged by the idempotent second resolve
    assert row.resolved_at is not None

    # an unknown action is rejected
    with pytest.raises(ValidationError):
        review.resolve(h, item_id, "frobnicate")


# --------------------------------------------------------------------------
# test_cli_project
# --------------------------------------------------------------------------

def test_phase5_cli_project_flow(tmp_path):
    from typer.testing import CliRunner

    from seedgraph.cli import app

    runner = CliRunner()
    root = str(tmp_path / "home")

    def run(*args):
        return runner.invoke(app, [*args, "--root", root])

    assert run("project", "new", "cliproj", "--name", "CLI Proj").exit_code == 0
    assert run(
        "project", "add", "cliproj", "--doi", "10.1/a", "--title", "Paper A", "--seed"
    ).exit_code == 0
    assert run(
        "project", "add", "cliproj", "--arxiv", "2101.1", "--title", "Paper B",
        "--status", "metadata_only",
    ).exit_code == 0
    assert run(
        "project", "add", "cliproj", "--title", "Paper C",
        "--status", "excluded", "--reason", "user_excluded",
    ).exit_code == 0

    show = run("project", "show", "cliproj")
    assert show.exit_code == 0
    assert "Paper A" in show.output and "included" in show.output
    assert "metadata_only" in show.output and "excluded" in show.output

    # doctor reports FK pragma ON for the project scope (uses the existing surface)
    doc = runner.invoke(app, ["doctor", "--project", "cliproj", "--root", root])
    assert doc.exit_code == 0
    assert "foreign_keys_project" in doc.output and "ON" in doc.output

    # FK enforcement is real: deleting a work cascades project_documents + identifiers
    h = service.open_project("cliproj", root=Path(root))
    with Session(h.engine) as s:
        work = s.exec(select(Work).where(Work.canonical_title == "Paper A")).one()
        wid = work.work_id
        s.delete(work)
        s.commit()
    with Session(h.engine) as s:
        assert s.get(ProjectDocument, wid) is None
        assert s.exec(select(Identifier).where(Identifier.work_id == wid)).all() == []


# --------------------------------------------------------------------------
# test_api_read_routes (scaffold-critic gap 1)
# --------------------------------------------------------------------------

def test_phase5_api_read_routes():
    from seedgraph.api.app import app
    from seedgraph.web.serve import require_local_session

    h = service.create_project("apiproj")
    service.add_work(h, ids={"doi": "10.1/a"}, title="Paper A", is_seed=True)
    service.add_work(h, ids={"arxiv": "2101.1"}, title="Paper B", inclusion_status="metadata_only")

    client = TestClient(app)
    assert "apiproj" in client.get("/projects").json()

    # /documents is a private-evidence GET behind require_local_session (cross-cutting
    # #1); exercise it via the dependency-override seam, restoring the gate after.
    app.dependency_overrides[require_local_session] = lambda: None
    try:
        resp = client.get("/projects/apiproj/documents")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        only_included = client.get("/projects/apiproj/documents", params={"status": "included"})
        titles = {row["title"] for row in only_included.json()}
        assert titles == {"Paper A"}

        assert client.get("/projects/does-not-exist/documents").status_code == 404
    finally:
        app.dependency_overrides.pop(require_local_session, None)


# --------------------------------------------------------------------------
# Milestone acceptance (doc 10 §6 Phase 4) — GROUPED CLI surface
# --------------------------------------------------------------------------

def test_phase5_milestone_acceptance(tmp_path):
    from typer.testing import CliRunner

    from seedgraph.cli import app

    runner = CliRunner()
    root = str(tmp_path / "home")

    def run(*args):
        result = runner.invoke(app, [*args, "--root", root])
        assert result.exit_code == 0, result.output
        return result

    run("project", "new", "did_demo", "--name", "DiD demo")
    run("project", "add", "did_demo", "--doi", "10.x/aaa", "--title", "Paper A", "--seed")
    run("project", "add", "did_demo", "--arxiv", "2101.00001", "--title", "Paper B",
        "--status", "metadata_only")
    run("project", "add", "did_demo", "--title", "Paper C", "--status", "excluded",
        "--reason", "user_excluded")

    show = run("project", "show", "did_demo").output
    # A=included(seed), B=metadata_only, C=excluded all present
    assert "Paper A" in show and "Paper B" in show and "Paper C" in show
    assert "included" in show and "metadata_only" in show and "excluded" in show

    # Isolation assertion: two projects' corpora are disjoint and each non-empty.
    run("project", "new", "did_demo2")
    run("project", "add", "did_demo2", "--doi", "10.y/zzz", "--title", "Paper Z")

    h1 = service.open_project("did_demo", root=Path(root))
    h2 = service.open_project("did_demo2", root=Path(root))
    c1 = {w.work_id for w in service.corpus_works(h1)}
    c2 = {w.work_id for w in service.corpus_works(h2)}
    assert c1 and c2  # each non-empty
    assert c1.isdisjoint(c2)  # disjoint — query only that project's corpus
