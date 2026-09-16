"""Bounded source reading through real cache/project databases, offline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from sqlmodel import Session
from typer import Typer
from typer.testing import CliRunner

from seedgraph import cache_access
from seedgraph.acquisition.bridge import write_bridge
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.db.adapter import raw_conn
from seedgraph.db.connection import cache_db_path
from seedgraph.project import service
from seedgraph.spans.store import ensure_span
from seedgraph.vocab import AccessClass, AcquisitionMethod


MD = "# First\n\nOwn text caf\u00e9 \U0001f600.\n\n## Child\n\nChild text.\n\n# Last\n\nLast text.\n"


def _checkpoint(handle):
    handle.engine.dispose()
    for path in (handle.db_path, cache_db_path(handle.root)):
        if path.exists():
            with sqlite3.connect(path) as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _document(home: Path, markdown=MD, access_class=AccessClass.open_access):
    handle = service.create_project("reader", root=home)
    work = service.add_work(handle, title="Reader paper", authors=["An Author"], year=2026)
    pdf = home / "reader.pdf"
    pdf.write_bytes(b"%PDF-1.4 reader fixture with sufficiently many words")
    method = (AcquisitionMethod.open_access_fetch if access_class == AccessClass.open_access
              else AcquisitionMethod.upload)
    source = ingest_file(pdf, access_class=access_class, acquisition_method=method, root=home)
    md = convert_source_file(source.source_file_id,
                             backend=FakeMarkerBackend(markdown=markdown), root=home)
    with Session(handle.engine) as session:
        write_bridge(session, work_id=work.work_id, source_file_id=source.source_file_id,
                     file_hash=source.file_hash, markdown_id=md.markdown_id,
                     markdown_hash=md.markdown_hash, acquisition_method="manual_upload")
        session.commit()
    _checkpoint(handle)
    return handle, work.work_id, source, md


def test_reader_paginates_unicode_and_sections_without_writes(isolated_home, monkeypatch):
    from seedgraph.work_read import read_work

    handle, work_id, _, _ = _document(isolated_home)
    # Any accidental initialization or write-capable connection fails this test.
    real_connect = sqlite3.connect
    def only_readonly(database, *args, **kwargs):
        assert "mode=ro" in str(database)
        conn = real_connect(database, *args, **kwargs)
        conn.set_authorizer(lambda action, *args: sqlite3.SQLITE_DENY
                            if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE,
                                          sqlite3.SQLITE_DELETE, sqlite3.SQLITE_CREATE_TABLE}
                            else sqlite3.SQLITE_OK)
        return conn
    monkeypatch.setattr(sqlite3, "connect", only_readonly)
    result = read_work(handle, work_id=work_id, max_chars=7, section_limit=1)
    first = result
    chunks = [result["text"]]
    while result["continuation"]:
        result = read_work(handle, work_id=work_id, max_chars=7,
                           cursor=result["continuation"], section_limit=1)
        assert result["start_char"] == sum(map(len, chunks))
        chunks.append(result["text"])
    assert "".join(chunks) == MD
    assert result["selection_complete"] is True
    assert len(first["sections"]["items"]) == 1
    assert first["sections"]["next_offset"] == 1
    section_id = first["sections"]["items"][0]["section_id"]
    own = read_work(handle, work_id=work_id, section_id=section_id)
    subtree = read_work(handle, work_id=work_id, section_id=section_id, subtree=True)
    assert "Child text" not in own["text"]
    assert "Child text" in subtree["text"]
    assert "Last text" not in subtree["text"]


def test_missing_text_is_metadata_only_without_acquisition(isolated_home):
    from seedgraph.work_read import read_work

    handle = service.create_project("missing", root=isolated_home)
    work = service.add_work(handle, title="Known paper")
    _checkpoint(handle)
    result = read_work(handle, work_id=work.work_id)
    assert result["status"] == "unavailable"
    assert result["reason"] == "source_missing"
    assert result["work"]["title"] == "Known paper"
    assert result["text"] is None
    assert "acqui" in result["next_action"].lower()


def test_continuation_and_span_pin_old_conversion(isolated_home):
    from seedgraph.work_read import read_work

    handle, work_id, source, md = _document(isolated_home)
    cache = cache_access.open_cache_ro(isolated_home)
    try:
        with Session(handle.engine) as session:
            span_id = ensure_span(raw_conn(session), cache, isolated_home, work_id=work_id,
                                  markdown_id=md.markdown_id, exact_quote="Child text.",
                                  access_class="open_access")
            session.commit()
    finally:
        cache.close()
    _checkpoint(handle)
    first = read_work(handle, work_id=work_id, max_chars=10)
    newer = convert_source_file(source.source_file_id,
                               backend=FakeMarkerBackend(markdown="# New\nCompletely new text.\n"),
                               force=True, root=isolated_home)
    _checkpoint(handle)
    second = read_work(handle, work_id=work_id, max_chars=100, cursor=first["continuation"])
    assert first["text"] + second["text"] == MD
    assert second["newer_version_available"] is True
    assert second["newer_markdown_id"] == newer.markdown_id
    span = read_work(handle, work_id=work_id, span_id=span_id)
    assert span["text"] == MD
    assert span["anchor"]["start_char"] == MD.index("Child text.")
    assert span["source"]["markdown_id"] == md.markdown_id
    with sqlite3.connect(cache_db_path(isolated_home)) as conn:
        uri = conn.execute("SELECT storage_uri FROM markdown_documents WHERE markdown_id=?",
                           (md.markdown_id,)).fetchone()[0]
    # Cache content is in this disposable test home only.
    (isolated_home / "cache" / uri).unlink()
    gone = read_work(handle, work_id=work_id, cursor=first["continuation"])
    assert gone["reason"] == "pinned_version_missing"
    assert gone["text"] is None and "restart" in gone["next_action"]


def test_policy_withholds_all_source_fields_and_bounds_headings(isolated_home):
    from seedgraph.work_read import read_work

    private_heading = "Private heading" * 100
    handle, work_id, _, _ = _document(isolated_home, f"# {private_heading}\nPrivate body.\n",
                                     AccessClass.user_supplied_private)
    local = read_work(handle, work_id=work_id)
    assert len(local["sections"]["items"][0]["heading_text"]) == 256
    assert local["sections"]["items"][0]["heading_truncated"] is True
    external = read_work(handle, work_id=work_id, external=True)
    redacted = read_work(handle, work_id=work_id, redact_private=True, confirm_external=True)
    for response in (external, redacted):
        assert response["status"] == "policy_withheld"
        assert response["text"] is None and response["sections"]["items"] == []
        assert "Private body" not in json.dumps(response)
        assert "Private heading" not in json.dumps(response)
    consent = read_work(handle, work_id=work_id, external=True, confirm_external=True)
    assert consent["status"] == "ok" and consent["text"] == local["text"]


def test_cursor_is_bound_and_limits_are_validated(isolated_home):
    from seedgraph.errors import ValidationError
    from seedgraph.work_read import read_work

    handle, work_id, _, _ = _document(isolated_home)
    cursor = read_work(handle, work_id=work_id, max_chars=5)["continuation"]
    for kwargs in ({"max_chars": 0}, {"max_chars": 48001}, {"section_limit": 101},
                   {"section_offset": -1}, {"cursor": "../../../cache"},
                   {"cursor": cursor, "section_id": "unrelated"}):
        with pytest.raises(ValidationError):
            read_work(handle, work_id=work_id, **kwargs)
    with pytest.raises(ValidationError):
        read_work(handle, work_id="different_work", cursor=cursor)


def test_cli_and_mcp_share_reader_and_external_policy(isolated_home):
    from seedgraph.mcp.context import ServerContext
    from seedgraph.mcp.tools_source import register_source_tools
    from seedgraph.work_read_cli import register_work_read

    handle, work_id, _, _ = _document(isolated_home, access_class=AccessClass.user_supplied_private)
    cli = Typer()
    register_work_read(cli)
    result = CliRunner().invoke(cli, ["reader", work_id, "--root", str(isolated_home),
                                     "--json", "--max-chars", "11"])
    assert result.exit_code == 0, result.output
    local = json.loads(result.stdout)
    assert local["text"] == MD[:11]
    class Registry:
        def tool(self):
            def register(fn):
                self.call = fn
                return fn
            return register
    registry = Registry()
    register_source_tools(registry, ServerContext(root=isolated_home))
    withheld = registry.call("reader", work_id)
    assert withheld["status"] == "policy_withheld"
    shared = registry.call("reader", work_id, confirm_external=True, max_chars=11)
    assert shared["text"] == local["text"]


def test_reupload_continues_only_with_existing_work_provenance(isolated_home):
    from seedgraph.sections.parser import parse_sections
    from seedgraph.sections.store import replace_sections
    from seedgraph.work_read import read_work

    handle, work_id, source, md = _document(isolated_home)
    first = read_work(handle, work_id=work_id, max_chars=10)
    pdf = isolated_home / "replacement.pdf"
    pdf.write_bytes(b"%PDF replacement source with other file bytes")
    new_source = ingest_file(pdf, access_class=AccessClass.open_access,
                             acquisition_method=AcquisitionMethod.open_access_fetch, root=isolated_home)
    new_md = convert_source_file(new_source.source_file_id,
                                 backend=FakeMarkerBackend(markdown="# Replacement\nNew text."),
                                 root=isolated_home)
    with Session(handle.engine) as session:
        write_bridge(session, work_id=work_id, source_file_id=new_source.source_file_id,
                     file_hash=new_source.file_hash, markdown_id=new_md.markdown_id,
                     markdown_hash=new_md.markdown_hash, acquisition_method="manual_upload")
        session.commit()
    _checkpoint(handle)
    unknown = read_work(handle, work_id=work_id, cursor=first["continuation"])
    assert unknown["reason"] == "pinned_lineage_unavailable"
    sections = parse_sections(MD, markdown_id=md.markdown_id, markdown_hash=md.markdown_hash,
                              source_file_id=source.source_file_id, source_file_hash=source.file_hash,
                              work_id=work_id)
    with sqlite3.connect(handle.db_path) as conn:
        replace_sections(conn, md.markdown_id, sections)
    _checkpoint(handle)
    continued = read_work(handle, work_id=work_id, cursor=first["continuation"])
    assert first["text"] + continued["text"] == MD
    assert continued["newer_version_available"] is True
    assert continued["newer_markdown_id"] == new_md.markdown_id


def test_cursor_rechecks_access_on_each_page(isolated_home):
    from seedgraph.work_read import read_work

    handle, work_id, source, _ = _document(isolated_home)
    first = read_work(handle, work_id=work_id, external=True, max_chars=10)
    with sqlite3.connect(cache_db_path(isolated_home)) as conn:
        conn.execute("UPDATE source_files SET access_class='user_supplied_private' WHERE source_file_id=?",
                     (source.source_file_id,))
    _checkpoint(handle)
    second = read_work(handle, work_id=work_id, external=True, cursor=first["continuation"])
    assert second["status"] == "policy_withheld"
    assert second["text"] is None and second["sections"]["items"] == []


def test_preamble_has_no_descendant_sections(isolated_home):
    from seedgraph.work_read import read_work

    handle, work_id, _, _ = _document(isolated_home, "Preamble only.\n\n" + MD)
    inventory = read_work(handle, work_id=work_id)["sections"]["items"]
    result = read_work(handle, work_id=work_id, section_id=inventory[0]["section_id"], subtree=True)
    assert result["text"] == "Preamble only.\n\n"


@pytest.mark.parametrize("cache_state,reason", [
    ("uninitialized", "cache_setup_required"),
    ("older", "cache_schema_mismatch"),
    ("missing_table", "cache_schema_unavailable"),
])
def test_unavailable_cache_schema_returns_metadata_without_migration(isolated_home, cache_state, reason):
    from seedgraph.work_read import read_work

    handle, work_id, _, _ = _document(isolated_home)
    with sqlite3.connect(cache_db_path(isolated_home)) as conn:
        if cache_state == "uninitialized":
            conn.execute("DROP TABLE schema_migrations")
        elif cache_state == "older":
            conn.execute("DELETE FROM schema_migrations WHERE version=(SELECT MAX(version) FROM schema_migrations)")
        conn.execute("DROP TABLE markdown_documents")
    _checkpoint(handle)
    before = cache_db_path(isolated_home).read_bytes()
    result = read_work(handle, work_id=work_id)
    assert result["status"] == "unavailable" and result["reason"] == reason
    assert result["work"]["title"] == "Reader paper"
    assert result["text"] is None and result["next_action"]
    assert cache_db_path(isolated_home).read_bytes() == before


def test_policy_denial_explains_disabled_external_handoff(isolated_home):
    from seedgraph.work_read import read_work

    handle, work_id, _, _ = _document(isolated_home)
    (isolated_home / "config.yaml").write_text("content_policy:\n  allow_external_llm: false\n", encoding="utf-8")
    result = read_work(handle, work_id=work_id, external=True, confirm_external=True)
    assert result["status"] == "policy_withheld"
    assert "allow_external_llm=false" in result["policy_reason"]
    assert "confirm" not in result["next_action"]


def test_shared_markdown_keeps_producer_lineage_when_access_representative_changes(isolated_home):
    from seedgraph.work_read import read_work

    handle, first_work, first_source, md = _document(isolated_home)
    first_page = read_work(handle, work_id=first_work, max_chars=10, external=True)
    assert first_page["status"] == "ok" and first_page["text"] == MD[:10]
    other = service.add_work(handle, title="Other source, identical converted text")
    pdf = isolated_home / "second_source.pdf"
    pdf.write_bytes(b"%PDF-1.4 distinct second source yielding exactly the same Markdown")
    source = ingest_file(pdf, access_class=AccessClass.open_access,
                         acquisition_method=AcquisitionMethod.open_access_fetch, root=isolated_home)
    shared = convert_source_file(source.source_file_id, backend=FakeMarkerBackend(markdown=MD),
                                 root=isolated_home)
    assert shared.markdown_id == md.markdown_id
    with Session(handle.engine) as session:
        write_bridge(session, work_id=other.work_id, source_file_id=source.source_file_id,
                     file_hash=source.file_hash, markdown_id=shared.markdown_id,
                     markdown_hash=shared.markdown_hash, acquisition_method="manual_upload")
        session.commit()
    _checkpoint(handle)
    second_work = read_work(handle, work_id=other.work_id)
    assert second_work["text"] == MD
    assert second_work["source"]["source_file_id"] == source.source_file_id
    # A third, private producer repoints the shared row only for conservative access.
    pdf = isolated_home / "private_source.pdf"
    pdf.write_bytes(b"%PDF-1.4 private third source yielding exactly the same Markdown")
    private = ingest_file(pdf, access_class=AccessClass.user_supplied_private,
                          acquisition_method=AcquisitionMethod.upload, root=isolated_home)
    convert_source_file(private.source_file_id, backend=FakeMarkerBackend(markdown=MD), root=isolated_home)
    _checkpoint(handle)
    continued = read_work(handle, work_id=first_work, cursor=first_page["continuation"])
    assert first_page["text"] + continued["text"] == MD
    assert continued["source"]["source_file_id"] == first_source.source_file_id
    assert continued["access_class"] == "user_supplied_private"
    assert continued["newer_version_available"] is False
    withheld = read_work(handle, work_id=first_work, cursor=first_page["continuation"], external=True)
    assert withheld["status"] == "policy_withheld" and withheld["text"] is None
