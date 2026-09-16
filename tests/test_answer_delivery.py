"""Public CLI/MCP delivery contracts for active agent sessions."""
import json
import sqlite3

import pytest

from typer.testing import CliRunner

from _phase8_helpers import build_fixture_project
from seedgraph.cli import app


def test_cli_delivers_one_json_value_even_when_answer_directory_is_unwritable():
    project = build_fixture_project("delivery")
    # A file in the directory's place deterministically fails on every OS/user.
    (project.db_path.parent / "answers").write_text("blocked", encoding="utf-8")
    result = CliRunner().invoke(
        app, ["ask", "across groups", "--project", "delivery", "--no-llm", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["citations"]
    assert payload["persistence"]["status"] == "not_saved"
    assert payload["persistence"]["reason"]


def test_cli_no_save_readonly_keeps_corpus_bytes_and_files_unchanged():
    project = build_fixture_project("readonly")
    project.engine.dispose()
    with sqlite3.connect(project.db_path) as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    before = {p.relative_to(project.root): p.read_bytes()
              for p in project.root.rglob("*") if p.is_file()}
    result = CliRunner().invoke(
        app, ["ask", "across groups", "--project", "readonly", "--no-llm", "--no-save", "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["citations"]
    assert payload["persistence"]["status"] == "skipped"
    after = {p.relative_to(project.root): p.read_bytes()
             for p in project.root.rglob("*") if p.is_file()}
    assert after == before


@pytest.mark.parametrize("problem,code", [("missing", "setup_required"), ("schema", "schema_mismatch"), ("wal", "setup_required"), ("journal", "setup_required")])
def test_readonly_setup_errors_are_typed_and_never_repair_the_corpus(problem, code):
    h = build_fixture_project("needs_setup")
    h.engine.dispose()
    writer = sqlite3.connect(h.db_path)
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if problem == "schema":
        writer.execute("DELETE FROM schema_migrations WHERE version=(SELECT MAX(version) FROM schema_migrations)")
        writer.commit()
        writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if problem == "wal":
        writer.execute("UPDATE works SET year=2029 WHERE work_id='work_a'")
        writer.commit()
    else:
        writer.close()
    if problem == "missing":
        h.db_path.unlink()
    if problem == "journal":
        h.db_path.with_name("project.db-journal").write_bytes(b"pending journal")
    before = {p.relative_to(h.root): p.read_bytes() for p in h.root.rglob("*") if p.is_file()}
    try:
        result = CliRunner().invoke(app, [
            "ask", "across groups", "--project", h.slug, "--no-llm", "--no-save", "--json",
        ])
        assert result.exit_code != 0
        assert json.loads(result.stdout)["error"]["code"] == code
        assert before == {p.relative_to(h.root): p.read_bytes() for p in h.root.rglob("*") if p.is_file()}
    finally:
        writer.close()


def test_cli_partial_save_reports_only_the_saved_artifact(monkeypatch):
    from seedgraph.answer import trace
    h = build_fixture_project("partial_save")

    def failed_trace(*args, **kwargs):
        raise PermissionError("trace storage unavailable")

    monkeypatch.setattr(trace, "save_trace", failed_trace)
    result = CliRunner().invoke(app, ["ask", "across groups", "--project", h.slug, "--no-llm", "--json"])
    assert result.exit_code == 0
    saved = json.loads(result.stdout)["persistence"]
    assert saved["status"] == "not_saved"
    assert "answer_path" in saved and "trace_path" not in saved
    assert len(list((h.db_path.parent / "answers").glob("*.json"))) == 1


def test_readonly_reader_detects_a_writer_that_starts_after_open():
    from seedgraph.db.connection import ReadOnlyCorpusError, connect_readonly
    h = build_fixture_project("writer_race")
    h.engine.dispose()
    writer = sqlite3.connect(h.db_path)
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    reader = connect_readonly(h.db_path)
    assert reader.execute("SELECT year FROM works WHERE work_id='work_a'").fetchone()[0] == 2020
    writer.execute("UPDATE works SET year=2029 WHERE work_id='work_a'")
    writer.commit()
    try:
        with pytest.raises(ReadOnlyCorpusError, match="checkpointed|changed"):
            reader.close()
    finally:
        writer.close()
