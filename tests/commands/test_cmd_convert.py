"""`seedgraph convert` smoke tests. The converter is the deterministic fake (no subprocess)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seedgraph.cli import app
from seedgraph.library.files import ConvertOutcome
from seedgraph.library.store import open_library_at


@pytest.fixture(autouse=True)
def _fake_converter(monkeypatch):
    monkeypatch.setenv("SEEDGRAPH_FAKE_MARKER", "1")


def _pdf(path: Path, text: str) -> Path:
    """A one-page PDF with a Helvetica text layer (invented text)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def test_convert_a_named_pdf_into_the_library(runner, migrated_home, tmp_path):
    pdf = _pdf(tmp_path / "loose.pdf", "Granular Pricing of Imaginary Orchards")
    result = runner.invoke(app, ["convert", str(pdf), "--root", str(migrated_home), "--json"])
    assert result.exit_code == 0, result.output
    outcomes = json.loads(result.output)
    assert [o["status"] for o in outcomes] == ["converted"]
    with open_library_at(migrated_home / "library") as lib:
        assert lib.conn.execute("SELECT 1 FROM markdown_documents WHERE markdown_id = ?",
                                (outcomes[0]["markdown_id"],)).fetchone()


def test_convert_queued_runs(runner, migrated_home):
    result = runner.invoke(app, ["convert", "--queued", "--root", str(migrated_home)])
    assert result.exit_code == 0, result.output
    with open_library_at(migrated_home / "library") as lib:
        assert lib.conn.execute("SELECT COUNT(*) FROM conversion_queue").fetchone()[0] == 0


def test_convert_with_nothing_to_do_is_refused(runner, migrated_home):
    result = runner.invoke(app, ["convert", "--root", str(migrated_home)])
    assert result.exit_code == 3
    assert "nothing to convert" in result.output


def test_convert_unknown_paper_is_refused(runner, migrated_home):
    assert runner.invoke(app, ["convert", "--paper", "lp_nope", "--root", str(migrated_home)]).exit_code == 3


def test_a_failed_conversion_exits_1(runner, migrated_home, monkeypatch):
    from seedgraph.commands import convert as convert_cmd

    failed = ConvertOutcome(source_file_id="sf_x", status="failed", markdown_id=None, error="boom", paper_ids=())
    monkeypatch.setattr(convert_cmd.files, "convert_files", lambda lib, **kw: [failed])
    result = runner.invoke(app, ["convert", "--queued", "--root", str(migrated_home)])
    assert result.exit_code == 1
    assert "boom" in result.output
