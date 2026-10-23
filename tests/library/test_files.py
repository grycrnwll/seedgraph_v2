"""Files and conversion into the library (v3 design §1.1, §2.1 files, §7.4, §11 S4).

All PDFs here are synthetic and all text is invented.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from seedgraph.cache.convert import ConversionError, convert_source_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend, MarkerResult
from seedgraph.library import files
from seedgraph.library.store import LibraryError, open_library_at

TITLE = "Bayesian Thermostats for Coastal Bakeries"
DOI = "10.9999/bake.2031.007"


# --- synthetic PDFs with a real text layer --------------------------------------------


def text_pdf(path: Path, pages: list[str], *, marker: str = "") -> Path:
    """A minimal valid PDF whose pages carry ``pages`` as a Helvetica text layer.

    ``marker`` is appended as a trailing comment (e.g. ``FAKE:ERROR`` for the fake
    converter engine) and also keeps otherwise-identical PDFs distinct.
    """
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()]
    for i, text in enumerate(pages):
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET".encode()
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    if marker:
        out += f"% {marker}\n".encode()
    path = Path(path)
    path.write_bytes(bytes(out))
    return path


def no_text_pdf(path: Path, *, marker: str = "") -> Path:
    """A PDF with pages but no text layer (a 'scan')."""
    return text_pdf(path, ["", ""], marker=marker)


# --- fixtures -------------------------------------------------------------------------


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def add_paper(lib, paper_id: str = "lp_00000000000000000001", title: str | None = TITLE,
              doi: str | None = None) -> str:
    now = datetime.now(timezone.utc).isoformat()
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES (?, ?, ?)",
                     (paper_id, title, now))
        if doi:
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) VALUES (?, 'doi', ?)",
                         (paper_id, doi))
    return paper_id


def paper_file(lib, paper_id: str):
    return lib.conn.execute("SELECT * FROM paper_files WHERE paper_id = ?", (paper_id,)).fetchone()


def open_items(lib, kind: str = "pdf_title_miss"):
    return lib.conn.execute(
        "SELECT * FROM attention_items WHERE kind = ? AND status = 'open'", (kind,)
    ).fetchall()


def queued(lib) -> list[str]:
    return [r[0] for r in lib.conn.execute(
        "SELECT source_file_id FROM conversion_queue ORDER BY source_file_id")]


PAGED_MD = (
    "{0}------------------------------------------------\n\n"
    f"# {TITLE}\n\n## Introduction\n\nInvented body text about ovens.\n\n"
    "{1}------------------------------------------------\n\n"
    "More invented text.\n\n## References\n\n[1] A reference.\n"
)


# --- attach and the standing PDF check -------------------------------------------------


def test_attach_records_a_passing_check_from_the_pdf_text(lib, tmp_path):
    pid = add_paper(lib)
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE + " by Jane Doe", "Invented second page."])

    res = files.attach_pdf(lib, pid, pdf)

    row = paper_file(lib, pid)
    assert row["source_file_id"] == res.source_file_id
    assert row["pdf_check_status"] == "pass" == res.check.status
    check = json.loads(row["pdf_check_json"])
    assert check["status"] == "pass" and check["source"] == "pdf"
    assert row["markdown_id"] is None
    # the bytes are in the library's content-addressed blob store
    sha = res.source_file_id.removeprefix("sf_")
    assert (lib.pdfs_dir / f"{sha}.pdf").read_bytes() == pdf.read_bytes()
    src = lib.conn.execute("SELECT storage_uri FROM source_files WHERE source_file_id = ?",
                           (res.source_file_id,)).fetchone()
    assert src["storage_uri"] == f"pdfs/{sha}.pdf"
    assert open_items(lib) == []
    assert queued(lib) == [res.source_file_id]
    assert lib.conn.execute("SELECT status FROM paper_status WHERE paper_id = ?",
                            (pid,)).fetchone()[0] == "held"


def test_doi_on_the_page_passes(lib, tmp_path):
    pid = add_paper(lib, title="A Title That Is Not On The Page At All", doi=DOI)
    pdf = text_pdf(tmp_path / "a.pdf", [f"Some cover sheet doi:{DOI}"])
    assert files.attach_pdf(lib, pid, pdf).check.via == "doi"


def test_a_miss_flags_the_paper_but_attach_succeeds(lib, tmp_path):
    pid = add_paper(lib)
    pdf = text_pdf(tmp_path / "wrong.pdf", ["Seasonal Migration of Alpine Goats", "Invented."])

    res = files.attach_pdf(lib, pid, pdf)

    assert res.check.status == "miss"
    row = paper_file(lib, pid)
    assert row["pdf_check_status"] == "miss"
    items = open_items(lib)
    assert len(items) == 1
    item = items[0]
    assert item["paper_id"] == pid
    assert item["hides_notes"] == 1
    assert "may not be this paper" in item["detail"]
    assert json.loads(item["subject_json"]) == {"paper_id": pid, "source_file_id": res.source_file_id}
    assert res.attention_item_id == item["item_id"]
    # never a block: the file is still queued for conversion
    assert queued(lib) == [res.source_file_id]


def test_no_text_layer_is_unknown_and_raises_nothing(lib, tmp_path):
    pid = add_paper(lib)
    res = files.attach_pdf(lib, pid, no_text_pdf(tmp_path / "scan.pdf"))
    assert res.check.status == "unknown"
    assert paper_file(lib, pid)["pdf_check_status"] == "unknown"
    assert open_items(lib) == []


def test_reattaching_the_same_pdf_is_idempotent(lib, tmp_path):
    pid = add_paper(lib)
    pdf = text_pdf(tmp_path / "wrong.pdf", ["Seasonal Migration of Alpine Goats"])
    first = files.attach_pdf(lib, pid, pdf)
    before = dict(paper_file(lib, pid))

    second = files.attach_pdf(lib, pid, pdf)

    assert second.source_file_id == first.source_file_id
    assert dict(paper_file(lib, pid)) == before
    assert lib.conn.execute("SELECT COUNT(*) FROM paper_files").fetchone()[0] == 1
    assert lib.conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 1
    assert len(open_items(lib)) == 1
    assert queued(lib) == [first.source_file_id]
    assert len(list(lib.pdfs_dir.iterdir())) == 1


def test_attach_refuses_an_unknown_paper(lib, tmp_path):
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    with pytest.raises(LibraryError, match="unknown paper"):
        files.attach_pdf(lib, "lp_nope", pdf)
    assert lib.conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 0


def test_a_second_different_pdf_needs_a_detach_first(lib, tmp_path):
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    with pytest.raises(LibraryError, match="detach"):
        files.attach_pdf(lib, pid, text_pdf(tmp_path / "b.pdf", [TITLE], marker="other"))
    assert lib.conn.execute("SELECT COUNT(*) FROM paper_files").fetchone()[0] == 1


def test_one_pdf_may_be_attached_to_two_papers(lib, tmp_path):
    a = add_paper(lib, "lp_00000000000000000001")
    b = add_paper(lib, "lp_00000000000000000002", title="Seasonal Migration of Alpine Goats")
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    files.attach_pdf(lib, a, pdf)
    res_b = files.attach_pdf(lib, b, pdf)
    assert res_b.check.status == "miss"
    assert res_b.queued  # the shared file is waiting, queued by the first attach
    assert [r["paper_id"] for r in open_items(lib)] == [b]


def test_attach_sets_the_dirty_flag(lib, tmp_path):
    pid = add_paper(lib)
    with lib.write() as conn:
        conn.execute("UPDATE library_meta SET value = '0' WHERE key = 'dirty'")
    files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    assert lib.conn.execute("SELECT value FROM library_meta WHERE key='dirty'").fetchone()[0] == "1"


def test_attach_inside_a_callers_transaction_rolls_back_with_it(lib, tmp_path):
    pid = add_paper(lib)
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    with pytest.raises(RuntimeError):
        with lib.write():
            files.attach_pdf(lib, pid, pdf)
            raise RuntimeError("caller changes its mind")
    assert lib.conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 0
    assert paper_file(lib, pid) is None


# --- conversion into the library -------------------------------------------------------


def test_conversion_through_the_fake_pool_lands_in_library_db(lib, tmp_path):
    from seedgraph.conversion import WarmMarkerBackend
    from support.converter import fake_pool

    pid = add_paper(lib)
    res = files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    pool = fake_pool()
    try:
        outcomes = files.convert_files(lib, queued=True, backend=WarmMarkerBackend(pool))
    finally:
        pool.shutdown()

    assert [(o.source_file_id, o.status) for o in outcomes] == [(res.source_file_id, "converted")]
    md_id = outcomes[0].markdown_id
    row = lib.conn.execute("SELECT * FROM markdown_documents WHERE markdown_id = ?", (md_id,)).fetchone()
    assert row["source_file_id"] == res.source_file_id
    assert (lib.root / row["storage_uri"]).is_file()
    assert row["storage_uri"] == f"markdown/{md_id.removeprefix('md_')}.md"
    run = lib.conn.execute("SELECT * FROM conversion_runs WHERE conversion_run_id = ?",
                           (row["conversion_run_id"],)).fetchone()
    assert run["run_status"] == "success" and run["converter_version"] == "1.10.2"
    assert paper_file(lib, pid)["markdown_id"] == md_id
    assert queued(lib) == []
    assert lib.conn.execute("SELECT converter_version FROM paper_converter WHERE paper_id = ?",
                            (pid,)).fetchone()[0] == "1.10.2"


def test_conversion_writes_nothing_under_a_cache_folder(lib, tmp_path):
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    files.convert_files(lib, queued=True, backend=FakeMarkerBackend(PAGED_MD))
    home = lib.root.parent
    assert not (home / "cache").exists()
    assert not (lib.root / "marker").exists()


class ProbingBackend:
    """Checks, while 'converting', that no transaction is held on the library."""

    version = "probe-1"

    def __init__(self, lib, markdown: str = PAGED_MD):
        self.lib = lib
        self.markdown = markdown
        self.seen: dict = {}

    def __call__(self, pdf_path, cfg):
        self.seen["in_transaction"] = self.lib.conn.in_transaction
        other = sqlite3.connect(str(self.lib.db_path), timeout=0, isolation_level=None)
        try:
            other.execute("BEGIN IMMEDIATE")  # fails at once if a writer lock is held
            self.seen["pending"] = other.execute(
                "SELECT COUNT(*) FROM conversion_runs WHERE run_status = 'pending'").fetchone()[0]
            other.execute("ROLLBACK")
        finally:
            other.close()
        return MarkerResult(markdown=self.markdown, block_json=None, warnings=[])


def test_no_transaction_is_held_across_the_conversion(lib, tmp_path):
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    probe = ProbingBackend(lib)

    [outcome] = files.convert_files(lib, queued=True, backend=probe)

    assert outcome.status == "converted"
    assert probe.seen == {"in_transaction": False, "pending": 1}


def test_converting_inside_a_callers_transaction_is_refused(lib, tmp_path):
    pid = add_paper(lib)
    res = files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    with pytest.raises(LibraryError, match="transaction"):
        with lib.write():
            convert_source_file(res.source_file_id, library=lib, backend=ProbingBackend(lib))


def test_a_failed_conversion_is_recorded_and_stays_queued(lib, tmp_path):
    class Boom(FakeMarkerBackend):
        def __call__(self, pdf_path, cfg):
            raise RuntimeError("invented converter failure")

    pid = add_paper(lib)
    res = files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))

    [outcome] = files.convert_files(lib, queued=True, backend=Boom())

    assert outcome.status == "failed" and "invented converter failure" in outcome.error
    runs = lib.conn.execute("SELECT run_status, error FROM conversion_runs").fetchall()
    assert [r[0] for r in runs] == ["failed"]
    assert queued(lib) == [res.source_file_id]
    assert paper_file(lib, pid)["markdown_id"] is None


def test_cache_api_raises_conversion_error_with_the_failed_row_kept(lib, tmp_path):
    class Boom(FakeMarkerBackend):
        def __call__(self, pdf_path, cfg):
            raise RuntimeError("invented")

    src = ingest_file(text_pdf(tmp_path / "a.pdf", [TITLE]), access_class="user_supplied_private",
                      acquisition_method="upload", library=lib)
    with pytest.raises(ConversionError):
        convert_source_file(src.source_file_id, library=lib, backend=Boom())
    assert lib.conn.execute("SELECT run_status FROM conversion_runs").fetchone()[0] == "failed"


def test_reconverting_reuses_the_markdown_and_runs_the_converter_once(lib, tmp_path):
    pid = add_paper(lib)
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    files.attach_pdf(lib, pid, pdf)
    backend = FakeMarkerBackend(PAGED_MD)
    [first] = files.convert_files(lib, queued=True, backend=backend)
    [again] = files.convert_files(lib, paper_id=pid, backend=backend)
    assert backend.call_count == 1
    assert again.markdown_id == first.markdown_id and again.status == "converted"


def test_convert_named_pdfs_ingests_them_unattached(lib, tmp_path):
    pdf = text_pdf(tmp_path / "loose.pdf", [TITLE])
    [outcome] = files.convert_files(lib, pdfs=[pdf], backend=FakeMarkerBackend(PAGED_MD))
    assert outcome.status == "converted" and outcome.paper_ids == ()
    assert lib.conn.execute("SELECT COUNT(*) FROM markdown_documents").fetchone()[0] == 1


def test_attaching_an_already_converted_pdf_links_its_markdown_without_queueing(lib, tmp_path):
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    [outcome] = files.convert_files(lib, pdfs=[pdf], backend=FakeMarkerBackend(PAGED_MD))
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, pdf)
    assert paper_file(lib, pid)["markdown_id"] == outcome.markdown_id
    assert queued(lib) == []


def test_an_unknown_check_is_rechecked_on_the_markdown_after_conversion(lib, tmp_path):
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, no_text_pdf(tmp_path / "scan.pdf"))
    files.convert_files(lib, queued=True, backend=FakeMarkerBackend(PAGED_MD))
    row = paper_file(lib, pid)
    assert row["pdf_check_status"] == "pass"
    assert json.loads(row["pdf_check_json"])["source"] == "markdown"


def test_an_unknown_check_that_misses_on_the_markdown_raises_the_item(lib, tmp_path):
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, no_text_pdf(tmp_path / "scan.pdf"))
    goats = PAGED_MD.replace(TITLE, "Seasonal Migration of Alpine Goats")
    files.convert_files(lib, queued=True, backend=FakeMarkerBackend(goats))
    assert paper_file(lib, pid)["pdf_check_status"] == "miss"
    assert [r["paper_id"] for r in open_items(lib)] == [pid]


def test_a_known_check_is_not_overwritten_by_conversion(lib, tmp_path):
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, text_pdf(tmp_path / "wrong.pdf", ["Seasonal Migration of Alpine Goats"]))
    files.convert_files(lib, queued=True, backend=FakeMarkerBackend(PAGED_MD))
    assert paper_file(lib, pid)["pdf_check_status"] == "miss"


def test_recheck_uses_the_markdown_when_there_is_one(lib, tmp_path):
    pid = add_paper(lib)
    files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    files.convert_files(lib, queued=True, backend=FakeMarkerBackend(PAGED_MD))
    with lib.write() as conn:
        conn.execute("UPDATE papers SET title = 'Seasonal Migration of Alpine Goats'")
    [check] = files.recheck_pdf(lib, pid)
    assert check.status == "miss"
    assert json.loads(paper_file(lib, pid)["pdf_check_json"])["source"] == "markdown"
    assert len(open_items(lib)) == 1


def test_convert_paper_without_a_file_is_refused(lib):
    pid = add_paper(lib)
    with pytest.raises(LibraryError, match="no PDF"):
        files.convert_files(lib, paper_id=pid, backend=FakeMarkerBackend(PAGED_MD))


# --- the cache API keeps its cache.db behaviour ----------------------------------------


def test_root_and_library_together_are_refused(lib, tmp_path):
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    with pytest.raises(Exception, match="not both"):
        ingest_file(pdf, access_class="user_supplied_private", acquisition_method="upload",
                    root=tmp_path / "other", library=lib)


def test_root_callers_still_get_a_cache_db(tmp_path):
    root = tmp_path / "oldhome"
    pdf = text_pdf(tmp_path / "a.pdf", [TITLE])
    src = ingest_file(pdf, access_class="user_supplied_private", acquisition_method="upload", root=root)
    md = convert_source_file(src.source_file_id, backend=FakeMarkerBackend(PAGED_MD), root=root)
    assert (root / "cache" / "cache.db").is_file()
    assert (root / "cache" / md.storage_uri).is_file()
    assert (root / "cache" / "marker").is_dir()  # manifest still written in cache mode
