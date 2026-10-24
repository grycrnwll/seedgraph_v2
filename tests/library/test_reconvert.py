"""Gate after conversion, forced-OCR re-conversion, re-matching and the re-note offer
(design §5.3, §5.5, §11 P5). All PDFs are synthetic and all text is invented."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from seedgraph.cache.marker_backend import MarkerResult
from seedgraph.library import attention, files
from seedgraph.library.reconvert import reconvert_ocr
from seedgraph.library.rematch import rematch
from seedgraph.library.store import open_library_at
from seedgraph.notes import ClaimDraft, NoteProvenance, current_note_id, save_note
from seedgraph.work import packages

from .test_files import text_pdf

TITLE = "Invented Ovens and the Morning Loaf"

# Quotes as the garbled default conversion has them, and as clean OCR text has them.
Q_FOLDED = ("The invented bakery estimator is identiÖed under di§erent oven settings.",
            "The invented bakery estimator is identified under different oven settings.")
Q_FFI = ("Its e¢ ciency reáects the made-up dough supply.",
         "Its efficiency reflects the made-up dough supply.")
Q_PLAIN = ("Bakeries open at an invented hour each morning.",
           "Bakeries open at an invented hour each morning.")
Q_SPACES = ("Ovens cool slowly\nafter the Örst batch.",
            "Ovens cool slowly after the first batch.")
Q_LOST = "A sentence that the invented OCR text lost entirely."


def _filler(garbled: bool) -> str:
    word = "speciÖc" if garbled else "specific"
    return " ".join(f"Sentence {k} notes a {word} invented detail about loaves." for k in range(40))


def garbled_md(*, with_lost: bool = True) -> str:
    quotes = " ".join(q[0] for q in (Q_FOLDED, Q_FFI, Q_PLAIN, Q_SPACES))
    lost = f" {Q_LOST}" if with_lost else ""
    return f"# {TITLE}\n\n{quotes}{lost}\n\n{_filler(True)}\n"


def clean_md(*, lost: bool = False) -> str:
    quotes = " ".join(q[1] for q in (Q_FOLDED, Q_FFI, Q_PLAIN, Q_SPACES))
    extra = f" {Q_LOST}" if lost else ""
    return f"# {TITLE}\n\n{quotes}{extra}\n\nA fresh line from the OCR pass.\n\n{_filler(False)}\n"


class OcrBackend:
    """Garbled text by default; ``ocr`` text when asked for forced OCR."""

    version = "fake-ocr-1"

    def __init__(self, default: str, ocr: str):
        self.default, self.ocr = default, ocr
        self.calls: list[bool] = []

    def __call__(self, pdf_path, cfg):
        self.calls.append(cfg.force_ocr)
        return MarkerResult(markdown=self.ocr if cfg.force_ocr else self.default,
                            block_json=None, warnings=[])


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def add_paper(lib, paper_id: str = "lp_00000000000000000042") -> str:
    now = datetime.now(timezone.utc).isoformat()
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES (?, ?, ?)",
                     (paper_id, TITLE, now))
    return paper_id


def converted_paper(lib, tmp_path, backend) -> tuple[str, str, str]:
    """A paper with a PDF converted by ``backend``'s default path: (paper, file, md)."""
    pid = add_paper(lib)
    res = files.attach_pdf(lib, pid, text_pdf(tmp_path / "a.pdf", [TITLE]))
    outcome = files.convert_file(lib, res.source_file_id, backend=backend)
    assert outcome.status == "converted"
    return pid, res.source_file_id, outcome.markdown_id


def markdown_of(lib, paper_id: str) -> str:
    return lib.conn.execute("SELECT markdown_id FROM paper_files WHERE paper_id = ?",
                            (paper_id,)).fetchone()[0]


def gate_row(lib, md_id: str):
    return lib.conn.execute("SELECT status, checker_version FROM text_quality WHERE markdown_id = ?",
                            (md_id,)).fetchone()


# --- the gate runs after every conversion ---------------------------------------------


def test_the_gate_runs_after_conversion_and_caches_its_verdict(lib, tmp_path):
    pid, _, md_id = converted_paper(lib, tmp_path, OcrBackend(garbled_md(), clean_md()))
    assert gate_row(lib, md_id)["status"] == "garbled"
    # Caching only: raising the hold-back item is batch creation's job.
    assert attention.list_items(lib, paper_id=pid, kind="text_garbled") == []


def test_a_clean_conversion_caches_a_pass(lib, tmp_path):
    _, _, md_id = converted_paper(lib, tmp_path, OcrBackend(clean_md(), clean_md()))
    assert gate_row(lib, md_id)["status"] == "pass"


# --- forced-OCR re-conversion ---------------------------------------------------------


def test_a_failed_forced_ocr_keeps_the_old_markdown_and_the_item_open(lib, tmp_path):
    backend = OcrBackend(garbled_md(), garbled_md(with_lost=False))
    pid, _, old_md = converted_paper(lib, tmp_path, backend)

    result = reconvert_ocr(lib, pid, backend=backend)

    assert backend.calls == [False, True]  # the second conversion asked for forced OCR
    assert result.status == "ocr_did_not_help"
    assert result.new_markdown_id not in (None, old_md)
    assert gate_row(lib, result.new_markdown_id)["status"] == "garbled"
    assert markdown_of(lib, pid) == old_md
    [item] = attention.list_items(lib, paper_id=pid, kind="text_garbled")
    assert item.status == "open" and "forced OCR did not help" in item.detail
    assert item.subject_json == attention.canonical_subject({"paper_id": pid, "markdown_id": old_md})
    assert result.rematches == ()


def test_a_failed_conversion_keeps_the_old_markdown(lib, tmp_path):
    class Boom(OcrBackend):
        def __call__(self, pdf_path, cfg):
            if cfg.force_ocr:
                raise RuntimeError("invented OCR failure")
            return super().__call__(pdf_path, cfg)

    backend = Boom(garbled_md(), clean_md())
    pid, _, old_md = converted_paper(lib, tmp_path, backend)

    result = reconvert_ocr(lib, pid, backend=backend)

    assert result.status == "conversion_failed" and "invented OCR failure" in result.detail
    assert markdown_of(lib, pid) == old_md


def test_identical_ocr_output_did_not_help(lib, tmp_path):
    backend = OcrBackend(garbled_md(), garbled_md())
    pid, _, old_md = converted_paper(lib, tmp_path, backend)

    result = reconvert_ocr(lib, pid, backend=backend)

    assert result.status == "ocr_did_not_help"
    assert markdown_of(lib, pid) == old_md


def test_a_passing_forced_ocr_switches_the_markdown_and_resolves_the_item(lib, tmp_path):
    backend = OcrBackend(garbled_md(), clean_md())
    pid, _, old_md = converted_paper(lib, tmp_path, backend)
    held = packages.hold_back_reason(lib, packages.load_paper_text(lib.conn, pid))
    assert held is not None  # batch creation raised the text_garbled item

    result = reconvert_ocr(lib, pid, backend=backend)

    assert result.status == "switched"
    assert markdown_of(lib, pid) == result.new_markdown_id != old_md
    item = attention.get_item(lib, held.attention_item_id)
    assert item.status == "resolved" and item.resolution == "reconverted_ocr"
    [rm] = result.rematches  # no note yet: nothing to carry, nothing offered
    assert rm.previously_anchored == 0 and rm.note_id is None and rm.attention_item_id is None
    assert result.as_dict()["status"] == "switched"


# --- re-matching the current note -----------------------------------------------------


def _draft(i: int, quote: str | None, *, status: str = "found") -> ClaimDraft:
    return ClaimDraft(claim_type="main_result", field_key=f"main_results[{i}]", status=status,
                      epistemic_type="llm_extracted", normalized_label=f"invented label {i}",
                      claim_text=f"Invented claim number {i}.", exact_quote=quote, confidence=0.7)


def note_on_garbled(lib, pid: str, md_id: str, *, with_lost: bool) -> str:
    """Save a note on the garbled markdown whose quotes are its garbled slices."""
    garbled = [Q_FOLDED[0], Q_FFI[0], Q_PLAIN[0], Q_SPACES[0]] + ([Q_LOST] if with_lost else [])
    drafts = [_draft(i, q) for i, q in enumerate(garbled)]
    drafts.append(_draft(99, None, status="not_found"))
    raw = json.dumps({"main_results": [{"exact_quote": q} for q in garbled]}, ensure_ascii=False)
    mh = lib.conn.execute("SELECT markdown_hash FROM markdown_documents WHERE markdown_id = ?",
                          (md_id,)).fetchone()[0]
    with lib.write() as conn:
        saved = save_note(conn, paper_id=pid, markdown_id=md_id, markdown_hash=mh, drafts=drafts,
                          note_text="An invented note.", archetype="empirical", raw_note_json=raw,
                          provenance=NoteProvenance(client="invented-agent", model="m-1"))
    assert saved.quotes_anchored == len(garbled)
    return saved.note_id


def note_row(lib, note_id: str):
    return lib.conn.execute(
        "SELECT n.*, r.extraction_mode, r.parent_extraction_run_id FROM structured_notes n "
        "JOIN extraction_runs r ON r.extraction_run_id = n.extraction_run_id WHERE n.note_id = ?",
        (note_id,)).fetchone()


def test_a_full_rematch_carries_the_note_as_a_rematch_version(lib, tmp_path):
    backend = OcrBackend(garbled_md(with_lost=False), clean_md())
    pid, _, old_md = converted_paper(lib, tmp_path, backend)
    old_note = note_on_garbled(lib, pid, old_md, with_lost=False)

    result = reconvert_ocr(lib, pid, backend=backend)

    [rm] = result.rematches
    assert (rm.status, rm.rematched, rm.previously_anchored) == ("carried", 4, 4)
    assert sorted(q.method for q in rm.quotes) == ["exact", "folded", "folded", "whitespace"]
    assert current_note_id(lib.conn, pid) == rm.note_id != old_note
    new, old = note_row(lib, rm.note_id), note_row(lib, old_note)
    assert new["markdown_id"] == result.new_markdown_id and new["version_no"] == old["version_no"] + 1
    assert new["extraction_mode"] == "rematch"
    assert new["parent_extraction_run_id"] == old["extraction_run_id"]
    claims = lib.conn.execute(
        "SELECT claim_text, status FROM extracted_claims WHERE structured_note_id = ? ORDER BY rowid",
        (rm.note_id,)).fetchall()
    old_claims = lib.conn.execute(
        "SELECT claim_text, status FROM extracted_claims WHERE structured_note_id = ? ORDER BY rowid",
        (old_note,)).fetchall()
    assert [tuple(c) for c in claims] == [tuple(c) for c in old_claims]
    quotes = [r[0] for r in lib.conn.execute(
        "SELECT es.exact_quote FROM extracted_claims c JOIN claim_spans cs ON cs.claim_id = c.claim_id "
        "JOIN evidence_spans es ON es.span_id = cs.span_id WHERE c.structured_note_id = ? "
        "AND es.markdown_id = ? ORDER BY c.rowid", (rm.note_id, result.new_markdown_id))]
    assert quotes == [Q_FOLDED[1], Q_FFI[1], Q_PLAIN[1], Q_SPACES[1]]
    assert Q_FOLDED[1] in new["raw_note_json"]
    assert attention.list_items(lib, paper_id=pid, kind="renote_offered") == []


def test_a_dropped_quote_offers_a_renote_and_the_old_version_stays_current(lib, tmp_path):
    backend = OcrBackend(garbled_md(), clean_md())  # the OCR text lost Q_LOST
    pid, _, old_md = converted_paper(lib, tmp_path, backend)
    old_note = note_on_garbled(lib, pid, old_md, with_lost=True)

    result = reconvert_ocr(lib, pid, backend=backend)

    assert result.status == "switched" and markdown_of(lib, pid) == result.new_markdown_id
    [rm] = result.rematches
    assert (rm.status, rm.rematched, rm.previously_anchored, rm.note_id) == ("renote_offered", 4, 5, None)
    assert current_note_id(lib.conn, pid) == old_note
    assert note_row(lib, old_note)["markdown_id"] == old_md
    storage = lib.conn.execute("SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?",
                               (old_md,)).fetchone()[0]
    assert (lib.root / storage).is_file()  # the old blob is kept
    [item] = attention.list_items(lib, paper_id=pid, kind="renote_offered")
    assert item.item_id == rm.attention_item_id and item.status == "open" and not item.hides_notes
    assert "4/5" in item.detail and "80%" in item.detail
    assert lib.conn.execute("SELECT COUNT(*) FROM structured_notes WHERE paper_id = ?",
                            (pid,)).fetchone()[0] == 1  # nothing re-noted


def test_rematch_without_a_note_does_nothing(lib, tmp_path):
    backend = OcrBackend(garbled_md(), clean_md())
    pid, _, old_md = converted_paper(lib, tmp_path, backend)
    rm = rematch(lib, pid, old_md, old_md)
    assert rm.status == "no_note" and rm.as_dict()["match_rate"] == 1.0
