"""Fixture library for the paper page tests (U8): a held paper with a synthetic PDF, Marker
page delimiters and a note; a held paper without page markers; a cited-only paper.
Invented text only."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from seedgraph.cache.marker_backend import MarkerResult
from seedgraph.library import files
from seedgraph.library.store import open_library_at
from seedgraph.notes import format as fmt
from seedgraph.notes import save



def text_pdf(path: Path, pages: list[str], *, marker: str = "") -> Path:
    """A minimal valid PDF whose pages carry ``pages`` as a Helvetica text layer (as in
    tests/library/test_files.py)."""
    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()]
    for i, text in enumerate(pages):
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({esc}) Tj ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode())
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


PAGES = [
    "We study whether invented widgets grow in a made-up market.",
    "Our two-step widget estimator counts widgets twice and averages. "
    "Units do not anticipate the invented programme before it starts.",
    "Widgets grow by an invented amount. Widgets grow by an invented amount. "
    "The frumious index appears once in this invented text.",
]
DELIM = "-" * 48
RQ_QUOTE = PAGES[0]
METHOD_QUOTE = "Our two-step widget estimator counts widgets twice and averages."
ASSUMPTION_QUOTE = "Units do not anticipate the invented programme before it starts."
RESULT_QUOTE = "The frumious index appears once in this invented text."

HELD = "lp_00000000000000000801"
PLAIN = "lp_00000000000000000802"
CITED = "lp_00000000000000000803"
TITLE = "An Invented Study of Widget Growth"


def filler(n: int) -> str:
    """Ordinary invented prose so each markdown page passes the text-quality gate."""
    return (f"On page {n + 1} of this made up study we write more about the people who make small parts, "
            "the time they spend at work each day, the places where they live and travel, the food they "
            "like to share with friends, and the many simple things they think about when the long day "
            "is finally over and the house is quiet again.")


def paged_md(pages: list[str]) -> str:
    return "# " + TITLE + "\n" + "".join(f"\n\n{{{n}}}{DELIM}\n\n{body}\n\n{filler(n)}"
                                         for n, body in enumerate(pages)) + "\n"


def plain_md(pages: list[str]) -> str:
    return "# " + TITLE + "\n\n" + "\n\n".join(f"{body}\n\n{filler(n)}" for n, body in enumerate(pages)) + "\n"


class FixedBackend:
    version = "fake-fixed-1"

    def __init__(self, markdown: str):
        self.markdown = markdown

    def __call__(self, pdf_path, cfg):
        return MarkerResult(markdown=self.markdown, block_json=None, warnings=[])


def raw_note(method_text: str = "A two-step widget estimator.") -> dict:
    return {
        "research_question": {"claim_text": "Do invented widgets grow?", "status": "found",
                              "exact_quote": RQ_QUOTE},
        "method_or_model": [{"claim_text": method_text, "normalized_label": "widget estimator",
                             "status": "found", "method_type": "estimator",
                             "plain_words": "Count the widgets twice and average.",
                             "exact_quote": METHOD_QUOTE}],
        "assumptions": [{"claim_text": "No anticipation of the invented programme.",
                         "normalized_label": "no anticipation", "status": "found",
                         "assumption_type": "identification", "exact_quote": ASSUMPTION_QUOTE},
                        {"claim_text": "Widgets are counted without error.",
                         "normalized_label": "exact counting", "status": "inferred",
                         "assumption_type": "general", "inferred_explanation": "invented reason"}],
        "main_results": [{"claim_text": "A frumious index appears.", "normalized_label": "frumious index",
                          "status": "found", "result_type": "main", "exact_quote": RESULT_QUOTE}],
    }


@dataclass
class Fixture:
    lib_dir: Path
    held: str
    plain: str
    cited: str
    note_id: str
    markdown_id: str
    source_file_id: str

    def open(self):
        return open_library_at(self.lib_dir)

    def claim(self, field_key: str, note_id: str | None = None) -> str:
        with self.open() as lib:
            return lib.conn.execute(
                "SELECT claim_id FROM extracted_claims WHERE structured_note_id = ? AND field_key = ?",
                (note_id or self.note_id, field_key)).fetchone()[0]


def _paper(lib, paper_id: str, title: str | None, **cols) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, title_source, authors_json, year, venue, created_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (paper_id, title, cols.get("title_source", "v2"),
                      json.dumps(cols.get("authors", ["Ada Inventor", "Bo Madeup"])),
                      cols.get("year", 2031), cols.get("venue", "Journal of Invented Results"), now))
        for id_type, value in cols.get("ids", ()):
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, ?, ?, 'v2')",
                         (paper_id, id_type, value))


def _held(lib, tmp: Path, paper_id: str, markdown: str, name: str) -> tuple[str, str]:
    res = files.attach_pdf(lib, paper_id, text_pdf(tmp / name, PAGES, marker=paper_id))
    outcome = files.convert_file(lib, res.source_file_id, backend=FixedBackend(markdown))
    assert outcome.status == "converted", outcome
    return res.source_file_id, outcome.markdown_id


def save_note(lib, paper_id: str, markdown_id: str, raw: dict | None = None):
    raw = raw or raw_note()
    note, errors = fmt.validate_note(raw)
    assert errors == []
    norm = fmt.normalize_parts([note])
    mh = lib.conn.execute("SELECT markdown_hash FROM markdown_documents WHERE markdown_id = ?",
                          (markdown_id,)).fetchone()[0]
    with lib.write() as conn:
        return save.save_note(
            conn, paper_id=paper_id, markdown_id=markdown_id, markdown_hash=mh, drafts=norm.drafts,
            note_text=norm.note_text, archetype=norm.archetype, raw_note_json=json.dumps(raw),
            provenance=save.NoteProvenance(client="claude", model="invented-model"))


def build(home: Path, tmp: Path, *, note: bool = True) -> Fixture:
    lib_dir = home / "library"
    lib = open_library_at(lib_dir)
    try:
        _paper(lib, HELD, TITLE, ids=[("doi", "10.9999/invented.2031.001")])
        _paper(lib, PLAIN, "A Paper Without Page Markers")
        _paper(lib, CITED, "A Cited-Only Invented Paper", title_source="openalex")
        sf, md = _held(lib, tmp, HELD, paged_md(PAGES), "held.pdf")
        _held(lib, tmp, PLAIN, plain_md(PAGES), "plain.pdf")
        note_id = save_note(lib, HELD, md).note_id if note else ""
    finally:
        lib.close()
    return Fixture(lib_dir, HELD, PLAIN, CITED, note_id, md, sf)


# --- HTTP helpers ------------------------------------------------------------------------

def token(client, path: str = "/") -> str:
    return re.search(r'name="csrf-token" content="([^"]+)"', client.get(path).text).group(1)


def post(client, path: str, data: dict | None = None, *, htmx: bool = True, csrf: bool = True, **kw):
    """POST a form the way the page does (CSRF field; htmx header for partial swaps)."""
    data = dict(data or {})
    if csrf:
        data = {"csrf_token": token(client), **data}
    headers = {"HX-Request": "true"} if htmx else {}
    return client.post(path, data=data, headers=headers, **kw)
