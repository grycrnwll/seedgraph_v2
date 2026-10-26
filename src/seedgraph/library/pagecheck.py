"""The quote → PDF highlight measurement, ``seedgraph library pagecheck`` (stage 3 §3.4).

``run(lib, home, *, set_name, mode)`` measures a set of papers:

- **sets** (``<home>/ui/highlight_sets.json``, made once by hand from the prototype's
  ids and hashes; no repo code reads the prototype branch):
  ``proto5`` = PDF sha256s resolved through ``source_files`` → ``paper_files``;
  ``proto31`` = v2 work ids (+ ``v2_project``) resolved through ``v2_id_map``, plus the
  8-hex claim-id prefixes of every quoted claim; ``current`` = every paper with a
  current note.
- **modes**: ``prototype`` counts exactly the prototype's claims (``proto31``: by the
  stored prefixes; ``proto5``: every quoted non-lens claim of the papers' notes carried
  from the prototype's v2 project), across all note versions, rank-0 anchor;
  ``current`` counts the anchored claims of the current claims (``pages``).
- **tiers per quote**: page assigned (``how`` = marker / reanchored / none, from
  :class:`pages.PageAssigner`); then, against pypdfium2's text layer of the PDF that
  supplied the page, folded (``textfold``): ``verified`` (a hit overlaps the assigned
  page range), ``page_wrong`` (hits, none on the assigned pages, needle of at least
  ``MIN_DISTINCTIVE`` folded characters: fails the check),
  ``page_only`` (assigned, no hit anywhere), ``no_page`` (nothing assigned) or
  ``not_checked`` (verification off or the PDF is missing).
- **papers without page markers** (current markdown has no delimiters) are listed with
  their converter version, to offer re-conversion. Nothing is guessed.

The prototype's figures in ``<home>/ui/highlight_expected.json`` are the bar in
prototype mode (§10.2): ``proto5`` page-assigned equals the expected total; ``proto31``
counts the same number of quotes and its page-assigned count and ratio are each at
least the prototype's. Any page-wrong quote fails in every mode.

``open_readonly(lib_dir)`` opens a library with a ``mode=ro`` connection: the
measurement never migrates or writes the library. ``write_report`` writes
``<home>/ui/pagecheck/<stamp>/report.{json,md}``.
"""

from __future__ import annotations

import bisect
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

from ..errors import SeedgraphError
from .pages import HOW_MARKER, HOW_NONE, HOW_REANCHORED, PageAssigner, current_anchored_claims, current_markdown, pdf_path
from .store import Library, library_db_path
from .textfold import find_passage, fold

SETS = ("proto5", "proto31", "current")
MODES = ("prototype", "current")
#: A folded needle shorter than this (e.g. a math-heavy quote reduced to "asand") is not
#: distinctive: a hit elsewhere is chance, so it never makes a quote page-wrong.
MIN_DISTINCTIVE = 20
SETS_FILE = "highlight_sets.json"
EXPECTED_FILE = "highlight_expected.json"


class PagecheckError(SeedgraphError):
    """The measurement cannot run (bad set/mode, missing sets file)."""


@dataclass(frozen=True)
class QuoteCheck:
    paper_id: str
    claim_id: str
    span_id: str
    page: int | None
    page_end: int | None
    how: str
    tier: str  # verified | page_only | page_wrong | no_page | not_checked
    found_pages: tuple[int, ...] = ()
    warning: str | None = None


@dataclass(frozen=True)
class PaperRow:
    paper_id: str
    quotes: int
    page_assigned: int
    marker: int
    reanchored: int
    none: int
    verified: int
    page_only: int
    page_wrong: int
    not_checked: int
    has_markers: bool


@dataclass
class PageReport:
    set_name: str
    mode: str
    verify: bool
    papers: list[PaperRow]
    quotes: list[QuoteCheck]
    totals: dict[str, int]
    no_marker_papers: list[dict]
    unresolved: list[str]
    expected: dict | None
    verdict: str
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"set": self.set_name, "mode": self.mode, "verify": self.verify, "verdict": self.verdict,
                "reasons": self.reasons, "totals": self.totals, "expected": self.expected,
                "unresolved": self.unresolved, "no_marker_papers": self.no_marker_papers,
                "papers": [asdict(p) for p in self.papers], "quotes": [asdict(q) for q in self.quotes]}


# --- opening ----------------------------------------------------------------------------


@contextmanager
def open_readonly(lib_dir: Path | str) -> Iterator[Library]:
    """A :class:`Library` on a read-only connection (no migration, no writes)."""
    lib_dir = Path(lib_dir)
    db = library_db_path(lib_dir)
    if not db.is_file():
        raise PagecheckError(f"no library at {lib_dir}")
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    lib = Library(root=lib_dir, conn=conn)
    try:
        yield lib
    finally:
        conn.close()


# --- set resolution ---------------------------------------------------------------------


def _load_json(home: Path, name: str, *, required: bool) -> dict | None:
    path = Path(home) / "ui" / name
    if not path.is_file():
        if required:
            raise PagecheckError(f"{path} not found: create the highlight sets first (design §3.4)")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rank0_span(conn: sqlite3.Connection, claim_id: str) -> str | None:
    row = conn.execute("SELECT span_id FROM claim_spans WHERE claim_id = ? ORDER BY rank, id LIMIT 1",
                       (claim_id,)).fetchone()
    return row[0] if row else None


def _resolve(lib: Library, home: Path, set_name: str, mode: str
             ) -> tuple[list[str], dict[str, list[tuple[str, str]]], list[str]]:
    """``(paper_ids, claims per paper as (claim_id, span_id), unresolved entries)``."""
    conn = lib.conn
    unresolved: list[str] = []
    papers: list[str] = []
    claims: dict[str, list[tuple[str, str]]] = {}
    if set_name == "current":
        papers = [r[0] for r in conn.execute(
            "SELECT DISTINCT paper_id FROM note_current ORDER BY paper_id")]
    else:
        sets = _load_json(home, SETS_FILE, required=True) or {}
        spec = sets.get(set_name)
        if not spec:
            raise PagecheckError(f"{SETS_FILE} has no '{set_name}' set")
        project = spec.get("v2_project")
        if set_name == "proto5":
            for sha in spec.get("pdf_sha256", []):
                rows = conn.execute(
                    "SELECT DISTINCT pf.paper_id FROM source_files s "
                    "JOIN paper_files pf ON pf.source_file_id = s.source_file_id "
                    "WHERE s.file_hash = ? ORDER BY pf.paper_id", (sha,)).fetchall()
                if not rows:
                    unresolved.append(f"pdf {sha}")
                papers += [r[0] for r in rows if r[0] not in papers]
            if mode == "prototype":
                for pid in papers:
                    rows = conn.execute(
                        "SELECT c.claim_id FROM extracted_claims c "
                        "JOIN structured_notes n ON n.note_id = c.structured_note_id "
                        "WHERE c.paper_id = ? AND n.origin = ? AND c.claim_type NOT LIKE 'lens:%' "
                        "ORDER BY n.version_no, c.field_key, c.claim_id", (pid, f"v2:{project}")).fetchall()
                    claims[pid] = [(r[0], s) for r in rows if (s := _rank0_span(conn, r[0]))]
        else:  # proto31
            for entry in spec.get("papers", []):
                work = entry["v2_work_id"]
                row = conn.execute(
                    "SELECT v3_id FROM v2_id_map WHERE v2_project = ? AND v2_table = 'works' AND v2_id = ?",
                    (project, work)).fetchone()
                if row is None:
                    unresolved.append(f"v2 work {work}")
                    continue
                pid = row[0]
                if pid not in papers:
                    papers.append(pid)
                if mode != "prototype":
                    continue
                for prefix in entry.get("claim_prefixes", []):
                    rows = conn.execute(
                        "SELECT claim_id FROM extracted_claims WHERE paper_id = ? AND claim_id LIKE ?",
                        (pid, f"claim_{prefix}%")).fetchall()
                    span = _rank0_span(conn, rows[0][0]) if len(rows) == 1 else None
                    if span is None:
                        unresolved.append(f"claim prefix {prefix} of {work}")
                        continue
                    claims.setdefault(pid, []).append((rows[0][0], span))
    if mode == "current":
        claims = {pid: current_anchored_claims(lib, pid) for pid in papers}
    return papers, claims, unresolved


# --- verification -----------------------------------------------------------------------


class _PdfText:
    """Folded pypdfium2 text of each PDF, concatenated, with each page's start offset."""

    def __init__(self, lib: Library):
        self.lib = lib
        self._cache: dict[str, tuple[str, list[int]] | None] = {}

    def get(self, source_file_id: str) -> tuple[str, list[int]] | None:
        if source_file_id not in self._cache:
            self._cache[source_file_id] = self._load(source_file_id)
        return self._cache[source_file_id]

    def _load(self, source_file_id: str) -> tuple[str, list[int]] | None:
        path = pdf_path(self.lib, source_file_id)
        if path is None:
            return None
        import pypdfium2 as pdfium

        try:
            doc = pdfium.PdfDocument(str(path))
        except Exception:  # noqa: BLE001 - unreadable PDF: not checkable
            return None
        parts: list[str] = []
        starts: list[int] = []
        pos = 0
        try:
            for i in range(len(doc)):
                page = doc[i]
                tp = page.get_textpage()
                try:
                    text = fold(tp.get_text_range()).text
                finally:
                    tp.close()
                    page.close()
                starts.append(pos)
                parts.append(text)
                pos += len(text)
        finally:
            doc.close()
        return "".join(parts), starts


def _tier(pdf: tuple[str, list[int]] | None, needle: str, page: int, page_end: int
          ) -> tuple[str, tuple[int, ...]]:
    if pdf is None or not needle:
        return "not_checked", ()
    text, starts = pdf
    found: set[int] = set()
    overlaps = False
    for s, e in find_passage(text, needle):
        first = bisect.bisect_right(starts, s)  # 1-based page of the hit's start
        last = bisect.bisect_right(starts, e - 1)
        found.update(range(first, last + 1))
        if first <= page_end and last >= page:
            overlaps = True
    if overlaps:
        return "verified", tuple(sorted(found))
    if found and len(needle) >= MIN_DISTINCTIVE:
        return "page_wrong", tuple(sorted(found))
    return "page_only", tuple(sorted(found))


# --- the run ----------------------------------------------------------------------------


def _converter_version(lib: Library, markdown_id: str | None) -> str | None:
    if markdown_id is None:
        return None
    row = lib.conn.execute(
        "SELECT r.converter_version FROM markdown_documents m "
        "JOIN conversion_runs r ON r.conversion_run_id = m.conversion_run_id WHERE m.markdown_id = ?",
        (markdown_id,)).fetchone()
    return row[0] if row else None


def _gate(set_name: str, mode: str, totals: dict[str, int], expected: dict | None) -> list[str]:
    reasons = []
    if totals["page_wrong"]:
        reasons.append(f"{totals['page_wrong']} page-wrong quote(s): found only on another page")
    if mode != "prototype" or not expected:
        return reasons
    want_q, want_a = expected.get("quoted"), expected.get("page_assigned")
    got_q, got_a = totals["quotes"], totals["page_assigned"]
    if set_name == "proto5":
        if want_a is not None and got_a != want_a:
            reasons.append(f"page-assigned {got_a} != expected {want_a}")
    else:
        if want_q is not None and got_q != want_q:
            reasons.append(f"{got_q} quotes counted, the prototype counted {want_q}")
        if want_a is not None and got_a < want_a:
            reasons.append(f"page-assigned {got_a} < the prototype's {want_a}")
        if want_q and want_a is not None and got_q and got_a / got_q < want_a / want_q:
            reasons.append(f"page-assigned ratio {got_a / got_q:.4f} < the prototype's {want_a / want_q:.4f}")
    return reasons


def run(lib: Library, home: Path | str, *, set_name: str, mode: str, verify: bool = True) -> PageReport:
    """Measure ``set_name`` in ``mode`` (see the module doc). Reads only."""
    if set_name not in SETS:
        raise PagecheckError(f"unknown set {set_name!r}; one of {', '.join(SETS)}")
    if mode not in MODES:
        raise PagecheckError(f"unknown mode {mode!r}; one of {', '.join(MODES)}")
    if mode == "prototype" and set_name == "current":
        raise PagecheckError("prototype mode needs a prototype set (proto5 or proto31)")
    home = Path(home)
    papers, claims, unresolved = _resolve(lib, home, set_name, mode)
    assigner = PageAssigner(lib)
    pdfs = _PdfText(lib)
    quotes: list[QuoteCheck] = []
    rows: list[PaperRow] = []
    no_markers: list[dict] = []
    for pid in papers:
        mine: list[QuoteCheck] = []
        for claim_id, span_id in claims.get(pid, []):
            a = assigner.assign(span_id, paper_id=pid)
            if a.page is None:
                tier, found = "no_page", ()
            elif not verify:
                tier, found = "not_checked", ()
            else:
                tier, found = _tier(pdfs.get(a.source_file_id) if a.source_file_id else None,
                                    fold(a.quote).text, a.page, a.page_end or a.page)
            mine.append(QuoteCheck(pid, claim_id, span_id, a.page, a.page_end, a.how, tier, found, a.warning))
        cur_md, _ = current_markdown(lib, pid)
        has_markers = assigner.has_markers(cur_md)
        count = {k: sum(1 for q in mine if q.how == k) for k in (HOW_MARKER, HOW_REANCHORED, HOW_NONE)}
        tiers = {k: sum(1 for q in mine if q.tier == k)
                 for k in ("verified", "page_only", "page_wrong", "not_checked")}
        rows.append(PaperRow(pid, len(mine), count[HOW_MARKER] + count[HOW_REANCHORED], count[HOW_MARKER],
                             count[HOW_REANCHORED], count[HOW_NONE], tiers["verified"], tiers["page_only"],
                             tiers["page_wrong"], tiers["not_checked"], has_markers))
        if not has_markers:
            no_markers.append({"paper_id": pid, "markdown_id": cur_md,
                               "converter_version": _converter_version(lib, cur_md),
                               "quotes": len(mine), "page_assigned": rows[-1].page_assigned})
        quotes += mine
    totals = {k: sum(getattr(r, k) for r in rows) for k in
              ("quotes", "page_assigned", "marker", "reanchored", "none", "verified", "page_only",
               "page_wrong", "not_checked")}
    totals["papers"] = len(rows)
    totals["papers_without_markers"] = len(no_markers)
    expected_all = _load_json(home, EXPECTED_FILE, required=False) or {}
    expected = expected_all.get(set_name) if set_name != "current" else None
    reasons = _gate(set_name, mode, totals, expected)
    return PageReport(set_name, mode, verify, rows, quotes, totals, no_markers, unresolved, expected,
                      "fail" if reasons else "pass", reasons)


# --- the report -------------------------------------------------------------------------


def render_markdown(report: PageReport) -> str:
    t = report.totals
    ratio = f"{t['page_assigned'] / t['quotes']:.4f}" if t["quotes"] else "n/a"
    lines = [
        f"# pagecheck: set `{report.set_name}`, mode `{report.mode}`", "",
        f"Verdict: **{report.verdict}**" + (": " + "; ".join(report.reasons) if report.reasons else ""), "",
        "| measure | count |", "|---|---|",
        f"| papers | {t['papers']} |", f"| quotes | {t['quotes']} |",
        f"| page assigned | {t['page_assigned']} (ratio {ratio}) |",
        f"| - marker / reanchored / none | {t['marker']} / {t['reanchored']} / {t['none']} |",
        f"| passage verified | {t['verified']} |", f"| page-only | {t['page_only']} |",
        f"| page wrong | {t['page_wrong']} |", f"| not checked | {t['not_checked']} |",
        f"| papers without page markers | {t['papers_without_markers']} |", "",
    ]
    if report.expected:
        lines += [f"Expected (prototype): {json.dumps(report.expected)}", ""]
    if report.unresolved:
        lines += ["## Unresolved set entries", ""] + [f"- {u}" for u in report.unresolved] + [""]
    if report.no_marker_papers:
        lines += ["## Papers without page markers (offer re-conversion)", "",
                  "| paper | markdown | converter | quotes |", "|---|---|---|---|"]
        lines += [f"| {p['paper_id']} | {p['markdown_id']} | {p['converter_version']} | {p['quotes']} |"
                  for p in report.no_marker_papers] + [""]
    lines += ["## Per paper", "",
              "| paper | quotes | assigned | marker | reanchored | none | verified | page-only | wrong |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r.paper_id} | {r.quotes} | {r.page_assigned} | {r.marker} | {r.reanchored} | {r.none} | "
              f"{r.verified} | {r.page_only} | {r.page_wrong} |" for r in report.papers]
    wrong = [q for q in report.quotes if q.tier == "page_wrong"]
    if wrong:
        lines += ["", "## Page-wrong quotes", ""]
        lines += [f"- {q.paper_id} {q.claim_id}: assigned {q.page}-{q.page_end}, found on {list(q.found_pages)}"
                  for q in wrong]
    return "\n".join(lines) + "\n"


def write_report(report: PageReport, home: Path | str, *, stamp: str | None = None) -> Path:
    """Write ``report.json`` and ``report.md`` to ``<home>/ui/pagecheck/<stamp>/``."""
    stamp = stamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    out = Path(home) / "ui" / "pagecheck" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report.as_dict(), indent=1), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return out
