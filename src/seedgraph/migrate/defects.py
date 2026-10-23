"""Defect rules of the migration (design §7.3). Pure: no DB, no clock, no network.

The plan builder (:mod:`seedgraph.migrate.plan`) calls these to:

* pick each held paper's markdown, re-linking a work that points at another PDF's
  markdown to its own PDF's latest successful markdown, or recording that it must be
  queued for conversion (:func:`choose_markdown`);
* recognise markdown produced by a test/fake converter (:func:`is_test_converter`);
* run the shared title matcher against a paper's markdown pages, falling back to the
  PDF text layer when the markdown check is not ``pass`` (:func:`check_paper`);
* recognise malformed titles and repair them offline from v2's ``provider_cache``
  (:func:`malformed_reason`, :func:`title_from_provider_cache`);
* describe attention items (:class:`AttentionSpec`); the apply stage writes them with
  :func:`seedgraph.library.attention.raise_item`.

Double imports need no rule here: they are the automatic merges of
:func:`seedgraph.library.identity.plan_identity`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from seedgraph.library.titlecheck import PdfCheck, check_pdf_text, first_pages_from_markdown

#: ``check_pdf_text`` statuses, best first. ``unknown`` (no evidence) never beats a miss.
STATUS_RANK = {"pass": 3, "weak": 2, "miss": 1, "unknown": 0}

#: Publisher / tooling boilerplate seen in place of real titles (lower-cased, exact).
BOILERPLATE_TITLES = frozenset({
    "untitled", "untitled document", "microsoft word", "title page", "front matter", "cover page",
    "table of contents", "author manuscript", "accepted manuscript", "full text", "pdf", "document",
    "abstract", "article", "contents",
})
_HASH_RE = re.compile(r"^[0-9a-f]{16,}$")
_UUID_RE = re.compile(r"^[0-9a-f-]{32,}$")
_FILE_EXT_RE = re.compile(r"\.(pdf|docx?|tex|txt|md|html?)$", re.IGNORECASE)


@dataclass(frozen=True)
class AttentionSpec:
    """One attention item to raise (``item_id`` = hash of ``kind`` + ``subject``)."""

    kind: str
    subject: Mapping[str, Any]
    paper_id: str | None
    detail: str | None
    hides_notes: bool = False

    def sort_key(self) -> tuple[str, str, str]:
        return (self.kind, self.paper_id or "", json.dumps(self.subject, sort_keys=True))


# --- converters and markdown choice ---------------------------------------------------

def is_test_converter(run: Mapping[str, Any] | None) -> bool:
    """A conversion run made by a test/fake converter (e.g. ``converter_version='fake-0.0.0'``)."""
    if not run:
        return False
    version = str(run.get("converter_version") or "").lower()
    name = str(run.get("converter_name") or "").lower()
    return version.startswith(("fake", "test")) or name in ("fake", "test", "fake_marker")


@dataclass(frozen=True)
class MarkdownChoice:
    """The paper's current markdown and what was repaired to get it.

    ``repair``: ``None`` (pointer kept), ``'relinked'`` (foreign markdown replaced by the
    PDF's own), ``'relink_failed'`` (foreign markdown and no own markdown: queue it),
    ``'linked'`` (no pointer, own markdown found), ``'test_converter'`` (only fake
    converter output: not carried as the paper's conversion), ``'pointer_missing'``.
    """

    markdown_id: str | None
    repair: str | None
    pointed_markdown_id: str | None
    foreign: bool
    test_converter_markdown_id: str | None = None


def choose_markdown(
    source_file_id: str,
    pointed_markdown_id: str | None,
    markdown_by_sf: Mapping[str, Sequence[Mapping[str, Any]]],
    markdown_by_id: Mapping[str, Mapping[str, Any]],
    runs_by_id: Mapping[str, Mapping[str, Any]],
) -> MarkdownChoice:
    """Pick the paper's markdown for its primary PDF ``source_file_id`` (§7.3)."""
    own = sorted(markdown_by_sf.get(source_file_id, ()), key=lambda m: (m["created_at"], m["markdown_id"]))
    good = [m for m in own if not is_test_converter(runs_by_id.get(m["conversion_run_id"]))]
    fake = [m for m in own if m not in good]
    fake_id = fake[-1]["markdown_id"] if fake else None
    latest = good[-1]["markdown_id"] if good else None
    pointed = markdown_by_id.get(pointed_markdown_id) if pointed_markdown_id else None

    if pointed_markdown_id and pointed is None:
        return MarkdownChoice(latest, "pointer_missing" if latest else "relink_failed",
                              pointed_markdown_id, False, fake_id)
    if pointed is not None and pointed["source_file_id"] != source_file_id:
        return MarkdownChoice(latest, "relinked" if latest else "relink_failed", pointed_markdown_id, True, fake_id)
    if pointed is not None:
        if is_test_converter(runs_by_id.get(pointed["conversion_run_id"])):
            return MarkdownChoice(latest, "relinked" if latest else "test_converter",
                                  pointed_markdown_id, False, pointed["markdown_id"])
        return MarkdownChoice(pointed["markdown_id"], None, pointed_markdown_id, False, fake_id)
    if latest:
        return MarkdownChoice(latest, "linked", None, False, fake_id)
    return MarkdownChoice(None, "test_converter" if fake_id else None, None, False, fake_id)


# --- title check -----------------------------------------------------------------------

def _best(checks: Sequence[tuple[str, PdfCheck]]) -> tuple[str, PdfCheck]:
    best = checks[0]
    for item in checks[1:]:
        if (STATUS_RANK[item[1].status], item[1].score) > (STATUS_RANK[best[1].status], best[1].score):
            best = item
    return best


def _check_pages(title: str | None, dois: Sequence[str], pages: Sequence[str]) -> PdfCheck:
    results = [check_pdf_text(title, doi, pages) for doi in (sorted(dois) or [None])]
    return _best([("", r) for r in results])[1]


@dataclass(frozen=True)
class PaperCheck:
    status: str
    json: str


def check_paper(
    title: str | None,
    dois: Sequence[str],
    markdown_text: str | None,
    pdf_pages: Callable[[], Sequence[str] | None],
) -> PaperCheck:
    """§7.2: check the markdown's first pages; if not ``pass``, also the PDF text layer.

    The best result wins (``pass`` > ``weak`` > ``miss`` > ``unknown``, then score).
    ``json`` records the winner and every check made.
    """
    checks: list[tuple[str, PdfCheck]] = []
    if markdown_text:
        checks.append(("markdown", _check_pages(title, dois, first_pages_from_markdown(markdown_text))))
    if not checks or checks[-1][1].status != "pass":
        pages = pdf_pages()
        if pages is not None:
            checks.append(("pdf", _check_pages(title, dois, pages)))
    if not checks:
        return PaperCheck("unknown", json.dumps({"status": "unknown", "score": 0.0, "via": None, "source": None,
                                                  "evidence": "no text", "checked": []},
                                                 sort_keys=True, separators=(",", ":")))
    source, best = _best(checks)
    payload = {
        "status": best.status, "score": best.score, "via": best.via, "evidence": best.evidence,
        "source": source,
        "checked": [{"source": s, "status": c.status, "score": c.score, "via": c.via} for s, c in checks],
    }
    # Same keys as library.files' standing check (status/score/via/evidence/source), plus ``checked``.
    return PaperCheck(best.status, json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


# --- malformed titles ----------------------------------------------------------------

def malformed_reason(title: str | None) -> str | None:
    """Why ``title`` is malformed (§7.3), or ``None`` if it looks like a real title."""
    if title is None or not title.strip():
        return "empty"
    t = title.strip()
    low = t.lower()
    if "�" in t:
        return "replacement_character"
    if _FILE_EXT_RE.search(t):
        return "filename_like"
    if _HASH_RE.match(low) or _UUID_RE.match(low):
        return "hash_like"
    if " " not in t and ("_" in t or "-" in t):
        return "filename_like"
    if low in BOILERPLATE_TITLES or low.startswith(("microsoft word - ", "untitled")):
        return "boilerplate"
    if len(t) < 10:
        return "too_short"
    if len(t) > 300:
        return "too_long"
    return None


def _title_in_response(response: Any) -> str | None:
    if isinstance(response, dict):
        for key in ("title", "display_name"):
            value = response.get(key)
            if isinstance(value, list):
                value = value[0] if value else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


@dataclass(frozen=True)
class TitleRepair:
    title: str
    provider: str
    request_key: str


def title_from_provider_cache(dois: Sequence[str], provider_rows: Sequence[Mapping[str, Any]]) -> TitleRepair | None:
    """A well-formed title for one of ``dois`` from v2's cached ``by_doi`` responses.

    Rows are tried in ``(provider, request_key)`` order; responses that are not a JSON
    object with a well-formed ``title``/``display_name`` are skipped. Offline only.
    """
    keys = {f"by_doi:doi={d}" for d in dois}
    for row in sorted(provider_rows, key=lambda r: (r["provider"], r["request_key"])):
        if row["request_key"] not in keys or row.get("response") is None:
            continue
        try:
            response = json.loads(row["response"])
        except (TypeError, ValueError):
            continue
        title = _title_in_response(response)
        if title and malformed_reason(title) is None:
            return TitleRepair(title, row["provider"], row["request_key"])
    return None
