"""The ``metadata`` task (design §2.2, §6): bibliographic metadata from the cover page.

- Package: the first :data:`HEAD_CHARS` characters of the current markdown (the cover
  and title page, as v2 ``extraction/metadata.py``) plus the deterministic DOI/arXiv
  hits from that same region (never a cited work's id from a later reference list).
- Output: ``{title, authors, year, venue, doi, arxiv}``, each null when unknown.
- Save: fill empty fields only, so a researcher-set field is never overwritten;
  the title is cleaned first (footnote marks, ``library.metadata_quality.clean_title``)
  and a generic one ("Contents", "Abstract", the venue's name …) is not saved: a
  ``metadata_generic_title`` attention item records the proposed value and its source,
  and ``metadata_needed`` stays open;
  ``title_source = 'ollama'`` for an Ollama worker, else ``'agent'``; new ids into
  ``identifiers`` (an id already held by another paper is skipped and reported); the
  worker's id wins, the deterministic hit fills a null; resolves the paper's open
  ``metadata_needed`` item once it has a title.
- After the commit (:meth:`MetadataTask.after_commit`, called by the routine outside the
  submit transaction): the paper's DOI is verified with the providers
  (``library.metadata_verify.verify``) when a chain is set for this context
  (``metadata_verify.set_chain``; the Ollama worker sets its ``chain``). No chain, no
  lookup: the title stays ``'ollama'`` / ``'agent'`` (unverified).
- Workers: Ollama (the default worker for this task, P6), interactive, background.
  Only the text-quality gate holds a paper back (a title miss is what metadata fixes).

The deterministic scan is a copy of v2 ``extraction.metadata.deterministic_identifiers``
(P8 trims that module), normalised with ``library.identity.normalize_ids``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..library.identity import normalize_ids
from ..library.metadata_quality import clean_title, generic_title_reason
from . import packages
from .tasks import (
    AcceptedPart,
    CheckResult,
    Eligibility,
    Instructions,
    PackagePlan,
    SaveContext,
    SaveResult,
    Selection,
    resolve_selection,
)

if TYPE_CHECKING:
    from ..library.store import Library
    from .store import BatchItem, WorkPackage

HEAD_CHARS = 2600
TASK_VERSION = "metadata-1"
METADATA_NEEDED = "metadata_needed"
GENERIC_TITLE = "metadata_generic_title"
FIELDS = ("title", "authors", "year", "venue", "doi", "arxiv")

_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"arxiv\.org/abs/([A-Za-z0-9][A-Za-z0-9.\-/]*)", re.IGNORECASE)
_ARXIV_LABELED_RE = re.compile(r"arxiv:\s*([A-Za-z0-9][A-Za-z0-9.\-/]*)", re.IGNORECASE)

INSTRUCTIONS = """# Metadata task (`metadata`, version {version})

Read the package text: the first page of one paper (its cover and title block), followed by
any DOI or arXiv id found on it. Write ONE JSON object with exactly these keys to the
package's `result_path` and submit it:

| key | type | what |
|---|---|---|
| `title` | string or null | the paper's own title, as printed |
| `authors` | array of strings or null | the authors in order, "First Last" |
| `year` | integer or null | the publication year |
| `venue` | string or null | the journal, conference or series |
| `doi` | string or null | the paper's own DOI (never a cited work's) |
| `arxiv` | string or null | the paper's own arXiv id |

Use only the package text. Use null for anything the text does not state. Ignore
publisher or archive boilerplate (download notices, terms of use).

Example: {{"title": "An Invented Study", "authors": ["Ada Inventor"], "year": 2019,
"venue": "Journal of Invented Results", "doi": null, "arxiv": null}}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def deterministic_identifiers(markdown: str) -> dict[str, str | None]:
    """The paper's own DOI / arXiv id from its cover region (v2 rule), normalised."""
    text = (markdown or "")[:HEAD_CHARS]
    raw: list[tuple[str, str]] = []
    m = _DOI_RE.search(text)
    if m is not None:
        raw.append(("doi", m.group(0).rstrip(".,;)]}>\"'")))
    um = _ARXIV_URL_RE.search(text) or _ARXIV_LABELED_RE.search(text)
    if um is not None:
        raw.append(("arxiv", um.group(1).rstrip(".,;)]}>\"'")))
    found = dict(normalize_ids(raw))
    return {"doi": found.get("doi"), "arxiv": found.get("arxiv")}


def needs_metadata(conn: sqlite3.Connection, paper_id: str) -> bool:
    """An open ``metadata_needed`` item, or no title."""
    if conn.execute("SELECT 1 FROM attention_items WHERE paper_id = ? AND kind = ? AND status = 'open'",
                    (paper_id, METADATA_NEEDED)).fetchone():
        return True
    row = conn.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return row is not None and not (row[0] or "").strip()


def _package_text(markdown: str) -> str:
    head = markdown[:HEAD_CHARS]
    ids = deterministic_identifiers(markdown)
    hits = [f"- {k}: {v}" for k, v in ids.items() if v]
    trailer = "\n\n---\nIdentifiers found on this page:\n" + ("\n".join(hits) if hits else "- none") + "\n"
    return head + trailer


class MetadataTask:
    name = "metadata"
    version = TASK_VERSION
    subject_kind = "paper"
    allowed_workers = frozenset({"ollama", "interactive", "background"})

    def __init__(self, lib: Library) -> None:
        self.lib = lib

    def resolve(self, lib: Library, query: Selection) -> list[str]:
        return resolve_selection(lib, query)

    def eligibility(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> Eligibility:
        if not lib.conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (subject_id,)).fetchone():
            return Eligibility("failed", reason="not a library paper")
        paper = packages.load_paper_text(lib.conn, subject_id)
        if paper is None:
            return Eligibility("failed", reason="no converted markdown (or its file is missing)")
        binding = dict(markdown_id=paper.markdown_id, markdown_hash=paper.markdown_hash)
        if not options.get("replace") and not needs_metadata(lib.conn, subject_id):
            return Eligibility("preserved", reason="the paper has a title and no open metadata_needed item",
                               **binding)
        held = packages.gate_hold(lib, paper)
        if held is not None:
            return Eligibility("held_back", reason=held.reason, attention_item_id=held.attention_item_id, **binding)
        return Eligibility("queued", **binding)

    def plan(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> list[PackagePlan]:
        paper = packages.load_paper_text(lib.conn, subject_id)
        assert paper is not None
        text = _package_text(paper.text)
        return [PackagePlan(1, 1, 0, min(HEAD_CHARS, len(paper.text)), text, max(1, len(text) // 4))]

    def instructions(self, lib: Library, *, has_parts: bool) -> Instructions:
        return Instructions(INSTRUCTIONS.format(version=TASK_VERSION), "{}\n")

    def check(self, lib: Library, package: WorkPackage, payload: dict[str, Any]) -> CheckResult:
        result = CheckResult()
        meta, errors = parse_meta(payload)
        result.errors = errors
        unknown = sorted(set(payload) - set(FIELDS))
        if unknown:
            result.warnings.append({"field_key": ",".join(unknown), "message": "unknown keys ignored"})
        result.claims_kept = sum(1 for v in meta.values() if v) if not errors else 0
        return result

    def save(self, conn: sqlite3.Connection, item: BatchItem, parts: Sequence[AcceptedPart],
             ctx: SaveContext) -> SaveResult:
        (part,) = parts
        meta, _ = parse_meta(part.payload)
        source = "ollama" if part.worker_kind == "ollama" else "agent"
        pid = item.subject_id
        title, authors, year, venue = conn.execute(
            "SELECT title, authors_json, year, venue FROM papers WHERE paper_id = ?", (pid,)).fetchone()
        filled: list[str] = []
        rejected: str | None = None
        proposed = clean_title(meta["title"])
        if not (title or "").strip() and proposed:
            reason = generic_title_reason(proposed, venue=meta["venue"] or venue)
            if reason is None:
                conn.execute("UPDATE papers SET title = ?, title_source = ? WHERE paper_id = ?",
                             (proposed, source, pid))
                filled.append("title")
            else:
                rejected = _reject_title(ctx.lib, pid, proposed, source, reason, part.model)
        if not authors and meta["authors"]:
            conn.execute("UPDATE papers SET authors_json = ? WHERE paper_id = ?",
                         (json.dumps(meta["authors"], ensure_ascii=False), pid))
            filled.append("authors")
        if year is None and meta["year"] is not None:
            conn.execute("UPDATE papers SET year = ? WHERE paper_id = ?", (meta["year"], pid))
            filled.append("year")
        if not (venue or "").strip() and meta["venue"]:
            conn.execute("UPDATE papers SET venue = ? WHERE paper_id = ?", (meta["venue"], pid))
            filled.append("venue")
        det = deterministic_identifiers(ctx.markdown_text)
        attached, collided = [], []
        raw = [(t, meta[t] or det[t]) for t in ("doi", "arxiv") if meta[t] or det[t]]
        for id_type, value in normalize_ids(raw):
            owner = conn.execute("SELECT paper_id FROM identifiers WHERE id_type = ? AND id_value = ?",
                                 (id_type, value)).fetchone()
            if owner is None:
                conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, ?, ?, ?)",
                             (pid, id_type, value, source))
                attached.append([id_type, value])
            elif owner[0] != pid:
                collided.append([id_type, value, owner[0]])
        has_title = conn.execute("SELECT COALESCE(TRIM(title), '') <> '' FROM papers WHERE paper_id = ?",
                                 (pid,)).fetchone()[0]
        resolved = 0
        if has_title:
            resolved = conn.execute(
                "UPDATE attention_items SET status = 'resolved', resolution = 'metadata_filled', resolved_at = ? "
                "WHERE paper_id = ? AND kind = ? AND status = 'open'", (_now(), pid, METADATA_NEEDED)).rowcount
        return SaveResult(pid, {"filled": filled, "ids_attached": attached, "ids_collided": collided,
                                "resolved_items": resolved, "generic_title_item": rejected})

    def after_commit(self, lib: Library, item: BatchItem, result: Any) -> Any:
        """Verify the paper's DOI with the providers once the save has committed (outside
        the submit transaction); a no-op when no chain is set for this context."""
        from ..library import metadata_verify

        chain = metadata_verify.current_chain()
        if chain is None or lib.conn.in_transaction:
            return None
        return metadata_verify.verify(lib, item.subject_id, chain=chain)


def _reject_title(lib: Library, paper_id: str, proposed: str, source: str, reason: str,
                  model: str | None) -> str:
    """Raise the ``metadata_generic_title`` item for a title that is not saved."""
    from ..library import attention

    who = source + (f" ({model})" if model else "")
    item = attention.raise_item(
        lib, GENERIC_TITLE, {"paper_id": paper_id, "title": proposed}, paper_id=paper_id,
        detail=f"{who} proposed the title '{proposed}', which was not saved: {reason}")
    return item.item_id


def _opt_str(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def parse_meta(payload: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The six fields, typed, plus type errors (an error never guesses a value)."""
    meta: dict[str, Any] = {k: None for k in FIELDS}
    errors: list[str] = []
    for key in ("title", "venue", "doi", "arxiv"):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"{key}: must be a string or null")
        meta[key] = _opt_str(value)
    authors = payload.get("authors")
    if authors is not None:
        if not isinstance(authors, list) or not all(isinstance(a, str) for a in authors):
            errors.append("authors: must be an array of strings or null")
        else:
            meta["authors"] = [a.strip() for a in authors if a.strip()] or None
    year = payload.get("year")
    if year is not None:
        if isinstance(year, bool):
            errors.append("year: must be an integer or null")
        elif isinstance(year, int):
            meta["year"] = year
        elif isinstance(year, str) and year.strip().isdigit():
            meta["year"] = int(year.strip())
        else:
            errors.append("year: must be an integer or null")
    if meta["year"] is not None and not 1000 <= meta["year"] <= 2999:
        errors.append("year: must be a four-digit year")
    return meta, errors
