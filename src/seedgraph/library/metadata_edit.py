"""The researcher's own metadata for a paper (docs/V3_UI_DESIGN.md §2.4 "Edit metadata", §12 q10).

    set_user_metadata(lib, paper_id, *, title, year, venue, authors) -> tuple[str, ...]

Writes the values in one library transaction with ``title_source = 'user'``. A ``'user'``
source is never looked up or overwritten by Ollama or a provider
(:mod:`library.metadata_verify` treats it as locked). Returns the changed field names.

Rules chosen where the design is silent:
- The title is required (a researcher edit never blanks it); year, venue and authors may
  be cleared. Whitespace is collapsed; empty author lines are dropped.
- ``title_source`` becomes ``'user'`` on any save, even when only the year changed: the
  source locks the whole record (there are no per-field sources).
- An open ``metadata_needed`` item for the paper is resolved (``edited_by_researcher``).
- After a title change on a held paper the standing PDF check re-runs
  (:func:`files.recheck_pdf`), outside the write transaction, as metadata_verify does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime, timezone

from .store import Library, LibraryError, is_held, paper_exists

USER_SOURCE = "user"
RESOLUTION = "edited_by_researcher"
METADATA_NEEDED = "metadata_needed"
YEAR_RANGE = (1000, 2200)

_WS = re.compile(r"\s+")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = _WS.sub(" ", value).strip()
    return value or None


def set_user_metadata(lib: Library, paper_id: str, *, title: str | None, year: int | str | None = None,
                      venue: str | None = None, authors: Sequence[str] = ()) -> tuple[str, ...]:
    """Save the researcher's metadata (see the module doc); the changed field names."""
    if not paper_exists(lib.conn, paper_id):
        raise LibraryError(f"unknown paper {paper_id!r}")
    title = _clean(title)
    if title is None:
        raise LibraryError("the title must not be empty")
    if isinstance(year, str):
        year = year.strip() or None
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            raise LibraryError(f"the year must be a number, not {year!r}") from None
        if not YEAR_RANGE[0] <= year <= YEAR_RANGE[1]:
            raise LibraryError(f"the year {year} is out of range")
    venue = _clean(venue)
    names = [a for a in (_clean(x) for x in authors) if a]
    authors_json = json.dumps(names, ensure_ascii=False) if names else None

    with lib.write() as conn:
        old = conn.execute("SELECT title, title_source, year, venue, authors_json FROM papers WHERE paper_id = ?",
                           (paper_id,)).fetchone()
        changed = tuple(name for name, before, after in (
            ("title", old["title"], title), ("year", old["year"], year), ("venue", old["venue"], venue),
            ("authors", old["authors_json"], authors_json)) if before != after)
        conn.execute("UPDATE papers SET title = ?, title_source = ?, year = ?, venue = ?, authors_json = ? "
                     "WHERE paper_id = ?", (title, USER_SOURCE, year, venue, authors_json, paper_id))
        conn.execute("UPDATE attention_items SET status = 'resolved', resolution = ?, resolved_at = ? "
                     "WHERE paper_id = ? AND kind = ? AND status = 'open'",
                     (RESOLUTION, datetime.now(timezone.utc).isoformat(), paper_id, METADATA_NEEDED))
    if "title" in changed and is_held(lib.conn, paper_id):
        from . import files

        files.recheck_pdf(lib, paper_id)
    return changed
