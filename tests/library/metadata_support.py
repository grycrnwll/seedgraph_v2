"""Shared fakes for the ingest and metadata tests: a provider chain answering from a dict,
papers with a synthetic PDF, and attention-item reads. All text is invented."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from seedgraph.library import files

from .test_files import text_pdf


class FakeChain:
    """``by_doi`` from a dict; records shaped as the real providers shape them (an OpenAlex
    record carries ``openalex_id``, a Crossref record does not). Counts calls."""

    def __init__(self, records: dict[str, dict] | None = None, *, error: Exception | None = None):
        self.records = {k.lower(): v for k, v in (records or {}).items()}
        self.error = error
        self.calls: list[str] = []

    async def by_doi(self, doi: str):
        self.calls.append(doi)
        if self.error is not None:
            raise self.error
        rec = self.records.get(doi.lower())
        return dict(rec) if rec is not None else None


def openalex_record(doi: str, title: str, year: int, authors=("Ada Inventor",), **extra) -> dict:
    return {"openalex_id": "W1234567", "doi": doi, "title": title, "year": year, "authors": list(authors), **extra}


def crossref_record(doi: str, title: str, year: int, authors=("Ada Inventor",), **extra) -> dict:
    return {"doi": doi, "title": title, "year": year, "authors": list(authors), **extra}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_paper(lib, paper_id: str, *, title: str | None = None, title_source: str | None = None,
                 year: int | None = None, venue: str | None = None, authors: list[str] | None = None,
                 dois: tuple[tuple[str, str | None], ...] = ()) -> str:
    """A paper row with its DOIs as ``(doi, source)`` pairs."""
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO papers (paper_id, title, title_source, year, venue, authors_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (paper_id, title, title_source, year, venue, json.dumps(authors) if authors else None, now()))
        for doi, source in dois:
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, 'doi', ?, ?)",
                         (paper_id, doi, source))
    return paper_id


def held_paper(lib, tmp_path: Path, paper_id: str, pages: list[str], **kw) -> str:
    """A paper holding a synthetic PDF whose text layer is ``pages``."""
    insert_paper(lib, paper_id, **kw)
    files.attach_pdf(lib, paper_id, text_pdf(tmp_path / f"{paper_id}.pdf", pages, marker=paper_id))
    return paper_id


def paper_row(lib, paper_id: str):
    return lib.conn.execute(
        "SELECT title, title_source, year, venue, authors_json FROM papers WHERE paper_id = ?",
        (paper_id,)).fetchone()


def dois_of(lib, paper_id: str) -> list[tuple[str, str | None]]:
    return [tuple(r) for r in lib.conn.execute(
        "SELECT id_value, source FROM identifiers WHERE paper_id = ? AND id_type = 'doi' ORDER BY id_value",
        (paper_id,))]


def items(lib, kind: str, status: str = "open"):
    return lib.conn.execute(
        "SELECT * FROM attention_items WHERE kind = ? AND status = ? ORDER BY item_id", (kind, status)).fetchall()
