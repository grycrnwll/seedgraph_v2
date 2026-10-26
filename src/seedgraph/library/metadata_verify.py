"""Verify a paper's metadata by its DOI against the providers (docs/V3_UI_DESIGN.md §7).

    verify(lib, paper_id, *, chain, pages=None)              -> VerifyOutcome
    refetch(lib, paper_ids=None, *, malformed=False, chain)  -> list[VerifyOutcome]
    malformed_papers(lib)                                    -> list[str]
    set_chain(chain) / current_chain()                       the post-commit hook's chain
    default_chain(lib)                                       the real provider chain

:func:`verify` looks the paper's DOI up with ``chain.by_doi`` (OpenAlex, then Crossref,
then Semantic Scholar in the real :class:`~seedgraph.providers.base.ProviderChain`) and
compares the provider title with the stored one through the tolerant matcher
(:func:`~seedgraph.library.titlecheck.title_similarity`):

- provider title ~ stored title -> ``verified``: the provider's title, year, venue and
  authors replace the stored ones, ``title_source`` = the provider;
- they differ but the provider title is in the paper's cover region (the first
  ``metadata_quality.COVER_CHARS`` characters of its first pages; never the whole first
  pages, which in a short paper already hold the reference list and so the cited work's
  title) -> ``provider_wins`` (the stored title was misread), same update;
- they differ and the provider title is not on the cover -> the DOI likely came from the
  reference list: a machine-found DOI row (source ``ollama`` / ``agent`` / ``pdf_scan``)
  is removed (``doi_removed``), any other is kept (``doi_mismatch``), the stored title
  stays, and a ``metadata_doi_mismatch`` attention item is raised.

The network call runs with no write transaction open. A paper whose ``title_source`` is
``'user'`` is never looked up or changed (``user_locked``): researcher values are never
overwritten.

Rules chosen where the design is silent:
- ``ProviderChain.by_doi`` returns a bare record, so the answering provider is inferred:
  a record's own ``provider`` key, else ``openalex`` when it has an ``openalex_id``, else
  ``semantic_scholar`` when it has an ``arxiv_id`` / ``s2_id``, else ``crossref``.
- ``papers`` has no ``metadata_json`` column, so the evidence (DOI, matcher score,
  provider) is returned on the outcome and written into attention item details.
- A ``'user'`` title source locks title, year, venue and authors alike (there are no
  per-field sources).
- A generic provider title (``metadata_quality``) is never taken (``not_found``).
- The paper's first DOI (by value) is verified; the first pages come from its current
  markdown, else from its PDF's text layer.
- After a title change on a held paper :func:`files.recheck_pdf` re-runs the PDF check,
  and an open ``metadata_needed`` item is resolved.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Literal, Protocol

from . import attention
from .metadata_quality import clean_title, cover_region, generic_title_reason
from .store import Library, LibraryError
from .titlecheck import (PASS_THRESHOLD, check_pdf_text, first_pages_from_markdown, first_pages_from_pdf,
                         title_similarity)

__all__ = ["LibraryError", "VerifyOutcome", "verify", "refetch", "malformed_papers", "set_chain",
           "current_chain", "default_chain", "provider_of", "run_sync"]

MISMATCH_KIND = "metadata_doi_mismatch"
GENERIC_KIND = "metadata_generic_title"
METADATA_NEEDED = "metadata_needed"
#: DOI sources a machine found (removable when the DOI turns out to be a cited work's).
MACHINE_DOI_SOURCES = frozenset({"ollama", "agent", "pdf_scan"})
#: Title-vs-title similarity at or above which two titles are the same title.
SAME_TITLE = PASS_THRESHOLD

Status = Literal["verified", "provider_wins", "doi_removed", "doi_mismatch", "unconfirmed", "no_doi",
                 "not_found", "lookup_failed", "user_locked"]


class DoiLookup(Protocol):
    async def by_doi(self, doi: str) -> dict | None: ...


@dataclass(frozen=True)
class VerifyOutcome:
    paper_id: str
    status: Status
    doi: str | None = None
    provider: str | None = None
    score: float | None = None  # title similarity, stored vs provider
    changed: tuple[str, ...] = ()
    attention_item_id: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "status": self.status, "doi": self.doi, "provider": self.provider,
                "score": self.score, "changed": list(self.changed),
                "attention_item_id": self.attention_item_id, "detail": self.detail}


# --- the chain the post-commit hook uses ------------------------------------------------------

_CHAIN: contextvars.ContextVar[Any] = contextvars.ContextVar("seedgraph_metadata_chain", default=None)


def set_chain(chain: Any) -> contextvars.Token:
    """Set the provider chain post-commit verification uses in this context (thread);
    ``None`` (the default) skips verification. Returns a token for ``_CHAIN.reset``."""
    return _CHAIN.set(chain)


def reset_chain(token: contextvars.Token) -> None:
    _CHAIN.reset(token)


def current_chain() -> Any:
    return _CHAIN.get()


def default_chain(lib: Library) -> Any:
    """The real provider chain (OpenAlex, Crossref, Unpaywall, Semantic Scholar, CORE when
    keyed; keys from environment variables) cached in the library's ``provider_cache.db``.
    The caller closes ``chain.cache_engine`` (a connection) when done."""
    from ..config.loader import load_global_config
    from ..providers.base import ProviderChain
    from .store import open_provider_cache

    return ProviderChain.from_config(load_global_config(lib.root.parent),
                                     cache_engine=open_provider_cache(lib.root))


# --- helpers -----------------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_LOOPS = threading.local()


def _thread_loop() -> asyncio.AbstractEventLoop:
    """This thread's long-lived event loop. Never closed: the real providers keep an
    ``httpx.AsyncClient`` bound to the loop of their first call, so every later lookup
    from this thread must run on that same loop."""
    loop = getattr(_LOOPS, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        _LOOPS.loop = loop
    return loop


def run_sync(value: Any) -> Any:
    """The result of ``value``: awaited when it is awaitable (on this thread's long-lived
    event loop; on a helper thread when this thread already runs a loop), else ``value``."""
    if not inspect.isawaitable(value):
        return value

    async def _await() -> Any:
        return await value

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _thread_loop().run_until_complete(_await())
    box: dict[str, Any] = {}

    def _run() -> None:
        try:
            box["value"] = asyncio.run(_await())
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            box["error"] = exc

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def provider_of(record: dict) -> str:
    """Which provider answered (see the module doc)."""
    if record.get("provider"):
        return str(record["provider"])
    if record.get("openalex_id"):
        return "openalex"
    if record.get("arxiv_id") or record.get("s2_id"):
        return "semantic_scholar"
    return "crossref"


def lookup_doi(chain: Any, doi: str) -> dict | None:
    """``chain.by_doi(doi)`` run to completion (no transaction may be open)."""
    return run_sync(chain.by_doi(doi))


def paper_pages(lib: Library, paper_id: str) -> list[str]:
    """The first pages of the paper's primary file: its markdown, else its PDF text layer."""
    row = lib.conn.execute(
        "SELECT s.storage_uri AS pdf_uri, m.storage_uri AS md_uri FROM paper_files f "
        "JOIN source_files s ON s.source_file_id = f.source_file_id "
        "LEFT JOIN markdown_documents m ON m.markdown_id = f.markdown_id "
        "WHERE f.paper_id = ? AND f.role = 'primary'", (paper_id,)).fetchone()
    if row is None:
        return []
    if row["md_uri"]:
        path = lib.root / row["md_uri"]
        if path.is_file():
            pages = first_pages_from_markdown(path.read_text(encoding="utf-8"))
            if any(p.strip() for p in pages):
                return pages
    if row["pdf_uri"]:
        return first_pages_from_pdf(lib.root / row["pdf_uri"]) or []
    return []


def record_values(record: dict) -> dict[str, Any]:
    """Title (cleaned), year, venue and authors from a provider record (``None`` when absent)."""
    title = record.get("title")
    if isinstance(title, list):
        title = title[0] if title else None
    year = record.get("year")
    authors = record.get("authors")
    return {
        "title": clean_title(title) if isinstance(title, str) else None,
        "year": year if isinstance(year, int) and not isinstance(year, bool) else None,
        "venue": clean_title(record["venue"]) if isinstance(record.get("venue"), str) else None,
        "authors": [a for a in authors if isinstance(a, str) and a.strip()] if isinstance(authors, list) else None,
    }


def apply_provider_values(conn, paper_id: str, values: dict[str, Any], provider: str) -> tuple[str, ...]:
    """Write the provider's non-null values onto the paper; the changed field names."""
    old = conn.execute("SELECT title, title_source, year, venue, authors_json FROM papers WHERE paper_id = ?",
                       (paper_id,)).fetchone()
    changed: list[str] = []
    if values["title"] and (values["title"] != old["title"] or old["title_source"] != provider):
        conn.execute("UPDATE papers SET title = ?, title_source = ? WHERE paper_id = ?",
                     (values["title"], provider, paper_id))
        changed.append("title")
    if values["year"] is not None and values["year"] != old["year"]:
        conn.execute("UPDATE papers SET year = ? WHERE paper_id = ?", (values["year"], paper_id))
        changed.append("year")
    if values["venue"] and values["venue"] != old["venue"]:
        conn.execute("UPDATE papers SET venue = ? WHERE paper_id = ?", (values["venue"], paper_id))
        changed.append("venue")
    if values["authors"]:
        authors_json = json.dumps(values["authors"], ensure_ascii=False)
        if authors_json != old["authors_json"]:
            conn.execute("UPDATE papers SET authors_json = ? WHERE paper_id = ?", (authors_json, paper_id))
            changed.append("authors")
    return tuple(changed)


def resolve_metadata_needed(conn, paper_id: str, resolution: str) -> int:
    return conn.execute(
        "UPDATE attention_items SET status = 'resolved', resolution = ?, resolved_at = ? "
        "WHERE paper_id = ? AND kind = ? AND status = 'open'",
        (resolution, _now(), paper_id, METADATA_NEEDED)).rowcount


def _paper(lib: Library, paper_id: str):
    row = lib.conn.execute("SELECT title, title_source, venue FROM papers WHERE paper_id = ?",
                           (paper_id,)).fetchone()
    if row is None:
        raise LibraryError(f"unknown paper {paper_id!r}")
    return row


def _is_held(lib: Library, paper_id: str) -> bool:
    return lib.conn.execute("SELECT 1 FROM paper_files WHERE paper_id = ? LIMIT 1", (paper_id,)).fetchone() is not None


# --- verify ------------------------------------------------------------------------------------


def verify(lib: Library, paper_id: str, *, chain: Any, pages: Sequence[str] | None = None) -> VerifyOutcome:
    """Verify the paper's DOI and prefer the provider's values (see the module doc).

    Must not be called inside an open library transaction (it waits on the network).
    """
    paper = _paper(lib, paper_id)
    if paper["title_source"] == "user":
        return VerifyOutcome(paper_id, "user_locked", detail="the researcher set this paper's metadata")
    doi_row = lib.conn.execute(
        "SELECT id_value, source FROM identifiers WHERE paper_id = ? AND id_type = 'doi' ORDER BY id_value LIMIT 1",
        (paper_id,)).fetchone()
    if doi_row is None:
        return VerifyOutcome(paper_id, "no_doi", detail="the paper has no DOI to verify")
    doi, doi_source = doi_row[0], doi_row[1]
    try:
        record = lookup_doi(chain, doi)
    except Exception as exc:  # noqa: BLE001 - a lookup failure changes nothing
        return VerifyOutcome(paper_id, "lookup_failed", doi=doi, detail=f"provider lookup failed: {exc}")
    if not record:
        return VerifyOutcome(paper_id, "not_found", doi=doi, detail="no provider knows this DOI")
    provider = provider_of(record)
    values = record_values(record)
    if not values["title"] or generic_title_reason(values["title"]) is not None:
        return VerifyOutcome(paper_id, "not_found", doi=doi, provider=provider,
                             detail=f"{provider} gave no usable title for this DOI")

    stored = paper["title"]
    score = title_similarity(stored, values["title"]) if stored else 0.0
    if stored and score >= SAME_TITLE:
        status: Status = "verified"
    else:
        cover = cover_region(list(pages) if pages is not None else paper_pages(lib, paper_id))
        found = check_pdf_text(values["title"], None, cover)
        if found.status == "pass":
            status = "provider_wins"
        elif found.status == "unknown" and not stored:
            return VerifyOutcome(paper_id, "unconfirmed", doi=doi, provider=provider, score=score,
                                 detail="the paper has no title and no text to confirm the provider's title")
        else:
            return _mismatch(lib, paper_id, doi, doi_source, provider, values["title"], stored, score)

    with lib.write() as conn:
        changed = apply_provider_values(conn, paper_id, values, provider)
        resolve_metadata_needed(conn, paper_id, "metadata_verified")
    if "title" in changed and _is_held(lib, paper_id):
        from . import files  # local: files pulls in the cache and converter layers

        files.recheck_pdf(lib, paper_id)
    how = "matches" if status == "verified" else "is in the PDF (the stored title was misread)"
    return VerifyOutcome(paper_id, status, doi=doi, provider=provider, score=score, changed=changed,
                         detail=f"{provider} title for {doi} {how}; score {score:.2f}")


def _mismatch(lib: Library, paper_id: str, doi: str, doi_source: str | None, provider: str,
              provider_title: str, stored: str | None, score: float) -> VerifyOutcome:
    removable = doi_source in MACHINE_DOI_SOURCES
    detail = (f"DOI {doi} belongs to '{provider_title}' ({provider}), which is not this paper's title "
              f"('{stored or 'unknown'}', score {score:.2f}) and is not on its cover page; "
              + ("it was probably read from the reference list and was removed" if removable
                 else "it was kept because it did not come from a machine scan"))
    with lib.write() as conn:
        if removable:
            conn.execute("DELETE FROM identifiers WHERE paper_id = ? AND id_type = 'doi' AND id_value = ?",
                         (paper_id, doi))
        item = attention.raise_item(lib, MISMATCH_KIND, {"paper_id": paper_id, "doi": doi}, paper_id=paper_id,
                                    detail=detail)
    return VerifyOutcome(paper_id, "doi_removed" if removable else "doi_mismatch", doi=doi, provider=provider,
                         score=score, attention_item_id=item.item_id, detail=detail)


# --- refetch -----------------------------------------------------------------------------------


def malformed_papers(lib: Library) -> list[str]:
    """Papers whose title is missing, carries footnote marks or is generic (never a
    researcher-set one), by ``paper_id``."""
    out: list[str] = []
    for pid, title, source, venue in lib.conn.execute(
            "SELECT paper_id, title, title_source, venue FROM papers ORDER BY paper_id"):
        if source == "user":
            continue
        if not (title or "").strip() or clean_title(title) != title or generic_title_reason(title, venue=venue):
            out.append(pid)
    return out


def refetch(lib: Library, paper_ids: Sequence[str] | None = None, *, malformed: bool = False,
            chain: Any) -> list[VerifyOutcome]:
    """``library refetch-metadata``: clean footnote marks locally, then verify each paper's
    DOI with the providers. Papers: ``paper_ids`` plus, with ``malformed``, every
    :func:`malformed_papers` one. A title still generic afterwards raises a
    ``metadata_generic_title`` item."""
    chosen = list(dict.fromkeys([*(paper_ids or ()), *(malformed_papers(lib) if malformed else ())]))
    if not paper_ids and not malformed:
        raise LibraryError("refetch-metadata: name papers or pass --malformed")
    outcomes: list[VerifyOutcome] = []
    for pid in chosen:
        paper = _paper(lib, pid)
        cleaned: tuple[str, ...] = ()
        if paper["title_source"] != "user" and paper["title"] and clean_title(paper["title"]) != paper["title"]:
            with lib.write() as conn:
                conn.execute("UPDATE papers SET title = ? WHERE paper_id = ?", (clean_title(paper["title"]), pid))
            cleaned = ("title",)
        out = verify(lib, pid, chain=chain)
        if cleaned and "title" not in out.changed:
            out = replace(out, changed=cleaned + out.changed)
        after = _paper(lib, pid)
        if after["title_source"] != "user" and after["title"]:
            reason = generic_title_reason(after["title"], venue=after["venue"])
            if reason is not None:
                attention.raise_item(lib, GENERIC_KIND, {"paper_id": pid, "title": after["title"]}, paper_id=pid,
                                     detail=f"{reason}; set the title on the paper page")
        outcomes.append(out)
    return outcomes

