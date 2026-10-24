"""Deterministic post-conversion bibliographic metadata backfill (v2 projects).

The converted markdown is the authoritative identity of a paper; the PDF filename
is just the user's acquisition convention (``2023_santanna.pdf`` -> the junk title
``2023 santanna``). This module reads a converted work's markdown and backfills the
strong-id scalar columns:

* :func:`deterministic_identifiers` scans ONLY the cover/head region (the first
  ~2600 chars) with regexes for a DOI / arXiv id — this catches the paper's OWN DOI
  (incl. a JSTOR stable-URL ``10.2307/...`` cover DOI) while NEVER picking up a CITED
  work's DOI/arXiv id from a later reference list.
* :func:`backfill_work_metadata` is the idempotent runner: resolve the work's
  markdown via the bridge, scan, and apply ``identity.update_work_bibliography``.

The LLM path (through the hosted ``llm.executor``) was removed in stage 2: metadata
from the model is now the work protocol's ``metadata`` task (``work/tasks_metadata.py``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .. import cache_access
from ..acquisition.bridge import resolve_work_markdown
from ..project import identity
from ..project.identity import normalize_id

if TYPE_CHECKING:
    from pathlib import Path

    from ..project.service import ProjectHandle

__all__ = [
    "BiblioMeta",
    "BackfillResult",
    "deterministic_identifiers",
    "backfill_work_metadata",
]

# Only the first ~2600 chars (title block / author line / abstract head) carry the
# paper's own metadata, not publisher/JSTOR cover boilerplate or references.
HEAD_CHARS = 2600


@dataclass
class BiblioMeta:
    """Typed bibliographic metadata for one paper (the extractor's output)."""

    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None


@dataclass
class BackfillResult:
    """Per-work outcome of :func:`backfill_work_metadata` (the CLI summary line)."""

    work_id: str
    status: str
    old_title: Optional[str] = None
    new_title: Optional[str] = None
    title_changed: bool = False
    ids_attached: list[tuple[str, str]] = field(default_factory=list)
    ids_collided: list[tuple[str, str]] = field(default_factory=list)
    fields_filled: list[str] = field(default_factory=list)
    message: Optional[str] = None


# --- deterministic identifiers (regex over the cover/HEAD region only) -------

# Crossref DOI shape (catches the JSTOR stable DOI 10.2307/<int>). Trailing
# sentence punctuation is stripped after the match.
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
# An arXiv id is trusted ONLY when EXPLICITLY labeled: an ``arxiv.org/abs/<id>``
# URL or an ``arXiv:<id>`` prefix. A bare ``NNNN.NNNNN`` numeric token is NOT an
# arXiv id (it collides with coefficients / table values / cited-paper ids), so no
# bare-number pattern exists — dropping it is deliberate (must-fix #2). The URL is
# tried first so ``arXiv: https://arxiv.org/abs/<id>`` yields the id, not "https".
_ARXIV_URL_RE = re.compile(r"arxiv\.org/abs/([A-Za-z0-9][A-Za-z0-9.\-/]*)", re.IGNORECASE)
_ARXIV_LABELED_RE = re.compile(r"arxiv:\s*([A-Za-z0-9][A-Za-z0-9.\-/]*)", re.IGNORECASE)


def _clean_tail(text: str) -> str:
    return text.rstrip(".,;)]}>\"'")


def deterministic_identifiers(markdown: str) -> dict[str, Optional[str]]:
    """Regex-scan the paper's cover/HEAD region for its OWN DOI / arXiv id (no LLM).

    Returns ``{"doi": <normalized doi | None>, "arxiv": <normalized arxiv | None>}``.

    The scan is bounded to the first :data:`HEAD_CHARS` characters — the same
    cover/title-block region the LLM sees. This is deliberate identity hygiene: a
    paper's OWN DOI (including a JSTOR ``Stable URL`` cover DOI) lives on the cover,
    whereas the DOIs / arXiv ids in a later reference list belong to CITED works and
    must NEVER be attached to the citing paper (must-fix #1). arXiv ids are accepted
    ONLY when explicitly labeled (``arXiv:<id>`` or an ``arxiv.org/abs/<id>`` URL);
    a bare ``NNNN.NNNNN`` numeric token is never treated as an arXiv id (must-fix #2).
    """
    text = (markdown or "")[:HEAD_CHARS]
    doi = None
    m = _DOI_RE.search(text)
    if m is not None:
        doi = normalize_id("doi", _clean_tail(m.group(0)))

    arxiv = None
    um = _ARXIV_URL_RE.search(text)
    if um is not None:
        arxiv = normalize_id("arxiv", _clean_tail(um.group(1)))
    if not arxiv:
        lm = _ARXIV_LABELED_RE.search(text)
        if lm is not None:
            arxiv = normalize_id("arxiv", _clean_tail(lm.group(1)))

    return {"doi": doi, "arxiv": arxiv}


# --- runner -----------------------------------------------------------------

def backfill_work_metadata(
    h: "ProjectHandle",
    work_id: str,
    *,
    cache_root: "Path | str | None" = None,
    force: bool = False,
    cache_conn=None,
) -> BackfillResult:
    """Backfill one work's DOI / arXiv id from its converted markdown (no LLM).

    Idempotent: a work whose current title is NOT filename-derived is skipped unless
    ``force``. Resolves the work's markdown via the bridge, scans the head region with
    :func:`deterministic_identifiers`, and applies ``identity.update_work_bibliography``
    (which never creates a work and routes id collisions to review).
    """
    from sqlmodel import Session

    from ..db.project_models import Work

    # 1. Resolve work -> markdown; short-circuit idempotently on a good title.
    with Session(h.engine, expire_on_commit=False) as session:
        resolved = resolve_work_markdown(session, work_id=work_id)
        work = session.get(Work, work_id)
        if work is None:
            return BackfillResult(work_id=work_id, status="skipped_no_work",
                                  message=f"no such work {work_id}")
        old_title = work.canonical_title
        if resolved is None:
            return BackfillResult(work_id=work_id, status="skipped_no_markdown",
                                  old_title=old_title, new_title=old_title,
                                  message=f"work {work_id} has no resolvable markdown")
        if not force and not identity.is_filename_derived_title(session, work):
            return BackfillResult(work_id=work_id, status="skipped_idempotent",
                                  old_title=old_title, new_title=old_title,
                                  message="title already authoritative (use --force)")
    markdown_id, _markdown_hash = resolved

    # 2. Read the markdown text.
    own_cache = cache_conn is None
    if own_cache:
        cache_conn = cache_access.open_cache_ro(cache_root)
    try:
        md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
    finally:
        if own_cache:
            cache_conn.close()
    if md is None:
        return BackfillResult(work_id=work_id, status="skipped_no_markdown",
                              old_title=old_title, new_title=old_title,
                              message=f"markdown {markdown_id} unresolvable in cache")

    # 3. Deterministic ids over the head region only, so a cited work's
    #    reference-list DOI/arXiv is never attached.
    det = deterministic_identifiers(md.text[:HEAD_CHARS])
    meta = BiblioMeta(doi=det.get("doi"), arxiv_id=det.get("arxiv"))

    # 4. Apply to the work (never creates one; id collisions -> review).
    with Session(h.engine, expire_on_commit=False) as session:
        upd = identity.update_work_bibliography(session, work_id, meta, allow_title_overwrite=force)
        session.commit()

    return BackfillResult(
        work_id=work_id,
        status="ok",
        old_title=upd.old_title,
        new_title=upd.new_title,
        title_changed=upd.title_changed,
        ids_attached=upd.ids_attached,
        ids_collided=upd.ids_collided,
        fields_filled=upd.fields_filled,
    )
