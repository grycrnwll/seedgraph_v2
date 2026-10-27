""""This isn't the paper's PDF" (paper page and Review): the chooser (cover-title suggestions,
a title search, a new paper) and the preview of :func:`seedgraph.library.reassign.plan_reassign_pdf`.
Pure reads; the confirm route calls :func:`seedgraph.library.reassign.reassign_pdf`.

``next`` (a local path) carries the page to return to after the move (Review); without it
the researcher lands on the paper that now holds the PDF.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

from ...library import reassign
from ...library.reassign import ReassignPlan, Suggestion
from ...library.store import Library, is_held, paper_exists


@dataclass(frozen=True)
class PaperBrief:
    paper_id: str
    title: str | None
    held: bool

    @property
    def display_title(self) -> str:
        return self.title or "(untitled paper)"


@dataclass(frozen=True)
class Choice:
    suggestion: Suggestion
    preview_url: str


@dataclass(frozen=True)
class Chooser:
    paper: PaperBrief
    source_file_id: str
    suggestions: tuple[Choice, ...]
    query: str
    hits: tuple[Choice, ...]
    new_paper_url: str
    next: str


@dataclass(frozen=True)
class Preview:
    paper: PaperBrief
    target: PaperBrief | None
    plan: ReassignPlan
    next: str
    chooser_url: str


def local_next(value: object) -> str:
    text = str(value or "")
    return text if text.startswith("/") and not text.startswith("//") else ""


def _brief(lib: Library, paper_id: str) -> PaperBrief:
    row = lib.conn.execute("SELECT title FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    return PaperBrief(paper_id, row[0] if row else None, is_held(lib.conn, paper_id))


def _url(paper_id: str, path: str, query: dict[str, str]) -> str:
    q = {k: v for k, v in query.items() if v}
    return f"/papers/{paper_id}/reassign-pdf{path}" + (f"?{urlencode(q)}" if q else "")


def chooser_url(paper_id: str, source_file_id: str | None = None, next_: str = "") -> str:
    return _url(paper_id, "", {"file": source_file_id or "", "next": next_})


def chooser(lib: Library, paper_id: str, source_file_id: str | None = None, *, query: str = "",
            next_: str = "") -> Chooser | None:
    """The chooser, or ``None`` when the paper is unknown. Raises
    :class:`seedgraph.library.reassign.ReassignError` when the paper has no such PDF."""
    if not paper_exists(lib.conn, paper_id):
        return None
    next_ = local_next(next_)
    found = reassign.suggest_targets(lib, paper_id, source_file_id)
    sf = source_file_id or lib.conn.execute(
        "SELECT source_file_id FROM paper_files WHERE paper_id = ? ORDER BY source_file_id LIMIT 1",
        (paper_id,)).fetchone()[0]

    def choice(s: Suggestion) -> Choice:
        return Choice(s, _url(paper_id, "/preview", {"file": sf, "to": s.paper_id, "next": next_}))

    hits = reassign.find_papers(lib, query, exclude=(paper_id,)) if query.strip() else ()
    return Chooser(paper=_brief(lib, paper_id), source_file_id=sf, suggestions=tuple(choice(s) for s in found),
                   query=query, hits=tuple(choice(s) for s in hits),
                   new_paper_url=_url(paper_id, "/preview", {"file": sf, "new": "1", "next": next_}), next=next_)


def preview(lib: Library, paper_id: str, source_file_id: str | None, *, to: str | None, new_paper: bool,
            next_: str = "") -> Preview | None:
    """The plan for one choice, or ``None`` when the paper is unknown. Raises
    :class:`seedgraph.library.reassign.ReassignError` for a request that cannot be planned."""
    if not paper_exists(lib.conn, paper_id):
        return None
    plan = reassign.plan_reassign_pdf(lib, paper_id, source_file_id, target_paper_id=to or None,
                                      new_paper=new_paper)
    next_ = local_next(next_)
    return Preview(paper=_brief(lib, paper_id), target=_brief(lib, to) if to else None, plan=plan, next=next_,
                   chooser_url=chooser_url(paper_id, plan.source_file_id, next_))
