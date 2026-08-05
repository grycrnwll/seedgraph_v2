"""Never-blank display labels (Build B chunk 9; ported from v1 ``render.py``).

One shared, DISPLAY-ONLY pure helper: :func:`derive_label` formats a human
label for a work from whatever metadata survives. It never fabricates or
mutates ``canonical_title`` (or any stored field) — placeholders are
deliberately non-title-shaped so neither a human nor the semantic layer can
mistake them for a real title.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["derive_label", "extract_surname"]


def extract_surname(name: str, *, whole_comma_head: bool = False) -> str:
    """Comma-head/last-token surname extraction shared by display labels and
    BibTeX cite keys (``semantic/export._cite_key``).

    Default (display mode): take the comma-head's last whitespace token so
    "Callaway, Brantly" and "Brantly Callaway" both yield "Callaway"; a
    token-less comma-head falls back to ``name`` verbatim.

    ``whole_comma_head=True`` (cite-key mode): when a comma is present the
    ENTIRE comma-head is kept ("Van Der Berg, Gerard" -> "Van Der Berg"),
    preserving the historical ``_cite_key`` behavior byte-for-byte.
    """
    head = name.split(",")[0].strip()
    if whole_comma_head and "," in name:
        return head
    return head.split()[-1] if head.split() else name


def derive_label(
    *,
    title: Optional[str] = None,
    authors: Optional[list] = None,
    year: Optional[int] = None,
    doi: Optional[str] = None,
    openalex_id: Optional[str] = None,
    arxiv_id: Optional[str] = None,
    work_id: Optional[str] = None,
) -> str:
    """Derive a NEVER-BLANK human display label for a work node/row.

    Priority chain (v1 ``render.py::derive_label``):

    1. a non-empty ``title`` → returned verbatim (a titled node gets
       ``label == title``; good titles are never regressed);
    2. otherwise ``"Surname[ et al.]( Year)"`` when an authors list is present;
    3. otherwise a VISIBLY-PLACEHOLDER identifier — ``"doi:…"`` /
       ``"OpenAlex W…"`` / ``"arXiv:…"`` — deliberately non-title-shaped;
    4. otherwise ``"untitled <id8>"`` (the pure-UUID residue) — and finally
       ``"untitled"`` if not even a ``work_id`` is available.
    """

    # (1) A real title wins outright — label == title for every titled node.
    if title is not None:
        t = str(title).strip()
        if t:
            return t

    # (2) "Surname[ et al.]( Year)" when authors are present (shared
    # comma-head/last-token extraction; see extract_surname).
    if authors:
        first = str(authors[0]).strip()
        if first:
            lead = extract_surname(first)
            et_al = " et al." if len(authors) > 1 else ""
            suffix = f" ({year})" if year else ""
            return f"{lead}{et_al}{suffix}"

    # (3) A visibly-placeholder identifier — never title-shaped.
    if doi:
        return f"doi:{str(doi).strip()}"
    if openalex_id:
        oa = str(openalex_id).strip()
        # OpenAlex work ids are bare "W…"; present them with the provider prefix.
        return f"OpenAlex {oa}" if oa else "untitled"
    if arxiv_id:
        return f"arXiv:{str(arxiv_id).strip()}"

    # (4) Pure-UUID residue: a short, collision-distinct id prefix.
    if work_id:
        wid = str(work_id).strip()
        if wid:
            return f"untitled {wid[:8]}"
    return "untitled"
