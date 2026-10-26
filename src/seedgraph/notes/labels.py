"""The ``labels.json`` input of a notes batch (design §4.2).

Per label group, the library's top ``normalized_label``s, ranked by the number of
distinct papers whose current note (agent audience) uses them as a ``found`` claim.
Agent audience = ``note_current`` (unlabelled versions only) minus papers with an open
``hides_notes`` attention item. Case, whitespace, hyphen and trailing-punctuation
variants of one label are listed once (:func:`label_groups`). Ties break by label, so
the output is byte-stable.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict

from ..library.attention import HIDDEN_PAPERS_SQL
from .format import SCHEMA_ID

#: Label group -> the claim types it covers (order = order in labels.json).
LABEL_GROUPS: dict[str, tuple[str, ...]] = {
    "methods_and_models": ("method", "model"),
    "identification_assumptions": ("identification_assumption",),
    "regularity_conditions": ("regularity_condition",),
    "general_assumptions": ("general_assumption",),
    "estimands": ("estimand",),
    "data_sources": ("data",),
    "results": ("result",),
    "limitations": ("limitation",),
    "robustness_checks": ("robustness_check",),
}

DEFAULT_LIMIT = 100


_SPACE_RE = re.compile(r"[\s\-‐‑‒–—]+")
_TRAILING_RE = re.compile(r"[\s.,;:!?]+$")


def label_key(label: str) -> str:
    """The grouping key of a label: casefolded, trailing punctuation stripped, and
    whitespace and hyphen/dash runs read as one space ("Difference-in-Differences."
    and "difference in differences" are one label)."""
    return _SPACE_RE.sub(" ", _TRAILING_RE.sub("", label.casefold())).strip()


def label_groups(conn: sqlite3.Connection, *, limit: int = DEFAULT_LIMIT,
                 schema_id: str = SCHEMA_ID) -> dict[str, list[str]]:
    """``{group: [label, ...]}`` for every group in :data:`LABEL_GROUPS` (empty lists kept).

    Spelling variants (see :func:`label_key`) are listed once, under the spelling used by
    the most papers (ties: the first in code-point order). A variant group ranks by the
    distinct papers using any of its spellings (a paper using two spellings counts once).
    """
    out: dict[str, list[str]] = {}
    for group, claim_types in LABEL_GROUPS.items():
        marks = ", ".join("?" for _ in claim_types)
        rows = conn.execute(
            "SELECT DISTINCT c.normalized_label, c.paper_id "
            "FROM extracted_claims c JOIN note_current n ON n.note_id = c.structured_note_id "
            f"WHERE n.schema_id IS ? AND c.claim_type IN ({marks}) AND c.status = 'found' "
            "AND c.normalized_label IS NOT NULL AND TRIM(c.normalized_label) <> '' "
            f"AND c.paper_id NOT IN ({HIDDEN_PAPERS_SQL})",
            (schema_id, *claim_types),
        ).fetchall()
        papers: dict[str, set[str]] = defaultdict(set)  # key -> papers
        spellings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for label, paper_id in rows:
            key = label_key(label) or label
            papers[key].add(paper_id)
            spellings[key][label].add(paper_id)
        ranked = []
        for key, members in papers.items():
            spelling = min(spellings[key].items(), key=lambda kv: (-len(kv[1]), kv[0]))[0]
            ranked.append((-len(members), spelling))
        out[group] = [spelling for _, spelling in sorted(ranked)[:limit]]
    return out


def labels_json(conn: sqlite3.Connection, *, limit: int = DEFAULT_LIMIT) -> str:
    """The frozen ``labels.json`` text for a new batch (stable bytes for equal data)."""
    return json.dumps(label_groups(conn, limit=limit), ensure_ascii=False, indent=1) + "\n"
