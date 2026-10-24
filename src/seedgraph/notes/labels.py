"""The ``labels.json`` input of a notes batch (design §4.2).

Per label group, the library's top ``normalized_label``s, ranked by the number of
distinct papers whose current note (agent audience) uses them as a ``found`` claim.
Agent audience = ``note_current`` (unlabelled versions only) minus papers with an open
``hides_notes`` attention item. Ties break by label, so the output is byte-stable.
"""

from __future__ import annotations

import json
import sqlite3

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


def label_groups(conn: sqlite3.Connection, *, limit: int = DEFAULT_LIMIT,
                 schema_id: str = SCHEMA_ID) -> dict[str, list[str]]:
    """``{group: [label, ...]}`` for every group in :data:`LABEL_GROUPS` (empty lists kept)."""
    out: dict[str, list[str]] = {}
    for group, claim_types in LABEL_GROUPS.items():
        marks = ", ".join("?" for _ in claim_types)
        rows = conn.execute(
            "SELECT c.normalized_label, COUNT(DISTINCT c.paper_id) AS papers "
            "FROM extracted_claims c JOIN note_current n ON n.note_id = c.structured_note_id "
            f"WHERE n.schema_id IS ? AND c.claim_type IN ({marks}) AND c.status = 'found' "
            "AND c.normalized_label IS NOT NULL AND TRIM(c.normalized_label) <> '' "
            f"AND c.paper_id NOT IN ({HIDDEN_PAPERS_SQL}) "
            "GROUP BY c.normalized_label ORDER BY papers DESC, c.normalized_label LIMIT ?",
            (schema_id, *claim_types, limit),
        ).fetchall()
        out[group] = [r[0] for r in rows]
    return out


def labels_json(conn: sqlite3.Connection, *, limit: int = DEFAULT_LIMIT) -> str:
    """The frozen ``labels.json`` text for a new batch (stable bytes for equal data)."""
    return json.dumps(label_groups(conn, limit=limit), ensure_ascii=False, indent=1) + "\n"
