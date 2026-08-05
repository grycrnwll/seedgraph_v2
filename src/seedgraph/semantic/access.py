"""Access-class resolution + export gating for the concept layer (plan §6, §7).

``resolve_access_class`` stamps the denormalized ``access_class`` on ``concepts``
and ``project_graph_edges`` = **MAX over every contributing claim's
``extracted_claims.access_class``** (decisions 30/60/76). It is a single
project.db-local query: it does NOT walk ``claim_spans → evidence_spans → … →
source_files`` and it never opens ``cache.db`` — the concept layer is
self-describing when the cache is detached. Missing column/value ⇒ fail closed
(treat as private / abort the build; doc 07 §7, §9).

The plain default-deny export gate (``is_shareable`` / ``field_allowed``) is the
phase_0 single source of truth and is re-exported here, NOT redefined (D8). Only
:func:`is_shareable_citation_edge` is phase_7-local, because it keys on
``EpistemicType`` — a vocabulary phase_0 has no notion of.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from ..errors import ValidationError

# Phase_0 owns the single default-deny export gate. Re-export, never duplicate.
from ..vocab import AccessClass, Provenance, field_allowed, is_shareable

__all__ = [
    "field_allowed",
    "is_shareable",
    "is_shareable_citation_edge",
    "resolve_access_class",
]

# Access-class lattice for the MAX fold (most-restrictive wins). Anything not
# listed (unknown / NULL / new) is treated as the most restrictive (fail-closed).
ACCESS_LATTICE_ORDER: tuple[str, ...] = (
    "open_access",
    "metadata_only",
    "licensed_future",
    "user_supplied_private",
)

# Citation-edge epistemic types that are shareable. Keyed on EpistemicType
# (D2 6-tier); deterministic / metadata_resolved → shareable, all else withheld.
_SHAREABLE_EDGE_EPISTEMIC: frozenset[str] = frozenset(
    {Provenance.deterministic.value, Provenance.metadata_resolved.value}
)


def has_access_class_column(conn: sqlite3.Connection) -> bool:
    """True iff ``extracted_claims`` carries the denormalized ``access_class`` column.

    The fail-closed prerequisite (doc 07 §9): the concept layer derives its
    ``access_class`` solely from this column. A single ``PRAGMA table_info`` read —
    no cache.db, no source walk.
    """
    rows = conn.execute("PRAGMA table_info(extracted_claims)").fetchall()
    return any(row[1] == "access_class" for row in rows)


def resolve_access_class(conn: sqlite3.Connection, claim_ids: Iterable[str]) -> str:
    """MAX (most-restrictive) ``access_class`` over the given contributing claims.

    Reads **only** ``extracted_claims.access_class`` for ``claim_ids`` — a single
    project.db query, no span/source walk, no cache.db. Folds via
    :func:`vocab.AccessClass.most_restrictive`; ``unknown``/missing values fail
    closed to ``user_supplied_private``. If the ``extracted_claims.access_class``
    column is absent entirely, raises (the build aborts rather than defaulting
    open; doc 07 §9). Empty ``claim_ids`` ⇒ ``user_supplied_private``.
    """
    if not has_access_class_column(conn):
        raise ValidationError(
            "extracted_claims.access_class is absent — the concept layer derives "
            "access_class solely from it and refuses to default open (doc 07 §9). "
            "Re-run note extraction so the column is populated before building "
            "concepts."
        )
    ids = list(dict.fromkeys(claim_ids))
    if not ids:
        return AccessClass.user_supplied_private.value
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT access_class FROM extracted_claims WHERE claim_id IN ({placeholders})",
        ids,
    ).fetchall()
    # Fail-closed coercion: NULL / '' / 'unknown' (or any unrecognized value) is
    # treated as the most-restrictive private class before the MAX fold.
    coerced: list[str] = []
    for (value,) in rows:
        if value in (None, "", AccessClass.unknown.value):
            coerced.append(AccessClass.user_supplied_private.value)
        else:
            coerced.append(value)
    return AccessClass.most_restrictive(*coerced).value


def is_shareable_citation_edge(epistemic_type: str) -> bool:
    """Phase_7-local allowlist for citation edges (which carry no ``access_class``).

    Keys on ``EpistemicType``: ``deterministic`` / ``metadata_resolved`` → True;
    everything else (notably ``llm_inferred``) → False. Used by export so
    deterministic/metadata citation edges stay shareable while interpretive
    overlay edges are withheld by default.
    """
    return epistemic_type in _SHAREABLE_EDGE_EPISTEMIC
