"""Concept identity primitives (doc 07 §4.1; plan §5/§6, implementation step 3).

``normalize`` produces the merge_key; ``concept_id`` derives the DETERMINISTIC
primary key (``'concept::' || normalized_label``) so that across rebuilds an
unchanged label keeps its identity and any ``user_confirmed``/``user_split``
status + ``concept_constraints`` survive (doc 07 §7; D10). Also hosts the
relabel-then-one-merge canonicalization and the ``field-typed-label-wins`` /
``paper_frequency`` helpers consumed by :mod:`.concepts`.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Sequence

#: Prefix that makes the concept primary key deterministic and human-readable.
CONCEPT_ID_PREFIX = "concept::"

#: Claim types treated as generic (NOT field-typed) for ``field_typed_label``.
#: A field-typed surface form always wins over a generic mention.
_GENERIC_CLAIM_TYPES: frozenset[str] = frozenset({"other", "claim", "background"})

_KEEP_RE = re.compile(r"[^a-z0-9 ]+")
_WS_RE = re.compile(r"\s+")


def normalize(label: str) -> str:
    """Return the merge_key for ``label``.

    NFC-normalize, lowercase, keep only ``[a-z0-9 ]``, and collapse runs of
    whitespace to single spaces (then strip). This is the canonical merge_key:
    two surface forms with the same ``normalize`` output are the same concept.
    Pinned by ``test_concept_normalize`` for determinism + NFC stability.
    """
    if not label:
        return ""
    text = unicodedata.normalize("NFC", label).lower()
    text = _KEEP_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def concept_id(normalized_label: str) -> str:
    """Deterministic concept primary key: ``'concept::' || normalized_label``.

    Trivial by design (doc 07 §4.1): identity is the normalized label, never an
    opaque uuid, so rebuilds are stable. Caller passes an already-normalized
    label (the output of :func:`normalize`).
    """
    return f"{CONCEPT_ID_PREFIX}{normalized_label}"


def _tokens(label: str) -> set[str]:
    """Whitespace token set of an already-normalized label."""
    return {tok for tok in label.split() if tok}


def token_jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two (normalized) labels in ``[0, 1]``."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_acronym_expansion(short: str, long: str) -> bool:
    """True if ``short`` is the initialism of ``long`` (e.g. ``MLM`` ↔
    ``Masked Language Modeling``).

    Deterministic acronym detection used by both the auto-fold path and the
    over-merge guard tests. Compares the uppercase initials of ``long``'s tokens
    against ``short``'s letters. Operates on the normalized forms, so it is stable
    regardless of the surface casing/punctuation.
    """
    short_n = normalize(short).replace(" ", "")
    long_tokens = _tokens(normalize(long))
    # an acronym short is a single bare token of >=2 chars; the long form expands
    # to >=2 tokens whose leading letters spell the acronym.
    if len(short_n) < 2 or len(long_tokens) < 2:
        return False
    if " " in normalize(short):  # multi-token short is not an acronym
        return False
    initials = "".join(tok[0] for tok in normalize(long).split() if tok)
    return short_n == initials


def field_typed_label(labels: Sequence[str], claim_types: Sequence[str]) -> str:
    """Pick the human-facing ``canonical_label`` for a merged cluster.

    ``field-typed-label-wins``: the surface form contributed by the most-specific
    / field-typed claim wins over a generic mention; ties break deterministically
    (longest, then lexicographically) so the choice is reproducible.
    """
    pairs = list(zip(labels, claim_types))
    if not pairs:
        raise ValueError("field_typed_label requires at least one label")
    typed = [lbl for lbl, ct in pairs if (ct or "").strip().lower() not in _GENERIC_CLAIM_TYPES]
    pool = typed if typed else [lbl for lbl, _ in pairs]
    # deterministic tie-break: longest first, then lexicographically smallest.
    return sorted(set(pool), key=lambda s: (-len(s), s))[0]


def paper_frequency(conn: sqlite3.Connection, concept_normalized_label: str) -> int:
    """Distinct ``works`` mentioning the concept (the ``paper_frequency`` value).

    Counts distinct ``claim_concepts.work_id`` for the concept; a project.db-local
    query (no cache.db).
    """
    cid = concept_id(concept_normalized_label)
    row = conn.execute(
        "SELECT COUNT(DISTINCT work_id) FROM claim_concepts WHERE concept_id = ?",
        (cid,),
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0
