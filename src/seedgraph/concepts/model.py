"""The concept model's identities (stage 4 design §1): label keys, sense keys, handles,
concept and decision ids, claim fingerprints, grounding, and handle resolution.

- A **label** is ``(family, label_key)`` with ``label_key = semantic.merge.normalize(label)``.
- A **sense** is ``(family, label_key, sense_id)``; every label starts with ``s0``.
- A **handle** is the stable public name of a sense: ``family/label_key`` for ``s0``,
  ``family/label_key~s2`` otherwise. It always resolves to the current concept
  (:func:`resolve`), whatever merges and splits reshaped the groups.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from ..semantic.merge import normalize

DEFAULT_SENSE = "s0"
_FIELD_INDEX_RE = re.compile(r"\[\d+\]$")


class SenseKey(NamedTuple):
    family: str
    label_key: str
    sense_id: str = DEFAULT_SENSE

    def __str__(self) -> str:
        return f"{self.family}/{self.label_key}~{self.sense_id}"


def label_key(label: str | None) -> str:
    return normalize(label or "")


def handle(key: SenseKey) -> str:
    """The public handle: ``family/label_key`` for the default sense, else ``…~sense_id``."""
    base = f"{key.family}/{key.label_key}"
    return base if key.sense_id == DEFAULT_SENSE else f"{base}~{key.sense_id}"


def parse_handle(text: str) -> tuple[str, str, str | None]:
    """``(family, label_key, sense_id | None)``; the label part is normalized, so a URL
    carrying the surface form ("Parallel-trends") still resolves."""
    family, sep, rest = text.strip().partition("/")
    if not sep or not family or not rest:
        raise ValueError(f"not a concept handle: {text!r} (expected family/label[~sense])")
    label, _, sense = rest.partition("~")
    return family.strip(), label_key(label), (sense.strip() or None)


def concept_id(family: str, smallest: SenseKey) -> str:
    """``'cn_' + hash(family + smallest member sense key)`` (§1.2)."""
    return "cn_" + hashlib.sha256(f"{family}|{smallest}".encode()).hexdigest()[:20]


def decision_id(kind: str, label_a: str, label_b: str, *, family: str | None = None,
                family_b: str | None = None, sense_a: str | None = None, sense_b: str | None = None) -> str:
    """``'cd_' + hash(kind + subjects)``. A family-free, sense-free decision hashes as the
    stage 1 migration did, so a new researcher decision on a migrated pair is that row."""
    if family is None and family_b is None and sense_a is None and sense_b is None:
        text = f"{kind}|{label_a}|{label_b}"
    else:
        text = f"{kind}|{family or ''}|{family_b or ''}|{label_a}~{sense_a or ''}|{label_b}~{sense_b or ''}"
    return "cd_" + hashlib.sha256(text.encode()).hexdigest()[:20]


def ordered_pair(label_a: str, sense_a: str | None, label_b: str, sense_b: str | None,
                 family_a: str | None = None, family_b: str | None = None) -> tuple:
    """The two sides ordered as the schema's CHECK wants: ``(label, sense, family)`` a < b.
    Returns ``(label_a, sense_a, family_a, label_b, sense_b, family_b)``."""
    a = (label_a, sense_a or "", family_a or "", sense_a, family_a)
    b = (label_b, sense_b or "", family_b or "", sense_b, family_b)
    lo, hi = sorted((a, b), key=lambda t: t[:3])
    return lo[0], lo[3], lo[4], hi[0], hi[3], hi[4]


# --- claims -------------------------------------------------------------------------------------

def claim_fingerprint(claim: Mapping[str, Any], span: Mapping[str, Any] | None) -> str:
    """Canonical JSON of the stage 3 fingerprint fields (``notes.overlay.fingerprint``)
    computed from a ``current_claims`` row as it reads now, plus its top quote. Works for
    researcher-added claims too (they have no ``extracted_claims`` row)."""
    d = {
        "claim_type": claim["claim_type"],
        "field_root": _FIELD_INDEX_RE.sub("", claim["field_key"] or ""),
        "label_fold": normalize(claim["normalized_label"] or ""),
        "text_tokens": sorted(set(normalize(claim["claim_text"] or "").split())),
        "quote_hash": span["quote_hash"] if span else None,
        "markdown_hash": span["markdown_hash"] if span else None,
        "start_char": span["start_char"] if span else None,
        "end_char": span["end_char"] if span else None,
    }
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def supports(claim: Mapping[str, Any]) -> bool:
    """Grounding (§1.3): an anchored agent quote (trust ``paper_states``), a researcher
    claim of kind ``quote`` or ``assertion``, or an agent claim the researcher edited that
    is still anchored. ``agent_inferred`` claims support nothing."""
    if claim["kind"] in ("quote", "assertion"):
        return True
    if claim["trust"] == "paper_states":
        return True
    return claim["trust"] == "researcher" and claim["span_id"] is not None and claim["status"] == "found"


# --- resolution ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class SenseChoice:
    handle: str
    sense_id: str
    concept_id: str
    gloss: str | None


@dataclass(frozen=True)
class Resolution:
    """``kind``: ``concept`` (``concept_id`` set), ``chooser`` (a split label: pick one of
    ``senses``) or ``missing``."""

    kind: str
    handle: str
    concept_id: str | None = None
    senses: tuple[SenseChoice, ...] = field(default_factory=tuple)


def resolve(conn: sqlite3.Connection, text: str) -> Resolution:
    """Resolve a handle against the derived tables (refresh first when stale)."""
    family, key, sense = parse_handle(text)
    rows = conn.execute(
        "SELECT m.sense_id, m.concept_id, s.gloss FROM concept_members m "
        "LEFT JOIN concept_senses s ON s.family = m.family AND s.label_key = m.label_key "
        "AND s.sense_id = m.sense_id WHERE m.family = ? AND m.label_key = ? ORDER BY m.sense_id",
        (family, key)).fetchall()
    canonical = handle(SenseKey(family, key, sense or DEFAULT_SENSE))
    if sense is not None:
        hit = next((r for r in rows if r[0] == sense), None)
        return Resolution("concept", canonical, hit[1]) if hit else Resolution("missing", canonical)
    s0 = next((r for r in rows if r[0] == DEFAULT_SENSE), None)
    if s0 is not None:
        return Resolution("concept", canonical, s0[1])
    if not rows:
        return Resolution("missing", canonical)
    return Resolution("chooser", canonical, None, tuple(
        SenseChoice(handle(SenseKey(family, key, r[0])), r[0], r[1], r[2]) for r in rows))
