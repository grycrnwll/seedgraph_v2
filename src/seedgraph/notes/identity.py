"""Claim identity across extractions (stage 3 design §4.4). Pure: no database.

``claim_id`` is minted fresh at every save and ``field_key`` carries a position, so a
claim is identified by its :class:`ClaimFingerprint` (``claim_type``, ``field_root``,
``label_fold``, ``text_tokens`` and, when anchored, its top quote's ``quote_hash``,
``markdown_hash`` and character span). ``overlay.fingerprint`` computes the dict form.

**Same claim** = the same ``claim_type`` (the agent's original type: an op's
fingerprint is taken from the agent's row at edit time, so a researcher re-type never
enters it) plus the first tier that holds:

- ``S``: a scalar field (one per note): same ``field_root``. Scalars match only this way.
- ``1``: quotes overlap (character IoU >= :data:`IOU_MIN` on the same markdown) and
  ``label_fold`` equal.
- ``2``: quotes overlap and token Jaccard of the claim texts >= :data:`JACCARD_MIN`.
- ``3``: neither side anchored, ``label_fold`` equal and Jaccard >= :data:`JACCARD_MIN`.

Quotes on different markdowns never overlap here: the caller relocates old quotes onto
the new markdown first (:meth:`ClaimFingerprint.relocated`; ``notes.replay`` does it).

:func:`match` is one-to-one and deterministic: candidates are sorted by (tier,
-overlap, -Jaccard, new field order, new key, old key) and taken greedily. A candidate
that ties a still-open rival (sharing its old or its new claim) on (tier, overlap,
Jaccard) is never picked: every claim in the tie goes to :attr:`Matching.ambiguous`
for the researcher to choose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from .format import SCALAR_FIELDS
from .interpretation import field_sort_key

#: Character IoU two quotes on the same markdown need to "overlap".
IOU_MIN = 0.5
#: Token Jaccard two claim texts need to be "similar".
JACCARD_MIN = 0.5
SCALAR_ROOTS: frozenset[str] = frozenset(name for name, _ in SCALAR_FIELDS)
TIERS: tuple[str, ...] = ("S", "1", "2", "3")


@dataclass(frozen=True)
class ClaimFingerprint:
    """One claim's identity. ``key`` names it for the caller (a claim id, usually)."""

    key: str
    claim_type: str
    field_root: str
    label_fold: str
    text_tokens: frozenset[str]
    field_key: str = ""
    quote_hash: str | None = None
    markdown_hash: str | None = None
    start_char: int | None = None
    end_char: int | None = None

    @property
    def anchored(self) -> bool:
        return self.markdown_hash is not None and self.start_char is not None and self.end_char is not None

    @property
    def span(self) -> tuple[int, int] | None:
        return (self.start_char, self.end_char) if self.anchored else None  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, key: str, d: Mapping[str, Any], *, field_key: str = "") -> ClaimFingerprint:
        """From ``overlay.fingerprint``'s dict (as stored in an op's ``target``)."""
        anchored = d.get("quote_hash") is not None and d.get("start_char") is not None
        return cls(
            key=key, claim_type=d["claim_type"], field_root=d.get("field_root") or "",
            label_fold=d.get("label_fold") or "", text_tokens=frozenset(d.get("text_tokens") or ()),
            field_key=field_key or d.get("field_root") or "",
            quote_hash=d.get("quote_hash") if anchored else None,
            markdown_hash=d.get("markdown_hash") if anchored else None,
            start_char=d.get("start_char") if anchored else None,
            end_char=d.get("end_char") if anchored else None)

    def relocated(self, markdown_hash: str, span: tuple[int, int] | None) -> ClaimFingerprint:
        """This fingerprint with its quote moved onto another markdown; ``span=None`` (the
        old quote no longer locates uniquely) leaves it unanchored."""
        if span is None:
            return replace(self, quote_hash=None, markdown_hash=None, start_char=None, end_char=None)
        return replace(self, markdown_hash=markdown_hash, start_char=span[0], end_char=span[1])


@dataclass(frozen=True)
class Pair:
    old: str
    new: str
    tier: str
    overlap: float
    jaccard: float


@dataclass(frozen=True)
class Ambiguity:
    """Claims that tie on (tier, overlap, Jaccard): the researcher chooses."""

    old: tuple[str, ...]
    new: tuple[str, ...]
    tier: str


@dataclass(frozen=True)
class Matching:
    pairs: tuple[Pair, ...]
    ambiguous: tuple[Ambiguity, ...]
    unmatched_old: tuple[str, ...]
    unmatched_new: tuple[str, ...]

    def new_for(self, old_key: str) -> str | None:
        return next((p.new for p in self.pairs if p.old == old_key), None)


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0  # two empty texts (a not_found row) are the same text
    return len(a & b) / len(a | b)


def new_order(f: ClaimFingerprint) -> tuple[int, int, str]:
    """The note's field order (numeric array index), then the key."""
    return field_sort_key(f.field_key or f.field_root, f.key)


def tier(old: ClaimFingerprint, new: ClaimFingerprint) -> tuple[str, float, float] | None:
    """``(tier, overlap, jaccard)`` when ``old`` and ``new`` are the same claim, else ``None``."""
    if old.claim_type != new.claim_type:
        return None
    if old.field_root in SCALAR_ROOTS or new.field_root in SCALAR_ROOTS:
        return ("S", 1.0, 1.0) if old.field_root == new.field_root else None
    jac = jaccard(old.text_tokens, new.text_tokens)
    same_label = old.label_fold == new.label_fold
    if old.anchored and new.anchored:
        overlap = iou(old.span, new.span) if old.markdown_hash == new.markdown_hash else 0.0  # type: ignore[arg-type]
        if overlap >= IOU_MIN and same_label:
            return "1", overlap, jac
        if overlap >= IOU_MIN and jac >= JACCARD_MIN:
            return "2", overlap, jac
        return None
    if not old.anchored and not new.anchored and same_label and jac >= JACCARD_MIN:
        return "3", 0.0, jac
    return None


def match(old: Sequence[ClaimFingerprint], new: Sequence[ClaimFingerprint]) -> Matching:
    """One-to-one deterministic matching of ``old`` onto ``new`` (see the module doc)."""
    new_rank = {f.key: new_order(f) for f in new}
    cands: list[tuple[tuple, Pair]] = []
    for o in old:
        for n in new:
            t = tier(o, n)
            if t is None:
                continue
            pair = Pair(o.key, n.key, *t)
            score = (TIERS.index(t[0]), -t[1], -t[2])
            cands.append(((score, new_rank[n.key], n.key, o.key), pair))
    cands.sort(key=lambda c: c[0])

    used_old: set[str] = set()
    used_new: set[str] = set()
    pairs: list[Pair] = []
    ambiguous: list[Ambiguity] = []
    for key, pair in cands:
        if pair.old in used_old or pair.new in used_new:
            continue
        rivals = [p for k, p in cands if p is not pair and k[0] == key[0]
                  and p.old not in used_old and p.new not in used_new
                  and (p.old == pair.old or p.new == pair.new)]
        if rivals:
            olds = sorted({pair.old, *(p.old for p in rivals)})
            news = sorted({pair.new, *(p.new for p in rivals)}, key=lambda k: (new_rank[k], k))
            ambiguous.append(Ambiguity(tuple(olds), tuple(news), pair.tier))
            used_old.update(olds)
            used_new.update(news)
            continue
        pairs.append(pair)
        used_old.add(pair.old)
        used_new.add(pair.new)
    return Matching(
        pairs=tuple(pairs), ambiguous=tuple(ambiguous),
        unmatched_old=tuple(sorted(f.key for f in old if f.key not in used_old)),
        unmatched_new=tuple(sorted((f.key for f in new if f.key not in used_new),
                                   key=lambda k: (new_rank[k], k))))
