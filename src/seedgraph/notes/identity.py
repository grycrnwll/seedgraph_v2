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
- ``4``: ``label_fold`` equal (casefolded; hyphens and whitespace collapsed by the fold),
  non-empty and unique within the claim type on **both** sides; quotes and anchoring
  are ignored (a re-note often quotes a different sentence for the same claim).
- ``5``: strong text overlap whatever the label or quotes: the overlap coefficient
  ``|A & B| / min(|A|, |B|)`` of the content words (:func:`content_words`: stopwords
  and one-letter tokens dropped, "fixed" read as "constant") >= :data:`OVERLAP_MIN`,
  with both sets at least :data:`MIN_CONTENT_WORDS` long. A rival tier-5 candidate
  within :data:`TIER5_MARGIN` of the best is a tie (ambiguous), not a pick.

Quotes on different markdowns never overlap here: the caller relocates old quotes onto
the new markdown first (:meth:`ClaimFingerprint.relocated`; ``notes.replay`` does it).

:func:`match` is one-to-one and deterministic: candidates are sorted by (tier,
-overlap, -Jaccard, -text overlap, new field order, new key, old key) and taken
greedily. A candidate that ties a still-open rival (sharing its old or its new claim)
on (tier, overlap, Jaccard, text overlap; within the margin for tier 5) is never picked: every claim in the tie goes to :attr:`Matching.ambiguous`
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
#: Tier 5: the share of the shorter claim's content words found in the other. 0.6 asks
#: for a clear majority of the shorter claim to recur; two different limitations of one
#: paper that share only their topic words stay around 0.5 or below.
OVERLAP_MIN = 0.6
#: Tier 5 needs this many content words on each side (tiny texts overlap by accident).
MIN_CONTENT_WORDS = 4
#: Tier 5: a rival within this much of the best candidate makes the choice ambiguous
#: (about one word of a typical 8-12 content-word claim).
TIER5_MARGIN = 0.15
TIERS: tuple[str, ...] = ("S", "1", "2", "3", "4", "5")
STOPWORDS: frozenset[str] = frozenset("""
a about above across after again against all also although am an and any are as at be because been
being below between both but by can could did do does doing down during each either else for from
further had has have having he her here hers him his how however i if in into is it its itself just
may might more most much must my neither no nor not of off on once one only or other our ours out
over own per same she should so some such than that the their theirs them then there these they
this those though through thus to too under until up upon us very via was we were what when where
whether which while who whom whose why will with within without would yet you your
""".split())
#: Words a re-wording swaps freely, read as one.
SYNONYMS: dict[str, str] = {"fixed": "constant"}


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
    text_overlap: float = 0.0  # tier 5: the content-word overlap coefficient


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


def content_words(tokens: frozenset[str]) -> frozenset[str]:
    """Folded tokens without stopwords and one-letter tokens, synonyms unified."""
    folded = (t.casefold() for t in tokens)
    return frozenset(SYNONYMS.get(t, t) for t in folded if len(t) > 1 and t not in STOPWORDS)


def overlap_coefficient(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def label_key(f: ClaimFingerprint) -> str:
    """The label as compared: casefolded, hyphens and whitespace runs collapsed."""
    return " ".join(f.label_fold.casefold().replace("-", " ").split())


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
    same_label = label_key(old) == label_key(new)
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


def _pair(o: ClaimFingerprint, n: ClaimFingerprint, label_counts: Mapping[tuple[str, str, str], int],
          o_words: frozenset[str], n_words: frozenset[str]) -> Pair | None:
    """The best tier for one candidate pair, tiers 4 and 5 included."""
    t = tier(o, n)
    if t is not None:
        return Pair(o.key, n.key, *t)
    if o.claim_type != n.claim_type or o.field_root in SCALAR_ROOTS or n.field_root in SCALAR_ROOTS:
        return None
    jac = jaccard(o.text_tokens, n.text_tokens)
    label = label_key(o)
    if (label and label == label_key(n) and label_counts[("old", o.claim_type, label)] == 1
            and label_counts[("new", n.claim_type, label)] == 1):
        return Pair(o.key, n.key, "4", 0.0, jac)
    coef = overlap_coefficient(o_words, n_words)
    if min(len(o_words), len(n_words)) >= MIN_CONTENT_WORDS and coef >= OVERLAP_MIN:
        return Pair(o.key, n.key, "5", 0.0, jac, coef)
    return None


def match(old: Sequence[ClaimFingerprint], new: Sequence[ClaimFingerprint]) -> Matching:
    """One-to-one deterministic matching of ``old`` onto ``new`` (see the module doc)."""
    new_rank = {f.key: new_order(f) for f in new}
    label_counts: dict[tuple[str, str, str], int] = {}
    for side, fps in (("old", old), ("new", new)):
        for f in fps:
            k = (side, f.claim_type, label_key(f))
            label_counts[k] = label_counts.get(k, 0) + 1
    words = {("old", f.key): content_words(f.text_tokens) for f in old}
    words.update({("new", f.key): content_words(f.text_tokens) for f in new})

    cands: list[tuple[tuple, Pair]] = []
    for o in old:
        for n in new:
            pair = _pair(o, n, label_counts, words[("old", o.key)], words[("new", n.key)])
            if pair is None:
                continue
            score = (TIERS.index(pair.tier), -pair.overlap, -pair.jaccard, -pair.text_overlap)
            cands.append(((score, new_rank[n.key], n.key, o.key), pair))
    cands.sort(key=lambda c: c[0])

    def rival(p: Pair, k: tuple, pair: Pair, key: tuple) -> bool:
        if pair.tier == "5":  # tier 5 also needs a clear margin over the next candidate
            return p.tier == "5" and p.text_overlap >= pair.text_overlap - TIER5_MARGIN
        return k[0] == key[0]

    used_old: set[str] = set()
    used_new: set[str] = set()
    pairs: list[Pair] = []
    ambiguous: list[Ambiguity] = []
    for key, pair in cands:
        if pair.old in used_old or pair.new in used_new:
            continue
        rivals = [p for k, p in cands if p is not pair and rival(p, k, pair, key)
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
