"""Phase 3b — strong-id-gated reference resolution + v1 precision guard.

Resolves each :class:`~seedgraph.citation.bib_parser.BibEntry` against the
``phase_5b`` provider chain behind the ``phase_5`` identity strong-id gate, then
applies v1's precision guard (``is_false_resolution`` +
``REFERENCE_WORK_DOI_PREFIXES``). This module is **PURE with respect to the DB**:
it performs resolution only (it may hit the provider chain, which is offline-safe
within its ``provider_cache`` TTL) and writes nothing — persistence and edge
projection live in :mod:`seedgraph.citation.parsed_bib`.

Resolution-confidence rubric (§7; decision 73(b); deterministic, no-LLM —
decisions 15/58/75):
- **Strong-id hit** (entry carries DOI/arXiv) → ``status='resolved'``,
  ``resolution_source`` ``doi``/``arxiv``, ``confidence≈1.0``. A parsed DOI the
  chain MISSES is unverified: it stays a candidate and the entry falls through to
  the title(+year) path — never a self-minted confident identity (decision 13).
  The arXiv self-record exception is deliberate v1 parity and stays.
- **Title(+year) hit** whose chain record carries a strong id (``canonical_key``
  is not ``None`` / not a ``title_hash``) and passes the precision guard →
  ``resolved``, ``resolution_source='title_year'``, ``confidence`` token-overlap
  scaled (~0.7–0.9).
- **Title-only / chain record with no strong id / unpinnable** → ``ambiguous``
  (no edge).
- **Precision-guard rejection** (refwork-DOI denylist with no own DOI, or
  ``<2``-content-token title overlap) → ``suspect`` with ``reject_reason`` (no edge).
- **No usable fields** → ``unresolved`` (entry kept raw — recall stays measurable).

Forbidden here: ``llm_inferred`` / ``llm_extracted`` (decision 14; AGENTS.md §2.2)
— resolution exchanges only public bibliographic metadata; restricted full text
never leaves the machine (decision 73; doc 03 §6).
"""

from __future__ import annotations

import asyncio
import inspect
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from seedgraph.citation.bib_parser import BibEntry

# ===========================================================================
# Resolution-precision guard (ported from v1 — kill false-positive refwork hits)
# ===========================================================================
#: Reference-work DOI SUB-TOKENS that, when a parsed entry resolved to them with
#: NO own DOI (a blind ``by_title`` shred), are rejected as false positives.
#: Two shapes coexist: 10.1093 named sub-prefixes (the bare 10.1093 registrant is
#: DELIBERATELY ABSENT — it also hosts legit scholarly journals) and whole
#: reference-work-only registrants (10.4135 SAGE, 10.4324 Routledge, …).
REFERENCE_WORK_DOI_PREFIXES: frozenset[str] = frozenset(
    {
        # 10.1093 named sub-prefixes (the registrant itself is NEVER denied)
        "benz",
        "gmo",
        "odnb",
        "ww",
        "anb",
        "acref",
        "acrefore",
        "oed",
        "gao",
        # whole reference-work-only registrants (match before the first '/')
        "10.4135",
        "10.4324",
        "10.5040",
        "10.1553",
    }
)

#: A leading qualifier on a 10.1093 reference-work suffix that must be STRIPPED
#: before the sub-token is read: ``10.1093/ref:odnb/107897`` keys as ``odnb``.
_REFWORK_QUALIFIER_RE = re.compile(r"^(?:ref|acprof):", re.IGNORECASE)


def _refwork_subtoken(doi: str) -> str:
    """The reference-work key of a DOI: the first ``/``-segment of the suffix.

    Split on ``/`` only, strip a leading ``ref:`` / ``acprof:`` qualifier,
    lowercase (the tokenizer is load-bearing).
    """
    if not doi:
        return ""
    parts = doi.split("/", 2)
    if len(parts) < 2:
        return ""
    suffix_head = parts[1].strip().lower()
    suffix_head = _REFWORK_QUALIFIER_RE.sub("", suffix_head, count=1)
    return suffix_head


def _doi_is_refwork(doi: Optional[str]) -> bool:
    """Whether ``doi`` is a denylisted reference-work DOI.

    True when EITHER the bare registrant (segment before the first ``/``) OR the
    sub-token (first ``/``-segment of the suffix, ``ref:``/``acprof:`` stripped) is
    in :data:`REFERENCE_WORK_DOI_PREFIXES`.
    """
    if not doi:
        return False
    d = doi.strip().lower()
    registrant = d.split("/", 1)[0]
    if registrant in REFERENCE_WORK_DOI_PREFIXES:
        return True
    return _refwork_subtoken(d) in REFERENCE_WORK_DOI_PREFIXES


#: Stopwords removed before counting SHARED CONTENT tokens for the overlap gate.
_OVERLAP_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
        "by", "with", "from", "as", "is", "are", "be", "this", "that", "into",
        "über", "der", "die", "das", "und", "le", "la", "les", "de", "du",
    }
)

#: Word-character token splitter for the overlap gate (drops punctuation).
_CONTENT_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _content_tokens(text: Optional[str]) -> set[str]:
    """Lowercased non-stopword alphabetic tokens of ``text`` (overlap gate input)."""
    if not text:
        return set()
    toks = {t.lower() for t in _CONTENT_TOKEN_RE.findall(text)}
    return {t for t in toks if t not in _OVERLAP_STOPWORDS and len(t) > 1}


@dataclass
class ResolvedRef:
    """Resolution outcome for one :class:`BibEntry`.

    ``status`` is one of the :class:`~seedgraph.vocab.ResolutionStatus` values
    (``resolved | unresolved | ambiguous | suspect``). ``resolved_work_id`` is set
    only downstream (this module is DB-pure). ``candidates`` holds the provider
    candidate records (the chosen record is ``candidates[0]`` for ``resolved``);
    ``rejected_record`` / ``reject_reason`` describe a ``suspect`` precision-guard
    rejection (``reject_reason`` ∈ ``{'refwork_doi_denylist','low_title_overlap'}``).
    """

    entry: BibEntry
    resolved_work_id: str | None
    status: str
    resolution_source: str | None
    confidence: float
    candidates: list[dict] = field(default_factory=list)
    rejected_record: dict | None = None
    reject_reason: str | None = None


def is_false_resolution(entry: BibEntry, record: dict) -> str | None:
    """v1 precision guard: decide whether ``record`` is a false resolution of ``entry``.

    Returns the rejection reason — ``'refwork_doi_denylist'`` (the record's DOI
    registrant is denylisted and ``entry`` carries no own DOI) or
    ``'low_title_overlap'`` (fewer than 2 shared content tokens between
    ``entry.title`` and the record title) — or ``None`` if the resolution is
    accepted. Ported from v1 ``_is_false_resolution`` (decision 73/75).
    """
    # --- (b) reference-work DOI denylist (entry had no own DOI) ----------------
    record_doi = record.get("doi")
    if entry.doi is None and _doi_is_refwork(record_doi):
        return "refwork_doi_denylist"

    # --- (c) title-token-overlap backstop --------------------------------------
    record_title = record.get("title")
    if entry.doi is None and entry.arxiv is None and entry.title and record_title:
        q_tokens = _content_tokens(entry.title)
        r_tokens = _content_tokens(record_title)
        shared = len(q_tokens & r_tokens)
        # Relative-threshold escape for short titles: when either side is 1–2
        # content tokens, a single shared content token is enough.
        min_side = min(len(q_tokens), len(r_tokens))
        needed = 1 if (min_side and min_side <= 2) else 2
        if q_tokens and r_tokens and shared < needed:
            return "low_title_overlap"

    return None


# ===========================================================================
# Provider-record helpers
# ===========================================================================
def _run(value: Any) -> Any:
    """Resolve a value that may be an awaitable (the async provider chain) offline.

    Supports both a plain sync mock (returns a dict directly) and the real
    ``ProviderChain`` (returns a coroutine), so resolution is callable without an
    event loop in hand. No running loop is expected at the call site.
    """
    if inspect.isawaitable(value):
        return asyncio.run(_drain(value))
    return value


async def _drain(awaitable: Any) -> Any:
    return await awaitable


# Provider records use ``*_id`` suffixed strong-id keys; ``identity.canonical_key``
# expects the bare ``doi/openalex/arxiv/s2/ssrn`` keys (phase_5). Normalize before
# the strong-id gate so the gate and the downstream upsert agree.
_RECORD_ID_MAP: tuple[tuple[str, str], ...] = (
    ("doi", "doi"),
    ("openalex_id", "openalex"),
    ("openalex", "openalex"),
    ("arxiv_id", "arxiv"),
    ("arxiv", "arxiv"),
    ("semantic_scholar_id", "s2"),
    ("s2_id", "s2"),
    ("s2", "s2"),
    ("ssrn_id", "ssrn"),
    ("ssrn", "ssrn"),
)


def record_to_identity(record: dict) -> dict:
    """Normalize a provider record's strong-id keys to identity's bare keys."""
    out: dict[str, Any] = {}
    for src, dst in _RECORD_ID_MAP:
        if record.get(src) and dst not in out:
            out[dst] = record[src]
    if record.get("title"):
        out["title"] = record["title"]
    if record.get("year") is not None:
        out["year"] = record["year"]
    if record.get("authors"):
        out["authors"] = record["authors"]
    return out


def _strong_key(identity, record: dict):
    """The record's canonical strong-id key, or ``None`` (the strong-id gate).

    A ``title_hash`` canonical key never produces an edge (decision 64 parity);
    ``identity.canonical_key`` returns only ``doi/openalex/arxiv/s2/ssrn`` or
    ``None`` (a title-only record), so ``None`` is exactly the "no strong id" case.
    """
    key = identity.canonical_key(record_to_identity(record))
    if key is None:
        return None
    id_type = key[0]
    type_name = getattr(id_type, "value", id_type)
    if str(type_name) == "title_hash":
        return None
    return key


def _title_year_confidence(entry: BibEntry, record: dict) -> float:
    """Token-overlap-scaled confidence in ``[0.7, 0.9]`` for a title(+year) hit.

    NOT a hardcoded 1.0 (1.0 is provider-only / strong-id). Deterministic, no-LLM.
    """
    a = _content_tokens(entry.title)
    b = _content_tokens(record.get("title"))
    if not a or not b:
        jac = 0.0
    else:
        jac = len(a & b) / len(a | b)
    return round(min(0.9, max(0.7, 0.7 + 0.2 * jac)), 4)


def _resolve_one(entry: BibEntry, providers, identity) -> ResolvedRef:
    """Resolve ONE entry per the §7 rubric (DB-pure)."""
    # 1. Strong-id direct: the entry carries its own DOI/arXiv.
    doi_candidate: dict | None = None
    if entry.doi is not None:
        rec = _run(providers.by_doi(entry.doi))
        if rec:
            return ResolvedRef(
                entry, None, "resolved", "doi", 1.0, candidates=[dict(rec)]
            )
        # Chain MISS: the parsed DOI is UNVERIFIED (possibly OCR-garbled) — never
        # self-mint a confident identity from it (decision 13; phase_3b §7). Keep
        # it as a candidate and fall through to the title(+year) path.
        doi_candidate = entry.to_record()
    if entry.arxiv is not None:
        # arXiv self-record exception (deliberate in v1): an arXiv id is
        # self-verifying enough to resolve without a chain hit.
        rec = _run(providers.by_doi(entry.arxiv)) or entry.to_record()
        return ResolvedRef(
            entry, None, "resolved", "arxiv", 1.0, candidates=[dict(rec)]
        )

    # 2. Title(+year) via the provider chain.
    if entry.title:
        rec = _run(providers.by_title(entry.title, entry.year))
        if rec:
            rec = dict(rec)
            if _strong_key(identity, rec) is not None:
                # Strong-id hit → precision guard before any edge.
                reason = is_false_resolution(entry, rec)
                if reason is not None:
                    return ResolvedRef(
                        entry, None, "suspect", None, 0.0,
                        candidates=[rec], rejected_record=rec, reject_reason=reason,
                    )
                conf = _title_year_confidence(entry, rec)
                return ResolvedRef(
                    entry, None, "resolved", "title_year", conf, candidates=[rec]
                )
            # Title-only chain hit (no strong id) → ambiguous (decision 64).
            candidates = [rec] + ([doi_candidate] if doi_candidate else [])
            return ResolvedRef(entry, None, "ambiguous", None, 0.0, candidates=candidates)
        # Chain miss but the entry has a title → ambiguous (unpinnable, kept raw).
        candidates = [doi_candidate] if doi_candidate else []
        return ResolvedRef(entry, None, "ambiguous", None, 0.0, candidates=candidates)

    if doi_candidate is not None:
        # Unconfirmed parsed DOI, no title to corroborate → ambiguous, reviewer
        # sees the DOI in candidates (decision 13: NO edge, NO confident identity).
        return ResolvedRef(entry, None, "ambiguous", None, 0.0, candidates=[doi_candidate])

    # 3. No usable fields → unresolved (kept raw — recall stays measurable).
    return ResolvedRef(entry, None, "unresolved", None, 0.0, candidates=[])


def resolve_references(
    entries: Iterable[BibEntry],
    providers,
    identity,
    *,
    citing_work_id: str,
) -> list[ResolvedRef]:
    """Resolve ``entries`` to works via DOI/arXiv first, else title(+year).

    For each entry: try strong ids (DOI/arXiv) directly; otherwise query the
    ``providers`` chain by title(+year). Gate every would-be edge through the
    ``identity`` strong-id rule (``canonical_key`` must not be ``None`` /
    ``title_hash``), then apply the :func:`is_false_resolution` precision guard.
    Emit one :class:`ResolvedRef` per entry with ``status``/``resolution_source``/
    ``confidence`` per the §7 rubric. PURE w.r.t. the DB (no writes).

    ``providers`` = the ``phase_5b`` provider chain; ``identity`` = the ``phase_5``
    identity module. ``citing_work_id`` is threaded for diagnostics / review-payload
    context only.
    """
    return [_resolve_one(entry, providers, identity) for entry in entries]
