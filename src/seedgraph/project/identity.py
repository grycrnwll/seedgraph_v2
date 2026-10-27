"""Identifier and title normalization (pure).

The v2 ORM merge flow (``upsert_work`` and the project-db review routing) was deleted in
stage 4; the library's own identity rules live in ``library/identity.py``. What remains
is pure and shared by the providers, the OA fetch, the provider cache, the library,
growth and migrate.

Strong-id precedence (highest canonical authority first):
``doi > openalex > arxiv > s2 > ssrn``.

The small title-normalization helpers (``sha1_hex`` / ``normalize_title`` /
``title_hash``) live here rather than in the phase_0 ``ids.py`` so this phase adds
no edit to a foundation module; they are pure and deterministic.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Literal, Mapping, TypeAlias

# Closed set of strong external id kinds (validated at the app boundary; the
# DB intentionally carries no CHECK on identifiers.id_type so new kinds can land
# in a later phase without a schema break).
IdType = Literal["doi", "openalex", "arxiv", "s2", "ssrn"]

# Canonical-key precedence: earlier wins.
ID_TYPE_PRECEDENCE: tuple[IdType, ...] = ("doi", "openalex", "arxiv", "s2", "ssrn")

# An incoming, user-supplied work record: a mapping of normalized fields
# (title/authors/year/venue + any subset of the strong id kinds above).
WorkLike: TypeAlias = Mapping[str, Any]


# --- Pure title-normalization helpers (trivial; implemented) ----------------

def sha1_hex(text: str) -> str:
    """SHA-1 of ``text`` (UTF-8) as a lowercase hex string."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_title(title: str) -> str:
    """Normalize a title for fuzzy matching: lower, NFKD-fold (drop diacritics),
    strip punctuation, collapse whitespace to single spaces.

    Deterministic and idempotent: ``normalize_title(normalize_title(t)) ==
    normalize_title(t)``. Recomputable on demand — no ``normalized_title`` column
    is stored (only the derived ``title_hash``).
    """
    decomposed = unicodedata.normalize("NFKD", title)
    no_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9]+", " ", no_marks.lower())
    return " ".join(cleaned.split())


def title_hash(title: str) -> str:
    """``sha1_hex(normalize_title(title))`` — the indexed fuzzy matcher.

    NEVER an auto-merge key (decision 13/64): a title-hash collision routes to
    review, it does not merge works.
    """
    return sha1_hex(normalize_title(title))


# --- ID normalization -------------------------------------------------------

# v1-ported surface-form normalizers (prototype identity.py:163-240, pure regex).
_DOI_PREFIX_RE = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/", re.IGNORECASE)
_ARXIV_URL_RE = re.compile(r"^(?:https?://)?arxiv\.org/(?:abs|pdf)/", re.IGNORECASE)
_ARXIV_LABEL_RE = re.compile(r"^arxiv:", re.IGNORECASE)
_ARXIV_VERSION_RE = re.compile(r"v\d+$")
_OPENALEX_URL_RE = re.compile(r"^(?:https?://)?openalex\.org/", re.IGNORECASE)

#: The arXiv-DOI alias family (``10.48550/arxiv.{id}``) — the identity/cache/fetch
#: layers must agree on this fold (gap scan §4.7, live-caught 2026-05-31). Single
#: definition: ``acquisition/fetch.py`` imports it from here.
ARXIV_DOI_RE = re.compile(r"(?i)^10\.48550/arxiv\.(.+)$")


def normalize_id(id_type: str, value: Any) -> str | None:
    """Normalize a raw external id value deterministically, or ``None`` if empty.

    Trims surrounding whitespace, then per type (v1 parity):

    * ``doi`` — strip ``doi:`` and any ``doi.org`` / ``dx.doi.org`` URL prefix
      (scheme optional); lowercase.
    * ``arxiv`` — strip ``arxiv.org/abs|pdf/`` URL prefixes and the ``arXiv:``
      label; DROP the ``v\\d+`` version suffix (v1/v2 of a preprint are the same
      work — the strip is load-bearing for the merge contract).
    * ``openalex`` — strip the ``openalex.org/`` URL prefix; uppercase a leading
      ``w``.
    * ``s2`` / ``ssrn`` — opaque, trimmed only.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if id_type == "doi":
        text = _DOI_PREFIX_RE.sub("", text)
        if text.lower().startswith("doi:"):
            text = text[len("doi:"):]
        text = text.strip().lower()
    elif id_type == "arxiv":
        text = _ARXIV_URL_RE.sub("", text)
        text = _ARXIV_LABEL_RE.sub("", text).strip()
        text = _ARXIV_VERSION_RE.sub("", text)
    elif id_type == "openalex":
        text = _OPENALEX_URL_RE.sub("", text).strip()
        if text and text[0] in ("w", "W"):
            text = "W" + text[1:]
    return text or None


# --- Strong ids ---------------------------------------------------------------

def canonical_key(rec: WorkLike) -> tuple[IdType, str] | None:
    """Return the single highest-precedence ``(id_type, normalized_value)`` present
    on ``rec``, or ``None`` if it carries no strong id.

    Precedence: ``doi > openalex > arxiv > s2 > ssrn`` (:data:`ID_TYPE_PRECEDENCE`).
    Defined as ``strong_ids(rec)[0]`` so key and lookup can never disagree (the
    arXiv-DOI fold applies to both by construction).
    """
    pairs = strong_ids(rec)
    return pairs[0] if pairs else None


def strong_ids(rec: WorkLike) -> list[tuple[IdType, str]]:
    """Return ALL strong ids present on ``rec`` as normalized ``(id_type, value)``
    pairs, in precedence order (used to look the record up by every id it carries).

    A DOI in the arXiv-DOI alias family (``10.48550/arxiv.{id}``) is FOLDED to an
    ``("arxiv", id)`` pair and the doi pair is NOT emitted — otherwise the doi-first
    precedence would keep ``{doi: 10.48550/arxiv.X}`` and ``{arxiv: X}`` as two
    disjoint identities (gap scan §4.7).
    """
    values: dict[IdType, str] = {}
    for id_type in ID_TYPE_PRECEDENCE:
        value = normalize_id(id_type, rec.get(id_type))
        if value:
            values[id_type] = value
    doi = values.get("doi")
    if doi is not None:
        match = ARXIV_DOI_RE.match(doi)
        if match:
            del values["doi"]
            folded = normalize_id("arxiv", match.group(1))
            if folded:
                values.setdefault("arxiv", folded)
    return [(id_type, values[id_type]) for id_type in ID_TYPE_PRECEDENCE if id_type in values]
