"""provider_cache read/write surface + the PINNED request-key grammar (§4.1/§6.5).

Owns the thin read/write surface over the ``provider_cache`` table in ``cache.db``
and the deterministic ``request_key`` builders. The
``referenced_works:src=<canonical id>`` key is a CROSS-PHASE CONTRACT: phase_2
re-derives it via the re-exported :func:`canonical_request_id` to recover each
work's reference list offline (binding r2-1 §4). DO NOT change the grammar without
amending phase_2.

Decisions implemented: 39/64/73 (cache lives in ``cache.db``, 30-day TTL, this
phase is the sole writer), D8 (raw payload may be cached locally but is
non-shareable — every export projects ONLY ``vocab.PROVIDER_SHAREABLE_FIELDS``),
D1 (content-addressed cache ids; ``canonical_request_id`` uses phase_5's identity
id-preference + ``normalize_title`` so a work and its cache key never drift).

ORM: :class:`ProviderCache` is ORM MAPPING ONLY and mirrors
``db/schema/cache/0003_provider_cache.sql`` (D6 — never ``create_all``-authored).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Index, UniqueConstraint
from sqlmodel import Field, SQLModel

from ..project.identity import ARXIV_DOI_RE, normalize_id, normalize_title

if TYPE_CHECKING:  # keep stdlib/optional imports out of module import for safety
    import sqlite3

# 30-day TTL (decision 39); mirrors the SQL DEFAULT.
DEFAULT_TTL_SECONDS = 2_592_000


class ProviderCache(SQLModel, table=True):
    """ORM mapping for ``provider_cache`` (cache.db). Mirrors 0003_provider_cache.sql.

    Mapping only (D6): the numbered ``.sql`` is the schema authority and a test
    asserts ORM metadata == migrated schema.
    """

    __tablename__ = "provider_cache"
    __table_args__ = (
        UniqueConstraint("provider", "request_key", name="uq_provider_cache_provider_request_key"),
        Index("ix_provider_cache_lookup", "provider", "request_key", "fetched_at"),
    )

    provider_cache_id: Optional[int] = Field(default=None, primary_key=True)
    provider: str
    request_key: str
    response: Optional[str] = None
    fetched_at: str
    ttl_seconds: int = DEFAULT_TTL_SECONDS


# --- request_key grammar (§6.5) — PINNED cross-phase contract ----------------
#
# ``canonical_request_id`` id-preference (§6.5): openalex > doi > arxiv > pmid > s2.
# The value is lowercased; the idtype name is the canonical short form. A work and
# the cache key it computes can never drift because both sides call THIS function.
#
# KEY-GRAMMAR NOTE (deliberate key move, Build A ch2): a DOI in the arXiv-DOI
# alias family (``10.48550/arxiv.{id}``) keys as ``arxiv=<id>`` — NOT
# ``doi=10.48550/...`` — so the cache key agrees with the identity/fetch layers'
# fold of the same family (gap scan §4.7, live-caught 2026-05-31). The grammar
# SHAPE is unchanged; only this one DOI family moves. Existing cached entries
# under the old ``doi=`` key re-fetch once (narrow blast radius). Pinned by
# ``test_request_key_contract`` (tests/test_phase_5b.py).
_CANON_PREF: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("openalex", ("openalex_id", "openalex")),
    ("doi", ("doi",)),
    ("arxiv", ("arxiv_id", "arxiv")),
    ("pmid", ("pmid", "pubmed_id")),
    ("s2", ("s2_id", "semantic_scholar_id", "s2")),
)


def _get_field(obj: Any, *names: str) -> Any:
    """Read the first present ``names`` field from ``obj`` (ORM attr OR mapping key)."""
    for name in names:
        value: Any = None
        if isinstance(obj, dict):
            value = obj.get(name)
        else:
            value = getattr(obj, name, None)
        if value not in (None, ""):
            return value
    return None


def canonical_request_id(work_or_record: Any) -> str:
    """Return the work's STRONGEST available id as ``'<idtype>=<value>'`` (lowercased).

    Preference order ``openalex > doi > arxiv > pmid > s2`` (§6.5). Falls back to a
    normalized-title key (``'title=<normalize_title(title)>'``) when no strong id is
    present, and ``'anon'`` when nothing is resolvable. Re-exported for phase_2 to
    reconstruct the ``referenced_works`` key and read a work's cached reference list
    offline — both sides call THIS function so a work and its cache key never drift.
    """
    for idtype, names in _CANON_PREF:
        value = _get_field(work_or_record, *names)
        if value is not None:
            text = str(value).strip().lower()
            if idtype == "doi":
                # arXiv-DOI alias family folds to the arxiv key (grammar note above).
                match = ARXIV_DOI_RE.match(text)
                if match:
                    folded = normalize_id("arxiv", match.group(1))
                    if folded:
                        return f"arxiv={folded.lower()}"
            return f"{idtype}={text}"
    title = _get_field(work_or_record, "title", "canonical_title")
    if title:
        return f"title={normalize_title(str(title))}"
    return "anon"


def key_by_doi(doi: str) -> str:
    """``by_doi:doi=<lowercased doi>`` (§6.5)."""
    return f"by_doi:doi={doi.lower()}"


def key_by_title(title: str, year: int | None = None) -> str:
    """``by_title:title=<normalize_title(title)>|year=<year or ''>`` (§6.5).

    Uses phase_5's ``normalize_title`` so the key matches the resolved work.
    """
    return f"by_title:title={normalize_title(title or '')}|year={year or ''}"


def key_referenced_works(work_or_record: Any) -> str:
    """``referenced_works:src=<canonical_request_id>`` — the load-bearing key (§6.5).

    This is the entry phase_2 reconstructs to project ``provider_reference`` edges
    offline. Deterministic; round-tripped by ``test_request_key_contract`` on both
    sides of the contract.
    """
    return f"referenced_works:src={canonical_request_id(work_or_record)}"


def key_oa_candidates(work_or_record: Any) -> str:
    """``oa_pdf_candidates:src=<canonical_request_id>`` (§6.5)."""
    return f"oa_pdf_candidates:src={canonical_request_id(work_or_record)}"


def key_by_openalex_ids(ids: Any) -> str:
    """``by_openalex_ids:ids=<'|'-joined sorted normalized W-ids>`` — ADDITIVE.

    Build D ch7: a NEW namespace ADDED to the §6.5 grammar (no existing key
    string or ``canonical_request_id`` behavior changes — the pinned
    ``test_request_key_contract`` is untouched). The distinct namespace is
    load-bearing (D-8): the batched W-id verb must never read the 30-day
    ``by_doi:`` NEGATIVE entries that pinned the untitled state in the first
    place. Ids are normalized via ``normalize_id('openalex', …)``, filtered to
    ``W…`` forms, de-duplicated, lowercased (the §6.5 lowercased-value
    convention), and SORTED — deterministic and order-insensitive, so the same
    id set always maps to the same cache row.
    """
    norm: set[str] = set()
    for raw in ids or []:
        oa = normalize_id("openalex", raw)
        if oa and oa.startswith("W"):
            norm.add(oa.lower())
    return f"by_openalex_ids:ids={'|'.join(sorted(norm))}"


# --- cache read/write over cache.db provider_cache ---------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(stamp: str) -> datetime:
    dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def cache_is_fresh(*, fetched_at: str, ttl_seconds: int, now: str | None = None) -> bool:
    """``now - fetched_at < ttl_seconds`` (port v1). Default ``now`` = current UTC."""
    try:
        fetched = _parse_iso(fetched_at)
    except (ValueError, TypeError):
        return False
    current = _parse_iso(now) if now is not None else datetime.now(timezone.utc)
    return (current - fetched).total_seconds() < ttl_seconds


def cache_get(conn: "sqlite3.Connection", *, provider: str, request_key: str) -> Any | None:
    """Return the parsed cached ``response`` for ``(provider, request_key)`` if a
    row exists AND :func:`cache_is_fresh`, else ``None``. A cached negative (empty
    list) hit is returned as-is (e.g. ``[]``) so a known miss is not re-queried
    within the TTL (port v1 ``cache_is_fresh`` semantics)."""
    row = conn.execute(
        "SELECT response, fetched_at, ttl_seconds FROM provider_cache "
        "WHERE provider = ? AND request_key = ?",
        (provider, request_key),
    ).fetchone()
    if row is None:
        return None
    response, fetched_at, ttl_seconds = row[0], row[1], row[2]
    if not cache_is_fresh(fetched_at=fetched_at, ttl_seconds=int(ttl_seconds)):
        return None
    if response is None:
        return None
    try:
        return json.loads(response)
    except (ValueError, TypeError):
        return None


def cache_put(
    conn: "sqlite3.Connection",
    *,
    provider: str,
    request_key: str,
    response: Any,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> None:
    """Upsert a ``(provider, request_key)`` row with the JSON-serialized
    ``response`` and a fresh ``fetched_at`` (UNIQUE(provider, request_key); the
    sole writer, decision 39/64/73)."""
    try:
        payload = json.dumps(response)
    except (TypeError, ValueError):
        payload = json.dumps(response, default=str)
    conn.execute(
        "INSERT INTO provider_cache (provider, request_key, response, fetched_at, ttl_seconds) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(provider, request_key) DO UPDATE SET "
        "response = excluded.response, fetched_at = excluded.fetched_at, "
        "ttl_seconds = excluded.ttl_seconds",
        (provider, request_key, payload, _now_iso(), int(ttl_seconds)),
    )
    conn.commit()
