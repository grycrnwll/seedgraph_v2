"""Build A (identity + merge correctness) acceptance tests.

Chunk 1 — identifier surface-form normalization: the v1-ported pure-regex
normalizers in ``project/identity.py:normalize_id`` must map every surface form
v1 accepted to one canonical form, idempotently, so the same id always compares
equal at the identity layer.

Chunk 2 — arXiv-DOI identity fold: ``{doi: 10.48550/arxiv.X}`` and
``{arxiv: X}`` resolve to ONE work (``strong_ids`` folds the alias family to an
arxiv pair; ``canonical_key`` is ``strong_ids[0]`` by construction).

Chunk 3 — conflicting-strong-id merge blocker (v1 Fix 1): a single-work match
whose incoming record carries a DIFFERENT value for a strong id the matched work
already owns never silently merges — it demotes to a ``duplicate_candidate``
review item (``reason="id_disagreement"``). Canonical fixture: preprint/published
pair sharing an arXiv id with two DOIs.

Chunk 4 — weak-id title-corroboration guard (v1 Fix 5): a single-work match
shared ONLY on weak ids (arxiv/s2/ssrn) with both title hashes present and
conflicting demotes to review (``reason="weak_id_title_conflict"``); a shared
doi/openalex id always merges regardless of title.

All tests offline / keyless.
"""

from __future__ import annotations


import pytest

from seedgraph.project.identity import canonical_key, normalize_id, strong_ids


# --------------------------------------------------------------------------
# Chunk 1 — normalizer table (mirrors v1 identity.py:163-240 docstring examples)
# --------------------------------------------------------------------------

NORMALIZE_CASES = [
    # doi: strip doi.org / dx.doi.org URL prefixes (scheme optional) + doi: label; lowercase
    ("doi", "https://doi.org/10.1234/ABC", "10.1234/abc"),
    ("doi", "https://dx.doi.org/10.1234/ABC", "10.1234/abc"),
    ("doi", "dx.doi.org/10.1/X", "10.1/x"),
    ("doi", "doi.org/10.1/X", "10.1/x"),
    ("doi", "doi:10.1234/X", "10.1234/x"),
    ("doi", "  10.1234/AbC ", "10.1234/abc"),
    # arxiv: strip abs/pdf URL prefixes + arXiv: label; DROP the v\d+ version
    # suffix (load-bearing: v1/v2 of a preprint are the same work)
    ("arxiv", "https://arxiv.org/abs/2401.01234v2", "2401.01234"),
    ("arxiv", "http://arxiv.org/pdf/2401.01234", "2401.01234"),
    ("arxiv", "arxiv.org/abs/2401.01234", "2401.01234"),
    ("arxiv", "arXiv:2401.01234v2", "2401.01234"),
    ("arxiv", "ARXIV:2401.01234", "2401.01234"),
    ("arxiv", "2401.01234v3", "2401.01234"),
    ("arxiv", "2401.01234", "2401.01234"),
    # openalex: strip openalex.org/ URL prefix; uppercase the leading W
    ("openalex", "https://openalex.org/W2044", "W2044"),
    ("openalex", "openalex.org/w2044", "W2044"),
    ("openalex", "w2044", "W2044"),
    ("openalex", "W2044", "W2044"),
    # s2 / ssrn: opaque provider ids — trimmed only, case preserved
    ("s2", " AbC123 ", "AbC123"),
    ("ssrn", " 4123456 ", "4123456"),
]


@pytest.mark.parametrize("id_type,raw,expected", NORMALIZE_CASES)
def test_normalize_id_surface_forms(id_type, raw, expected):
    assert normalize_id(id_type, raw) == expected


@pytest.mark.parametrize("id_type,raw,expected", NORMALIZE_CASES)
def test_normalize_id_idempotent(id_type, raw, expected):
    canonical = normalize_id(id_type, raw)
    assert normalize_id(id_type, canonical) == canonical


def test_normalize_id_empty_inputs_are_none():
    assert normalize_id("doi", None) is None
    assert normalize_id("arxiv", "   ") is None
    assert normalize_id("openalex", "") is None
    # a bare arXiv label with no id collapses to nothing, not an empty string
    assert normalize_id("arxiv", "arXiv:") is None


# --------------------------------------------------------------------------
# Chunk 1 — round trip: two arXiv surface forms resolve to ONE work
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Chunk 2 — arXiv-DOI alias fold: {doi: 10.48550/arxiv.X} == {arxiv: X}
# --------------------------------------------------------------------------

def test_arxiv_doi_alias_folds_to_arxiv_pair():
    # The alias doi yields the arxiv pair INSTEAD OF a doi pair, with the
    # chunk-1 normalizer applied to the captured id (version suffix dropped).
    assert strong_ids({"doi": "10.48550/arXiv.2401.01234v2"}) == [("arxiv", "2401.01234")]
    assert (
        canonical_key({"doi": "10.48550/arXiv.2401.01234"})
        == ("arxiv", "2401.01234")
        == canonical_key({"arxiv": "2401.01234"})
    )
    # The fold moves ONLY the 10.48550 family — any other DOI keys as doi.
    assert canonical_key({"doi": "10.1234/X"}) == ("doi", "10.1234/x")


def test_canonical_key_is_first_strong_id():
    # canonical_key is defined as strong_ids[0], so key and lookup can never
    # disagree; the folded arxiv pair outranks the weaker s2 id.
    rec = {"doi": "10.48550/arxiv.2401.01234", "s2": "S2X"}
    assert canonical_key(rec) == strong_ids(rec)[0] == ("arxiv", "2401.01234")
    assert canonical_key({}) is None
    assert strong_ids({}) == []


def test_arxiv_doi_alias_disagreeing_arxiv_id_drops_the_alias_doi():
    # Senior-review R3 pin (behavior, not endorsement): a record carrying BOTH
    # the alias doi 10.48550/arxiv.X and a DIFFERENT arxiv id Y keeps Y and
    # silently DROPS X — the fold's setdefault never overwrites an existing
    # arxiv pair, and the doi pair was already deleted.
    # ponytail: deliberate tradeoff — the X-vs-Y disagreement is DISCARDED in
    # this pure normalizer (no session to enqueue from), NOT routed to review
    # as an id_disagreement; ceiling: such a record never surfaces X anywhere
    # downstream (lookup, scalar mirrors, review payloads).
    rec = {"doi": "10.48550/arXiv.2401.01234", "arxiv": "9999.88888"}
    assert strong_ids(rec) == [("arxiv", "9999.88888")]
    # canonical_key is strong_ids[0] by construction — X is gone there too.
    assert canonical_key(rec) == ("arxiv", "9999.88888")


# --------------------------------------------------------------------------
# Chunk 3 — conflicting-strong-id merge blocker (v1 Fix 1)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Chunk 4 — weak-id title-corroboration guard (v1 Fix 5)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Chunks 6-8 — merge executor, review appliers, manual override
# --------------------------------------------------------------------------

import sqlite3  # noqa: E402



def _conn(h) -> sqlite3.Connection:
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


_NOW = "2026-07-02T00:00:00+00:00"


# --------------------------------------------------------------------------
# Chunk 7 — duplicate_candidate appliers (merge / exclude / reject)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Chunk 8 — manual-override applier for citation_resolution (+ migration 0014)
# --------------------------------------------------------------------------

MO_MD = (
    "# Citing Paper\n\nBody text.\n\n## References\n\n"
    "Doe, J. (2020). A quite unfindable manuscript title. Some Journal. "
    "https://doi.org/10.7777/miss\n"
)


class _EmptyProviders:
    """Provider chain double that misses everything (offline)."""

    async def by_doi(self, doi):
        return None

    async def by_title(self, title, year=None):
        return None


