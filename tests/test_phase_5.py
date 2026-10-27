"""Phase 5 (Project Model MVP) acceptance tests.

Each test maps to a plan §11 acceptance item and runs fully offline / keyless
(no LLM, no network). The module-level imports double as an import-cleanliness
check: the whole phase_5 surface must import without error.
"""

from __future__ import annotations

import re


# Import-cleanliness: these must all import without error at collection time.
from seedgraph.project import identity


# --------------------------------------------------------------------------
# test_ids
# --------------------------------------------------------------------------

def test_phase5_ids_shape_and_normalization():
    from seedgraph.ids import new_id

    a, b = new_id("work"), new_id("work")
    assert re.fullmatch(r"work_[0-9a-f]{32}", a)
    assert a != b  # unique across calls

    # case + diacritic + punctuation + whitespace folding, idempotent
    raw = "  Héllo,   WORLD!!  "
    norm = identity.normalize_title(raw)
    assert norm == "hello world"
    assert identity.normalize_title(norm) == norm  # idempotent
    assert identity.normalize_title("Café") == identity.normalize_title("cafe")
    # title_hash is sha1 over the normalized form
    assert identity.title_hash(raw) == identity.sha1_hex(identity.normalize_title(raw))


# --------------------------------------------------------------------------
# test_schema_parity (D6 — ORM-vs-migration parity)
# --------------------------------------------------------------------------

def _affinity(type_str: str) -> str:
    """SQLite type affinity (with the SQLModel JSON column treated as TEXT, since the
    0002 migration declares ``authors`` TEXT and SQLModel serializes JSON to it)."""
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"  # TEXT/VARCHAR/CHAR/CLOB/JSON


# --------------------------------------------------------------------------
# test_slug_layout
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_identity_merge
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_project_service
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_inclusion_status
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_corpus_filter_isolation
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_review_queue
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# test_cli_project
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# test_api_read_routes (scaffold-critic gap 1)
# --------------------------------------------------------------------------

