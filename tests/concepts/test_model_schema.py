"""0010_concepts.sql carries existing concept_decisions rows forward (the upgrade path a
stage 1-3 library takes on first open). Invented rows only."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import seedgraph

SCHEMA = Path(seedgraph.__file__).parent / "db" / "schema" / "library"


def _ddl_0006_concept_decisions() -> str:
    text = (SCHEMA / "0006_lenses_concepts_attention.sql").read_text(encoding="utf-8")
    start = text.index("CREATE TABLE concept_decisions")
    end = text.index("CREATE INDEX ix_concept_decisions_pair")
    end = text.index(";", end) + 1
    return text[start:end]


def test_0010_copies_old_rows_unchanged_in_meaning():
    conn = sqlite3.connect(":memory:")
    conn.executescript(_ddl_0006_concept_decisions())
    old = [("cd_a", "merge", "common trends", "parallel trends", "agent", "pending_apply", "v2 bulk merge",
            "v2:test_proj", "2024-01-01"),
           ("cd_b", "never_merge", "ate", "att", "researcher", "applied", None, "library", "2025-01-01"),
           ("cd_c", "link", "autoregression", "serial correlation", "researcher", "undone", "invented",
            "library", "2025-02-01")]
    conn.executemany("INSERT INTO concept_decisions (decision_id, kind, label_a, label_b, decided_by, state, note, "
                     "origin, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", old)
    sql = (SCHEMA / "0010_concepts.sql").read_text(encoding="utf-8")
    conn.executescript(sql)
    rows = conn.execute("SELECT decision_id, kind, label_a, label_b, decided_by, state, note, origin, decided_at "
                        "FROM concept_decisions ORDER BY decision_id").fetchall()
    assert rows == old
    new_cols = conn.execute("SELECT family, family_b, sense_a, sense_b, name, quote_a_json, confidence, "
                            "veto_reason, batch_id FROM concept_decisions").fetchall()
    assert all(v is None for r in new_cols for v in r)
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}
    assert {"ix_concept_decisions_pair", "ix_concept_decisions_state"} <= indexes
    # the new kinds, makers and states are accepted; a misordered pair is not
    conn.execute("INSERT INTO concept_decisions (decision_id, kind, label_a, label_b, decided_by, state, decided_at) "
                 "VALUES ('cd_d', 'never_connect', 'a', 'b', 'rule', 'vetoed', 'x')")
    try:
        conn.execute("INSERT INTO concept_decisions (decision_id, kind, label_a, label_b, decided_by, decided_at) "
                     "VALUES ('cd_e', 'merge', 'z', 'a', 'agent', 'x')")
    except sqlite3.IntegrityError:
        pass
    else:
        raise AssertionError("label_a > label_b was accepted")
