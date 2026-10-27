"""Stage 4 acceptance checks against a real home (design §11; slice C11). Opt-in:

    SEEDGRAPH_STAGE4_HOME=C:/path/to/home  uv run pytest tests/acceptance/stage4

Optional: ``SEEDGRAPH_STAGE4_BATCH`` (default: the ``batch_id`` in
``<home>/concepts/acceptance/inputs.json``) and ``SEEDGRAPH_STAGE4_SAMPLE`` (default: the
newest review sample). The library is opened read-only (``mode=ro``): these tests never
refresh or write. Assertion messages carry counts and decision ids only, never labels.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from seedgraph.concepts import acceptance as acc
from seedgraph.concepts import cleanup as cu
from seedgraph.concepts import decide
from seedgraph.concepts import researcher as rs

HOME_ENV = "SEEDGRAPH_STAGE4_HOME"

pytestmark = pytest.mark.skipif(not os.environ.get(HOME_ENV), reason=f"set {HOME_ENV} to a v3 home to run")


@pytest.fixture(scope="module")
def home() -> Path:
    return Path(os.environ[HOME_ENV])


@pytest.fixture(scope="module")
def conn(home):
    db = home / "library" / "library.db"
    if not db.is_file():
        pytest.fail(f"no library at {db}")
    c = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    yield c
    c.close()


@pytest.fixture(scope="module")
def inputs(home) -> dict:
    path = home / "concepts" / "acceptance" / "inputs.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    batch = os.environ.get("SEEDGRAPH_STAGE4_BATCH") or data.get("batch_id")
    if not batch:
        pytest.skip("no batch: set SEEDGRAPH_STAGE4_BATCH or write concepts/acceptance/inputs.json")
    return {**data, "batch_id": batch}


@pytest.fixture(scope="module")
def sample(conn) -> rs.ReviewSample:
    sid = os.environ.get("SEEDGRAPH_STAGE4_SAMPLE")
    s = rs.review_sample(conn, sid) if sid else next(iter(rs.review_samples(conn)), None)
    if s is None:
        pytest.skip("no review sample yet (seedgraph concepts review-sample)")
    return s


def test_concept_guardrails_hold(conn):
    """§11.4: no concept spans families, holds a never-merge / never-connect pair, is named
    outside its members, has a stoplist conflict or an unchecked ambiguous merge; the seed
    pairs are apart."""
    res = decide.audit(conn)
    assert res.ok, res.as_dict()


def test_researcher_never_merge_pairs_are_apart(conn, inputs):
    data = acc.guardrail_checks(conn, inputs["batch_id"], acc.rf.plan(conn))
    assert data["never_merge_together"] == 0


def test_bridge_batch_has_no_unsettled_conflicts_inside_rows_or_dead_senses(conn, inputs):
    f = cu.findings(conn, inputs["batch_id"])
    assert [c.decision_ids for c in f.applied_conflicts] == []
    assert [c.decision_ids for c in f.unsettled] == []
    assert [r["decision_id"] for r in f.inside] == []
    assert [c.unrelated_id for c in f.contradicted] == []
    assert [r["decision_id"] for r in f.dead] == []


def test_review_sample_is_stratified_from_agent_merges_and_links(conn, inputs, sample):
    items = acc.sample_items(conn, sample)
    assert len(items) == sample.size == len(sample.decision_ids)
    assert {i["kind"] for i in items} <= {"merge", "link"}
    assert all(i["decided_by"] == "agent" for i in items)
    comp = acc.composition(items)
    assert set(comp["by_verdict"]) == {"merge", "link"}
    assert len(comp["by_family"]) > 1


def test_wrong_merge_gate_once_every_item_is_marked(conn, sample):
    s = rs.review_summary(conn, sample.sample_id)
    if s.gate_passed is None:
        pytest.skip(f"review pending: {s.marked} of {s.size} marked")
    assert s.gate_passed, f"wrong merges {len(s.wrong_merges)} > {s.gate_limit:g}: {list(s.wrong_merges)}"


def test_acceptance_report_renders(conn, home, inputs, tmp_path):
    if "estimate" not in inputs or "measured" not in inputs:
        pytest.skip("inputs.json lacks the cost figures")
    text = acc.render(acc.gather(conn, home, inputs))
    for n in range(1, 8):
        assert f"## {n}." in text
