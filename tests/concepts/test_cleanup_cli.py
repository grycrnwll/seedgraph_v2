"""`seedgraph concepts review-sample | acceptance` (slice C11): the frozen 50-item review
sample (with a top-up from migrated agent merges when the batch has too few) and the
stage 4 acceptance report with its researcher-review section and wrong-merge gate.
Invented text only."""

from __future__ import annotations

import json
import re

import pytest
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.concepts import cleanup as cu
from seedgraph.concepts import refresh as rf
from seedgraph.concepts import researcher as rs
from seedgraph.library import store

from ._world import World, a
from .test_cleanup import BATCH, agent

runner = CliRunner()
SHAPES = ("round", "square", "oval", "flat", "tall", "wide", "thin", "bent")


def _home(tmp_path, *, migrated: int = 0):
    """Eight invented labels; the batch merges two pairs and links four; ``migrated`` agent
    merges without a batch (the top-up pool)."""
    home = tmp_path / "home"
    lib = store.open_library_at(home / "library")
    w = World(lib)
    for i, s in enumerate(SHAPES):
        w.note(f"cc_p{i}", [a(f"{s} widget", f"Invented: every {s} widget turns.")])
        w.note(f"cc_q{i}", [a(f"{s} gizmo", f"Invented: every {s} gizmo spins.")])
    agent(lib, "merge", "round widget", "round gizmo")
    agent(lib, "merge", "square widget", "square gizmo")
    for s in ("oval", "flat", "tall", "wide"):
        agent(lib, "link", f"{s} widget", f"{s} gizmo")
    agent(lib, "unrelated", "thin widget", "thin gizmo")  # never sampled
    for s in SHAPES[6:6 + migrated]:
        w.decide("merge", f"{s} widget", f"{s} gizmo", by="agent", family="assumption")
    rf.refresh(lib)
    lib.close()
    return home


def _sample_id(output: str) -> str:
    m = re.search(r"/concepts/review/(rs_[0-9a-f]+)", output)
    assert m, output
    return m.group(1)


def test_review_sample_prints_the_review_path_and_writes_the_list(tmp_path):
    home = _home(tmp_path)
    r = runner.invoke(app, ["concepts", "review-sample", "--n", "4", "--seed", "7", "--root", str(home)])
    assert r.exit_code == 0, r.output
    sid = _sample_id(r.output)
    listing = (home / "concepts" / "review" / f"{sid}.md").read_text(encoding="utf-8")
    assert listing.count("| merge |") + listing.count("| link |") == 4
    assert "top-up" not in r.output
    again = runner.invoke(app, ["concepts", "review-sample", "--n", "4", "--seed", "7", "--root", str(home)])
    assert _sample_id(again.output) == sid  # same population and seed: the same sample


def test_review_sample_tops_up_from_migrated_agent_merges_and_says_so(tmp_path):
    home = _home(tmp_path, migrated=2)
    r = runner.invoke(app, ["concepts", "review-sample", "--n", "8", "--root", str(home), "--json"])
    assert r.exit_code == 0, r.output
    body = json.loads(r.output)
    assert body["eligible"] == 6 and body["topped_up"] == 2 and body["size"] == 8
    assert body["path"] == f"/concepts/review/{body['sample_id']}"
    lib = store.open_library_at(home / "library")
    try:
        assert len(rs.review_sample(lib.conn, body["sample_id"]).decision_ids) == 8
    finally:
        lib.close()


def _inputs(tmp_path, home):
    ev = home / "concepts" / "eval" / "20260101T000000Z"
    ev.mkdir(parents=True)
    (ev / "report.json").write_text(json.dumps({
        "verdict": {"passed": False, "checks": {"separation": True, "recall": False}},
        "modes": {"keyword": {"recall_at_10": 0.5, "mrr": 0.25, "separation": 0.5},
                  "meaning": {"recall_at_10": 0.25, "mrr": 0.2, "separation": 0.75}}}), encoding="utf-8")
    (ev / "report.md").write_text("# invented\n", encoding="utf-8")
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps({
        "batch_id": BATCH,
        "estimate": {"agent_tokens_low": 1000, "agent_tokens_high": 2000, "packages": 3},
        "measured": {"agent_tokens": 2500, "source": "invented orchestrator count"},
        "ticket11": {"tokens": "2-4M", "pairs": "about 10k", "packages": "about 200"},
        "link_rule": {"before": ["w1"], "label_before": "old rule", "label_after": "new rule"},
    }), encoding="utf-8")
    return path


def test_acceptance_report_has_every_section_and_the_gate_after_marks(tmp_path):
    home = _home(tmp_path)
    inputs = _inputs(tmp_path, home)
    r = runner.invoke(app, ["concepts", "review-sample", "--n", "6", "--root", str(home)])
    sid = _sample_id(r.output)
    out = tmp_path / "acc"
    r = runner.invoke(app, ["concepts", "acceptance", "--inputs", str(inputs), "--sample", sid, "--out", str(out),
                            "--root", str(home)])
    assert r.exit_code == 0, r.output
    text = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## 1. Embeddings vs keyword", "## 2. Cost vs estimate", "## 3. Guardrails",
                    "## 4. Verdicts", "## 5. Queue", "## 6. Review sample", "## 7. Researcher review"):
        assert heading in text
    assert "FAIL" in text and "keyword kept" in text
    assert "2,500" in text and "invented orchestrator count" in text
    assert "pending: 0 of 6 marked" in text

    lib = store.open_library_at(home / "library")
    try:
        sample = rs.review_sample(lib.conn, sid)
        kinds = dict(lib.conn.execute("SELECT decision_id, kind FROM concept_decisions"))
        wrong = next(d for d in sample.decision_ids if kinds[d] == "merge")
        for d in sample.decision_ids:
            rs.mark(lib, sid, d, "wrong" if d == wrong else "right", note="invented")
    finally:
        lib.close()
    r = runner.invoke(app, ["concepts", "acceptance", "--inputs", str(inputs), "--sample", sid, "--out", str(out),
                            "--root", str(home)])
    assert r.exit_code == 0, r.output
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "wrong merges: 1" in text and "gate: FAIL" in text  # 2 merges sampled: 10% allows 0.2
    assert wrong in text
    body = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert body["review"]["gate_passed"] is False


def test_acceptance_reports_the_cleanup_and_the_residual_checks(tmp_path):
    home = _home(tmp_path)
    inputs = _inputs(tmp_path, home)
    lib = store.open_library_at(home / "library")
    try:
        cu.record(lib, cu.run(lib, BATCH, apply=True))
    finally:
        lib.close()
    out = tmp_path / "acc"
    r = runner.invoke(app, ["concepts", "acceptance", "--inputs", str(inputs), "--out", str(out), "--root", str(home)])
    assert r.exit_code == 0, r.output
    body = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert body["guardrails"]["audit"]["ok"] is True
    assert body["guardrails"]["residual"]["conflicts_unsettled"] == 0
    assert body["cleanup"]["counts"] == {r: 0 for r in cu.RULES}
    assert "no review sample yet" in (out / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("missing", ["inputs"])
def test_acceptance_refuses_without_inputs(tmp_path, missing):
    home = _home(tmp_path)
    r = runner.invoke(app, ["concepts", "acceptance", "--root", str(home)])
    assert r.exit_code != 0 and "inputs" in r.output
