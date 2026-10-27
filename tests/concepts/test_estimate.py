"""The concept tasks' dry run (stage 4 design §4.4-§4.5): it writes nothing, it matches the
batch create_batch makes, and a token budget cuts the packaging. Invented text only."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from seedgraph.concepts import estimate as cest
from seedgraph.library import store
from seedgraph.work import estimate as west
from seedgraph.work import routine
from seedgraph.work.tasks_concepts import load_manifest

from ._world import World, a, m

T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
WORDS = ["cog", "lever", "flange", "pivot", "spindle", "ratchet", "detent", "bushing", "sprocket", "gimbal",
         "pawl", "cam", "spline", "collet", "arbor", "toggle"]


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def w(lib):
    return World(lib)


def busy_world(w: World, n: int = 16) -> None:
    """Invented senses whose claim texts overlap in pairs and triples."""
    for i in range(n):
        x, y, z = WORDS[i % 16], WORDS[(i + 1) % 16], WORDS[(i + 3) % 16]
        w.note(f"lp_b{i:02d}", [
            a(f"{x} {y} condition variant", f"Invented: the {x} and the {y} move together in widget {i}."),
            a(f"{x} {y} restriction", f"Invented: the {x} restricts the {y} for gadget {i}."),
            m(f"{y} {z} estimator", f"Invented: estimate the {y} from the {z} for widget {i}."),
        ])
    with w.lib.write() as conn:
        for i in range(0, n - 1, 2):
            conn.execute("INSERT INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, "
                         "created_at) VALUES (?, ?, 'manual_override', 1.0, '2026-01-01')",
                         (f"lp_b{i:02d}", f"lp_b{i + 1:02d}"))


def ro_connection(lib) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{lib.db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def batch_totals(lib, batch_id: str) -> dict:
    pkgs = lib.conn.execute("SELECT est_tokens, text_path FROM work_packages WHERE batch_id = ? ORDER BY ordinal",
                            (batch_id,)).fetchall()
    subjects = [r[0] for r in lib.conn.execute("SELECT subject_id FROM batch_items WHERE batch_id = ? ORDER BY ordinal",
                                               (batch_id,))]
    items = [it for sid in subjects for it in load_manifest(lib, batch_id, sid)["items"]]
    labels = json.loads((Path(lib.root).parent / "work" / batch_id / "labels.json").read_text(encoding="utf-8"))
    assert len(json.dumps(labels)) < 1000  # the agent-facing file stays small; manifests live apart
    return {"packages": len(pkgs), "text_tokens": sum(p[0] for p in pkgs),
            "pairs": sum(1 for it in items if it["kind"] == "pair"),
            "sense_checks": sum(1 for it in items if it["kind"] == "sense_check")}


@pytest.mark.parametrize("task, options", [
    ("bridge", {"families": "assumption,method"}),
    ("merge", {}),
])
def test_the_dry_run_writes_nothing_and_matches_the_created_batch(lib, w, task, options):
    busy_world(w)
    ro = ro_connection(lib)
    try:  # never refreshed and read-only: any write would raise
        dry = cest.estimate(ro, task, options)
    finally:
        ro.close()
    assert dry.packages >= 1 and dry.pairs >= 1
    assert dry.agent_tokens_low < dry.agent_tokens_high and dry.text_tokens > 0
    hook = west.estimate(lib, f"concepts:{task}", options=options)  # the C3 estimate hook
    view = routine.create_batch(lib, f"concepts:{task}", options=options, now=T0)
    got = batch_totals(lib, view.batch_id)
    for est in (dry, hook):
        assert (est.packages, est.text_tokens, est.pairs, est.sense_checks) == (
            got["packages"], got["text_tokens"], got["pairs"], got["sense_checks"])
        assert list(est.subjects) == [i.subject_id for i in view.items]


def test_the_dry_run_reports_families_and_citation_distance_shares(lib, w):
    busy_world(w)
    dry = cest.estimate(lib.conn, "bridge", {"families": "assumption"})
    assert set(dry.per_family) == {"assumption"}
    assert dry.per_family["assumption"]["pairs"] == dry.pairs
    assert dry.distance_share and abs(sum(dry.distance_share.values()) - 1.0) < 1e-3
    assert list(dry.distance_share)[0] == "inf"  # farthest first
    d = dry.as_dict()
    assert d["candidate_source"] == "keyword" and d["task"] == "concepts:bridge"


def test_a_token_budget_stops_packaging_beyond_it(lib, w):
    busy_world(w)
    full = cest.estimate(lib.conn, "bridge", {"families": "assumption,method"})
    budget = full.agent_tokens_high // 2
    cut = cest.estimate(lib.conn, "bridge", {"families": "assumption,method", "budget_tokens": budget})
    assert cut.items_total == full.items_total
    assert 0 < cut.items_packaged < full.items_packaged
    assert cut.cut_at == cut.items_packaged + 1
    assert cut.agent_tokens_high <= budget
    assert "by priority fit" in cut.as_dict()["cut"]
    view = routine.create_batch(lib, "concepts:bridge",
                                options={"families": "assumption,method", "budget_tokens": budget}, now=T0)
    got = batch_totals(lib, view.batch_id)
    assert got["pairs"] + got["sense_checks"] == cut.items_packaged
    assert got["packages"] == cut.packages


def test_bad_options_are_refused(lib, w):
    with pytest.raises(cest.OptionError, match="family"):
        cest.estimate(lib.conn, "bridge", {"families": "widgets"})
    with pytest.raises(cest.OptionError, match="budget_tokens"):
        cest.estimate(lib.conn, "bridge", {"budget_tokens": 0})
    with pytest.raises(cest.OptionError, match="unknown concept task"):
        cest.estimate(lib.conn, "tidy", {})
