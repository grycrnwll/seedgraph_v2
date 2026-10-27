"""The protocol seam for non-paper tasks (stage 4 design §3.1, slice C3).

A fake task registered here claims subjects that are not papers (invented "widget
groups"). It defines the three optional hooks - ``select`` (choose the subjects from
options), ``subject_label`` (a human label) and ``estimate`` (the dry run) - plus
``task_options`` (its own batch options with defaults). It cites refs and writes no
quotes, so no anchor preview runs. Paper tasks are covered by the rest of tests/work.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from seedgraph.work import estimate as estimate_mod
from seedgraph.work import routine, tasks
from seedgraph.work.claims import WorkerRef
from seedgraph.work.store import WorkError

from .conftest import T0, at

FAKE = "widgets"
SAVED: list[dict[str, Any]] = []


@dataclass(frozen=True)
class FakeEstimate:
    groups: int
    text_tokens: int
    agent_tokens_low: int
    agent_tokens_high: int


class FakeWidgetTask:
    """Invented subjects ``wg_01..wg_NN``; options ``groups`` (how many) and ``family``."""

    name = FAKE
    version = "0.1.0"
    subject_kind = "widget_group"
    allowed_workers = frozenset({"interactive", "background"})
    task_options = {"groups": 3, "family": "gadget"}

    def __init__(self, lib):
        self.lib = lib

    # --- the C3 hooks ---------------------------------------------------------------------
    def select(self, lib, options: Mapping[str, Any]) -> list[str]:
        return [f"wg_{i:02d}" for i in range(1, int(options["groups"]) + 1)]

    def subject_label(self, lib, subject_id: str) -> str:
        n = int(subject_id.split("_")[1])
        return f"gadget pairs {50 * (n - 1) + 1}-{50 * n}"

    def estimate(self, lib, options: Mapping[str, Any]) -> FakeEstimate:
        g = int(options["groups"])
        return FakeEstimate(groups=g, text_tokens=100 * g, agent_tokens_low=150 * g, agent_tokens_high=270 * g)

    # --- the stage 2 seam -----------------------------------------------------------------
    def resolve(self, lib, query):  # never called: select replaces paper selection
        raise AssertionError("resolve must not be called for a select task")

    def eligibility(self, lib, subject_id, options):
        return tasks.Eligibility("queued")

    def plan(self, lib, subject_id, options):
        text = f"# {options['family']} group {subject_id}\n\n1. c1 vs c2\n"
        return [tasks.PackagePlan(1, 1, None, None, text, 12)]

    def instructions(self, lib, *, has_parts):
        return tasks.Instructions("# Invented widget instructions\n", "{}")

    def check(self, lib, package, payload):
        if payload.get("verdict") not in ("same", "different"):
            return tasks.CheckResult(errors=["verdict must be same or different"])
        return tasks.CheckResult()

    def save(self, conn, item, parts: Sequence[tasks.AcceptedPart], ctx: tasks.SaveContext):
        SAVED.append({"subject_id": item.subject_id, "verdict": parts[0].payload["verdict"],
                      "family": ctx.options["family"]})
        return tasks.SaveResult(result_ref=f"decision_{item.subject_id}")


@pytest.fixture(autouse=True)
def fake_task(monkeypatch):
    SAVED.clear()
    monkeypatch.setitem(tasks._BY_NAME, FAKE, FakeWidgetTask)  # noqa: SLF001 - test registration
    tasks._ensure_builtins()  # noqa: SLF001
    yield
    SAVED.clear()


W1 = WorkerRef("interactive", "w1", client="claude")


def test_a_select_task_chooses_its_subjects_from_options(lib):
    view = routine.create_batch(lib, FAKE, options={"groups": 2, "claim_ttl_s": 900}, now=T0)
    assert [(i.subject_kind, i.subject_id, i.state) for i in view.items] == [
        ("widget_group", "wg_01", "queued"), ("widget_group", "wg_02", "queued")]
    assert view.options["groups"] == 2 and view.options["family"] == "gadget"  # default filled in
    assert view.options["claim_ttl_s"] == 900


def test_a_fake_non_paper_task_runs_create_claim_submit_finish(lib):
    view = routine.create_batch(lib, FAKE, options={"groups": 2, "family": "widget"}, client="claude", now=T0)
    for n, verdict in enumerate(("same", "different"), start=1):
        reply = routine.claim_next(lib, view.batch_id, W1, now=at(n))
        assert reply.status == "claimed"
        msg = reply.message
        assert msg["subject_kind"] == "widget_group" and msg["subject_id"] == f"wg_{n:02d}"
        assert msg["subject_label"] == f"gadget pairs {50 * (n - 1) + 1}-{50 * n}"
        assert msg["title"] == msg["subject_label"] and msg["paper_id"] is None and msg["year"] is None
        res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"verdict": verdict}, now=at(n + 0.5))
        assert res.status == "accepted" and res.paper_status == "done"
        assert res.quotes_found == 0 and res.failed_quotes == []  # no anchor preview
        assert res.note_id == f"decision_wg_{n:02d}"
    assert SAVED == [{"subject_id": "wg_01", "verdict": "same", "family": "widget"},
                     {"subject_id": "wg_02", "verdict": "different", "family": "widget"}]
    done = routine.batch_view(lib, view.batch_id, now=at(5))
    assert done.state.state == "finished"
    assert [i.subject_label for i in done.items] == ["gadget pairs 1-50", "gadget pairs 51-100"]
    assert all(i["subject_label"] for i in done.as_dict()["items"])


def test_an_invalid_answer_keeps_the_claim(lib):
    view = routine.create_batch(lib, FAKE, options={"groups": 1}, now=T0)
    msg = routine.claim_next(lib, view.batch_id, W1, now=at(1)).message
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"verdict": "maybe"}, now=at(2))
    assert res.status == "invalid" and res.errors == ["verdict must be same or different"]
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note={"verdict": "same"}, now=at(3))
    assert res.status == "accepted" and res.paper_status == "done"


@pytest.mark.parametrize("kw", [{"paper_ids": ["lp_01"]}, {"query": "held"}])
def test_a_select_task_refuses_paper_arguments(lib, kw):
    with pytest.raises(WorkError, match="chooses its own subjects"):
        routine.create_batch(lib, FAKE, now=T0, **kw)


def test_a_select_task_records_a_graph_and_passes_it_on(lib, monkeypatch):
    seen = {}

    def select(self, lib, options):
        seen.update(options)
        return ["wg_01"]

    monkeypatch.setattr(FakeWidgetTask, "select", select)
    view = routine.create_batch(lib, FAKE, graph="invented_graph", now=T0)
    assert view.graph == "invented_graph" and seen["graph"] == "invented_graph"


def test_task_options_are_refused_for_tasks_that_do_not_declare_them(lib):
    with pytest.raises(WorkError, match="unknown batch option"):
        routine.create_batch(lib, "notes", paper_ids=["lp_01"], options={"groups": 2}, now=T0)
    with pytest.raises(WorkError, match="unknown batch option"):
        routine.create_batch(lib, FAKE, options={"colour": "red"}, now=T0)


def test_selection_limits_count_subjects(lib, monkeypatch):
    assert routine.MAX_SUBJECTS == 1000 and routine.MAX_PAPERS == routine.MAX_SUBJECTS
    monkeypatch.setattr(routine, "MAX_SUBJECTS", 2)
    with pytest.raises(WorkError, match="at most 2 subjects, not 3"):
        routine.create_batch(lib, FAKE, now=T0)
    with pytest.raises(WorkError, match="selected no subjects"):
        routine.create_batch(lib, FAKE, options={"groups": 0}, now=T0)


def test_duplicate_selected_subjects_are_refused(lib, monkeypatch):
    monkeypatch.setattr(FakeWidgetTask, "select", lambda self, lib, options: ["wg_01", "wg_01"])
    with pytest.raises(WorkError, match="distinct"):
        routine.create_batch(lib, FAKE, now=T0)


def test_the_estimate_hook_is_the_dry_run(lib):
    est = estimate_mod.estimate(lib, FAKE, options={"groups": 4})
    assert est == FakeEstimate(groups=4, text_tokens=400, agent_tokens_low=600, agent_tokens_high=1080)
    assert estimate_mod.estimate(lib, FAKE).groups == 3  # the task's default options
    with pytest.raises(WorkError, match="chooses its own subjects"):
        estimate_mod.estimate(lib, FAKE, ["lp_01"])
    with pytest.raises(WorkError, match="unknown batch option"):
        estimate_mod.estimate(lib, FAKE, options={"colour": "red"})


def test_a_task_without_an_estimate_hook_or_whole_paper_shape_has_no_estimate(lib):
    with pytest.raises(WorkError, match="no estimate"):
        estimate_mod.estimate(lib, "metadata", ["lp_01"])


def test_subject_labels_fall_back_to_the_subject_id_when_the_task_version_changed(lib, monkeypatch):
    view = routine.create_batch(lib, FAKE, options={"groups": 1}, now=T0)
    monkeypatch.setattr(FakeWidgetTask, "version", "0.2.0")
    again = routine.batch_view(lib, view.batch_id, now=at(1))  # must not raise
    assert [i.subject_label for i in again.items] == ["wg_01"]
