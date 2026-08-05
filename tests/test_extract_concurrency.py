"""Build C chunk 6 — reader concurrency lever (``extract notes --concurrency``), OFFLINE.

Design D6: an opt-in ThreadPoolExecutor fans out over work_ids at the WORKS
level; each worker owns its own ``Session(h.engine)`` and its own read-only
cache connection; the shared :class:`BudgetState` serializes spend accumulation
and the stop latch on an internal ``threading.Lock`` (``add_spend`` /
``is_stopped``), so no update is ever lost and a budget stop in one worker
prevents later dispatches on every worker. The sequential path (``--concurrency
1`` / flag omitted, the default) must remain byte-identical to the pre-change
behavior.

Covered here: the lock-level no-lost-update hammer; latch semantics through the
new methods (whole-note + chunked per-paper form); a concurrency-2 CLI run over
4 works with a barrier-ish fake that FORCES two workers in-flight together (all
terminal rows written, ``budget.spent_usd`` equals the exact DB sum); the
cross-thread stop latch (2 dispatched, 2 ``skipped_budget``, zero extra LLM
calls); and two concurrency-1 goldens (same-project dry-run byte-identity and a
cross-project id-normalized real-run comparison against the default path).
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.config.models import LlmBudget
from seedgraph.extraction import runner as runner_mod
from seedgraph.extraction.runner import BudgetState
from seedgraph.llm.backend import FakeLLMBackend
from seedgraph.project import service

from test_phase_4 import MD, _add_doc, good_note_dict

cli = CliRunner()


# --------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------

def _seed_project(slug: str, n: int, *, distinct_markdown: bool = False):
    """Project with ``n`` included, bridged-markdown works; return (h, [work_ids]).

    ``distinct_markdown`` appends a per-work tail paragraph so every work gets
    its OWN content-addressed markdown_id — concurrency tests use this so two
    workers never race on one markdown's span space (the canned quotes still
    anchor: the shared body is unchanged).
    """
    h = service.create_project(slug)
    wids = []
    for i in range(n):
        md_text = MD + (f"\nDistinct tail paragraph {i}.\n" if distinct_markdown else "")
        wid, _md = _add_doc(h, f"{slug}_{i}", md_text, doi=f"10.1/{slug}{i}")
        wids.append(wid)
    return h, wids


def _write_project_yaml(slug: str, mutate) -> None:
    proj_yaml = Path(os.environ["SEEDGRAPH_HOME"]) / "projects" / slug / "project.yaml"
    data = yaml.safe_load(proj_yaml.read_text()) if proj_yaml.exists() else {}
    data = data or {}
    mutate(data)
    proj_yaml.write_text(yaml.safe_dump(data))


def _route_to_anthropic(slug: str) -> None:
    """Route note_extraction to the priced anthropic profile (nonzero est cost)."""
    def mutate(data):
        data.setdefault("llm", {}).setdefault("routes", {})["note_extraction"] = {
            "task_type": "note_extraction",
            "preferred_profile": "anthropic_api_default",
        }
    _write_project_yaml(slug, mutate)


def _arm_stop_budget(slug: str) -> None:
    """Any completed run (even $0 local) trips the stop latch: 0.0 >= 0.0."""
    def mutate(data):
        data["budget"] = {"per_run_soft_limit_usd": 0.0, "stop_on_budget_exceeded": True}
    _write_project_yaml(slug, mutate)


@dataclass
class _BarrierFake(FakeLLMBackend):
    """FakeLLMBackend that holds each call at a 2-party barrier (barrier-ish slow
    fake): two pool workers are forced in-flight INSIDE ``complete`` together, so
    the subsequent ``add_spend`` calls interleave — a lost update or an unlocked
    latch shows up deterministically. ``max_active`` records the peak overlap."""

    barrier: threading.Barrier | None = None
    max_active: int = 0
    _active: int = 0
    _gauge_lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self):
        self._gauge_lock = threading.Lock()

    def complete(self, system_prompt, user_prompt, **kwargs):
        with self._gauge_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.barrier is not None:
                try:
                    self.barrier.wait(timeout=10)
                except threading.BrokenBarrierError:
                    pass  # bounded fallback: assertions below still catch it
            return super().complete(system_prompt, user_prompt, **kwargs)
        finally:
            with self._gauge_lock:
                self._active -= 1


def _run_rows(h) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(h.db_path))
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT work_id, run_status, estimated_cost FROM extraction_runs"
        ).fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# BudgetState: locked accounting (unit)
# --------------------------------------------------------------------------

def test_budget_state_add_spend_no_lost_updates():
    """8 threads x 200 add_spend(0.001) with a tiny switch interval: runs and
    spent_usd are EXACT — the internal lock loses no read-modify-write update."""
    budget = LlmBudget()
    state = BudgetState()
    n_threads, n_iter = 8, 200
    start = threading.Barrier(n_threads)

    def hammer():
        start.wait(timeout=10)
        for _ in range(n_iter):
            state.add_spend(0.001, budget)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # force aggressive thread interleaving
    try:
        threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)

    assert state.runs == n_threads * n_iter
    assert state.spent_usd == pytest.approx(n_threads * n_iter * 0.001, abs=1e-9)
    assert not state.is_stopped()


def test_budget_state_latch_methods_whole_note_and_chunked_forms():
    """add_spend trips the same latches as the pre-lock accounting: per-run on the
    work's own cost (whole-note form) or on the per-PAPER total (chunked form),
    monthly on prior + cumulative; is_stopped reads the latch."""
    budget = LlmBudget(per_run_soft_limit_usd=0.5, stop_on_budget_exceeded=True)
    state = BudgetState()
    assert not state.is_stopped()
    state.add_spend(0.1, budget)
    assert not state.is_stopped()
    state.add_spend(0.6, budget)  # 0.6 >= 0.5
    assert state.is_stopped()
    assert state.stop_reason == "per_run_soft_limit_usd exceeded"
    assert state.spent_usd == pytest.approx(0.7)
    assert state.runs == 2

    # chunked per-paper form: a small chunk cost with a big paper total latches.
    chunked = BudgetState()
    chunked.add_spend(0.1, budget, per_run_cost=0.9)
    assert chunked.is_stopped()
    assert chunked.spent_usd == pytest.approx(0.1)  # accrual is the CHUNK cost

    # monthly: prior DB spend + accrual (Stage C wiring preserved).
    monthly_budget = LlmBudget(monthly_soft_limit_usd=1.0, stop_on_budget_exceeded=True)
    seeded = BudgetState(prior_month_spent_usd=0.9)
    seeded.add_spend(0.2, monthly_budget)
    assert seeded.is_stopped()
    assert seeded.stop_reason == "monthly_soft_limit_usd exceeded"

    # no stop_on_budget_exceeded -> accounting only, latch never trips.
    off = BudgetState()
    off.add_spend(99.0, LlmBudget(per_run_soft_limit_usd=0.5))
    assert not off.is_stopped()


# --------------------------------------------------------------------------
# CLI --concurrency 2: all terminal rows + exact budget sum (forced interleave)
# --------------------------------------------------------------------------

def test_concurrency_two_all_rows_written_exact_budget_sum(monkeypatch):
    """4 works, 2 workers, a barrier forcing pairwise in-flight overlap, a PRICED
    route (anthropic profile, FakeLLMBackend override — no network): every work
    gets its terminal extraction_runs row and the shared BudgetState's spent_usd
    equals the exact sum of the recorded per-run costs — no lost update."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    h, wids = _seed_project("conc4", 4, distinct_markdown=True)
    _route_to_anthropic("conc4")

    fake = _BarrierFake(response=good_note_dict(), barrier=threading.Barrier(2))
    monkeypatch.setattr(runner_mod, "_BACKEND_OVERRIDE", fake)

    created: list[BudgetState] = []

    class _Recording(BudgetState):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(runner_mod, "BudgetState", _Recording)

    r = cli.invoke(app, ["extract", "notes", "conc4", "--concurrency", "2"])
    assert r.exit_code == 0, (r.output, r.exception)

    rows = _run_rows(h)
    assert {row["work_id"] for row in rows} == set(wids)  # all 4 terminal rows
    assert all(row["run_status"] == "success" for row in rows)
    for wid in wids:
        assert f"{wid}\tsuccess" in r.output  # per-completion echo, self-identified

    expected = sum(row["estimated_cost"] for row in rows)
    assert expected > 0  # the priced route makes a lost update visible
    assert len(created) == 1  # the one shared CLI BudgetState
    budget = created[0]
    assert budget.spent_usd == pytest.approx(expected, abs=1e-12)
    assert budget.runs == 4
    assert len(fake.calls) == 4
    assert fake.max_active == 2  # the barrier really forced two workers in-flight


# --------------------------------------------------------------------------
# CLI --concurrency 2: stop latch honored ACROSS threads
# --------------------------------------------------------------------------

def test_concurrency_stop_latch_prevents_later_dispatches(monkeypatch):
    """per_run_soft_limit_usd=0.0 + stop_on_budget_exceeded: the first pair of
    works (held in-flight together by the barrier) both dispatch and both trip
    the latch on completion; the remaining works are NEVER dispatched — each
    worker re-checks the shared latch before its next work -> skipped_budget."""
    h, wids = _seed_project("conclatch", 4, distinct_markdown=True)
    _arm_stop_budget("conclatch")

    fake = _BarrierFake(response=good_note_dict(), barrier=threading.Barrier(2))
    monkeypatch.setattr(runner_mod, "_BACKEND_OVERRIDE", fake)

    r = cli.invoke(app, ["extract", "notes", "conclatch", "--concurrency", "2"])
    assert r.exit_code == 0, (r.output, r.exception)

    assert len(fake.calls) == 2  # exactly the first pair; later works never dispatch

    rows = _run_rows(h)
    statuses = sorted(row["run_status"] for row in rows)
    assert statuses == ["skipped_budget", "skipped_budget", "success", "success"]
    assert {row["work_id"] for row in rows} == set(wids)  # 4 terminal rows still
    assert r.output.count("skipped_budget") >= 2
    assert "per_run_soft_limit_usd exceeded" in r.output


# --------------------------------------------------------------------------
# concurrency-1 goldens: the sequential path is unchanged
# --------------------------------------------------------------------------

def _fake_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_mod, "_BACKEND_OVERRIDE", FakeLLMBackend(response=good_note_dict())
    )


def test_concurrency_one_dry_run_byte_identical_to_default(monkeypatch):
    """Same project, deterministic dry-run: omitting the flag and passing
    ``--concurrency 1`` produce byte-identical output."""
    _fake_llm(monkeypatch)
    _seed_project("concgold", 3)

    r_default = cli.invoke(app, ["extract", "notes", "concgold", "--dry-run"])
    r_one = cli.invoke(
        app, ["extract", "notes", "concgold", "--dry-run", "--concurrency", "1"]
    )
    assert r_default.exit_code == 0, r_default.output
    assert r_one.exit_code == 0, r_one.output
    assert r_default.output == r_one.output


_ID_RE = re.compile(r"\b(work|note|extr)_[0-9a-f]{32}\b")


def test_concurrency_one_real_run_matches_default_golden(monkeypatch):
    """Two identically-shaped projects, one real run each (default path vs
    ``--concurrency 1``): after normalizing minted ids, the outputs are
    byte-identical — the sequential golden (worklist header, per-work success
    lines with claims/spans/message/running_cost) is preserved."""
    _fake_llm(monkeypatch)
    _seed_project("goldseq", 2)
    _seed_project("goldone", 2)

    r_default = cli.invoke(app, ["extract", "notes", "goldseq"])
    r_one = cli.invoke(app, ["extract", "notes", "goldone", "--concurrency", "1"])
    assert r_default.exit_code == 0, r_default.output
    assert r_one.exit_code == 0, r_one.output

    norm_default = _ID_RE.sub(r"\1_X", r_default.output)
    norm_one = _ID_RE.sub(r"\1_X", r_one.output)
    assert norm_default == norm_one
    # pin the golden line shape itself (not just mutual agreement).
    assert "worklist: 2 works ordered by citation in-degree; cap=none" in norm_one
    assert "work_X\tsuccess\tnote=note_X" in norm_one
    assert "running_cost=$0.000000" in norm_one


def test_concurrency_rejects_zero():
    """min=1: --concurrency 0 is a usage error, never a silent no-work pool."""
    r = cli.invoke(app, ["extract", "notes", "whatever", "--concurrency", "0"])
    assert r.exit_code != 0
