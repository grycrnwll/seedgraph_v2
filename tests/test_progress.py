"""Unit tests for the shared progress abstraction (``seedgraph.progress``).

Fully offline: the echo path is captured with ``capsys``; the emit path uses a real
``ensure_run`` + ``read_events`` under the autouse isolated ``SEEDGRAPH_HOME``. No
network, no LLM, no new dependency.
"""

from __future__ import annotations

import pytest

from seedgraph.progress import Progress, conversion_summary, make_emitter
from seedgraph.project import service
from seedgraph.run import ensure_run, read_events


# --------------------------------------------------------------------------
# Progress.step — the stdout heartbeat
# --------------------------------------------------------------------------

def test_step_echoes_n_of_m(capsys):
    prog = Progress(2, "works")
    prog.step("alpha", edges=1)
    prog.step("beta", edges=3)
    out = capsys.readouterr().out
    assert "1/2 works — alpha  edges=1" in out
    assert "2/2 works — beta  edges=3" in out


def test_emit_none_is_stdout_only(capsys):
    prog = Progress(1, "works")  # echo defaults True, emit=None
    seq = prog.step("only")
    assert seq == -1  # emit=None never touches the filesystem, returns -1
    assert "1/1 works — only" in capsys.readouterr().out


def test_echo_false_is_silent(capsys):
    prog = Progress(1, "works", echo=False)
    assert prog.step("silent") == -1
    assert capsys.readouterr().out == ""


def test_zero_total_and_overshoot(capsys):
    prog = Progress(0, "items")
    prog.step()  # 1/0
    prog.step()  # 2/0 — done may exceed total
    out = capsys.readouterr().out
    assert "1/0 items" in out
    assert "2/0 items" in out
    assert prog.done == 2


def test_context_manager_never_swallows():
    with pytest.raises(ValueError):
        with Progress(1) as p:
            p.step()
            raise ValueError("boom")


# --------------------------------------------------------------------------
# Progress.step — the events.jsonl emit path (identical append_event shape)
# --------------------------------------------------------------------------

def test_step_emits_progress_event_with_done_total():
    service.create_project("progemit")
    run_id = ensure_run("progemit")
    emit = make_emitter("progemit", run_id, phase="cite")
    prog = Progress(3, "works", emit=emit, event="work", echo=False)

    seq0 = prog.step("w1", edges=2)
    seq1 = prog.step("w2", edges=5)
    assert seq0 >= 0 and seq1 == seq0 + 1

    events = read_events("progemit", run_id)
    work = [e for e in events if e["event"] == "work"]
    assert len(work) == 2
    assert work[0]["data"]["done"] == 1
    assert work[0]["data"]["total"] == 3
    assert work[0]["data"]["edges"] == 2
    assert work[1]["data"]["done"] == 2
    # The emitted message equals the (would-be) echoed line.
    assert work[0]["message"] == "1/3 works — w1  edges=2"


def test_make_emitter_stamps_run_id():
    service.create_project("progrid")
    run_id = ensure_run("progrid")
    emit = make_emitter("progrid", run_id, phase="cite")
    assert emit.run_id == run_id


# --------------------------------------------------------------------------
# conversion_summary — the pure corpus-bar read-model
# --------------------------------------------------------------------------

def test_conversion_summary_math():
    rows = [{"has_markdown": True}, {"has_markdown": True}, {"has_markdown": False}]
    queue = {
        "w1": {"state": "queued", "detail": ""},
        "w2": {"state": "converting", "detail": ""},
        "w3": {"state": "failed", "detail": "boom"},
    }
    s = conversion_summary(rows, queue)
    assert s["converted"] == 2
    assert s["queued"] == 1
    assert s["converting"] == 1
    assert s["failed"] == 1
    assert s["pending"] == 2  # queued + converting
    assert s["total"] == 4  # converted + pending (failed excluded from the denominator)
    assert s["done_pct"] == 50


def test_conversion_summary_empty_is_zero():
    s = conversion_summary([], {})
    assert s == {
        "converted": 0, "pending": 0, "queued": 0, "converting": 0,
        "failed": 0, "total": 0, "done_pct": 0,
    }


def test_conversion_summary_all_converted_is_full():
    rows = [{"has_markdown": True}, {"has_markdown": True}]
    s = conversion_summary(rows, {})
    assert s["total"] == 2 and s["converted"] == 2 and s["done_pct"] == 100
