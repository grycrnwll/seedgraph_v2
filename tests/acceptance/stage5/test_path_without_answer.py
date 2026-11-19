"""§10.3: calls with no ``record_answer`` show "in progress", then "path without answer" once
the clock passes PATH_IDLE, or at once when the connection closed; in Records and on Home.
The live variant (stop an agent mid-search) is on the researcher's checklist."""

from __future__ import annotations

from . import probes


def check(out: dict) -> None:
    assert out["now"] == {"idle-1432": "in_progress", "closed-1432": "path_without_answer"}
    assert out["later"] == {"idle-1432": "path_without_answer", "closed-1432": "path_without_answer"}
    assert out["list_filter_later"] == ["closed-1432", "idle-1432"]
    assert len([t for t in out["feed_now"] if t.startswith("path without answer")]) == 1
    assert len([t for t in out["feed_later"] if t.startswith("path without answer")]) == 2
    assert len([t for t in out["home_later"] if t.startswith("path without answer")]) == 2


def test_path_without_answer_on_an_invented_library(invented_home, work_home):
    check(probes.path_without_answer(invented_home, work_home))


def test_path_without_answer_on_a_copy_of_the_real_library(real_home, work_home):
    out = probes.path_without_answer(real_home, work_home)
    # the real library may already hold paths without answer; look at the probe's own streams only
    assert out["now"] == {"idle-1432": "in_progress", "closed-1432": "path_without_answer"}
    assert out["later"] == {"idle-1432": "path_without_answer", "closed-1432": "path_without_answer"}
