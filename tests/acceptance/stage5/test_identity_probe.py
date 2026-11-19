"""§10.4: session identity. A parent plus two named helpers gives the parent's record with
helper-tagged reads and no cross-ownership; the same without names is one connection stream
marked "session told apart by connection only"; a restart with a name keeps one record while an
unnamed stream splits; the ``_meta`` keys (never values) and the client are kept. The live
variant with real Claude Code subagents is on the researcher's checklist."""

from __future__ import annotations

from seedgraph.records.manage import NOTE_CONNECTION_ONLY

from . import probes


def check(out: dict) -> None:
    assert out["named_records"] == 2 and not out["cross_owned"]
    assert out["parent_events"] == [("", "search"), ("a", "search"), ("b", "paper_read")]
    assert out["other_events"] == 1
    assert out["unnamed_records"] == 1 and out["unnamed_events"] == 2
    assert NOTE_CONNECTION_ONLY in out["unnamed_notes"]
    assert out["restart_named_records"] == 1 and out["restart_named_events"] == 2
    assert out["restart_unnamed_streams"] == 2
    assert "probe_marker" in out["meta"]["keys"] and not out["meta"]["values_stored"]
    assert out["meta"]["client"]


def test_identity_probe_on_an_invented_library(invented_home, work_home):
    check(probes.identity_probe(invented_home, work_home))


def test_identity_probe_on_a_copy_of_the_real_library(real_home, work_home):
    check(probes.identity_probe(real_home, work_home))
