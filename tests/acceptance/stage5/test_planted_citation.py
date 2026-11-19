"""§10.2: a planted unread citation is caught, through the real MCP tools on a copy of the
library. Always on an invented library; against the real home when SEEDGRAPH_STAGE5_HOME is
set. The live variant (ask the agent to cite a named span without opening it) is on the
researcher's checklist."""

from __future__ import annotations

from . import probes


def check(result: dict) -> None:
    assert result["never_returned"] == [probes.FLAG_NOT_READ]
    assert result["search_only"] == [probes.FLAG_SEARCH_ONLY]
    assert result["anchor_only"] == [probes.FLAG_NOT_READ]
    assert result["read"] == [] and result["read_next_action"] is None
    assert result["unknown_code"] == "unknown_evidence" and result["unknown_stored"] == 0


def test_planted_citation_on_an_invented_library(invented_home, work_home):
    check(probes.planted_citation(invented_home, work_home))


def test_planted_citation_on_a_copy_of_the_real_library(real_home, work_home):
    check(probes.planted_citation(real_home, work_home))
