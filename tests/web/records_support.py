"""An invented records library for the R5 screen tests (stage 5 design §6.1-6.4): two noted
papers in two graphs and four records - one spanning both graphs, one library-wide with a
flag, one path without answer from a second client, one still in progress. Built through
the R2 helpers in ``tests/records/_world.py``; textbook terms only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from records._world import T0, Agent, span_hit, standard

from seedgraph.library import graphs
from seedgraph.library.store import open_library_at

DID_GRAPH, ATE_GRAPH = "did-reading", "matching"
NOW = T0 + timedelta(hours=2)
JUDGEMENT = "I think the invented pre-trends look parallel, but the corpus does not say so outright."


@dataclass
class RecordsFixture:
    home: Path
    did: object
    ate: object
    both: str          # searched both graphs: "spans 2 graphs"
    flagged: str       # library-wide, cites the DiD paper, one citation not read
    no_answer: str     # a codex connection stream that went idle
    in_progress: str   # a search a minute before NOW
    claim_id: str      # the cited parallel-trends claim
    span_id: str       # the cited (read) parallel-trends span
    unread_span: str   # the cited span no reply returned


def make_graph(lib, slug: str, name: str, *members) -> None:
    g = graphs.create_graph(lib, slug, display_name=name)
    try:
        for p in members:
            graphs.set_membership(g, p.paper_id, "member")
    finally:
        g.close()


def build(home: Path) -> RecordsFixture:
    lib = open_library_at(home / "library")
    try:
        w, did, ate = standard(lib)
        make_graph(lib, DID_GRAPH, "DiD reading", did)
        make_graph(lib, ATE_GRAPH, "Matching estimators", ate)
        sp, serial = w.span(did, "parallel"), w.span(did, "serial")
        claim = did.claims["assumptions[0]"]

        b = Agent(lib, "mcp-bbbb0002", client="codex")
        b.search("matching estimators", hits=[span_hit(w.span(ate, "matching"), ate)])

        a = Agent(lib, "mcp-aaaa0001", name="parallel-trends-1432", start=T0 + timedelta(minutes=5))
        a.search("parallel trends", graph=DID_GRAPH)
        a.search("propensity score", graph=ATE_GRAPH)
        a.paper_show(ate)
        both = a.answer([{"paper_id": ate.paper_id}], question="Which graphs discuss two invented designs?",
                        answer="Both invented graphs do.").record_id
        a.search("parallel trends", hits=[span_hit(sp, did), span_hit(serial, did)])
        a.paper_read(did, *did.at("parallel"))
        a.paper_show(did)
        flagged = a.answer([{"span_id": sp, "why": "the assumption in the paper's words"},
                            {"claim_id": claim}, {"span_id": serial}],
                           question="Is parallel trends plausible in the invented county study?",
                           answer="The paper assumes parallel trends between treated and control counties.",
                           judgement=JUDGEMENT).record_id

        c = Agent(lib, "mcp-cccc0003", start=NOW - timedelta(minutes=2))
        c.search("serial correlation")
        rows = lib.conn.execute("SELECT record_id, opened_at FROM traversal_records WHERE state = 'open' "
                                "ORDER BY opened_at").fetchall()
        no_answer, in_progress = rows[0][0], rows[1][0]
        return RecordsFixture(home, did, ate, both, flagged, no_answer, in_progress, claim, sp, serial)
    finally:
        lib.close()


def clock(at: datetime = NOW):
    return lambda: at


def with_lib(home: Path, fn):
    lib = open_library_at(home / "library")
    try:
        return fn(lib)
    finally:
        lib.close()
