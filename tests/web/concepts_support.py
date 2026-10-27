"""Fixture library for the concept page tests (stage 4 slice C7). Invented text only; every
home is under pytest's tmp.

``build(home)`` writes into ``<home>/library``:

- ``lp_c1``..``lp_c3`` with assumption claims; ``widget overlap`` (lp_c1, lp_c2) and
  ``gadget overlap`` (lp_c3) joined by an applied agent merge (batch ``b_c7``, quotes);
- ``doohickey rank`` (lp_c1, lp_c2) linked to ``gizmo rank`` (lp_c3) by an applied agent
  link with a note and a quote from each side;
- ``sprocket condition`` (lp_c2) vs ``gizmo rank``: a queued agent ``unsure`` item, and
  ``widget overlap`` vs ``doohickey rank``: a queued agent merge (reach);
- ``cog rank`` split by the researcher into two glossed senses (a chooser page);
- ``unanchored estimator`` (method): only agent-inferred claims, so it is unsupported;
- graph ``cwidgets`` with lp_c1 and lp_c2;
- a frozen review sample of the applied agent merge and link.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from concepts._embed import index_fts
from concepts._world import World, a, m

from seedgraph.concepts import refresh as rf
from seedgraph.concepts import researcher as rs
from seedgraph.concepts.model import SenseKey
from seedgraph.library import graphs
from seedgraph.library.store import open_library_at

GRAPH = "cwidgets"
BATCH = "b_c7"


@dataclass
class Fixture:
    home: Path
    merge_id: str = ""
    link_id: str = ""
    queued_merge: str = ""
    queued_unsure: str = ""
    sample_id: str = ""
    cog_claims: tuple[str, ...] = ()


def _agent(lib, w, kind, x, y, *, state, reason=None, note=None, quotes=None):
    did = w.decide(kind, x, y, by="agent", state=state, family="assumption")
    qa, qb = quotes or (None, None)
    with lib.write() as conn:
        conn.execute("UPDATE concept_decisions SET veto_reason = ?, batch_id = ?, note = ?, confidence = 'medium', "
                     "quote_a_json = ?, quote_b_json = ? WHERE decision_id = ?", (reason, BATCH, note, qa, qb, did))
    return did


def build(home: Path) -> Fixture:
    fx = Fixture(home=home)
    lib = open_library_at(home / "library")
    try:
        w = World(lib)
        c1 = w.note("lp_c1", [a("widget overlap", "Invented: each widget has a twin widget."),
                              a("doohickey rank", "Invented: the doohickey matrix has full rank."),
                              a("cog rank", "Invented: the cog matrix has full rank.")])
        c2 = w.note("lp_c2", [a("widget overlap", "Invented: widgets overlap in size."),
                              a("doohickey rank", "Invented: doohickeys are of full rank."),
                              a("sprocket condition", "Invented: sprockets turn freely."),
                              a("cog rank", "Invented: cogs are ranked by size.")])
        w.note("lp_c3", [a("gadget overlap", "Invented: each gadget has a twin gadget."),
                         a("gizmo rank", "Invented: the gizmo system is of full rank."),
                         m("unanchored estimator", "Invented: an estimator nobody quoted.", quote=False)])
        index_fts(lib)
        fx.cog_claims = (c1["cog rank"], c2["cog rank"])
        with graphs.create_graph(lib, GRAPH, display_name="Invented concept widgets") as g:
            graphs.set_membership(g, "lp_c1", "member", is_seed=True)
            graphs.set_membership(g, "lp_c2", "member")
        rf.refresh(lib)

        def quotes(x, y):
            return (rs._quote_for(lib.conn, SenseKey("assumption", x), None),
                    rs._quote_for(lib.conn, SenseKey("assumption", y), None))

        fx.merge_id = _agent(lib, w, "merge", "gadget overlap", "widget overlap", state="applied",
                             quotes=quotes("gadget overlap", "widget overlap"))
        fx.link_id = _agent(lib, w, "link", "doohickey rank", "gizmo rank", state="applied",
                            note="Invented: gizmo rank is the older name.",
                            quotes=quotes("doohickey rank", "gizmo rank"))
        fx.queued_unsure = _agent(lib, w, "unrelated", "gizmo rank", "sprocket condition", state="queued",
                                  reason="unsure")
        fx.queued_merge = _agent(lib, w, "merge", "doohickey rank", "widget overlap", state="queued", reason="reach")
        rf.refresh(lib)
        rs.split_label(lib, "assumption", "cog rank",
                       [("full rank of a matrix", [fx.cog_claims[0]]), ("an ordering by size", [fx.cog_claims[1]])])
        fx.sample_id = rs.create_review_sample(lib, n=50, seed=0).sample_id
    finally:
        lib.close()
    return fx
