"""End to end on a synthetic library (design §11 P4): create -> two workers claim different
packages -> a lookalike-character quote gets the right feedback -> resubmit -> current
notes; held-back papers are listed and never packaged."""

from __future__ import annotations

import json
from pathlib import Path

from seedgraph.notes import current_note_id
from seedgraph.work import routine, store
from seedgraph.work.claims import WorkerRef

from .routine_support import MU_QUOTE, good_note, make_paper, open_lib
from .test_routine_items import GARBLED


def test_two_workers_lookalike_feedback_resubmit_current_notes(tmp_path):
    lib = open_lib(tmp_path)
    try:
        a, b = make_paper(lib, 1), make_paper(lib, 2)
        garbled = make_paper(lib, 3, text=GARBLED, sections=[])
        view = routine.create_batch(lib, "notes", paper_ids=[a.paper_id, b.paper_id, garbled.paper_id],
                                    client="claude")
        assert [i.state for i in view.items] == ["queued", "queued", "held_back"]
        assert view.state.state == "waiting"
        assert view.as_dict()["commands"]["claude_code"].endswith(view.batch_id)

        w1, w2 = WorkerRef("interactive", "w1", client="claude"), WorkerRef("interactive", "w2", client="codex")
        m1 = routine.claim_next(lib, view.batch_id, w1).message
        m2 = routine.claim_next(lib, view.batch_id, w2).message
        assert m1["packet_id"] != m2["packet_id"]
        assert {m1["paper_id"], m2["paper_id"]} == {a.paper_id, b.paper_id}
        assert routine.batch_view(lib, view.batch_id).state.state == "working"

        # w1 copies the micro sign instead of the Greek mu: feedback hands back the exact slice
        lookalike = good_note(mu_quote=MU_QUOTE.replace("μ", "µ"))
        Path(m1["result_path"]).write_bytes(json.dumps(lookalike, ensure_ascii=False).encode("utf-8"))
        r1 = routine.submit(lib, m1["packet_id"], m1["claim_id"], note_path=m1["result_path"])
        assert r1.status == "needs_revision"
        (failed,) = r1.failed_quotes
        assert failed["reason"] == "lookalike_characters" and failed["markdown_slice"] == MU_QUOTE

        r2 = routine.submit(lib, m2["packet_id"], m2["claim_id"], note=good_note(), usage={"input_tokens": 900})
        assert r2.status == "accepted" and r2.paper_status == "done"

        fixed = good_note(mu_quote=failed["markdown_slice"])
        Path(m1["result_path"]).write_bytes(json.dumps(fixed, ensure_ascii=False).encode("utf-8"))
        r1b = routine.submit(lib, m1["packet_id"], m1["claim_id"], note_path=m1["result_path"])
        assert r1b.status == "accepted" and r1b.paper_status == "done"

        assert current_note_id(lib.conn, m1["paper_id"]) == r1b.note_id
        assert current_note_id(lib.conn, m2["paper_id"]) == r2.note_id
        assert current_note_id(lib.conn, garbled.paper_id) is None
        final = routine.batch_view(lib, view.batch_id)
        assert final.state.state == "finished"
        assert final.state.items == {"done": 2, "held_back": 1}
        tokens = lib.conn.execute("SELECT r.input_tokens, r.provider FROM structured_notes n JOIN extraction_runs r "
                                  "ON r.extraction_run_id = n.extraction_run_id WHERE n.note_id = ?",
                                  (r2.note_id,)).fetchone()
        assert tuple(tokens) == (900, "codex")
        kinds = [e.kind for e in store.list_events(lib.conn, view.batch_id)]
        assert kinds.count("claimed") == 2 and kinds.count("paper_done") == 2
        assert kinds.index("claimed") < kinds.index("submitted")
        claims = store.list_claims(lib.conn, batch_id=view.batch_id)
        assert len({c.packet_id for c in claims}) == len(claims) == 2
        assert routine.list_batches(lib, active_only=True) == []
    finally:
        lib.close()
