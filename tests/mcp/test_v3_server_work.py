"""v3 MCP protocol tools (design §7.2, §1): one in-memory call per tool, over an invented
library. The routine's own behaviour is tested in tests/work; these pin the adapters:
arguments reach the routine, replies are its JSON, refusals are structured errors."""

from __future__ import annotations

import json
from pathlib import Path

from v3_mcp_support import GRAPH, call, err, ok, home, seeded, server  # noqa: F401 (fixtures)
from work.routine_support import MU_QUOTE, good_note


def _batch(server, seeded, **extra):
    args = {"task": "notes", "paper_ids": [seeded.p2.paper_id, seeded.p3.paper_id], **extra}
    return ok(call(server, "batch_create", args))


def test_batch_create_by_ids_and_commands(server, seeded):
    body = _batch(server, seeded, options={"claim_ttl_s": 900})
    assert body["task_type"] == "notes" and body["state"] == "waiting"
    assert [i["state"] for i in body["items"]] == ["queued", "queued"]
    assert body["options"]["claim_ttl_s"] == 900
    assert body["commands"]["claude_code"] == f"/seedgraph-v3-work {body['batch_id']}"
    assert "seedgraph_v3" in body["commands"]["codex"]


def test_batch_create_by_graph_and_query(server, seeded):
    by_graph = ok(call(server, "batch_create", {"task": "notes", "graph": GRAPH}))
    assert by_graph["graph"] == GRAPH
    # p1 already has a fresh note (preserved); the frontier paper is not held in the graph
    states = {i["subject_id"]: i["state"] for i in by_graph["items"]}
    assert states[seeded.p1.paper_id] == "preserved" and states[seeded.p2.paper_id] == "queued"
    by_query = ok(call(server, "batch_create", {"task": "notes",
                                                "query": f"held in graph {GRAPH} without current note"}))
    assert [i["subject_id"] for i in by_query["items"]] == [seeded.p2.paper_id]


def test_batch_create_refusals(server, seeded):
    body = err(call(server, "batch_create", {"task": "notes"}))
    assert body["code"] == "work_refused" and body["next_action"]
    assert err(call(server, "batch_create", {"task": "notes", "paper_ids": [seeded.p2.paper_id],
                                             "options": {"bogus": 1}}))["code"] == "work_refused"


def test_claim_submit_revision_accept(server, seeded):
    batch = _batch(server, seeded)
    claim = ok(call(server, "work_claim", {"batch_id": batch["batch_id"], "worker": "w1"}))
    assert claim["status"] == "claimed"
    pkg = claim["package"]
    assert Path(pkg["text_path"]).is_file() and Path(pkg["instructions_path"]).is_file()
    assert pkg["paper_id"] == seeded.p2.paper_id

    status = ok(call(server, "batch_status", {"batch_id": batch["batch_id"]}))
    assert status["state"] == "working" and status["workers"] == ["w1"]

    lookalike = good_note(mu_quote=MU_QUOTE.replace("μ", "µ"))
    Path(pkg["result_path"]).write_bytes(json.dumps(lookalike, ensure_ascii=False).encode("utf-8"))
    rev = ok(call(server, "work_submit", {"packet_id": pkg["packet_id"], "claim_id": pkg["claim_id"],
                                          "note_path": pkg["result_path"]}))
    assert rev["status"] == "needs_revision"
    (failed,) = rev["failed_quotes"]
    assert failed["reason"] == "lookalike_characters" and failed["markdown_slice"] == MU_QUOTE

    done = ok(call(server, "work_submit", {"packet_id": pkg["packet_id"], "claim_id": pkg["claim_id"],
                                           "note": good_note(mu_quote=failed["markdown_slice"]),
                                           "usage": {"input_tokens": 1200}, "model": "invented-model"}))
    assert done["status"] == "accepted" and done["paper_status"] == "done" and done["note_id"]


def test_submit_inline_string_note_and_invalid(server, seeded):
    batch = _batch(server, seeded)
    pkg = ok(call(server, "work_claim", {"batch_id": batch["batch_id"]}))["package"]
    bad = ok(call(server, "work_submit", {"packet_id": pkg["packet_id"], "claim_id": pkg["claim_id"],
                                          "note": "no json here"}))
    assert bad["status"] == "invalid" and bad["errors"]
    good = ok(call(server, "work_submit", {"packet_id": pkg["packet_id"], "claim_id": pkg["claim_id"],
                                           "note": json.dumps(good_note(), ensure_ascii=False)}))
    assert good["status"] == "accepted"


def test_wrong_claim_id_is_structured_refusal(server, seeded):
    batch = _batch(server, seeded)
    pkg = ok(call(server, "work_claim", {"batch_id": batch["batch_id"]}))["package"]
    body = err(call(server, "work_submit", {"packet_id": pkg["packet_id"], "claim_id": "cl_wrong",
                                            "note": good_note()}))
    assert body["code"] == "claim_unknown"
    assert body["detail"] == {"packet_id": pkg["packet_id"], "claim_id": "cl_wrong"}
    assert body["next_action"]


def test_release_give_back_and_paper_problem(server, seeded):
    batch = _batch(server, seeded)
    pkg = ok(call(server, "work_claim", {"batch_id": batch["batch_id"]}))["package"]
    back = ok(call(server, "work_release", {"packet_id": pkg["packet_id"], "claim_id": pkg["claim_id"],
                                            "reason": "give_back"}))
    assert back["state"] == "waiting"
    again = ok(call(server, "work_claim", {"batch_id": batch["batch_id"]}))["package"]
    assert again["packet_id"] == pkg["packet_id"]
    failed = ok(call(server, "work_release", {"packet_id": again["packet_id"], "claim_id": again["claim_id"],
                                              "reason": "paper_problem", "message": "the text is unreadable"}))
    assert failed["item_counts"]["failed"] == 1
    assert err(call(server, "work_release", {"packet_id": pkg["packet_id"], "claim_id": pkg["claim_id"],
                                             "reason": "lost"}))["code"] == "work_refused"


def test_stop_continue_and_paused_claim(server, seeded):
    batch_id = _batch(server, seeded)["batch_id"]
    stopped = ok(call(server, "batch_stop", {"batch_id": batch_id, "mode": "now"}))
    assert stopped["state"] == "stopped"
    paused = ok(call(server, "work_claim", {"batch_id": batch_id}))
    assert paused["status"] == "paused" and paused["next_action"]
    assert ok(call(server, "batch_continue", {"batch_id": batch_id}))["state"] == "waiting"
    assert err(call(server, "batch_stop", {"batch_id": batch_id, "mode": "later"}))["code"] == "invalid_argument"


def test_report_limit_releases_this_workers_claims(server, seeded):
    batch_id = _batch(server, seeded)["batch_id"]
    ok(call(server, "work_claim", {"batch_id": batch_id, "worker": "w9"}))
    body = ok(call(server, "batch_report_limit", {"batch_id": batch_id, "worker": "w9",
                                                  "resets_at": "2026-11-01T10:00:00+00:00",
                                                  "message": "usage limit reached"}))
    assert body["state"] == "usage_limit" and body["resets_at"].startswith("2026-11-01")
    assert body["live_claims"] == 0


def test_batch_list_and_status_paging(server, seeded):
    batch_id = _batch(server, seeded)["batch_id"]
    listed = ok(call(server, "batch_list"))["batches"]
    assert [b["batch_id"] for b in listed] == [seeded.noted_batch, batch_id]
    assert ok(call(server, "batch_list", {"active_only": True}))["batches"][0]["batch_id"] == batch_id
    page = ok(call(server, "batch_status", {"batch_id": batch_id, "offset": 1, "limit": 1}))
    assert page["item_total"] == 2 and [i["ordinal"] for i in page["items"]] == [2]
    assert err(call(server, "batch_status", {"batch_id": "b_missing"}))["code"] == "work_refused"


def test_unexpected_failures_are_structured(server, seeded, monkeypatch):
    import sqlite3

    from seedgraph.work import routine

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(routine, "list_batches", locked)
    assert err(call(server, "batch_list"))["code"] == "library_busy"

    def broken(*a, **k):
        raise RuntimeError("invented failure")

    monkeypatch.setattr(routine, "list_batches", broken)
    body = err(call(server, "batch_list"))
    assert body["code"] == "internal_error" and "invented failure" in body["message"]
