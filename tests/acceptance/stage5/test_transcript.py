"""The transcript reader and the cross-check (design §10.1), on an invented Claude Code
transcript. The structure matches what Claude Code writes: ``assistant`` lines with ``tool_use``
blocks, ``user`` lines with ``tool_result`` blocks, ``api-request-blob`` lines repeating both,
and subagent transcripts in ``<session>/subagents/``. Nothing here is a real transcript."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from seedgraph.records import acceptance as acc

T0 = datetime(2026, 11, 4, 14, 0, 0, tzinfo=timezone.utc)
PFX = "mcp__seedgraph_v3__"


def ts(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


class Writer:
    def __init__(self) -> None:
        self.lines: list[dict] = []
        self.n = 0

    def call(self, tool, args, result, *, t, error=False, blocks=False, sidechain=False, server="seedgraph_v3"):
        self.n += 1
        uid = f"toolu_{self.n:04d}"
        use = {"type": "tool_use", "id": uid, "name": f"mcp__{server}__{tool}" if "__" not in tool else tool,
               "input": args}
        self.lines.append({"type": "assistant", "timestamp": ts(t), "isSidechain": sidechain,
                           "message": {"role": "assistant", "content": [use]}})
        self.lines.append({"type": "api-request-blob", "message": {"content": [use]}})  # a repeat, to be ignored
        body = result if isinstance(result, str) else json.dumps(result)
        content = [{"type": "text", "text": body}] if blocks else body
        self.lines.append({"type": "user", "timestamp": ts(t + 0.5), "isSidechain": sidechain,
                           "message": {"role": "user", "content": [
                               {"type": "tool_result", "tool_use_id": uid, "is_error": error, "content": content}]}})
        self.lines.append({"type": "api-request-blob", "message": {"content": [
            {"type": "tool_result", "tool_use_id": uid, "content": [{"type": "text", "text": body}]}]}})

    def write(self, path):
        path.write_text("\n".join(json.dumps(x) for x in self.lines) + "\nnot json at all\n", encoding="utf-8")


def make(tmp_path):
    """Two answered questions: q1 (named session, a helper in a subagent file), q2 (unnamed)."""
    w = Writer()
    w.call("search", {"query": "parallel trends", "session": "pt-1400"}, {"hits": []}, t=1)
    w.call("paper_show", {"paper_id": "lp_a", "session": "pt-1400"}, {"paper_id": "lp_a"}, t=3, blocks=True)
    w.call("graph_list", {}, {"graphs": []}, t=4)                                         # not logged
    w.call("paper_list", {"graph": "g"}, {"papers": []}, t=5)                              # no query: not logged
    w.call("paper_read", {"paper_id": "lp_a", "start": 0, "end": 50, "session": "pt-1400"},
           {"error": {"code": "invalid_argument"}}, t=6, error=True)                       # failed
    w.call("search", {"query": "x", "session": "Bad Name"}, {"error": {"code": "invalid_session"}}, t=7, error=True)
    w.call("search", {"query": "other server"}, {"hits": []}, t=8, server="another_server")
    w.call("record_answer", {"question": "q", "answer": "a", "evidence": [], "session": "pt-1400"},
           {"record_id": "tr_one", "flags": []}, t=10)
    w.call("search", {"query": "matching"}, {"hits": []}, t=20)
    w.call("concept_show", {"handle": "h1"}, {"handle": "h1"}, t=22)
    w.call("record_answer", {"question": "q2", "answer": "a", "evidence": []}, {"record_id": "tr_two"}, t=30)
    main = tmp_path / "session-1.jsonl"
    w.write(main)
    sub = Writer()
    sub.n = 100
    sub.call("search", {"query": "helper query", "session": "pt-1400/a"}, {"hits": []}, t=2, sidechain=True)
    d = tmp_path / "session-1" / "subagents"
    d.mkdir(parents=True)
    sub.write(d / "agent-a.jsonl")
    return main


def test_reader_pairs_results_dedupes_blobs_and_reads_subagents(tmp_path):
    tr = acc.read_transcript(make(tmp_path))
    assert tr.files == ("session-1.jsonl", "agent-a.jsonl")
    assert len(tr.calls) == 11                                    # blobs and the other server's call are out
    assert len({c.tool_use_id for c in tr.calls}) == 11
    by_tool = [c.tool for c in tr.calls]
    assert by_tool.count("search") == 4 and "graph_list" in by_tool
    helper = next(c for c in tr.calls if c.input.get("query") == "helper query")
    assert helper.sidechain and helper.ok and helper.stream == "pt-1400/a"
    failed = [c for c in tr.calls if not c.ok]
    assert sorted(c.tool for c in failed) == ["paper_read", "search"]
    assert [c.when for c in tr.calls] == sorted(c.when for c in tr.calls)


def test_the_server_name_is_an_option(tmp_path):
    tr = acc.read_transcript(make(tmp_path), server="another_server")
    assert [c.tool for c in tr.calls] == ["search"]


def test_logged_calls_keep_only_successful_calls_of_the_nine_tools(tmp_path):
    calls = acc.logged_calls(acc.read_transcript(make(tmp_path)).calls)
    assert [(c.tool, c.input.get("query") or c.input.get("paper_id") or c.input.get("handle")) for c in calls] == [
        ("search", "parallel trends"), ("search", "helper query"), ("paper_show", "lp_a"),
        ("search", "matching"), ("concept_show", "h1")]


def test_a_record_owns_the_calls_between_the_answers(tmp_path):
    calls = acc.read_transcript(make(tmp_path)).calls
    one = acc.record_window(calls, "tr_one")
    assert [c.input.get("query") or c.input.get("paper_id") for c in one] == [
        "parallel trends", "helper query", "lp_a"]
    two = acc.record_window(calls, "tr_two")
    assert [c.tool for c in two] == ["search", "concept_show"]
    try:
        acc.record_window(calls, "tr_nope")
    except acc.AcceptanceError as exc:
        assert "tr_nope" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown record must be refused")


def ev(seq, tool, stream=None, **args):
    return {"seq": seq, "tool": tool, "args": args, "stream": stream, "at": ts(seq)}


def test_a_matching_log_is_clean_even_when_streams_interleave(tmp_path):
    calls = acc.record_window(acc.read_transcript(make(tmp_path)).calls, "tr_one")
    events = [ev(1, "search", "pt-1400", query="parallel trends"), ev(2, "paper_show", "pt-1400", paper_id="lp_a"),
              ev(3, "search", "pt-1400/a", query="helper query")]            # logged in a different order
    cc = acc.cross_check(calls, events)
    assert cc.ok and (cc.transcript, cc.events) == (3, 3)


def test_missing_extra_and_out_of_order_events_are_reported(tmp_path):
    calls = acc.record_window(acc.read_transcript(make(tmp_path)).calls, "tr_one")
    missing = acc.cross_check(calls, [ev(1, "search", "pt-1400", query="parallel trends"),
                                      ev(2, "search", "pt-1400/a", query="helper query")], unlogged_calls=1)
    assert not missing.ok and missing.missing == (("paper_show", "lp_a", "pt-1400"),) and missing.unlogged_calls == 1
    extra = acc.cross_check(calls, [ev(1, "search", "pt-1400", query="parallel trends"),
                                    ev(2, "paper_show", "pt-1400", paper_id="lp_a"),
                                    ev(3, "search", "pt-1400/a", query="helper query"),
                                    ev(4, "paper_show", "pt-1400", paper_id="lp_b")])
    assert not extra.ok and extra.extra == (("paper_show", "lp_b", "pt-1400"),)
    swapped = acc.cross_check(calls, [ev(1, "paper_show", "pt-1400", paper_id="lp_a"),
                                      ev(2, "search", "pt-1400", query="parallel trends"),
                                      ev(3, "search", "pt-1400/a", query="helper query")])
    assert not swapped.ok and swapped.order == ("pt-1400",) and not swapped.missing
    assert "FAIL" in acc.render_cross_check("tr_one", swapped)
    different_query = acc.cross_check(calls, [ev(1, "search", "pt-1400", query="something else"),
                                              ev(2, "paper_show", "pt-1400", paper_id="lp_a"),
                                              ev(3, "search", "pt-1400/a", query="helper query")])
    assert len(different_query.missing) == 1 and len(different_query.extra) == 1


def test_a_transcript_path_that_is_missing_is_refused(tmp_path):
    try:
        acc.read_transcript(tmp_path / "nope.jsonl")
    except acc.AcceptanceError:
        pass
    else:  # pragma: no cover
        raise AssertionError("missing transcript must be refused")
