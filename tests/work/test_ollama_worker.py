"""The Ollama worker loop (design §6, §11 P6): claims metadata packages through the
routine, asks a (fake, loopback) Ollama for the JSON, submits; invalid output gets one
retry then ``give_back``; a model or connection failure releases and stops, never
pauses. Invented text only."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from llm.fake_ollama import FakeOllama, chat_reply, closed_port_url
from seedgraph.library import attention
from seedgraph.work import ollama_worker, routine
from seedgraph.work.claims import WorkerRef

from .routine_support import filler, make_paper, open_lib

MODEL = "invented-model"
COVER = ("# A Made-Up Study of Gadget Markets\n\nAda Inventor and Bo Example\n\n"
         "Journal of Invented Results, 2019.\n\n" + filler("cover", 400) + "\n")
GOOD = {"title": "A Made-Up Study of Gadget Markets", "authors": ["Ada Inventor", "Bo Example"],
        "year": 2019, "venue": "Journal of Invented Results", "doi": None, "arxiv": None}


@pytest.fixture
def lib(tmp_path):
    library = open_lib(tmp_path)
    yield library
    library.close()


@pytest.fixture
def fake():
    server = FakeOllama().start()
    yield server
    server.close()


def _untitled(lib, n):
    paper = make_paper(lib, n, text=COVER.replace("Gadget", f"Gadget {n}"), sections=[])
    with lib.write() as conn:
        conn.execute("UPDATE papers SET title = NULL, year = NULL WHERE paper_id = ?", (paper.paper_id,))
    attention.raise_item(lib, "metadata_needed", {"paper_id": paper.paper_id}, paper_id=paper.paper_id,
                         detail="title unknown")
    return paper


def _metadata_batch(lib, n=1):
    papers = [_untitled(lib, i) for i in range(1, n + 1)]
    return routine.create_batch(lib, "metadata", paper_ids=[p.paper_id for p in papers]).batch_id, papers


def _run(lib, batch_id, fake, **kw):
    return ollama_worker.run_ollama_worker(lib, batch_id, model=MODEL, base_url=fake.url, **kw)


def test_the_worker_fills_metadata_and_stops_when_finished(lib, fake):
    batch_id, (paper,) = _metadata_batch(lib)
    fake.script(chat_reply(json.dumps(GOOD)))
    report = _run(lib, batch_id, fake)
    assert report.stopped == "finished" and report.saved == 1 and report.error is None
    row = lib.conn.execute("SELECT title, title_source, year FROM papers WHERE paper_id = ?",
                           (paper.paper_id,)).fetchone()
    assert tuple(row) == (GOOD["title"], "ollama", 2019)
    (chat,) = fake.chats
    assert chat["model"] == MODEL and isinstance(chat["format"], dict)
    assert "Metadata task" in chat["messages"][0]["content"]  # instructions.md is the system prompt
    assert chat["messages"][1]["content"].startswith("# A Made-Up Study")  # the package text
    assert len(chat["messages"][1]["content"]) < 3200  # the cover only, not the whole markdown
    sub = lib.conn.execute("SELECT model, usage_json FROM package_submissions").fetchone()
    assert sub[0] == MODEL and json.loads(sub[1]) == {"input_tokens": 120, "output_tokens": 30}


def _state(lib, batch_id):
    return routine.batch_view(lib, batch_id).state


def test_ollama_is_offered_only_metadata_packages(lib, fake):
    paper = make_paper(lib, 1)
    notes = routine.create_batch(lib, "notes", paper_ids=[paper.paper_id]).batch_id
    report = _run(lib, notes, fake)
    assert report.stopped == "not_allowed" and report.claimed == 0
    assert fake.requests == []
    assert _state(lib, notes).packages == {"open": 1}


def test_a_connection_failure_releases_and_never_pauses(lib):
    batch_id, _ = _metadata_batch(lib, 2)
    report = ollama_worker.run_ollama_worker(lib, batch_id, model=MODEL, base_url=closed_port_url(), timeout=5)
    assert report.stopped == "ollama_error" and "unreachable" in report.error
    assert len(report.released) == 1 and report.saved == 0
    st = _state(lib, batch_id)
    assert st.packages == {"open": 2} and st.live_claims == 0
    assert not st.paused and st.pause_kind is None and st.resets_at is None
    # another worker can take the released package straight away
    assert routine.claim_next(lib, batch_id, WorkerRef("interactive", "i1")).status == "claimed"


def test_a_model_error_releases_without_a_retry(lib, fake):
    batch_id, _ = _metadata_batch(lib)
    fake.script((404, {"error": "model not found"}))
    report = _run(lib, batch_id, fake)
    assert report.stopped == "ollama_error" and report.retries == 0 and len(fake.chats) == 1
    assert _state(lib, batch_id).packages == {"open": 1}


def test_invalid_json_is_retried_once_then_saved(lib, fake):
    batch_id, _ = _metadata_batch(lib)
    fake.script(chat_reply("Sorry, I cannot find a title."), chat_reply(json.dumps(GOOD)))
    report = _run(lib, batch_id, fake)
    assert report.stopped == "finished" and report.saved == 1 and report.retries == 1
    assert "previous answer was rejected" in fake.chats[1]["messages"][1]["content"]
    outcomes = [r[0] for r in lib.conn.execute("SELECT outcome FROM package_submissions ORDER BY submission_no")]
    assert outcomes == ["invalid", "accepted"]


def test_invalid_output_twice_releases_and_stops(lib, fake):
    batch_id, _ = _metadata_batch(lib, 2)
    fake.script(chat_reply("not json at all"), chat_reply(json.dumps({"title": 7, "authors": "Ada"})))
    report = _run(lib, batch_id, fake)
    assert report.stopped == "invalid_output" and report.retries == 1 and len(fake.chats) == 2
    assert "title: must be a string" in report.error
    st = _state(lib, batch_id)
    assert st.packages == {"open": 2} and st.live_claims == 0 and st.resets_at is None
    assert not st.paused and st.pause_kind is None


def test_an_empty_answer_counts_as_invalid_output(lib, fake):
    batch_id, _ = _metadata_batch(lib)
    fake.script(chat_reply(""), chat_reply(json.dumps(GOOD)))
    report = _run(lib, batch_id, fake)
    assert report.saved == 1 and report.retries == 1


def test_max_packages_stops_the_loop(lib, fake):
    batch_id, _ = _metadata_batch(lib, 3)
    fake.script(chat_reply(json.dumps(GOOD)), chat_reply(json.dumps(GOOD)))
    report = _run(lib, batch_id, fake, max_packages=2)
    assert report.stopped == "max_packages" and report.saved == 2
    assert _state(lib, batch_id).packages == {"open": 1, "saved": 2}


def test_a_paused_batch_stops_the_worker(lib, fake):
    batch_id, _ = _metadata_batch(lib)
    routine.report_limit(lib, batch_id, WorkerRef("background", "b1"), resets_at="2099-01-01T00:00:00Z")
    report = _run(lib, batch_id, fake)
    assert report.stopped == "paused" and fake.requests == []


def test_an_unexpected_error_releases_and_reraises(lib):
    batch_id, _ = _metadata_batch(lib)

    class Broken:
        def complete_json(self, *a, **k):
            raise RuntimeError("invented bug")

    with pytest.raises(RuntimeError):
        ollama_worker.run_ollama_worker(lib, batch_id, model=MODEL, client=Broken())
    assert _state(lib, batch_id).packages == {"open": 1}


def test_the_worker_does_not_import_the_hosted_layer():
    tree = ast.parse(Path(ollama_worker.__file__).read_text(encoding="utf-8"))
    mods = {("." * n.level) + (n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    hosted = ("executor", "routing", "secrets", "backend", "providers", "profiles", "cost", "usage")
    assert not [m for m in mods if any(h in m for h in hosted)]
    assert "..llm.ollama" in mods
