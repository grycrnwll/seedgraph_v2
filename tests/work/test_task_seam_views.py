"""The non-paper seam seen from outside (stage 4 design §3.1, slice C3): the batch page
shows subject labels (never a paper link), and the MCP and CLI batch views work for a
fake non-paper task. The fake task lives in ``test_task_seam``."""

from __future__ import annotations

import json
import warnings

import anyio
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from typer.testing import CliRunner

from seedgraph.cli import app as cli_app
from seedgraph.library.store import open_library_at
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef

from .test_task_seam import FAKE, fake_task  # noqa: F401 (the autouse registration fixture)

PORT = 8765


@pytest.fixture
def home(isolated_home):
    open_library_at(isolated_home / "library").close()
    return isolated_home


def _with_lib(home, fn):
    lib = open_library_at(home / "library")
    try:
        return fn(lib)
    finally:
        lib.close()


def _page(home, path: str) -> str:
    from seedgraph.web.app import create_app

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from starlette.testclient import TestClient

        with TestClient(create_app(home, port=PORT), base_url=f"http://127.0.0.1:{PORT}",
                        client=("127.0.0.1", 50000), follow_redirects=False) as client:
            resp = client.get(path)
    assert resp.status_code == 200, resp.text
    return resp.text


def test_the_batch_page_shows_subject_labels_not_paper_links(home):
    batch_id = _with_lib(home, lambda lib: routine.create_batch(lib, FAKE, options={"groups": 2}).batch_id)
    html = _page(home, f"/batches/{batch_id}")
    assert "gadget pairs 1-50" in html and "gadget pairs 51-100" in html
    assert "/papers/wg_01" not in html and "Subject" in html
    items = _page(home, f"/batches/{batch_id}/items")
    assert "gadget pairs 51-100" in items
    listing = _page(home, "/batches")
    assert batch_id in listing and "2 selected subjects" in listing


def test_the_batch_view_rows_carry_kind_and_label(home):
    from seedgraph.web.views import batch as batch_view

    batch_id = _with_lib(home, lambda lib: routine.create_batch(lib, FAKE, options={"groups": 1}).batch_id)
    detail = _with_lib(home, lambda lib: batch_view.detail(lib, batch_id))
    row = detail.items[0]
    assert (row.subject_kind, row.paper_id, row.title, row.held) == (
        "widget_group", "wg_01", "gadget pairs 1-50", False)
    assert detail.selection == "1 selected subject" and detail.subject_noun == "subject"


def _mcp(home, name: str, args: dict) -> dict:
    from seedgraph.mcp.server import build_server

    server = build_server(home=home)

    async def _driver():
        async with create_connected_server_and_client_session(server._mcp_server) as client:  # noqa: SLF001
            return await client.call_tool(name, args)

    result = anyio.run(_driver)
    assert result.isError is False, result.content[0].text
    return json.loads(result.content[0].text)


def test_mcp_batch_tools_work_for_a_non_paper_task(home):
    created = _mcp(home, "batch_create", {"task": FAKE, "options": {"groups": 2, "family": "widget"}})
    batch_id = created["batch_id"]
    assert [i["subject_label"] for i in created["items"]] == ["gadget pairs 1-50", "gadget pairs 51-100"]
    claimed = _mcp(home, "work_claim", {"batch_id": batch_id, "worker": "w1"})
    pkg = claimed["package"]
    assert pkg["subject_kind"] == "widget_group" and pkg["subject_label"] == "gadget pairs 1-50"
    submitted = _mcp(home, "work_submit", {"packet_id": pkg["packet_id"], "claim_id": pkg["claim_id"],
                                           "note": {"verdict": "same"}})
    assert submitted["status"] == "accepted" and submitted["paper_status"] == "done"
    status = _mcp(home, "batch_status", {"batch_id": batch_id})
    assert status["item_counts"]["done"] == 1 and status["items"][1]["subject_label"] == "gadget pairs 51-100"
    assert any(b["batch_id"] == batch_id for b in _mcp(home, "batch_list", {})["batches"])


def test_cli_work_commands_work_for_a_non_paper_task(home, tmp_path):
    runner = CliRunner()

    def work(*args) -> dict:
        result = runner.invoke(cli_app, ["work", *args, "--root", str(home)])
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    batch_id = work("create", FAKE)["batch_id"]
    pkg = work("claim", batch_id, "--worker", "w1")["package"]
    assert pkg["subject_label"] == "gadget pairs 1-50"
    note = tmp_path / "answer.json"
    note.write_text(json.dumps({"verdict": "different"}), encoding="utf-8")
    assert work("submit", pkg["packet_id"], "--claim", pkg["claim_id"], "--note-file", str(note))["status"] == "accepted"
    status = work("status", batch_id)
    assert status["item_total"] == 3 and status["items"][0]["state"] == "done"
    assert [i["subject_label"] for i in status["items"]][:2] == ["gadget pairs 1-50", "gadget pairs 51-100"]


def test_paper_batches_still_name_papers(home):
    """A paper batch keeps its paper wording and links (the stage 3 page is unchanged)."""
    from .routine_support import make_paper

    def make(lib):
        paper = make_paper(lib, 1, title="Invented gadget subsidies")
        return paper.paper_id, routine.create_batch(lib, "notes", paper_ids=[paper.paper_id]).batch_id

    pid, batch_id = _with_lib(home, make)
    html = _page(home, f"/batches/{batch_id}")
    assert f'href="/papers/{pid}"' in html and "<th>Paper</th>" in html
    msg = _with_lib(home, lambda lib: routine.claim_next(lib, batch_id, WorkerRef("interactive", "w1")).message)
    assert msg["paper_id"] == pid and msg["title"] == "Invented gadget subsidies"
    assert msg["subject_kind"] == "paper" and msg["subject_label"] == "Invented gadget subsidies"
