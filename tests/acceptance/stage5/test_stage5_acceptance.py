"""Stage 5 acceptance checks against a real home (design §10; slice R8). Opt-in:

    SEEDGRAPH_STAGE5_HOME=C:/path/to/home  uv run pytest tests/acceptance/stage5

The library is opened read-only (``mode=ro``); nothing is written. Each check skips while the
researcher's part is pending (questions not asked, sample not marked), and fails only on what is
done and wrong. Assertion messages carry counts and ids only."""

from __future__ import annotations

import json
import sqlite3

import pytest

from seedgraph.library import store as libstore
from seedgraph.records import acceptance as acc
from seedgraph.records import manage


@pytest.fixture
def ro(real_home):
    db = libstore.library_db_path(libstore.library_dir(real_home))
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    yield libstore.Library(root=libstore.library_dir(real_home), conn=conn)
    conn.close()


def test_the_report_renders_from_the_real_home(ro, real_home):
    text = acc.render(acc.gather(ro, real_home))
    for n in range(1, 7):
        assert f"## {n}." in text


def test_the_five_questions_records_exist_and_are_checked(ro, real_home):
    d = acc.stage_dir(real_home)
    if not (d / acc.QUESTIONS_FILE).is_file():
        pytest.skip("no questions.yaml yet (seedgraph-v3 records acceptance-report)")
    questions = acc.load_questions(d / acc.QUESTIONS_FILE)
    linked = [q for q in questions if q.record]
    if len(linked) < acc.QUESTION_COUNT:
        pytest.skip(f"{len(linked)} of {acc.QUESTION_COUNT} questions linked to a record")
    missing = [q.id for q in linked if _absent(ro, q.record)]
    assert not missing, f"records not in the library for {missing}"
    checklist = (d / acc.CHECKLIST_FILE).read_text(encoding="utf-8")
    ticks = acc.parse_ticks(checklist)
    open_ = [q.id for q in linked if ticks.get(q.record, (0, 1))[0] != ticks.get(q.record, (0, 1))[1]]
    if open_:
        pytest.skip(f"checklist not finished for {open_}")
    cross = json.loads((d / acc.CROSSCHECK_FILE).read_text(encoding="utf-8")) if (d / acc.CROSSCHECK_FILE).is_file() else {}
    dirty = [q.id for q in linked if q.client != "codex" and not cross.get(q.record, {}).get("ok")]
    assert not dirty, f"no clean transcript cross-check for {dirty}"
    assert any(q.client == "codex" for q in linked), "no question was asked from Codex"


def _absent(lib, record_id: str) -> bool:
    try:
        manage.get_record(lib, record_id)
    except manage.RecordNotFound:
        return True
    return False


def test_the_spotcheck_sample_is_fully_marked_and_the_fix_left_the_figure_alone(ro, real_home):
    d = acc.spotcheck_data(ro)
    if d is None:
        pytest.skip("no spot-check sample yet")
    if not d["all_marked"]:
        pytest.skip(f"spot-check pending: {d['marked']} of {d['items']} marked")
    assert d["items"] == d["size"], f"sample holds {d['items']} of {d['size']} asked"
    if d["unchanged_after_fix"] is None:
        pytest.skip("no fix yet: run `records acceptance-report`, fix one wrong claim, run it again")
    assert d["unchanged_after_fix"], "the accuracy figure moved after a fix"
