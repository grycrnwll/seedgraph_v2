"""The concept task instructions (stage 4 design §3.3-§3.4): link criteria that keep mere
topical relatedness out of links, the usage wording that matches the CLI, the note on
repeated claims, and a version bump that does not strand batches created earlier.
Invented text only."""

from __future__ import annotations

import pytest

from seedgraph.concepts import instructions as ins
from seedgraph.library import store
from seedgraph.work import tasks
from seedgraph.work.store import WorkError


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


def _flat(text: str) -> str:
    return " ".join(text.split()).lower()


@pytest.mark.parametrize("text", [ins.MERGE, ins.BRIDGE], ids=["merge", "bridge"])
def test_pair_instructions_give_a_concrete_link_test_and_prefer_unrelated_over_a_weak_link(text):
    flat = _flat(text)
    assert "when unsure between merge and link, answer **link**" in flat
    assert "when unsure between link and unrelated, answer **unrelated**" in flat
    assert "swapping one label for the other" in flat and "a careful reader would object" in flat
    assert "merely on related topics" in flat and "one is an input to the other" in flat
    assert flat.count("**unrelated (topical)**") >= 2


@pytest.mark.parametrize("text", [ins.MERGE, ins.BRIDGE], ids=["merge", "bridge"])
def test_usage_wording_matches_the_cli_and_mcp_and_is_optional(text):
    flat = _flat(text)
    assert "--usage-file" in flat and "`usage` (mcp)" in flat and "otherwise skip it" in flat
    assert "pass `usage` (your token usage) when you submit" not in flat


@pytest.mark.parametrize("text", [ins.MERGE, ins.BRIDGE], ids=["merge", "bridge"])
def test_instructions_say_a_claim_may_repeat_under_different_refs(text):
    flat = _flat(text)
    assert "may repeat a claim under different refs" in flat and "cite whichever ref the item lists" in flat


def test_instructions_carry_their_version_and_the_version_was_bumped():
    assert ins.INSTRUCTIONS_VERSION == "1.1.0"
    for text in (ins.MERGE, ins.BRIDGE, ins.NAME_LITERATURES):
        assert f"instructions version {ins.INSTRUCTIONS_VERSION}" in text.lower()


def test_new_concept_batches_use_the_new_version_and_earlier_batches_still_load(lib):
    assert tasks.get_task(lib, "concepts:bridge").version == ins.INSTRUCTIONS_VERSION
    assert tasks.task_for_batch(lib, "concepts:bridge", "1.0.0").version == "1.0.0"
    with pytest.raises(WorkError, match="create a new batch"):
        tasks.task_for_batch(lib, "concepts:bridge", "0.9.0")
