"""Stage 5 slice R3 (design §8): the consultation skill's "Record your answer" section and the
work skill's `model` wording."""

from __future__ import annotations

from seedgraph.agent_setup import register


def _flat(name: str) -> str:
    return " ".join(register.skill_text(name).split())


def test_consultation_skill_has_a_record_your_answer_section():
    raw = register.skill_text("seedgraph-v3")
    assert "## Record your answer" in raw
    text = _flat("seedgraph-v3")
    lower = text.lower()
    assert "record_answer(question, answer, evidence, judgement?, session?)" in text
    assert "once per answer, at the end" in lower
    assert "do not record answers that did not use the library" in lower
    assert "`parallel-trends-1432`" in text and "pass it as `session` on every seedgraph call" in lower
    assert "`<yours>/a`" in text and "if you are a subagent and were given no name, choose your own" in lower
    assert "cite only what you opened" in lower and "`paper_read`" in text and "`concept_show`" in text
    assert "a search hit alone counts as found, not read" in lower
    assert "`answer`" in text and "`judgement`" in text
    assert "read them and record again" in lower
    # "Find your footing" and "Ground a claim" stay
    assert "## Find your footing" in raw and "## Ground a claim" in raw


def test_work_skill_asks_for_model_on_every_submit():
    lower = _flat("seedgraph-v3-work").lower()
    assert ("pass `model` on every submit: your model id as your client states it (for example in your system "
            "prompt). spot-check accuracy is reported per model.") in lower
    assert "pass `model` if your client shows it" not in lower
