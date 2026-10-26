"""The consultation skill states the three researcher-edit rules (stage 3 design §4.3, §11 U6)."""

from __future__ import annotations

from seedgraph.agent_setup import register


def test_read_skill_states_the_researcher_edit_rules():
    text = " ".join(register.skill_text("seedgraph-v3").split())
    lower = text.lower()
    for field in ("trust", "trust_note", "edited_by_researcher", "original", "researcher_notes"):
        assert f"`{field}" in text, field
    assert "follow the researcher's version" in lower and "say so" in lower
    assert "never present a researcher assertion as the paper's" in lower
    assert 'say "researcher"' in lower
    assert "project" not in lower
