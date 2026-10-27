"""The skills' concept sections (stage 4 design §9, slice C9): the read skill explains senses,
ambiguous labels, links vs merges and that researcher decisions win; the work skill covers
concept packages."""

from __future__ import annotations

from seedgraph.agent_setup import register


def _flat(name: str) -> str:
    return " ".join(register.skill_text(name).split())


def test_read_skill_names_the_concept_tools_and_rules():
    text = _flat("seedgraph-v3")
    lower = text.lower()
    for tool in ("concept_search(", "concept_show(", "literatures(", "concept_hubs(", "seedgraph://graphs/"):
        assert tool in text, tool
    assert "label + sense" in lower
    assert "ambiguous label" in lower
    assert '"same words, sense not checked"' in lower and "never treat" in lower
    assert "closely related link" in lower and "merge" in lower
    assert "researcher decisions win" in lower or "researcher decisions always win" in lower
    assert 'say "agent inferred"' in lower
    assert "search_mode" in text
    assert "search concepts before spans" in lower


def test_work_skill_has_a_concept_packages_section():
    text = _flat("seedgraph-v3-work")
    lower = text.lower()
    assert "## concept packages" in register.skill_text("seedgraph-v3-work").lower()
    for word in ("labels.json", "manifest", "claim refs", "never write quotes", "quote_a", "quote_b",
                 "same_sense", "unsure", "`usage`", "instructions.md"):
        assert word.lower() in lower, word
    for verdict in ("merge", "link", "split", "unrelated"):
        assert f"`{verdict}`" in text, verdict
    assert "when unsure between merge and link" in lower


def test_work_skill_usage_wording_matches_the_cli_flag_and_is_optional():
    lower = _flat("seedgraph-v3-work").lower()
    assert "if your client exposes token usage, write it to a json file and pass `--usage-file`" in lower
    assert "`usage` (mcp); otherwise skip it" in lower
    assert "pass `usage` (token counts) on submit when your client shows it" not in lower


def test_work_skill_says_pair_packages_may_repeat_a_claim_and_unsure_links_are_unrelated():
    lower = _flat("seedgraph-v3-work").lower()
    assert "when unsure between link and unrelated, answer unrelated" in lower
    assert "cite whichever ref the item lists" in lower
