"""v3 skills and the setup bundle (design §7.3, §7.4, §8): the two skill texts, installing
them into a given skills or prompts directory, and `plan_setup` showing everything before
writing. Temp directories only."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

from seedgraph.agent_setup import register
from seedgraph.work.commands import work_commands


def _frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    assert m, "missing frontmatter"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines())


def test_skill_frontmatter():
    read = _frontmatter(register.skill_text("seedgraph-v3"))
    work = _frontmatter(register.skill_text("seedgraph-v3-work"))
    assert read["name"] == "seedgraph-v3" and "disable-model-invocation" not in read
    assert work["name"] == "seedgraph-v3-work" and work["disable-model-invocation"] == "true"
    assert len(read["description"]) <= 1024 and len(work["description"]) <= 1024


def test_skills_say_graph_never_project_and_name_v3_tools():
    for name in ("seedgraph-v3", "seedgraph-v3-work"):
        text = register.skill_text(name)
        assert "project" not in text.lower(), name
        assert "seedgraph_v3" in text
        for gone in ("search_spans", "work_read", "extraction_", "ask("):
            assert gone not in text, (name, gone)
    read = register.skill_text("seedgraph-v3")
    for tool in ("graph_list", "graph_show", "paper_list", "search(", "paper_show", "paper_read", "lens_results"):
        assert tool in read
    work = register.skill_text("seedgraph-v3-work")
    for tool in ("work_claim", "work_submit", "note_path", "result_path", "needs_revision", "markdown_slice",
                 "work_release", "batch_report_limit", "batch_status", "$ARGUMENTS"):
        assert tool in work


def test_work_skill_matches_copyable_command():
    """The Claude Code command is `/<skill name> <batch_id>` (design §8)."""
    name = _frontmatter(register.skill_text("seedgraph-v3-work"))["name"]
    assert work_commands("b_x").claude_code == f"/{name} b_x"


def test_unknown_skill_refused():
    with pytest.raises(register.RegisterError):
        register.skill_text("seedgraph-extract")


@pytest.mark.parametrize("layout, rel", [("skills", "seedgraph-v3-work/SKILL.md"),
                                         ("prompts", "seedgraph-v3-work.md")])
def test_plan_and_install_skills(tmp_path, layout, rel):
    tmp_path = tmp_path / "skills_dir"
    planned = register.plan_skills(tmp_path, layout=layout)
    assert [s.action for s in planned] == ["add", "add"]
    assert register.apply_skills(planned) == []
    assert not tmp_path.exists()
    written = register.apply_skills(planned, write=True)
    assert tmp_path / rel in written
    assert (tmp_path / rel).read_text(encoding="utf-8") == register.skill_text("seedgraph-v3-work")
    assert [s.action for s in register.plan_skills(tmp_path, layout=layout)] == ["unchanged", "unchanged"]
    (tmp_path / rel).write_text("old text", encoding="utf-8")
    assert register.plan_skills(tmp_path, layout=layout)[1].action == "replace"


def test_default_targets_only_names_paths(tmp_path):
    tmp_path = tmp_path / "user"
    t = register.default_targets(tmp_path)
    assert t.claude_config == tmp_path / ".claude.json"
    assert t.codex_config == tmp_path / ".codex" / "config.toml"
    assert t.claude_skills_dir == tmp_path / ".claude" / "skills"
    assert t.codex_skills_dir == tmp_path / ".codex" / "skills"
    assert not tmp_path.exists()


def test_plan_setup_both_shows_then_writes(tmp_path):
    tmp_path = tmp_path / "user"
    t = register.default_targets(tmp_path)
    entry = register.server_entry(worktree=Path(r"C:\wt"), home=Path(r"C:\h\.seedgraph-v3"))
    plan = register.plan_setup("both", entry, claude_config=t.claude_config, claude_skills_dir=t.claude_skills_dir,
                               codex_config=t.codex_config, codex_skills_dir=t.codex_prompts_dir,
                               codex_layout="prompts")
    shown = plan.render()
    assert "Nothing has been written" in shown
    assert "claude mcp add --scope user seedgraph_v3" in shown and "[mcp_servers.seedgraph_v3]" in shown
    assert plan.apply() == []
    assert not tmp_path.exists()
    written = plan.apply(write=True)
    assert t.claude_config in written and t.codex_config in written
    assert json.loads(t.claude_config.read_text(encoding="utf-8"))["mcpServers"]["seedgraph_v3"]
    assert tomllib.loads(t.codex_config.read_text(encoding="utf-8"))["mcp_servers"]["seedgraph_v3"]
    assert (t.claude_skills_dir / "seedgraph-v3" / "SKILL.md").is_file()
    assert (t.codex_prompts_dir / "seedgraph-v3-work.md").is_file()


def test_plan_setup_needs_paths():
    entry = register.server_entry(worktree=Path("wt"), home=Path("h"))
    with pytest.raises(register.RegisterError):
        register.plan_setup("claude", entry)
