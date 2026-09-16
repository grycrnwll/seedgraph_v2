"""Ambient-trigger install (``seedgraph project agent-setup``) unit tests.

Fully offline: pure functions over bundled templates + tmp_path filesystem. No
network, no LLM, no seedgraph home.
"""

from __future__ import annotations

from seedgraph.agent_setup import (
    install_skill,
    render_claude_block,
    write_claude_block,
)

_BEGIN = "<!-- seedgraph:begin -->"


def test_render_claude_block_substitutes_slug():
    rendered = render_claude_block("my-proj.v2")
    assert "{slug}" not in rendered
    assert "my-proj.v2" in rendered


def test_write_claude_block_is_idempotent(tmp_path):
    path = write_claude_block(tmp_path, "demo")
    first = path.read_text(encoding="utf-8")
    path2 = write_claude_block(tmp_path, "demo")
    second = path2.read_text(encoding="utf-8")

    assert path == path2
    assert first == second  # re-run produced no change
    assert second.count(_BEGIN) == 1  # exactly one fenced block, never duplicated


def test_write_claude_block_preserves_surrounding_content(tmp_path):
    claude = tmp_path / "CLAUDE.md"
    claude.write_text("# My project rules\n\nKeep this line.\n", encoding="utf-8")

    write_claude_block(tmp_path, "demo")
    after_first = claude.read_text(encoding="utf-8")
    assert "# My project rules" in after_first
    assert "Keep this line." in after_first
    assert after_first.count(_BEGIN) == 1

    write_claude_block(tmp_path, "demo")
    after_second = claude.read_text(encoding="utf-8")
    assert "# My project rules" in after_second  # pre-existing content survives re-run
    assert "Keep this line." in after_second
    assert after_second.count(_BEGIN) == 1


def test_install_skill_writes_then_reports_unchanged(tmp_path):
    path, changed = install_skill(tmp_path)
    assert changed is True
    assert path == tmp_path / "seedgraph" / "SKILL.md"
    assert path.read_text(encoding="utf-8").strip()

    path2, changed2 = install_skill(tmp_path)
    assert path2 == path
    assert changed2 is False  # second run is a no-op


def test_extraction_skill_is_explicit_and_separate(tmp_path):
    from seedgraph.agent_setup import install_extraction_skill

    ambient, _ = install_skill(tmp_path)
    original = ambient.read_bytes()
    skill, changed = install_extraction_skill(tmp_path, client="codex")
    assert changed is True
    assert skill.parent.name == "seedgraph-extract"
    text = skill.read_text(encoding="utf-8")
    assert "extraction_submit" in text and "--replace" in text
    assert "allow_implicit_invocation: false" in (skill.parent / "agents/openai.yaml").read_text()
    assert install_extraction_skill(tmp_path, client="codex")[1] is False
    assert ambient.read_bytes() == original
    claude, _ = install_extraction_skill(tmp_path / "claude", client="claude")
    assert "disable-model-invocation: true" in claude.read_text()


def test_explicit_skill_destination_is_preserved_even_if_it_matches_other_host_default(tmp_path, monkeypatch):
    from pathlib import Path
    from typer.testing import CliRunner
    from seedgraph.cli import app
    from seedgraph.project import service

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    handle = service.create_project("install_explicit")
    requested = tmp_path / ".claude" / "skills"
    result = CliRunner().invoke(app, ["project", "agent-setup", handle.slug, "--extraction",
                                     "--client", "codex", "--skills-dir", str(requested)])
    assert result.exit_code == 0, result.output
    assert (requested / "seedgraph-extract/SKILL.md").exists()
    assert not (tmp_path / ".agents").exists()
