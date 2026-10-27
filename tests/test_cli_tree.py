"""The trimmed v3 CLI (docs/V3_UI_DESIGN.md §1.3, §11 U9).

* the top-level command set is pinned, so a v2 group cannot come back unnoticed;
* every command in the tree answers ``--help`` (a smoke test per command);
* every ``seedgraph-v3 <command>`` named in the agent instructions
  (``agent_setup/*.md``) and in ``docs/DEPENDENCY_POLICY.md`` resolves in the app.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer.main
from typer.testing import CliRunner

from seedgraph.cli import app

REPO = Path(__file__).resolve().parents[1]
AGENT_DOCS = sorted((REPO / "src" / "seedgraph" / "agent_setup").glob("*.md"))
COMMAND_DOCS = [*AGENT_DOCS, REPO / "docs" / "DEPENDENCY_POLICY.md"]

CORE = {"setup", "migrate", "serve", "mcp", "convert", "work", "golden", "library", "concepts", "version"}
REMOVED = {"corpus", "cite", "sections", "spans", "lens", "graph", "cache", "project",
           "review", "doctor", "work-read"}


def _root():
    return typer.main.get_command(app)


def _is_group(cmd) -> bool:
    # Typer vendors its own click, so test the shape, not the class.
    return isinstance(getattr(cmd, "commands", None), dict)


def _paths(cmd, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    out = [prefix]
    if _is_group(cmd):
        for name, sub in sorted(cmd.commands.items()):
            out.extend(_paths(sub, (*prefix, name)))
    return out


ALL_PATHS = _paths(_root())


def test_the_top_level_commands_are_exactly_the_v3_core():
    assert set(_root().commands) == CORE
    assert not (set(_root().commands) & REMOVED)


def test_the_library_maintenance_group_has_its_commands():
    library = _root().commands["library"]
    for name in ("status", "search", "show", "check", "attention", "remove", "delete", "textcheck",
                 "reconvert", "pagecheck", "refetch-metadata", "prune-replaced", "graphs", "merge", "reassign-pdf"):
        assert name in library.commands, name
    assert "agents" in _root().commands["setup"].commands
    assert "serve" in _root().commands["mcp"].commands


@pytest.mark.parametrize("path", ALL_PATHS, ids=lambda p: " ".join(p) or "<root>")
def test_every_command_answers_help(path):
    result = CliRunner().invoke(app, [*path, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_version_runs():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.startswith("seedgraph ")


_REF = re.compile(r"seedgraph-v3 ([^`\n]*)")


def _resolve(tokens: list[str]) -> str | None:
    """Walk ``tokens`` down the command tree; return the first token that fails, else None."""
    cmd = _root()
    for tok in tokens:
        if not _is_group(cmd):
            return None  # a leaf: the rest are its arguments
        if tok.startswith(("-", "<")) or tok.isupper() or tok in {"...", "…"}:
            return None
        alternatives = tok.split("|")  # `work claim|submit|release`
        if not all(a in cmd.commands for a in alternatives):
            return tok
        cmd = cmd.commands[alternatives[0]]
    return None


def _references() -> list[tuple[str, str]]:
    refs = []
    for doc in COMMAND_DOCS:
        for m in _REF.finditer(doc.read_text(encoding="utf-8")):
            tokens = m.group(1).split()
            if tokens and re.fullmatch(r"[a-z][a-z-]*(\|[a-z][a-z-]*)*", tokens[0]):
                refs.append((doc.name, m.group(1).strip()))
    return refs


def test_the_docs_name_some_commands():
    assert len(_references()) >= 3


@pytest.mark.parametrize("doc,ref", _references(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_command_named_in_the_docs_resolves(doc, ref):
    bad = _resolve(ref.split())
    assert bad is None, f"{doc}: `seedgraph-v3 {ref}` — {bad!r} is not a command"


def test_agent_docs_name_no_bare_v2_commands():
    # The v2 carriers (`seedgraph concepts …`, `seedgraph ask …`) are gone; any CLI
    # command an agent is told to run uses the v3 name.
    for doc in AGENT_DOCS:
        text = doc.read_text(encoding="utf-8")
        assert not re.search(r"`seedgraph (?!-v3)[a-z]", text), doc.name
