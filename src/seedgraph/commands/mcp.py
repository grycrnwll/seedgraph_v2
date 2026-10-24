"""`seedgraph mcp serve` (design §7.1) and `seedgraph setup agents` (§7.4).

- ``mcp serve`` runs the v3 stdio MCP server (:mod:`seedgraph.mcp.server`) on the
  home given by ``--root`` / ``$SEEDGRAPH_HOME``. ``--redact-private`` is gone (ADR-0012).
- ``setup agents`` renders the ``seedgraph_v3`` registration and the two skills for
  Claude Code and/or Codex through :mod:`seedgraph.agent_setup.register`. It prints the
  plan and writes only with ``--write``. Codex skills go to ``~/.codex/skills/<name>/SKILL.md``.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from . import _common

mcp_app = typer.Typer(name="mcp", help="The Seedgraph MCP server.", no_args_is_help=True)


@mcp_app.command("serve")
def serve(root: Optional[Path] = _common.ROOT_OPTION) -> None:
    """Run the stdio MCP server over the library, its context graphs and the work protocol."""
    from ..mcp import server

    server.main(home=_common.home(root))


class AgentClient(str, Enum):
    claude = "claude"
    codex = "codex"
    all = "all"


class Launcher(str, Enum):
    exe = "exe"
    python = "python"


def default_worktree() -> Path:
    """The checkout whose ``.venv`` runs this program (``<worktree>/.venv``)."""
    return Path(sys.prefix).resolve().parent


def agents(
    client: AgentClient = typer.Option(AgentClient.all, "--client", help="claude, codex or all."),
    write: bool = typer.Option(False, "--write", help="Apply exactly the change shown."),
    root: Optional[Path] = typer.Option(
        None, "--root", help="The v3 home the server will read (default: $SEEDGRAPH_HOME, else ~/.seedgraph-v3)."),
    user_home: Optional[Path] = typer.Option(
        None, "--user-home", help="Where .claude.json, .claude/ and .codex/ live (default: your user folder)."),
    worktree: Optional[Path] = typer.Option(
        None, "--worktree", help="The checkout whose .venv runs the server (default: this program's)."),
    launcher: Launcher = typer.Option(Launcher.exe, "--launcher",
                                      help="exe: .venv/Scripts/seedgraph-v3.exe; python: python -m seedgraph."),
) -> None:
    """Show (and with --write apply) the seedgraph_v3 MCP registration and skills."""
    from ..agent_setup import register

    with _common.cli_errors():
        targets = register.default_targets(user_home or Path.home())
        entry = register.server_entry(worktree=worktree or default_worktree(), home=_common.home(root),
                                      launcher=launcher.value)
        plan = register.plan_setup(
            "both" if client is AgentClient.all else client.value, entry,
            claude_config=targets.claude_config, claude_skills_dir=targets.claude_skills_dir,
            codex_config=targets.codex_config, codex_skills_dir=targets.codex_skills_dir,
            codex_layout="skills")
        if not write:
            typer.echo(plan.render())
            return
        typer.echo(plan.render().replace(
            "Nothing has been written. Re-run with --write to apply exactly this.\n", ""))
        written = plan.apply(write=True)
    for path in written:
        typer.echo(f"wrote {path}")
    if not written:
        typer.echo("nothing to write; everything is already as shown")
