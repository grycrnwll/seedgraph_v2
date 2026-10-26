"""The v3 core commands (design §1.3, §10). ``register(app)`` mounts them on the root app;
it is the only place a command is added (``cli.py`` holds just the app and ``version``).

Every module here is a thin adapter: parse options, call one function of a deep
module (:mod:`seedgraph.library`, :mod:`seedgraph.migrate`, :mod:`seedgraph.golden`),
render.
"""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    from ..golden.cli import golden_app
    from . import convert, library, mcp, migrate, serve, setup, work

    setup.setup_app.command("agents")(mcp.agents)
    app.add_typer(setup.setup_app, name="setup")
    app.add_typer(mcp.mcp_app, name="mcp")
    app.add_typer(migrate.migrate_app, name="migrate")
    app.command("convert")(convert.convert)
    app.add_typer(library.library_app, name="library")
    app.add_typer(work.work_app, name="work")
    app.command("serve")(serve.serve)
    app.add_typer(golden_app, name="golden")
