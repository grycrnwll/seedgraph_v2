"""The v3 core commands (design §10). ``register(app)`` mounts them on the root app.

Every module here is a thin adapter: parse options, call one function of a deep
module (:mod:`seedgraph.library`, :mod:`seedgraph.migrate`), render.
"""

from __future__ import annotations

import typer


def register(app: typer.Typer) -> None:
    from . import convert, library, mcp, migrate, setup, work

    setup.setup_app.command("agents")(mcp.agents)
    app.add_typer(setup.setup_app, name="setup")
    app.add_typer(mcp.mcp_app, name="mcp")
    app.add_typer(migrate.migrate_app, name="migrate")
    app.command("convert")(convert.convert)
    app.add_typer(library.library_app, name="library")
    app.add_typer(work.work_app, name="work")
