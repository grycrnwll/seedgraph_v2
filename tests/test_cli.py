from typer.testing import CliRunner

from seedgraph import __version__
from seedgraph.cli import app

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "library" in result.output


def test_version_prints_versions():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "sqlite" in result.output


def test_v3_core_commands_are_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("setup", "migrate", "convert", "library"):
        assert name in result.output


def test_stage1_deleted_commands_are_gone():
    """Design §9: the old top-level `search --project` is deleted, and `migrate` is now the
    v3 migration group (the old v2 schema upgrade, `migrate [--project]`, is gone)."""
    old_search = runner.invoke(app, ["search", "x", "--project", "p"])
    assert old_search.exit_code == 2
    assert "No such command" in old_search.output
    old_migrate = runner.invoke(app, ["migrate", "--project", "p"])
    assert old_migrate.exit_code == 2
    assert "cache: schema at version" not in old_migrate.output


def test_top_level_command_names_are_unique():
    import typer.main

    group = typer.main.get_command(app)
    registered = len(app.registered_commands) + len(app.registered_groups)
    # Click keys commands by name, so a duplicate name would silently shadow another.
    assert len(group.commands) == registered
    assert "search" not in group.commands


# --- `graph.analyze.analyze_summary` (reserved for stage 4; the `graph analyze` CLI is gone)


# --- Build E ch5a: informational GPU probe (design 8 — probe never gates) ----


# --- Build D ch13: `corpus run --dry-run` — closed-form preview, ZERO network


