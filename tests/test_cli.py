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


def test_analyze_summary_shape_on_fixture_project():
    """`analyze_summary(h)` returns the shared dict the CLI/MCP adapters render:
    run header + counts, community sizes, god nodes (work_id + title), bridges,
    and read-next — composed offline from the fixture's single citation edge."""
    from _phase8_helpers import build_fixture_project

    from seedgraph.graph.analyze import analyze_summary

    h = build_fixture_project("as_shape")
    summary = analyze_summary(h)

    # Top-level keys (the MCP `graph_analyze` contract).
    assert set(summary) == {
        "run_id", "node_count", "edge_count", "community_count",
        "communities", "god_nodes", "bridge_edge_count", "bridge_node_count",
        "bridges", "unread_count", "read_next",
    }

    # Run header / counts (no run -> "adhoc"; citation projection = 4 works, 1 edge).
    assert summary["run_id"] == "adhoc"
    assert summary["node_count"] == 4
    assert summary["edge_count"] == 1

    # Communities: list of {community_id, size}, sorted by id, sizes covering all works.
    assert summary["community_count"] == len(summary["communities"]) == 3
    assert [c["community_id"] for c in summary["communities"]] == [0, 1, 2]
    assert all(c["size"] >= 1 for c in summary["communities"])
    assert sum(c["size"] for c in summary["communities"]) == 4

    # God nodes: work_id + resolved title, one entry per top-centrality work.
    assert all({"work_id", "title"} == set(g) for g in summary["god_nodes"])
    gods = {g["work_id"]: g["title"] for g in summary["god_nodes"]}
    assert gods["work_a"] == "Paper A: Difference-in-Differences"

    # Bridges: empty for this single-edge fixture (both endpoints separate comms,
    # but no cross-community CITATION carries positive betweenness here).
    assert summary["bridges"] == []
    assert summary["bridge_edge_count"] == 0
    assert summary["bridge_node_count"] == 0

    # Read-next: the lone metadata_only work_c, carrying its display title.
    assert summary["unread_count"] == 1
    assert summary["read_next"] == [
        {"work_id": "work_c", "title": "Paper C: Foundational (metadata only)"}
    ]


def test_analyze_summary_bridge_shape_on_two_triangle_project():
    """Non-vacuous bridge-shape check: two citation triangles joined by one edge
    give one cross-community bridge; `analyze_summary` surfaces it with the
    source/target, community-pair, and flagged-vs-derived audit fields."""
    from _phase8_helpers import _now, _raw_conn

    from seedgraph.graph.analyze import analyze_summary
    from seedgraph.project import service

    handle = service.create_project("as_bridge")
    conn = _raw_conn(handle)
    for wid in ("a1", "a2", "a3", "b1", "b2", "b3"):
        conn.execute(
            "INSERT INTO works (work_id, canonical_title, year, created_at) "
            "VALUES (?,?,?,?)",
            (wid, f"Paper {wid}", 2020, _now()),
        )
    for u, v in [
        ("a1", "a2"), ("a2", "a3"), ("a3", "a1"),
        ("b1", "b2"), ("b2", "b3"), ("b3", "b1"),
        ("a1", "b1"),
    ]:
        conn.execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, "
            "provenance, confidence, run_id, created_at) "
            "VALUES (?,?,'cites','provider_reference',1.0,'cite1',?)",
            (u, v, _now()),
        )
    conn.commit()
    conn.close()

    summary = analyze_summary(handle)

    assert summary["bridge_edge_count"] == 1
    assert summary["bridge_node_count"] == 2
    assert len(summary["bridges"]) == 1
    bridge = summary["bridges"][0]
    assert set(bridge) == {
        "source_work_id", "target_work_id",
        "source_community", "target_community", "flagged_is_bridge",
    }
    assert (bridge["source_work_id"], bridge["target_work_id"]) == ("a1", "b1")
    assert bridge["source_community"] != bridge["target_community"]
    assert bridge["flagged_is_bridge"] is True


# --- Build E ch5a: informational GPU probe (design 8 — probe never gates) ----


def test_doctor_gpu_probe_never_imports_torch_and_reports_gpus(monkeypatch):
    """The probe counts GPUs with nvidia-smi (the app has no torch) and reports the
    converter version pinned in converter/uv.lock. Informational: always ok."""
    from seedgraph import doctor as doctor_mod
    from seedgraph.conversion import pool as pool_mod

    monkeypatch.setattr(pool_mod, "_nvidia_smi_gpu_count", lambda: 2)
    check = doctor_mod._check_gpu_probe()
    assert check.ok is True
    assert "gpus=2" in check.detail
    assert "marker-pdf 1.10.2" in check.detail
    assert "conservative" in check.detail


def test_doctor_gpu_probe_ok_when_no_gpu(monkeypatch):
    from seedgraph import doctor as doctor_mod
    from seedgraph.conversion import pool as pool_mod

    monkeypatch.setattr(pool_mod, "_nvidia_smi_gpu_count", lambda: 0)
    check = doctor_mod._check_gpu_probe()
    assert check.ok is True
    assert "gpus=0" in check.detail


def test_doctor_gpu_probe_ok_when_probe_raises(monkeypatch):
    from seedgraph import doctor as doctor_mod
    from seedgraph.conversion import pool as pool_mod

    def boom():
        raise RuntimeError("driver mismatch")

    monkeypatch.setattr(pool_mod, "_nvidia_smi_gpu_count", boom)
    check = doctor_mod._check_gpu_probe()
    assert check.ok is True
    assert "unknown" in check.detail


# --- Build D ch13: `corpus run --dry-run` — closed-form preview, ZERO network


