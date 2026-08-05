from typer.testing import CliRunner

from seedgraph import __version__
from seedgraph.cli import app

runner = CliRunner()


def test_help_exits_zero():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output


def test_version_prints_versions():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output
    assert "sqlite" in result.output


def test_migrate_applies_and_is_idempotent():
    from seedgraph.db.migrations import latest_version

    expected = f"version {latest_version('cache')}"
    first = runner.invoke(app, ["migrate"])
    assert first.exit_code == 0
    assert expected in first.output

    second = runner.invoke(app, ["migrate"])
    assert second.exit_code == 0
    assert expected in second.output


def test_doctor_exits_zero_on_fresh_home():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "RESULT: OK" in result.output


# --- Build B ch4: `graph analyze` — decision 38's deterministic floor, visible


def test_graph_analyze_prints_summary_block():
    """`graph analyze <slug>` exits 0 and prints the summary block (communities,
    god nodes, bridges, read-next) on a fixture project; no run -> 'adhoc'."""
    from _phase8_helpers import build_fixture_project

    build_fixture_project("cli_an")
    result = runner.invoke(app, ["graph", "analyze", "cli_an"])
    assert result.exit_code == 0
    assert "run adhoc:" in result.output
    assert "epistemic_type=deterministic" in result.output
    assert "communities:" in result.output
    assert "god nodes:" in result.output
    assert "bridges:" in result.output
    # the fixture's single unread work (metadata_only work_c) is the read-next.
    assert "read next (top 10 of 1 unread):" in result.output
    assert "work_c" in result.output

    # --run-id threads through to the header line.
    named = runner.invoke(app, ["graph", "analyze", "cli_an", "--run-id", "r9"])
    assert named.exit_code == 0
    assert "run r9:" in named.output


def test_graph_analyze_lists_bridge_edges_with_flagged_audit():
    """Cross-community bridge edges are LISTED under the count line, each with
    the flagged-vs-derived audit marker (Build B plan item 6 — bridge surfacing
    via ``answer/traverse.bridges_in``). Two citation triangles joined by one
    edge -> two communities; the joining edge derives AND was flagged at
    annotate time, so it prints ``flagged``."""
    from _phase8_helpers import _now, _raw_conn
    from seedgraph.project import service

    handle = service.create_project("cli_br")
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

    result = runner.invoke(app, ["graph", "analyze", "cli_br"])
    assert result.exit_code == 0
    assert "bridges: 1 edge(s), 2 node(s)" in result.output
    assert "  a1 -> b1 [community" in result.output
    assert "; flagged]" in result.output
    assert "derived-only" not in result.output


# --- M10 lift: `graph analyze` renders `graph.analyze.analyze_summary` --------

# Golden literal captured from the UNMODIFIED pre-lift command body on the
# `build_fixture_project` corpus (deterministic: seeded Louvain, fixed fixture,
# no run -> "adhoc"). Pins the renderer output byte-for-byte across the lift so a
# formatting drift is caught, not just the presence of substrings.
_GOLDEN_ANALYZE_CLI_AN = (
    "run adhoc: 4 work(s), 1 citation edge(s) "
    "[citation projection; epistemic_type=deterministic]\n"
    "communities: 3\n"
    "  community 0: 2 work(s)\n"
    "  community 1: 1 work(s)\n"
    "  community 2: 1 work(s)\n"
    "god nodes: 4\n"
    "  work_a\tPaper A: Difference-in-Differences\n"
    "  work_c\tPaper C: Foundational (metadata only)\n"
    "  work_b\tPaper B: Linear IV\n"
    "  work_x\tPaper X: Excluded after extraction\n"
    "bridges: 0 edge(s), 0 node(s)\n"
    "read next (top 10 of 1 unread):\n"
    "  [1] work_c\tPaper C: Foundational (metadata only)\n"
)


def test_graph_analyze_output_byte_identical_to_pre_lift_golden():
    """The M10 renderer produces byte-identical stdout to the pre-lift inline body."""
    from _phase8_helpers import build_fixture_project

    build_fixture_project("cli_gold")
    result = runner.invoke(app, ["graph", "analyze", "cli_gold"])
    assert result.exit_code == 0
    assert result.output == _GOLDEN_ANALYZE_CLI_AN


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


def test_doctor_gpu_probe_ok_when_torch_import_raises(monkeypatch):
    """The check NEVER fails, even when `import torch` itself raises."""
    import sys

    from seedgraph import doctor as doctor_mod

    # None in sys.modules makes `import torch` raise ImportError deterministically,
    # so this holds on machines that DO have torch installed.
    monkeypatch.setitem(sys.modules, "torch", None)
    check = doctor_mod._check_gpu_probe()
    assert check.ok is True
    assert "torch not importable" in check.detail


def test_doctor_gpu_probe_ok_when_device_probe_raises(monkeypatch):
    """torch imports but the device probe explodes → still ok=True, device=unknown."""
    import sys
    import types

    from seedgraph import doctor as doctor_mod

    broken = types.ModuleType("torch")
    broken.cuda = types.SimpleNamespace(
        is_available=lambda: (_ for _ in ()).throw(RuntimeError("driver mismatch"))
    )
    monkeypatch.setitem(sys.modules, "torch", broken)
    check = doctor_mod._check_gpu_probe()
    assert check.ok is True
    assert "device=unknown" in check.detail


def test_doctor_gpu_probe_reports_cuda_but_guidance_stays_conservative(monkeypatch):
    """Happy path reports device + VRAM, and the recommended profile is still the
    hard-coded conservative floor (a wrong probe degrades to slow, never to crash)."""
    import sys
    import types

    from seedgraph import doctor as doctor_mod

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=16 * 1024**3),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    check = doctor_mod._check_gpu_probe()
    assert check.ok is True
    assert "device=cuda" in check.detail
    assert "gpu_memory_gb=16.0" in check.detail
    assert "conservative" in check.detail


def test_doctor_exits_zero_when_torch_import_raises(monkeypatch):
    """Acceptance rider (Build E item 5): doctor exits 0 on a torch-less machine."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "gpu_probe" in result.output
    assert "RESULT: OK" in result.output


# --- Build D ch13: `corpus run --dry-run` — closed-form preview, ZERO network


def test_corpus_run_dry_run_zero_network_exits_zero(monkeypatch):
    """`corpus run <slug> --dry-run` prints the closed-form acquisition budget
    (papers / API calls / disk) and exits 0 BEFORE constructing the provider
    chain: a ProviderChain.from_config that fails-if-called pins the
    zero-network claim (Build D ch13 / D-13)."""
    from _phase8_helpers import build_fixture_project
    from seedgraph.providers.base import ProviderChain

    build_fixture_project("cli_dry")  # 2 included works -> 2 seeds

    def _boom(*args, **kwargs):
        raise AssertionError("--dry-run must never construct the provider chain")

    monkeypatch.setattr(ProviderChain, "from_config", _boom)
    monkeypatch.delenv("SEEDGRAPH_FAKE_PROVIDERS", raising=False)

    result = runner.invoke(app, ["corpus", "run", "cli_dry", "--dry-run"])
    assert result.exit_code == 0, result.output
    # Defaults depth=2 / per-gen-cap=50 over the fixture's 2 included seeds:
    # expected_papers = 2 + 50*2 = 102, api_calls = 3x, est_disk = 102 x 1.9 MB.
    assert "expected_papers=102 (seeds=2 + cap=50 x depth=2)" in result.output
    assert "api_calls=~306" in result.output
    assert "est_disk=~193.8 MB" in result.output


def test_corpus_run_dry_run_honors_depth_and_cap(monkeypatch):
    """--depth/--per-gen-cap flow into the preview arithmetic unchanged."""
    from _phase8_helpers import build_fixture_project
    from seedgraph.providers.base import ProviderChain

    build_fixture_project("cli_dry2")

    def _boom(*args, **kwargs):
        raise AssertionError("--dry-run must never construct the provider chain")

    monkeypatch.setattr(ProviderChain, "from_config", _boom)
    result = runner.invoke(
        app,
        ["corpus", "run", "cli_dry2", "--dry-run", "--depth", "3", "--per-gen-cap", "10"],
    )
    assert result.exit_code == 0, result.output
    assert "expected_papers=32 (seeds=2 + cap=10 x depth=3)" in result.output
    assert "api_calls=~96" in result.output
