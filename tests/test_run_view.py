"""Direct tests for :mod:`seedgraph.graph.run_view` — the run-view orchestration
both the CLI and the web run-planner go through.

These exercise ``build_and_export`` at its OWN interface (no Typer, no CliRunner):
the whole point of lifting this out of ``cli.py`` was that the orchestration is
now testable through the module seam both adapters cross, not only by booting the
CLI. Fully offline — no network, no LLM.
"""

from __future__ import annotations

from seedgraph.citation.edges import write_edge
from seedgraph.config.loader import config_fingerprint
from seedgraph.db.connection import connect_project_raw
from seedgraph.graph.export import load_graph
from seedgraph.graph.run_view import build_and_export
from seedgraph.project import service


def _seed_project_with_edge(slug: str):
    """A project with two works and one provider_reference edge A->B under run 'R'."""
    h = service.create_project(slug)
    a = service.add_work(h, ids={"doi": "10.1000/aaa", "openalex": "W_A"}, title="Paper A")
    b = service.add_work(h, ids={"doi": "10.1000/bbb", "openalex": "W_B"}, title="Paper B")
    conn = connect_project_raw(h.db_path)
    try:
        write_edge(
            conn,
            source=a.work_id,
            target=b.work_id,
            provenance="provider_reference",
            confidence=1.0,
            run_id="R",
        )
        conn.commit()
    finally:
        conn.close()
    return h


def test_build_and_export_materializes_and_writes_run_view():
    """``build_and_export`` builds the NetworkX view AND writes graph.json +
    manifest.json to the run dir, returning the in-memory graph."""
    h = _seed_project_with_edge("runview")

    graph, out_dir = build_and_export(h.slug, h.root, h.root, "R", open_world=False)

    # In-memory view: both works + the shareable provider edge.
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1

    # On-disk artifacts landed under runs/R/.
    assert out_dir.name == "R"
    assert (out_dir / "graph.json").is_file()
    assert (out_dir / "manifest.json").is_file()

    # graph.json round-trips to the same shareable view.
    reloaded = load_graph(out_dir / "graph.json")
    assert reloaded.number_of_edges() == 1


def test_build_and_export_manifest_fingerprint_round_trips():
    """The citation manifest section carries the run_id and a config_fingerprint
    that verifies against its own config_snapshot (decision 35 / Build F ch5)."""
    import json

    h = _seed_project_with_edge("runviewmanifest")
    _graph, out_dir = build_and_export(h.slug, h.root, h.root, "R", open_world=False)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    citation = manifest["sections"]["citation"]
    assert citation["run_id"] == "R"
    assert citation["edge_count"] == 1
    assert citation["config_fingerprint"] == config_fingerprint(citation["config_snapshot"])


def test_web_cite_job_crosses_the_run_view_seam():
    """The WEB adapter path: driving ``web.planner._job_cite`` end-to-end executes
    its function-local imports of ``build_and_export`` / ``connect_project_raw`` /
    ``open_cache_ro`` from their real homes and writes graph.json — proving the web
    run-planner no longer reaches into cli.py for the run-view orchestration."""
    from seedgraph.web.planner import _job_cite, plan_job

    h = _seed_project_with_edge("webcite")

    events: list = []

    def emit(event, message="", *, level="info", **data):
        events.append({"event": event, "message": message, "level": level, "data": data})
        return len(events) - 1

    emit.run_id = "run-web-cite"

    fn = _job_cite(plan_job(h, "cite"), {})
    fn(emit, h)

    out_dir = h.root / "projects" / h.slug / "runs" / "run-web-cite"
    assert (out_dir / "graph.json").is_file()
    assert any(e["event"] == "cite" for e in events)
