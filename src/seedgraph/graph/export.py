"""Export the NetworkX citation view to ``runs/{run_id}/`` artifacts.

:func:`export_graph` writes two derived artifacts under a run directory
(decision 17):

* ``graph.json`` — a NetworkX node-link export of the graph, **shareable-filtered**
  on the way out via :func:`citation.edges.is_shareable_edge` so only metadata-class
  (``provider_reference``) edges are serialized. Because every phase_2 edge is
  ``provider_reference``, ``graph.json`` doubles as the metadata-only shareable
  export and is safe to share (decision 5, D8). It reloads into NetworkX with
  node/edge counts matching ``authoritative_edges(run_id)`` (criterion 4).
* ``manifest.json`` section(s) — the secret-stripped effective ``config_snapshot``
  plus its round-trippable ``config_fingerprint`` (sha256 over canonical key-sorted
  JSON; ``config.loader.config_fingerprint(section["config_snapshot"])`` rebuilds
  it), pinned tool/provider versions, the walked source-work set,
  ``provider_cache`` freshness, ``run_id``,
  and per-provenance / edge / node counts (decision 35). Written via the run
  manager's shallow section-keyed atomic replacement (D5) — this stage owns its
  own disjoint top-level section.

``networkx`` is imported lazily inside the function (phase_2 runtime dependency;
wiring stage adds it to pyproject — see needs_wiring).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import networkx

# The node-link document key under which edges are serialized. Pinned so the
# write (:func:`export_graph`) and the read (:func:`load_graph`) always agree,
# regardless of the installed networkx default (3.x deprecated the implicit key).
_EDGES_KEY = "links"


def _node_link_data(graph: "networkx.DiGraph") -> dict:
    import networkx as nx

    try:
        return nx.node_link_data(graph, edges=_EDGES_KEY)
    except TypeError:  # pragma: no cover - older networkx without the edges kwarg
        return nx.node_link_data(graph)


def load_graph(path: Path | str) -> "networkx.DiGraph":
    """Reload an exported ``graph.json`` node-link document into an ``nx.DiGraph``.

    The inverse of :func:`export_graph`'s serialization — shared by the export
    round-trip test and the ``GET .../graph.json`` read route so the edges key never
    drifts between writer and reader.
    """
    import networkx as nx

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return nx.node_link_graph(data, directed=True, edges=_EDGES_KEY)
    except TypeError:  # pragma: no cover - older networkx without the edges kwarg
        return nx.node_link_graph(data, directed=True)


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def export_graph(
    g: "networkx.DiGraph",
    out_dir: Path | str,
    *,
    manifest: dict,
) -> None:
    """Write ``graph.json`` (shareable-filtered node-link) + the run ``manifest``.

    Serializes ``g`` to ``{out_dir}/graph.json`` as a NetworkX node-link document,
    dropping any edge for which :func:`citation.edges.is_shareable_edge` is False
    (all phase_2 edges are shareable, so the filter is a no-op identity here but is
    exercised as the export seam). Writes the supplied ``manifest`` section(s) into
    ``{out_dir}/manifest.json`` (decision 35) using the run manager's shallow,
    section-keyed atomic replacement (D5). The exported ``graph.json`` round-trips
    back into NetworkX with node/edge counts equal to the run's authoritative edge
    set (criterion 4).
    """
    import networkx as nx

    from ..citation.edges import is_shareable_edge

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Shareable-export seam (decision 5, D8): drop any edge whose provenance is not
    # metadata-class. Every phase_2 edge is provider_reference, so this is identity
    # now, but the seam is exercised so a non-shareable provenance is filtered out.
    shareable = nx.DiGraph()
    shareable.add_nodes_from(g.nodes(data=True))
    for source, target, data in g.edges(data=True):
        if is_shareable_edge(data.get("provenance", "")):
            shareable.add_edge(source, target, **data)

    _atomic_write_json(out / "graph.json", _node_link_data(shareable))

    # manifest.json: merge the supplied section(s) into the run manifest's
    # ``sections`` without clobbering OTHER stages' sections (D5). Re-running the
    # export overwrites only this stage's own ``citation`` section (idempotent).
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        document = {"sections": {}}
    sections = document.setdefault("sections", {})
    for key, value in manifest.items():
        sections[key] = value
    _atomic_write_json(manifest_path, document)
