"""NetworkX graph view + export utilities (phase_2).

``build.py`` assembles an ``nx.DiGraph`` from the durable ``citation_edges`` rows;
``export.py`` writes the ``runs/{run_id}/graph.json`` + ``manifest.json`` artifacts.
Reused later by the semantic-graph overlay (phase_7).

This ``__init__`` imports nothing eagerly so the package stays import-clean even
when the optional ``networkx`` runtime dependency is not installed; import the
submodules (``graph.build`` / ``graph.export``) directly.
"""

from __future__ import annotations
