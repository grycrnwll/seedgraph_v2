"""Citation-graph subsystem.

Phase 2 owns the provider-metadata edge layer (``walk.py``, ``edges.py``,
``graph/``). Phase 3b adds the parsed-bibliography tier under the **same**
``citation_edges`` table: ``bib_parser`` (deterministic references parse over
``document_sections`` char ranges), ``resolve`` (strong-id-gated resolution +
v1 precision guard), and ``parsed_bib`` (the two-stage per-work orchestrator).

Phase 2's surface (``edges`` / ``project_edges``) is import-clean (no network, no
networkx) so it is re-exported eagerly here; the parsed-bibliography modules stay
lazy until phase_3b lands.
"""

from __future__ import annotations

from .edges import (
    PROVENANCE_AUTHORITY,
    PROVIDER_REFERENCE,
    authoritative_edges,
    is_shareable_edge,
    write_edge,
)
from .project_edges import project_provider_edges

__all__ = [
    "PROVENANCE_AUTHORITY",
    "PROVIDER_REFERENCE",
    "authoritative_edges",
    "is_shareable_edge",
    "write_edge",
    "project_provider_edges",
]
