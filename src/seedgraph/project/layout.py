"""Slug-safe, relocatable path helpers for a project directory (decision 17).

No absolute path is ever stored in a row — every path is derived from
``(root, slug)`` at access time, so a project directory can be relocated freely.

This module is the phase_5 single import surface for project paths. ``validate_slug``,
``project_dir`` and ``project_db_path`` already exist in the phase_0 foundation
(:mod:`seedgraph.paths`, :mod:`seedgraph.db.connection`) and are re-exported here as
trivial delegations so the project layer has one place to import from; this phase
adds the new ``project_yaml_path``. Layout:

    {root or $SEEDGRAPH_HOME or ~/.seedgraph}/projects/{slug}/
        project.db
        project.yaml
        runs/                  # scaffolded empty; manifests written by later phases
"""

from __future__ import annotations

from pathlib import Path

from .. import paths
from ..db.connection import project_db_path

# Re-exported phase_0 helpers (single import surface for the project layer).
validate_slug = paths.validate_slug
project_dir = paths.project_dir

__all__ = [
    "validate_slug",
    "project_dir",
    "project_db_path",
    "project_yaml_path",
    "project_runs_dir",
]


def project_yaml_path(slug: str, root: Path | str | None = None) -> Path:
    """Resolve ``projects/{slug}/project.yaml`` (slug validated first)."""
    return paths.project_dir(slug, root) / "project.yaml"


def project_runs_dir(slug: str, root: Path | str | None = None) -> Path:
    """Resolve ``projects/{slug}/runs/`` (slug validated first)."""
    return paths.project_dir(slug, root) / "runs"
