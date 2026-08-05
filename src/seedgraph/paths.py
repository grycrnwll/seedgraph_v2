"""Filesystem layout + slug safety (decision 17).

Root resolves by ``--root`` override → ``$SEEDGRAPH_HOME`` → default
``~/.seedgraph``. ``validate_slug`` rejects anything that is not a single
``^[a-z0-9._-]+$`` path segment *before* it reaches the filesystem.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .errors import ValidationError

SLUG_RE = re.compile(r"^[a-z0-9._-]+$")
_DEFAULT_HOME = Path.home() / ".seedgraph"


def resolve_home(root: Path | str | None = None) -> Path:
    """Resolve the seedgraph home directory."""
    if root is not None:
        return Path(root)
    env = os.environ.get("SEEDGRAPH_HOME")
    if env:
        return Path(env)
    return _DEFAULT_HOME


def cache_root(root: Path | str | None = None) -> Path:
    return resolve_home(root) / "cache"


def validate_slug(slug: str) -> str:
    """Return ``slug`` unchanged, or raise :class:`ValidationError`.

    Rejects empty strings, uppercase, path separators, ``.``/``..``, and any
    character outside ``[a-z0-9._-]`` — i.e. anything that is not a safe single
    path segment.
    """
    if not isinstance(slug, str) or not slug:
        raise ValidationError(
            f"invalid project slug {slug!r}: must be a non-empty single path segment "
            f"matching ^[a-z0-9._-]+$"
        )
    if slug in (".", "..") or not SLUG_RE.match(slug):
        raise ValidationError(
            f"invalid project slug {slug!r}: only lowercase [a-z0-9._-] single path "
            f"segments are allowed (no '/', '\\', '..', uppercase, or whitespace)"
        )
    return slug


def project_dir(slug: str, root: Path | str | None = None) -> Path:
    """Resolve ``projects/{slug}`` after validating the slug (raises first)."""
    validate_slug(slug)
    return resolve_home(root) / "projects" / slug
