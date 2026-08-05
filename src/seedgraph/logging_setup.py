"""Stdlib logging configuration (level via env/config)."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: str | None = None) -> logging.Logger:
    """Configure the ``seedgraph`` logger. Level: arg → env → INFO."""
    resolved = level or os.environ.get("SEEDGRAPH_LOG_LEVEL", "INFO")
    numeric = getattr(logging, str(resolved).upper(), logging.INFO)
    logging.basicConfig(level=numeric, format=_FORMAT)
    return logging.getLogger("seedgraph")
