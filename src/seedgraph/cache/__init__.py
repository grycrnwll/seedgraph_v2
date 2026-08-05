"""Local content cache (phase_1) — public surface.

Turns an uploaded/fetched PDF into a content-addressed, deduplicated,
fully-provenanced, validated markdown artifact in ``~/.seedgraph/cache/`` +
``cache.db``. Same bytes are never hashed-and-stored twice; same
``(file, converter version, config)`` is never re-converted.

Authoring decisions: D1 (content-addressed ``sf_``/``md_`` ids), D6 (numbered
``.sql`` is the schema source of truth; SQLModel classes mirror it), decision 11
(``access_class`` stored once on ``source_files``), decision 83 / gap 10
(``paginate_output=True`` for page-citation fidelity).
"""

from __future__ import annotations

from .convert import (
    CacheError,
    ConversionError,
    MarkerConfig,
    add_document,
    conversion_epistemic_type,
    conversion_fingerprint,
    convert_source_file,
    write_manifest,
)
from .db import cache_engine, cache_session, init_cache_db
from .events import record_event
from .hashing import sha256_bytes, sha256_file, sha256_text
from .ingest import ingest_file, reconcile_access_class
from .marker_backend import (
    FakeMarkerBackend,
    LocalMarkerBackend,
    MarkerBackend,
    MarkerResult,
)
from .models import CacheEvent, ConversionRun, MarkdownDocument, SourceFile
from .read import markdown_path, markdown_text, resolve_markdown, source_access_class
from .validate import ValidationResult, validate_markdown

__all__ = [
    # ingest / convert / orchestration
    "ingest_file",
    "reconcile_access_class",
    "convert_source_file",
    "add_document",
    "MarkerConfig",
    "conversion_fingerprint",
    "conversion_epistemic_type",
    "write_manifest",
    "CacheError",
    "ConversionError",
    # backend
    "MarkerBackend",
    "MarkerResult",
    "LocalMarkerBackend",
    "FakeMarkerBackend",
    # validation
    "ValidationResult",
    "validate_markdown",
    # read APIs
    "resolve_markdown",
    "markdown_text",
    "markdown_path",
    "source_access_class",
    # db / events
    "init_cache_db",
    "cache_engine",
    "cache_session",
    "record_event",
    # hashing
    "sha256_file",
    "sha256_bytes",
    "sha256_text",
    # ORM models (mapping-only, mirror the migrated schema — D6)
    "SourceFile",
    "ConversionRun",
    "MarkdownDocument",
    "CacheEvent",
]
