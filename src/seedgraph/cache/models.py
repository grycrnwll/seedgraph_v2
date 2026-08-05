"""SQLModel table classes — ORM MAPPING ONLY (decision D6).

These classes MIRROR the migrated ``cache.db`` schema authored in
``db/schema/cache/0001_baseline.sql`` + ``0002_cache.sql``. They are used for
typed ORM reads/writes; ``SQLModel.metadata.create_all`` is **never** used to
author or evolve the schema (a phase_1 test asserts ORM metadata == migrated
schema). When the two disagree, the numbered ``.sql`` wins and these classes are
corrected to match.

Content-addressed ids (D1): ``SourceFile.source_file_id == 'sf_' + file_hash`` and
``MarkdownDocument.markdown_id == 'md_' + markdown_hash`` (the id IS the content
hash; the ``*_hash`` columns are explicit duplicates for query clarity).
``ConversionRun`` is a mutable identity row -> opaque ``conv_`` uuid.
``CacheEvent`` is a pure log row -> autoincrement integer pk.
"""

from __future__ import annotations

from typing import Optional

from sqlmodel import Field, SQLModel


class SourceFile(SQLModel, table=True):
    """Mirror of ``source_files`` (created in 0001, columns added in 0002).

    ``access_class`` is stored ONCE here (decision 11) — never denormalized onto
    markdown. ``updated_at`` is bumped only on access_class reconciliation.
    """

    __tablename__ = "source_files"

    source_file_id: str = Field(primary_key=True)  # 'sf_' + file_hash (D1)
    file_hash: str = Field(unique=True, index=True)  # dedup key; == id's hash
    file_type: Optional[str] = None  # 'pdf' | 'html'
    access_class: str = Field(default="user_supplied_private")
    acquisition_method: Optional[str] = None
    storage_uri: Optional[str] = None  # POSIX path RELATIVE to cache root
    byte_size: int = 0
    created_at: str = ""  # ISO-8601 UTC
    updated_at: Optional[str] = None  # bumped only on access_class reconcile


class ConversionRun(SQLModel, table=True):
    """Mirror of ``conversion_runs`` — full Marker provenance (doc 04 §8 / 11 §4/§5)."""

    __tablename__ = "conversion_runs"

    conversion_run_id: str = Field(primary_key=True)  # 'conv_' + uuid4hex (mutable, D1)
    source_file_id: str = Field(foreign_key="source_files.source_file_id", index=True)
    source_file_hash: str = Field(index=True)  # denormalized anchor + dedup
    converter_name: str
    converter_package: str
    converter_version: str
    python_version: str
    config_json: str  # canonical JSON (sorted keys), incl. paginate_output
    conversion_fingerprint: str = Field(index=True)
    llm_enabled: int = 0
    llm_service: Optional[str] = None
    llm_model: Optional[str] = None
    llm_prompt_version: Optional[str] = None
    conversion_epistemic_type: str  # doc 11 §5 enum
    run_status: str  # 'pending' | 'success' | 'failed'
    warnings_json: Optional[str] = None  # JSON array of validation signals
    error: Optional[str] = None
    markdown_hash: Optional[str] = None  # this run's output hash, set on success (dedup resolves run->markdown by content)
    created_at: str = ""
    completed_at: Optional[str] = None


class MarkdownDocument(SQLModel, table=True):
    """Mirror of ``markdown_documents`` — content-addressed markdown + cross-scope anchor."""

    __tablename__ = "markdown_documents"

    markdown_id: str = Field(primary_key=True)  # 'md_' + markdown_hash (D1); cross-DB anchor
    conversion_run_id: str = Field(foreign_key="conversion_runs.conversion_run_id")
    source_file_id: str = Field(foreign_key="source_files.source_file_id", index=True)
    markdown_hash: str = Field(index=True)  # == id's hash (explicit duplicate)
    storage_uri: str  # POSIX path RELATIVE to cache root
    conversion_status: str = "success"  # only successes get a markdown row
    byte_size: int = 0
    created_at: str = ""


class CacheEvent(SQLModel, table=True):
    """Mirror of ``cache_events`` — append-only audit log (autoincrement pk)."""

    __tablename__ = "cache_events"

    cache_event_id: Optional[int] = Field(default=None, primary_key=True)
    object_type: str  # 'source_file' | 'conversion_run' | 'markdown_document'
    object_id: str
    event_type: str  # see cache_events event vocab (plan §4)
    detail_json: Optional[str] = None
    created_at: str = ""


# phase_5b owns ``provider_cache`` (cache.db). Its ORM mapping lives in
# ``cache/provider_cache.py`` next to the request-key grammar; re-export it here so
# all cache.db ORM mappings register/import from one module (mapping-only — D6).
from .provider_cache import ProviderCache  # noqa: E402,F401
