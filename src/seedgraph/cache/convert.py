"""Marker conversion orchestration: dedup, run, validate, store, provenance.

Implements the load-bearing dedup + validation algorithm (plan §6):

1. Load ``source_files`` row; reject ``file_type='html'`` (deferred — §2).
2. ``fp = conversion_fingerprint(backend.version, cfg)``.
3. If not ``force``: ``SELECT … WHERE source_file_hash=? AND conversion_fingerprint=?
   AND run_status='success' ORDER BY created_at DESC, conversion_run_id DESC LIMIT 1``
   -> hit returns that run's ``markdown_documents`` row + emits ``conversion_reused``
   (Marker NOT invoked). ``force`` skips this SELECT and always mints a new run.
4. Else INSERT ``conversion_runs(run_status='pending')``, emit ``conversion_started``,
   invoke ``backend``.
5. ``validate_markdown``: ``ok=False`` -> ``run_status='failed'`` + ``error`` + merged
   warnings + ``conversion_failed``; NO markdown row/file written. ``ok=True`` ->
   structural warnings appended, run continues.
6. Hash output, content-address-store the ``.md`` (atomic), INSERT
   ``markdown_documents``, set run ``success``, write best-effort ``meta.json`` +
   ``manifest.json``, emit ``conversion_succeeded`` + ``markdown_written``.
7. Backend exception -> run ``failed`` + ``error`` + ``conversion_failed`` (append-only;
   retry mints a fresh run).

Dedup is SELECT-based, not constraint-based (no unique index on
``(source_file_hash, conversion_fingerprint)``): multiple ``success`` rows with the
same fingerprint may coexist under append-only / ``--force``; the canonical run is
the latest success by ``created_at`` (tie-break ``conversion_run_id``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..db.connection import open_cache_db
from ..errors import SeedgraphError
from ..ids import new_id
from ..vocab import (
    AccessClass,
    AcquisitionMethod,
    ConversionEpistemicType,
    FileType,
    RunStatus,
)
from . import store
from .db import init_cache_db
from .events import record_event
from .hashing import sha256_text
from .marker_backend import FakeMarkerBackend, LocalMarkerBackend, MarkerBackend
from .models import MarkdownDocument, SourceFile
from .validate import validate_markdown


class CacheError(SeedgraphError):
    """A cache operation could not be completed (bad selector, deferred feature,
    access-policy refusal)."""


class ConversionError(CacheError):
    """A Marker conversion run failed (backend exception or fatal validation gate).
    The failing ``conversion_runs`` row is retained with ``run_status='failed'``."""


# Local LLM services treated as on-machine (no text leaves the host). Anything else
# under ``use_llm`` is an EXTERNAL assist and is gated for private documents (§8).
_LOCAL_LLM_SERVICES = frozenset({"local", "ollama", "llama_cpp", "vllm", "lmstudio"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(obj) -> str:
    """Canonical JSON: sorted keys, compact separators (stable across runs)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _default_backend() -> MarkerBackend:
    """Default conversion backend. Real :class:`LocalMarkerBackend` unless the
    ``SEEDGRAPH_FAKE_MARKER`` env flag is set (offline CLI demos / manual smoke),
    in which case a deterministic :class:`FakeMarkerBackend` is used. Tests always
    pass an explicit backend, so this flag never affects assertions."""
    if os.environ.get("SEEDGRAPH_FAKE_MARKER"):
        return FakeMarkerBackend()
    return LocalMarkerBackend()


@dataclass(frozen=True)
class MarkerConfig:
    """Immutable Marker conversion config — part of the conversion IDENTITY.

    Every field perturbs :func:`conversion_fingerprint`. ``paginate_output=True``
    (gap 10 / decision 83) makes Marker emit in-markdown page-delimiter markers
    that phase_3 parses for real page citations; toggling it changes the markdown
    bytes -> a new ``markdown_hash`` under a new fingerprint (a paginated run is
    never deduped against a non-paginated one).
    """

    use_llm: bool = False
    redo_inline_math: bool = False  # only meaningful with use_llm
    force_ocr: bool = False
    output_format: str = "markdown"
    paginate_output: bool = True  # emit page-delimiter markers (part of identity)
    llm_service: str | None = None
    llm_model: str | None = None
    llm_prompt_version: str | None = None

    def to_canonical_dict(self) -> dict:
        """Config as a plain dict for canonical (sorted-key) JSON serialization
        into ``config_json`` and the fingerprint tuple."""
        return dataclasses.asdict(self)


def conversion_fingerprint(converter_version: str, cfg: MarkerConfig) -> str:
    """SHA-256 (lowercase hex) of the canonical conversion-identity tuple:
    ``converter_version`` + every :class:`MarkerConfig` field (incl.
    ``paginate_output``), serialized as canonical JSON (sorted keys, fixed field
    set). Stable under key reordering; changes when the version or any config
    field changes. This is the dedup key."""
    payload = {"converter_version": converter_version, "config": cfg.to_canonical_dict()}
    return sha256_text(_canonical_json(payload))


def conversion_epistemic_type(cfg: MarkerConfig) -> "ConversionEpistemicType":
    """Map config -> doc 11 §5 epistemic type: no-LLM ->
    ``local_deterministic_conversion``; ``use_llm`` + external service ->
    ``local_conversion_with_external_llm_assist``; local LLM ->
    ``local_conversion_with_local_llm_assist`` (plan §7)."""
    if not cfg.use_llm:
        return ConversionEpistemicType.local_deterministic_conversion
    service = (cfg.llm_service or "").strip().lower()
    if service in _LOCAL_LLM_SERVICES:
        return ConversionEpistemicType.local_conversion_with_local_llm_assist
    return ConversionEpistemicType.local_conversion_with_external_llm_assist


def write_manifest(
    conversion_run_id: str,
    *,
    source_file_hash: str,
    markdown_hash: str,
    cfg: MarkerConfig,
    converter_version: str,
    root: Path | str | None = None,
) -> Path:
    """Write the per-conversion ``marker/{conv}/manifest.json`` provenance mirror
    (doc 11 §4): ``sha256:``-prefixed ``source_file_hash`` + ``markdown_hash``,
    converter/python versions, and ``paginate_output``. Returns its path."""
    manifest = {
        "schema": "seedgraph/marker-manifest/v1",
        "conversion_run_id": conversion_run_id,
        "converter": {
            "name": "marker",
            "package": "marker-pdf",
            "version": converter_version,
            "python_version": platform.python_version(),
        },
        "config": cfg.to_canonical_dict(),
        "paginate_output": cfg.paginate_output,
        "source_file_hash": f"sha256:{source_file_hash}",
        "markdown_hash": f"sha256:{markdown_hash}",
        "created_at": _now_iso(),
    }
    dest = store.marker_dir(conversion_run_id, root) / "manifest.json"
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    store.write_bytes_atomic(payload, dest)
    return dest


def _insert_pending_run(
    conn,
    *,
    run_id: str,
    src: SourceFile,
    cfg: MarkerConfig,
    converter_version: str,
    fingerprint: str,
    created_at: str,
) -> None:
    epistemic = conversion_epistemic_type(cfg)
    conn.execute(
        "INSERT INTO conversion_runs "
        "(conversion_run_id, source_file_id, source_file_hash, converter_name, "
        " converter_package, converter_version, python_version, config_json, "
        " conversion_fingerprint, llm_enabled, llm_service, llm_model, "
        " llm_prompt_version, conversion_epistemic_type, run_status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            src.source_file_id,
            src.file_hash,
            "marker",
            "marker-pdf",
            converter_version,
            platform.python_version(),
            _canonical_json(cfg.to_canonical_dict()),
            fingerprint,
            1 if cfg.use_llm else 0,
            cfg.llm_service if cfg.use_llm else None,
            cfg.llm_model if cfg.use_llm else None,
            cfg.llm_prompt_version if cfg.use_llm else None,
            epistemic.value,
            RunStatus.pending.value,
            created_at,
        ),
    )


def _finish_run(
    conn,
    run_id: str,
    *,
    status: RunStatus,
    warnings: list[str],
    error: str | None,
    markdown_hash: str | None = None,
) -> None:
    # markdown_hash is set ONLY on a success finish (NULL for pending/failed); it
    # records this run's output content hash so dedup can resolve run -> markdown by
    # content, independent of which run currently owns the shared markdown row.
    conn.execute(
        "UPDATE conversion_runs SET run_status = ?, warnings_json = ?, error = ?, "
        "markdown_hash = ?, completed_at = ? WHERE conversion_run_id = ?",
        (
            status.value,
            _canonical_json(warnings) if warnings else None,
            error,
            markdown_hash,
            _now_iso(),
            run_id,
        ),
    )


def store_markdown_document(
    conn,
    *,
    source_file_id: str,
    source_access_class: str,
    conversion_run_id: str,
    markdown_text: str,
    root: Path | str | None = None,
) -> "MarkdownDocument":
    """Content-address-store markdown bytes + upsert the shared ``markdown_documents``
    row — the SINGLE markdown-storage surface (shared by :func:`convert_source_file`
    and the external-import path so a converted and an imported doc are byte-identical
    in shape).

    Hashes the text (``markdown_id == 'md_' + sha256``), atomically writes
    ``markdown/{hash}.md`` under the cache root, and inserts the ``markdown_documents``
    row pointing at ``conversion_run_id`` (``conversion_status='success'``). Does NOT
    touch ``conversion_runs`` (the caller owns that row's lifecycle) and does NOT
    commit (the caller commits).

    ``markdown_id`` IS the content hash (PK), so identical markdown bytes produced by
    a later run collide on the PK. These collisions are NOT only ``--force`` re-runs
    of the same file: a DISTINCT source file (different file_hash, maybe a different
    access_class) can convert to byte-identical markdown, and a new
    conversion_fingerprint can too. The single content-addressed row is SHARED across
    all such producers, and its ``source_file_id`` is the authoritative access-walk
    target (``source_access_class``, decision 11). It is therefore reconciled
    MOST-RESTRICTIVE / never-downgrade, exactly like ``source_files`` ingest
    reconciliation: a later, LESS-restrictive writer (e.g. an open_access fetch) must
    NOT repoint a user_supplied_private row and leak it.
    """
    markdown_bytes = markdown_text.encode("utf-8")
    markdown_hash = sha256_text(markdown_text)
    markdown_id = f"md_{markdown_hash}"
    blob_path = store.markdown_blob_path(markdown_hash, root)
    store.write_bytes_atomic(markdown_bytes, blob_path)
    storage_uri = store.relative_uri(blob_path, root)
    md_created_at = _now_iso()

    existing = conn.execute(
        "SELECT s.access_class FROM markdown_documents m "
        "JOIN source_files s ON s.source_file_id = m.source_file_id "
        "WHERE m.markdown_id = ?",
        (markdown_id,),
    ).fetchone()
    if existing is None:
        conn.execute(
            "INSERT INTO markdown_documents "
            "(markdown_id, conversion_run_id, source_file_id, markdown_hash, storage_uri, "
            " conversion_status, byte_size, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                markdown_id,
                conversion_run_id,
                source_file_id,
                markdown_hash,
                storage_uri,
                "success",
                len(markdown_bytes),
                md_created_at,
            ),
        )
    else:
        # Shared row already exists (identical bytes). storage_uri/byte_size are
        # content-derived (identical), so only the access-walk pointer matters:
        # tighten it to this run iff this source is STRICTLY more restrictive.
        stored_class = AccessClass.most_restrictive(existing[0])
        incoming_class = AccessClass.most_restrictive(source_access_class)
        if (
            AccessClass.most_restrictive(stored_class, incoming_class) == incoming_class
            and incoming_class != stored_class
        ):
            conn.execute(
                "UPDATE markdown_documents SET source_file_id = ?, conversion_run_id = ? "
                "WHERE markdown_id = ?",
                (source_file_id, conversion_run_id, markdown_id),
            )
        # else: existing pointer is equal/more restrictive -> keep it (never downgrade).
    return MarkdownDocument(
        markdown_id=markdown_id,
        conversion_run_id=conversion_run_id,
        source_file_id=source_file_id,
        markdown_hash=markdown_hash,
        storage_uri=storage_uri,
        conversion_status="success",
        byte_size=len(markdown_bytes),
        created_at=md_created_at,
    )


def import_external_markdown(
    source_file_id: str,
    *,
    markdown_text: str,
    converter_name: str = "external_import",
    root: Path | str | None = None,
) -> "MarkdownDocument":
    """Store EXTERNALLY-converted markdown for a source file WITHOUT running Marker.

    The GPU-free sibling of :func:`convert_source_file`: it records an HONEST
    ``conversion_runs`` row (``run_status='success'``,
    ``converter_name='external_import'`` so provenance shows Marker did NOT run) and
    stores the provided markdown through the SAME :func:`store_markdown_document`
    surface, yielding a ``markdown_documents`` row indistinguishable in shape from a
    converted one (``sections build`` / ``cite parse`` / bridge reconciliation all
    read it the same way). No dedup SELECT, no backend, no validation gate — the
    caller vouches for the markdown. Append-only (a re-import mints a fresh run; the
    content-addressed markdown row is idempotent on ``markdown_id``).
    """
    init_cache_db(root)
    conn = open_cache_db(root)
    try:
        src_row = conn.execute(
            "SELECT * FROM source_files WHERE source_file_id = ?", (source_file_id,)
        ).fetchone()
        if src_row is None:
            raise CacheError(
                f"import_external_markdown: unknown source_file_id {source_file_id!r}"
            )
        src = SourceFile(**dict(src_row))

        run_id = new_id("conv")
        now = _now_iso()
        markdown_hash = sha256_text(markdown_text)
        # HONEST provenance: converter_name says Marker never ran; the fingerprint is
        # namespaced so a later real Marker convert of the same PDF never dedup-reuses
        # this externally-imported markdown (its fingerprint is Marker-version derived).
        fingerprint = sha256_text(f"external_import:{markdown_hash}")
        conn.execute(
            "INSERT INTO conversion_runs "
            "(conversion_run_id, source_file_id, source_file_hash, converter_name, "
            " converter_package, converter_version, python_version, config_json, "
            " conversion_fingerprint, llm_enabled, llm_service, llm_model, "
            " llm_prompt_version, conversion_epistemic_type, run_status, "
            " warnings_json, error, markdown_hash, created_at, completed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                src.source_file_id,
                src.file_hash,
                converter_name,
                converter_name,
                "n/a",
                platform.python_version(),
                _canonical_json({}),
                fingerprint,
                0,
                None,
                None,
                None,
                ConversionEpistemicType.local_deterministic_conversion.value,
                RunStatus.success.value,
                None,
                None,
                markdown_hash,
                now,
                now,
            ),
        )
        md = store_markdown_document(
            conn,
            source_file_id=src.source_file_id,
            source_access_class=src.access_class,
            conversion_run_id=run_id,
            markdown_text=markdown_text,
            root=root,
        )
        record_event(
            conn,
            "conversion_run",
            run_id,
            "markdown_imported",
            {"markdown_id": md.markdown_id, "source_file_id": source_file_id,
             "converter_name": converter_name},
        )
        record_event(
            conn,
            "markdown_document",
            md.markdown_id,
            "markdown_written",
            {"conversion_run_id": run_id, "byte_size": md.byte_size},
        )
        conn.commit()
        return md
    finally:
        conn.close()


def convert_source_file(
    source_file_id: str,
    *,
    config: MarkerConfig = MarkerConfig(),
    backend: MarkerBackend | None = None,
    force: bool = False,
    allow_external_llm: bool = False,
    root: Path | str | None = None,
) -> "MarkdownDocument":
    """Convert a stored source file to validated markdown (or reuse a cache hit).

    Runs the §6 dedup+validate algorithm. ``backend`` defaults to
    :class:`LocalMarkerBackend` (or :class:`FakeMarkerBackend` under
    ``SEEDGRAPH_FAKE_MARKER`` for offline demos). Rejects ``file_type='html'`` with
    a clear deferred-feature error. Returns the canonical ``MarkdownDocument`` (the
    reused one on a cache hit, ``markdown_id == 'md_' + markdown_hash``); raises
    :class:`ConversionError` when validation's fatal gate trips or the backend
    raises.

    ``allow_external_llm`` (decision 15 / §8): an external-LLM assist on a
    ``user_supplied_private`` source is refused unless this override is passed —
    the document text would otherwise leave the machine. ``root`` (additive) lets
    the CLI thread its ``--root`` override.
    """
    if backend is None:
        backend = _default_backend()
    init_cache_db(root)
    conn = open_cache_db(root)
    try:
        src_row = conn.execute(
            "SELECT * FROM source_files WHERE source_file_id = ?", (source_file_id,)
        ).fetchone()
        if src_row is None:
            raise CacheError(f"convert_source_file: unknown source_file_id {source_file_id!r}")
        src = SourceFile(**dict(src_row))

        if src.file_type == FileType.html.value:
            raise CacheError(
                "HTML -> markdown conversion is deferred (phase_1 out of scope): Marker is "
                "PDF-only. The HTML source is ingested/hashed/deduped/stored, but "
                f"`convert` does not run for file_type='html' (source_file_id={source_file_id!r})."
            )

        # Content-access gate for external LLM on private documents (§8).
        if (
            config.use_llm
            and conversion_epistemic_type(config)
            == ConversionEpistemicType.local_conversion_with_external_llm_assist
            and AccessClass(src.access_class) == AccessClass.user_supplied_private
            and not allow_external_llm
        ):
            raise CacheError(
                "refusing external-LLM conversion of a user_supplied_private document: its text "
                "would leave the machine. Pass allow_external_llm=True (CLI --allow-external-llm) "
                "to override, or use --no-llm (the local, deterministic default)."
            )

        converter_version = backend.version
        fingerprint = conversion_fingerprint(converter_version, config)

        if not force:
            # SELECT-based dedup (no unique index): latest SUCCESS run for this
            # (source_file_hash, conversion_fingerprint), per plan §6. We drive off
            # conversion_runs (the dedup authority) and resolve the markdown by its
            # CONTENT hash (c.markdown_hash -> m.markdown_hash), NOT by the markdown
            # row's owning-run pointer. The single content-addressed markdown row can
            # only point at ONE producing run, so a pointer-join would miss a valid
            # prior success whose markdown was later repointed to a different
            # fingerprint's run (e.g. version bump then revert) and re-invoke Marker.
            hit = conn.execute(
                "SELECT m.* FROM conversion_runs c "
                "JOIN markdown_documents m ON m.markdown_hash = c.markdown_hash "
                "WHERE c.source_file_hash = ? AND c.conversion_fingerprint = ? "
                "AND c.run_status = 'success' AND c.markdown_hash IS NOT NULL "
                "ORDER BY c.created_at DESC, c.conversion_run_id DESC LIMIT 1",
                (src.file_hash, fingerprint),
            ).fetchone()
            if hit is not None:
                md = MarkdownDocument(**dict(hit))
                record_event(
                    conn,
                    "conversion_run",
                    md.conversion_run_id,
                    "conversion_reused",
                    {"markdown_id": md.markdown_id, "source_file_id": source_file_id},
                )
                conn.commit()
                return md

        # Mint a fresh run (append-only). --force lands here directly.
        run_id = new_id("conv")
        created_at = _now_iso()
        _insert_pending_run(
            conn,
            run_id=run_id,
            src=src,
            cfg=config,
            converter_version=converter_version,
            fingerprint=fingerprint,
            created_at=created_at,
        )
        record_event(
            conn,
            "conversion_run",
            run_id,
            "conversion_started",
            {"source_file_id": source_file_id, "conversion_fingerprint": fingerprint},
        )
        conn.commit()

        # Invoke the backend.
        pdf_path = store.resolve_uri(src.storage_uri, root)
        try:
            result = backend(pdf_path, config)
        except Exception as exc:  # backend failure -> failed run (append-only; retry mints fresh)
            _finish_run(
                conn, run_id, status=RunStatus.failed, warnings=[], error=f"backend error: {exc}"
            )
            record_event(
                conn, "conversion_run", run_id, "conversion_failed", {"error": str(exc)}
            )
            conn.commit()
            raise ConversionError(
                f"Marker backend failed for source_file_id={source_file_id!r}: {exc}"
            ) from exc

        validation = validate_markdown(result.markdown)
        warnings = list(result.warnings) + list(validation.warnings)
        if not validation.ok:
            fatal_reason = validation.warnings[0] if validation.warnings else "validation failed"
            _finish_run(
                conn,
                run_id,
                status=RunStatus.failed,
                warnings=warnings,
                error=f"markdown validation failed: {fatal_reason}",
            )
            record_event(
                conn, "conversion_run", run_id, "conversion_failed", {"error": fatal_reason}
            )
            conn.commit()
            raise ConversionError(
                f"conversion produced invalid markdown ({fatal_reason}) for "
                f"source_file_id={source_file_id!r}; run {run_id} marked failed, no markdown stored."
            )

        # Success: content-address-store the markdown + upsert the shared row (the
        # single markdown-storage surface, shared with the external-import path), then
        # finish this run.
        md = store_markdown_document(
            conn,
            source_file_id=src.source_file_id,
            source_access_class=src.access_class,
            conversion_run_id=run_id,
            markdown_text=result.markdown,
            root=root,
        )
        markdown_id = md.markdown_id
        markdown_hash = md.markdown_hash
        _finish_run(
            conn,
            run_id,
            status=RunStatus.success,
            warnings=warnings,
            error=None,
            markdown_hash=markdown_hash,
        )

        # Best-effort raw block/metadata dump (never parsed; doc 11 §7).
        if result.block_json is not None:
            meta_path = store.marker_dir(run_id, root) / "meta.json"
            store.write_bytes_atomic(
                json.dumps(result.block_json, indent=2, default=str).encode("utf-8"), meta_path
            )
        write_manifest(
            run_id,
            source_file_hash=src.file_hash,
            markdown_hash=markdown_hash,
            cfg=config,
            converter_version=converter_version,
            root=root,
        )
        record_event(
            conn,
            "conversion_run",
            run_id,
            "conversion_succeeded",
            {"markdown_id": markdown_id, "warnings": warnings},
        )
        record_event(
            conn,
            "markdown_document",
            markdown_id,
            "markdown_written",
            {"conversion_run_id": run_id, "byte_size": md.byte_size},
        )
        conn.commit()
        return md
    finally:
        conn.close()


def add_document(
    path: Path,
    *,
    access_class: AccessClass = AccessClass.user_supplied_private,
    acquisition_method: "AcquisitionMethod" = AcquisitionMethod.upload,
    config: MarkerConfig = MarkerConfig(),
    force: bool = False,
    backend: MarkerBackend | None = None,
    allow_external_llm: bool = False,
    root: Path | str | None = None,
) -> "MarkdownDocument":
    """Convenience: :func:`ingest_file` then :func:`convert_source_file` in one call.

    This is the milestone entry point: ``add_document(pdf)`` then
    ``add_document(same bytes)`` creates no new ``source_files`` row, runs Marker
    zero additional times, and resolves to the same ``markdown_id``. HTML is
    ingested but rejected at convert with a clear deferred message.

    ``backend`` (additive over the plan signature) lets callers/tests inject a
    :class:`FakeMarkerBackend` so the milestone's "Marker runs zero additional
    times" (``call_count``) is observable offline.
    """
    from .ingest import ingest_file  # local import avoids an ingest<->convert cycle

    src = ingest_file(
        path,
        access_class=access_class,
        acquisition_method=acquisition_method,
        root=root,
    )
    return convert_source_file(
        src.source_file_id,
        config=config,
        backend=backend,
        force=force,
        allow_external_llm=allow_external_llm,
        root=root,
    )
