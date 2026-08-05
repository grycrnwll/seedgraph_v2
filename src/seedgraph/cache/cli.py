"""Typer ``cache`` sub-app (plan §6 CLI surface).

Defines the ``seedgraph cache …`` verbs, mounted on the root CLI as
``seedgraph cache``. Command bodies call the cache library (ingest / convert /
read) and print human-readable provenance; errors surface as a clear message +
non-zero exit (never a traceback).

Conversion verbs (``add`` / ``convert``) use the real ``LocalMarkerBackend`` by
default; set ``SEEDGRAPH_FAKE_MARKER=1`` to drive a deterministic
:class:`FakeMarkerBackend` for an offline round-trip without ``marker-pdf``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..db.connection import open_cache_db
from ..errors import SeedgraphError
from ..vocab import AccessClass, AcquisitionMethod
from . import store
from .convert import MarkerConfig, add_document, convert_source_file
from .db import init_cache_db
from .ingest import ingest_file

cache_app = typer.Typer(
    name="cache",
    help="Local content cache: ingest + Marker-convert PDFs (content-addressed).",
    no_args_is_help=True,
)

_root_option = typer.Option(None, "--root", help="Override the seedgraph home directory.")


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def _resolve_source_file_id(selector: str, root: Optional[Path]) -> str:
    """Map a ``file_hash`` or ``source_file_id`` selector to a ``source_file_id``."""
    if selector.startswith("sf_"):
        return selector
    init_cache_db(root)
    conn = open_cache_db(root)
    try:
        row = conn.execute(
            "SELECT source_file_id FROM source_files WHERE file_hash = ?", (selector,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        _fail(f"no source file with file_hash or id {selector!r}")
    return row[0]


@cache_app.command("add")
def add(
    path: Path = typer.Argument(..., help="PDF (or HTML) file to ingest + convert."),
    access_class: str = typer.Option(
        "user_supplied_private", "--access-class", help="Content-access class (fail-closed default)."
    ),
    acquisition_method: str = typer.Option(
        "upload", "--acquisition-method", help="upload | folder_import | open_access_fetch."
    ),
    use_llm: bool = typer.Option(False, "--use-llm/--no-llm", help="Enable Marker LLM-assist."),
    allow_external_llm: bool = typer.Option(
        False, "--allow-external-llm", help="Permit external-LLM assist on private documents (§8)."
    ),
    force: bool = typer.Option(False, "--force", help="Force a new conversion run even on a cache hit."),
    root: Optional[Path] = _root_option,
) -> None:
    """Ingest + convert a PDF in one step (convenience over ingest + convert)."""
    try:
        md = add_document(
            path,
            access_class=AccessClass(access_class),
            acquisition_method=AcquisitionMethod(acquisition_method),
            config=MarkerConfig(use_llm=use_llm),
            force=force,
            allow_external_llm=allow_external_llm,
            root=root,
        )
    except (SeedgraphError, ValueError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        json.dumps(
            {
                "markdown_id": md.markdown_id,
                "markdown_hash": md.markdown_hash,
                "source_file_id": md.source_file_id,
                "conversion_run_id": md.conversion_run_id,
                "storage_uri": md.storage_uri,
                "byte_size": md.byte_size,
            },
            indent=2,
        )
    )


@cache_app.command("ingest")
def ingest(
    path: Path = typer.Argument(..., help="File to hash, dedup, and store."),
    access_class: str = typer.Option("user_supplied_private", "--access-class"),
    acquisition_method: str = typer.Option("upload", "--acquisition-method"),
    root: Optional[Path] = _root_option,
) -> None:
    """Ingest (hash/dedup/store) a source file without converting it."""
    try:
        src = ingest_file(
            path,
            access_class=AccessClass(access_class),
            acquisition_method=AcquisitionMethod(acquisition_method),
            root=root,
        )
    except (SeedgraphError, ValueError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        json.dumps(
            {
                "source_file_id": src.source_file_id,
                "file_hash": src.file_hash,
                "file_type": src.file_type,
                "access_class": src.access_class,
                "storage_uri": src.storage_uri,
                "byte_size": src.byte_size,
            },
            indent=2,
        )
    )


@cache_app.command("convert")
def convert(
    selector: str = typer.Argument(..., help="file_hash or source_file_id to convert."),
    use_llm: bool = typer.Option(False, "--use-llm/--no-llm"),
    allow_external_llm: bool = typer.Option(
        False, "--allow-external-llm", help="Permit external-LLM assist on private documents (§8)."
    ),
    force: bool = typer.Option(False, "--force"),
    root: Optional[Path] = _root_option,
) -> None:
    """Convert an already-ingested PDF (HTML rejected as deferred)."""
    source_file_id = _resolve_source_file_id(selector, root)
    try:
        md = convert_source_file(
            source_file_id,
            config=MarkerConfig(use_llm=use_llm),
            force=force,
            allow_external_llm=allow_external_llm,
            root=root,
        )
    except (SeedgraphError, ValueError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        json.dumps(
            {
                "markdown_id": md.markdown_id,
                "markdown_hash": md.markdown_hash,
                "source_file_id": md.source_file_id,
                "conversion_run_id": md.conversion_run_id,
                "storage_uri": md.storage_uri,
            },
            indent=2,
        )
    )


@cache_app.command("show")
def show(
    selector: str = typer.Argument(..., help="file_hash or markdown_hash."),
    root: Optional[Path] = _root_option,
) -> None:
    """Print provenance / manifest for a source file or markdown document."""
    init_cache_db(root)
    conn = open_cache_db(root)
    try:
        sf = conn.execute(
            "SELECT * FROM source_files WHERE file_hash = ? OR source_file_id = ?",
            (selector, selector),
        ).fetchone()
        md = conn.execute(
            "SELECT * FROM markdown_documents WHERE markdown_hash = ? OR markdown_id = ?",
            (selector, selector),
        ).fetchone()
        runs = []
        if sf is not None:
            runs = conn.execute(
                "SELECT * FROM conversion_runs WHERE source_file_id = ? "
                "ORDER BY created_at ASC",
                (sf["source_file_id"],),
            ).fetchall()
        elif md is not None:
            runs = conn.execute(
                "SELECT * FROM conversion_runs WHERE conversion_run_id = ?",
                (md["conversion_run_id"],),
            ).fetchall()
    finally:
        conn.close()

    if sf is None and md is None:
        _fail(f"no source file or markdown document matches {selector!r}")

    payload = {
        "source_file": dict(sf) if sf is not None else None,
        "markdown_document": dict(md) if md is not None else None,
        "conversion_runs": [dict(r) for r in runs],
    }
    typer.echo(json.dumps(payload, indent=2))


@cache_app.command("stat")
def stat(root: Optional[Path] = _root_option) -> None:
    """Print cache counts, total bytes, and the resolved cache-root path."""
    init_cache_db(root)
    conn = open_cache_db(root)
    try:

        def _count(table: str) -> int:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

        def _bytes(table: str) -> int:
            return conn.execute(f"SELECT COALESCE(SUM(byte_size), 0) FROM {table}").fetchone()[0]

        summary = {
            "cache_root": str(store.cache_root(root)),
            "source_files": _count("source_files"),
            "conversion_runs": _count("conversion_runs"),
            "markdown_documents": _count("markdown_documents"),
            "cache_events": _count("cache_events"),
            "source_bytes": _bytes("source_files"),
            "markdown_bytes": _bytes("markdown_documents"),
        }
    finally:
        conn.close()
    typer.echo(json.dumps(summary, indent=2))
