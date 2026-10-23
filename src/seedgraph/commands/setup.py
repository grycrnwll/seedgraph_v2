"""`seedgraph setup`: create the v3 home and library.db, check the converter env and GPUs.

Idempotent. The converter check is static (folder, lock file, uv on PATH, GPU count);
it never starts a converter.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer

from ..conversion import pool
from ..library.store import is_dirty, library_db_path, library_dir, open_library
from . import _common


def setup(
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
) -> None:
    """Create the v3 home and its library (if missing) and check the PDF converter."""
    with _common.cli_errors():
        home = _common.home(root)
        lib_dir = library_dir(home)
        created = not library_db_path(lib_dir).is_file()
        with open_library(home) as lib:
            lib.graphs_root.mkdir(parents=True, exist_ok=True)
            dirty = is_dirty(lib.conn)
        conv_dir = pool.converter_dir()
        try:
            gpus: int | None = pool._nvidia_smi_gpu_count()
        except Exception:  # noqa: BLE001 - informational only
            gpus = None
        report = {
            "home": str(home),
            "library": str(library_db_path(lib_dir)),
            "library_created": created,
            "library_dirty": dirty,
            "graphs": str(lib_dir.parent / "graphs"),
            "converter_dir": str(conv_dir),
            "converter_found": conv_dir.is_dir(),
            "marker_version": pool.locked_marker_version(),
            "uv_found": shutil.which("uv") is not None,
            "gpus": gpus,
        }
    if json_out:
        _common.echo_json(report)
        return
    typer.echo(f"home: {home}")
    typer.echo(f"library: {report['library']} ({'created' if created else 'already exists'})")
    typer.echo(f"graphs: {report['graphs']}")
    typer.echo(f"converter: {conv_dir} ({'found' if report['converter_found'] else 'MISSING'}); "
               f"marker-pdf {report['marker_version'] or 'unknown'}; "
               f"uv {'found' if report['uv_found'] else 'NOT on PATH'}")
    typer.echo(f"gpus: {'unknown' if gpus is None else gpus} (nvidia-smi)")
    if not report["converter_found"] or not report["uv_found"]:
        typer.echo("warning: PDF conversion will not work until the converter environment is set up")
