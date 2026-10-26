"""`seedgraph setup`: create the v3 home and library.db, check the converter env and GPUs.

Idempotent. The converter check is static (folder, lock file, uv on PATH, GPU count);
it never starts a converter. It also lists which scholarly-API environment variables
are set (names only) and, with ``--probe-ollama``, asks the local Ollama server for its
models (no model call). These replaced the v2 ``doctor`` command (design §1.3).
``seedgraph setup agents`` (in :mod:`.mcp`) is mounted on the same group.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import typer

from ..conversion import pool
from ..library.store import ORIGIN_SETUP, is_dirty, library_db_path, library_dir, open_library
from . import _common

setup_app = typer.Typer(name="setup", help="Create the v3 home and library; `setup agents` registers "
                                           "the MCP server and skills with Claude Code and Codex.")


@setup_app.callback(invoke_without_command=True)
def setup(
    ctx: typer.Context,
    root: Optional[Path] = _common.ROOT_OPTION,
    json_out: bool = _common.JSON_OPTION,
    probe_ollama: bool = typer.Option(
        False, "--probe-ollama", help="Also check that the local Ollama server answers (no model call)."),
) -> None:
    """Create the v3 home and its library (if missing) and check the PDF converter."""
    if ctx.invoked_subcommand is not None:
        return
    with _common.cli_errors():
        home = _common.home(root)
        lib_dir = library_dir(home)
        created = not library_db_path(lib_dir).is_file()
        with open_library(home, new_origin=ORIGIN_SETUP) as lib:
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
            "scholarly_env": _scholarly_env(),
        }
        if probe_ollama:
            report["ollama"] = _ollama()
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
    typer.echo("scholarly API variables (optional): "
               + ", ".join(f"{name}={'set' if on else 'unset'}" for name, on in report["scholarly_env"].items()))
    if probe_ollama:
        o = report["ollama"]
        if o["reachable"]:
            typer.echo(f"ollama: reachable at {o['base_url']} ({', '.join(o['models'][:8]) or 'no models pulled'})")
        else:
            typer.echo(f"ollama: not reachable at {o['base_url']} ({o['error']}); `work ollama` needs it")
    if not report["converter_found"] or not report["uv_found"]:
        typer.echo("warning: PDF conversion will not work until the converter environment is set up")


def _scholarly_env() -> dict[str, bool]:
    """Which scholarly-API environment variables are set (names only, never values)."""
    from ..providers import SCHOLARLY_ENV_VARS, env_key

    return {name: bool(env_key(name)) for name in SCHOLARLY_ENV_VARS}


def _ollama() -> dict:
    """``GET /api/tags`` on the local Ollama server; unreachable is information, not an error."""
    from ..llm import ollama

    r = ollama.check_reachable(timeout=1.0)
    return {"reachable": bool(r.ok), "base_url": r.base_url, "models": list(r.models), "error": r.error}
