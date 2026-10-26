"""`seedgraph serve [--port 8765] [--no-open] [--root PATH]`: the web UI
(docs/V3_UI_DESIGN.md §1.1, §6, §8). A thin adapter over
:func:`seedgraph.web.app.create_app`.

- Always binds ``127.0.0.1``; there is no ``--host`` option. The socket is bound here
  (exclusively on Windows) so a busy port fails with a plain message and is never
  shared or swapped for another port.
- Holds ``<home>/run/converter.lock`` for its lifetime, so ``convert`` / ``reconvert``
  refuse while it runs; a second ``serve`` on the same home refuses too.
- uvicorn runs with ``proxy_headers=False``: a forwarded-for header can never make a
  remote client look like loopback.
"""

from __future__ import annotations

import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any, Optional

import typer

from . import _common

DEFAULT_PORT = 8765


def bind_loopback(port: int) -> socket.socket:
    """A listening-ready TCP socket on ``127.0.0.1:port`` that no other socket shares."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform == "win32":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(128)
    except OSError:
        sock.close()
        raise
    return sock


def server_config(app: Any, port: int):
    import uvicorn  # local: keep CLI start-up light

    return uvicorn.Config(app, host="127.0.0.1", port=port, proxy_headers=False,
                          server_header=False, log_level="warning", lifespan="on")


def run_server(app: Any, sock: socket.socket) -> None:
    import uvicorn

    host, port = sock.getsockname()[:2]
    uvicorn.Server(server_config(app, port)).run(sockets=[sock])


def open_browser(url: str) -> None:
    threading.Timer(1.0, webbrowser.open, (url,)).start()


def serve(
    port: int = typer.Option(DEFAULT_PORT, "--port", min=1, max=65535,
                             help="Port on 127.0.0.1 (never another address)."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the browser."),
    root: Optional[Path] = _common.ROOT_OPTION,
) -> None:
    """Serve the web UI on http://127.0.0.1:PORT/ (this computer only)."""
    from ..library.store import library_db_path, library_dir
    from ..web.jobs import ConverterLockHeld, acquire_converter_lock

    home = _common.home(root)
    lib_dir = library_dir(home)
    if not library_db_path(lib_dir).is_file():
        _common.fail(f"no library at {lib_dir}; run `seedgraph setup` or `seedgraph migrate run` first")
    url = f"http://127.0.0.1:{port}/"
    try:
        lock = acquire_converter_lock(home)
    except ConverterLockHeld as exc:
        _common.fail(f"seedgraph serve is already running for this home (pid {exc.pid}); "
                     f"open it in the browser or stop it first")
        return
    try:
        try:
            sock = bind_loopback(port)
        except OSError as exc:
            _common.fail(f"port {port} is already in use on 127.0.0.1 ({exc.strerror or exc}); "
                         f"stop the other program or choose another port with --port")
            return
        from ..web.app import create_app  # local: Starlette/Jinja2 only when serving

        app = create_app(home, port=port)
        typer.echo(f"seedgraph is serving {home} at {url} (Ctrl+C to stop)")
        if not no_open:
            open_browser(url)
        try:
            run_server(app, sock)
        finally:
            sock.close()
    finally:
        lock.release()
