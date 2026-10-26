"""Shared pieces of the stage 3 acceptance harness (docs/V3_UI_DESIGN.md §10).

- **Run folders and inputs**: reports go to ``<home>/ui/acceptance/<run>/``; the
  researcher's choices (the PDF to drop, the paper for the edits-survive check) live in
  ``<home>/ui/acceptance/inputs.json`` (template: ``inputs.template.json`` next to this
  file, which holds no real values).
- **A live server**: :class:`LiveServer` spawns ``seedgraph serve --no-open`` on a free
  port with this interpreter (not through ``uv run``, so stopping it stops the server
  that holds ``run/converter.lock``) and stops it with CTRL_BREAK, then a kill.
- **A browserless client**: :class:`PageClient` speaks to the app the way the page does
  (the CSRF token read from ``<meta name="csrf-token">``, the ``HX-Request`` header for
  partial swaps), over a live socket or in-process through the full ASGI security stack.
- **Snapshots** for the edits-survive check: the current interpretation (agent and
  researcher audiences), the search result *sets* for fixed queries, and the claims MCP
  ``paper_show`` returns. :func:`canonical` gives the byte-exact form that is compared.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[2]
TEMPLATE = Path(__file__).with_name("inputs.template.json")
ACCEPTANCE = Path("ui") / "acceptance"
INPUTS = "inputs.json"
CSRF_META = re.compile(r'name="csrf-token" content="([^"]+)"')
VERSION_ATTR = re.compile(r'data-expected-version="(\d+)"')
NOTES_MESSAGE = re.compile(r'class="sg-flash sg-flash-(\w+)" role="alert" data-notes-message>([^<]*)<')


# --- run folders and inputs ------------------------------------------------------------------


def acceptance_dir(home: Path | str) -> Path:
    return Path(home) / ACCEPTANCE


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_run(home: Path | str, name: str | None = None) -> Path:
    run = acceptance_dir(home) / (name or stamp())
    run.mkdir(parents=True, exist_ok=False)
    return run


def load_inputs(home: Path | str) -> dict[str, Any]:
    path = acceptance_dir(home) / INPUTS
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_inputs(home: Path | str, section: str, values: dict[str, Any]) -> Path:
    """Merge ``values`` into ``inputs.json[section]`` (other keys kept; created from the
    template's shape when missing)."""
    path = acceptance_dir(home) / INPUTS
    path.parent.mkdir(parents=True, exist_ok=True)
    data = load_inputs(home) or json.loads(TEMPLATE.read_text(encoding="utf-8"))
    data.setdefault(section, {}).update(values)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def canonical(obj: Any) -> str:
    """The byte-exact JSON form compared by the edits-survive check."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def write_json(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --- sockets ---------------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lan_ipv4() -> str | None:
    """This machine's primary non-loopback IPv4 address (a UDP connect sends no packet),
    or ``None`` when it has none."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # TEST-NET-1: never routed, nothing is sent
            ip = s.getsockname()[0]
    except OSError:
        ip = None
    if ip and not ip.startswith("127.") and ip != "0.0.0.0":
        return ip
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addr = info[4][0]
            if not addr.startswith("127."):
                return addr
    except OSError:
        pass
    return None


def try_connect(host: str, port: int, timeout: float = 5.0) -> str:
    """``connected``, ``refused`` or ``timeout`` (or ``error: ...``) for a TCP connect."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "connected"
    except ConnectionRefusedError:
        return "refused"
    except (socket.timeout, TimeoutError):
        return "timeout"
    except OSError as exc:
        return f"error: {exc}"


def listening_endpoints(port: int) -> list[str]:
    """Every local ``address:port`` (any process) listening on TCP ``port``, read from
    ``netstat`` (Windows) or ``lsof`` elsewhere; ``[]`` when none is found.

    Matched on the port, not the PID: on Windows a venv's ``python.exe`` is a launcher
    whose child process owns the socket. The port was free when chosen, so a listener
    on it is the server."""
    found: list[str] = []
    if sys.platform == "win32":
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                             check=False).stdout
        out += subprocess.run(["netstat", "-ano", "-p", "TCPv6"], capture_output=True, text=True,
                              check=False).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0].upper().startswith("TCP") and parts[3].upper() == "LISTENING":
                if parts[1].rsplit(":", 1)[-1] == str(port):
                    found.append(parts[1])
        return found
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                         capture_output=True, text=True, check=False).stdout
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 9:
            found.append(parts[8])
    return found


class LiveServer:
    """``seedgraph serve --root HOME --no-open --port PORT`` as a child process."""

    def __init__(self, home: Path | str, port: int | None = None, *, startup_timeout: float = 60.0):
        self.home = Path(home)
        self.port = port or free_port()
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None
        self.output = ""

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "LiveServer":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "seedgraph", "serve", "--root", str(self.home), "--no-open",
             "--port", str(self.port)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=flags)
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.output = self.proc.stdout.read().decode("utf-8", "replace") if self.proc.stdout else ""
                raise RuntimeError(f"serve exited with {self.proc.returncode}: {self.output.strip()}")
            if try_connect("127.0.0.1", self.port, timeout=0.5) == "connected":
                return self
            time.sleep(0.2)
        self.stop()
        raise RuntimeError(f"serve did not listen on 127.0.0.1:{self.port} within {self.startup_timeout}s")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            if sys.platform == "win32":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=15)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            self.proc.kill()
            self.proc.wait(timeout=15)
        finally:
            if self.proc.stdout:
                try:
                    self.output += self.proc.stdout.read().decode("utf-8", "replace")
                except (OSError, ValueError):
                    pass

    def __enter__(self) -> "LiveServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class QuietServer:
    """The app on ``127.0.0.1:<free port>`` in a background thread **without its
    lifespan**: real sockets for a browser, but no JobRunner, so a real home's
    conversion queue is not resumed. For read-mostly browser passes."""

    def __init__(self, home: Path | str, port: int | None = None):
        self.home = Path(home)
        self.port = port or free_port()
        self._server = None
        self._thread = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "QuietServer":
        import threading

        import uvicorn

        from seedgraph.commands.serve import bind_loopback

        app = inprocess_app(self.home, port=self.port)
        sock = bind_loopback(self.port)
        config = uvicorn.Config(app, host="127.0.0.1", port=self.port, proxy_headers=False, server_header=False,
                                log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [sock]}, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 30
        while not self._server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=15)


# --- a browserless client --------------------------------------------------------------------


def inprocess_app(home: Path | str, *, port: int = 8765):
    """The app for ``home`` without its lifespan (so no JobRunner starts and the
    conversion queue is not resumed): safe on a real home."""
    from seedgraph.web.app import create_app

    return create_app(home, port=port)


def testclient(app, *, port: int = 8765):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from starlette.testclient import TestClient

        return TestClient(app, base_url=f"http://127.0.0.1:{port}", client=("127.0.0.1", 50000),
                          follow_redirects=False)


class PageClient:
    """GET/POST the way the page does. ``http`` is a ``TestClient`` or an ``httpx.Client``
    whose base URL is the app's own origin."""

    def __init__(self, http: Any):
        self.http = http
        self._token: str | None = None

    @classmethod
    def live(cls, base_url: str, timeout: float = 60.0) -> "PageClient":
        import httpx

        return cls(httpx.Client(base_url=base_url, follow_redirects=False, timeout=timeout))

    @classmethod
    def inprocess(cls, home: Path | str, *, port: int = 8765) -> "PageClient":
        return cls(testclient(inprocess_app(home, port=port), port=port))

    def token(self) -> str:
        if self._token is None:
            m = CSRF_META.search(self.get("/").text)
            if not m:
                raise RuntimeError("no csrf-token meta on /")
            self._token = m.group(1)
        return self._token

    def get(self, path: str, **kw):
        return self.http.get(path, **kw)

    def post(self, path: str, data: dict[str, Any] | None = None, *, htmx: bool = True, csrf: bool = True,
             headers: dict[str, str] | None = None, **kw):
        body = dict(data or {})
        if csrf:
            body = {"csrf_token": self.token(), **body}
        hdrs = dict(headers or {})
        if htmx:
            hdrs.setdefault("HX-Request", "true")
        return self.http.post(path, data=body, headers=hdrs, **kw)

    def close(self) -> None:
        self.http.close()


def pane_version(html: str) -> int:
    m = VERSION_ATTR.search(html)
    if not m:
        raise RuntimeError("the notes pane has no data-expected-version")
    return int(m.group(1))


def pane_message(html: str) -> tuple[str, str] | None:
    m = NOTES_MESSAGE.search(html)
    return (m.group(1), m.group(2).strip()) if m else None


# --- snapshots -------------------------------------------------------------------------------


def _interp_dict(interp) -> dict[str, Any] | None:
    if interp is None:
        return None
    d = asdict(interp)
    d.pop("newer_extraction")  # freshness is expected to change; compared separately
    return d


def interpretation_snapshot(lib, paper_id: str) -> dict[str, Any]:
    """Both audiences of ``notes.interpretation.current`` (the researcher's with hidden
    claims shown), the hidden set and the researcher notes."""
    from seedgraph.notes import interpretation

    agent = interpretation.current(lib, paper_id, "agent")
    researcher = interpretation.current(lib, paper_id, "researcher", show_hidden=True)
    return {
        "agent": _interp_dict(agent),
        "researcher": _interp_dict(researcher),
        "hidden": sorted(c.claim_id for c in (researcher.claims if researcher else ()) if c.hidden),
        "researcher_notes": researcher.researcher_notes if researcher else None,
    }


def search_sets(lib, queries: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    """For each ``{"query", "kind", "audience"}``: the sorted set of (id, text) pairs
    ``library.search`` returns with no limit. Scores and order are not compared."""
    from seedgraph.library import search

    out = []
    for q in queries:
        hits = search.search(lib, q["query"], kind=q["kind"], audience=q.get("audience", "agent"), limit=None)
        pairs = sorted({(canonical(list(h.key) if isinstance(h.key, tuple) else h.key), h.text) for h in hits})
        out.append({**q, "results": [list(p) for p in pairs]})
    return out


def paper_show_claims(lib, paper_id: str) -> list[dict[str, Any]] | None:
    """The claims MCP ``paper_show`` returns for the paper (the tool's own function)."""
    from seedgraph.mcp.tools_library import _paper_show

    note = _paper_show(lib, paper_id).get("current_note")
    return None if note is None else note["claims"]


def invariant(lib, paper_id: str, queries: Iterable[dict[str, str]]) -> dict[str, Any]:
    """Everything that must be byte-identical between the setup and the re-note."""
    return {"interpretation": interpretation_snapshot(lib, paper_id),
            "search": search_sets(lib, queries),
            "paper_show_claims": paper_show_claims(lib, paper_id)}


def freshness(lib, paper_id: str) -> dict[str, Any]:
    """The values a re-note is expected to move: ``note_latest`` and the newer-extraction
    flag (the interpretation's ``note_current`` stays pinned)."""
    from seedgraph.notes import interpretation
    from seedgraph.notes.format import SCHEMA_ID

    row = lib.conn.execute("SELECT note_id, version_no FROM note_latest WHERE paper_id = ? AND schema_id IS ?",
                           (paper_id, SCHEMA_ID)).fetchone()
    interp = interpretation.current(lib, paper_id, "researcher")
    return {"note_latest_id": row[0] if row else None, "note_latest_version": row[1] if row else None,
            "note_current_id": interp.note_id if interp else None,
            "newer_extraction": bool(interp and interp.newer_extraction)}
