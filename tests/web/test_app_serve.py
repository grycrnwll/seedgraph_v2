"""``seedgraph serve`` (design §1.1, §6, §8): binds 127.0.0.1 only, refuses a busy port
plainly, holds the converter lock while it runs, opens the browser unless --no-open.

The server loop itself is replaced by a recorder (``serve.run_server``), so no real
server starts; the bound socket is real.
"""

from __future__ import annotations

import socket

import pytest
from typer.testing import CliRunner

from seedgraph.cli import app as cli_app
from seedgraph.commands import serve as serve_mod
from seedgraph.web.jobs import acquire_converter_lock, converter_lock_holder


@pytest.fixture
def recorded(monkeypatch, home):
    calls = {"run": [], "open": []}

    def fake_run(app, sock):
        calls["run"].append((app, sock.getsockname()))
        calls["lock_during_run"] = converter_lock_holder(home)

    monkeypatch.setattr(serve_mod, "run_server", fake_run)
    monkeypatch.setattr(serve_mod, "open_browser", lambda url: calls["open"].append(url))
    return calls


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def invoke(*args):
    return CliRunner().invoke(cli_app, ["serve", *args])


def test_serve_binds_loopback_and_holds_the_lock_while_running(home, recorded):
    import os

    port = free_port()
    r = invoke("--root", str(home), "--port", str(port), "--no-open")
    assert r.exit_code == 0, r.output
    [(app, sockname)] = recorded["run"]
    assert sockname == ("127.0.0.1", port)
    assert app.state.web.port == port and app.state.web.home == home
    assert recorded["lock_during_run"] == os.getpid()
    assert converter_lock_holder(home) is None  # released at exit
    assert recorded["open"] == []
    assert f"http://127.0.0.1:{port}/" in r.output


def test_serve_opens_the_browser_by_default(home, recorded):
    port = free_port()
    r = invoke("--root", str(home), "--port", str(port))
    assert r.exit_code == 0, r.output
    assert recorded["open"] == [f"http://127.0.0.1:{port}/"]


def test_serve_has_no_host_option(home, recorded):
    r = invoke("--root", str(home), "--host", "0.0.0.0", "--no-open")
    assert r.exit_code == 2
    assert recorded["run"] == []


def test_a_busy_port_fails_with_a_plain_message(home, recorded):
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        port = busy.getsockname()[1]
        r = invoke("--root", str(home), "--port", str(port), "--no-open")
    assert r.exit_code == 3
    assert f"port {port} is already in use" in r.output
    assert recorded["run"] == []
    assert converter_lock_holder(home) is None


def test_serve_refuses_while_another_serve_holds_the_converters(home, recorded):
    with acquire_converter_lock(home):
        r = invoke("--root", str(home), "--port", str(free_port()), "--no-open")
    assert r.exit_code == 3
    assert "already running" in r.output
    assert recorded["run"] == []


def test_serve_refuses_a_home_without_a_library(tmp_path, recorded):
    r = invoke("--root", str(tmp_path / "empty"), "--port", str(free_port()), "--no-open")
    assert r.exit_code == 3
    assert "no library" in r.output
    assert not (tmp_path / "empty" / "library").exists()


def test_the_real_server_config_is_loopback_without_proxy_headers(home):
    port = free_port()
    config = serve_mod.server_config(object(), port)
    assert config.host == "127.0.0.1" and config.port == port
    assert config.proxy_headers is False  # X-Forwarded-For can never fake a loopback client
