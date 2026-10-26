"""Live bind, Host and CSRF test (docs/V3_UI_DESIGN.md §10.2 #4, the live part).

Spawns ``seedgraph serve --root <tmp home> --no-open --port <free>`` and checks over real
sockets: it listens on ``127.0.0.1`` only, a connection to the machine's LAN address is
refused, a foreign ``Host`` gets 421, a POST without a token (or with a wrong one, or a
foreign ``Origin``) gets 403, and the positive controls (a GET with ``Host: localhost``,
a POST with the token) get 200. Opt-in: ``pytest -m live_serve``.

``python -m tests.acceptance.test_live_bind [--report DIR]`` runs the same checks outside
pytest and prints (and optionally writes) the results.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import pytest

from . import harness as h


def run_bind_checks(home: Path) -> dict:
    """Start serve on ``home`` and return every observation (no assertions)."""
    import httpx

    out: dict = {}
    with h.LiveServer(home) as server:
        port = server.port
        out["port"] = port
        out["listening"] = h.listening_endpoints(port)
        lan = h.lan_ipv4()
        out["lan_ip"] = lan
        out["lan_connect"] = h.try_connect(lan, port) if lan else "skipped: no LAN address"
        out["loopback_connect"] = h.try_connect("127.0.0.1", port)
        with httpx.Client(base_url=server.base_url, follow_redirects=False, timeout=30) as c:
            home_page = c.get("/")
            out["get_own_host"] = home_page.status_code
            out["get_localhost_host"] = c.get("/", headers={"Host": f"localhost:{port}"}).status_code
            out["get_foreign_host"] = c.get("/", headers={"Host": f"evil.example:{port}"}).status_code
            out["post_foreign_host"] = c.post("/library/add-dois", data={}, headers={
                "Host": f"evil.example:{port}"}).status_code
            out["get_ip_host_wrong_port"] = c.get("/", headers={"Host": f"127.0.0.1:{port + 1}"}).status_code
            token = h.CSRF_META.search(home_page.text).group(1)
            out["post_no_token"] = c.post("/library/add-dois", data={"dois": ""}).status_code
            out["post_wrong_token"] = c.post("/library/add-dois", data={"dois": "", "csrf_token": "x" * 43}).status_code
            out["post_foreign_origin"] = c.post("/library/add-dois", data={"dois": "", "csrf_token": token},
                                                headers={"Origin": "http://evil.example"}).status_code
            out["post_cross_site_fetch"] = c.post("/library/add-dois", data={"dois": "", "csrf_token": token},
                                                  headers={"Sec-Fetch-Site": "cross-site"}).status_code
            ok = c.post("/library/add-dois", data={"dois": "", "csrf_token": token}, headers={"HX-Request": "true"})
            out["post_with_token"] = ok.status_code
            out["post_with_token_header"] = c.post("/library/add-dois", data={"dois": ""}, headers={
                "X-CSRF-Token": token, "HX-Request": "true"}).status_code
            out["set_cookie_seen"] = any("set-cookie" in r.headers for r in (home_page, ok))
    out["server_output"] = server.output.strip()[-2000:]
    return out


def verdicts(obs: dict) -> dict[str, bool]:
    listening = obs["listening"]
    v = {
        "listens on 127.0.0.1 only": bool(listening) and all(a.startswith("127.0.0.1:") for a in listening),
        "loopback connect works": obs["loopback_connect"] == "connected",
        "foreign Host -> 421 (GET)": obs["get_foreign_host"] == 421,
        "foreign Host -> 421 (POST)": obs["post_foreign_host"] == 421,
        "wrong port in Host -> 421": obs["get_ip_host_wrong_port"] == 421,
        "GET with own Host -> 200": obs["get_own_host"] == 200,
        "GET with localhost Host -> 200": obs["get_localhost_host"] == 200,
        "POST without token -> 403": obs["post_no_token"] == 403,
        "POST with wrong token -> 403": obs["post_wrong_token"] == 403,
        "POST with foreign Origin -> 403": obs["post_foreign_origin"] == 403,
        "POST with Sec-Fetch-Site cross-site -> 403": obs["post_cross_site_fetch"] == 403,
        "POST with token (form field) -> 200": obs["post_with_token"] == 200,
        "POST with token (header) -> 200": obs["post_with_token_header"] == 200,
        "no Set-Cookie": not obs["set_cookie_seen"],
    }
    if obs["lan_ip"]:
        v["connection to the LAN address refused"] = obs["lan_connect"] == "refused"
    return v


def _temp_home(root: Path) -> Path:
    from seedgraph.library.store import open_library_at

    home = root / "home"
    open_library_at(home / "library").close()
    return home


@pytest.mark.live_serve
def test_live_bind_host_and_csrf(tmp_path):
    obs = run_bind_checks(_temp_home(tmp_path))
    if not obs["lan_ip"]:
        pytest.skip("no LAN address on this machine; the LAN refusal cannot be shown")
    failed = {k: v for k, v in verdicts(obs).items() if not v}
    assert not failed, (failed, {k: obs[k] for k in obs if k != "server_output"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="live bind / Host / CSRF check on a temp home")
    ap.add_argument("--report", type=Path, default=None, help="write bind.json into this folder")
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="sg-bind-") as tmp:
        obs = run_bind_checks(_temp_home(Path(tmp)))
    v = verdicts(obs)
    for name, ok in v.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    shown = {k: obs[k] for k in obs if k != "server_output"}
    print(json.dumps(shown, indent=1))
    if args.report:
        args.report.mkdir(parents=True, exist_ok=True)
        h.write_json(args.report / "bind.json", {"observations": obs, "verdicts": v})
    return 0 if all(v.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
