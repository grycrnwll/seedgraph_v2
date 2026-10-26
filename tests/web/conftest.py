"""Shared fixtures for the web tests: an app on a temp home and a loopback client."""

from __future__ import annotations

import warnings

import pytest

from seedgraph.library.store import open_library_at

PORT = 8765
ORIGIN = f"http://127.0.0.1:{PORT}"


@pytest.fixture
def home(tmp_path):
    h = tmp_path / "home"
    open_library_at(h / "library").close()
    return h


@pytest.fixture
def make_app(home):
    from seedgraph.web.app import create_app

    def _make(**kw):
        kw.setdefault("port", PORT)
        return create_app(home, **kw)

    return _make


@pytest.fixture
def client_for():
    """``client_for(app, client=("127.0.0.1", 50000), base_url=ORIGIN)`` -> TestClient."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from starlette.testclient import TestClient

    made = []

    def _client(app, *, client=("127.0.0.1", 50000), base_url=ORIGIN, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c = TestClient(app, base_url=base_url, client=client, follow_redirects=False, **kw)
        made.append(c)
        return c

    yield _client
    for c in made:
        c.close()
