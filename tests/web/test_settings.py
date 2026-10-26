"""Settings (design §2.7): paths, converter, Ollama and which scholarly-API environment
variables are set (names only, never values). The Ollama probe runs only on the
"check again" POST; a GET never touches the network or the converter pool."""

from __future__ import annotations

import pytest

from seedgraph.llm.ollama import Reachability

from .pages_support import no_inline_script_or_style, post_forms_lead_with_csrf, token
from .test_jobs import FakePool


@pytest.fixture
def no_probe(monkeypatch):
    import seedgraph.conversion.pool as pool_mod
    import seedgraph.llm.ollama as ollama

    def boom(*a, **k):
        raise AssertionError("a GET must not probe Ollama or build the converter pool")

    monkeypatch.setattr(ollama, "check_reachable", boom)
    monkeypatch.setattr(pool_mod, "_POOL", None)
    monkeypatch.setattr(pool_mod, "pool_from_environment", boom)
    monkeypatch.setattr(pool_mod, "_nvidia_smi_gpu_count", boom)


def test_settings_shows_paths_and_the_converter_without_building_the_pool(home, make_app, client_for,
                                                                          no_probe, monkeypatch):
    import seedgraph.conversion.pool as pool_mod

    monkeypatch.setattr(pool_mod, "locked_marker_version", lambda project=None: "1.2.3-invented")
    html = client_for(make_app()).get("/settings").text
    assert str(home) in html and str(home / "library") in html and str(home / "graphs") in html
    assert "1.2.3-invented" in html
    assert "not started" in html  # no pool yet: nothing was probed
    assert no_inline_script_or_style(html) and post_forms_lead_with_csrf(html)


def test_settings_shows_a_running_pool(make_app, client_for, no_probe):
    html = client_for(make_app(pool=FakePool(size=2))).get("/settings").text
    assert "2 converter slots" in html


def test_scholarly_variables_are_listed_by_name_never_by_value(make_app, client_for, no_probe, monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "invented-secret-value-123")
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    html = client_for(make_app()).get("/settings").text
    assert "S2_API_KEY" in html and "OPENALEX_API_KEY" in html
    assert "invented-secret-value-123" not in html
    assert "<code>S2_API_KEY</code></td><td>set</td>" in html
    assert "<code>OPENALEX_API_KEY</code></td><td>not set</td>" in html


def test_ollama_is_checked_only_on_the_post_and_the_result_is_shown(make_app, client_for, no_probe,
                                                                     monkeypatch):
    import seedgraph.llm.ollama as ollama

    monkeypatch.setenv("SEEDGRAPH_OLLAMA_MODEL", "invented:1b")
    c = client_for(make_app())
    first = c.get("/settings").text
    assert "not checked yet" in first and "invented:1b" in first

    calls = []

    def fake_check(base_url=None, **kw):
        calls.append(base_url)
        return Reachability(ok=True, base_url="http://localhost:11434", models=["invented:1b", "other:7b"])

    monkeypatch.setattr(ollama, "check_reachable", fake_check)
    assert c.post("/settings/ollama/check").status_code == 403  # no token
    r = c.post("/settings/ollama/check", data={"csrf_token": token(c)})
    assert r.status_code == 303 and len(calls) == 1
    html = c.get(r.headers["location"]).text
    assert "reachable" in html and "other:7b" in html


def test_an_unreachable_ollama_says_so(make_app, client_for, no_probe, monkeypatch):
    import seedgraph.llm.ollama as ollama

    monkeypatch.setattr(ollama, "check_reachable", lambda base_url=None, **kw: Reachability(
        ok=False, base_url="http://localhost:11434", error="connection refused (invented)"))
    c = client_for(make_app())
    r = c.post("/settings/ollama/check", data={"csrf_token": token(c)})
    assert "not reachable" in c.get(r.headers["location"]).text
