"""Build C chunk 10 (D11): opt-in Haiku-class profile for concept canonicalization.

``anthropic_api_cheap`` ships in ``default_profiles()`` restricted to
``semantic_graph_extraction`` (canon is guardrail-disposed subset selection, so
the cheap model is strictly better there); no default route references it
(ADR-0006 local-first) — users opt in via ``llm route set``.

Offline / keyless throughout: the autouse ``isolated_home`` fixture strips real
provider keys and stubs the OS keyring; tests set a fake ``ANTHROPIC_API_KEY``
where profile availability matters.
"""

from __future__ import annotations

from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.config.loader import load_project_config
from seedgraph.config.models import (
    ContentPolicy,
    LLMConfig,
    ProjectConfig,
    TaskRoute,
    default_llm_config,
    default_profiles,
    default_routes,
)
from seedgraph.llm.routing import NoLlmRoute, Route, resolve_route

runner = CliRunner()


def _home(tmp_path):
    return tmp_path / "home"


# --- profile shape + default-config validation -------------------------------


def test_cheap_profile_in_defaults_with_expected_fields():
    prof = default_profiles()["anthropic_api_cheap"]
    assert prof.provider == "anthropic"
    assert prof.model == "claude-haiku-4-5"
    assert prof.env_var == "ANTHROPIC_API_KEY"
    assert prof.access_mode == "api_key"
    # Canon-only: the allowed_tasks fence is the point of D11.
    assert prof.allowed_tasks == ["semantic_graph_extraction"]


def test_default_config_validates_and_no_default_route_references_it():
    # Constructing LLMConfig runs the route→profile referential validator.
    cfg = default_llm_config()
    assert "anthropic_api_cheap" in cfg.profiles
    # ADR-0006 local-first: strictly opt-in, never a default route target.
    for route in default_routes().values():
        assert route.preferred_profile != "anthropic_api_cheap"
        assert route.fallback_profile != "anthropic_api_cheap"


# --- CLI surface ---------------------------------------------------------------


def test_llm_profiles_list_shows_cheap_profile(tmp_path):
    res = runner.invoke(app, ["llm", "profiles", "list", "--root", str(_home(tmp_path))])
    assert res.exit_code == 0, res.output
    line = next(
        ln for ln in res.output.splitlines() if ln.startswith("anthropic_api_cheap\t")
    )
    assert "\tanthropic\t" in line
    assert "\tclaude-haiku-4-5\t" in line


def test_route_set_round_trips_and_resolve_route_selects_it(tmp_path, monkeypatch):
    root = _home(tmp_path)
    res = runner.invoke(
        app,
        [
            "llm", "route", "set",
            "--task", "semantic_graph_extraction",
            "--preferred", "anthropic_api_cheap",
            "--project", "proj",
            "--root", str(root),
        ],
    )
    assert res.exit_code == 0, res.output
    cfg = load_project_config("proj", root)
    assert (
        cfg.llm.routes["semantic_graph_extraction"].preferred_profile
        == "anthropic_api_cheap"
    )

    # With the key present, the resolver actually selects the cheap profile.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    route = resolve_route("semantic_graph_extraction", "open_access", cfg)
    assert isinstance(route, Route)
    assert route.profile_id == "anthropic_api_cheap"
    assert route.provider == "anthropic"


# --- allowed_tasks fence ---------------------------------------------------------


def _cfg(routes):
    return ProjectConfig(
        slug="t",
        llm=LLMConfig(profiles=default_profiles(), routes=routes),
        content_policy=ContentPolicy(),
    )


def test_allowed_tasks_blocks_note_extraction_falls_to_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = _cfg(
        {
            "note_extraction": TaskRoute(
                task_type="note_extraction",
                preferred_profile="anthropic_api_cheap",
                fallback_profile="anthropic_api_default",
                requires_source_text=True,
            )
        }
    )
    route = resolve_route("note_extraction", "open_access", cfg)
    assert isinstance(route, Route)
    # The fence skipped the cheap profile despite its key being present.
    assert route.profile_id == "anthropic_api_default"


def test_allowed_tasks_blocks_note_extraction_no_fallback_degrades(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg = _cfg(
        {
            "note_extraction": TaskRoute(
                task_type="note_extraction",
                preferred_profile="anthropic_api_cheap",
                requires_source_text=True,
            )
        }
    )
    # Nothing authorized for the task → honest NoLlmRoute, never a fabricated pick.
    assert isinstance(resolve_route("note_extraction", "open_access", cfg), NoLlmRoute)
