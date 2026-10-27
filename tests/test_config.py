import pytest
import yaml

from seedgraph.config.loader import (
    clamp_project_ceiling,
    load_global_config,
    load_project_config,
    write_project_overrides,
)
from seedgraph.errors import ConfigError
from seedgraph.paths import project_dir, resolve_home


def _write_yaml(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _global_yaml(data):
    _write_yaml(resolve_home() / "config.yaml", data)


def _project_yaml(slug, data):
    _write_yaml(project_dir(slug) / "project.yaml", data)


def test_defaults_load_in_clean_env():
    cfg = load_global_config()
    assert "no_llm" in cfg.llm.profiles
    assert "anthropic_api_default" in cfg.llm.profiles
    assert "note_extraction" in cfg.llm.routes


def test_routing_missing_profile_raises_config_error():
    _write_yaml(
        resolve_home() / "config.yaml",
        {"llm": {"routes": {"bad": {"task_type": "bad", "preferred_profile": "ghost"}}}},
    )
    with pytest.raises(ConfigError) as exc:
        load_global_config()
    assert "ghost" in str(exc.value)


def test_project_overrides_merge_over_global():
    _write_yaml(
        resolve_home() / "config.yaml",
        {"log_level": "DEBUG", "budget": {"usd_limit": 10.0}},
    )
    _write_yaml(
        project_dir("proj") / "project.yaml",
        {"log_level": "WARNING"},
    )
    cfg = load_project_config("proj")
    assert cfg.slug == "proj"
    assert cfg.log_level == "WARNING"      # project wins
    assert cfg.budget.usd_limit == 10.0    # global value preserved


# ==========================================================================
# Settings inheritance: tighten-only ceilings (clamp_project_ceiling)
# ==========================================================================

def test_ceiling_clamps_privacy_bool_project_cannot_loosen():
    # global gate is CLOSED; project attempts to OPEN it (loosen) -> clamped closed.
    _global_yaml({"content_policy": {"external_llm_for_private_full_text": False}})
    _project_yaml("proj", {"content_policy": {"external_llm_for_private_full_text": True}})
    cfg = load_project_config("proj")
    assert cfg.content_policy.external_llm_for_private_full_text is False


def test_ceiling_privacy_bool_project_may_tighten():
    # global gate is OPEN; project TIGHTENS it (allowed) -> effective closed.
    _global_yaml({"content_policy": {"external_llm_for_private_full_text": True}})
    _project_yaml("proj", {"content_policy": {"external_llm_for_private_full_text": False}})
    cfg = load_project_config("proj")
    assert cfg.content_policy.external_llm_for_private_full_text is False


def test_ceiling_clamps_usd_limit_above_global():
    _global_yaml({"budget": {"usd_limit": 5.0}})
    # project above the global cap is clamped DOWN to the global ceiling
    _project_yaml("hi", {"budget": {"usd_limit": 999.0}})
    assert load_project_config("hi").budget.usd_limit == 5.0
    # project BELOW the cap is honored verbatim (it is the tighter value)
    _project_yaml("lo", {"budget": {"usd_limit": 2.0}})
    assert load_project_config("lo").budget.usd_limit == 2.0


def test_ceiling_min_no_global_cap_lets_project_set_any_value():
    # global usd_limit unset (None == unlimited) -> project may impose any cap.
    _project_yaml("proj", {"budget": {"usd_limit": 12.0}})
    assert load_project_config("proj").budget.usd_limit == 12.0


def test_ceiling_allow_unverified_pricing_cannot_loosen():
    _global_yaml({"budget": {"allow_unverified_pricing": False}})
    _project_yaml("proj", {"budget": {"allow_unverified_pricing": True}})
    assert load_project_config("proj").budget.allow_unverified_pricing is False


def test_ceiling_answer_policy_external_search_cannot_loosen():
    _global_yaml({"answer_policy": {"allow_external_search_by_default": False}})
    _project_yaml("proj", {"answer_policy": {"allow_external_search_by_default": True}})
    assert load_project_config("proj").answer_policy.allow_external_search_by_default is False


def test_ceiling_intersects_route_access_classes():
    _global_yaml(
        {"llm": {"routes": {"note_extraction": {
            "allowed_access_classes": ["open_access", "user_supplied_private"]}}}}
    )
    # project set is NOT a subset -> effective = project ∩ global == [open_access]
    _project_yaml(
        "proj",
        {"llm": {"routes": {"note_extraction": {
            "allowed_access_classes": ["open_access", "restricted"]}}}},
    )
    route = load_project_config("proj").llm.routes["note_extraction"]
    assert route.allowed_access_classes == ["open_access"]


def test_ceiling_route_access_classes_cannot_widen_to_null():
    # a hand-edited project.yaml tries to REMOVE the global constraint (null == open).
    _global_yaml(
        {"llm": {"routes": {"note_extraction": {
            "allowed_access_classes": ["open_access"]}}}}
    )
    _project_yaml(
        "proj",
        {"llm": {"routes": {"note_extraction": {"allowed_access_classes": None}}}},
    )
    route = load_project_config("proj").llm.routes["note_extraction"]
    assert route.allowed_access_classes == ["open_access"]  # clamped back to global


def test_clamp_project_ceiling_robust_to_missing_sections():
    # empty / partial dicts must not raise; global-None ceiling keeps the project value.
    assert clamp_project_ceiling({}, {}) == {}
    merged = {"budget": {"usd_limit": 10.0}}
    out = clamp_project_ceiling({}, merged)
    assert out["budget"]["usd_limit"] == 10.0


def test_ceiling_privacy_bool_none_global_fails_closed():
    # A malformed config.yaml with an explicit `null` privacy bool must NOT let a
    # project loosen it: a None global ceiling clamps the 'and' gate to False (closed).
    out = clamp_project_ceiling(
        {"content_policy": {"external_llm_for_private_full_text": None}},
        {"content_policy": {"external_llm_for_private_full_text": True}},
    )
    assert out["content_policy"]["external_llm_for_private_full_text"] is False


# ==========================================================================
# Settings inheritance: LIVE inheritance of unset fields
# ==========================================================================

def test_live_inheritance_unset_field_tracks_current_global():
    _global_yaml({"budget": {"usd_limit": 7.0}})
    _project_yaml("proj", {"log_level": "WARNING"})  # never sets budget
    assert load_project_config("proj").budget.usd_limit == 7.0
    # rewrite the global; the project (which never overrode it) sees the NEW value.
    _global_yaml({"budget": {"usd_limit": 3.0}})
    assert load_project_config("proj").budget.usd_limit == 3.0


def test_live_inheritance_answer_policy_from_global():
    # answer_policy has a GLOBAL anchor now, so an unset project field inherits it.
    _global_yaml({"answer_policy": {"require_project_sources": False}})
    _project_yaml("proj", {"log_level": "INFO"})  # no answer_policy override
    assert load_project_config("proj").answer_policy.require_project_sources is False


# ==========================================================================
# Settings inheritance: free overrides (no ceiling)
# ==========================================================================

def test_free_override_routing_profile_choice_wins():
    _global_yaml(
        {"llm": {"routes": {"note_extraction": {
            "preferred_profile": "local_ollama_default"}}}}
    )
    # profile CHOICE is a free override (never a ceiling).
    _project_yaml(
        "proj",
        {"llm": {"routes": {"note_extraction": {
            "preferred_profile": "anthropic_api_default"}}}},
    )
    route = load_project_config("proj").llm.routes["note_extraction"]
    assert route.preferred_profile == "anthropic_api_default"


def test_free_override_require_project_sources_wins():
    _global_yaml({"answer_policy": {"require_project_sources": True}})
    _project_yaml("proj", {"answer_policy": {"require_project_sources": False}})
    assert load_project_config("proj").answer_policy.require_project_sources is False


# ==========================================================================
# write_project_overrides: thin overlay, validation, secret-safety
# ==========================================================================

def test_write_project_overrides_writes_thin_overlay():
    write_project_overrides("thin", {"budget": {"usd_limit": 3.0}})
    raw = yaml.safe_load((project_dir("thin") / "project.yaml").read_text(encoding="utf-8"))
    # only the overridden group is materialized; omitted keys stay inherited.
    assert raw == {"budget": {"usd_limit": 3.0}}
    assert load_project_config("thin").budget.usd_limit == 3.0


def test_write_project_overrides_rejects_raw_secret():
    with pytest.raises(ConfigError):
        write_project_overrides(
            "proj", {"llm": {"profiles": {"x": {"api_key": "sk-should-never-persist"}}}}
        )
    assert not (project_dir("proj") / "project.yaml").exists()


def test_write_project_overrides_rejects_invalid_patch_before_writing():
    # a route pointing at an unknown profile fails runtime validation -> nothing written.
    with pytest.raises(ConfigError):
        write_project_overrides(
            "badp", {"llm": {"routes": {"note_extraction": {"preferred_profile": "ghost"}}}}
        )
    assert not (project_dir("badp") / "project.yaml").exists()


# ==========================================================================
# Credential boundary: a project may CHOOSE a profile but never DEFINE/REDIRECT one
# ==========================================================================

def test_write_project_overrides_rejects_llm_profiles_patch():
    # PROBE2: a project tries to ship its OWN profile registry (define/redirect
    # an endpoint + bind a machine-global key) -> refused, nothing written.
    with pytest.raises(ConfigError):
        write_project_overrides(
            "prof",
            {"llm": {"profiles": {"anthropic_api_default": {
                "base_url": "http://attacker.example/v1"}}}},
        )
    assert not (project_dir("prof") / "project.yaml").exists()


def test_write_project_overrides_rejects_profile_definition_field_anywhere():
    # a profile-definition field (base_url) anywhere under llm is refused even
    # without a top-level `profiles` key -> nothing written.
    with pytest.raises(ConfigError):
        write_project_overrides(
            "burl",
            {"llm": {"routes": {"note_extraction": {
                "base_url": "http://attacker.example/v1"}}}},
        )
    assert not (project_dir("burl") / "project.yaml").exists()


def test_write_project_overrides_allows_pure_profile_choice():
    # the ONE legitimate llm override — a route profile CHOICE among machine-global
    # profiles — still writes a thin overlay.
    write_project_overrides(
        "choice",
        {"llm": {"routes": {"note_extraction": {
            "preferred_profile": "anthropic_api_default"}}}},
    )
    route = load_project_config("choice").llm.routes["note_extraction"]
    assert route.preferred_profile == "anthropic_api_default"


def test_load_project_config_ignores_handedited_profile_base_url():
    # PROBE1: a hand-edited project.yaml redirects an existing profile's endpoint
    # AND rebinds its key. The live load path must strip both back to the
    # machine-global registry (defense-in-depth, no write API involved).
    _project_yaml(
        "hand",
        {"llm": {"profiles": {"anthropic_api_default": {
            "base_url": "http://attacker.example/v1",
            "env_var": "ATTACKER_KEY"}}}},
    )
    prof = load_project_config("hand").llm.profiles["anthropic_api_default"]
    assert prof.base_url is None                 # endpoint NOT redirected (global)
    assert prof.env_var == "ANTHROPIC_API_KEY"   # key NOT rebound (global)


# ==========================================================================
# Clobber-safety: override groups survive the declarative writer
# ==========================================================================

