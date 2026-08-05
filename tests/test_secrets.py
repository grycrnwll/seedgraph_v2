import logging

import pytest

from seedgraph.config.models import LLMProfile
from seedgraph.errors import ConfigError
from seedgraph.llm import secrets
from seedgraph.llm.secrets import (
    # Bound at import time (pre-fixture): conftest nulls the module attribute for
    # hermeticity, so this name is the only way to reach the real implementation.
    _keyring_get as real_keyring_get,
    resolve_named_secret,
    resolve_profile_key,
    resolve_secret,
)


def _profile(**kw) -> LLMProfile:
    kw.setdefault("profile_id", "p")
    kw.setdefault("provider", "anthropic")
    return LLMProfile(**kw)


# --- resolve_secret (explicit source; reference-only) ------------------------

def test_env_var_present_returns_value(monkeypatch):
    monkeypatch.setenv("MYKEY", "topsecret")
    assert resolve_secret("environment", "MYKEY") == "topsecret"


def test_env_var_absent_returns_none(monkeypatch):
    monkeypatch.delenv("MYKEY", raising=False)
    assert resolve_secret("environment", "MYKEY") is None


def test_keyring_source_resolves_via_store(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring_get", {"seedgraph/openai/default": "rk"}.get)
    assert resolve_secret("system_keyring", "seedgraph/openai/default") == "rk"


def test_unknown_source_raises():
    with pytest.raises(ConfigError):
        resolve_secret("vault", "anything")


def test_secret_value_not_logged(monkeypatch, caplog):
    monkeypatch.setenv("MYKEY", "topsecret")
    with caplog.at_level(logging.DEBUG):
        assert resolve_secret("environment", "MYKEY") == "topsecret"
    assert "topsecret" not in caplog.text


# --- keyring name validation --------------------------------------------------

def test_invalid_name_prefix_raises():
    # Real implementation: the prefix guard fires before any keyring import.
    with pytest.raises(ConfigError):
        real_keyring_get("openai/default")


def test_bare_prefix_raises():
    with pytest.raises(ConfigError):
        real_keyring_get("seedgraph/")


# --- resolve_named_secret (shared chain for acquisition keys, ADR-0001) -------

def test_named_secret_keyring_wins_over_env(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring_get", {"seedgraph/s2/default": "ring"}.get)
    monkeypatch.setenv("S2_API_KEY", "env")
    assert resolve_named_secret("seedgraph/s2/default", env_var="S2_API_KEY") == "ring"


def test_named_secret_env_fallback(monkeypatch):
    monkeypatch.setenv("S2_API_KEY", "env")
    assert resolve_named_secret("seedgraph/s2/default", env_var="S2_API_KEY") == "env"


def test_named_secret_total_miss_returns_none(monkeypatch):
    monkeypatch.delenv("S2_API_KEY", raising=False)
    assert resolve_named_secret("seedgraph/s2/default", env_var="S2_API_KEY") is None


# --- resolve_profile_key (the auto chain, ADR-0002) ---------------------------

def test_auto_keyring_wins_over_env(monkeypatch):
    monkeypatch.setattr(
        secrets, "_keyring_get", {"seedgraph/anthropic/default": "ring"}.get
    )
    monkeypatch.setenv("AKEY", "env")
    assert resolve_profile_key(_profile(env_var="AKEY")) == "ring"


def test_auto_keyring_miss_falls_to_env(monkeypatch):
    monkeypatch.setenv("AKEY", "env")
    assert resolve_profile_key(_profile(env_var="AKEY")) == "env"


def test_auto_total_miss_returns_none(monkeypatch):
    monkeypatch.delenv("AKEY", raising=False)
    assert resolve_profile_key(_profile(env_var="AKEY")) is None


def test_auto_respects_key_name_override(monkeypatch):
    monkeypatch.setattr(secrets, "_keyring_get", {"seedgraph/anthropic/work": "ring"}.get)
    assert resolve_profile_key(_profile(key_name="seedgraph/anthropic/work")) == "ring"


def test_auto_no_keyring_backend_falls_to_env(monkeypatch):
    """Headless CI simulation: NoKeyringError behaves exactly like a miss."""
    import keyring
    import keyring.errors

    def boom(service, username):
        raise keyring.errors.NoKeyringError("headless")

    monkeypatch.setattr(secrets, "_keyring_get", real_keyring_get)
    monkeypatch.setattr(keyring, "get_password", boom)
    monkeypatch.setenv("AKEY", "env")
    assert resolve_profile_key(_profile(env_var="AKEY")) == "env"


def test_explicit_environment_never_touches_keyring(monkeypatch):
    def fail(name):
        raise AssertionError("keyring consulted for key_source=environment")

    monkeypatch.setattr(secrets, "_keyring_get", fail)
    monkeypatch.setenv("AKEY", "env")
    assert resolve_profile_key(_profile(env_var="AKEY", key_source="environment")) == "env"


def test_explicit_keyring_never_reads_env(monkeypatch):
    monkeypatch.setenv("AKEY", "env")
    profile = _profile(env_var="AKEY", key_source="system_keyring")
    assert resolve_profile_key(profile) is None


def test_unknown_profile_source_raises():
    with pytest.raises(ConfigError):
        resolve_profile_key(_profile(key_source="vault"))


# --- executor injection --------------------------------------------------------

def test_build_backend_injects_keyring_resolved_key(monkeypatch):
    from seedgraph.config.models import ProjectConfig
    from seedgraph.llm.executor import build_backend

    monkeypatch.setattr(
        secrets, "_keyring_get", {"seedgraph/anthropic/default": "ring-key"}.get
    )
    monkeypatch.delenv("UNSET_VAR", raising=False)
    profile = _profile(env_var="UNSET_VAR", model="m")
    backend = build_backend(profile, ProjectConfig(slug="proj"))
    assert backend._api_key == "ring-key"
