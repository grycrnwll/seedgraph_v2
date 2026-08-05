"""`seedgraph secrets set|list|remove` + keyring-aware presence checks (ADR-0001).

Offline/keyless: every keyring interaction is monkeypatched (dict-backed store) —
the suite must never read or write this machine's real credential store. The
autouse ``isolated_home`` fixture nulls ``_keyring_get``; tests that need hits
re-patch it (their later patch wins).
"""

from __future__ import annotations

import keyring
from typer.testing import CliRunner

from seedgraph.cli import _env_present, app
from seedgraph.llm import secrets as secrets_mod
from seedgraph.project import service

cli = CliRunner()


def _store_backed(monkeypatch, store: dict):
    """Patch keyring writes and seedgraph reads onto one in-memory dict."""
    monkeypatch.setattr(
        keyring, "set_password", lambda svc, user, val: store.__setitem__((svc, user), val)
    )

    def _delete(svc, user):
        try:
            del store[(svc, user)]
        except KeyError:
            raise Exception(f"no such entry {svc}/{user}")  # noqa: TRY002 — stands in for PasswordDeleteError

    monkeypatch.setattr(keyring, "delete_password", _delete)
    monkeypatch.setattr(
        secrets_mod,
        "_keyring_get",
        lambda name: store.get(("seedgraph", name[len("seedgraph/"):])),
    )


# --- secrets set / list / remove round-trip ----------------------------------

def test_secrets_roundtrip_bare_alias(monkeypatch):
    store: dict = {}
    _store_backed(monkeypatch, store)

    # set: bare alias expands, value comes from the hidden prompt (never argv/echo).
    r = cli.invoke(app, ["secrets", "set", "openai"], input="sk-test-value\n")
    assert r.exit_code == 0, r.output
    assert "stored seedgraph/openai/default" in r.output
    assert "sk-test-value" not in r.output  # value never printed
    assert store[("seedgraph", "openai/default")] == "sk-test-value"

    # list: presence marker only, never the value.
    r = cli.invoke(app, ["secrets", "list"])
    assert r.exit_code == 0, r.output
    assert "seedgraph/openai/default\tset" in r.output
    assert "seedgraph/anthropic/default\tabsent" in r.output
    assert "sk-test-value" not in r.output

    # remove: deletes; a second remove is tolerated.
    r = cli.invoke(app, ["secrets", "remove", "openai"])
    assert r.exit_code == 0, r.output
    assert "removed seedgraph/openai/default" in r.output
    assert store == {}
    r = cli.invoke(app, ["secrets", "remove", "openai"])
    assert r.exit_code == 0, r.output
    assert "was not set" in r.output


def test_secrets_list_shows_all_conventional_names():
    r = cli.invoke(app, ["secrets", "list"])
    assert r.exit_code == 0, r.output
    for svc in ("anthropic", "core", "gemini", "openai", "openalex", "s2"):
        assert f"seedgraph/{svc}/default\tabsent" in r.output


def test_secrets_set_full_name_and_invalid_names(monkeypatch):
    store: dict = {}
    _store_backed(monkeypatch, store)

    r = cli.invoke(app, ["secrets", "set", "seedgraph/papersflow/default"], input="v\n")
    assert r.exit_code == 0, r.output
    assert store[("seedgraph", "papersflow/default")] == "v"

    # Too few segments / empty segment -> exit 2, nothing stored.
    for bad in ("seedgraph/openai", "seedgraph//default"):
        r = cli.invoke(app, ["secrets", "set", bad], input="v\n")
        assert r.exit_code == 2, r.output
        assert "invalid secret name" in r.output
    assert len(store) == 1


def test_secrets_set_blank_value_rejected(monkeypatch):
    store: dict = {}
    _store_backed(monkeypatch, store)
    # A whitespace-only value passes the prompt but fails our guard.
    r = cli.invoke(app, ["secrets", "set", "openai"], input=" \n")
    assert r.exit_code == 2, r.output
    assert "empty value" in r.output
    assert store == {}


def test_secrets_set_backend_write_fault_is_actionable(monkeypatch):
    def _boom(svc, user, val):
        raise RuntimeError("no backend")

    monkeypatch.setattr(keyring, "set_password", _boom)
    r = cli.invoke(app, ["secrets", "set", "openai"], input="v\n")
    assert r.exit_code == 2, r.output
    assert "keyring write failed" in r.output


# --- keyring-aware presence in llm keys / _env_present -----------------------

def test_env_present_auto_infers_leg_by_shape(monkeypatch):
    monkeypatch.setattr(
        secrets_mod, "_keyring_get",
        lambda name: "kv" if name == "seedgraph/openai/default" else None,
    )
    assert _env_present("auto", "seedgraph/openai/default") == "yes"
    assert _env_present("auto", "seedgraph/gemini/default") == "no"
    monkeypatch.setenv("SOME_KEY", "x")
    assert _env_present("auto", "SOME_KEY") == "yes"
    assert _env_present("auto", "OTHER_KEY") == "no"
    assert _env_present("bogus_source", "SOME_KEY") == "?"
    assert _env_present("auto", None) == "?"


def test_llm_keys_test_keyring_row(monkeypatch):
    service.create_project("krproj")
    r = cli.invoke(app, [
        "llm", "keys", "set", "--project", "krproj",
        "--provider", "anthropic", "--key-source", "keyring",
        "--env-var", "seedgraph/anthropic/default",
    ])
    assert r.exit_code == 0, r.output

    # conftest nulls the keyring -> absent.
    r = cli.invoke(app, ["llm", "keys", "test", "--project", "krproj"])
    assert r.exit_code == 1
    assert "present=no" in r.output

    monkeypatch.setattr(secrets_mod, "_keyring_get", lambda name: "sk-from-ring")
    r = cli.invoke(app, ["llm", "keys", "test", "--project", "krproj"])
    assert r.exit_code == 0, r.output
    assert "present=yes" in r.output

    r = cli.invoke(app, ["llm", "keys", "list", "--project", "krproj"])
    row = next(ln for ln in r.output.splitlines() if ln.startswith("anthropic"))
    assert "seedgraph/anthropic/default" in row and "yes" in row


# --- doctor reports keyring-resolved keys ------------------------------------

def test_doctor_key_ref_uses_chain(monkeypatch):
    from seedgraph.doctor import collect_checks

    # Keyring-only presence (no env var set) now flips the check to present.
    monkeypatch.setattr(
        secrets_mod, "_keyring_get",
        lambda name: "kv" if name == "seedgraph/anthropic/default" else None,
    )
    results = collect_checks()
    anth = next(r for r in results if r.name == "key_ref_anthropic_api_default")
    assert anth.ok, anth.detail
    assert "keyring seedgraph/anthropic/default" in anth.detail
