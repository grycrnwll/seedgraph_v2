"""Build F ops slice (chunks 5, 6, 7, 9) — config snapshot/fingerprint round-trip,
``doctor --json`` shape contract, the fail-loud import guard's symbol list, and the
done-vs-crashed ``interrupted`` sweep.

Offline / keyless throughout: CLI verbs run under Typer's ``CliRunner`` on the
autouse throwaway ``SEEDGRAPH_HOME``. (The old web UI's serve-sweep and events-route
tests went with web/ in stage 2.)
"""

from __future__ import annotations

import importlib

from typer.testing import CliRunner

from seedgraph.config.loader import (
    _FORBIDDEN_SECRET_KEYS,
    config_fingerprint,
    effective_config_snapshot,
)

runner = CliRunner()


# --------------------------------------------------------------------------
# ch7 — fail-loud contract import guard (the guard's real test is CI turning
# red when a symbol vanishes; this pins that the list itself stays importable).
# --------------------------------------------------------------------------

def test_contract_symbol_list_is_importable():
    import conftest

    for module_name, symbol in conftest._CONTRACT_SYMBOLS:
        module = importlib.import_module(module_name)
        assert hasattr(module, symbol), f"{module_name}.{symbol} missing"
    conftest._assert_contract_symbols()  # must not raise on a healthy tree


# --------------------------------------------------------------------------
# ch5 — effective_config_snapshot + config_fingerprint
# --------------------------------------------------------------------------

def _all_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _all_keys(item)


def test_fingerprint_is_key_order_invariant():
    assert config_fingerprint({"a": 1, "b": [2, 3]}) == config_fingerprint({"b": [2, 3], "a": 1})
    assert config_fingerprint({"a": 1}) != config_fingerprint({"a": 2})


def test_snapshot_strips_secret_named_keys_from_hostile_mapping():
    """Defense-in-depth: even a hostile merged patch never lands a secret-named
    field in the snapshot (values under legitimate keys survive)."""
    hostile = {
        "llm": {"profiles": {"p": {"api_key": "sk-live-oops", "model": "m"}}},
        "token": "raw",
        "nested": [{"password": "x", "keep": 1}],
    }
    snap = effective_config_snapshot(hostile)
    forbidden = {k.lower() for k in _FORBIDDEN_SECRET_KEYS}
    assert not [k for k in _all_keys(snap) if isinstance(k, str) and k.lower() in forbidden]
    assert snap["llm"]["profiles"]["p"]["model"] == "m"
    assert snap["nested"][0]["keep"] == 1


# --------------------------------------------------------------------------
# ch6 — doctor --json
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# ch9 — done-vs-crashed: `interrupted` terminal + mark_interrupted_runs sweep
# --------------------------------------------------------------------------


