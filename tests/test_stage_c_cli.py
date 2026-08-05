"""Stage C (LLM CLI group + doctor checks + cost controls) — offline/keyless.

Covers the new ``llm profiles|keys|usage|estimate`` verbs, the Stage C doctor
checks (pricing-snapshot age, capability + structured-output mismatch,
private-content policy, opt-in ``--probe-ollama``), and the cost controls
(monthly soft limit wired to ``monthly_spend + this_run`` + the CLI
``require_confirmation_above_usd`` gate). No paid call is ever made: providers
are never dispatched (default ``llm keys test`` is presence-only) and the
confirmation gate uses a local-cost-$0 route.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.config.models import GlobalConfig, LlmBudget
from seedgraph.db.bootstrap import ensure_project_db
from seedgraph.db.connection import open_project_db
from seedgraph.doctor import collect_checks, has_failure
from seedgraph.llm.cost import budget_status, this_month
from seedgraph.llm.usage import UsageEvent, log_usage
from seedgraph.project import service

cli = CliRunner()


# --- llm profiles list ------------------------------------------------------

def test_llm_profiles_list_global_and_routes():
    r = cli.invoke(app, ["llm", "profiles", "list"])
    assert r.exit_code == 0, r.output
    assert "anthropic_api_default" in r.output
    assert "local_ollama_default" in r.output
    # the routing table is appended
    assert "note_extraction" in r.output
    assert "answer_generation" in r.output


def test_llm_profiles_availability_registry_aware(monkeypatch):
    # local is always available; anthropic flips on the key being present.
    r0 = cli.invoke(app, ["llm", "profiles", "list"])
    anth0 = [ln for ln in r0.output.splitlines() if ln.startswith("anthropic_api_default")]
    assert anth0 and anth0[0].split("\t")[-1] == "0"  # no key -> unavailable
    local0 = [ln for ln in r0.output.splitlines() if ln.startswith("local_ollama_default")]
    assert local0 and local0[0].split("\t")[-1] == "1"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r1 = cli.invoke(app, ["llm", "profiles", "list"])
    anth1 = [ln for ln in r1.output.splitlines() if ln.startswith("anthropic_api_default")]
    assert anth1 and anth1[0].split("\t")[-1] == "1"  # key present -> available


# --- llm keys (reference-only; never values) --------------------------------

def test_llm_keys_set_list_remove_no_values():
    service.create_project("keysproj")
    r = cli.invoke(app, [
        "llm", "keys", "set", "--project", "keysproj",
        "--provider", "anthropic", "--env-var", "ANTHROPIC_API_KEY",
    ])
    assert r.exit_code == 0, r.output
    assert "no value stored" in r.output

    r = cli.invoke(app, ["llm", "keys", "list", "--project", "keysproj"])
    assert r.exit_code == 0, r.output
    assert "anthropic" in r.output
    assert "ANTHROPIC_API_KEY" in r.output  # the env-var NAME is the reference

    # The table has no secret value: confirm no value column exists in llm_key_refs.
    conn = open_project_db("keysproj")
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_key_refs)")}
        assert {"key_value", "secret", "value", "api_key"}.isdisjoint(cols)
        stored = conn.execute("SELECT key_reference, env_var FROM llm_key_refs").fetchone()
        assert tuple(stored) == ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    finally:
        conn.close()

    r = cli.invoke(app, [
        "llm", "keys", "remove", "--project", "keysproj",
        "--provider", "anthropic", "--env-var", "ANTHROPIC_API_KEY",
    ])
    assert r.exit_code == 0, r.output
    assert "removed 1" in r.output
    r = cli.invoke(app, ["llm", "keys", "list", "--project", "keysproj"])
    assert "no key references" in r.output


def test_llm_keys_test_presence_only_no_dispatch(monkeypatch):
    service.create_project("ktest")
    cli.invoke(app, [
        "llm", "keys", "set", "--project", "ktest",
        "--provider", "anthropic", "--env-var", "ANTHROPIC_API_KEY",
    ])
    # Absent key (conftest scrubs it) -> present=no, non-zero exit, NO dispatch.
    r = cli.invoke(app, ["llm", "keys", "test", "--project", "ktest"])
    assert r.exit_code == 1
    assert "present=no" in r.output
    assert "smoke" not in r.output  # default makes no paid call

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r2 = cli.invoke(app, ["llm", "keys", "test", "--project", "ktest"])
    assert r2.exit_code == 0, r2.output
    assert "present=yes" in r2.output
    # presence test stamped last_validated_at but logged NO usage row (no call).
    conn = open_project_db("ktest")
    try:
        assert conn.execute("SELECT COUNT(*) FROM llm_usage_events").fetchone()[0] == 0
        assert conn.execute(
            "SELECT last_validated_at FROM llm_key_refs"
        ).fetchone()[0] is not None
    finally:
        conn.close()


# --- llm usage --------------------------------------------------------------

def test_llm_usage_aggregates_by_dimension():
    ensure_project_db("usageproj")
    conn = open_project_db("usageproj")
    try:
        log_usage(conn, UsageEvent(
            task_type="note_extraction", provider="anthropic", model="claude-sonnet-4-6",
            input_tokens=10, output_tokens=5, estimated_cost=0.01,
        ))
        log_usage(conn, UsageEvent(
            task_type="answer_generation", provider="anthropic", model="claude-sonnet-4-6",
            input_tokens=20, output_tokens=8, estimated_cost=0.02,
        ))
    finally:
        conn.close()

    r = cli.invoke(app, ["llm", "usage", "--project", "usageproj", "--group-by", "provider"])
    assert r.exit_code == 0, r.output
    assert "anthropic" in r.output
    assert "TOTAL" in r.output
    assert "0.030000" in r.output  # summed estimated cost

    r2 = cli.invoke(app, ["llm", "usage", "--project", "usageproj", "--group-by", "task"])
    assert r2.exit_code == 0
    assert "note_extraction" in r2.output and "answer_generation" in r2.output

    r3 = cli.invoke(app, ["llm", "usage", "--project", "usageproj", "--group-by", "nonsense"])
    assert r3.exit_code == 2


# --- llm estimate (no dispatch) ---------------------------------------------

def test_llm_estimate_real_route(monkeypatch):
    service.create_project("estproj")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    r = cli.invoke(app, [
        "llm", "estimate", "--project", "estproj", "--task", "answer_generation",
        "--input-tokens", "1000000", "--output-tokens", "1000000",
    ])
    assert r.exit_code == 0, r.output
    assert "claude-sonnet-4-6" in r.output
    # 1M in @ $3 + 1M out @ $15 = $18
    assert "18.000000" in r.output
    # no dispatch happened: no usage rows written by estimate.
    conn = open_project_db("estproj")
    try:
        assert conn.execute("SELECT COUNT(*) FROM llm_usage_events").fetchone()[0] == 0
    finally:
        conn.close()


def test_llm_estimate_no_llm_task():
    service.create_project("estproj2")
    r = cli.invoke(app, [
        "llm", "estimate", "--project", "estproj2", "--task", "reference_parsing",
        "--input-tokens", "1000",
    ])
    assert r.exit_code == 0, r.output
    assert "no LLM route" in r.output


# --- doctor: Stage C LLM checks ---------------------------------------------

def test_doctor_llm_checks_present_and_pass():
    results = collect_checks()
    names = {r.name for r in results}
    assert "pricing_snapshot_age" in names
    assert "llm_capability_coverage" in names
    assert "llm_structured_output_match" in names
    assert "private_content_policy" in names
    assert not has_failure(results), [r for r in results if not r.ok and r.severity == "error"]
    assert "ollama_probe" not in names  # not probed unless requested


def test_doctor_probe_ollama_opt_in():
    results = collect_checks(probe_ollama=True)
    names = {r.name for r in results}
    assert "ollama_probe" in names
    # a closed Ollama port is a warning, never a hard failure.
    assert not has_failure(results)


def test_doctor_cli_probe_flag():
    r = cli.invoke(app, ["doctor", "--probe-ollama"])
    assert r.exit_code == 0, r.output
    assert "ollama_probe" in r.output
    assert "RESULT: OK" in r.output


def test_doctor_structured_output_mismatch_warns():
    # A project route that requires structured output from llama3 (no native API)
    # surfaces a WARNING, not a failure.
    service.create_project("smis")
    proj_yaml = Path(os.environ["SEEDGRAPH_HOME"]) / "projects" / "smis" / "project.yaml"
    data = yaml.safe_load(proj_yaml.read_text()) if proj_yaml.exists() else {}
    data = data or {}
    data.setdefault("llm", {}).setdefault("routes", {})["note_extraction"] = {
        "task_type": "note_extraction",
        "preferred_profile": "local_ollama_default",
        "requires_structured_output": True,
    }
    proj_yaml.write_text(yaml.safe_dump(data))
    results = collect_checks(slug="smis")
    match = next(r for r in results if r.name == "llm_structured_output_match")
    assert not match.ok and match.severity == "warning"
    assert not has_failure(results)


# --- cost controls ----------------------------------------------------------

def test_budget_status_wires_monthly_spend_plus_this_run():
    ensure_project_db("bstat")
    conn = open_project_db("bstat")
    try:
        log_usage(conn, UsageEvent(task_type="t", estimated_cost=0.5))
        budget = LlmBudget(monthly_soft_limit_usd=0.6, require_confirmation_above_usd=0.3)
        st = budget_status(conn, budget, 0.4)
        assert st.year_month == this_month()
        assert st.prior_spend_usd == 0.5
        assert st.this_run_usd == 0.4
        assert st.projected_usd == 0.9  # monthly_spend + this_run
        assert st.over_monthly_soft_limit  # 0.9 > 0.6
        assert st.requires_confirmation  # 0.4 >= 0.3
    finally:
        conn.close()


def test_runner_monthly_limit_uses_prior_month_spend():
    from seedgraph.extraction.runner import BudgetState, _account_budget

    cfg = GlobalConfig(budget=LlmBudget(monthly_soft_limit_usd=1.0, stop_on_budget_exceeded=True))
    # Seeded prior spend pushes this run over the monthly soft limit.
    seeded = BudgetState(prior_month_spent_usd=0.9)
    _account_budget(seeded, cfg, 0.2)  # 0.9 + 0.2 = 1.1 >= 1.0
    assert seeded.stopped and seeded.stop_reason == "monthly_soft_limit_usd exceeded"
    # Without the prior seed the same run is under the limit (0.2 < 1.0).
    fresh = BudgetState()
    _account_budget(fresh, cfg, 0.2)
    assert not fresh.stopped


def test_extract_notes_confirmation_gate(monkeypatch):
    from test_phase_4 import _make_doc, good_note_dict

    from seedgraph.extraction import runner as runner_mod
    from seedgraph.llm.backend import FakeLLMBackend

    monkeypatch.setattr(runner_mod, "_BACKEND_OVERRIDE", FakeLLMBackend(response=good_note_dict()))
    _make_doc("confirmgate")

    # Arm the confirmation gate: any run (even $0 local) is at/above the threshold.
    proj_yaml = Path(os.environ["SEEDGRAPH_HOME"]) / "projects" / "confirmgate" / "project.yaml"
    data = yaml.safe_load(proj_yaml.read_text()) if proj_yaml.exists() else {}
    data = data or {}
    data["budget"] = {"require_confirmation_above_usd": 0.0}
    proj_yaml.write_text(yaml.safe_dump(data))

    # --yes skips the prompt and proceeds.
    r = cli.invoke(app, ["extract", "notes", "confirmgate", "--yes"])
    assert r.exit_code == 0, r.output
    assert "success" in r.output

    # Declining the prompt aborts before any work (exit 1), prompt fired pre-loop.
    r2 = cli.invoke(app, ["extract", "notes", "confirmgate"], input="n\n")
    assert r2.exit_code == 1, r2.output
    assert "aborted" in r2.output


def test_extract_notes_default_config_no_cost_prompt(monkeypatch):
    # Default config (no monthly limit / no confirmation threshold) is a no-op:
    # the cost-control path adds no output and never prompts.
    from test_phase_4 import _make_doc, good_note_dict

    from seedgraph.extraction import runner as runner_mod
    from seedgraph.llm.backend import FakeLLMBackend

    monkeypatch.setattr(runner_mod, "_BACKEND_OVERRIDE", FakeLLMBackend(response=good_note_dict()))
    _make_doc("nogate")
    r = cli.invoke(app, ["extract", "notes", "nogate"])
    assert r.exit_code == 0, r.output
    assert "success" in r.output
    assert "[budget]" not in r.output
    assert "aborted" not in r.output
