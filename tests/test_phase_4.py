"""Phase 4 — default note extractor acceptance tests (plan §11), fully OFFLINE.

Every plan §11 unit/acceptance test is implemented with real assertions and runs
with NO network, NO real LLM, and NO key: a synthetic markdown is pushed through
the real cache (ingest + FakeMarkerBackend convert), bridged work->markdown, then
extracted through the phase-4 surface with a deterministic FakeLLMBackend that
returns canned ``default_research_note_v1`` JSON + token counts. The module-level
imports double as a whole-phase import-cleanliness check.

Contracts exercised: vocab normalize_claim_type; the field-agnostic schema
(data_sources + setting); the JSON validator (valid / wrapped / repair /
unrepairable); normalize (found/not_found/inferred rows, decision-47 assumption
typing, D2 epistemic/assertion independence); resolve_source_access_class
(oa/private/missing fail-closed); staleness + --force; the runner happy path
(claim->span coverage, FTS, provenance, access_class on run+note+claims+SPANS,
raw_note_json round-trip); span access_class propagation (should_fix #2);
anchor-miss downgrade-to-ambiguous; the five non-success run statuses
(extraction_failed / skipped_oversize / skipped_policy / skipped_budget /
skipped_no_llm); the routing identifier; FK enforcement; D6 ORM/migration parity
(incl. the nullable FK rulings); D7 raw_conn atomicity; and no-stemming FTS.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlmodel import Session
from typer.testing import CliRunner

# Import-clean surface checks (collection fails loudly if a module is broken).
from seedgraph import cache_access
from seedgraph.acquisition.bridge import write_bridge
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.cli import app
from seedgraph.config.models import (
    ContentPolicy,
    GlobalConfig,
    LlmBudget,
    LlmCapabilities,
    LLMConfig,
    LLMProfile,
    ModelCapability,
    TaskRoute,
    default_profiles,
)
from seedgraph.db.adapter import raw_conn
from seedgraph.db.fts import ensure_fts_tables, reindex_claim_fts, reindex_note_fts  # noqa: F401
from seedgraph.db.migrations import run_migrations
from seedgraph.db.models_project import ExtractedClaim, ExtractionRun, StructuredNote
from seedgraph.extraction import runner as runner_mod
from seedgraph.extraction.normalize import (
    _derive_archetype,
    map_assumption_claim_type,
    normalize_note,
)
from seedgraph.extraction.prompt import build_prompt  # noqa: F401
from seedgraph.extraction.runner import (
    BudgetState,
    current_note,
    extract_note,
    resolve_source_access_class,
)
from seedgraph.extraction.schema import (
    PROMPT_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    DefaultNoteV1,
)
from seedgraph.extraction.validator import extract_json_object, validate_note
from seedgraph.ids import new_id
from seedgraph.llm.backend import FakeLLMBackend
from seedgraph.project import service
from seedgraph.vocab import AccessClass, AcquisitionMethod, normalize_claim_type

cli = CliRunner()


# --------------------------------------------------------------------------
# Offline fixtures / helpers
# --------------------------------------------------------------------------

MD = (
    "# Minimum Wage and Teen Employment\n\n"
    "We study whether minimum wage increases reduce teen employment.\n\n"
    "## Contribution\n\n"
    "Our main contribution is a new difference-in-differences estimator.\n\n"
    "## Methods\n\n"
    "We use a two-way fixed effects regression model.\n\n"
    "## Data\n\n"
    "We rely on the Current Population Survey microdata.\n\n"
    "## Setting\n\n"
    "The setting is the United States labor market in the 1990s.\n\n"
    "## Identification\n\n"
    "We rely on the parallel trends assumption for identification.\n\n"
    "## Results\n\n"
    "We find that teen employment fell by three percent.\n\n"
    "## Limitations\n\n"
    "A key limitation is the short panel length.\n"
)


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def good_note_dict() -> dict:
    """A full canned note whose every ``found`` quote is a unique verbatim MD substring."""
    return {
        "research_question": {
            "claim_text": "Does the minimum wage reduce teen employment?",
            "normalized_label": "research question",
            "status": "found",
            "assertion_status": "stated",
            "confidence": 0.9,
            "exact_quote": "We study whether minimum wage increases reduce teen employment.",
        },
        "main_contribution": {
            "claim_text": "A new difference-in-differences estimator.",
            "status": "found",
            "assertion_status": "stated",
            "exact_quote": "Our main contribution is a new difference-in-differences estimator.",
        },
        "method_or_model": [
            {
                "claim_text": "Two-way fixed effects regression.",
                "method_type": "regression",
                "status": "found",
                "assertion_status": "stated",
                "exact_quote": "We use a two-way fixed effects regression model.",
            }
        ],
        "data_sources": [
            {
                "claim_text": "Current Population Survey microdata.",
                "status": "found",
                "assertion_status": "stated",
                "exact_quote": "We rely on the Current Population Survey microdata.",
            }
        ],
        "setting": {
            "claim_text": "US labor market in the 1990s.",
            "status": "found",
            "assertion_status": "stated",
            "exact_quote": "The setting is the United States labor market in the 1990s.",
        },
        # estimand_or_target_object intentionally omitted -> not_found row
        "assumptions": [
            {
                "claim_text": "Parallel trends holds.",
                "normalized_label": "parallel trends",
                "assumption_type": "identification",
                "status": "found",
                "assertion_status": "stated",
                "exact_quote": "We rely on the parallel trends assumption for identification.",
            }
        ],
        "main_results": [
            {
                "claim_text": "Teen employment fell by three percent.",
                "result_type": "point_estimate",
                "status": "found",
                "assertion_status": "inferred",
                "inferred_explanation": "derived from the reported headline figure",
                "exact_quote": "We find that teen employment fell by three percent.",
            }
        ],
        "limitations": [
            {
                "claim_text": "Short panel length.",
                "limitation_type": "data_quality",
                "status": "found",
                "assertion_status": "stated",
                "exact_quote": "A key limitation is the short panel length.",
            }
        ],
        "robustness_checks": [],
        "open_questions": [],
    }


def make_config(
    *,
    preferred: str = "local_ollama_default",
    fallback: str | None = None,
    requires_source_text: bool = True,
    private_external: bool = False,
    budget: LlmBudget | None = None,
    profiles: dict | None = None,
) -> GlobalConfig:
    """Build a routing config for the runner (resolve_route consumes .llm/.content_policy)."""
    profs = profiles if profiles is not None else default_profiles()
    routes = {
        "note_extraction": TaskRoute(
            task_type="note_extraction",
            preferred_profile=preferred,
            fallback_profile=fallback,
            requires_source_text=requires_source_text,
        )
    }
    return GlobalConfig(
        llm=LLMConfig(profiles=profs, routes=routes),
        content_policy=ContentPolicy(external_llm_for_private_full_text=private_external),
        budget=budget or LlmBudget(),
    )


def _make_doc(slug, markdown=MD, *, access_class=AccessClass.open_access, doi="10.1/x"):
    """Create a project + one bridged converted markdown; return (handle, work_id, md_doc)."""
    h = service.create_project(slug)
    wid, md = _add_doc(h, slug, markdown, access_class=access_class, doi=doi)
    return h, wid, md


def _add_doc(h, slug_file, markdown, *, access_class=AccessClass.open_access, doi="10.1/x"):
    """Add one work + bridged converted markdown to an existing project handle."""
    method = (
        AcquisitionMethod.open_access_fetch
        if access_class == AccessClass.open_access
        else AcquisitionMethod.upload
    )
    w = service.add_work(h, ids={"doi": doi}, title="W " + slug_file)
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{slug_file}.pdf"
    p.write_bytes(b"%PDF-1.4 " + slug_file.encode() + b" body content with words here")
    src = ingest_file(p, access_class=access_class, acquisition_method=method, root=None)
    md = convert_source_file(
        src.source_file_id, backend=FakeMarkerBackend(markdown=markdown), root=None
    )
    with Session(h.engine) as s:
        write_bridge(
            s,
            work_id=w.work_id,
            source_file_id=src.source_file_id,
            file_hash=src.file_hash,
            markdown_id=md.markdown_id,
            markdown_hash=md.markdown_hash,
            acquisition_method=method.value,
        )
        s.commit()
    return w.work_id, md


def _dbconn(h) -> sqlite3.Connection:
    conn = sqlite3.connect(str(h.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _run_extract(h, wid, *, note=None, responses=None, config=None, capabilities=None, **kwargs):
    """Run extract_note with a FakeLLMBackend; return (result, fake_backend)."""
    fake = FakeLLMBackend(response=note, responses=responses)
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            result = extract_note(
                session,
                cache_conn,
                work_id=wid,
                backend=fake,
                config=config if config is not None else make_config(),
                cache_root=None,
                capabilities=capabilities,
                **kwargs,
            )
    finally:
        cache_conn.close()
    return result, fake


# --------------------------------------------------------------------------
# vocab / schema
# --------------------------------------------------------------------------

def test_normalize_claim_type_passthrough():
    """normalize_claim_type passes through doc-05 §6 members; unknown -> 'claim'; no rename."""
    for value in (
        "data",
        "setting",
        "identification_assumption",
        "regularity_condition",
        "general_assumption",
        "method",
        "result",
        "limitation",
        "research_question",
    ):
        assert normalize_claim_type(value) == value
    assert normalize_claim_type("totally_unknown_kind") == "claim"
    # NO data -> dataset rewrite.
    assert normalize_claim_type("data") == "data"
    assert normalize_claim_type("data") != "dataset"


def test_schema_accepts_data_sources_and_setting():
    """DefaultNoteV1 accepts data_sources + setting -> claims of type data and setting."""
    note = DefaultNoteV1.model_validate(
        {
            "data_sources": [{"claim_text": "Some panel", "status": "found"}],
            "setting": {"claim_text": "A country", "status": "found"},
        }
    )
    drafts, _note_text, _archetype = normalize_note(note)
    types = {d.field_key: d for d in drafts}
    assert types["data_sources[0]"].claim_type == "data"
    assert types["setting"].claim_type == "setting"


# --------------------------------------------------------------------------
# validator
# --------------------------------------------------------------------------

def test_validator_valid_json():
    note, errors = validate_note(json.dumps(good_note_dict()))
    assert note is not None
    assert errors == []
    assert note.research_question.exact_quote.startswith("We study whether")


def test_validator_extracts_wrapped_json():
    """Prose-preamble / code-fence-wrapped JSON is bracket-extracted and validated."""
    body = json.dumps(good_note_dict())
    wrapped = f"Sure, here is the note:\n```json\n{body}\n```\nThanks!"
    assert extract_json_object(wrapped) == body
    note, errors = validate_note(wrapped)
    assert note is not None and errors == []


def test_validator_repair_path_via_runner():
    """A malformed first response drives the single repair re-prompt path (runner)."""
    h, wid, _md = _make_doc("v_repair")
    result, fake = _run_extract(
        h, wid, responses=["this is not JSON at all", json.dumps(good_note_dict())]
    )
    assert result.run_status == "success"
    assert len(fake.calls) == 2  # first attempt + one repair


def test_validator_unrepairable():
    note, errors = validate_note("no json object here, just prose")
    assert note is None
    assert errors  # non-empty error list


def test_validator_coerces_unambiguous_nonstring_method_type():
    """Live catch (test_proj, gpt-5.4-mini): stochastic non-string method_type is
    coerced pre-Pydantic instead of failing schema validation."""
    # Multi-element list of strings -> "; "-joined.
    d = good_note_dict()
    d["method_or_model"][0]["method_type"] = ["regression", "panel methods"]
    note, errors = validate_note(json.dumps(d))
    assert errors == [] and note is not None
    assert note.method_or_model[0].method_type == "regression; panel methods"

    # Single-element list -> that element.
    d["method_or_model"][0]["method_type"] = ["regression"]
    note, errors = validate_note(json.dumps(d))
    assert errors == []
    assert note.method_or_model[0].method_type == "regression"

    # Single-key wrapped dict -> the wrapped string.
    d["method_or_model"][0]["method_type"] = {"value": "regression"}
    note, errors = validate_note(json.dumps(d))
    assert errors == []
    assert note.method_or_model[0].method_type == "regression"

    # None stays None (explicit null is a legitimate absent value).
    d["method_or_model"][0]["method_type"] = None
    note, errors = validate_note(json.dumps(d))
    assert errors == []
    assert note.method_or_model[0].method_type is None


def test_validator_coercion_covers_sibling_string_fields_for_free():
    """The same helper covers every envelope string field (no per-field logic):
    result_type / limitation_type / normalized_label get the identical pass."""
    d = good_note_dict()
    d["main_results"][0]["result_type"] = ["point_estimate"]
    d["limitations"][0]["limitation_type"] = {"type": "data_quality"}
    d["research_question"]["normalized_label"] = ["research question", "RQ"]
    note, errors = validate_note(json.dumps(d))
    assert errors == [] and note is not None
    assert note.main_results[0].result_type == "point_estimate"
    assert note.limitations[0].limitation_type == "data_quality"
    assert note.research_question.normalized_label == "research question; RQ"


@pytest.mark.parametrize(
    "bad",
    [
        {"a": "x", "b": "y"},  # multi-key dict: which value? ambiguous
        [{"type": "regression"}],  # list of dicts, not strings
        ["regression", 7],  # mixed-type list
        [],  # empty list carries no string
        42,  # a number is not an unambiguous string
    ],
)
def test_validator_still_rejects_unusable_method_type(bad):
    """Genuinely unusable shapes still fail schema validation (no over-tolerance)."""
    d = good_note_dict()
    d["method_or_model"][0]["method_type"] = bad
    note, errors = validate_note(json.dumps(d))
    assert note is None
    assert errors


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------

def test_normalize_mixed_statuses():
    """found/not_found rows: every scalar once, every element once, absent -> not_found."""
    note = DefaultNoteV1.model_validate(good_note_dict())
    drafts, note_text, archetype = normalize_note(note)
    by_key = {d.field_key: d for d in drafts}
    # Every scalar appears exactly once.
    for key in ("research_question", "main_contribution", "setting", "estimand_or_target_object"):
        assert key in by_key
    # estimand absent -> not_found; populated arrays -> found; empty arrays -> not_found.
    assert by_key["estimand_or_target_object"].status == "not_found"
    assert by_key["research_question"].status == "found"
    assert by_key["robustness_checks"].status == "not_found"
    assert by_key["open_questions"].status == "not_found"
    found = [d for d in drafts if d.status == "found"]
    assert len(found) == 8
    # Chunk 4 multi-label: found data (empirical) + found identification
    # assumption (theoretical) combine in canonical order.
    assert note_text and archetype == "theoretical+empirical"


def _found_env(text: str, **extra) -> dict:
    """A minimal ``found`` claim envelope for the archetype derivation table."""
    return {"claim_text": text, "status": "found", **extra}


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # data only -> empirical
        ({"data_sources": [_found_env("CPS microdata")]}, "empirical"),
        # assumptions only -> theoretical
        (
            {"assumptions": [_found_env("Parallel trends holds", assumption_type="identification")]},
            "theoretical",
        ),
        # data + assumptions combine (v1 could not represent this)
        (
            {
                "data_sources": [_found_env("CPS microdata")],
                "assumptions": [_found_env("Regularity", assumption_type="regularity")],
            },
            "theoretical+empirical",
        ),
        # simulation regex firing on a method claim_text
        (
            {"method_or_model": [_found_env("We run 10,000 Monte Carlo replications")]},
            "simulation",
        ),
        # simulation regex firing on claim_subtype (method_type carries it)
        (
            {"method_or_model": [_found_env("A generative benchmark", method_type="simulation")]},
            "simulation",
        ),
        # hyphenated monte-carlo variant + data -> canonical order, no '+theoretical'
        (
            {
                "data_sources": [_found_env("Synthetic panel")],
                "method_or_model": [_found_env("A monte-carlo study of the estimator")],
            },
            "empirical+simulation",
        ),
        # regex is fenced to method_or_model: 'simulation' in a RESULT does not fire
        (
            {"main_results": [_found_env("Simulation results show a 3% drop", result_type="other")]},
            "empirical",  # default case: found claims but no label fired
        ),
        # non-simulation method + no data/assumptions -> default empirical
        ({"method_or_model": [_found_env("Two-way fixed effects regression")]}, "empirical"),
        # zero found claims -> 'empty' (deliberate deviation from v1's default)
        ({}, "empty"),
    ],
)
def test_archetype_derivation_table(fields: dict, expected: str):
    """Chunk 4: multi-label archetype derivation over {theoretical, empirical, simulation}."""
    note = DefaultNoteV1.model_validate(fields)
    _drafts, _note_text, archetype = normalize_note(note)
    assert archetype == expected


def test_archetype_canonical_ordering_stable():
    """All three labels emit in fixed theoretical->empirical->simulation order,
    independent of draft order."""
    note = DefaultNoteV1.model_validate(
        {
            "data_sources": [_found_env("CPS microdata")],
            "assumptions": [_found_env("Parallel trends", assumption_type="identification")],
            "method_or_model": [_found_env("Monte Carlo simulation of the estimator")],
        }
    )
    drafts, _note_text, archetype = normalize_note(note)
    assert archetype == "theoretical+empirical+simulation"
    # Ordering comes from the canonical label order, not draft order.
    assert _derive_archetype(list(reversed(drafts))) == "theoretical+empirical+simulation"


def test_normalize_inferred_requires_explanation_and_epistemic_type():
    """inferred without explanation rejected; epistemic_type independent of assertion (D2)."""
    # Stated -> llm_extracted.
    stated = DefaultNoteV1.model_validate(
        {"research_question": {"status": "found", "assertion_status": "stated", "claim_text": "x"}}
    )
    rq = {d.field_key: d for d in normalize_note(stated)[0]}["research_question"]
    assert rq.assertion_status == "stated" and rq.epistemic_type == "llm_extracted"

    # Inferred WITH explanation -> llm_inferred (the two fields are independent).
    inferred = DefaultNoteV1.model_validate(
        {
            "research_question": {
                "status": "found",
                "assertion_status": "inferred",
                "inferred_explanation": "inferred from the abstract",
                "claim_text": "x",
            }
        }
    )
    rq2 = {d.field_key: d for d in normalize_note(inferred)[0]}["research_question"]
    assert rq2.assertion_status == "inferred" and rq2.epistemic_type == "llm_inferred"

    # Inferred WITHOUT explanation -> rejected.
    bad = DefaultNoteV1.model_validate(
        {"research_question": {"status": "found", "assertion_status": "inferred", "claim_text": "x"}}
    )
    with pytest.raises(ValueError):
        normalize_note(bad)


def test_normalize_identification_assumption_claim_type():
    """identification assumption -> claim_type='identification_assumption', subtype IS NULL."""
    assert map_assumption_claim_type("identification") == "identification_assumption"
    assert map_assumption_claim_type("regularity") == "regularity_condition"
    assert map_assumption_claim_type("something_else") == "general_assumption"
    assert map_assumption_claim_type(None) == "general_assumption"
    note = DefaultNoteV1.model_validate(
        {"assumptions": [{"assumption_type": "identification", "status": "found", "claim_text": "p"}]}
    )
    draft = {d.field_key: d for d in normalize_note(note)[0]}["assumptions[0]"]
    assert draft.claim_type == "identification_assumption"
    assert draft.claim_subtype is None  # decision 47 — assumptions never use claim_subtype


def test_normalize_method_subtype():
    """A method element -> claim_type='method' with claim_subtype == method_type."""
    note = DefaultNoteV1.model_validate(
        {"method_or_model": [{"method_type": "regression", "status": "found", "claim_text": "m"}]}
    )
    draft = {d.field_key: d for d in normalize_note(note)[0]}["method_or_model[0]"]
    assert draft.claim_type == "method"
    assert draft.claim_subtype == "regression"


# --------------------------------------------------------------------------
# access class / staleness
# --------------------------------------------------------------------------

def test_resolve_source_access_class():
    """open_access / user_supplied_private resolvable; missing source -> fail-closed private."""
    h_oa, _wid, md_oa = _make_doc("acc_oa", access_class=AccessClass.open_access, doi="10.5/oa")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        assert resolve_source_access_class(cache_conn, md_oa.markdown_id) == "open_access"
        assert resolve_source_access_class(cache_conn, "md_not_present") == "user_supplied_private"
    finally:
        cache_conn.close()

    priv_md = MD.replace("Minimum Wage", "Private Variant")
    _h2, _wid2, md_p = _make_doc(
        "acc_priv", priv_md, access_class=AccessClass.user_supplied_private, doi="10.5/p"
    )
    cache_conn = cache_access.open_cache_ro(None)
    try:
        assert resolve_source_access_class(cache_conn, md_p.markdown_id) == "user_supplied_private"
    finally:
        cache_conn.close()


def test_staleness_changed_hash_returns_none():
    """A changed cached markdown_hash => current_note returns None (re-extract)."""
    h, wid, md = _make_doc("stale_hash")
    result, _fake = _run_extract(h, wid, note=good_note_dict())
    assert result.run_status == "success"
    with Session(h.engine) as session:
        assert current_note(session, wid, SCHEMA_ID, md.markdown_hash) is not None
        assert current_note(session, wid, SCHEMA_ID, "deadbeef_changed_hash") is None


def test_force_reextracts():
    """--force re-extracts despite an existing current note (a new run row)."""
    h, wid, _md = _make_doc("force")
    _run_extract(h, wid, note=good_note_dict())
    # Without --force -> idempotent skip (no new run).
    again, _fake = _run_extract(h, wid, note=good_note_dict())
    assert again.run_status == "skipped_idempotent"
    conn = _dbconn(h)
    try:
        runs_before = conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0]
    finally:
        conn.close()
    forced, _fake2 = _run_extract(h, wid, note=good_note_dict(), force=True)
    assert forced.run_status == "success"
    conn = _dbconn(h)
    try:
        runs_after = conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0]
    finally:
        conn.close()
    assert runs_after == runs_before + 1


# --------------------------------------------------------------------------
# runner happy path + provenance
# --------------------------------------------------------------------------

def test_runner_happy_path():
    """One note; found claims have >=1 span; not_found have 0; FTS populated; provenance +
    access_class on run+note+claims+SPANS; raw_note_json round-trips."""
    h, wid, md = _make_doc("happy")
    result, _fake = _run_extract(h, wid, note=good_note_dict())
    assert result.run_status == "success"
    assert result.claim_count == 11 and result.span_count == 8

    conn = _dbconn(h)
    try:
        # Exactly one note / one successful run.
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE run_status='success'"
        ).fetchone()[0] == 1
        note_row = conn.execute("SELECT * FROM structured_notes").fetchone()

        # Every found claim has >=1 claim_spans -> evidence_spans; not_found has zero.
        found_rows = conn.execute(
            "SELECT claim_id FROM extracted_claims WHERE status='found'"
        ).fetchall()
        assert len(found_rows) == 8
        for (claim_id,) in [(r["claim_id"],) for r in found_rows]:
            n = conn.execute(
                "SELECT COUNT(*) FROM claim_spans cs JOIN evidence_spans es "
                "ON es.span_id = cs.span_id WHERE cs.claim_id = ?",
                (claim_id,),
            ).fetchone()[0]
            assert n >= 1
        for r in conn.execute(
            "SELECT claim_id FROM extracted_claims WHERE status='not_found'"
        ).fetchall():
            assert conn.execute(
                "SELECT COUNT(*) FROM claim_spans WHERE claim_id=?", (r["claim_id"],)
            ).fetchone()[0] == 0

        # claim_fts / note_fts populated.
        assert conn.execute("SELECT COUNT(*) FROM claim_fts").fetchone()[0] == 11
        assert conn.execute("SELECT COUNT(*) FROM note_fts").fetchone()[0] == 1

        # Provenance complete on the run.
        run = conn.execute("SELECT * FROM extraction_runs WHERE run_status='success'").fetchone()
        assert run["schema_id"] == SCHEMA_ID
        assert run["schema_version"] == SCHEMA_VERSION
        assert run["prompt_version"] == PROMPT_VERSION
        assert run["markdown_hash"] == md.markdown_hash
        assert run["input_tokens"] and run["input_tokens"] > 0
        assert run["created_at"]

        # access_class stamped on run + note + claims + the created spans.
        assert run["access_class"] == "open_access"
        assert note_row["access_class"] == "open_access"
        claim_classes = {
            r[0] for r in conn.execute("SELECT DISTINCT access_class FROM extracted_claims")
        }
        assert claim_classes == {"open_access"}
        span_classes = {
            r[0] for r in conn.execute("SELECT DISTINCT access_class FROM evidence_spans")
        }
        assert span_classes == {"open_access"}

        # raw_note_json round-trips.
        parsed = DefaultNoteV1.model_validate(json.loads(note_row["raw_note_json"]))
        assert parsed.research_question.exact_quote.startswith("We study whether")
    finally:
        conn.close()


def test_span_access_class_propagation():
    """should_fix #2 / D76: an open_access source yields open_access spans, not the private default."""
    h, wid, _md = _make_doc("oa_prop", access_class=AccessClass.open_access)
    _run_extract(h, wid, note=good_note_dict())
    conn = _dbconn(h)
    try:
        classes = {r[0] for r in conn.execute("SELECT access_class FROM evidence_spans")}
        assert classes == {"open_access"}
        assert "user_supplied_private" not in classes
    finally:
        conn.close()


def test_runner_anchoring_miss_downgrades_ambiguous():
    """A found claim whose quote isn't verbatim -> downgraded to ambiguous, NO fabricated span."""
    h, wid, _md = _make_doc("anchor_miss")
    note = {
        "research_question": {
            "status": "found",
            "assertion_status": "stated",
            "claim_text": "Some RQ",
            "exact_quote": "THIS EXACT QUOTE DOES NOT APPEAR ANYWHERE IN THE DOCUMENT.",
        }
    }
    result, _fake = _run_extract(h, wid, note=note)
    assert result.run_status == "success"
    conn = _dbconn(h)
    try:
        rq = conn.execute(
            "SELECT status, inferred_explanation FROM extracted_claims "
            "WHERE field_key='research_question'"
        ).fetchone()
        assert rq["status"] == "ambiguous"
        assert "anchored" in (rq["inferred_explanation"] or "")
        # No fabricated span at all.
        assert conn.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM claim_spans").fetchone()[0] == 0
    finally:
        conn.close()


def test_runner_list_method_type_succeeds_without_retry():
    """Live catch e2e (test_proj, gpt-5.4-mini): attempt 1 returns method_type as a
    LIST of strings -> tolerant coercion validates it, run succeeds, and NO repair
    re-prompt is dispatched (exactly one backend call). The coerced value lands in
    claim_subtype and the persisted raw_note_json round-trips."""
    h, wid, _md = _make_doc("mtype_list")
    d = good_note_dict()
    d["method_or_model"][0]["method_type"] = ["regression", "panel methods"]
    result, fake = _run_extract(h, wid, responses=[json.dumps(d)])
    assert result.run_status == "success"
    assert len(fake.calls) == 1  # coercion rescued attempt 1: no repair re-prompt
    conn = _dbconn(h)
    try:
        row = conn.execute(
            "SELECT claim_subtype FROM extracted_claims WHERE field_key='method_or_model[0]'"
        ).fetchone()
        assert row["claim_subtype"] == "regression; panel methods"
        raw = conn.execute("SELECT raw_note_json FROM structured_notes").fetchone()[0]
        parsed = DefaultNoteV1.model_validate(json.loads(raw))
        assert parsed.method_or_model[0].method_type == "regression; panel methods"
    finally:
        conn.close()


def test_runner_json_failure_extraction_failed():
    """should_fix #5: unrepairable JSON => extraction_failed run, NO structured_notes row."""
    h, wid, _md = _make_doc("jsonfail")
    result, fake = _run_extract(
        h, wid, responses=["not json", "still not json after repair"]
    )
    assert result.run_status == "extraction_failed"
    assert len(fake.calls) == 2
    conn = _dbconn(h)
    try:
        assert conn.execute(
            "SELECT run_status FROM extraction_runs"
        ).fetchone()["run_status"] == "extraction_failed"
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
    finally:
        conn.close()


# --------------------------------------------------------------------------
# gates: context / content / budget / no-llm / routing
# --------------------------------------------------------------------------

def test_context_gate_skipped_oversize():
    """Over-window prompt => skipped_oversize, no note, message -> phase_4b, NO dispatch."""
    h, wid, _md = _make_doc("oversize")
    tiny_caps = LlmCapabilities(
        snapshot_date=date(2026, 1, 1),
        models={
            "llama3": ModelCapability(
                provider="ollama",
                context_window_tokens=5,  # far smaller than the prompt
                pricing_status="not_applicable",
                access_modes=["local"],
            )
        },
    )
    result, fake = _run_extract(
        h, wid, note=good_note_dict(), capabilities=tiny_caps
    )
    assert result.run_status == "skipped_oversize"
    assert "phase_4b" in (result.message or "")
    assert fake.calls == []  # no backend dispatch
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        assert conn.execute(
            "SELECT run_status FROM extraction_runs"
        ).fetchone()["run_status"] == "skipped_oversize"
    finally:
        conn.close()


def test_context_gate_unknown_model_fails_closed():
    """A routed model ABSENT from the bundled llm_capabilities.yaml snapshot fails
    CLOSED: the oversize gate refuses to dispatch rather than risk silent truncation
    (issue #1/#4 — the gate must not fail open on an unknown model). Uses the bundled
    snapshot (capabilities=None), the path the §11 oversize test never exercised."""
    h, wid, _md = _make_doc("unknown_model")
    mystery = LLMProfile(
        profile_id="local_mystery",
        provider="ollama",
        access_mode="local",
        model="mystery-model-not-in-snapshot",
        is_local=True,
        allowed_tasks=["note_extraction"],
    )
    config = make_config(preferred="local_mystery", profiles={"local_mystery": mystery})
    # capabilities=None -> the runner loads the bundled snapshot (no mystery model).
    result, fake = _run_extract(h, wid, note=good_note_dict(), config=config)
    assert result.run_status == "skipped_oversize"
    assert fake.calls == []  # nothing dispatched to the backend
    assert "mystery-model-not-in-snapshot" in (result.message or "")
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        assert conn.execute(
            "SELECT run_status FROM extraction_runs"
        ).fetchone()["run_status"] == "skipped_oversize"
    finally:
        conn.close()


def test_default_local_profile_model_in_capabilities_snapshot():
    """Regression for issue #1/#4: the local-first default profile's model
    (`local_ollama_default` -> llama3) MUST be present in the bundled snapshot, or
    the oversize gate has no window to compare against for the default route."""
    from seedgraph.config.loader import load_llm_capabilities

    caps = load_llm_capabilities()
    model = default_profiles()["local_ollama_default"].model
    assert model in caps.models
    assert caps.models[model].context_window_tokens > 0


def test_content_gate_no_confirmation(monkeypatch):
    """private + external profile + no confirmation => skipped_policy; NO external dispatch."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    priv_md = MD.replace("Minimum Wage", "Confidential Draft")
    h, wid, _md = _make_doc(
        "content_gate", priv_md, access_class=AccessClass.user_supplied_private, doi="10.7/p"
    )
    profiles = {"anthropic_api_default": default_profiles()["anthropic_api_default"]}
    config = make_config(
        preferred="anthropic_api_default", fallback=None, private_external=False, profiles=profiles
    )
    result, fake = _run_extract(h, wid, note=good_note_dict(), config=config)
    assert result.run_status == "skipped_policy"
    assert fake.calls == []  # source text NEVER dispatched externally
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        run = conn.execute("SELECT * FROM extraction_runs").fetchone()
        assert run["run_status"] == "skipped_policy"
        assert run["external_full_text"] == 0  # nothing left the machine
    finally:
        conn.close()


@pytest.mark.parametrize(
    "access_class",
    [
        AccessClass.metadata_only,
        AccessClass.licensed_future,
        AccessClass.unknown,
    ],
)
def test_content_gate_blocks_all_restricted_classes(monkeypatch, access_class):
    """Regression (content-gate leak): every restricted class — not just
    user_supplied_private — must be blocked from an external dispatch without
    --confirm-external (plan §8; decisions 30/60/76). Source text NEVER leaves the
    machine; with no local profile the work records skipped_policy and stamps
    external_full_text=0."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    secret_md = MD.replace("Minimum Wage", "Embargoed Licensed Text")
    h, wid, _md = _make_doc(
        f"leak_{access_class.value}", secret_md, access_class=access_class, doi="10.9/p"
    )
    profiles = {"anthropic_api_default": default_profiles()["anthropic_api_default"]}
    config = make_config(
        preferred="anthropic_api_default", fallback=None, private_external=False, profiles=profiles
    )
    result, fake = _run_extract(h, wid, note=good_note_dict(), config=config)
    assert result.run_status == "skipped_policy"
    assert fake.calls == []  # secret source text NEVER dispatched externally
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        run = conn.execute("SELECT * FROM extraction_runs").fetchone()
        assert run["run_status"] == "skipped_policy"
        assert run["external_full_text"] == 0  # nothing left the machine
    finally:
        conn.close()


def test_oversize_external_route_stamps_no_full_text_leak(monkeypatch):
    """Regression (issue #3): an external preferred profile that trips the
    context-window gate must stamp external_full_text=0 — the gate returns before
    any dispatch, so NO source text leaves the machine (plan §8)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    h, wid, _md = _make_doc("oversize_ext", doi="10.91/p")
    profiles = {"anthropic_api_default": default_profiles()["anthropic_api_default"]}
    model = profiles["anthropic_api_default"].model
    tiny_caps = LlmCapabilities(
        snapshot_date=date(2026, 1, 1),
        models={
            model: ModelCapability(
                provider="anthropic",
                context_window_tokens=5,  # far smaller than the prompt
                pricing_status="verified",
                input_usd_per_mtok=0.0,
                output_usd_per_mtok=0.0,
                access_modes=["api"],
            )
        },
    )
    config = make_config(
        preferred="anthropic_api_default", fallback=None, private_external=True, profiles=profiles
    )
    result, fake = _run_extract(
        h, wid, note=good_note_dict(), config=config, capabilities=tiny_caps, confirm_external=True
    )
    assert result.run_status == "skipped_oversize"
    assert fake.calls == []  # nothing dispatched
    conn = _dbconn(h)
    try:
        run = conn.execute("SELECT * FROM extraction_runs").fetchone()
        assert run["run_status"] == "skipped_oversize"
        assert run["external_full_text"] == 0  # gate returned before any dispatch
    finally:
        conn.close()


def test_budget_loop_stop_on_exceeded(monkeypatch):
    """should_fix #3: stop_on_budget_exceeded + low limit over 3 works -> remaining skipped_budget."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    h = service.create_project("budget")
    wids = []
    for i in range(3):
        body = MD.replace("Minimum Wage", f"Paper Number {i}")
        wid, _md = _add_doc(h, f"budget{i}", body, access_class=AccessClass.open_access, doi=f"10.8/{i}")
        wids.append(wid)

    budget_cfg = LlmBudget(monthly_soft_limit_usd=0.001, stop_on_budget_exceeded=True)
    config = make_config(
        preferred="anthropic_api_default", fallback="no_llm", budget=budget_cfg
    )
    budget = BudgetState()
    statuses = []
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            for wid in wids:
                fake = FakeLLMBackend(response=good_note_dict())
                res = extract_note(
                    session,
                    cache_conn,
                    work_id=wid,
                    backend=fake,
                    config=config,
                    cache_root=None,
                    budget_state=budget,
                )
                statuses.append(res.run_status)
    finally:
        cache_conn.close()

    assert statuses[0] == "success"
    assert statuses[1] == "skipped_budget"
    assert statuses[2] == "skipped_budget"
    assert budget.spent_usd > 0.001
    conn = _dbconn(h)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE run_status='skipped_budget'"
        ).fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
    finally:
        conn.close()


def test_no_llm_skipped_no_llm():
    """no_llm profile => skipped_no_llm, no note (honest degradation; no rule-based substitute)."""
    h, wid, _md = _make_doc("nollm")
    config = make_config(preferred="no_llm")
    result, fake = _run_extract(h, wid, note=good_note_dict(), config=config)
    assert result.run_status == "skipped_no_llm"
    assert fake.calls == []
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        assert conn.execute(
            "SELECT run_status FROM extraction_runs"
        ).fetchone()["run_status"] == "skipped_no_llm"
    finally:
        conn.close()


def test_routing_identifier_validation():
    """note_extraction validates against profile.allowed_tasks (should_fix #4 / §8 / §13).

    The config-validation mechanism itself is asserted: every default LLM-generation
    profile lists ``note_extraction`` in ``allowed_tasks``, ``no_llm`` does not, and a
    profile NOT authorized for ``note_extraction`` is refused by routing (never
    dispatched) even though it is otherwise available.
    """
    # The allowed_tasks gate exists on the default profiles (config validation, §13).
    # Every hosted/local generation profile (incl. openai_api_default, ADR-0003)
    # authorizes note_extraction; no_llm authorizes nothing.
    profs = default_profiles()
    assert "note_extraction" in profs["local_ollama_default"].allowed_tasks
    assert "note_extraction" in profs["anthropic_api_default"].allowed_tasks
    assert "note_extraction" in profs["openai_api_default"].allowed_tasks
    assert "note_extraction" not in (profs["no_llm"].allowed_tasks or [])

    # A real (available, authorized, non-no-LLM) profile -> the route is usable.
    h_ok, wid_ok, _md = _make_doc("route_ok")
    ok, _fake = _run_extract(
        h_ok, wid_ok, note=good_note_dict(), config=make_config(preferred="local_ollama_default")
    )
    assert ok.run_status == "success"

    # no_llm -> not a usable note_extraction backend.
    h_no, wid_no, _md2 = _make_doc("route_no", MD.replace("Minimum Wage", "Other"), doi="10.9/n")
    no, _fake2 = _run_extract(
        h_no, wid_no, note=good_note_dict(), config=make_config(preferred="no_llm")
    )
    assert no.run_status == "skipped_no_llm"

    # A profile that is available + local but whose allowed_tasks OMITS note_extraction
    # fails config validation: routing refuses it and the backend is never dispatched
    # (honest degradation -> skipped_no_llm), proving the allowed_tasks gate is enforced.
    h_un, wid_un, _md3 = _make_doc("route_unauth", MD.replace("Minimum Wage", "Unauth"), doi="10.9/u")
    unauth = LLMProfile(
        profile_id="local_embeddings_only",
        provider="ollama",
        access_mode="local",
        model="llama3",
        is_local=True,
        allowed_tasks=["embeddings"],  # NOT authorized for note_extraction
    )
    un_cfg = make_config(
        preferred="local_embeddings_only", profiles={"local_embeddings_only": unauth}
    )
    un, fake_un = _run_extract(h_un, wid_un, note=good_note_dict(), config=un_cfg)
    assert un.run_status == "skipped_no_llm"
    assert fake_un.calls == []  # the unauthorized profile was never dispatched


# --------------------------------------------------------------------------
# schema integrity / bridge / FTS
# --------------------------------------------------------------------------

def _seed_run_note(conn, wid, md, access="open_access"):
    """Raw-insert a minimal extraction_runs + structured_notes pair; return (run_id, note_id)."""
    run_id = new_id("extr")
    note_id = new_id("note")
    now = _iso()
    conn.execute(
        "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, "
        "schema_id, schema_version, prompt_version, access_class, external_full_text, run_status, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (run_id, wid, md.markdown_id, md.markdown_hash, SCHEMA_ID, SCHEMA_VERSION, PROMPT_VERSION,
         access, 0, "success", now),
    )
    conn.execute(
        "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
        "markdown_hash, schema_id, schema_version, prompt_version, access_class, raw_note_json, "
        "note_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (note_id, run_id, wid, md.markdown_id, md.markdown_hash, SCHEMA_ID, SCHEMA_VERSION,
         PROMPT_VERSION, access, "{}", "x", now),
    )
    return run_id, note_id


def test_fk_dangling_claim_span_raises():
    """Inserting claim_spans with a dangling span_id raises (foreign_keys ON)."""
    h, wid, md = _make_doc("fk_dangle")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        run_id, note_id = _seed_run_note(conn, wid, md)
        claim_id = new_id("claim")
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
            "work_id, claim_type, field_key, status, epistemic_type, access_class, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (claim_id, note_id, run_id, wid, "result", "main_results[0]", "found",
             "llm_extracted", "open_access", _iso()),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, ?, ?)",
                (claim_id, "span_does_not_exist", 0, _iso()),
            )
    finally:
        conn.close()


def _affinity(type_str: str) -> str:
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"


def _assert_table_parity(insp, table_name, model):
    """Column set + PK + per-column affinity & nullability parity (ORM == migrated, D6).

    Index column-set equality is intentionally out of scope: per D6 the ORM is
    column MAPPING ONLY (the numbered .sql authors the composite indexes), and the
    plan §11 parity bullet enumerates columns + nullability + the D2 columns.
    """
    orm_cols = {c.name: c for c in model.__table__.columns}
    mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
    assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"

    mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
    orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
    assert mig_pk == orm_pk, f"{table_name}: PK drift"

    orm_dialect = sqlite_dialect.dialect()
    for name, orm_col in orm_cols.items():
        mig_col = mig_cols[name]
        orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
        mig_aff = _affinity(str(mig_col["type"]))
        assert orm_aff == mig_aff, f"{table_name}.{name}: type {orm_aff} != {mig_aff}"
        if name not in mig_pk:
            assert orm_col.nullable == mig_col["nullable"], f"{table_name}.{name}: nullability drift"


def test_migration_parity_orm_equals_sql(tmp_path):
    """D6/D2: 0007 schema == ORM metadata for the 3 mapped tables + claim_spans junction;
    extracted_claims has epistemic_type (NOT NULL) + assertion_status; the nullable FK rulings hold."""
    from seedgraph.db.migrations import discover_migrations

    db = tmp_path / "parity.db"
    conn = sqlite3.connect(str(db))
    # Apply migrations THROUGH 0009: phase_4 birth-DDL 0007 + phase_4b's additive
    # chunk-provenance ALTERs 0008 + phase_6's lens_definition_hash ALTER 0009, all
    # of which the ExtractionRun ORM class now mirrors (D6). 0009 also creates
    # lenses/lens_outputs (parity asserted in test_phase_6), not inspected here.
    for version, path in discover_migrations("project"):
        if version > 9:
            continue
        conn.executescript(path.read_text(encoding="utf-8"))
    conn.close()

    insp = inspect(create_engine(f"sqlite:///{db.as_posix()}"))
    names = insp.get_table_names()
    for t in ("extraction_runs", "structured_notes", "extracted_claims", "claim_spans",
              "claim_fts", "note_fts"):
        assert t in names, t

    _assert_table_parity(insp, "extraction_runs", ExtractionRun)
    _assert_table_parity(insp, "structured_notes", StructuredNote)
    _assert_table_parity(insp, "extracted_claims", ExtractedClaim)

    cols = {c["name"]: c for c in insp.get_columns("extracted_claims")}
    # D2: both columns present; epistemic_type NOT NULL, assertion_status present (nullable).
    assert cols["epistemic_type"]["nullable"] is False
    assert "assertion_status" in cols
    # phase_4b chunk-provenance columns now mirrored on the ORM + present in 0008.
    run_cols_4b = {c["name"]: c for c in insp.get_columns("extraction_runs")}
    assert run_cols_4b["extraction_mode"]["nullable"] is False  # NOT NULL DEFAULT 'whole'
    for n in ("parent_extraction_run_id", "chunk_index", "chunk_count", "chunk_section_ids"):
        assert n in run_cols_4b and run_cols_4b[n]["nullable"] is True
    assert cols["source_chunk_index"]["nullable"] is True
    assert cols["source_extraction_run_id"]["nullable"] is True
    # Binding nullable rulings (scaffold overrides the plan's NOT NULL text here).
    assert cols["structured_note_id"]["nullable"] is True
    note_cols = {c["name"]: c for c in insp.get_columns("structured_notes")}
    assert note_cols["schema_id"]["nullable"] is True
    run_cols = {c["name"]: c for c in insp.get_columns("extraction_runs")}
    # schema_id is nullable: lens-origin runs leave it NULL (sibling schema,
    # mutually exclusive w/ lens_id; decision 22 / phase_6 §4.3).
    assert run_cols["schema_id"]["nullable"] is True

    # claim_spans is a pure junction (no ORM class): assert its columns + FK targets.
    cs_cols = {c["name"] for c in insp.get_columns("claim_spans")}
    assert {"id", "claim_id", "span_id", "rank", "created_at"} <= cs_cols
    referred = {fk["referred_table"] for fk in insp.get_foreign_keys("claim_spans")}
    assert {"extracted_claims", "evidence_spans"} <= referred


def test_adapter_bridge_atomic_txn():
    """D7: ensure_span INSERT + ORM claim + raw claim_spans commit atomically; rollback hides both."""
    from seedgraph.spans.store import ensure_span

    # --- commit path ---
    h, wid, md = _make_doc("d7commit")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            run_id, note_id = _seed_run_note(conn, wid, md)
            sid = ensure_span(
                conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
                exact_quote="parallel trends assumption", access_class="open_access",
            )
            assert sid is not None
            claim = ExtractedClaim(
                claim_id=new_id("claim"), structured_note_id=note_id, extraction_run_id=run_id,
                work_id=wid, claim_type="result", field_key="main_results[0]", status="found",
                epistemic_type="llm_extracted", access_class="open_access", created_at=_iso(),
            )
            session.add(claim)
            session.flush()
            conn.execute(
                "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, ?, ?)",
                (claim.claim_id, sid, 0, _iso()),
            )
            session.commit()
            committed_claim = claim.claim_id
    finally:
        cache_conn.close()
    check = _dbconn(h)
    try:
        assert check.execute("SELECT COUNT(*) FROM evidence_spans WHERE span_id=?", (sid,)).fetchone()[0] == 1
        assert check.execute(
            "SELECT COUNT(*) FROM extracted_claims WHERE claim_id=?", (committed_claim,)
        ).fetchone()[0] == 1
        assert check.execute(
            "SELECT COUNT(*) FROM claim_spans WHERE span_id=?", (sid,)
        ).fetchone()[0] == 1
    finally:
        check.close()

    # --- rollback path (fresh doc, fresh quote) ---
    h2, wid2, md2 = _make_doc("d7rollback", MD.replace("Minimum Wage", "Rollback Variant"), doi="10.11/r")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h2.engine) as session:
            conn = raw_conn(session)
            run_id, note_id = _seed_run_note(conn, wid2, md2)
            sid2 = ensure_span(
                conn, cache_conn, None, markdown_id=md2.markdown_id, work_id=wid2,
                exact_quote="parallel trends assumption", access_class="open_access",
            )
            assert sid2 is not None
            claim = ExtractedClaim(
                claim_id=new_id("claim"), structured_note_id=note_id, extraction_run_id=run_id,
                work_id=wid2, claim_type="result", field_key="main_results[0]", status="found",
                epistemic_type="llm_extracted", access_class="open_access", created_at=_iso(),
            )
            session.add(claim)
            session.flush()
            conn.execute(
                "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, ?, ?)",
                (claim.claim_id, sid2, 0, _iso()),
            )
            rolled_claim = claim.claim_id
            session.rollback()
    finally:
        cache_conn.close()
    check = _dbconn(h2)
    try:
        assert check.execute("SELECT COUNT(*) FROM evidence_spans WHERE span_id=?", (sid2,)).fetchone()[0] == 0
        assert check.execute(
            "SELECT COUNT(*) FROM extracted_claims WHERE claim_id=?", (rolled_claim,)
        ).fetchone()[0] == 0
        assert check.execute(
            "SELECT COUNT(*) FROM claim_spans WHERE span_id=?", (sid2,)
        ).fetchone()[0] == 0
    finally:
        check.close()


def test_fts_exact_phrase_query():
    """Exact phrase "parallel trends" returns the seeded claim + note (unicode61, no porter)."""
    h, wid, _md = _make_doc("fts")
    _run_extract(h, wid, note=good_note_dict())
    conn = _dbconn(h)
    try:
        claim_hits = conn.execute(
            'SELECT claim_id FROM claim_fts WHERE claim_fts MATCH ?', ('"parallel trends"',)
        ).fetchall()
        assert len(claim_hits) >= 1
        note_hits = conn.execute(
            'SELECT note_id FROM note_fts WHERE note_fts MATCH ?', ('"parallel trends"',)
        ).fetchall()
        assert len(note_hits) == 1
        # No stemming: "trend" (singular stem) must NOT match the verbatim "trends".
        assert conn.execute(
            'SELECT COUNT(*) FROM claim_fts WHERE claim_fts MATCH ?', ("trend",)
        ).fetchone()[0] == 0
    finally:
        conn.close()


# --------------------------------------------------------------------------
# milestone (doc 10 §8 acceptance) — end-to-end via the CLI, offline
# --------------------------------------------------------------------------

def test_milestone_extract_notes_corpus(monkeypatch):
    """`seedgraph extract notes <slug>` over a 1-paper fixture: one current note; every field a
    claim with a valid status; found claims have >=1 span; absent fields explicit not_found;
    created spans carry the source access_class; a re-run without --force is an idempotent skip."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    # Offline backend injected through the CLI seam (the CLI can't pass backend= via Typer).
    monkeypatch.setattr(runner_mod, "_BACKEND_OVERRIDE", FakeLLMBackend(response=good_note_dict()))

    h, wid, _md = _make_doc("mile")

    r1 = cli.invoke(app, ["extract", "notes", "mile"])
    assert r1.exit_code == 0, r1.output
    assert "success" in r1.output

    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
        statuses = {
            r[0] for r in conn.execute("SELECT DISTINCT status FROM extracted_claims")
        }
        assert statuses <= {"found", "not_found", "ambiguous", "not_applicable"}
        # Every found substantive claim has >=1 evidence span.
        for (claim_id,) in conn.execute(
            "SELECT claim_id FROM extracted_claims WHERE status='found'"
        ).fetchall():
            assert conn.execute(
                "SELECT COUNT(*) FROM claim_spans WHERE claim_id=?", (claim_id,)
            ).fetchone()[0] >= 1
        # Absent fields are explicit not_found rows.
        assert conn.execute(
            "SELECT COUNT(*) FROM extracted_claims WHERE status='not_found'"
        ).fetchone()[0] >= 1
        # Created spans carry the source access_class.
        assert {r[0] for r in conn.execute("SELECT access_class FROM evidence_spans")} == {"open_access"}
        runs_after_first = conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0]
    finally:
        conn.close()

    # Re-run without --force => idempotent skip (no new runs).
    r2 = cli.invoke(app, ["extract", "notes", "mile"])
    assert r2.exit_code == 0, r2.output
    assert "skipped_idempotent" in r2.output
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == runs_after_first
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
    finally:
        conn.close()
