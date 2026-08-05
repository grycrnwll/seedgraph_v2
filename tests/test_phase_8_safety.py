"""Phase 8 — safety-contract invariants (plan §11 "Safety contract invariants";
decisions 62/82). These are the acceptance gate the eval phase later consumes.

Each runs without an API key (fake backend) against the fixture ``project.db``. The
invariants are checked deterministically from the envelope as DATA — no behaviour is
trusted.
"""

from __future__ import annotations

import sqlite3

import pytest
from _phase8_helpers import (
    build_fixture_project,
    canned_envelope_json,
    make_answer_config,
)

from seedgraph.answer import answer, AnswerEnvelope  # noqa: F401
from seedgraph.answer import compose
from seedgraph.answer.guard import NOT_FOUND_MESSAGE
from seedgraph.answer.types import AnswerCategory, AnswerMode
from seedgraph.llm.backend import FakeLLMBackend

GROUNDED_Q = 'According to Paper A, what holds "across groups"?'


def _assert_no_unsupported_prose(env: AnswerEnvelope) -> None:
    """The unconditional must-fix invariant (plan §11): an envelope may NEVER carry
    non-empty prose with zero supporting citations — with the SOLE exception of the
    honest not-found banner (``insufficient_evidence=True`` AND the canonical
    message). Crucially this does NOT exempt the ``insufficient_evidence=True`` slice
    (the laundering path issue #1/#5): a model cannot smuggle base-model prose by
    setting that flag, because the only citation-less prose permitted is the banner
    the guard itself writes. (The ``allow_outside`` labeled-outside carve-out is a
    distinct mode, not exercised by these project_only probes.)"""
    if env.answer_text.strip() and not env.citations:
        assert env.insufficient_evidence is True, env
        assert env.answer_text == NOT_FOUND_MESSAGE, env


@pytest.fixture
def project():
    return build_fixture_project("ph8_safety")


@pytest.fixture
def conn(project):
    c = sqlite3.connect(str(project.db_path))
    c.execute("PRAGMA foreign_keys=ON")
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _clear_backend():
    compose._BACKEND_OVERRIDE = None
    yield
    compose._BACKEND_OVERRIDE = None


def _grounded_answer(project, *, markers=(1,)) -> AnswerEnvelope:
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="It holds across groups [E1].",
                                      cited_markers=list(markers))
    )
    return answer(GROUNDED_Q, project, config=make_answer_config())[0]


# --- Faithfulness (09 §8) --------------------------------------------------

def test_faithfulness_cited_works_subset_of_retrieved(project):
    """``set(cited_work_ids)`` ⊆ the retrieved allowed set — no citation outside what
    was actually shown (09 §8)."""
    env = _grounded_answer(project)
    assert set(env.cited_work_ids) <= set(env.retrieved_item_ids)


def test_faithfulness_cited_spans_subset_of_retrieved(project):
    """``set(cited_span_ids)`` ⊆ the retrieved allowed set (09 §8)."""
    env = _grounded_answer(project)
    assert set(env.cited_span_ids) <= set(env.retrieved_item_ids)


def test_faithfulness_invented_citation_cannot_escape(project):
    """A model citing an out-of-range marker has it dropped — it never reaches the
    envelope (the number→id + membership guarantee, 09 §8)."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="It holds across groups [E1][E99].",
                                      cited_markers=[1, 99])
    )
    env, _ = answer(GROUNDED_Q, project, config=make_answer_config())
    assert "work_99" not in env.cited_work_ids
    assert set(env.cited_span_ids) <= set(env.retrieved_item_ids)


def test_faithfulness_every_cited_work_exists_and_is_included_or_metadata_only(project, conn):
    """Every cited ``work_id`` exists in ``works`` AND is ``included`` /
    ``metadata_only`` in ``project_documents`` (the §11 faithfulness invariant, 09 §8).

    NON-VACUOUS: the fixture's ``work_x`` was excluded *after* extraction and still
    RETAINS a citable span (``s_x``) that shares the ``"across groups"`` phrase with the
    included ``work_a``'s ``s_a`` — so a grounded query reaches both, and the fake
    backend tries to cite both markers ([E1][E2]). The retrieval/guard inclusion filter
    must keep the EXCLUDED work out of ``cited_work_ids`` while a real INCLUDED work IS
    cited. This FAILS against the pre-fix code (which surfaced + cited ``work_x``)."""
    # sanity: the fixture is genuinely adversarial — work_x IS excluded yet retains a span.
    assert conn.execute(
        "SELECT inclusion_status FROM project_documents WHERE work_id='work_x'"
    ).fetchone()[0] == "excluded"
    assert conn.execute(
        "SELECT COUNT(*) FROM evidence_spans WHERE work_id='work_x'"
    ).fetchone()[0] >= 1

    # the fake backend cites BOTH the included evidence and (attempts) the excluded one.
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="It holds across groups [E1][E2].",
                                      cited_markers=[1, 2])
    )
    env, _ = answer(GROUNDED_Q, project, config=make_answer_config())

    # the EXCLUDED work is never citable ...
    assert "work_x" not in env.cited_work_ids, env
    # ... while a real INCLUDED work IS cited (non-vacuous), and every cited work
    # resolves to an included/metadata_only project_documents row.
    assert env.cited_work_ids
    for work_id in env.cited_work_ids:
        assert conn.execute("SELECT 1 FROM works WHERE work_id=?", (work_id,)).fetchone()
        status = conn.execute(
            "SELECT inclusion_status FROM project_documents WHERE work_id=?", (work_id,)
        ).fetchone()
        assert status is not None and status[0] in ("included", "metadata_only")


def test_faithfulness_no_prose_with_zero_supporting_citations(project):
    """The UNCONDITIONAL must-fix invariant (plan §11) closing the 09 §8 hole: no
    envelope carries non-empty prose with zero supporting citations, save the honest
    not-found banner. The predicate ``_assert_no_unsupported_prose`` does NOT exclude
    the ``insufficient_evidence=True`` slice — it is the exact laundering path the old
    (weaker) ``not insufficient_evidence`` predicate let through."""
    # 1) grounded answer -> prose + citations (passes; has citations).
    grounded = _grounded_answer(project)
    _assert_no_unsupported_prose(grounded)
    assert grounded.answer_text and grounded.citations

    # 2) model cites ONLY an invented marker -> guard abstains (banner + flag set).
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="The corpus proves it [E99].",
                                      cited_markers=[99])
    )
    abstained, _ = answer(GROUNDED_Q, project, config=make_answer_config())
    _assert_no_unsupported_prose(abstained)
    assert abstained.insufficient_evidence is True
    assert abstained.citations == []
    assert abstained.answer_category == AnswerCategory.unresolved
    assert "unsupported_prose_suppressed" in abstained.warnings

    # 3) no-evidence question -> abstains, never fabricates grounded prose.
    none_env, _ = answer("What is the capital of France?", project, config=make_answer_config())
    _assert_no_unsupported_prose(none_env)


def test_faithfulness_insufficient_flag_cannot_launder_base_model_prose(project):
    """A model that self-declares ``insufficient_evidence=true`` WHILE emitting
    unsupported base-model prose and ZERO citations cannot launder that prose into the
    envelope: over a question that retrieves a real span (so the LLM IS called), the
    guard forces clean abstention — suppressing the prose, dropping the contradictory
    category, and warning. This is the precise HIGH-severity hole (issues #5/#7); the
    old guard gated post-drop abstention behind ``not insufficient_evidence`` and let
    it through."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(
            answer_text="The capital of France is Paris, recalled from general knowledge.",
            cited_markers=[],
            answer_category="source_grounded",
            insufficient_evidence=True,
        )
    )
    env, _ = answer(GROUNDED_Q, project, config=make_answer_config())
    _assert_no_unsupported_prose(env)                       # the load-bearing invariant
    assert env.insufficient_evidence is True
    assert env.citations == []
    assert "Paris" not in env.answer_text                   # base-model prose suppressed
    assert env.answer_category == AnswerCategory.unresolved  # contradictory category fixed
    assert "unsupported_prose_suppressed" in env.warnings


def test_insufficient_flag_reconciled_when_grounded_citation_survives(project):
    """Defect 2 — a model that self-declares ``insufficient_evidence=true`` WHILE
    emitting prose AND ≥1 valid SURVIVING citation must NOT yield a self-contradictory
    envelope. The guard has final say (decision 82): it reconciles the flag FROM the
    evidence — ≥support_floor valid citations ⇒ grounded ⇒ ``insufficient_evidence`` is
    forced False, the prose + citation are kept, and the category is reconciled. The
    envelope is never internally contradictory (insufficient=true + prose + citations).

    Pre-fix this slipped through: ``reconcile`` and the post-drop block both skipped when
    the citation count was ≥ support_floor or the flag was already True, leaving the
    contradiction intact."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(
            answer_text="Paris is the capital [E1].",
            cited_markers=[1],
            answer_category="source_grounded",
            insufficient_evidence=True,
        )
    )
    env, _ = answer(GROUNDED_Q, project, config=make_answer_config())
    # evidence-driven branch: ≥1 valid surviving citation ⇒ grounded.
    assert env.citations
    assert env.insufficient_evidence is False
    assert env.answer_text == "Paris is the capital [E1]."     # prose kept
    assert env.answer_category == AnswerCategory.source_grounded  # reconciled from evidence
    assert "insufficient_flag_overridden" in env.warnings
    # the envelope is NOT internally contradictory.
    assert not (env.insufficient_evidence and env.citations and env.answer_text.strip())


# --- Corpus-leakage abstention (09 §9) -------------------------------------

def test_corpus_leakage_abstention_on_base_model_known_question(project):
    """A base-model-known question with no corpus support yields
    ``insufficient_evidence=True`` rather than an answer from model memory (09 §9)."""
    compose._BACKEND_OVERRIDE = FakeLLMBackend(
        response=canned_envelope_json(answer_text="Paris.", cited_markers=[],
                                      answer_category="outside_corpus")
    )
    env, _ = answer("What is the capital of France?", project, config=make_answer_config())
    assert env.insufficient_evidence is True
    assert env.answer_category == AnswerCategory.unresolved


# --- Content-access boundary (09 §10) --------------------------------------

def test_content_access_no_query_touches_another_project_db(project, conn):
    """No query during ``answer()`` reads another project's rows — cited works stay
    within THIS project (09 §10 boundary)."""
    other = build_fixture_project("ph8_safety_other")
    # mark the other project's work with a sentinel so a cross-read would be visible.
    oc = sqlite3.connect(str(other.db_path))
    oc.execute("UPDATE works SET canonical_title='OTHER_PROJECT_SENTINEL' WHERE work_id='work_a'")
    oc.commit()
    oc.close()

    env = _grounded_answer(project)
    this_works = {
        r[0] for r in conn.execute("SELECT work_id FROM works").fetchall()
    }
    assert set(env.cited_work_ids) <= this_works
    for c in env.citations:
        assert c.title != "OTHER_PROJECT_SENTINEL"


def test_content_access_private_fragments_not_sent_to_external_profile(project, monkeypatch):
    """``user_supplied_private`` fragments are not sent to an external profile when
    ``content_policy.external_llm_for_answer_generation`` forbids it (09 §10)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    from seedgraph.config.models import default_profiles

    backend = FakeLLMBackend(
        response=canned_envelope_json(answer_text="leak [E1].", cited_markers=[1])
    )
    compose._BACKEND_OVERRIDE = backend
    # external preferred, NO local profile available, policy forbids private external.
    profs = {
        "anthropic_api_default": default_profiles()["anthropic_api_default"],
        "no_llm": default_profiles()["no_llm"],
    }
    cfg = make_answer_config(preferred="anthropic_api_default", private_external=False,
                             profiles=profs)
    # 'hidden data' retrieves the user_supplied_private span s_priv.
    env, _ = answer('Where is the "hidden data" used?', project, config=cfg)
    assert backend.calls == []  # the private fragment was NEVER dispatched
    assert env.mode == AnswerMode.RETRIEVAL_ONLY
    assert "private_content_local_only" in env.warnings


def test_content_access_non_private_restricted_fragment_not_sent_external(project, monkeypatch):
    """A NON-shareable but non-``user_supplied_private`` fragment (``licensed_future``)
    is also gated from an external profile under the default-deny lattice (issue #9):
    the §7 gate keys on ``is_shareable`` (open_access / metadata_only only), so a
    restricted fragment cannot leak to an external API when policy forbids it (09 §10,
    decisions 76/30/60)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    from datetime import datetime, timezone

    from seedgraph.config.models import default_profiles

    # add a licensed_future span (restricted, non-shareable, but not 'private').
    oc = sqlite3.connect(str(project.db_path))
    oc.execute("PRAGMA foreign_keys=ON")
    now = datetime.now(timezone.utc).isoformat()
    oc.execute(
        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
        "source_file_hash, work_id, start_char, end_char, exact_quote, quote_hash, "
        "access_class, created_at) VALUES ('s_lic','md_work_a','h','sf','sfh','work_a',0,33,"
        "'licensed cohort microdata table 7','qh_lic','licensed_future',?)",
        (now,),
    )
    oc.execute(
        "INSERT INTO span_fts(quote_text, span_id, markdown_id, work_id, section_id) "
        "VALUES ('licensed cohort microdata table 7','s_lic','md_work_a','work_a',NULL)"
    )
    oc.commit()
    oc.close()

    backend = FakeLLMBackend(
        response=canned_envelope_json(answer_text="leak [E1].", cited_markers=[1])
    )
    compose._BACKEND_OVERRIDE = backend
    profs = {
        "anthropic_api_default": default_profiles()["anthropic_api_default"],
        "no_llm": default_profiles()["no_llm"],
    }
    cfg = make_answer_config(preferred="anthropic_api_default", private_external=False,
                             profiles=profs)
    env, _ = answer('Where is the "licensed cohort microdata table 7" used?', project, config=cfg)
    assert backend.calls == []  # the restricted fragment was NEVER dispatched externally
    assert env.mode == AnswerMode.RETRIEVAL_ONLY
    assert "private_content_local_only" in env.warnings


def test_content_access_envelope_has_no_full_text_dump(project, conn):
    """The envelope carries no full-text dump: every citation quote is the verbatim
    ``evidence_spans.exact_quote`` of one of its OWN spans — a single stored bounded
    fragment, never a whole document or a cross-span concatenation (09 §10, plan §7,
    decision 53). Checked STRUCTURALLY (the quote must BE a stored single-span fragment)
    so the invariant holds for a span of any length — not incidentally because the
    fixture span quotes happen to be short (a >max_fragment_chars span is still only a
    bounded fragment, and a char-length cap would be the wrong proxy)."""
    env = _grounded_answer(project)
    assert env.citations  # the grounded answer cites a real span
    for c in env.citations:
        if c.quote is None or not c.span_ids:
            continue
        placeholders = ",".join("?" * len(c.span_ids))
        stored = {
            r[0]
            for r in conn.execute(
                f"SELECT exact_quote FROM evidence_spans WHERE span_id IN ({placeholders})",
                c.span_ids,
            ).fetchall()
        }
        # the quote IS exactly one of this citation's stored span fragments — never a
        # full-document dump nor a concatenation across spans.
        assert c.quote in stored, c
    # provenance attests no full source text left the machine.
    assert env.llm_provenance is None or env.llm_provenance.get("source_text_left_machine") is False
