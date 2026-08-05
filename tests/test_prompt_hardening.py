"""Build C chunk 3 — prompt hardening acceptance tests (OFFLINE).

Covers the two prompt-hardening items shipped under the single PROMPT_VERSION
1.1.0 bump (design D4/D5):

* standard-names rule — ``normalized_label`` MUST be the standard community
  canonical name, pinned by the distinctive phrase (critic C-5) plus the worked
  examples, in BOTH the system prompt and the user-prompt claim-field line;
* references-tail exclusion — ``_strip_references`` splices references-kind
  section ranges out of the PROMPT text only (visible ``[references omitted]``
  marker, fail-open on no-sections / all-references), the whole-doc runner
  dispatches the trimmed prompt while ``ensure_span`` still anchors against the
  FULL markdown, the oversize gate estimates the trimmed prompt, and the chunked
  planner never reads a references section (``_pseudo_sections`` never filtered).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import date
from pathlib import Path

from sqlmodel import Session

from seedgraph import cache_access
from seedgraph.acquisition.bridge import write_bridge
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.config.models import (
    ContentPolicy,
    GlobalConfig,
    LlmBudget,
    LlmCapabilities,
    LLMConfig,
    ModelCapability,
    TaskRoute,
    default_profiles,
)
from seedgraph.db.adapter import raw_conn
from seedgraph.extraction.chunked_runner import extract_note_chunked
from seedgraph.extraction.prompt import _SYSTEM_PROMPT, build_prompt
from seedgraph.extraction.runner import (
    REFERENCES_OMITTED_MARKER,
    _strip_references,
    extract_note,
)
from seedgraph.extraction.schema import PROMPT_VERSION
from seedgraph.llm.backend import FakeLLMBackend
from seedgraph.project import service
from seedgraph.sections.parser import parse_sections
from seedgraph.sections.store import replace_sections
from seedgraph.vocab import AccessClass, AcquisitionMethod

# --------------------------------------------------------------------------
# Fixtures / helpers (offline patterns shared with test_phase_4 / test_phase_4b)
# --------------------------------------------------------------------------

BODY_Q = "We study whether widgets improve production outcomes in a panel of firms."
METHOD_Q = "We use a two-way fixed effects regression model on the widget panel."
REFS_LINE = (
    "Doe, J. (1999). Widget dynamics and firm outcomes. Journal of Widgetry, 1(1), 1-10."
)

MD_REFS = (
    "# Widgets and Outcomes\n\n"
    f"{BODY_Q}\n\n"
    "## Methods\n\n"
    f"{METHOD_Q}\n\n"
    "## References\n\n"
    f"{REFS_LINE}\n"
    "Roe, R. (2001). Panel methods for widget data. Journal of Panels, 2(2), 11-20.\n"
)


def _note() -> dict:
    """A canned note whose every found quote lives in the RETAINED (non-refs) body."""
    return {
        "research_question": {
            "claim_text": "Do widgets improve outcomes?",
            "normalized_label": "widget productivity",
            "status": "found",
            "assertion_status": "stated",
            "confidence": 0.9,
            "exact_quote": BODY_Q,
        },
        "method_or_model": [
            {
                "claim_text": "Two-way fixed effects regression.",
                "method_type": "regression",
                "status": "found",
                "assertion_status": "stated",
                "exact_quote": METHOD_Q,
            }
        ],
    }


def make_config(*, preferred: str = "local_ollama_default") -> GlobalConfig:
    routes = {
        "note_extraction": TaskRoute(
            task_type="note_extraction",
            preferred_profile=preferred,
            fallback_profile=None,
            requires_source_text=True,
        )
    }
    return GlobalConfig(
        llm=LLMConfig(profiles=default_profiles(), routes=routes),
        content_policy=ContentPolicy(external_llm_for_private_full_text=False),
        budget=LlmBudget(),
    )


def _caps(window: int, *, max_output: int | None = None) -> LlmCapabilities:
    return LlmCapabilities(
        snapshot_date=date(2026, 1, 1),
        models={
            "llama3": ModelCapability(
                provider="ollama",
                context_window_tokens=window,
                max_output_tokens=max_output,
                input_usd_per_mtok=0.0,
                output_usd_per_mtok=0.0,
                supports_structured_output=False,
                access_modes=["local"],
                pricing_status="not_applicable",
            )
        },
    )


def _make_doc(slug, markdown, *, doi="10.1/x"):
    h = service.create_project(slug)
    w = service.add_work(h, ids={"doi": doi}, title="W " + slug)
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{slug}.pdf"
    p.write_bytes(b"%PDF-1.4 " + slug.encode() + b" body content with words here")
    src = ingest_file(
        p,
        access_class=AccessClass.open_access,
        acquisition_method=AcquisitionMethod.open_access_fetch,
        root=None,
    )
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
            acquisition_method=AcquisitionMethod.open_access_fetch.value,
        )
        s.commit()
    return h, w.work_id, md


def _build_sections(h, wid, markdown_id) -> list:
    """Parse + store document_sections for a work; return the parsed sections."""
    cache_conn = cache_access.open_cache_ro(None)
    try:
        mdrow = cache_access.read_markdown(cache_conn, None, markdown_id)
        secs = parse_sections(
            mdrow.text,
            markdown_id=markdown_id,
            markdown_hash=mdrow.markdown_hash,
            source_file_id=mdrow.source_file_id,
            source_file_hash=mdrow.source_file_hash,
            work_id=wid,
        )
        with Session(h.engine) as session:
            replace_sections(raw_conn(session), markdown_id, secs)
            session.commit()
    finally:
        cache_conn.close()
    return secs


def _parse_only(markdown: str) -> list:
    """parse_sections with dummy ids (unit tests that never touch a store)."""
    return parse_sections(
        markdown,
        markdown_id="md_x",
        markdown_hash="h_x",
        source_file_id="src_x",
        source_file_hash="sh_x",
        work_id="w_x",
    )


def _run_extract(h, wid, *, note=None, capabilities=None, **kwargs):
    fake = FakeLLMBackend(response=note)
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            result = extract_note(
                session,
                cache_conn,
                work_id=wid,
                backend=fake,
                config=make_config(),
                cache_root=None,
                capabilities=capabilities,
                **kwargs,
            )
    finally:
        cache_conn.close()
    return result, fake


def _run_chunked(h, wid, *, note=None, caps=None, **kwargs):
    fake = FakeLLMBackend(response=note)
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            res = extract_note_chunked(
                session,
                cache_conn,
                work_id=wid,
                backend=fake,
                config=make_config(),
                capabilities=caps,
                cache_root=None,
                **kwargs,
            )
    finally:
        cache_conn.close()
    return res, fake


def _dbconn(h) -> sqlite3.Connection:
    conn = sqlite3.connect(str(h.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# --------------------------------------------------------------------------
# prompt text — standard-names rule (D4; acceptance #5's distinctive phrase)
# --------------------------------------------------------------------------


def test_system_prompt_carries_standard_name_rule():
    """The distinctive phrase (critic C-5) + worked examples + normalized_label scope."""
    assert "standard community canonical name" in _SYSTEM_PROMPT.lower()
    # 2-3 worked, domain-neutral examples are present.
    assert "Masked Language Modeling" in _SYSTEM_PROMPT
    assert "difference-in-differences" in _SYSTEM_PROMPT
    assert "Monte Carlo simulation" in _SYSTEM_PROMPT
    # The rule scopes to normalized_label ONLY; quotes keep the paper's language.
    assert "ONLY to `normalized_label`" in _SYSTEM_PROMPT
    assert "`exact_quote` keep the paper's own language" in _SYSTEM_PROMPT


def test_user_prompt_claim_field_line_carries_rule():
    _system, user = build_prompt("SOME PAPER TEXT")
    assert "standard community canonical name" in user.lower()
    assert "same concept, same label across papers" in user


def test_prompt_version_bumped_once_for_both_items():
    assert PROMPT_VERSION == "1.1.0"


# --------------------------------------------------------------------------
# _strip_references (D5) — splice, marker, fail-open
# --------------------------------------------------------------------------


def test_strip_references_splices_range_and_inserts_marker():
    secs = _parse_only(MD_REFS)
    assert any(s.section_kind == "references" for s in secs)
    trimmed = _strip_references(MD_REFS, secs)
    assert REFS_LINE not in trimmed
    assert REFERENCES_OMITTED_MARKER in trimmed
    # Retained body is untouched (verbatim quotes still findable in the prompt).
    assert BODY_Q in trimmed
    assert METHOD_Q in trimmed


def test_strip_references_fail_open_paths():
    # No sections rows -> full text unchanged.
    assert _strip_references(MD_REFS, []) == MD_REFS
    # Sections exist but none are references-kind -> unchanged.
    no_refs_md = "# Title\n\nBody text here.\n\n## Methods\n\nMore body.\n"
    assert _strip_references(no_refs_md, _parse_only(no_refs_md)) == no_refs_md
    # An all-references document (annotated bibliography) -> trimming would empty
    # the prompt, so the FULL text is dispatched (fail-open).
    all_refs = "# References\n\nDoe, J. (1999). Widget dynamics. J. Widgetry.\n"
    assert _strip_references(all_refs, _parse_only(all_refs)) == all_refs


def test_build_prompt_embeds_trimmed_text():
    trimmed = _strip_references(MD_REFS, _parse_only(MD_REFS))
    _system, user = build_prompt(trimmed)
    assert REFERENCES_OMITTED_MARKER in user
    assert REFS_LINE not in user
    assert BODY_Q in user


# --------------------------------------------------------------------------
# runner (whole-doc) — trimmed dispatch, full-markdown anchoring, no-sections
# --------------------------------------------------------------------------


def test_runner_dispatches_trimmed_prompt_but_anchors_full_markdown():
    """Sectioned fixture: refs body absent from the dispatched user prompt (via
    FakeLLMBackend.calls) + visible marker, while ensure_span still anchors the
    retained-body quotes against the FULL stored markdown (span_count >= 1)."""
    h, wid, md = _make_doc("refs_trim", MD_REFS)
    _build_sections(h, wid, md.markdown_id)
    result, fake = _run_extract(h, wid, note=_note())
    assert result.run_status == "success"
    assert len(fake.calls) == 1
    user = fake.calls[0]["user"]
    assert REFS_LINE not in user
    assert REFERENCES_OMITTED_MARKER in user
    assert BODY_Q in user  # retained body was dispatched
    # Spans anchored against the FULL markdown (offsets index MD_REFS itself).
    assert result.span_count >= 1
    conn = _dbconn(h)
    try:
        rows = conn.execute(
            "SELECT exact_quote, start_char, end_char FROM evidence_spans"
        ).fetchall()
        quotes = {r["exact_quote"]: (r["start_char"], r["end_char"]) for r in rows}
        assert BODY_Q in quotes
        start, end = quotes[BODY_Q]
        assert MD_REFS[start:end] == BODY_Q  # offsets are FULL-markdown coordinates
    finally:
        conn.close()


def test_runner_no_sections_dispatches_untrimmed_full_text():
    h, wid, _md = _make_doc("refs_nosec", MD_REFS)
    # NO sections built -> fail-open: the full text (references included) goes out.
    result, fake = _run_extract(h, wid, note=_note())
    assert result.run_status == "success"
    user = fake.calls[0]["user"]
    assert REFS_LINE in user
    assert REFERENCES_OMITTED_MARKER not in user


# --------------------------------------------------------------------------
# oversize gate — the token estimate reflects the TRIMMED prompt (D5)
# --------------------------------------------------------------------------

# Small body + a references tail big enough to blow a 4096-token window on its
# own: full prompt ~9k tokens, trimmed prompt ~1k tokens.
_BIG_REFS_MD = (
    "# Widgets and Outcomes\n\n"
    f"{BODY_Q}\n\n"
    "## Methods\n\n"
    f"{METHOD_Q}\n\n"
    "## References\n\n"
    + "\n".join(
        f"Author{i}, A. ({1900 + i}). A long and winding reference entry number {i} "
        f"about widget dynamics. Journal of Widgetry, {i}({i}), 1-10."
        for i in range(250)
    )
    + "\n"
)


def test_oversize_gate_estimates_trimmed_prompt():
    """With sections, the trimmed prompt fits the window -> dispatch succeeds;
    the identical doc without sections overflows -> skipped_oversize, no dispatch
    (the gate estimates the BUILT prompt, so gate and dispatch agree — D5)."""
    caps = _caps(4096)
    # Sectioned: references trimmed -> under the window -> success.
    h1, wid1, md1 = _make_doc("gate_trim", _BIG_REFS_MD, doi="10.1/a")
    _build_sections(h1, wid1, md1.markdown_id)
    r1, fake1 = _run_extract(h1, wid1, note=_note(), capabilities=caps)
    assert r1.run_status == "success"
    assert len(fake1.calls) == 1
    assert REFERENCES_OMITTED_MARKER in fake1.calls[0]["user"]
    # No sections: full text -> over the window -> skipped_oversize, NO dispatch.
    h2, wid2, _md2 = _make_doc("gate_full", _BIG_REFS_MD, doi="10.1/b")
    r2, fake2 = _run_extract(h2, wid2, note=_note(), capabilities=caps)
    assert r2.run_status == "skipped_oversize"
    assert fake2.calls == []


# --------------------------------------------------------------------------
# chunked planner — references sections filtered from the plan (D5)
# --------------------------------------------------------------------------

_FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed eiusmod tempor. "

_CHUNK_MD = (
    f"# Section One\n\n{BODY_Q}\n\n" + _FILLER * 18 + "\n\n"
    f"# Section Two\n\n{METHOD_Q}\n\n" + _FILLER * 18 + "\n\n"
    "# Section Three\n\nWe find that outcomes improved by four widget percent.\n\n"
    + _FILLER * 18
    + "\n\n"
    "# References\n\n"
    f"{REFS_LINE}\n"
    "Roe, R. (2001). Panel methods for widget data. Journal of Panels, 2(2), 11-20.\n"
)


def test_chunked_plan_excludes_references_sections():
    """No chunk overlaps a references section range: the refs section_id appears in
    no map run's chunk_section_ids and the refs body reaches no dispatched prompt."""
    h, wid, md = _make_doc("chunk_refs", _CHUNK_MD)
    secs = _build_sections(h, wid, md.markdown_id)
    refs = [s for s in secs if s.section_kind == "references"]
    assert len(refs) == 1
    # Window sized so each body section is its own chunk (overhead ~745 tok,
    # sections ~380 tok each, two contiguous ~770): input_budget = 1450-745 = 705.
    res, fake = _run_chunked(
        h, wid, note=_note(), caps=_caps(1450, max_output=0), overlap_tokens=0
    )
    assert res.status == "success"
    assert res.chunk_count >= 2
    # The references body text never reached the backend.
    for call in fake.calls:
        assert REFS_LINE not in call["user"]
    # No planned chunk covers the references section (chunk_section_ids provenance),
    # and no chunk's char range overlaps the references [start, end) range.
    ref_sec = refs[0]
    conn = _dbconn(h)
    try:
        rows = conn.execute(
            "SELECT chunk_section_ids FROM extraction_runs "
            "WHERE extraction_mode = 'chunked_map'"
        ).fetchall()
        assert rows
        for row in rows:
            assert ref_sec.section_id not in json.loads(row["chunk_section_ids"])
    finally:
        conn.close()
    # Spans still anchor against the FULL markdown (references intact in cache).
    assert res.merged_claim_count > 0


def test_chunked_pseudo_sections_never_filtered():
    """No sections rows -> the whole-doc _pseudo_sections fallback is NEVER trimmed:
    the references body is dispatched as part of the (single) chunk."""
    h, wid, _md = _make_doc("chunk_nosec", MD_REFS)
    res, fake = _run_chunked(h, wid, note=_note(), caps=_caps(50_000, max_output=1024))
    assert res.status == "success"
    assert res.chunk_count == 1
    assert any(REFS_LINE in call["user"] for call in fake.calls)
