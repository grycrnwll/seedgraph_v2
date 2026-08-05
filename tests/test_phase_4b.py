"""Phase 4b — Chunked / Section Extraction acceptance tests (plan §11), OFFLINE.

Every plan §11 unit/acceptance test is implemented with real assertions and runs
with NO network, NO real LLM, and NO key: synthetic markdown is pushed through the
real cache (ingest + FakeMarkerBackend convert), bridged work->markdown, sectioned
via phase-3 ``parse_sections``/``replace_sections``, then map-reduced through the
phase-4b surface with a deterministic stub backend that returns canned per-chunk
``default_research_note_v1`` JSON + token counts. The module-level imports double
as a whole-phase import-cleanliness check.

Decisions exercised: D4 (oversize -> one merged note), D2 (two orthogonal
provenance fields), D5 (disjoint manifest section), D6 (numbered .sql authority +
ORM parity), D7 (raw_conn atomicity), D9 (capability-snapshot sizing).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlmodel import Session
from typer.testing import CliRunner

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
    ModelCapability,
    TaskRoute,
    default_profiles,
)
from seedgraph.db.adapter import raw_conn
from seedgraph.db.migrations import discover_migrations, latest_version
from seedgraph.db.models_project import ExtractedClaim, ExtractionRun
from seedgraph.extraction import chunked_runner as chunked_mod

# Import-cleanliness gate: collecting this module imports the full phase_4b surface.
from seedgraph.extraction.chunked_runner import (  # noqa: F401
    ChunkedExtractionResult,
    extract_note_chunked,
)
from seedgraph.extraction.chunker import Chunk, plan_chunks  # noqa: F401
from seedgraph.extraction.normalize import normalize_note
from seedgraph.extraction.reduce import MergedDraft, merge_chunk_drafts  # noqa: F401
from seedgraph.extraction.runner import extract_note
from seedgraph.extraction.schema import (
    PROMPT_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    DefaultNoteV1,
)
from seedgraph.ids import new_id
from seedgraph.llm.backend import LLMCompletion
from seedgraph.llm.tokens import estimate_tokens
from seedgraph.project import service
from seedgraph.sections.parser import parse_sections
from seedgraph.sections.store import replace_sections
from seedgraph.vocab import AccessClass, AcquisitionMethod

cli = CliRunner()


# ==========================================================================
# Offline fixtures / helpers
# ==========================================================================


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FakeSection:
    """A minimal ``document_sections``-shaped object for chunker unit tests."""

    section_id: str
    start_char: int
    end_char: int
    ordinal: int


def _caps(
    window: int,
    *,
    model: str = "llama3",
    provider: str = "ollama",
    max_output: int | None = None,
    pricing: str = "not_applicable",
    inp: float = 0.0,
    out: float = 0.0,
) -> LlmCapabilities:
    """A single-model synthetic capability snapshot (D9) with a chosen window."""
    return LlmCapabilities(
        snapshot_date=date(2026, 1, 1),
        models={
            model: ModelCapability(
                provider=provider,
                context_window_tokens=window,
                max_output_tokens=max_output,
                input_usd_per_mtok=inp,
                output_usd_per_mtok=out,
                supports_structured_output=False,
                access_modes=["local"] if provider == "ollama" else ["api_key"],
                pricing_status=pricing,
            )
        },
    )


def make_config(
    *,
    preferred: str = "local_ollama_default",
    fallback: str | None = None,
    private_external: bool = False,
    budget: LlmBudget | None = None,
    profiles: dict | None = None,
) -> GlobalConfig:
    profs = profiles if profiles is not None else default_profiles()
    routes = {
        "note_extraction": TaskRoute(
            task_type="note_extraction",
            preferred_profile=preferred,
            fallback_profile=fallback,
            requires_source_text=True,
        )
    }
    return GlobalConfig(
        llm=LLMConfig(profiles=profs, routes=routes),
        content_policy=ContentPolicy(external_llm_for_private_full_text=private_external),
        budget=budget or LlmBudget(),
    )


def _add_doc(h, tag, markdown, *, access_class=AccessClass.open_access, doi="10.1/x"):
    """Add one work + bridged converted markdown; return (work_id, md_convert)."""
    method = (
        AcquisitionMethod.open_access_fetch
        if access_class == AccessClass.open_access
        else AcquisitionMethod.upload
    )
    w = service.add_work(h, ids={"doi": doi}, title="W " + tag)
    p = Path(os.environ["SEEDGRAPH_HOME"]) / f"{tag}.pdf"
    p.write_bytes(b"%PDF-1.4 " + tag.encode() + b" body content with words here")
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


def _make_doc(slug, markdown, *, access_class=AccessClass.open_access, doi="10.1/x"):
    h = service.create_project(slug)
    wid, md = _add_doc(h, slug, markdown, access_class=access_class, doi=doi)
    return h, wid, md


def _build_sections(h, wid, markdown_id) -> int:
    """Parse + store ``document_sections`` for a work (phase-3 substrate)."""
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
            conn = raw_conn(session)
            n = replace_sections(conn, markdown_id, secs)
            session.commit()
    finally:
        cache_conn.close()
    return n


def _dbconn(h) -> sqlite3.Connection:
    conn = sqlite3.connect(str(h.db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _run_chunked(h, wid, *, backend, caps, config=None, **kwargs):
    """Run extract_note_chunked offline; return the ChunkedExtractionResult."""
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            res = extract_note_chunked(
                session,
                cache_conn,
                work_id=wid,
                backend=backend,
                config=config if config is not None else make_config(),
                capabilities=caps,
                cache_root=None,
                **kwargs,
            )
    finally:
        cache_conn.close()
    return res


class MappingBackend:
    """Deterministic stub: return a canned response keyed on a marker in the prompt.

    ``mapping`` maps a marker substring (present in a chunk's text) to its canned
    note (dict -> JSON). A prompt with no recognized marker returns ``default``
    (used to force an unrepairable chunk). Records every call so a test can assert
    NO chunk text was dispatched when a gate refuses.
    """

    def __init__(self, mapping: dict, default: object = "NOT VALID JSON AT ALL"):
        self.mapping = mapping
        self.default = default
        self.calls: list[dict] = []

    def complete(self, system, user, *, model=None, temperature=0.0, max_tokens=4096):
        self.calls.append({"system": system, "user": user, "model": model})
        chosen = None
        for marker, resp in self.mapping.items():
            if marker in user:
                chosen = resp
                break
        if chosen is None:
            chosen = self.default
        text = chosen if isinstance(chosen, str) else json.dumps(chosen)
        return LLMCompletion(
            text=text,
            input_tokens=estimate_tokens(system) + estimate_tokens(user),
            output_tokens=estimate_tokens(text),
        )


# Reusable canned per-claim envelope builders -------------------------------


def _env(text, quote, *, label=None, status="found", assertion="stated", conf=0.8, **extra):
    d = {"claim_text": text, "status": status, "assertion_status": assertion, "confidence": conf}
    if quote is not None:
        d["exact_quote"] = quote
    if label is not None:
        d["normalized_label"] = label
    if assertion == "inferred":
        d["inferred_explanation"] = extra.pop("inferred_explanation", "inferred from context")
    d.update(extra)
    return d


# Three-section fixture: each section is its own chunk under a small window, each
# carries a doc-wide-UNIQUE quotable sentence so spans anchor against the full doc.
FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed eiusmod tempor. "

S1A = "We study whether minimum wage increases reduce teen employment in alpha."
S1B = "The empirical setting is the United States labor market during alpha years."
S2A = "Our main contribution is a brand new beta difference in differences estimator."
S2B = "We rely on the Current Population Survey beta microdata for the analysis."
S3A = "We invoke the parallel trends assumption for identification in the gamma case."
S3B = "We find that teen employment fell by three gamma percent in headline terms."


def _section(heading: str, *sentences: str, pad: int = 20) -> str:
    body = " ".join(sentences)
    return f"# {heading}\n\n{body}\n\n" + (FILLER * pad).strip() + "\n\n"


THREE_SECTION_MD = (
    _section("Section One Alpha", S1A, S1B)
    + _section("Section Two Beta", S2A, S2B)
    + _section("Section Three Gamma", S3A, S3B)
)


def _note0() -> dict:
    return {
        "research_question": _env("Does min wage reduce teen employment?", S1A, label="research question"),
        "setting": _env("US labor market.", S1B, label="setting"),
    }


def _note1() -> dict:
    return {
        "main_contribution": _env("A new DiD estimator.", S2A, label="contribution"),
        "data_sources": [_env("CPS microdata.", S2B, label="cps")],
    }


def _note2() -> dict:
    return {
        "assumptions": [
            _env("Parallel trends holds.", S3A, label="parallel trends", assumption_type="identification")
        ],
        "main_results": [
            _env(
                "Teen employment fell three percent.",
                S3B,
                label="headline effect",
                result_type="point_estimate",
                assertion="inferred",
                inferred_explanation="derived from the reported headline figure",
            )
        ],
    }


def _three_chunk_backend() -> MappingBackend:
    # Markers are the section-unique sentences; each chunk's text contains exactly one.
    return MappingBackend({S1A: _note0(), S2A: _note1(), S3A: _note2()})


# Window sized so each ~section packs as its OWN chunk (3 chunks). overhead ~745
# (Build C chunk 3 standard-names rule grew the prompt from ~573); reserve_output=0
# -> input_budget = window-745. Each section (~420 tok) < budget; two together
# (~838 contiguous) exceed it. dispatch_budget = window leaves room for the default
# 128-token overlap on chunks after the first (chunk est ~546+745=1291 <= window).
_THREE_WINDOW = 1450


def _three_caps() -> LlmCapabilities:
    return _caps(_THREE_WINDOW, max_output=0)


def _setup_three_section(slug="threesec", doi="10.4b/3"):
    h, wid, md = _make_doc(slug, THREE_SECTION_MD, doi=doi)
    _build_sections(h, wid, md.markdown_id)
    return h, wid, md


# ==========================================================================
# chunker (plan §11 unit)
# ==========================================================================


def _even_sections(full: str, n: int) -> list[FakeSection]:
    step = len(full) // n
    secs = []
    for i in range(n):
        start = i * step
        end = len(full) if i == n - 1 else (i + 1) * step
        secs.append(FakeSection(f"sec{i}", start, end, i))
    return secs


def test_chunker_packs_into_deterministic_chunks():
    sentence = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
    full = "".join(f"# H{i}\n\n{sentence * 2}\n\n" for i in range(4))
    secs = _even_sections(full, 4)
    cap = _caps(200).models["llama3"]
    # Budget = 200 - 10 - 10 = 180 tokens. Each ~section ~ a few dozen tokens, so
    # several pack per chunk; whole-section boundaries are kept and offsets index
    # the FULL markdown.
    chunks = plan_chunks(
        secs, full, model_caps=cap, prompt_overhead_tokens=10, reserved_output_tokens=10, overlap_tokens=0
    )
    assert len(chunks) >= 1
    # Offsets index the FULL markdown, are ordered, and tile forward without going back.
    assert chunks[0].start_char == 0
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char >= a.start_char
        assert a.end_char <= b.end_char
    assert chunks[-1].end_char == len(full)
    # Every chunk carries the section ids it covers.
    covered = [sid for c in chunks for sid in c.section_ids]
    assert covered == [f"sec{i}" for i in range(4)]


def test_chunker_oversized_section_is_paragraph_subsplit():
    para = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda. "
    section_text = "\n\n".join([para.strip()] * 4)  # 4 paragraphs
    full = section_text
    secs = [FakeSection("big", 0, len(full), 0)]
    budget = estimate_tokens(para)  # ~ one paragraph
    cap = _caps(budget + 12).models["llama3"]
    chunks = plan_chunks(
        secs, full, model_caps=cap, prompt_overhead_tokens=6, reserved_output_tokens=6, overlap_tokens=0
    )
    # The single oversized section is split on paragraph boundaries into >1 chunk.
    assert len(chunks) > 1
    assert all(c.section_ids == ["big"] for c in chunks)


def test_chunker_oversized_paragraph_flagged_skipped_oversize_section():
    huge = "word " * 400  # one paragraph, no blank lines, ~ 500 tokens
    full = huge.strip()
    secs = [FakeSection("huge", 0, len(full), 0)]
    cap = _caps(60).models["llama3"]
    input_budget = 60 - 5 - 5
    chunks = plan_chunks(
        secs, full, model_caps=cap, prompt_overhead_tokens=5, reserved_output_tokens=5, overlap_tokens=0
    )
    # A single paragraph that still exceeds the budget is its own over-budget chunk
    # (the runner then records skipped_oversize_section for that span — D4).
    assert len(chunks) == 1
    content_tokens = chunks[0].est_prompt_tokens - 5
    assert content_tokens > input_budget


def test_chunker_replanning_identical_markdown_is_deterministic():
    full = "".join(f"# H{i}\n\nsentence number {i} here with words.\n\n" for i in range(5))
    secs = _even_sections(full, 5)
    cap = _caps(80).models["llama3"]
    kw = dict(model_caps=cap, prompt_overhead_tokens=8, reserved_output_tokens=8, overlap_tokens=4)
    a = plan_chunks(secs, full, **kw)
    b = plan_chunks(secs, full, **kw)
    assert a == b  # frozen dataclass equality (incl. section_ids lists)


def test_chunker_overlap_is_honored():
    full = "".join(f"# H{i}\n\n{('lexeme ' * 20).strip()}\n\n" for i in range(4))
    secs = _even_sections(full, 4)
    cap = _caps(120).models["llama3"]
    base = dict(model_caps=cap, prompt_overhead_tokens=10, reserved_output_tokens=10)
    no_overlap = plan_chunks(secs, full, overlap_tokens=0, **base)
    with_overlap = plan_chunks(secs, full, overlap_tokens=6, **base)
    assert len(no_overlap) == len(with_overlap) >= 2
    # The fixed overlap pulls each subsequent chunk's read scope earlier (trailing
    # text of the previous chunk is carried into the next).
    assert with_overlap[1].start_char < no_overlap[1].start_char
    overlap_chars = 6 * 4
    assert with_overlap[1].start_char == max(0, no_overlap[1].start_char - overlap_chars)


def test_chunker_reads_window_from_capabilities_yaml():
    full = "".join(f"# H{i}\n\n{('token ' * 25).strip()}\n\n" for i in range(6))
    secs = _even_sections(full, 6)
    base = dict(prompt_overhead_tokens=10, reserved_output_tokens=10, overlap_tokens=0)
    big = plan_chunks(secs, full, model_caps=_caps(400).models["llama3"], **base)
    small = plan_chunks(secs, full, model_caps=_caps(120).models["llama3"], **base)
    # D9: a smaller context window yields MORE chunks (budget = window - overhead -
    # reserved-output, so the packer fits fewer sections per chunk).
    assert len(small) > len(big)


# ==========================================================================
# reduce (plan §11 unit)
# ==========================================================================


def test_reduce_array_union_and_dedup():
    c0 = DefaultNoteV1.model_validate(
        {"assumptions": [_env("PT v0", "quote alpha unique", label="parallel trends", conf=0.4,
                              assumption_type="identification")]}
    )
    c1 = DefaultNoteV1.model_validate(
        {"assumptions": [_env("PT v1", "quote beta unique", label="Parallel Trends", conf=0.9,
                              assumption_type="identification")]}
    )
    d0, _, _ = normalize_note(c0)
    d1, _, _ = normalize_note(c1)
    merged, _, _ = merge_chunk_drafts([d0, d1])
    asm = [m for m in merged if m.field_key.startswith("assumptions[")]
    # Same (claim_type, normalized_label casefold) across two chunks -> ONE claim.
    assert len(asm) == 1
    claim = asm[0]
    # BOTH chunks' spans are unioned (no contributing evidence is lost — §12).
    assert set(claim.exact_quotes) == {"quote alpha unique", "quote beta unique"}
    # The highest-confidence text is kept.
    assert claim.claim_text == "PT v1"


def test_reduce_scalar_best_confidence_selection():
    notes = [
        DefaultNoteV1.model_validate({"research_question": _env("RQ mid", "q mid", conf=0.6)}),
        DefaultNoteV1.model_validate({"research_question": _env("RQ high", "q high", conf=0.95)}),
        DefaultNoteV1.model_validate({"research_question": _env("RQ low", "q low", conf=0.2)}),
    ]
    per_chunk = [normalize_note(n)[0] for n in notes]
    merged, _, _ = merge_chunk_drafts(per_chunk)
    rq = [m for m in merged if m.field_key == "research_question"]
    assert len(rq) == 1
    assert rq[0].claim_text == "RQ high"
    assert rq[0].source_chunk_index == 1

    # Deterministic tie-break by earliest chunk_index when confidences tie.
    tie = [
        normalize_note(DefaultNoteV1.model_validate({"research_question": _env("first", "qa", conf=0.7)}))[0],
        normalize_note(DefaultNoteV1.model_validate({"research_question": _env("second", "qb", conf=0.7)}))[0],
    ]
    merged2, _, _ = merge_chunk_drafts(tie)
    rq2 = [m for m in merged2 if m.field_key == "research_question"][0]
    assert rq2.claim_text == "first" and rq2.source_chunk_index == 0


def test_reduce_not_found_only_when_absent_in_all_chunks():
    # open_questions absent in both chunks; research_question found in one.
    c0 = DefaultNoteV1.model_validate({"research_question": _env("RQ", "q rq")})
    c1 = DefaultNoteV1.model_validate({"setting": _env("S", "q s")})
    merged, _, _ = merge_chunk_drafts([normalize_note(c0)[0], normalize_note(c1)[0]])
    by_key = {}
    for m in merged:
        by_key.setdefault(m.field_key, []).append(m)
    # A field absent in EVERY chunk -> exactly one not_found.
    oq = by_key["open_questions"]
    assert len(oq) == 1 and oq[0].status == "not_found"
    # A field found in ANY chunk -> no not_found row.
    rq = by_key["research_question"]
    assert len(rq) == 1 and rq[0].status == "found"
    setting = by_key["setting"]
    assert len(setting) == 1 and setting[0].status == "found"


def test_reduce_carries_provenance_fields():
    c0 = DefaultNoteV1.model_validate(
        {
            "main_results": [
                _env("fell 3pct", "q result", label="effect", result_type="point_estimate",
                     assertion="inferred", inferred_explanation="derived from headline", conf=0.8)
            ]
        }
    )
    merged, _, _ = merge_chunk_drafts([normalize_note(c0)[0]], source_run_ids=["extr_map0"])
    res = [m for m in merged if m.field_key.startswith("main_results[")][0]
    # D2 carried from the winning draft; inferred winner keeps llm_inferred + inferred.
    assert res.epistemic_type == "llm_inferred"
    assert res.assertion_status == "inferred"
    assert res.inferred_explanation == "derived from headline"
    # phase_4b provenance: chunk origin + map run back-link.
    assert res.source_chunk_index == 0
    assert res.source_extraction_run_id == "extr_map0"


# ==========================================================================
# chunked_runner (plan §11 unit + integration)
# ==========================================================================


def test_chunked_runner_happy_path_three_chunks():
    h, wid, md = _setup_three_section()
    backend = _three_chunk_backend()
    res = _run_chunked(h, wid, backend=backend, caps=_three_caps())
    assert res.status == "success", res.skipped_reason
    assert res.chunk_count == 3 and res.chunks_succeeded == 3

    conn = _dbconn(h)
    try:
        # Exactly ONE current structured note.
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
        note_row = conn.execute("SELECT * FROM structured_notes").fetchone()
        # 3 chunked_map runs (each with chunk_index + chunk_section_ids) + 1 reduce.
        maps = conn.execute(
            "SELECT * FROM extraction_runs WHERE extraction_mode='chunked_map' ORDER BY chunk_index"
        ).fetchall()
        assert len(maps) == 3
        for i, m in enumerate(maps):
            assert m["chunk_index"] == i
            assert m["chunk_count"] == 3
            assert json.loads(m["chunk_section_ids"])  # non-empty section id list
            assert m["parent_extraction_run_id"] == res.reduce_run_id
        reduces = conn.execute(
            "SELECT * FROM extraction_runs WHERE extraction_mode='chunked_reduce'"
        ).fetchall()
        assert len(reduces) == 1
        reduce = reduces[0]
        assert reduce["run_status"] == "success"
        # Reduce run bound 1:1 to the note.
        assert note_row["extraction_run_id"] == reduce["extraction_run_id"] == res.reduce_run_id
        # Aggregate tokens = sum over map runs.
        assert reduce["input_tokens"] == sum(m["input_tokens"] for m in maps)
        assert reduce["output_tokens"] == sum(m["output_tokens"] for m in maps)

        # Every found claim has >=1 claim_spans -> evidence_spans; not_found has 0.
        found = conn.execute(
            "SELECT claim_id FROM extracted_claims WHERE status='found'"
        ).fetchall()
        assert len(found) >= 6
        for (claim_id,) in [(r["claim_id"],) for r in found]:
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

        # access_class stamped most-restrictive on all runs + note + claims + spans.
        run_classes = {r[0] for r in conn.execute("SELECT DISTINCT access_class FROM extraction_runs")}
        assert run_classes == {"open_access"}
        assert note_row["access_class"] == "open_access"
        claim_classes = {r[0] for r in conn.execute("SELECT DISTINCT access_class FROM extracted_claims")}
        assert claim_classes == {"open_access"}
        span_classes = {r[0] for r in conn.execute("SELECT DISTINCT access_class FROM evidence_spans")}
        assert span_classes == {"open_access"}

        # claim_fts / note_fts populated.
        assert conn.execute("SELECT COUNT(*) FROM claim_fts").fetchone()[0] >= 6
        assert conn.execute("SELECT COUNT(*) FROM note_fts").fetchone()[0] == 1

        # Provenance back-links: merged claims carry source_chunk_index + map run id.
        prov = conn.execute(
            "SELECT COUNT(*) FROM extracted_claims "
            "WHERE status='found' AND source_chunk_index IS NOT NULL "
            "AND source_extraction_run_id IS NOT NULL"
        ).fetchone()[0]
        assert prov >= 6
    finally:
        conn.close()


def test_d7_span_and_claim_commit_atomically():
    """D7: ensure_span INSERT (via raw_conn) + ORM claim + raw claim_spans commit in
    one txn; a forced rollback hides both — no orphan span, no orphan claim."""
    from seedgraph.spans.store import ensure_span

    def _seed(conn, wid, md, access="open_access"):
        run_id = new_id("extr")
        note_id = new_id("note")
        now = _iso()
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, "
            "schema_id, schema_version, prompt_version, access_class, external_full_text, run_status, "
            "created_at, extraction_mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, wid, md.markdown_id, md.markdown_hash, SCHEMA_ID, SCHEMA_VERSION, PROMPT_VERSION,
             access, 0, "success", now, "chunked_reduce"),
        )
        conn.execute(
            "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
            "markdown_hash, schema_id, schema_version, prompt_version, access_class, raw_note_json, "
            "note_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (note_id, run_id, wid, md.markdown_id, md.markdown_hash, SCHEMA_ID, SCHEMA_VERSION,
             PROMPT_VERSION, access, "{}", "x", now),
        )
        return run_id, note_id

    # rollback path: nothing persists.
    h, wid, md = _setup_three_section("d7rollback", doi="10.4b/d7")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            run_id, note_id = _seed(conn, wid, md)
            sid = ensure_span(
                conn, cache_conn, None, markdown_id=md.markdown_id, work_id=wid,
                exact_quote=S1A, access_class="open_access",
            )
            assert sid is not None
            claim = ExtractedClaim(
                claim_id=new_id("claim"), structured_note_id=note_id, extraction_run_id=run_id,
                work_id=wid, claim_type="result", field_key="main_results[0]", status="found",
                epistemic_type="llm_extracted", access_class="open_access", created_at=_iso(),
                source_chunk_index=0, source_extraction_run_id=run_id,
            )
            session.add(claim)
            session.flush()
            conn.execute(
                "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, ?, ?)",
                (claim.claim_id, sid, 0, _iso()),
            )
            rolled_claim = claim.claim_id
            session.rollback()
    finally:
        cache_conn.close()
    check = _dbconn(h)
    try:
        assert check.execute("SELECT COUNT(*) FROM evidence_spans WHERE span_id=?", (sid,)).fetchone()[0] == 0
        assert check.execute(
            "SELECT COUNT(*) FROM extracted_claims WHERE claim_id=?", (rolled_claim,)
        ).fetchone()[0] == 0
        assert check.execute("SELECT COUNT(*) FROM claim_spans WHERE span_id=?", (sid,)).fetchone()[0] == 0
    finally:
        check.close()

    # commit path through the real runner: span + claim + claim_spans all persist.
    h2, wid2, _md2 = _setup_three_section("d7commit", doi="10.4b/d7c")
    res = _run_chunked(h2, wid2, backend=_three_chunk_backend(), caps=_three_caps())
    assert res.status == "success"
    check = _dbconn(h2)
    try:
        # Each persisted claim_spans points at a real evidence_span (no orphans).
        orphans = check.execute(
            "SELECT COUNT(*) FROM claim_spans cs LEFT JOIN evidence_spans es "
            "ON es.span_id = cs.span_id WHERE es.span_id IS NULL"
        ).fetchone()[0]
        assert orphans == 0
        assert check.execute("SELECT COUNT(*) FROM evidence_spans").fetchone()[0] >= 6
    finally:
        check.close()


def test_d2_every_claim_sets_both_provenance_fields():
    h, wid, _md = _setup_three_section("d2", doi="10.4b/d2")
    res = _run_chunked(h, wid, backend=_three_chunk_backend(), caps=_three_caps())
    assert res.status == "success"
    conn = _dbconn(h)
    try:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(extracted_claims)")}
        assert "stated_or_inferred" not in cols  # D2 naming, never the old conflated field
        assert {"epistemic_type", "assertion_status"} <= cols
        rows = conn.execute("SELECT epistemic_type, assertion_status FROM extracted_claims").fetchall()
        assert rows
        for r in rows:
            assert r["epistemic_type"] in ("llm_extracted", "llm_inferred")  # all phase-4b claims are llm_*
        # The inferred result keeps llm_inferred + inferred.
        inferred = conn.execute(
            "SELECT COUNT(*) FROM extracted_claims "
            "WHERE epistemic_type='llm_inferred' AND assertion_status='inferred'"
        ).fetchone()[0]
        assert inferred >= 1
    finally:
        conn.close()


def test_span_anchored_against_full_document():
    """A quote unique doc-wide anchors with correct start_char/section_id even when
    surfaced from a later chunk; a doc-wide-non-unique quote -> ambiguous, no span."""
    # Markdown where a sentence physically in section ONE is quoted by the chunk that
    # reads section THREE; and a duplicated sentence appears in two sections.
    dup = "This duplicated sentence appears twice across the document body."
    md_text = (
        _section("Sec One Alpha", S1A, dup)
        + _section("Sec Two Beta", S2A)
        + _section("Sec Three Gamma", S3A, dup)
    )
    h, wid, md = _make_doc("fulldoc", md_text, doi="10.4b/fd")
    _build_sections(h, wid, md.markdown_id)
    # chunk reading section THREE claims a quote that lives in section ONE (S1A) and
    # also claims the non-unique duplicated sentence.
    note_for_three = {
        "research_question": _env("cross-doc anchor", S1A, label="rq"),
        "setting": _env("ambiguous", dup, label="dup"),
    }
    backend = MappingBackend({S3A: note_for_three, S1A: {}, S2A: {}})
    res = _run_chunked(h, wid, backend=backend, caps=_three_caps())
    assert res.status == "success"

    cache_conn = cache_access.open_cache_ro(None)
    full = cache_access.read_markdown(cache_conn, None, md.markdown_id).text
    cache_conn.close()
    conn = _dbconn(h)
    try:
        # The cross-doc quote anchored at its true full-document offset (in section one).
        span = conn.execute(
            "SELECT start_char, end_char, section_id FROM evidence_spans WHERE exact_quote=?", (S1A,)
        ).fetchone()
        assert span is not None
        assert full[span["start_char"]:span["end_char"]] == S1A
        assert span["start_char"] == full.index(S1A)
        # The section_id resolves to the section actually containing S1A (section one).
        sec = conn.execute(
            "SELECT section_id FROM document_sections WHERE start_char <= ? AND ? < end_char "
            "ORDER BY level DESC LIMIT 1",
            (span["start_char"], span["start_char"]),
        ).fetchone()
        assert span["section_id"] == sec["section_id"]
        # The doc-wide-non-unique quote did NOT anchor: claim downgraded ambiguous,
        # no fabricated span.
        dup_claim = conn.execute(
            "SELECT status FROM extracted_claims WHERE field_key='setting'"
        ).fetchone()
        assert dup_claim["status"] == "ambiguous"
        assert conn.execute(
            "SELECT COUNT(*) FROM evidence_spans WHERE exact_quote=?", (dup,)
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_partial_chunk_failure_reduce_proceeds():
    # Chunk two (marker beta / S2A) is NOT in the mapping -> unrepairable -> that map
    # run extraction_failed; the reduce proceeds over chunks 0 and 2.
    h, wid, _md = _setup_three_section("partial", doi="10.4b/pf")
    backend = MappingBackend({S1A: _note0(), S3A: _note2()})  # S2A absent -> default invalid
    res = _run_chunked(h, wid, backend=backend, caps=_three_caps())
    assert res.status == "success"
    assert res.chunks_failed == 1 and res.chunks_succeeded == 2
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM extraction_runs "
            "WHERE extraction_mode='chunked_map' AND run_status='extraction_failed'"
        ).fetchone()[0] == 1
    finally:
        conn.close()

    # ALL chunks fail -> no note, reduce run extraction_failed.
    h2, wid2, _md2 = _setup_three_section("allfail", doi="10.4b/af")
    res2 = _run_chunked(h2, wid2, backend=MappingBackend({}), caps=_three_caps())
    assert res2.status == "extraction_failed"
    conn2 = _dbconn(h2)
    try:
        assert conn2.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        assert conn2.execute(
            "SELECT run_status FROM extraction_runs WHERE extraction_mode='chunked_reduce'"
        ).fetchone()["run_status"] == "extraction_failed"
    finally:
        conn2.close()


def test_per_chunk_context_gate():
    # A markdown with two small sections + one giant single-paragraph section that
    # overflows the window. The giant chunk skips (skipped_context); the work
    # continues over the small chunks and a note is written.
    giant = "overflow " * 400  # one paragraph ~ 800 tokens > window
    md_text = (
        _section("Small One Alpha", S1A)
        + "# Giant Two\n\n" + giant.strip() + "\n\n"
        + _section("Small Three Gamma", S3A)
    )
    h, wid, md = _make_doc("ctxgate", md_text, doi="10.4b/cg")
    _build_sections(h, wid, md.markdown_id)
    backend = MappingBackend({S1A: _note0(), S3A: _note2()})
    res = _run_chunked(h, wid, backend=backend, caps=_caps(1250, max_output=0))
    assert res.status == "success"
    conn = _dbconn(h)
    try:
        # The overflowing chunk is recorded skipped_context; the work still produced
        # a note from the other chunks.
        assert conn.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE run_status='skipped_context'"
        ).fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
    finally:
        conn.close()

    # Every chunk overflows -> skipped_oversize_section, no note.
    big_md = (
        "# A\n\n" + ("aaaa " * 200).strip() + "\n\n"
        + "# B\n\n" + ("bbbb " * 200).strip() + "\n\n"
    )
    h2, wid2, md2 = _make_doc("allover", big_md, doi="10.4b/ao")
    _build_sections(h2, wid2, md2.markdown_id)
    res2 = _run_chunked(h2, wid2, backend=MappingBackend({}), caps=_caps(600, max_output=0))
    assert res2.status == "skipped_oversize_section"
    conn2 = _dbconn(h2)
    try:
        assert conn2.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
    finally:
        conn2.close()


def test_content_gate_blocks_external_dispatch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    priv_md = THREE_SECTION_MD.replace("Section One", "Confidential One")
    h, wid, md = _make_doc(
        "contentgate", priv_md, access_class=AccessClass.user_supplied_private, doi="10.4b/cg2"
    )
    _build_sections(h, wid, md.markdown_id)
    profiles = {"anthropic_api_default": default_profiles()["anthropic_api_default"]}
    config = make_config(
        preferred="anthropic_api_default", fallback=None, private_external=False, profiles=profiles
    )
    backend = _three_chunk_backend()
    res = _run_chunked(
        h, wid, backend=backend, caps=_caps(1250, model="claude-sonnet-4-6", provider="anthropic",
                                            max_output=0, pricing="snapshot", inp=3.0, out=15.0),
        config=config,
    )
    assert res.status == "skipped_policy"
    assert backend.calls == []  # NO chunk text dispatched externally
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        run = conn.execute(
            "SELECT * FROM extraction_runs WHERE extraction_mode='chunked_reduce'"
        ).fetchone()
        assert run["run_status"] == "skipped_policy"
        assert run["external_full_text"] == 0  # nothing left the machine
    finally:
        conn.close()


def test_budget_across_chunks_and_works(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    h = service.create_project("budget4b")
    caps = _caps(1250, model="claude-sonnet-4-6", provider="anthropic", max_output=0,
                 pricing="snapshot", inp=3.0, out=15.0)
    profiles = {"anthropic_api_default": default_profiles()["anthropic_api_default"]}
    budget_cfg = LlmBudget(monthly_soft_limit_usd=0.0005, stop_on_budget_exceeded=True)
    config = make_config(preferred="anthropic_api_default", fallback=None,
                         private_external=True, budget=budget_cfg, profiles=profiles)
    wids = []
    for i in range(2):
        body = THREE_SECTION_MD.replace("Section One", f"Paper {i} One")
        wid, md = _add_doc(h, f"budget4b_{i}", body, access_class=AccessClass.open_access, doi=f"10.4b/b{i}")
        _build_sections(h, wid, md.markdown_id)
        wids.append(wid)

    from seedgraph.extraction.runner import BudgetState

    budget = BudgetState()
    statuses = []
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            for wid in wids:
                backend = _three_chunk_backend()
                res = extract_note_chunked(
                    session, cache_conn, work_id=wid, backend=backend, config=config,
                    capabilities=caps, cache_root=None, confirm_external=True, budget_state=budget,
                )
                statuses.append(res.status)
    finally:
        cache_conn.close()

    # The corpus halts mid-stream once the cumulative limit is crossed; the second
    # work is recorded skipped_budget (no note).
    assert budget.stopped is True
    assert budget.spent_usd >= 0.0005
    assert statuses[1] == "skipped_budget"
    conn = _dbconn(h)
    try:
        assert conn.execute(
            "SELECT run_status FROM extraction_runs "
            "WHERE extraction_mode='chunked_reduce' AND work_id=?", (wids[1],)
        ).fetchone()["run_status"] == "skipped_budget"
    finally:
        conn.close()

    # --dry-run projects the total and writes/dispatches NOTHING.
    h2, wid2, md2 = _make_doc("dryrun4b", THREE_SECTION_MD, doi="10.4b/dry")
    _build_sections(h2, wid2, md2.markdown_id)
    backend2 = _three_chunk_backend()
    res2 = _run_chunked(h2, wid2, backend=backend2, caps=_three_caps(), dry_run=True)
    assert res2.status == "dry_run"
    assert res2.chunk_count == 3
    assert backend2.calls == []
    conn2 = _dbconn(h2)
    try:
        assert conn2.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == 0
        assert conn2.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
    finally:
        conn2.close()


def test_no_llm_records_single_skipped_reduce_run():
    h, wid, md = _setup_three_section("nollm4b", doi="10.4b/nollm")
    config = make_config(preferred="no_llm")
    backend = _three_chunk_backend()
    res = _run_chunked(h, wid, backend=backend, caps=_three_caps(), config=config)
    assert res.status == "skipped_no_llm"
    assert backend.calls == []
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
        rows = conn.execute("SELECT run_status, extraction_mode FROM extraction_runs").fetchall()
        assert len(rows) == 1
        assert rows[0]["run_status"] == "skipped_no_llm"
        assert rows[0]["extraction_mode"] == "chunked_reduce"
    finally:
        conn.close()


# ==========================================================================
# cross-cutting decisions (plan §11)
# ==========================================================================


def _affinity(type_str: str) -> str:
    t = type_str.upper()
    if "INT" in t:
        return "INTEGER"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "TEXT"


def test_d6_orm_mirrors_migrated_chunk_columns(tmp_path):
    """D6: ORM metadata for ExtractionRun/ExtractedClaim equals the migrated schema
    (the phase-4b chunk-provenance columns present); applies migrations THROUGH 0009
    so the ExtractionRun ORM's now-mirrored lens_definition_hash (added at 0009) is
    covered; create_all is not the authoring path."""
    from sqlalchemy.dialects import sqlite as sqlite_dialect

    db = tmp_path / "parity4b.db"
    conn = sqlite3.connect(str(db))
    # Apply migrations THROUGH 0009 (phase_4 birth-DDL + phase_4b chunk ALTERs +
    # phase_6 lens_definition_hash ALTER, all mirrored on the ExtractionRun ORM).
    for version, path in discover_migrations("project"):
        if version > 9:
            continue
        conn.executescript(path.read_text(encoding="utf-8"))
    conn.close()

    insp = inspect(create_engine(f"sqlite:///{db.as_posix()}"))
    orm_dialect = sqlite_dialect.dialect()
    for table_name, model in (("extraction_runs", ExtractionRun), ("extracted_claims", ExtractedClaim)):
        orm_cols = {c.name: c for c in model.__table__.columns}
        mig_cols = {c["name"]: c for c in insp.get_columns(table_name)}
        assert set(orm_cols) == set(mig_cols), f"{table_name}: column set drift"
        mig_pk = set(insp.get_pk_constraint(table_name)["constrained_columns"])
        orm_pk = {c.name for c in model.__table__.columns if c.primary_key}
        assert mig_pk == orm_pk
        for name, orm_col in orm_cols.items():
            mig_col = mig_cols[name]
            orm_aff = _affinity(str(orm_col.type.compile(dialect=orm_dialect)))
            assert orm_aff == _affinity(str(mig_col["type"])), f"{table_name}.{name}: type drift"
            if name not in mig_pk:
                assert orm_col.nullable == mig_col["nullable"], f"{table_name}.{name}: nullability drift"

    # The chunk-provenance columns are actually present (not just set-equal to a stale
    # ORM) and nullable as authored.
    run_cols = {c["name"]: c for c in insp.get_columns("extraction_runs")}
    assert run_cols["extraction_mode"]["nullable"] is False
    for n in ("parent_extraction_run_id", "chunk_index", "chunk_count", "chunk_section_ids"):
        assert run_cols[n]["nullable"] is True
    claim_cols = {c["name"]: c for c in insp.get_columns("extracted_claims")}
    assert claim_cols["source_chunk_index"]["nullable"] is True
    assert claim_cols["source_extraction_run_id"]["nullable"] is True

    # The migration set the ORM mirrors includes the phase-4b version (0008), and a
    # migrated project reports an applied version that covers it (doctor parity).
    assert latest_version("project") >= 8
    assert any(v == 8 for v, _ in discover_migrations("project"))
    h = service.create_project("d6_4b")
    pconn = sqlite3.connect(str(h.db_path))
    try:
        applied = pconn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        assert applied >= 8
    finally:
        pconn.close()
    r = cli.invoke(app, ["doctor", "--project", "d6_4b"])
    assert r.exit_code == 0, r.output


def test_d5_manifest_section_is_disjoint_and_atomic():
    """D5: phase-4b writes only the chunked_extraction section; a prior note_extraction
    section persists untouched (disjoint); a second phase-4b run replaces atomically;
    writing note_extraction from this stage RAISES (contract violation)."""
    from seedgraph.errors import SeedgraphError
    from seedgraph.run import ensure_run, update_manifest

    service.create_project("manifest4b")
    run_a = ensure_run("manifest4b")
    # phase_4 owns 'note_extraction'; phase_4b owns the disjoint 'chunked_extraction'.
    update_manifest("manifest4b", run_a, {"note_extraction": {"works": 3, "notes": 3}})
    update_manifest("manifest4b", run_a, {"chunked_extraction": {"works": 1, "chunks_total": 4}})

    manifest_path = (
        Path(os.environ["SEEDGRAPH_HOME"]) / "projects" / "manifest4b" / "runs" / run_a / "manifest.json"
    )
    sections = json.loads(manifest_path.read_text())["sections"]
    assert sections["note_extraction"] == {"works": 3, "notes": 3}  # untouched (disjoint)
    assert sections["chunked_extraction"] == {"works": 1, "chunks_total": 4}

    # A second write to an existing section (from this stage) RAISES (D5 contract).
    with pytest.raises(SeedgraphError):
        update_manifest("manifest4b", run_a, {"note_extraction": {"works": 9}})
    with pytest.raises(SeedgraphError):
        update_manifest("manifest4b", run_a, {"chunked_extraction": {"works": 9}})

    # A second phase-4b RUN atomically replaces the section (a fresh manifest, no
    # accumulation).
    run_b = ensure_run("manifest4b")
    update_manifest("manifest4b", run_b, {"chunked_extraction": {"works": 2, "chunks_total": 7}})
    sections_b = json.loads(
        (Path(os.environ["SEEDGRAPH_HOME"]) / "projects" / "manifest4b" / "runs" / run_b / "manifest.json")
        .read_text()
    )["sections"]
    assert sections_b["chunked_extraction"] == {"works": 2, "chunks_total": 7}
    assert "note_extraction" not in sections_b


def test_staleness_and_idempotency_and_resume():
    h, wid, md = _setup_three_section("stale4b", doi="10.4b/stale")
    caps = _three_caps()
    res1 = _run_chunked(h, wid, backend=_three_chunk_backend(), caps=caps)
    assert res1.status == "success"
    conn = _dbconn(h)
    try:
        runs_after_first = conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0]
        notes_after_first = conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0]
    finally:
        conn.close()
    assert notes_after_first == 1

    # Re-run WITHOUT --force -> idempotent skip (no new runs, no new note).
    res2 = _run_chunked(h, wid, backend=_three_chunk_backend(), caps=caps)
    assert res2.status == "skipped"
    conn = _dbconn(h)
    try:
        assert conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == runs_after_first
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
    finally:
        conn.close()

    # A FORCED re-extract re-plans deterministically over the SAME inputs (the plan's
    # persisted shadow — chunk_section_ids — is stable) and recomputes the reduce over a
    # freshly RE-DISPATCHED map set. NOTE: --force is a full re-EXTRACTION, NOT the
    # optional §7 "MAY" map-run reuse (skip re-dispatch) — that cost optimization is
    # deferred; genuine resume-to-completion of an interrupted run is covered by
    # test_resume_completes_partial_extraction_recomputing_reduce.
    res3 = _run_chunked(h, wid, backend=_three_chunk_backend(), caps=caps, force=True)
    assert res3.status == "success"
    conn = _dbconn(h)
    try:
        plans = conn.execute(
            "SELECT chunk_section_ids FROM extraction_runs "
            "WHERE extraction_mode='chunked_map' ORDER BY created_at, chunk_index"
        ).fetchall()
        # Both runs produced identical chunk plans (deterministic re-plan).
        first = [p["chunk_section_ids"] for p in plans[:3]]
        second = [p["chunk_section_ids"] for p in plans[3:6]]
        assert first == second
    finally:
        conn.close()

    # A CHANGED markdown_hash invalidates the current note -> re-plan + re-extract.
    new_md_text = THREE_SECTION_MD.replace("Section One Alpha", "Section One Alpha Revised")
    _wid_same, md2 = _add_doc(h, "stale4b_v2", new_md_text, doi="10.4b/stale")  # same doi -> same work
    assert md2.markdown_hash != md.markdown_hash
    _build_sections(h, wid, md2.markdown_id)
    res4 = _run_chunked(h, wid, backend=_three_chunk_backend(), caps=caps)
    assert res4.status == "success"
    conn = _dbconn(h)
    try:
        # A new current note exists for the new markdown_hash.
        cur = conn.execute(
            "SELECT COUNT(*) FROM structured_notes WHERE markdown_hash=?", (md2.markdown_hash,)
        ).fetchone()[0]
        assert cur == 1
    finally:
        conn.close()


def test_resume_completes_partial_extraction_recomputing_reduce():
    """§7/§11 resume: an INTERRUPTED prior run (committed ``chunked_map`` success runs
    but no note/reduce — the process died before the reduce txn) is COMPLETED on a
    no-force re-run, producing exactly ONE current note whose reduce recomputes over
    the CURRENT map set (merged found-claims back-link via ``source_extraction_run_id``
    to the freshly-dispatched map runs bound to THIS reduce, never the orphaned prior
    runs). The no-force path skips only when a current NOTE exists; an interrupted
    paper (orphan map runs, no note) is resumed to completion, not skipped.

    Cost-saving reuse of the prior map runs *instead of re-dispatch* is the optional
    §7 "MAY" optimization and is deferred (per-chunk drafts are not persisted, so the
    reduce always re-derives them) — this test asserts the binding outcome (one merged
    note + reduce recomputed from the current map set), not the optional optimization.
    """
    h, wid, md = _setup_three_section("resume4b", doi="10.4b/resume")
    caps = _three_caps()

    # Simulate an interrupted prior run: two committed chunked_map 'success' rows for
    # the CURRENT markdown_hash, with NO structured_notes / chunked_reduce row.
    orphan_ids = [new_id("extr"), new_id("extr")]
    conn = _dbconn(h)
    try:
        for ci, rid in enumerate(orphan_ids):
            conn.execute(
                "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, "
                "markdown_hash, schema_id, schema_version, prompt_version, access_class, "
                "external_full_text, run_status, created_at, extraction_mode, chunk_index, "
                "chunk_count, chunk_section_ids) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rid, wid, md.markdown_id, md.markdown_hash, SCHEMA_ID, SCHEMA_VERSION,
                 PROMPT_VERSION, "open_access", 0, "success", _iso(), "chunked_map", ci, 3,
                 json.dumps([f"prior_sec_{ci}"])),
            )
        conn.commit()
        # Precondition: no current note (the prior run never reached the reduce).
        assert conn.execute(
            "SELECT COUNT(*) FROM structured_notes WHERE work_id=?", (wid,)
        ).fetchone()[0] == 0
    finally:
        conn.close()

    # No current note exists -> a no-force run RESUMES (completes) the paper.
    res = _run_chunked(h, wid, backend=_three_chunk_backend(), caps=caps)
    assert res.status == "success"

    conn = _dbconn(h)
    try:
        # Exactly ONE current note for the work (the resume produced it).
        assert conn.execute(
            "SELECT COUNT(*) FROM structured_notes WHERE work_id=?", (wid,)
        ).fetchone()[0] == 1
        note = conn.execute(
            "SELECT note_id, extraction_run_id FROM structured_notes WHERE work_id=?", (wid,)
        ).fetchone()
        reduce_run = note["extraction_run_id"]
        # The reduce recomputed over the CURRENT map set: the map runs bound to THIS
        # reduce are the freshly-dispatched ones, disjoint from the orphaned prior runs.
        current_maps = {
            r["extraction_run_id"]
            for r in conn.execute(
                "SELECT extraction_run_id FROM extraction_runs WHERE extraction_mode="
                "'chunked_map' AND parent_extraction_run_id=?", (reduce_run,)
            ).fetchall()
        }
        assert current_maps and current_maps.isdisjoint(set(orphan_ids))
        # Merged found-claims back-link only into the current map set (no claim points
        # at an orphaned prior run).
        srcs = {
            r["source_extraction_run_id"]
            for r in conn.execute(
                "SELECT source_extraction_run_id FROM extracted_claims WHERE structured_note_id=?",
                (note["note_id"],),
            ).fetchall()
            if r["source_extraction_run_id"] is not None
        }
        assert srcs, "merged found-claims must back-link to their map runs"
        assert srcs <= current_maps
    finally:
        conn.close()


def test_fk_claim_spans_dangling_span_raises():
    h, wid, md = _setup_three_section("fk4b", doi="10.4b/fk")
    conn = sqlite3.connect(str(h.db_path))
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        run_id = new_id("extr")
        note_id = new_id("note")
        now = _iso()
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, "
            "schema_id, schema_version, prompt_version, access_class, external_full_text, run_status, "
            "created_at, extraction_mode) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, wid, md.markdown_id, md.markdown_hash, SCHEMA_ID, SCHEMA_VERSION, PROMPT_VERSION,
             "open_access", 0, "success", now, "chunked_reduce"),
        )
        conn.execute(
            "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
            "markdown_hash, schema_id, schema_version, prompt_version, access_class, raw_note_json, "
            "note_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (note_id, run_id, wid, md.markdown_id, md.markdown_hash, SCHEMA_ID, SCHEMA_VERSION,
             PROMPT_VERSION, "open_access", "{}", "x", now),
        )
        claim_id = new_id("claim")
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, work_id, "
            "claim_type, field_key, status, epistemic_type, access_class, created_at) "
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


def test_fts_exact_phrase_searchable_post_merge():
    h, wid, _md = _setup_three_section("fts4b", doi="10.4b/fts")
    # "parallel trends" is surfaced from chunk 3 (S3A assumption); it must be exact-
    # phrase searchable in claim_fts AND note_fts AFTER the merge.
    res = _run_chunked(h, wid, backend=_three_chunk_backend(), caps=_three_caps())
    assert res.status == "success"
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
        # No stemming collapse: singular "trend" must not match verbatim "trends".
        assert conn.execute(
            'SELECT COUNT(*) FROM claim_fts WHERE claim_fts MATCH ?', ("trend",)
        ).fetchone()[0] == 0
    finally:
        conn.close()


# ==========================================================================
# milestone (plan §11, doc-10 §8, D4-narrowed) — end-to-end via the CLI, offline
# ==========================================================================


def _big_markdown(*, sentences: list[str], sections: int = 4, pad_chars: int = 9000) -> str:
    """An oversize (> llama3 8192-token window) sectioned markdown with unique quotes."""
    blocks = []
    pad = (FILLER * ((pad_chars // len(FILLER)) + 1))[:pad_chars]
    for i in range(sections):
        body = sentences[i] if i < len(sentences) else f"Section body number {i} sentinel."
        blocks.append(f"# Big Section {i}\n\n{body}\n\n{pad}\n\n")
    return "".join(blocks)


def test_milestone_oversize_paper_end_to_end(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    # Canned per-chunk note quoting doc-wide-unique sentences embedded in the paper.
    milestone_note = {
        "research_question": _env("Does X cause Y?", S1A, label="research question"),
        "main_contribution": _env("A new estimator.", S2A, label="contribution"),
        "assumptions": [
            _env("Parallel trends.", S3A, label="parallel trends", assumption_type="identification")
        ],
        "main_results": [
            _env("Effect is three percent.", S3B, label="effect", result_type="point_estimate",
                 assertion="inferred", inferred_explanation="from headline figure")
        ],
    }
    monkeypatch.setattr(chunked_mod, "_BACKEND_OVERRIDE", MappingBackend({}, default=milestone_note))

    h = service.create_project("mile4b")
    # Pin the project's note_extraction route to the local 8192-window model so the
    # CLI (which loads project routing) sizes chunks against the same window phase_4
    # used to mark the paper oversize.
    (Path(os.environ["SEEDGRAPH_HOME"]) / "config.yaml").write_text(
        "llm:\n"
        "  routes:\n"
        "    note_extraction:\n"
        "      task_type: note_extraction\n"
        "      preferred_profile: local_ollama_default\n"
        "      fallback_profile: no_llm\n"
        "      requires_source_text: true\n",
        encoding="utf-8",
    )
    big_md = _big_markdown(sentences=[S1A, S2A, S3A + " " + S3B, "tail sentinel."])
    assert estimate_tokens(big_md) > 8192  # genuinely oversize for the bundled llama3 window
    wid_a, md_a = _add_doc(h, "mile4b_a", big_md, doi="10.4b/mileA")
    _build_sections(h, wid_a, md_a.markdown_id)

    # A residual paper: ONE giant single-paragraph section larger than the window.
    # The heading shares the paragraph (single newline, NO blank line) so the whole
    # section is one irreducible over-budget unit — nothing in it fits the window.
    giant_para = ("residualword " * 4000).strip()  # one paragraph, > 8192 tokens
    residual_md = "# Residual Giant\n" + giant_para + "\n"
    wid_b, md_b = _add_doc(h, "mile4b_b", residual_md, doi="10.4b/mileB")
    _build_sections(h, wid_b, md_b.markdown_id)

    # Precondition: phase 4 records BOTH papers skipped_oversize (bundled llama3 window).
    config = make_config(preferred="local_ollama_default")
    cache_conn = cache_access.open_cache_ro(None)
    try:
        with Session(h.engine) as session:
            for wid in (wid_a, wid_b):
                pre = extract_note(
                    session, cache_conn, work_id=wid, backend=_three_chunk_backend(),
                    config=config, cache_root=None,
                )
                assert pre.run_status == "skipped_oversize", pre.run_status
    finally:
        cache_conn.close()

    # Run phase_4b through the CLI over the skipped_oversize worklist.
    r = cli.invoke(app, ["extract", "notes-chunked", "mile4b"])
    assert r.exit_code in (0, 1), r.output  # 1 = the residual skipped_oversize_section work

    conn = _dbconn(h)
    try:
        # Work A now has EXACTLY ONE current structured note.
        notes_a = conn.execute(
            "SELECT * FROM structured_notes WHERE work_id=?", (wid_a,)
        ).fetchall()
        assert len(notes_a) == 1
        note_a = notes_a[0]

        # Every schema field is a merged claim with a valid status.
        statuses = {
            r2[0] for r2 in conn.execute(
                "SELECT DISTINCT status FROM extracted_claims WHERE work_id=?", (wid_a,)
            )
        }
        assert statuses <= {"found", "not_found", "ambiguous", "not_applicable"}
        # Every found substantive claim has >=1 evidence span resolved against the doc.
        for (cid,) in conn.execute(
            "SELECT claim_id FROM extracted_claims WHERE work_id=? AND status='found'", (wid_a,)
        ).fetchall():
            assert conn.execute(
                "SELECT COUNT(*) FROM claim_spans cs JOIN evidence_spans es ON es.span_id=cs.span_id "
                "WHERE cs.claim_id=?", (cid,)
            ).fetchone()[0] >= 1
        # The map runs enumerate the chunk plan; the reduce run is bound 1:1.
        maps = conn.execute(
            "SELECT * FROM extraction_runs WHERE work_id=? AND extraction_mode='chunked_map'", (wid_a,)
        ).fetchall()
        assert len(maps) >= 2
        assert {m["chunk_count"] for m in maps} == {len(maps)}
        reduce = conn.execute(
            "SELECT * FROM extraction_runs WHERE work_id=? AND extraction_mode='chunked_reduce'", (wid_a,)
        ).fetchone()
        assert note_a["extraction_run_id"] == reduce["extraction_run_id"]
        # Created spans carry the source access_class; claims carry D2 provenance.
        assert {r2[0] for r2 in conn.execute(
            "SELECT DISTINCT access_class FROM evidence_spans WHERE work_id=?", (wid_a,)
        )} == {"open_access"}
        for r2 in conn.execute(
            "SELECT epistemic_type, assertion_status FROM extracted_claims WHERE work_id=? AND status='found'",
            (wid_a,),
        ):
            assert r2["epistemic_type"] in ("llm_extracted", "llm_inferred")

        # The residual paper is recorded skipped_oversize_section, NO note (never dropped).
        assert conn.execute(
            "SELECT COUNT(*) FROM structured_notes WHERE work_id=?", (wid_b,)
        ).fetchone()[0] == 0
        residual_run = conn.execute(
            "SELECT run_status FROM extraction_runs WHERE work_id=? AND extraction_mode='chunked_reduce'",
            (wid_b,),
        ).fetchone()
        assert residual_run["run_status"] == "skipped_oversize_section"
    finally:
        conn.close()

    # The manifest's chunked_extraction section is present and disjoint from note_extraction.
    runs_dir = Path(os.environ["SEEDGRAPH_HOME"]) / "projects" / "mile4b" / "runs"
    manifests = list(runs_dir.glob("*/manifest.json"))
    assert manifests
    found_chunked = False
    for mpath in manifests:
        sections = json.loads(mpath.read_text()).get("sections", {})
        if "chunked_extraction" in sections:
            found_chunked = True
            assert "note_extraction" not in sections  # disjoint — phase_4b never writes it
    assert found_chunked

    # Re-run without --force is an idempotent skip (no new runs for work A).
    conn = _dbconn(h)
    try:
        runs_a_before = conn.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE work_id=?", (wid_a,)
        ).fetchone()[0]
    finally:
        conn.close()
    r2 = cli.invoke(app, ["extract", "notes-chunked", "mile4b"])
    assert r2.exit_code in (0, 1), r2.output
    conn = _dbconn(h)
    try:
        runs_a_after = conn.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE work_id=?", (wid_a,)
        ).fetchone()[0]
        assert runs_a_after == runs_a_before  # idempotent — no new runs for the noted work
        assert conn.execute(
            "SELECT COUNT(*) FROM structured_notes WHERE work_id=?", (wid_a,)
        ).fetchone()[0] == 1
    finally:
        conn.close()
