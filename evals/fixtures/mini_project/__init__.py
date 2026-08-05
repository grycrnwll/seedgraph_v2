"""Synthetic ``mini_project`` builder for the phase_9 key-free CI gate (plan §5/§11).

Builds a tiny, fully-offline project.db + cache.db whose rows exercise every MVP
§12 invariant WITHOUT any LLM or network: dedup (distinct file_hash), markdown
present + re-sliceable, window-fitting note coverage WITH a recorded
``skipped_oversize`` paper (decision D4), included-paper citation edges + a
referenced-but-absent ``unresolved_target`` diagnostic (decision D3), a substantive
claim with a real evidence span + a claim with an explicit ``not_found`` record,
metadata-resolution + reference-extraction + concept-merge audit subjects, and a
``user_supplied_private`` span that must never escape into a shareable surface.

Built via the real numbered migrations (D6) + raw SQL inserts (mirroring the
phase_8 offline fixture style). Hand-authored — NO restricted full text.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from seedgraph import anchor
from seedgraph.db.bootstrap import ensure_cache_db
from seedgraph.db.connection import cache_db_path, project_db_path
from seedgraph.project import layout, service

SLUG = "mini_eval"

# --- markdown bodies (hand-authored; no restricted text). A known substring of
#     each body is stored as an evidence span so md[start:end] == quote re-verifies.
MD_P1 = (
    "# Paper P1: Difference-in-Differences\n\n"
    "The parallel trends assumption is central to identification in "
    "difference-in-differences designs.\n\n"
    "We discuss negative weights under heterogeneous treatment effects.\n"
)
MD_P2 = (
    "# Paper P2: Event Studies\n\n"
    "The rank condition must hold for point identification in the event-study model.\n\n"
    "Cohort-specific effects are aggregated into an interaction-weighted estimator.\n"
)

# The verbatim span quote drawn from MD_P1 (offsets computed at build time).
QUOTE_P1 = "The parallel trends assumption is central to identification"
QUOTE_P2 = "The rank condition must hold for point identification"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _md_id(text: str) -> str:
    return "md_" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sf_id(text: str) -> str:
    return "sf_" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_cache(root: Path) -> dict:
    """Migrate cache.db; write two markdown docs (one per window-fitting paper)."""
    ensure_cache_db(root)
    cache_path = cache_db_path(root)
    cache_root = cache_path.parent
    md_dir = cache_root / "markdown"
    md_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(cache_path))
    conn.execute("PRAGMA foreign_keys=ON")
    out: dict = {}
    try:
        for tag, body in (("p1", MD_P1), ("p2", MD_P2)):
            file_bytes = f"%PDF-fake-{tag}".encode("utf-8")
            sf_id = _sf_id(tag)
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            md_id = _md_id(body)
            md_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            storage_uri = f"markdown/{md_id}.md"
            # write BYTES (no newline translation) so the on-disk hash == markdown_hash.
            (cache_root / storage_uri).write_bytes(body.encode("utf-8"))

            conn.execute(
                "INSERT INTO source_files (source_file_id, file_hash, file_type, access_class, "
                "storage_uri, acquisition_method, created_at) VALUES (?,?,?,?,?,?,?)",
                (sf_id, file_hash, "pdf", "open_access", f"pdf/{sf_id}.pdf", "upload", _now()),
            )
            conv_id = f"conv_{tag}"
            conn.execute(
                "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, "
                "converter_name, converter_package, converter_version, python_version, config_json, "
                "conversion_fingerprint, conversion_epistemic_type, run_status, markdown_hash, "
                "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    conv_id, sf_id, file_hash, "marker", "marker-pdf", "1.0", "3.13",
                    "{}", f"fp_{tag}", "local_deterministic_conversion", "success", md_hash, _now(),
                ),
            )
            conn.execute(
                "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
                "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (md_id, conv_id, sf_id, md_hash, storage_uri, "success", len(body.encode()), _now()),
            )
            out[tag] = {"sf_id": sf_id, "file_hash": file_hash, "md_id": md_id, "md_hash": md_hash}
        conn.commit()
    finally:
        conn.close()
    out["cache_root"] = str(cache_root)
    return out


def _build_project(root: Path, cache: dict) -> dict:
    """Create + populate project.db with the §12 fixture rows."""
    handle = service.create_project(SLUG, root=root)
    conn = sqlite3.connect(str(handle.db_path))
    conn.execute("PRAGMA foreign_keys=ON")

    p1 = cache["p1"]
    p2 = cache["p2"]

    def add_work(work_id, title, status, year=2021):
        conn.execute(
            "INSERT INTO works (work_id, canonical_title, year, created_at) VALUES (?,?,?,?)",
            (work_id, title, year, _now()),
        )
        conn.execute(
            "INSERT INTO project_documents (work_id, inclusion_status, is_seed, created_at, "
            "updated_at) VALUES (?,?,?,?,?)",
            (work_id, status, 1, _now(), _now()),
        )

    add_work("work_p1", "Difference-in-Differences with staggered adoption", "included")
    add_work("work_p2", "Event-study estimation under heterogeneity", "included")
    add_work("work_p3", "Foundational survey (metadata only)", "metadata_only")
    add_work("work_oversize", "A very long monograph exceeding the context window", "included")

    # identifiers — DOI/arXiv that resolve to the named work (verify-corpus §10(b)).
    conn.execute(
        "INSERT INTO identifiers (work_id, id_type, id_value, resolution_source, confidence) "
        "VALUES ('work_p1','doi','10.0000/p1','user',1.0)"
    )
    conn.execute(
        "INSERT INTO identifiers (work_id, id_type, id_value, resolution_source, confidence) "
        "VALUES ('work_p2','arxiv','2099.00002','user',1.0)"
    )

    # work_source_files bridge (acquired + converted) for the two window-fitting papers.
    conn.execute(
        "INSERT INTO work_source_files (work_id, source_file_id, file_hash, markdown_id, "
        "markdown_hash, acquisition_method, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        ("work_p1", p1["sf_id"], p1["file_hash"], p1["md_id"], p1["md_hash"], "upload", _now(), _now()),
    )
    conn.execute(
        "INSERT INTO work_source_files (work_id, source_file_id, file_hash, markdown_id, "
        "markdown_hash, acquisition_method, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        ("work_p2", p2["sf_id"], p2["file_hash"], p2["md_id"], p2["md_hash"], "upload", _now(), _now()),
    )

    def add_run(rid, work_id, md_id, md_hash, status, access="open_access"):
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, "
            "schema_version, prompt_version, access_class, run_status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (rid, work_id, md_id, md_hash, "v1", "p1", access, status, _now()),
        )

    def add_note(nid, rid, work_id, md_id, md_hash, text):
        conn.execute(
            "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
            "markdown_hash, schema_version, prompt_version, access_class, raw_note_json, "
            "note_text, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (nid, rid, work_id, md_id, md_hash, "v1", "p1", "open_access", "{}", text, _now()),
        )

    # window-fitting papers: success run + a default note each.
    add_run("run_p1", "work_p1", p1["md_id"], p1["md_hash"], "success")
    add_run("run_p2", "work_p2", p2["md_id"], p2["md_hash"], "success")
    add_note("note_p1", "run_p1", "work_p1", p1["md_id"], p1["md_hash"], "P1 default note.")
    add_note("note_p2", "run_p2", "work_p2", p2["md_id"], p2["md_hash"], "P2 default note.")

    # D4 — the oversize paper: a not-processed record, NO note. Never a coverage failure.
    add_run("run_oversize", "work_oversize", "md_oversize", "h_oversize", "skipped_oversize")

    def add_span(span_id, work_id, md_id, md_hash, body, quote, access="open_access"):
        start = body.index(quote)
        end = start + len(quote)
        qh = anchor.quote_hash(quote)  # real NFC quote_hash (doctor invariant audit)
        conn.execute(
            "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
            "source_file_hash, work_id, start_char, end_char, exact_quote, quote_hash, "
            "span_kind, access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (span_id, md_id, md_hash, "sf", "sfh", work_id, start, end, quote, qh,
             "paragraph", access, _now()),
        )
        return {"span_id": span_id, "work_id": work_id, "markdown_id": md_id,
                "start": start, "end": end, "quote": quote}

    span_p1 = add_span("span_p1", "work_p1", p1["md_id"], p1["md_hash"], MD_P1, QUOTE_P1)
    span_p2 = add_span("span_p2", "work_p2", p2["md_id"], p2["md_hash"], MD_P2, QUOTE_P2)
    # a user_supplied_private span (must never reach a shareable surface).
    add_span("span_priv", "work_p1", p1["md_id"], p1["md_hash"], MD_P1,
             "negative weights under heterogeneous treatment effects",
             access="user_supplied_private")

    def add_claim(cid, nid, rid, work_id, ctype, label, text, status="found", access="open_access"):
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
            "work_id, claim_type, field_key, normalized_label, claim_text, status, epistemic_type, "
            "assertion_status, access_class, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (cid, nid, rid, work_id, ctype, "f", label, text, status, "llm_extracted",
             "stated", access, _now()),
        )

    # a substantive claim WITH >=1 evidence span ...
    add_claim("claim_p1", "note_p1", "run_p1", "work_p1", "identification_assumption",
              "parallel trends", "Parallel trends is central to identification.")
    add_claim("claim_p2", "note_p2", "run_p2", "work_p2", "regularity_condition",
              "rank condition", "The rank condition must hold.")
    # ... and a claim with an explicit not_found record (no span).
    add_claim("claim_notfound", "note_p1", "run_p1", "work_p1", "result",
              "missing estimand", None, status="not_found")

    conn.execute(
        "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES ('claim_p1','span_p1',0,?)",
        (_now(),),
    )
    conn.execute(
        "INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES ('claim_p2','span_p2',0,?)",
        (_now(),),
    )

    # included-paper citation edges (provider_reference — shareable, metadata-class).
    for src, tgt in (("work_p1", "work_p2"), ("work_p1", "work_p3"), ("work_p2", "work_p3")):
        conn.execute(
            "INSERT INTO citation_edges (source_work_id, target_work_id, edge_type, provenance, "
            "confidence, run_id, created_at) VALUES (?,?,'cites','provider_reference',1.0,'run_mini',?)",
            (src, tgt, _now()),
        )

    # reference_entries — phase_3b parse/resolution quality. One unresolved target
    # (referenced-but-absent → D3 diagnostic, NEVER materialized as a works row),
    # plus an ambiguous + a suspect entry (reference_extraction audit subjects).
    def add_ref(ref_id, citing, status, resolved=None, text="Some Reference (1999)."):
        conn.execute(
            "INSERT INTO reference_entries (reference_id, citing_work_id, raw_reference_text, "
            "resolved_work_id, resolution_status, created_at) VALUES (?,?,?,?,?,?)",
            (ref_id, citing, text, resolved, status, _now()),
        )

    add_ref("ref_unresolved", "work_p1", "unresolved", None,
            "Ghost et al. (1990), A work absent from the corpus.")
    add_ref("ref_ambiguous", "work_p1", "ambiguous", None, "Smith (2001), ambiguous match.")
    add_ref("ref_suspect", "work_p2", "suspect", None, "Doe (2002), suspect parse.")
    add_ref("ref_resolved", "work_p2", "resolved", "work_p1", "Clean (2003), resolved.")

    # review_queue — metadata-resolution (duplicate_candidate) subjects + a borderline
    # concept-merge routed to review (D10: borderline merges go to review_queue).
    conn.execute(
        "INSERT INTO review_queue (item_id, item_type, target_type, target_id, status, created_at) "
        "VALUES ('rq_dup1','duplicate_candidate','identifier','work_p1','open',?)",
        (_now(),),
    )
    conn.execute(
        "INSERT INTO review_queue (item_id, item_type, target_type, target_id, status, created_at) "
        "VALUES ('rq_merge1','concept_merge_candidate','concept','concept::parallel trends','open',?)",
        (_now(),),
    )

    # concepts + aliases (concept_merge audit subjects). The alias fold is a merge.
    conn.execute(
        "INSERT INTO concepts (concept_id, normalized_label, canonical_label, concept_type, "
        "paper_frequency, status, epistemic_type, access_class, created_at, updated_at) "
        "VALUES ('concept::parallel trends','parallel trends','Parallel Trends','assumption',2,"
        "'auto','deterministic','open_access',?,?)",
        (_now(), _now()),
    )
    conn.execute(
        "INSERT INTO concept_aliases (concept_id, alias_label, fold_reason, epistemic_type, "
        "created_at) VALUES ('concept::parallel trends','common trends','acronym','deterministic',?)",
        (_now(),),
    )

    conn.commit()
    conn.close()

    return {
        "span_p1": span_p1,
        "span_p2": span_p2,
        "window_fitting_work_ids": ["work_p1", "work_p2"],
        "oversize_work_id": "work_oversize",
        "private_span_id": "span_priv",
        "unresolved_ref_id": "ref_unresolved",
        "absent_target_hint": "Ghost et al.",
        "not_found_claim_id": "claim_notfound",
    }


def _write_gold_files(root: Path) -> None:
    """Author the per-project gold + leakage-probe files under projects/{slug}/eval/."""
    eval_dir = layout.project_dir(SLUG, root) / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "retrieval_gold.jsonl").write_text(
        '{"question": "What is the parallel trends assumption?", "query_type": "factual", '
        '"relevant_work_ids": ["work_p1"], "relevant_span_ids": ["span_p1"]}\n'
        '{"question": "What is the rank condition?", "query_type": "factual", '
        '"relevant_work_ids": ["work_p2"], "relevant_span_ids": ["span_p2"]}\n',
        encoding="utf-8",
    )
    (eval_dir / "leakage_probes.jsonl").write_text(
        '{"question": "What is the synthetic control estimator of Abadie et al.?", '
        '"rationale": "Synthetic control is not in the mini corpus; must abstain."}\n'
        '{"question": "Who won the 2021 Nobel Prize in Economics?", '
        '"rationale": "General-knowledge fact absent from corpus full text; must abstain."}\n',
        encoding="utf-8",
    )


def build(root: Path | str) -> dict:
    """Build the synthetic ``mini_project`` into ``root`` (a SEEDGRAPH_HOME). Returns
    a facts dict the acceptance tests assert against. Idempotent only across fresh
    homes — ``create_project`` hard-errors if the project already exists."""
    root = Path(root)
    cache = _build_cache(root)
    facts = _build_project(root, cache)
    _write_gold_files(root)
    facts.update(
        {
            "slug": SLUG,
            "root": str(root),
            "db_path": str(project_db_path(SLUG, root)),
            "cache_root": cache["cache_root"],
        }
    )
    return facts
