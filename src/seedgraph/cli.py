"""Typer CLI entry point (``seedgraph-v3`` / ``python -m seedgraph``).

The v3 core commands (``setup`` incl. ``setup agents``, ``migrate``, ``convert``,
``library``, ``work``, ``mcp serve``) live in :mod:`seedgraph.commands` and are
mounted by one ``commands.register(app)`` call. The old v2 groups below stay
registered until the stage that replaces each one deletes it (v3 design §9, §10);
stage 2 deleted ``serve``, ``extract``, ``agent-extract``, ``ask``, ``answer``,
``eval``, ``llm``, ``secrets``, the v2 ``mcp`` group and the lens runner commands."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import typer

from . import __version__, commands, doctor as doctor_mod
from .cache.cli import cache_app
from .db.connection import connect_project_raw
from .golden.cli import golden_app
from .graph.run_view import build_and_export
from .project.cli import project_app, review_app
from .run import latest_run_id

# Windows consoles default to cp1252; CLI output carries typographic dashes,
# ellipses, and other non-cp1252 glyphs (citation quotes, concept labels), which
# made `typer.echo` die with UnicodeEncodeError mid-listing (e.g. `ask`). Force
# UTF-8 at import so every entry path (console script + `python -m`) is safe;
# errors="replace" is the floor if a stream can't be reconfigured.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # captured/replaced streams (pytest) — leave as-is
        pass

app = typer.Typer(
    name="seedgraph",
    help="Seedgraph v3 — a local-first paper library, context graphs and agent work batches.",
    no_args_is_help=True,
    add_completion=False,
)

_root_option = typer.Option(None, "--root", help="Override the seedgraph home directory.")
_project_option = typer.Option(None, "--project", help="Project slug (^[a-z0-9._-]+$).")


@app.command()
def version() -> None:
    """Print package + SQLite versions."""
    typer.echo(f"seedgraph {__version__}")
    typer.echo(f"sqlite {sqlite3.sqlite_version}")


def _span_doctor_checks(root: Optional[Path], slug: str) -> list:
    """phase_3 span/section/FTS reconcile as ``doctor.CheckResult`` rows (additive)."""
    from sqlmodel import Session

    from . import cache_access, doctor_spans
    from .db.adapter import raw_conn
    from .project.service import open_project

    try:
        h = open_project(slug, root=root)
    except Exception as exc:  # noqa: BLE001
        return [doctor_mod.CheckResult("span_section_fts_reconcile", False, f"cannot open project: {exc}")]

    cache_conn = cache_access.open_cache_ro(root)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            report = doctor_spans.run(conn, cache_conn, root)
    finally:
        cache_conn.close()

    detail = (
        f"{report.spans_total} span(s); stale={report.spans_stale} "
        f"orphaned={report.spans_orphaned} missing={report.spans_missing} "
        f"dangling_section_refs={report.dangling_section_refs} "
        f"invariant_failures={report.invariant_failures}"
    )
    return [
        doctor_mod.CheckResult("span_fts5_available", report.fts5_available,
                               "span_fts FTS5 present" if report.fts5_available
                               else "FTS5 NOT compiled — span search unavailable"),
        doctor_mod.CheckResult("span_section_fts_reconcile", report.ok, detail),
    ]


@app.command()
def doctor(
    root: Optional[Path] = _root_option,
    project: Optional[str] = _project_option,
    probe_ollama: bool = typer.Option(
        False, "--probe-ollama", help="Also check that the local Ollama server answers (no model call)."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON object instead of the text render."
    ),
) -> None:
    """Validate the foundation; exit non-zero on any failure.

    Extended (phase_5b): when ``--project`` is given, also runs the cross-DB
    ``work_source_files`` bridge reconcile + a cross-scope foreign_key_check
    (decision r2-10) without editing phase_0's ``doctor.py``. Lists which
    scholarly-API environment variables are set (S2_API_KEY, OPENALEX_API_KEY,
    CORE_API_KEY, SEEDGRAPH_CONTACT_EMAIL; names only) and, with ``--probe-ollama``,
    checks the local Ollama server.
    """
    results = doctor_mod.collect_checks(root, project, probe_ollama=probe_ollama)
    if project is not None:
        from .acquisition.doctor_reconcile import cross_db_bridge_reconcile

        try:
            results = results + cross_db_bridge_reconcile(root, project)
        except Exception as exc:  # noqa: BLE001 - a reconcile error is a check failure
            results = results + [
                doctor_mod.CheckResult(
                    "cross_db_bridge_reconcile", False, f"reconcile failed: {exc}"
                )
            ]
        # phase_3: real span/section/FTS reconcile (replaces the no-op stub of the
        # same name emitted by collect_checks; additive, does not edit phase_0 doctor).
        results = [r for r in results if r.name != "span_section_fts_reconcile"]
        results = results + _span_doctor_checks(root, project)
        # Build A ch6: soft work-ref + identifier-normalization-drift scan
        # (same no-edit-to-doctor.py registration pattern).
        from .project.doctor_reconcile import work_refs_reconcile

        try:
            results = results + work_refs_reconcile(root, project)
        except Exception as exc:  # noqa: BLE001 - a reconcile error is a check failure
            results = results + [
                doctor_mod.CheckResult(
                    "work_refs_reconcile", False, f"reconcile failed: {exc}"
                )
            ]
    # ``--json`` serializes the SAME final results list (Build F ch6): text render
    # and exit-code semantics unchanged either way.
    typer.echo(doctor_mod.render_json(results) if as_json else doctor_mod.render(results))
    if doctor_mod.has_failure(results):
        raise typer.Exit(code=1)


# ===========================================================================
# phase_5b: corpus acquisition + resolution walk (real grouped sub-app).
# ===========================================================================

corpus_app = typer.Typer(
    name="corpus",
    help="Acquisition + resolution walk (sole network phase).",
    no_args_is_help=True,
)


@corpus_app.command("resolve")
def corpus_resolve(
    slug: str = typer.Argument(..., help="Project slug."),
    root: Optional[Path] = _root_option,
) -> None:
    """Resolve every work's metadata; fill identifiers; route mid-band to review."""
    import asyncio

    from .acquisition.resolve import resolve_corpus
    from .acquisition.service import corpus_io
    from .project.service import open_project
    from .run import ensure_run

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    chain, _client, _backend = corpus_io(root, run_id)
    report = asyncio.run(resolve_corpus(h, chain, run_id=run_id))
    typer.echo(
        f"resolved={report.resolved} ambiguous={report.ambiguous} "
        f"unresolved={report.unresolved} (run {run_id})"
    )


@corpus_app.command("walk")
def corpus_walk(
    slug: str = typer.Argument(..., help="Project slug."),
    depth: int = typer.Option(2, "--depth"),
    per_gen_cap: int = typer.Option(50, "--per-gen-cap"),
    root: Optional[Path] = _root_option,
) -> None:
    """Outbound referenced_works walk; grow metadata_only corpus; cache ref lists."""
    import asyncio

    from .acquisition.service import corpus_io
    from .acquisition.walk import walk_corpus
    from .progress import Progress
    from .project.service import open_project
    from .run import ensure_run

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    chain, _client, _backend = corpus_io(root, run_id)
    # Stdout-only per-source heartbeat (emit=None => no non-terminal events.jsonl
    # write that could strand _status_from_events on 'running' — D-7). Total 0 is
    # a placeholder: the walk grows it per generation as frontiers are known.
    prog = Progress(0, "works")
    report = asyncio.run(
        walk_corpus(h, chain, run_id=run_id, depth=depth, per_gen_cap=per_gen_cap, progress=prog)
    )
    typer.echo(f"discovered={report.discovered} frontier_sizes={report.frontier_sizes} (run {run_id})")


@corpus_app.command("acquire")
def corpus_acquire(
    slug: str = typer.Argument(..., help="Project slug."),
    promote: bool = typer.Option(
        True, "--promote/--no-promote",
        help="Promote a fetched+converted metadata_only work to included (default on).",
    ),
    max_papers: Optional[int] = typer.Option(
        None, "--max-papers",
        help="Attempt at most N papers; the remainder is counted skipped_budget (0 = attempt nothing).",
    ),
    budget_seconds: Optional[float] = typer.Option(
        None, "--budget-seconds",
        help="Wall-clock budget for the pass; on expiry the remainder is counted skipped_budget (0 = attempt nothing).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Fetch OA PDFs for target works; ingest+convert; write the single bridge row."""
    from .acquisition.service import acquire_corpus, corpus_io
    from .progress import Progress
    from .project.service import open_project
    from .run import ensure_run

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    chain, client, backend = corpus_io(root, run_id)
    # Stdout-only per-paper heartbeat (emit=None — D-7); the acquire pass
    # reconciles the placeholder total once it selects its targets.
    prog = Progress(0, "works")
    report = acquire_corpus(
        h, chain, run_id=run_id, promote=promote, cache_root=root,
        http_client=client, backend=backend,
        max_papers=max_papers, budget_seconds=budget_seconds,
        progress=prog,
    )
    typer.echo(
        f"acquired={report.acquired} already_cached={report.already_cached} "
        f"requires_upload={report.requires_upload} skipped_budget={report.skipped_budget} "
        f"(run {run_id})"
    )


@corpus_app.command("upload")
def corpus_upload(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: str = typer.Argument(..., help="work_ id."),
    pdf_path: Path = typer.Argument(..., help="Path to the user-supplied PDF."),
    promote: bool = typer.Option(
        True, "--promote/--no-promote",
        help="Promote a provided metadata_only work to included (default on).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Lawful manual upload: ingest (user_supplied_private) -> convert -> bridge."""
    from .acquisition.service import manual_upload
    from .project.service import open_project

    h = open_project(slug, root=root)
    row = manual_upload(h, work_id=work_id, pdf_path=pdf_path, cache_root=root, promote=promote)
    typer.echo(f"{work_id}\t{row.source_file_id}\t{row.markdown_id or ''}")


@corpus_app.command("import-markdown")
def corpus_import_markdown(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: str = typer.Option(..., "--work", help="work_ id to bridge."),
    pdf_path: Path = typer.Option(..., "--pdf", help="Source PDF (ingested for provenance)."),
    markdown_path: Path = typer.Option(
        ..., "--markdown", help="Externally-converted markdown file to import."
    ),
    promote: bool = typer.Option(
        True, "--promote/--no-promote",
        help="Promote a provided metadata_only work to included (default on).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """GPU-free import: bridge a work to EXTERNALLY-converted markdown (no Marker)."""
    from sqlmodel import Session

    from .acquisition.service import import_markdown_and_bridge
    from .db.project_models import ProjectDocument
    from .project.service import open_project

    h = open_project(slug, root=root)
    row = import_markdown_and_bridge(
        h, work_id=work_id, pdf_path=pdf_path, markdown_path=markdown_path,
        cache_root=root, promote=promote,
    )
    with Session(h.engine) as s:
        doc = s.get(ProjectDocument, work_id)
        promoted = doc is not None and doc.inclusion_status == "included"
    typer.echo(f"{work_id}\t{row.markdown_id or ''}\tpromoted={promoted}")


@corpus_app.command("ingest-folder")
def corpus_ingest_folder(
    slug: str = typer.Argument(..., help="Project slug."),
    folder: Path = typer.Argument(..., help="Folder of PDFs to bulk-ingest + auto-match."),
    promote: bool = typer.Option(
        True, "--promote/--no-promote",
        help="Flip a matched metadata_only work to included (inclusion_reason='provided_pdf').",
    ),
    promote_unmatched: bool = typer.Option(
        False, "--promote-unmatched",
        help="Create a new included work for an unmatched PDF instead of routing to review.",
    ),
    recursive: bool = typer.Option(False, "--recursive", help="Recurse into subfolders."),
    root: Optional[Path] = _root_option,
) -> None:
    """Bulk-ingest a folder of PDFs; auto-match each to a pending work (DOI→arXiv→title),
    OCR-fallback convert, and bridge; route misses to review (or --promote-unmatched)."""
    from .acquisition.service import ingest_folder
    from .project.service import open_project

    h = open_project(slug, root=root)
    report = ingest_folder(
        h, folder, cache_root=root, promote=promote,
        promote_unmatched=promote_unmatched, recursive=recursive,
    )
    for row in report.per_pdf:
        typer.echo(
            f"{row['filename']}\t{row['outcome']}\t{row.get('work_id') or ''}\t"
            f"{row.get('matched_by') or ''}\t{'ocr' if row.get('ocr_used') else ''}"
        )
    typer.echo(
        f"matched_doi={report.matched_doi} matched_arxiv={report.matched_arxiv} "
        f"matched_title={report.matched_title} promoted={report.promoted} "
        f"unmatched_review={report.unmatched_review} ambiguous_review={report.ambiguous_review} "
        f"already_bridged={report.already_bridged} conversion_failed={report.conversion_failed} "
        f"ocr_recovered={report.ocr_recovered} skipped_non_pdf={report.skipped_non_pdf}"
    )


@corpus_app.command("run")
def corpus_run(
    slug: str = typer.Argument(..., help="Project slug."),
    depth: int = typer.Option(2, "--depth"),
    per_gen_cap: int = typer.Option(50, "--per-gen-cap"),
    promote: bool = typer.Option(True, "--promote/--no-promote"),
    max_papers: Optional[int] = typer.Option(
        None, "--max-papers",
        help="Attempt at most N papers in the acquire pass; the remainder is counted skipped_budget (0 = attempt nothing).",
    ),
    budget_seconds: Optional[float] = typer.Option(
        None, "--budget-seconds",
        help="Wall-clock budget for the acquire pass; on expiry the remainder is counted skipped_budget (0 = attempt nothing).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Print the closed-form acquisition budget preview (papers / API calls / disk) and exit 0; ZERO network — the provider chain is never constructed.",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Umbrella: ensure_run -> resolve -> walk -> acquire under ONE run_id (D5)."""
    from .acquisition.service import corpus_io, run_corpus
    from .progress import Progress
    from .project.service import open_project

    h = open_project(slug, root=root)
    if dry_run:
        # Build D ch13 (D-13): closed-form no-network budget preview — print and
        # exit 0 BEFORE constructing the chain (no ensure_run, no providers, no
        # network). Seeds = the same included-works set walk_corpus expands from
        # as generation 0; --dry-run rides `corpus run` ONLY (the umbrella verb —
        # a second flag on `corpus walk` was cut as over-engineering).
        import sqlite3

        from .acquisition.preview import preview_acquisition

        conn = sqlite3.connect(str(h.db_path))
        try:
            seed_count = conn.execute(
                "SELECT COUNT(*) FROM works w "
                "JOIN project_documents d ON d.work_id = w.work_id "
                "WHERE d.inclusion_status = 'included'"
            ).fetchone()[0]
        finally:
            conn.close()
        pv = preview_acquisition(seed_count, depth, per_gen_cap)
        typer.echo(
            f"expected_papers={pv['expected_papers']} "
            f"(seeds={pv['seed_count']} + cap={pv['cap']} x depth={pv['depth']})"
        )
        typer.echo(f"api_calls=~{pv['api_calls']}")
        typer.echo(f"est_disk=~{pv['est_disk_mb']} MB")
        return
    chain, client, backend = corpus_io(root, None)
    # Stdout-only per-item heartbeats for both stages (emit=None — D-7): a CLI
    # run never appends a non-terminal 'progress' event to events.jsonl, so
    # _status_from_events can never be stranded on 'running' by this verb.
    walk_prog = Progress(0, "works")
    acquire_prog = Progress(0, "works")
    out = run_corpus(
        h, chain, depth=depth, per_gen_cap=per_gen_cap, root=root,
        http_client=client, backend=backend, promote=promote,
        max_papers=max_papers, budget_seconds=budget_seconds,
        walk_progress=walk_prog, acquire_progress=acquire_prog,
    )
    typer.echo(
        f"run {out['run_id']}: resolved={out['resolution'].resolved} "
        f"walked={out['walk'].discovered} acquired={out['acquisition'].acquired} "
        f"skipped_budget={out['acquisition'].skipped_budget}"
    )


@corpus_app.command("status")
def corpus_status(
    slug: str = typer.Argument(..., help="Project slug."),
    root: Optional[Path] = _root_option,
) -> None:
    """Table: work_id | title | inclusion_status | access_status | acquisition_state | has_markdown."""
    from .acquisition.service import corpus_rows
    from .project.service import open_project

    h = open_project(slug, root=root)
    typer.echo("work_id\ttitle\tinclusion_status\taccess_status\tacquisition_state\thas_markdown")
    # Shared read-model (lifted to acquisition.service.corpus_rows so the corpus UI
    # screen and this CLI table render the same projection).
    for r in corpus_rows(h, cache_root=root):
        typer.echo(
            f"{r['work_id']}\t{r['title'] or ''}\t{r['inclusion_status']}\t"
            f"{r['access_status'] or ''}\t{r['acquisition_state']}\t{int(r['has_markdown'])}"
        )


@corpus_app.command("identify")
def corpus_identify(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: Optional[List[str]] = typer.Option(
        None, "--work-id", help="Backfill only these work ids (repeatable; default: all with markdown)."
    ),
    force: bool = typer.Option(False, "--force", help="Re-run even if a work already has a non-filename title."),
    root: Optional[Path] = _root_option,
) -> None:
    """Deterministic post-conversion metadata backfill: attach doi/arxiv from each
    included work's converted markdown (the LLM title path moved to `work create metadata`)."""
    from .extraction.metadata import backfill_work_metadata
    from .project.service import open_project

    h = open_project(slug, root=root)
    work_ids = list(work_id) if work_id else _included_works_with_markdown(h)
    if not work_ids:
        typer.echo("no works with markdown to identify")
        return
    for wid in work_ids:
        # One work's failure must not abort the whole batch (batch resilience).
        try:
            res = backfill_work_metadata(h, wid, cache_root=root, force=force)
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            typer.echo(f"{wid}\terror\t{type(exc).__name__}: {exc}")
            continue
        ids = ", ".join(f"{t}:{v}" for t, v in res.ids_attached) or "-"
        collided = ", ".join(f"{t}:{v}" for t, v in res.ids_collided)
        line = (
            f"{wid}\t{res.status}\t{(res.old_title or '')!r} -> {(res.new_title or '')!r}"
            f"\tids={ids}"
        )
        if collided:
            line += f"\tid_collision(review)={collided}"
        typer.echo(line)


@corpus_app.command("backfill-titles")
def corpus_backfill_titles(
    slug: str = typer.Argument(..., help="Project slug."),
    root: Optional[Path] = _root_option,
) -> None:
    """Post-walk title read-repair (§5.10, Build D ch9): batch-fill NULL/empty
    titles for works that hold an OpenAlex W-id, via the batched by_openalex_ids
    namespace (a stale negative by_doi cache entry can never starve it). NO-LLM;
    idempotent — a re-run reports queried=0."""
    import asyncio

    from .acquisition.backfill import backfill_titles
    from .acquisition.service import corpus_io
    from .project.service import open_project
    from .run import ensure_run

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    chain, _client, _backend = corpus_io(root, run_id)
    result = asyncio.run(backfill_titles(h, chain))
    typer.echo(
        f"queried={result.queried} fetched={result.fetched} "
        f"filled={result.filled} still_missing={result.still_missing} (run {run_id})"
    )


# ===========================================================================
# phase_2: citation graph — OFFLINE provider-edge projection (real sub-app).
# Marker-independent, no full text, no LLM, NO NETWORK. `project`/`run` mint the
# run_id up front (must-fix #3); `build`/`table` are read-only and never mint edges.
# ===========================================================================

cite_app = typer.Typer(
    name="cite",
    help="Citation graph: OFFLINE provider-edge projection + NetworkX build/export.",
    no_args_is_help=True,
)


def _work_lookup(conn: sqlite3.Connection) -> tuple[dict[str, str], dict[str, str]]:
    """``(work_id -> title, work_id -> inclusion_status)`` for the table view."""
    titles: dict[str, str] = {}
    status: dict[str, str] = {}
    cursor = conn.execute(
        "SELECT w.work_id, w.canonical_title, d.inclusion_status "
        "FROM works w LEFT JOIN project_documents d ON d.work_id = w.work_id"
    )
    for work_id, title, inclusion_status in cursor.fetchall():
        titles[work_id] = title or ""
        status[work_id] = inclusion_status
    return titles, status


@cite_app.command("project")
def cite_project(
    slug: str = typer.Argument(..., help="Project slug."),
    root: Optional[Path] = _root_option,
) -> None:
    """Project provider_reference edges OFFLINE from provider_cache (no network).

    Mints a run_id at the start (must-fix #3), then for each existing work reads its
    cached referenced_works list and writes an edge to each target that ALREADY
    exists as a works row. No walk, no upsert, no frontier (phase_5b owns growth).
    """
    from . import cache_access
    from .citation.project_edges import project_provider_edges
    from .project.service import open_project
    from .run import ensure_run

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    conn = connect_project_raw(h.db_path)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        result = project_provider_edges(conn, cache_conn, run_id=run_id)
    finally:
        conn.close()
        cache_conn.close()
    typer.echo(
        f"run {run_id}: edges={result['edges']} "
        f"unresolved_targets={len(result['unresolved_targets'])}"
    )


@cite_app.command("build")
def cite_build(
    slug: str = typer.Argument(..., help="Project slug."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run to build (default: latest)."),
    open_world: bool = typer.Option(False, "--open-world", help="Keep metadata_only stub targets."),
    root: Optional[Path] = _root_option,
) -> None:
    """Assemble the NetworkX graph from a run's authoritative edges; export artifacts."""
    from .project.service import open_project
    from .run import ensure_run

    h = open_project(slug, root=root)
    if run_id is None:
        run_id = latest_run_id(slug, root=h.root)
        if run_id is None:
            typer.echo("no runs found; run `seedgraph cite project` or `cite run` first")
            raise typer.Exit(2)
    ensure_run(slug, run_id=run_id, root=h.root)  # idempotent; ensures the run dir
    graph, out_dir = build_and_export(slug, root, h.root, run_id, open_world=open_world)
    typer.echo(
        f"run {run_id}: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
        f"-> {out_dir / 'graph.json'}"
    )


@cite_app.command("table")
def cite_table(
    slug: str = typer.Argument(..., help="Project slug."),
    show_all: bool = typer.Option(False, "--all", help="Include metadata_only stub targets."),
    root: Optional[Path] = _root_option,
) -> None:
    """Print the closed-world included->included citation matrix (default).

    ``--all`` additionally surfaces edges to metadata_only stub targets.
    """
    from .citation.edges import authoritative_edges
    from .project.service import open_project

    h = open_project(slug, root=root)
    run_id = latest_run_id(slug, root=h.root)
    typer.echo("source_work_id\ttarget_work_id\tsource_title\ttarget_title")
    if run_id is None:
        return
    conn = connect_project_raw(h.db_path)
    try:
        edges = authoritative_edges(conn, run_id)
        titles, status = _work_lookup(conn)
    finally:
        conn.close()
    for edge in edges:
        source = edge["source_work_id"]
        target = edge["target_work_id"]
        if status.get(source) != "included":
            continue
        if not show_all and status.get(target) != "included":
            continue
        typer.echo(f"{source}\t{target}\t{titles.get(source, '')}\t{titles.get(target, '')}")


@cite_app.command("run")
def cite_run(
    slug: str = typer.Argument(..., help="Project slug."),
    open_world: bool = typer.Option(False, "--open-world", help="Keep metadata_only stub targets."),
    root: Optional[Path] = _root_option,
) -> None:
    """Umbrella: mint ONE run_id, then project (offline) -> build under it."""
    from . import cache_access
    from .citation.project_edges import project_provider_edges
    from .project.service import open_project
    from .run import ensure_run

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    conn = connect_project_raw(h.db_path)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        result = project_provider_edges(conn, cache_conn, run_id=run_id)
    finally:
        conn.close()
        cache_conn.close()
    graph, out_dir = build_and_export(slug, root, h.root, run_id, open_world=open_world)
    typer.echo(
        f"run {run_id}: edges={result['edges']} "
        f"unresolved_targets={len(result['unresolved_targets'])} "
        f"nodes={graph.number_of_nodes()} graph={out_dir / 'graph.json'}"
    )


# ===========================================================================
# phase_3b: parsed-bibliography pipeline — parse each included work's references
# section into resolved, provenance-tagged parsed_bibliography edges (two-stage
# lifecycle; §7). NO LLM, offline-resilient (providers ride provider_cache).
# ===========================================================================


def _included_works_with_markdown(h) -> list[str]:
    """Included works that have a bridged markdown (the default cite-parse loop)."""
    from sqlmodel import Session, select

    from .acquisition.bridge import resolve_work_markdown
    from .db.project_models import ProjectDocument, Work

    out: list[str] = []
    with Session(h.engine) as session:
        works = session.exec(
            select(Work)
            .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
            .where(ProjectDocument.inclusion_status == "included")
            .order_by(Work.created_at)
        ).all()
        for work in works:
            if resolve_work_markdown(session, work_id=work.work_id) is not None:
                out.append(work.work_id)
    return out


@cite_app.command("parse")
def cite_parse(
    slug: str = typer.Argument(..., help="Project slug."),
    work: Optional[str] = typer.Option(None, "--work", help="Single work id (default: all included with markdown)."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Target run (default: latest; minted if none)."),
    force: bool = typer.Option(False, "--force", help="Re-parse even when markdown_hash is unchanged."),
    root: Optional[Path] = _root_option,
) -> None:
    """Parse + resolve references into parsed_bibliography edges (Stage A + Stage B).

    Resolves the target ``run_id`` via the run-manager (``--run-id``, default = latest
    run; mints one if none exists). For the named ``--work`` or every included work
    with a bridged markdown, runs Stage A (parse/resolve, skipped on unchanged
    ``markdown_hash`` unless ``--force``) and Stage B (project current resolved refs
    into the run; ALWAYS runs). Routes ambiguous/suspect entries to ``review_queue``
    and merges per-work counts into ``manifest.json``.
    """
    from .acquisition.service import corpus_io
    from .citation.parsed_bib import build_parsed_edges
    from .errors import SeedgraphError
    from .progress import Progress
    from .project import identity as identity_mod
    from .project.service import open_project
    from .run import ensure_run, update_manifest

    h = open_project(slug, root=root)
    if run_id is None:
        run_id = latest_run_id(slug, root=h.root)
    run_id = ensure_run(slug, run_id=run_id, root=h.root)

    chain, _client, _backend = corpus_io(root, run_id)
    work_ids = [work] if work is not None else _included_works_with_markdown(h)

    totals = {
        "run_id": run_id, "works": 0, "entries": 0, "resolved": 0,
        "ambiguous": 0, "suspect": 0, "edges": 0, "reparsed": 0, "skipped": 0,
    }
    # Stdout-only N/M heartbeat (emit=None => no non-terminal events.jsonl write).
    prog = Progress(len(work_ids), "works")
    for wid in work_ids:
        res = build_parsed_edges(
            h, root, chain, identity_mod, work_id=wid, run_id=run_id, force=force
        )
        totals["works"] += 1
        totals["entries"] += res.entries
        totals["resolved"] += res.resolved
        totals["ambiguous"] += res.ambiguous
        totals["suspect"] += res.suspect
        totals["edges"] += res.edges_written
        totals["reparsed"] += 1 if res.reparsed else 0
        totals["skipped"] += 1 if res.skipped_reason else 0
        state = res.skipped_reason or ("reparsed" if res.reparsed else "current")
        prog.step(
            f"{wid} ({state})",
            entries=res.entries, resolved=res.resolved,
            ambiguous=res.ambiguous, suspect=res.suspect,
            edges=totals["edges"],
        )

    try:
        update_manifest(slug, run_id, {"parsed_bibliography": totals}, root=h.root)
    except SeedgraphError:
        # The section is owned by this stage; a re-run of cite parse under the same
        # run rewrites disjointly only if absent — tolerate an existing section.
        pass
    typer.echo(
        f"run {run_id}: works={totals['works']} entries={totals['entries']} "
        f"resolved={totals['resolved']} ambiguous={totals['ambiguous']} "
        f"suspect={totals['suspect']} edges={totals['edges']} "
        f"reparsed={totals['reparsed']} skipped={totals['skipped']}"
    )


@cite_app.command("disagreements")
def cite_disagreements(
    slug: str = typer.Argument(..., help="Project slug."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run to inspect (default: latest)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Print (source, target) pairs under the run carrying 2+ distinct provenances.

    The confirm / overlap surface (decision 26): each pair with its provenance set
    and the authoritative winner. Single-provenance adds-coverage edges are NOT
    listed here -- they appear in ``cite table``.
    """
    from .citation.edges import edge_disagreements
    from .project.service import open_project

    h = open_project(slug, root=root)
    if run_id is None:
        run_id = latest_run_id(slug, root=h.root)
    typer.echo("source_work_id\ttarget_work_id\tprovenances\tauthoritative_provenance")
    if run_id is None:
        return
    conn = connect_project_raw(h.db_path)
    try:
        rows = edge_disagreements(conn, run_id)
    finally:
        conn.close()
    for row in rows:
        typer.echo(
            f"{row['source']}\t{row['target']}\t{','.join(row['provenances'])}\t"
            f"{row['authoritative_provenance']}"
        )


# ===========================================================================
# phase_3: evidence spans — sectioning, span index/create/get/verify/reanchor,
# and exact-term FTS search (real grouped sub-apps). Fully deterministic: cache.db
# is opened READ-ONLY; span/section writes share the caller's ORM transaction via
# the D7 raw_conn bridge (the '--work' path resolves to markdown_id through the
# phase_5b bridge, decision NEW-B).
# ===========================================================================

sections_app = typer.Typer(
    name="sections",
    help="Document sectioning for evidence spans.",
    no_args_is_help=True,
)
spans_app = typer.Typer(
    name="spans",
    help="Evidence spans: index/create/get/verify/reanchor.",
    no_args_is_help=True,
)


def _resolve_md_for_cli(session, *, work_id: Optional[str], markdown_id: Optional[str]):
    """Resolve ``(work_id, markdown_id)`` for a phase_3 command.

    ``--work`` resolves to the current markdown via the phase_5b bridge
    (``resolve_work_markdown``); ``--markdown-id`` reverse-resolves its owning work
    from the same bridge. Returns ``None`` when nothing resolves.
    """
    from sqlmodel import select

    from .acquisition.bridge import resolve_work_markdown
    from .db.project_models import WorkSourceFile

    if work_id is not None:
        resolved = resolve_work_markdown(session, work_id=work_id)
        if resolved is None:
            return None
        return work_id, resolved[0]
    row = session.exec(
        select(WorkSourceFile).where(WorkSourceFile.markdown_id == markdown_id)
    ).first()
    if row is None or row.markdown_id is None:
        return None
    return row.work_id, markdown_id


@sections_app.command("build")
def sections_build(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: Optional[str] = typer.Option(None, "--work", help="Work id (resolves markdown via bridge)."),
    markdown_id: Optional[str] = typer.Option(None, "--markdown-id", help="Explicit markdown id."),
    root: Optional[Path] = _root_option,
) -> None:
    """Parse + store ``document_sections`` for a doc (idempotent deterministic ids)."""
    from sqlmodel import Session

    from . import cache_access
    from .db.adapter import raw_conn
    from .project.service import open_project
    from .sections.parser import parse_sections
    from .sections.store import replace_sections

    if (work_id is None) == (markdown_id is None):
        typer.echo("provide exactly one of --work or --markdown-id")
        raise typer.Exit(2)

    h = open_project(slug, root=root)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            resolved = _resolve_md_for_cli(session, work_id=work_id, markdown_id=markdown_id)
            if resolved is None:
                typer.echo("no markdown resolved for the given --work/--markdown-id")
                raise typer.Exit(2)
            wid, mid = resolved
            md = cache_access.read_markdown(cache_conn, root, mid)
            if md is None:
                typer.echo(f"markdown {mid} unresolvable")
                raise typer.Exit(2)
            sections = parse_sections(
                md.text,
                markdown_id=mid,
                markdown_hash=md.markdown_hash,
                source_file_id=md.source_file_id,
                source_file_hash=md.source_file_hash,
                work_id=wid,
            )
            n = replace_sections(conn, mid, sections)
            session.commit()
        typer.echo(f"{wid}\t{mid}\tsections={n}")
    finally:
        cache_conn.close()


@spans_app.command("index")
def spans_index(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: str = typer.Option(..., "--work", help="Work id to index."),
    root: Optional[Path] = _root_option,
) -> None:
    """Materialize paragraph spans + reindex span_fts for a work (idempotent)."""
    from sqlmodel import Session

    from . import cache_access
    from .acquisition.bridge import resolve_work_markdown
    from .db.adapter import raw_conn
    from .project.service import open_project
    from .spans.index import index_document

    h = open_project(slug, root=root)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            resolved = resolve_work_markdown(session, work_id=work_id)
            if resolved is None:
                typer.echo(f"no markdown for work {work_id}")
                raise typer.Exit(2)
            mid = resolved[0]
            n = index_document(conn, cache_conn, root, work_id=work_id, markdown_id=mid)
            session.commit()
        typer.echo(f"{work_id}\t{mid}\tspans={n}")
    finally:
        cache_conn.close()


@spans_app.command("create")
def spans_create(
    slug: str = typer.Argument(..., help="Project slug."),
    markdown_id: str = typer.Option(..., "--markdown-id", help="Markdown id to anchor in."),
    start: Optional[int] = typer.Option(None, "--start", help="Start code-point offset."),
    end: Optional[int] = typer.Option(None, "--end", help="End code-point offset."),
    quote: Optional[str] = typer.Option(None, "--quote", help="Exact verbatim quote (unique)."),
    kind: str = typer.Option("manual", "--kind", help="span_kind (default manual)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Create a manual span by unique --quote OR explicit --start/--end offsets."""
    from sqlmodel import Session

    from . import cache_access
    from .db.adapter import raw_conn
    from .project.service import open_project
    from .spans.store import _write_span, ensure_span

    h = open_project(slug, root=root)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            resolved = _resolve_md_for_cli(session, work_id=None, markdown_id=markdown_id)
            if resolved is None:
                typer.echo(f"no work bridged to markdown {markdown_id}")
                raise typer.Exit(2)
            wid, mid = resolved
            if quote is not None:
                span_id = ensure_span(
                    conn, cache_conn, root,
                    markdown_id=mid, work_id=wid, exact_quote=quote, span_kind=kind,
                )
                if span_id is None:
                    typer.echo("quote not found or not unique in markdown")
                    raise typer.Exit(2)
            elif start is not None and end is not None:
                md = cache_access.read_markdown(cache_conn, root, mid)
                if md is None:
                    typer.echo(f"markdown {mid} unresolvable")
                    raise typer.Exit(2)
                span_id = _write_span(
                    conn, cache_conn, root,
                    markdown_id=mid, work_id=wid, start=start, end=end,
                    exact_quote=md.text[start:end], span_kind=kind,
                )
            else:
                typer.echo("provide --quote or both --start and --end")
                raise typer.Exit(2)
            session.commit()
        typer.echo(span_id)
    finally:
        cache_conn.close()


@spans_app.command("get")
def spans_get(
    slug: str = typer.Argument(..., help="Project slug."),
    span_id: str = typer.Argument(..., help="Span id."),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON."),
    root: Optional[Path] = _root_option,
) -> None:
    """Show a span: exact source text, section breadcrumb, work, page, status."""
    import json as _json

    from sqlmodel import Session

    from . import cache_access
    from .db.adapter import raw_conn
    from .project.service import open_project
    from .spans.store import get_span

    h = open_project(slug, root=root)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            span = get_span(conn, cache_conn, root, span_id)
    finally:
        cache_conn.close()

    if json_out:
        typer.echo(_json.dumps({
            "span_id": span.span_id, "work_id": span.work_id, "markdown_id": span.markdown_id,
            "section_id": span.section_id, "heading_path": span.heading_path,
            "exact_quote": span.exact_quote, "resolved_text": span.resolved_text,
            "page_start": span.page_start, "page_end": span.page_end,
            "anchor_status": span.anchor_status, "access_class": span.access_class,
        }))
    else:
        typer.echo(f"span_id: {span.span_id}")
        typer.echo(f"work_id: {span.work_id}")
        typer.echo(f"section: {span.heading_path or '(preamble)'}")
        typer.echo(f"pages: {span.page_start}-{span.page_end}")
        typer.echo(f"status: {span.anchor_status}")
        typer.echo(f"text: {span.exact_quote}")


@spans_app.command("verify")
def spans_verify(
    slug: str = typer.Argument(..., help="Project slug."),
    span_id: Optional[str] = typer.Argument(None, help="Span id (omit with --all)."),
    verify_all: bool = typer.Option(False, "--all", help="Verify every span."),
    root: Optional[Path] = _root_option,
) -> None:
    """Verify markdown[start:end]==exact_quote & quote_hash; lineage-based stale-mark."""
    from sqlmodel import Session

    from . import cache_access
    from .db.adapter import raw_conn
    from .project.service import open_project
    from .spans.store import verify_span

    if (span_id is None) == (not verify_all):
        typer.echo("provide a span_id OR --all")
        raise typer.Exit(2)

    h = open_project(slug, root=root)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            if verify_all:
                ids = [r[0] for r in conn.execute("SELECT span_id FROM evidence_spans").fetchall()]
            else:
                ids = [span_id]
            counts: dict[str, int] = {}
            for sid in ids:
                status = verify_span(conn, cache_conn, root, sid)
                counts[status] = counts.get(status, 0) + 1
            session.commit()
    finally:
        cache_conn.close()
    typer.echo(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no spans")


@spans_app.command("reanchor")
def spans_reanchor(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: str = typer.Option(..., "--work", help="Work id whose stale spans to relocate."),
    root: Optional[Path] = _root_option,
) -> None:
    """Relocate stale spans into current markdown (exact); unique-miss -> review_queue."""
    from sqlmodel import Session

    from . import cache_access
    from .db.adapter import raw_conn
    from .project.service import open_project
    from .spans.store import reanchor_spans

    h = open_project(slug, root=root)
    cache_conn = cache_access.open_cache_ro(root)
    try:
        with Session(h.engine) as session:
            conn = raw_conn(session)
            moved = reanchor_spans(conn, cache_conn, root, work_id=work_id)
            session.commit()
    finally:
        cache_conn.close()
    typer.echo(f"{work_id}\trelocated={moved}")


# ===========================================================================
# phase_6: project lens definitions (new/list/show/validate; runs moved to the
# work protocol in stage 2). Resolves the project from --project / cwd.
# ===========================================================================

lens_app = typer.Typer(
    name="lens",
    help="Project lens definitions: new/list/show/validate (runs: `work create lens:<id>`).",
    no_args_is_help=True,
)

_lens_project_option = typer.Option(
    None, "--project", help="Project slug (default: resolve from the current directory)."
)


def _resolve_lens_project(project: Optional[str], root: Optional[Path]):
    """Open the project for a lens command, resolving the slug from --project or cwd."""
    from .errors import ValidationError
    from .project.service import open_project

    slug = project
    if slug is None:
        # Resolve from cwd: a directory whose name is a project under projects/.
        from . import paths

        cwd_name = Path.cwd().name
        if (paths.project_dir(cwd_name, root) / "project.yaml").exists():
            slug = cwd_name
    if slug is None:
        typer.echo("no project: pass --project SLUG (or run inside a project directory)")
        raise typer.Exit(2)
    try:
        return open_project(slug, root=root)
    except ValidationError as exc:
        typer.echo(f"project error: {exc}")
        raise typer.Exit(2)


def _lens_dir(h) -> Path:
    return h.root / "projects" / h.slug / "lenses"


@lens_app.command("new")
def lens_new(
    lens_id: str = typer.Argument(..., help="New lens id (slug+version, e.g. regularity_conditions_v1)."),
    from_template: str = typer.Option(..., "--from-template", help="Built-in template name to scaffold from."),
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Scaffold projects/{slug}/lenses/{lens_id}.yaml from the built-in template."""
    from sqlmodel import Session

    from .errors import SeedgraphError
    from .lenses import registry

    h = _resolve_lens_project(project, root)
    project_dir = h.root / "projects" / h.slug
    try:
        with Session(h.engine) as session:
            row = registry.create_lens_from_template(
                session, project_dir, lens_id, from_template
            )
    except SeedgraphError as exc:
        typer.echo(str(exc))
        raise typer.Exit(2)
    dest = registry.lens_yaml_path(project_dir, lens_id)
    typer.echo(f"created {dest}\nregistered lens {row.lens_id} status={row.status} hash={row.definition_hash[:12]}")


@lens_app.command("list")
def lens_list(
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """List registered lenses with status + hash."""
    from sqlmodel import Session

    from .lenses import registry

    h = _resolve_lens_project(project, root)
    with Session(h.engine) as session:
        rows = registry.list_lenses(session)
    if not rows:
        typer.echo("no lenses registered")
        return
    typer.echo("lens_id\tstatus\tobject_type\thash")
    for r in rows:
        typer.echo(f"{r.lens_id}\t{r.status}\t{r.object_type}\t{r.definition_hash[:12]}")


@lens_app.command("validate")
def lens_validate(
    lens_id: str = typer.Argument(..., help="Lens id to validate."),
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Validate the on-disk lens YAML; exit non-zero on any validation error."""
    from .lenses import registry

    h = _resolve_lens_project(project, root)
    project_dir = h.root / "projects" / h.slug
    try:
        lens = registry.load_lens(project_dir, lens_id)
    except Exception as exc:  # noqa: BLE001 — any parse/validation error is a CLI failure
        typer.echo(f"INVALID {lens_id}: {exc}")
        raise typer.Exit(1)
    typer.echo(f"OK {lens.lens_id} object_type={lens.object_type} fields={len(lens.output_schema)} hash={lens.definition_hash()[:12]}")


@lens_app.command("show")
def lens_show(
    lens_id: str = typer.Argument(..., help="Lens id to show."),
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Show the resolved lens definition. (Runs and results moved to the work
    protocol's ``lens:<id>`` task in stage 2.)"""
    from sqlmodel import Session

    from .lenses import registry

    h = _resolve_lens_project(project, root)
    project_dir = h.root / "projects" / h.slug
    lens = registry.load_lens(project_dir, lens_id)
    with Session(h.engine) as session:
        row = registry.get_lens_row(session, lens_id)
    status = row.status if row else "unregistered"
    typer.echo(
        f"lens {lens.lens_id} ({lens.name})\n"
        f"  status={status} object_type={lens.object_type} hash={lens.definition_hash()[:12]}\n"
        f"  fields={list(lens.output_schema)}"
    )


# ===========================================================================
# Wiring-stage additive command surface. Phase logic is NOT implemented here;
# every verb below is a stub that prints its owning phase and exits 2.
# ===========================================================================

# ===========================================================================
# phase_7 — semantic concepts + graph export (real grouped sub-apps).
# ===========================================================================
concepts_app = typer.Typer(
    name="concepts",
    help="Semantic concepts: build/list/show (show = the milestone query).",
    no_args_is_help=True,
)
graph_app = typer.Typer(
    name="graph",
    help="Semantic graph export (default-deny access guard).",
    no_args_is_help=True,
)


@concepts_app.command("build")
def concepts_build(
    slug: str = typer.Argument(..., help="Project slug."),
    no_llm: bool = typer.Option(False, "--no-llm", hidden=True,
                                help="Ignored: the build is deterministic since stage 2."),
    tau: float = typer.Option(0.6, "--tau", help="Token-Jaccard clustering threshold."),
    min_shared: int = typer.Option(
        2, "--min-shared",
        help="Min shared works for a co_occurs_with edge (drops within-single-paper cliques).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Extract -> cluster -> guard -> merge (deterministic); write concepts/aliases/
    claim_concepts; build Work--discusses-->Concept + co_occurs_with edges."""
    from .run import ensure_run
    from .semantic import build_semantic_overlay
    from .semantic.export import update_graph_manifest
    from .project.service import open_project

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    conn = connect_project_raw(h.db_path)
    try:
        report = build_semantic_overlay(conn, run_id=run_id, tau=tau, min_shared=min_shared)
    finally:
        conn.close()
    update_graph_manifest(slug=slug, run_id=run_id, report=report, root=h.root)
    typer.echo(
        f"run {run_id}: mode={report.concept_mode} concepts={report.concepts_written} "
        f"aliases={report.aliases_written} claim_concepts={report.claim_concepts_written} "
        f"discusses={report.discusses_edges} co_occurs={report.co_occurs_edges} "
        f"review_items={report.review_items_enqueued}"
    )


@concepts_app.command("list")
def concepts_list(
    slug: str = typer.Argument(..., help="Project slug."),
    type: Optional[str] = typer.Option(None, "--type", help="Filter by concept_type."),
    status: Optional[str] = typer.Option(None, "--status", help="Filter by status."),
    root: Optional[Path] = _root_option,
) -> None:
    """Tabular concept list (concept_id | label | type | papers | weight | status |
    access), sharp-first (weight = IDF discriminativeness, not importance)."""
    from .project.service import open_project
    from .semantic.query import list_concepts

    h = open_project(slug, root=root)
    conn = connect_project_raw(h.db_path)
    try:
        rows = list_concepts(conn, concept_type=type, status=status)
    finally:
        conn.close()
    typer.echo(
        "concept_id\tcanonical_label\ttype\tpaper_frequency\tweight\tstatus\taccess_class"
    )
    for row in rows:
        typer.echo(
            f"{row['concept_id']}\t{row['canonical_label']}\t{row['concept_type']}\t"
            f"{row['paper_frequency']}\t{row['weight']:.4f}\t{row['status']}\t"
            f"{row['access_class']}"
        )


@concepts_app.command("overview")
def concepts_overview(
    slug: str = typer.Argument(..., help="Project slug."),
    per_type_cap: int = typer.Option(
        12, "--per-type-cap", help="Max featured concepts shown per type."
    ),
    min_pf: int = typer.Option(
        2, "--min-pf", help="Min paper_frequency for a featured (non-background) concept."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the overview dict as JSON instead of the text sheet."
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """ORIENTATION sheet: scale, a background frame of ubiquitous 'assumed context'
    concepts, per-type featured concepts (top by distinct-paper recurrence, generics
    downweighted), and a queryable tail hint. Read-only aggregation over the concepts
    table (never mutates) — the compact alternative to dumping the whole `list`."""
    import json as _json

    from .project.service import open_project
    from .semantic.query import concept_overview

    h = open_project(slug, root=root)
    conn = connect_project_raw(h.db_path)
    try:
        ov = concept_overview(conn, per_type_cap=per_type_cap, min_pf=min_pf)
    finally:
        conn.close()

    if as_json:
        typer.echo(_json.dumps(ov, indent=2))
        return

    sc = ov["scale"]
    typer.echo(
        f"{slug}: {sc['total_concepts']} concepts across {sc['total_papers']} papers "
        f"| {sc['singleton_count']} singletons (pf<=1) | {len(sc['type_counts'])} types"
    )
    # Trust-tier label (CORE_CONCEPT §4): this layer is an LLM-extracted interpretive
    # overlay, not ground truth; recurrence = distinct papers; ground the specifics
    # downstream rather than reading them off the synthesized labels.
    typer.echo(
        "note: LLM-extracted SEMANTIC OVERLAY; pf = distinct papers (recurrence, not "
        "importance); ground specifics via `concepts show` / `search` / spans, not these labels."
    )
    bg = " | ".join(
        f"{c['canonical_label']}·pf{c['paper_frequency']}" for c in ov["background_frame"]
    )
    typer.echo(f"Background frame (assumed context): {bg}")
    for block in ov["by_type"]:
        feats = " | ".join(
            f"{c['canonical_label']}·pf{c['paper_frequency']}" for c in block["featured"]
        )
        more = f" (+{block['more_count']} more)" if block["more_count"] > 0 else ""
        typer.echo(f"{block['concept_type'].upper()} (n={block['type_count']}): {feats}{more}")
    th = ov["tail_hint"]
    typer.echo(
        f"tail: {th['not_shown_count']} concept(s) not shown — drill with "
        f"`{th['commands'][0]}` or `{th['commands'][1]}`."
    )


@concepts_app.command("show")
def concepts_show(
    slug: str = typer.Argument(..., help="Project slug."),
    concept_id: str = typer.Argument(..., help="concept:: id."),
    root: Optional[Path] = _root_option,
) -> None:
    """MILESTONE QUERY: linked papers, claims, and evidence spans for one concept."""
    from .project.service import open_project
    from .semantic.query import concept_detail

    h = open_project(slug, root=root)
    conn = connect_project_raw(h.db_path)
    try:
        detail = concept_detail(conn, concept_id)
    finally:
        conn.close()
    if detail is None:
        typer.echo(f"no concept {concept_id!r}", err=True)
        raise typer.Exit(code=1)
    c = detail["concept"]
    typer.echo(f"concept\t{c['concept_id']}\t{c['canonical_label']}\t{c['concept_type']}\t{c['access_class']}")
    for p in detail["papers"]:
        typer.echo(f"paper\t{p['work_id']}\t{p['title'] or ''}\t{p['year'] or ''}")
    for cl in detail["claims"]:
        typer.echo(f"claim\t{cl['claim_id']}\t{cl['work_id']}\t{cl['claim_type']}\t{cl['claim_text'] or ''}")
    for s in detail["spans"]:
        typer.echo(f"span\t{s['span_id']}\t{s['work_id']}\t{s['exact_quote']}")


@graph_app.command("export")
def graph_export(
    slug: str = typer.Argument(..., help="Project slug."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run id (default: latest)."),
    format: str = typer.Option("json", "--format", help="json|graphml|csv|bibtex|ris."),
    allow_private: bool = typer.Option(False, "--allow-private", help="Include private concepts/edges."),
    root: Optional[Path] = _root_option,
) -> None:
    """Computed view -> runs/{run_id}/ exports, through the default-deny guard."""
    from .project.service import open_project
    from .semantic.export import export_graph

    h = open_project(slug, root=root)
    rid = run_id or latest_run_id(slug, root=h.root)
    if rid is None:
        typer.echo("no run to export (run `concepts build` first)", err=True)
        raise typer.Exit(code=1)
    conn = connect_project_raw(h.db_path)
    try:
        written = export_graph(
            conn, slug=slug, run_id=rid, fmt=format, allow_private=allow_private, root=h.root
        )
    finally:
        conn.close()
    for path in written:
        typer.echo(str(path))


@graph_app.command("analyze")
def graph_analyze(
    slug: str = typer.Argument(..., help="Project slug."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run id (default: latest)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Deterministic citation-graph analysis summary: communities, god nodes,
    bridges, read-next (decision 38's no-LLM floor made visible)."""
    from .graph.analyze import analyze_summary
    from .project.service import open_project

    # Thin renderer over the shared assembly (decision M10): `graph.analyze
    # .analyze_summary` is the one source of truth the MCP `graph_analyze` tool
    # also renders. This body only formats the dict it returns.
    h = open_project(slug, root=root)
    summary = analyze_summary(h, run_id=run_id)

    typer.echo(
        f"run {summary['run_id']}: {summary['node_count']} work(s), "
        f"{summary['edge_count']} citation "
        "edge(s) [citation projection; epistemic_type=deterministic]"
    )
    typer.echo(f"communities: {summary['community_count']}")
    for c in summary["communities"]:
        typer.echo(f"  community {c['community_id']}: {c['size']} work(s)")
    typer.echo(f"god nodes: {len(summary['god_nodes'])}")
    for g in summary["god_nodes"]:
        typer.echo(f"  {g['work_id']}\t{g['title']}")
    typer.echo(
        f"bridges: {summary['bridge_edge_count']} edge(s), "
        f"{summary['bridge_node_count']} node(s)"
    )
    for e in summary["bridges"]:
        mark = "flagged" if e["flagged_is_bridge"] else "derived-only"
        typer.echo(
            f"  {e['source_work_id']} -> {e['target_work_id']} "
            f"[community {e['source_community']} -> {e['target_community']}; {mark}]"
        )
    typer.echo(f"read next (top 10 of {summary['unread_count']} unread):")
    for i, r in enumerate(summary["read_next"][:10], start=1):
        typer.echo(f"  [{i}] {r['work_id']}\t{r['title']}")


# Grouped sub-apps (group -> verbs), mounted additively onto the root app.
app.add_typer(cache_app, name="cache")  # phase_1: real sub-app (NotImplementedError bodies)
app.add_typer(golden_app, name="golden")  # v3 stage 0: golden set + converter-upgrade check
# phase_5: real grouped sub-apps wired to the project.service / project.review core.
app.add_typer(project_app, name="project")
app.add_typer(review_app, name="review")
app.add_typer(corpus_app, name="corpus")  # phase_5b: real grouped sub-app
# phase_2: real grouped sub-app (project|build|table|run). phase_3b adds the
# parsed-bibliography verbs (parse|disagreements) below; build/table pick up the
# parsed tier automatically via authoritative_edges.
app.add_typer(cite_app, name="cite")
app.add_typer(sections_app, name="sections")  # phase_3: real grouped sub-app
app.add_typer(spans_app, name="spans")  # phase_3: real grouped sub-app
app.add_typer(lens_app, name="lens")  # phase_6: real grouped sub-app
app.add_typer(concepts_app, name="concepts")  # phase_7: real grouped sub-app
app.add_typer(graph_app, name="graph")  # phase_7: real grouped sub-app

# v3 core commands: setup, migrate, convert, library (design §10).
commands.register(app)


from .work_read_cli import register_work_read

register_work_read(app)


if __name__ == "__main__":
    app()
