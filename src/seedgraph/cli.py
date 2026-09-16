"""Typer CLI.

Phase-0 verbs (``version``, ``migrate``, ``doctor``) are the fully-wired surface.
The wiring stage additively mounts grouped sub-apps and top-level commands for
every later phase as STUB callables (each echoes ``not yet implemented (<phase>)``
then exits 2) so ``seedgraph --help`` lists the full command surface before the
phase logic lands (decision 81 — CLI is the complete surface)."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

import typer

from . import __version__, doctor as doctor_mod
from .cache.cli import cache_app
from .db.bootstrap import ensure_cache_db, ensure_project_db
from .db.connection import cache_db_path, connect_project_raw, project_db_path
from .db.migrations import current_version
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
    help="Seedgraph v2 — local-first citation/knowledge graph substrate.",
    no_args_is_help=True,
    add_completion=False,
)

_root_option = typer.Option(None, "--root", help="Override the seedgraph home directory.")
_project_option = typer.Option(None, "--project", help="Project slug (^[a-z0-9._-]+$).")


def _scope_version(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return current_version(conn)
    finally:
        conn.close()


@app.command()
def version() -> None:
    """Print package + SQLite versions."""
    typer.echo(f"seedgraph {__version__}")
    typer.echo(f"sqlite {sqlite3.sqlite_version}")


@app.command()
def migrate(
    root: Optional[Path] = _root_option,
    project: Optional[str] = _project_option,
) -> None:
    """Apply pending numbered .sql migrations for the named scope(s). Idempotent."""
    ensure_cache_db(root)
    typer.echo(f"cache: schema at version {_scope_version(cache_db_path(root))}")
    if project is not None:
        ensure_project_db(project, root)
        typer.echo(
            f"project '{project}': schema at version "
            f"{_scope_version(project_db_path(project, root))}"
        )


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
        False, "--probe-ollama", help="Opt-in TCP probe of the configured Ollama endpoint (no model call)."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit one machine-readable JSON object instead of the text render."
    ),
) -> None:
    """Validate the foundation; exit non-zero on any failure.

    Extended (phase_5b): when ``--project`` is given, also runs the cross-DB
    ``work_source_files`` bridge reconcile + a cross-scope foreign_key_check
    (decision r2-10) without editing phase_0's ``doctor.py``. Stage C adds
    LLM-backend checks (pricing-snapshot age, task/model capability +
    structured-output mismatch, private-content policy conflict) and the opt-in
    ``--probe-ollama`` TCP probe — none of which make a paid call.
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


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (loopback by default)."),
    port: int = typer.Option(8765, help="Preferred port; auto-increments if busy."),
    allow_remote: bool = typer.Option(
        False,
        "--allow-remote",
        help=(
            "Permit binding a non-loopback interface. This exposes private evidence "
            "to anyone who can reach the host AND holds the session cookie."
        ),
    ),
    root: Optional[Path] = _root_option,
    open_browser: bool = typer.Option(
        False, "--open-browser", help="Open the one-time /auth URL in a browser."
    ),
) -> None:
    """Run the localhost web UI (FastAPI + uvicorn) over the project service layer.

    Pins ``$SEEDGRAPH_HOME``, refuses a non-loopback bind without ``--allow-remote``,
    auto-increments past a busy port, and prints a one-time ``/auth?t=<token>`` URL
    that exchanges the token for an HttpOnly, SameSite=Strict session cookie (the
    token never reappears in the address bar; cross-cutting #1).
    """
    from . import paths
    from .api.app import app as web_app
    from .project.service import list_projects
    from .run import mark_interrupted_runs
    from .web.serve import bind_is_allowed, find_free_port

    home = paths.resolve_home(root)
    os.environ["SEEDGRAPH_HOME"] = str(home)

    # Done-vs-crashed sweep (Build F ch9): stamp every run whose events.jsonl ends
    # non-terminal as `interrupted` BEFORE uvicorn binds, so no status request can
    # ever be served inside the restart window (web jobs die with this process).
    for slug in list_projects(home):
        for stale_run in mark_interrupted_runs(slug, root=home):
            typer.echo(f"marked stale run as interrupted: {slug}/{stale_run}", err=True)

    if not bind_is_allowed(host, allow_remote):
        typer.echo(
            f"refusing to bind non-loopback host {host!r} without --allow-remote",
            err=True,
        )
        raise typer.Exit(2)

    import uvicorn  # lazy: keep the CLI import-light and socket-free until we serve

    web_app.state.remote_bind = allow_remote
    if allow_remote and host not in {"127.0.0.1", "::1", "localhost"}:
        typer.echo(
            f"WARNING: binding {host!r} exposes private evidence to anyone who can "
            f"reach this host and presents the session cookie.",
            err=True,
        )

    port = find_free_port(host, port)
    url = f"http://{host}:{port}/auth?t={web_app.state.session_token}"
    typer.echo("seedgraph web UI — open this one-time sign-in URL:")
    typer.echo(f"  {url}")
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    uvicorn.run(web_app, host=host, port=port)


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
    """LLM-assisted post-conversion metadata backfill: replace filename-derived titles
    and attach doi/arxiv from each included work's converted markdown."""
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
            f"\tids={ids}\tllm={int(res.used_llm)}"
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
# phase_4: default note extraction (real grouped sub-app).
# ===========================================================================

extract_app = typer.Typer(
    name="extract",
    help="Note extraction (whole + chunked map-reduce).",
    no_args_is_help=True,
)


def _apply_cost_controls(
    h,
    config,
    work_ids,
    *,
    schema_id,
    profile_id,
    confirm_external,
    cache_conn,
    cache_root,
    budget_state,
    assume_yes,
    dry_run,
) -> None:
    """Stage C cost controls: monthly soft limit + require_confirmation gate (CLI).

    No-op unless the project config arms a monthly soft limit OR a confirmation
    threshold, so the default-config CLI path is byte-identical. Seeds
    ``budget_state.prior_month_spent_usd`` from the DB so the monthly soft limit is
    enforced against ``monthly_spend(conn, this_month) + this run`` (plan §Cost),
    estimates the run cost via a quiet dry-run pass (writes nothing), warns on a
    soft-limit overshoot, and prompts (``typer.confirm`` unless ``--yes``) when the
    run estimate is at or above ``require_confirmation_above_usd``."""
    from sqlmodel import Session

    from .errors import ConfigError
    from .extraction.runner import BudgetState, extract_note
    from .llm.cost import budget_status, monthly_spend, this_month

    b = config.budget
    if b.monthly_soft_limit_usd is None and b.require_confirmation_above_usd is None:
        return  # no cost controls configured -> no-op (default behavior unchanged)

    pconn = sqlite3.connect(str(h.db_path))
    try:
        # Monthly soft limit = prior DB spend this month + this run.
        budget_state.prior_month_spent_usd = monthly_spend(pconn, this_month())
        est = 0.0
        if not dry_run:
            with Session(h.engine) as s2:
                for wid in work_ids:
                    try:
                        dres = extract_note(
                            s2,
                            cache_conn,
                            work_id=wid,
                            schema_id=schema_id,
                            profile_id=profile_id,
                            force=True,
                            confirm_external=confirm_external,
                            dry_run=True,
                            budget_state=BudgetState(),
                            config=config,
                            cache_root=cache_root,
                        )
                    except ConfigError:
                        continue
                    est += dres.estimated_cost or 0.0
        status = budget_status(pconn, b, est)
    finally:
        pconn.close()

    if status.over_monthly_soft_limit:
        typer.echo(
            f"[budget] projected monthly spend ${status.projected_usd:.4f} exceeds "
            f"monthly_soft_limit_usd ${status.monthly_soft_limit_usd:.4f} "
            f"(prior ${status.prior_spend_usd:.4f} + est this run ${status.this_run_usd:.4f})"
        )
    if status.requires_confirmation and not dry_run:
        if not assume_yes and not typer.confirm(
            f"Estimated run cost ${status.this_run_usd:.4f} is at or above the confirmation "
            f"threshold ${status.confirmation_threshold_usd:.4f}. Proceed?"
        ):
            typer.echo("aborted: run cost not confirmed (pass --yes to skip this prompt)")
            raise typer.Exit(1)


def _worklist_by_centrality(h, cap: Optional[int] = None, *, work_ids: Optional[list] = None) -> list[str]:
    """Extraction worklist ordered by citation in-degree, most-cited first (D7).

    One ``LEFT JOIN citation_edges ON target_work_id`` + ``GROUP BY`` SQL read —
    no NetworkX, no new state. Works with zero in-edges sort last; ties break
    deterministically on ``work_id``. ``cap`` (when set) truncates the ordered
    list, so a mid-run budget stop strands the least-central tail instead of a
    random one (prototype semantic_extract.py ``_select_papers``).

    Membership: with ``work_ids=None``, the included works with a bridged
    markdown (the ``extract notes`` default — same membership as
    ``_included_works_with_markdown``, different order); with an explicit
    ``work_ids`` candidate list (the ``notes-chunked`` skipped_oversize
    default), exactly that set reordered. ``cite parse`` keeps
    ``_included_works_with_markdown`` and its created_at order untouched.
    """
    from sqlmodel import Session

    from .acquisition.bridge import resolve_work_markdown

    conn = sqlite3.connect(str(h.db_path))
    conn.row_factory = sqlite3.Row
    try:
        if work_ids is None:
            rows = conn.execute(
                "SELECT w.work_id AS work_id FROM works w "
                "JOIN project_documents pd ON pd.work_id = w.work_id "
                "LEFT JOIN citation_edges ce ON ce.target_work_id = w.work_id "
                "WHERE pd.inclusion_status = 'included' "
                "GROUP BY w.work_id "
                "ORDER BY COUNT(DISTINCT ce.source_work_id) DESC, w.work_id"
            ).fetchall()
        elif not work_ids:
            return []
        else:
            marks = ",".join("?" for _ in work_ids)
            rows = conn.execute(
                "SELECT w.work_id AS work_id FROM works w "
                "LEFT JOIN citation_edges ce ON ce.target_work_id = w.work_id "
                f"WHERE w.work_id IN ({marks}) "
                "GROUP BY w.work_id "
                "ORDER BY COUNT(DISTINCT ce.source_work_id) DESC, w.work_id",
                list(work_ids),
            ).fetchall()
    finally:
        conn.close()
    ordered = [row["work_id"] for row in rows]
    if work_ids is None:
        # The bridged-markdown membership gate, applied AFTER ordering so --cap
        # counts extractable works (a capped run never under-fills its budget
        # slot with markdown-less works).
        with Session(h.engine) as session:
            ordered = [
                wid for wid in ordered
                if resolve_work_markdown(session, work_id=wid) is not None
            ]
    return ordered[:cap] if cap is not None else ordered


def _echo_worklist_header(work_ids: list, cap: Optional[int]) -> None:
    """The CLI header line naming the worklist ordering choice (Build C chunk 5)."""
    typer.echo(
        f"worklist: {len(work_ids)} works ordered by citation in-degree; "
        f"cap={cap if cap is not None else 'none'}"
    )


@extract_app.command("notes")
def extract_notes(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: Optional[str] = typer.Option(None, "--work", help="Single work id (default: all included with markdown)."),
    schema: Optional[str] = typer.Option(None, "--schema", help="Schema id (default: default_research_note_v1)."),
    profile: Optional[str] = typer.Option(None, "--profile", help="Override the routing preferred profile."),
    force: bool = typer.Option(False, "--force", help="Re-extract even if a current note exists."),
    confirm_external: bool = typer.Option(False, "--confirm-external", help="Permit private full text to leave the machine externally."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate cost + show planned work; write nothing."),
    chunked: bool = typer.Option(False, "--chunked", help="Auto-route skipped_oversize works to phase_4b chunked map-reduce extraction."),
    cap: Optional[int] = typer.Option(None, "--cap", min=1, help="Extract at most N works (most-cited first; ignored with --work)."),
    concurrency: int = typer.Option(
        1, "--concurrency", min=1,
        help=(
            "Parallel workers over works (default 1 = today's sequential path). "
            "Opt-in for hosted-API runs; start at 2-4 — the prototype's live run "
            "hit Anthropic's Tier-1 30k tokens-per-minute ceiling on a burst, so "
            "keep the fan-out modest and let the executor's 429 backoff (the "
            "paired safety leg) absorb residual bursts."
        ),
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the require_confirmation_above_usd cost prompt."),
    root: Optional[Path] = _root_option,
) -> None:
    """Run default_research_note_v1 over each included work's markdown (loop)."""
    from sqlmodel import Session

    from . import cache_access
    from .config.loader import load_project_config as _load_routing_config
    from .errors import ConfigError
    from .extraction.chunked_runner import extract_note_chunked
    from .extraction.runner import BudgetState, extract_note
    from .extraction.schema import SCHEMA_ID
    from .project.service import open_project
    from .run import ensure_run, update_manifest

    h = open_project(slug, root=root)
    schema_id = schema or SCHEMA_ID
    try:
        routing_config = _load_routing_config(slug, root)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}")
        raise typer.Exit(2)

    # Default worklist: citation in-degree desc (+ optional --cap), so a mid-run
    # budget stop strands the least-central tail (Build C chunk 5 / D7).
    work_ids = [work_id] if work_id is not None else _worklist_by_centrality(h, cap)
    if not work_ids:
        typer.echo("no works with markdown to extract")
        return
    if work_id is None:
        _echo_worklist_header(work_ids, cap)

    budget = BudgetState()
    cache_conn = cache_access.open_cache_ro(root)
    try:
        _apply_cost_controls(
            h, routing_config, work_ids,
            schema_id=schema_id, profile_id=profile, confirm_external=confirm_external,
            cache_conn=cache_conn, cache_root=root, budget_state=budget,
            assume_yes=yes, dry_run=dry_run,
        )
        # Durable run-level reproducibility record of the worklist choice (gap
        # scan §5.2 build-entails / critic C-4: the CLI echo alone is not
        # durable). Written BEFORE the loop so a mid-run budget stop still
        # leaves the record; a fresh run is minted, so the D5 single-write
        # section contract can never trip.
        if work_id is None and not dry_run:
            manifest_run_id = ensure_run(slug, root=root)
            update_manifest(
                slug,
                manifest_run_id,
                {"note_extraction": {
                    "run_id": manifest_run_id,
                    "ordering": "citation_in_degree",
                    "cap": cap,
                    "works": len(work_ids),
                }},
                root=root,
            )
        def _process(session, conn, wid) -> "tuple[list[str], int]":
            """Per-work body shared by the sequential loop and the --concurrency
            pool (Build C chunk 6 / D6). Returns (echo lines, exit-code
            contribution); the caller echoes, so pool output never interleaves
            mid-line — each line already self-identifies by work_id."""
            try:
                result = extract_note(
                    session,
                    conn,
                    work_id=wid,
                    schema_id=schema_id,
                    profile_id=profile,
                    force=force,
                    confirm_external=confirm_external,
                    dry_run=dry_run,
                    budget_state=budget,
                    config=routing_config,
                    cache_root=root,
                )
            except ConfigError as exc:
                return [f"{wid}\tconfig_error\t{exc}"], 2
            line = f"{wid}\t{result.run_status}"
            if result.note_id:
                line += f"\tnote={result.note_id} claims={result.claim_count} spans={result.span_count}"
            if result.message:
                line += f"\t{result.message}"
            line += f"\trunning_cost=${budget.spent_usd:.6f}"
            lines = [line]
            code = 0
            # --chunked: auto-route an oversize work to phase_4b map-reduce so the
            # paper is processed, not dropped (D4). The same BudgetState threads
            # on, and the re-route stays inside the same worker (D6).
            if chunked and result.run_status == "skipped_oversize":
                cres = extract_note_chunked(
                    session,
                    conn,
                    work_id=wid,
                    schema_id=schema_id,
                    profile_id=profile,
                    force=force,
                    confirm_external=confirm_external,
                    dry_run=dry_run,
                    budget_state=budget,
                    config=routing_config,
                    cache_root=root,
                )
                cline = f"{wid}\t{cres.status} (chunked)\tchunks={cres.chunk_count}"
                if cres.note_id:
                    cline += f" note={cres.note_id} claims={cres.merged_claim_count}"
                if cres.skipped_reason:
                    cline += f"\t{cres.skipped_reason}"
                lines.append(cline)
                if cres.status in ("skipped_policy", "skipped_oversize_section"):
                    code = 1
            elif result.run_status in ("skipped_policy", "skipped_oversize"):
                code = 1
            return lines, code

        exit_code = 0
        if concurrency <= 1:
            with Session(h.engine) as session:
                for wid in work_ids:
                    lines, code = _process(session, cache_conn, wid)
                    for ln in lines:
                        typer.echo(ln)
                    exit_code = max(exit_code, code)
        else:
            # D6 works-level fan-out: each worker owns its own Session AND its own
            # read-only cache connection (sqlite3 connections must stay on the
            # thread that opened them); the shared BudgetState is lock-protected,
            # so spend accounting and the stop latch are race-free. Results echo
            # per-completion; exit-code aggregation is unchanged.
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _worker(wid):
                wconn = cache_access.open_cache_ro(root)
                try:
                    with Session(h.engine) as wsession:
                        return _process(wsession, wconn, wid)
                finally:
                    wconn.close()

            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_worker, wid) for wid in work_ids]
                for fut in as_completed(futures):
                    lines, code = fut.result()
                    for ln in lines:
                        typer.echo(ln)
                    exit_code = max(exit_code, code)
    finally:
        cache_conn.close()
    if exit_code:
        raise typer.Exit(exit_code)


def _skipped_oversize_worklist(h) -> list[str]:
    """Works phase_4 recorded ``skipped_oversize`` that still have no current note.

    The phase_4b default worklist (plan §6). Distinct works with a
    ``run_status='skipped_oversize'`` extraction_runs row and no surviving
    ``structured_notes`` row for the default schema.
    """
    conn = sqlite3.connect(str(h.db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT DISTINCT r.work_id AS work_id FROM extraction_runs r "
            "WHERE r.run_status = 'skipped_oversize' "
            "AND NOT EXISTS (SELECT 1 FROM structured_notes n WHERE n.work_id = r.work_id) "
            "ORDER BY r.work_id"
        ).fetchall()
    finally:
        conn.close()
    return [row["work_id"] for row in rows]


@extract_app.command("notes-chunked")
def extract_notes_chunked(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: Optional[str] = typer.Option(None, "--work", help="Single work id (default: phase_4 skipped_oversize works with no current note)."),
    schema: Optional[str] = typer.Option(None, "--schema", help="Schema id (default: default_research_note_v1)."),
    profile: Optional[str] = typer.Option(None, "--profile", help="Override the routing preferred profile."),
    overlap_tokens: int = typer.Option(128, "--overlap-tokens", help="Fixed inter-chunk overlap (tokens)."),
    reserve_output: Optional[float] = typer.Option(None, "--reserve-output", help="Fraction of the window reserved for output (else max_output_tokens)."),
    keep_chunk_json: bool = typer.Option(False, "--keep-chunk-json", help="Persist per-chunk JSON on map runs for audit (default off)."),
    force: bool = typer.Option(False, "--force", help="Re-extract even if a current note exists."),
    confirm_external: bool = typer.Option(False, "--confirm-external", help="Permit private full text to leave the machine externally."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Plan chunks + project total cost; write nothing."),
    cap: Optional[int] = typer.Option(None, "--cap", min=1, help="Extract at most N works (most-cited first; ignored with --work)."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Target manifest run (default: minted)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Chunked map-reduce extraction over phase_4 ``skipped_oversize`` works (phase_4b)."""
    from sqlmodel import Session

    from . import cache_access
    from .config.loader import load_project_config as _load_routing_config
    from .errors import ConfigError
    from .extraction.chunked_runner import extract_note_chunked
    from .extraction.runner import BudgetState
    from .extraction.schema import SCHEMA_ID
    from .project.service import open_project
    from .run import ensure_run, update_manifest

    h = open_project(slug, root=root)
    schema_id = schema or SCHEMA_ID
    try:
        routing_config = _load_routing_config(slug, root)
    except ConfigError as exc:
        typer.echo(f"config error: {exc}")
        raise typer.Exit(2)

    # Default worklist: the skipped_oversize set reordered by citation in-degree
    # desc (+ optional --cap) — Build C chunk 5 / D7.
    work_ids = (
        [work_id]
        if work_id is not None
        else _worklist_by_centrality(h, cap, work_ids=_skipped_oversize_worklist(h))
    )
    if not work_ids:
        typer.echo("no skipped_oversize works to extract")
        return
    if work_id is None:
        _echo_worklist_header(work_ids, cap)

    build_run_id = ensure_run(slug, run_id=run_id, root=root)
    budget = BudgetState()
    totals = {
        "works": 0, "chunks_total": 0, "notes_written": 0,
        "skipped_oversize_section": 0, "skipped_budget": 0, "skipped_policy": 0,
        "skipped_no_llm": 0, "extraction_failed": 0, "input_tokens": 0,
        "output_tokens": 0,
    }
    est_cost = 0.0
    cache_conn = cache_access.open_cache_ro(root)
    exit_code = 0
    try:
        with Session(h.engine) as session:
            for wid in work_ids:
                try:
                    res = extract_note_chunked(
                        session,
                        cache_conn,
                        work_id=wid,
                        schema_id=schema_id,
                        profile_id=profile,
                        overlap_tokens=overlap_tokens,
                        reserve_output=reserve_output,
                        keep_chunk_json=keep_chunk_json,
                        force=force,
                        confirm_external=confirm_external,
                        dry_run=dry_run,
                        build_run_id=build_run_id,
                        budget_state=budget,
                        config=routing_config,
                        cache_root=root,
                    )
                except ConfigError as exc:
                    typer.echo(f"{wid}\tconfig_error\t{exc}")
                    exit_code = 2
                    continue
                totals["works"] += 1
                totals["chunks_total"] += res.chunk_count
                totals["input_tokens"] += res.input_tokens
                totals["output_tokens"] += res.output_tokens
                est_cost += res.estimated_cost
                if res.status == "success":
                    totals["notes_written"] += 1
                elif res.status in totals:
                    totals[res.status] += 1
                line = (
                    f"{wid}\t{res.status}\tchunks={res.chunk_count} "
                    f"(ok={res.chunks_succeeded} fail={res.chunks_failed}) "
                    f"tokens={res.input_tokens}+{res.output_tokens} cost=${res.estimated_cost:.6f}"
                )
                if res.note_id:
                    line += f"\tnote={res.note_id} claims={res.merged_claim_count}"
                if res.skipped_reason:
                    line += f"\t{res.skipped_reason}"
                line += f"\trunning_cost=${budget.spent_usd:.6f}"
                typer.echo(line)
                if res.status in ("skipped_policy", "skipped_oversize_section"):
                    exit_code = max(exit_code, 1)
    finally:
        cache_conn.close()

    if not dry_run:
        update_manifest(
            slug,
            build_run_id,
            {
                "chunked_extraction": {
                    **totals,
                    "est_cost": est_cost,
                    "run_id": build_run_id,
                    # Worklist reproducibility record (gap scan §5.2 / C-4).
                    "ordering": "citation_in_degree" if work_id is None else "explicit_work",
                    "cap": cap if work_id is None else None,
                }
            },
            root=root,
        )
    if exit_code:
        raise typer.Exit(exit_code)


# ===========================================================================
# phase_6: project lenses (real grouped sub-app — new/list/show/validate/
# calibrate/run/results/status). Resolves the project from --project / cwd.
# ===========================================================================

lens_app = typer.Typer(
    name="lens",
    help="Project lenses: new/list/show/validate/calibrate/run/results/status.",
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
    """Show the resolved lens definition + per-status coverage counts."""
    from sqlmodel import Session

    from .lenses import registry, results as lens_results_mod

    h = _resolve_lens_project(project, root)
    project_dir = h.root / "projects" / h.slug
    lens = registry.load_lens(project_dir, lens_id)
    with Session(h.engine) as session:
        row = registry.get_lens_row(session, lens_id)
        cov = lens_results_mod.coverage(session, lens_id)
    status = row.status if row else "unregistered"
    typer.echo(
        f"lens {lens.lens_id} ({lens.name})\n"
        f"  status={status} object_type={lens.object_type} hash={lens.definition_hash()[:12]}\n"
        f"  fields={list(lens.output_schema)}\n"
        f"  coverage: found={cov.found} not_found={cov.not_found} ambiguous={cov.ambiguous} "
        f"skipped_no_markdown={cov.skipped_no_markdown} works={cov.works_total}"
    )


def _lens_router(routing_config):
    """Build the LensRouter for the CLI via the shared service lift."""
    from .lenses import runner as lens_runner

    return lens_runner.build_lens_router(routing_config)


def _included_lens_works(h) -> list[str]:
    from .lenses import runner as lens_runner

    return lens_runner.included_lens_work_ids(h)


@lens_app.command("calibrate")
def lens_calibrate(
    lens_id: str = typer.Argument(..., help="Lens id to calibrate."),
    sample: int = typer.Option(3, "--sample", help="Number of sample works."),
    work: Optional[list[str]] = typer.Option(None, "--work", help="Explicit work id(s)."),
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Sample-run the lens (sets status=calibrating) and print an FP/FN review report."""
    from sqlmodel import Session

    from .config.loader import load_project_config
    from .lenses import calibrate, registry

    h = _resolve_lens_project(project, root)
    project_dir = h.root / "projects" / h.slug
    routing_config = load_project_config(h.slug, root)
    with Session(h.engine) as session:
        registry.sync_lens(session, project_dir, lens_id)
        lens = registry.load_lens(project_dir, lens_id)
        report = calibrate.run_calibration(
            session, root, lens, _lens_router(routing_config),
            sample=sample, work_ids=list(work) if work else None,
        )
    typer.echo(
        f"calibrated {lens_id} on {len(report.sample_work_ids)} works: "
        f"found={len(report.found_records)} not_found={len(report.not_found_work_ids)} "
        f"review_items={len(report.review_items)}"
    )
    for rec in report.found_records:
        typer.echo(f"  found {rec['work_id']}\t{rec['normalized_label']}\t{rec['claim_text']}")


@lens_app.command("run")
def lens_run(
    lens_id: str = typer.Argument(..., help="Lens id to run."),
    all_works: bool = typer.Option(False, "--all", help="Run over the whole included corpus."),
    work: Optional[list[str]] = typer.Option(None, "--work", help="Explicit work id(s)."),
    force: bool = typer.Option(False, "--force", help="Reprocess every targeted work (append-only)."),
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Run the lens over the corpus; default skips current works, --force reprocesses."""
    from sqlmodel import Session

    from . import run as run_mod
    from .config.loader import load_project_config
    from .lenses import registry
    from .lenses.runner import run_lens

    h = _resolve_lens_project(project, root)
    project_dir = h.root / "projects" / h.slug
    routing_config = load_project_config(h.slug, root)
    with Session(h.engine) as session:
        registry.sync_lens(session, project_dir, lens_id)
        lens = registry.load_lens(project_dir, lens_id)
        work_ids = list(work) if work else _included_lens_works(h)
        if not work_ids:
            typer.echo("no included works to run")
            return
        result = run_lens(session, root, lens, work_ids, _lens_router(routing_config), force=force)
        # First full run promotes the lens to 'active' + snapshots definition_yaml
        # once (immutability-on-use, decision 50); idempotent thereafter.
        registry.promote_to_active(session, lens)

    build_run_id = run_mod.ensure_run(h.slug, root=root)
    run_mod.update_manifest(
        h.slug, build_run_id,
        {"lens_run": {
            "lens_id": lens_id,
            "definition_hash": lens.definition_hash(),
            "found": result.found, "not_found": result.not_found,
            "ambiguous": result.ambiguous, "extraction_failed": result.extraction_failed,
            "skipped_current": result.skipped_current,
            "skipped_no_markdown": result.skipped_no_markdown,
            "router_calls": result.router_calls,
            "degraded": result.degraded, "capability_note": result.capability_note,
        }},
        root=root,
    )
    typer.echo(
        f"lens {lens_id}: found={result.found} not_found={result.not_found} "
        f"ambiguous={result.ambiguous} failed={result.extraction_failed} "
        f"skipped_current={len(result.skipped_current)} "
        f"skipped_no_markdown={len(result.skipped_no_markdown)} "
        f"router_calls={result.router_calls}"
        + (" [degraded:no-LLM]" if result.degraded else "")
    )


@lens_app.command("results")
def lens_results_cmd(
    lens_id: str = typer.Argument(..., help="Lens id."),
    status: str = typer.Option("found", "--status", help="found | not_found | all."),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON."),
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Retrieve lens outputs across the project (the success criterion)."""
    import json as _json

    from sqlmodel import Session

    from .lenses import results as lens_results_mod
    from .lenses import runner as lens_runner

    h = _resolve_lens_project(project, root)
    if as_json:
        # Shared serializable payload (lifted to lenses.runner.lens_results so the
        # CLI --json and the web lens-detail screen render the same rows).
        with Session(h.engine) as session:
            payload = lens_runner.lens_results(session, lens_id, status=status)
        typer.echo(_json.dumps(payload, indent=2))
        return
    with Session(h.engine) as session:
        views = lens_results_mod.lens_results(session, lens_id, status=status)
    if not views:
        typer.echo("no results")
        return
    typer.echo("work_id\tstatus\tnormalized_label\tsection\tspans\tquote")
    for v in views:
        quote = (v.claim_text or "").replace("\n", " ")
        if len(quote) > 60:
            quote = quote[:57] + "..."
        typer.echo(
            f"{v.work_id}\t{v.status}\t{v.normalized_label or ''}\t{v.section or ''}\t"
            f"{len(v.span_ids)}\t{quote}"
        )


@lens_app.command("status")
def lens_status(
    lens_id: str = typer.Argument(..., help="Lens id."),
    project: Optional[str] = _lens_project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Report coverage, stale runs, and skipped works for a lens."""
    from sqlmodel import Session

    from .lenses import registry

    h = _resolve_lens_project(project, root)
    project_dir = h.root / "projects" / h.slug
    # Shared coverage+staleness read-model (lifted to registry.lens_staleness).
    with Session(h.engine) as session:
        st = registry.lens_staleness(session, project_dir, lens_id)
    typer.echo(
        f"lens {lens_id} status={st['status']}\n"
        f"  found={st['found']} not_found={st['not_found']} ambiguous={st['ambiguous']} "
        f"extraction_failed={st['extraction_failed']}\n"
        f"  works_total={st['works_total']} works_covered={st['works_covered']} "
        f"skipped_no_markdown={st['skipped_no_markdown']}\n"
        f"  stale_runs={st['stale_count']}"
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


def _resolve_concept_profile(handle, profile_name: Optional[str]):
    """Resolve an LLM profile for concept proposal, or ``None`` (deterministic).

    Loads the RUNTIME LLM config via ``config.loader.load_project_config`` (review
    #3) — ``handle.config`` is the lighter project handle config, not the merged
    runtime ``llm.profiles``, so a named ``--profile`` only actually selects a
    profile when resolved here. Returns ``None`` (deterministic) when unnamed."""
    if not profile_name:
        return None
    from .config.loader import load_project_config

    cfg = load_project_config(handle.slug, getattr(handle, "root", None))
    return cfg.llm.profiles.get(profile_name)


@concepts_app.command("build")
def concepts_build(
    slug: str = typer.Argument(..., help="Project slug."),
    profile: Optional[str] = typer.Option(None, "--profile", help="LLM profile for synonym proposal."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Deterministic only (exact-key + acronym + co_occurs)."),
    tau: float = typer.Option(0.6, "--tau", help="Token-Jaccard clustering threshold."),
    min_shared: int = typer.Option(
        2, "--min-shared",
        help="Min shared works for a co_occurs_with edge (drops within-single-paper cliques).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Extract -> cluster -> (propose) -> guard -> merge; write concepts/aliases/
    claim_concepts; build Work--discusses-->Concept + co_occurs_with edges."""
    from .run import ensure_run
    from .semantic import build_semantic_overlay
    from .semantic.export import update_graph_manifest
    from .project.service import open_project

    h = open_project(slug, root=root)
    run_id = ensure_run(slug, root=h.root)
    prof = None if no_llm else _resolve_concept_profile(h, profile)
    routing_config = None
    if prof is not None:
        from .config.loader import load_project_config

        routing_config = load_project_config(slug, root)
    conn = connect_project_raw(h.db_path)
    try:
        report = build_semantic_overlay(
            conn, run_id=run_id, profile=prof, tau=tau, min_shared=min_shared,
            config=routing_config,
        )
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


# ===========================================================================
# Stage C — LLM backend admin surface: `llm profiles|keys|usage|estimate`.
# Reference-only key rows (never values), usage aggregation, preflight cost
# estimate. NO default paid call — only `llm keys test --smoke` (opt-in) dispatches.
# ===========================================================================
llm_app = typer.Typer(
    name="llm",
    help="LLM backend admin: profiles, key references, usage, cost estimate.",
    no_args_is_help=True,
)
llm_profiles_app = typer.Typer(name="profiles", help="LLM access profiles.", no_args_is_help=True)
llm_keys_app = typer.Typer(
    name="keys", help="Key REFERENCES (env-var names; never values) in llm_key_refs.", no_args_is_help=True
)
llm_route_app = typer.Typer(
    name="route", help="Task-type → profile routing (read the table via `llm profiles list`).",
    no_args_is_help=True,
)


def _llm_load_config(project: Optional[str], root: Optional[Path]):
    from .config.loader import load_global_config, load_project_config

    if project is not None:
        return load_project_config(project, root)
    return load_global_config(root)


def _open_project_conn(project: str, root: Optional[Path]) -> sqlite3.Connection:
    """Open (migrating if needed) the project's project.db (FKs ON) for the llm
    admin verbs. Resolves the path directly — no project.yaml is required, just a
    valid slug (path-traversal guarded by ``validate_slug``)."""
    from . import paths

    paths.validate_slug(project)
    ensure_project_db(project, root)
    conn = sqlite3.connect(str(project_db_path(project, root)))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _env_present(key_source: str, reference: Optional[str]) -> str:
    """'yes'/'no'/'?' — whether the referenced secret RESOLVES (never the value)."""
    from .errors import ConfigError
    from .llm.secrets import resolve_named_secret, resolve_secret

    if not reference:
        return "?"
    try:
        if (key_source or "").lower() == "auto":
            # A single reference can't name both chain legs (ADR-0002) — infer
            # by shape: keyring entry names always carry the seedgraph/ prefix.
            if reference.startswith("seedgraph/"):
                return "yes" if resolve_named_secret(reference) is not None else "no"
            return "yes" if os.environ.get(reference) is not None else "no"
        return "yes" if resolve_secret(key_source, reference) is not None else "no"
    except ConfigError:
        return "?"  # unknown key_source or malformed keyring name


@llm_profiles_app.command("list")
def llm_profiles_list(
    project: Optional[str] = _project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """List configured profiles (+ registry-aware availability) and the routing table."""
    from .errors import SeedgraphError
    from .llm.profiles import is_local_profile, is_profile_available

    try:
        cfg = _llm_load_config(project, root)
    except SeedgraphError as exc:
        typer.echo(f"config error: {exc}")
        raise typer.Exit(2)
    typer.echo("profile_id\tprovider\tmodel\taccess_mode\tlocal\tavailable")
    for pid, p in sorted(cfg.llm.profiles.items()):
        typer.echo(
            f"{pid}\t{p.provider}\t{p.model or ''}\t{p.access_mode}\t"
            f"{int(is_local_profile(p))}\t{int(is_profile_available(p))}"
        )
    typer.echo("\ntask_type\tpreferred_profile\tfallback_profile")
    for task_type, r in sorted(cfg.llm.routes.items()):
        typer.echo(f"{task_type}\t{r.preferred_profile}\t{r.fallback_profile or ''}")


@llm_keys_app.command("list")
def llm_keys_list(
    project: str = typer.Option(..., "--project", help="Project slug."),
    root: Optional[Path] = _root_option,
) -> None:
    """List key REFERENCE rows (provider + env-var name only — never a secret value)."""
    conn = _open_project_conn(project, root)
    try:
        rows = conn.execute(
            "SELECT provider, access_mode, key_source, key_reference, env_var, last_validated_at "
            "FROM llm_key_refs ORDER BY provider, key_reference"
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        typer.echo("no key references registered")
        return
    typer.echo("provider\taccess_mode\tkey_source\tkey_reference\tenv_present\tlast_validated_at")
    for provider, access_mode, key_source, key_reference, env_var, lv in rows:
        ref = env_var or key_reference
        present = _env_present(key_source or "environment", ref)
        typer.echo(
            f"{provider}\t{access_mode or ''}\t{key_source or 'environment'}\t{ref or ''}\t"
            f"{present}\t{lv or ''}"
        )


@llm_keys_app.command("set")
def llm_keys_set(
    project: str = typer.Option(..., "--project", help="Project slug."),
    provider: str = typer.Option(..., "--provider", help="anthropic|ollama|..."),
    env_var: str = typer.Option(
        ..., "--env-var",
        help="Key REFERENCE (never the value): env var NAME, or for --key-source keyring "
        "the entry name (seedgraph/<service>/default).",
    ),
    access_mode: str = typer.Option("api_key", "--access-mode", help="api_key|local|none."),
    key_source: str = typer.Option("environment", "--key-source", help="environment|keyring."),
    root: Optional[Path] = _root_option,
) -> None:
    """Register a key REFERENCE row (provider + env-var NAME). Stores NO secret value."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conn = _open_project_conn(project, root)
    try:
        conn.execute(
            "INSERT INTO llm_key_refs (provider, access_mode, key_source, key_reference, "
            "env_var, created_at, last_validated_at) VALUES (?,?,?,?,?,?,NULL) "
            "ON CONFLICT(provider, key_reference) DO UPDATE SET "
            "access_mode=excluded.access_mode, key_source=excluded.key_source, "
            "env_var=excluded.env_var",
            (provider, access_mode, key_source, env_var, env_var, now),
        )
        conn.commit()
    finally:
        conn.close()
    typer.echo(f"registered key reference: provider={provider} env_var={env_var} (no value stored)")


@llm_keys_app.command("remove")
def llm_keys_remove(
    project: str = typer.Option(..., "--project", help="Project slug."),
    provider: str = typer.Option(..., "--provider", help="Provider to remove references for."),
    env_var: Optional[str] = typer.Option(
        None, "--env-var", help="Key reference (env var name); default removes ALL rows for the provider."
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Remove key reference row(s) for a provider."""
    conn = _open_project_conn(project, root)
    try:
        if env_var is not None:
            cur = conn.execute(
                "DELETE FROM llm_key_refs WHERE provider=? AND key_reference=?", (provider, env_var)
            )
        else:
            cur = conn.execute("DELETE FROM llm_key_refs WHERE provider=?", (provider,))
        conn.commit()
        n = cur.rowcount
    finally:
        conn.close()
    typer.echo(f"removed {n} key reference row(s)")


def _llm_smoke_dispatch(provider: str, reference: str, key_source: str, *, project: str, root):
    """OPT-IN single real dispatch to verify a key works (a paid call). Returns (ok, detail)."""
    from .config.models import LLMProfile
    from .errors import SeedgraphError
    from .llm.executor import build_backend

    try:
        cfg = _llm_load_config(project, root)
        profile = next((p for p in cfg.llm.profiles.values() if p.provider == provider), None)
        if profile is None:
            # The reference lands in the field matching its source: a keyring row's
            # reference is an entry NAME (seedgraph/...), not an env var.
            ref_field = "key_name" if key_source in ("keyring", "system_keyring") else "env_var"
            profile = LLMProfile(
                profile_id=f"{provider}_smoke", provider=provider, key_source=key_source,
                **{ref_field: reference},
            )
        backend = build_backend(profile, cfg)
        comp = backend.complete("ping", "Reply with the single word: ok", model=profile.model, max_tokens=8)
        return True, f"{len((comp.text or '').strip())} chars returned"
    except (SeedgraphError, Exception) as exc:  # noqa: BLE001 - report, never crash the CLI
        return False, str(exc)[:80]


@llm_keys_app.command("test")
def llm_keys_test(
    project: str = typer.Option(..., "--project", help="Project slug."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Test one provider (default: all)."),
    smoke: bool = typer.Option(
        False, "--smoke", help="OPT-IN: ONE real dispatch to verify the key works (a paid call)."
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Validate key REFERENCES (default makes NO paid call — presence check only).

    Default: report whether each referenced env var resolves + stamp
    ``last_validated_at`` on present rows. ``--smoke`` (opt-in) additionally makes
    ONE real provider dispatch — the only path that can spend money."""
    from datetime import datetime, timezone

    conn = _open_project_conn(project, root)
    any_fail = False
    try:
        q = "SELECT provider, access_mode, key_source, key_reference, env_var FROM llm_key_refs"
        params: tuple = ()
        if provider is not None:
            q += " WHERE provider=?"
            params = (provider,)
        rows = conn.execute(q + " ORDER BY provider, key_reference", params).fetchall()
        if not rows:
            typer.echo("no key references registered")
            return
        now = datetime.now(timezone.utc).isoformat()
        for prov, _access_mode, key_source, key_reference, env_var in rows:
            ref = env_var or key_reference
            present = _env_present(key_source or "environment", ref)
            line = f"{prov}\t{ref}\tpresent={present}"
            if present == "yes":
                conn.execute(
                    "UPDATE llm_key_refs SET last_validated_at=? WHERE provider=? AND key_reference=?",
                    (now, prov, key_reference),
                )
                if smoke:
                    ok, detail = _llm_smoke_dispatch(
                        prov, ref, key_source or "environment", project=project, root=root
                    )
                    line += f"\tsmoke={'ok' if ok else 'FAIL'} ({detail})"
                    any_fail = any_fail or not ok
            else:
                any_fail = True
            typer.echo(line)
        conn.commit()
    finally:
        conn.close()
    if any_fail:
        raise typer.Exit(1)


@llm_app.command("usage")
def llm_usage(
    project: str = typer.Option(..., "--project", help="Project slug."),
    group_by: str = typer.Option("task", "--group-by", help="task|model|provider|date."),
    month: Optional[str] = typer.Option(None, "--month", help="Filter to a calendar month (YYYY-MM)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Aggregate llm_usage_events by task/model/provider/date (calls/tokens/cost)."""
    dims = {
        "task": "task_type",
        "model": "model",
        "provider": "provider",
        "date": "substr(created_at,1,10)",
    }
    if group_by not in dims:
        typer.echo(f"--group-by must be one of {sorted(dims)}")
        raise typer.Exit(2)
    dim = dims[group_by]
    where = ""
    params: tuple = ()
    if month is not None:
        where = "WHERE substr(created_at,1,7)=?"
        params = (month,)
    conn = _open_project_conn(project, root)
    try:
        rows = conn.execute(
            f"SELECT {dim} AS k, COUNT(*), COALESCE(SUM(input_tokens),0), "
            f"COALESCE(SUM(output_tokens),0), COALESCE(SUM(estimated_cost),0.0) "
            f"FROM llm_usage_events {where} GROUP BY k ORDER BY k",
            params,
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(estimated_cost),0.0) FROM llm_usage_events {where}",
            params,
        ).fetchone()
    finally:
        conn.close()
    typer.echo(f"{group_by}\tcalls\tinput_tokens\toutput_tokens\test_cost_usd")
    for k, n, it, ot, cost in rows:
        typer.echo(f"{k if k is not None else '(none)'}\t{n}\t{it}\t{ot}\t{float(cost):.6f}")
    typer.echo(f"TOTAL\t{total[0]}\t-\t-\t{float(total[1]):.6f}")


@llm_app.command("estimate")
def llm_estimate(
    project: str = typer.Option(..., "--project", help="Project slug."),
    task: str = typer.Option(..., "--task", help="task_type to route (e.g. answer_generation)."),
    input_tokens: int = typer.Option(..., "--input-tokens", help="Estimated prompt tokens."),
    output_tokens: Optional[int] = typer.Option(
        None, "--output-tokens", help="Estimated output tokens (default: mirror input)."
    ),
    access_class: str = typer.Option("open_access", "--access-class", help="Source access class (content gate)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Preflight cost estimate for a task (NO dispatch); shows the monthly budget status."""
    from .config.loader import load_llm_capabilities, load_project_config
    from .errors import SeedgraphError
    from .llm.cost import budget_status, preflight_estimate
    from .llm.routing import NoLlmRoute, resolve_route

    try:
        cfg = load_project_config(project, root)
    except SeedgraphError as exc:
        typer.echo(f"config error: {exc}")
        raise typer.Exit(2)
    caps = load_llm_capabilities()
    try:
        route = resolve_route(task, access_class, cfg)
    except SeedgraphError as exc:
        typer.echo(f"task={task} BLOCKED by the content/policy gate: {exc}")
        raise typer.Exit(1)
    if isinstance(route, NoLlmRoute):
        typer.echo(
            f"task={task} -> no LLM route (deterministic_fallback="
            f"{route.deterministic_fallback}); estimated cost $0.000000"
        )
        return
    profile = cfg.llm.profiles.get(route.profile_id)
    model = profile.model if profile else None
    cap = caps.models.get(model) if model else None
    est = preflight_estimate(cap, input_tokens, output_tokens)
    pricing = cap.pricing_status if cap is not None else "unknown"
    out_tokens = output_tokens if output_tokens is not None else input_tokens
    typer.echo(f"task={task} profile={route.profile_id} provider={route.provider} model={model or '?'}")
    typer.echo(f"  input_tokens={input_tokens} output_tokens={out_tokens} pricing_status={pricing}")
    typer.echo(f"  estimated cost: ${est:.6f}")
    conn = _open_project_conn(project, root)
    try:
        status = budget_status(conn, cfg.budget, est)
    finally:
        conn.close()
    soft = ""
    if status.monthly_soft_limit_usd is not None:
        soft = f" (soft limit ${status.monthly_soft_limit_usd:.6f}{' EXCEEDED' if status.over_monthly_soft_limit else ''})"
    typer.echo(
        f"  monthly[{status.year_month}]: prior ${status.prior_spend_usd:.6f} + this "
        f"${status.this_run_usd:.6f} = ${status.projected_usd:.6f}{soft}"
    )


@llm_route_app.command("set")
def llm_route_set(
    task: str = typer.Option(..., "--task", help="Existing task_type to re-route (e.g. note_extraction)."),
    preferred: str = typer.Option(..., "--preferred", help="Profile id to route this task to."),
    fallback: Optional[str] = typer.Option(
        None,
        "--fallback",
        help="Fallback profile id. Omit to PRESERVE the current fallback; pass \"\" to clear it (no fallback).",
    ),
    project: Optional[str] = _project_option,
    root: Optional[Path] = _root_option,
) -> None:
    """Change the profile a task_type routes to (writes a thin config overlay).

    Global scope by default (``config.yaml``); with ``--project`` a project override
    (``project.yaml``) is written — a project may CHOOSE a profile but never define
    one. ``--task`` must ALREADY be a known route. Fallback handling: omit
    ``--fallback`` to carry the task's current fallback forward unchanged; pass an
    empty string (``--fallback ""``) to clear it. The write path validates the
    effective config first, so an unknown ``--preferred``/``--fallback`` profile
    raises before anything is written.
    """
    from .config.loader import write_global_config, write_project_overrides
    from .errors import SeedgraphError

    try:
        cfg = _llm_load_config(project, root)
    except SeedgraphError as exc:
        typer.echo(f"config error: {exc}")
        raise typer.Exit(2)

    if task not in cfg.llm.routes:
        known = ", ".join(sorted(cfg.llm.routes))
        typer.echo(f"unknown task '{task}' — known tasks: {known}")
        raise typer.Exit(2)

    # Carry the full route sub-dict forward so a deep-merge can actually CLEAR the
    # fallback (deep-merge overwrites a scalar/None, but cannot delete an absent key).
    if fallback is None:  # --fallback omitted → preserve current
        new_fallback = cfg.llm.routes[task].fallback_profile
    elif fallback == "":  # --fallback "" → clear
        new_fallback = None
    else:
        new_fallback = fallback
    patch = {"llm": {"routes": {task: {"preferred_profile": preferred, "fallback_profile": new_fallback}}}}

    try:
        if project is not None:
            write_project_overrides(project, patch, root=root)
            scope = f"project:{project}"
        else:
            write_global_config(patch, root)
            scope = "global"
    except SeedgraphError as exc:
        typer.echo(f"config error: {exc}")
        raise typer.Exit(2)

    typer.echo(
        f"route set: {task} -> preferred={preferred} "
        f"fallback={new_fallback or '(none)'} [scope={scope}]"
    )


llm_app.add_typer(llm_profiles_app, name="profiles")
llm_app.add_typer(llm_keys_app, name="keys")
llm_app.add_typer(llm_route_app, name="route")


# ===========================================================================
# Machine-global secret store: `secrets set|list|remove` (ADR-0001).
# One flat namespace (seedgraph/<service>/default) shared by LLM profiles and
# acquisition providers. Values go ONLY to the OS keyring — never argv, never a
# DB, never printed. Unlike `llm keys` (project-scoped REFERENCE registry),
# this group has no --project: the keyring is per-user by construction.
# ===========================================================================
secrets_app = typer.Typer(
    name="secrets",
    help="Machine-global OS-keyring secrets (seedgraph/<service>/default). Values are never shown.",
    no_args_is_help=True,
)

# Conventional service names (ADR-0001 flat namespace) shown by `secrets list`
# even when unset, so users see what CAN be configured.
_SECRET_SERVICES = ("anthropic", "core", "gemini", "openai", "openalex", "s2")


def _secret_full_name(name: str) -> str:
    """Expand a bare service alias ('openai') to 'seedgraph/openai/default'."""
    full = name if name.startswith("seedgraph/") else f"seedgraph/{name}/default"
    parts = full.split("/")
    if len(parts) < 3 or not all(parts):
        typer.echo(
            f"invalid secret name '{name}': expected 'seedgraph/<service>/default' "
            "or a bare service alias like 'openai'"
        )
        raise typer.Exit(2)
    return full


@secrets_app.command("set")
def secrets_set(
    name: str = typer.Argument(
        ..., help="Secret name: 'seedgraph/<service>/default' or a bare alias like 'openai'."
    ),
) -> None:
    """Store a secret VALUE in the OS keyring (prompted with hidden input, never argv)."""
    from .errors import ConfigError
    from .llm import secrets as _secrets

    full = _secret_full_name(name)
    value = typer.prompt(f"value for {full}", hide_input=True)
    if not value.strip():
        typer.echo("empty value — nothing stored")
        raise typer.Exit(2)
    try:
        _secrets._keyring_set(full, value)
    except ConfigError as exc:
        typer.echo(f"keyring write failed: {exc}")
        raise typer.Exit(2)
    typer.echo(f"stored {full} (value not shown)")


@secrets_app.command("list")
def secrets_list(root: Optional[Path] = _root_option) -> None:
    """List conventional + configured secret names with presence — never values."""
    from .llm import secrets as _secrets
    from .llm.profiles import is_local_profile

    names = {f"seedgraph/{svc}/default" for svc in _SECRET_SERVICES}
    try:  # include any custom key_name a configured profile references
        cfg = _llm_load_config(None, root)
        for p in cfg.llm.profiles.values():
            if not is_local_profile(p):
                names.add(p.key_name or f"seedgraph/{p.provider}/default")
    except Exception:  # noqa: BLE001 — config problems are `doctor`'s job, not list's
        pass
    typer.echo("name\tpresent")
    for full in sorted(names):
        present = _secrets._keyring_get(full) is not None
        typer.echo(f"{full}\t{'set' if present else 'absent'}")


@secrets_app.command("remove")
def secrets_remove(
    name: str = typer.Argument(
        ..., help="Secret name: 'seedgraph/<service>/default' or a bare alias like 'openai'."
    ),
) -> None:
    """Delete a secret from the OS keyring (missing entries are tolerated)."""
    from .llm import secrets as _secrets

    full = _secret_full_name(name)
    if _secrets._keyring_delete(full):
        typer.echo(f"removed {full}")
    else:
        typer.echo(f"{full} was not set")


def _stub(phase: str):
    """Build a no-arg stub command body for ``phase``."""

    def _command() -> None:
        typer.echo(f"not yet implemented ({phase})")
        raise typer.Exit(2)

    return _command


# ---------------------------------------------------------------------------
# phase_9 — eval Typer sub-app (the complete review surface; decision 81).
# `sample/verdict/report` use a raw project conn; `retrieval/leakage/answers`
# call the LIVE phase-8 harness in-process; `verify-corpus` is the REAL live
# corpus-oracle gate (network for unfetched PDFs).
# ---------------------------------------------------------------------------
eval_app = typer.Typer(
    name="eval",
    help="Evaluation harness: sample/retrieval/leakage/answers/verdict/report/verify-corpus.",
    no_args_is_help=True,
)


def _eval_project_conn(slug: str, root: Optional[Path]) -> sqlite3.Connection:
    conn = sqlite3.connect(str(project_db_path(slug, root)))
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@eval_app.command("sample")
def eval_sample(
    project: str = typer.Option(..., "--project", help="Project slug."),
    run: Optional[str] = typer.Option(None, "--run", help="Build run id."),
    seed: int = typer.Option(0, "--seed", help="RNG seed (same seed -> same draw)."),
    root: Optional[Path] = _root_option,
) -> None:
    """Draw the §11 sample (seeded) + metadata_resolution/reference_extraction
    subjects + score edge disagreements into open audit_records."""
    from .eval import audit as eval_audit, runners as eval_runners

    conn = _eval_project_conn(project, root)
    try:
        batch = eval_audit.draw_sample(conn, run, seed)
        meta = eval_audit.sample_metadata_resolution(conn, run, seed)
        refs = eval_audit.sample_reference_extraction(conn, run, seed)
        disagreements = eval_runners.score_edge_disagreements(conn, run)
    finally:
        conn.close()
    typer.echo(f"sample_batch_id {batch.sample_batch_id} seed {batch.sample_seed}")
    typer.echo(f"opened {len(batch.audit_ids)} sample audits; counts {batch.counts}")
    typer.echo(f"metadata_resolution {len(meta)}; reference_extraction {len(refs)}; "
               f"edge_disagreements {disagreements}")


@eval_app.command("verdict")
def eval_verdict(
    project: str = typer.Option(..., "--project", help="Project slug."),
    audit_id: str = typer.Option(..., "--audit-id", help="Audit row id."),
    decision: str = typer.Option(..., "--decision", help="accept|reject|edit."),
    verdict: Optional[str] = typer.Option(None, "--verdict"),
    severity: Optional[str] = typer.Option(None, "--severity"),
    problem: Optional[str] = typer.Option(None, "--problem"),
    fix: Optional[str] = typer.Option(None, "--fix"),
    edit_json: Optional[str] = typer.Option(None, "--edit-json"),
    root: Optional[Path] = _root_option,
) -> None:
    """Record a grade in audit_records (idempotent once resolved). For concept_merge
    the graph-state fix is enacted separately via review_queue (§4 boundary)."""
    from .eval import audit as eval_audit

    conn = _eval_project_conn(project, root)
    try:
        eval_audit.record_verdict(
            conn, audit_id, decision, verdict=verdict, severity=severity,
            problem=problem, fix=fix, edit_payload=edit_json,
        )
    finally:
        conn.close()
    typer.echo(f"recorded {decision} on {audit_id}")


@eval_app.command("report")
def eval_report(
    project: str = typer.Option(..., "--project", help="Project slug."),
    run: Optional[str] = typer.Option(None, "--run", help="Build run id."),
    root: Optional[Path] = _root_option,
) -> None:
    """Write runs/{run}/eval/metrics.{json,md} (D10/D4/D3 sections + open audits)."""
    from .eval import report as eval_report_mod

    conn = _eval_project_conn(project, root)
    try:
        path = eval_report_mod.build_report(conn, project, run, root=root)
    finally:
        conn.close()
    typer.echo(str(path))


@eval_app.command("retrieval")
def eval_retrieval(
    project: str = typer.Option(..., "--project", help="Project slug."),
    run: Optional[str] = typer.Option(None, "--run", help="Build run id."),
    k: int = typer.Option(10, "--k", help="Cutoff k."),
    root: Optional[Path] = _root_option,
) -> None:
    """Run retrieval_gold through the LIVE harness; write work+span-level metrics."""
    from .eval import report as eval_report_mod

    conn = _eval_project_conn(project, root)
    try:
        path = eval_report_mod.write_retrieval_metrics(conn, project, run, k=k, root=root)
    finally:
        conn.close()
    typer.echo(str(path))


@eval_app.command("leakage")
def eval_leakage(
    project: str = typer.Option(..., "--project", help="Project slug."),
    run: Optional[str] = typer.Option(None, "--run", help="Build run id."),
    root: Optional[Path] = _root_option,
) -> None:
    """Run leakage_probes through the LIVE harness; write the abstention rate."""
    from .eval import report as eval_report_mod

    conn = _eval_project_conn(project, root)
    try:
        path = eval_report_mod.write_leakage_metrics(conn, project, run, root=root)
    finally:
        conn.close()
    typer.echo(str(path))


@eval_app.command("answers")
def eval_answers(
    project: str = typer.Option(..., "--project", help="Project slug."),
    run: Optional[str] = typer.Option(None, "--run", help="Build run id."),
    root: Optional[Path] = _root_option,
) -> None:
    """Run the answer set; write faithfulness + unsupported_synthesis_rate."""
    from .eval import report as eval_report_mod

    conn = _eval_project_conn(project, root)
    try:
        path = eval_report_mod.write_answer_metrics(conn, project, run, root=root)
    finally:
        conn.close()
    typer.echo(str(path))


@eval_app.command("verify-corpus")
def eval_verify_corpus(
    project: str = typer.Option(..., "--project", help="Project slug."),
    run: Optional[str] = typer.Option(None, "--run", help="Build run id."),
    corpus: Optional[Path] = typer.Option(None, "--corpus", help="Corpus yaml path."),
    fetch: bool = typer.Option(False, "--fetch", help="Allow network PDF fetch."),
    root: Optional[Path] = _root_option,
) -> None:
    """REAL live corpus-oracle gate: promote first_corpus.yaml proposed->verified
    (fail-closed). Any unfetchable PDF / unresolved id / missing edge / unattachable
    claim leaves it proposed and BLOCKS first_corpus acceptance."""
    from .eval import runners as eval_runners

    result = eval_runners.verify_corpus(
        project, run, corpus_path=corpus, root=root, allow_fetch=fetch
    )
    if result.verified:
        typer.echo("verified: first_corpus promoted proposed -> verified")
    else:
        typer.echo("NOT verified (fail-closed); status stays proposed")
        for reason in result.failures:
            typer.echo(f"  - {reason}")
        raise typer.Exit(1)


def _stub_group(name: str, phase: str, help_text: str, verbs: list[str]) -> typer.Typer:
    """A grouped sub-app whose every verb is a phase stub (echo + exit 2)."""
    sub = typer.Typer(name=name, help=help_text, no_args_is_help=True)
    for verb in verbs:
        sub.command(verb)(_stub(phase))
    return sub


# ===========================================================================
# MCP server (design 00 / plan 01): a thin stdio consumer of the read/query
# surface. Behind the optional `[mcp]` extra — the command body lazy-imports the
# SDK-bearing `seedgraph.mcp.server` so the base install never requires it.
# ===========================================================================

mcp_app = typer.Typer(
    name="mcp",
    help="Model Context Protocol server over the read/query surface (optional 'mcp' extra).",
    no_args_is_help=True,
)


@mcp_app.command("serve")
def mcp_serve(
    root: Optional[Path] = _root_option,
    redact_private: bool = typer.Option(
        False,
        "--redact-private",
        help=(
            "Blank private full-text-derived fields (span/claim/citation text) "
            "in tool responses for non-shareable works. Best-effort valve, not "
            "the load-bearing access gate."
        ),
    ),
) -> None:
    """Run the stdio MCP server (``python -m seedgraph mcp serve``).

    Lazy-imports the SDK-bearing server module (marker-extra pattern): a missing
    ``[mcp]`` extra prints the install hint and exits 1. Pins ``$SEEDGRAPH_HOME``
    from ``--root`` (like ``serve``) only after the import succeeds, so a
    missing-extra boot has no side effects.
    """
    from . import paths

    try:
        from .mcp.server import build_server
    except ImportError:
        typer.echo(
            "the MCP server requires the optional 'mcp' extra — install it with:\n"
            "    pip install -e .[mcp]",
            err=True,
        )
        raise typer.Exit(1)

    home = paths.resolve_home(root)
    os.environ["SEEDGRAPH_HOME"] = str(home)
    server = build_server(root=home, redact_private=redact_private)
    server.run()  # FastMCP stdio loop; blocks until the host disconnects.


# ===========================================================================
# Answer artifacts: standalone AnswerTrace export (trace_plans 00 §6.1, 01 chunk 4).
# `ask` itself stays a top-level command (unchanged); `answer` groups the artifact
# utilities that operate on an already-persisted answer_id.
# ===========================================================================

answer_app = typer.Typer(
    name="answer",
    help="AnswerTrace artifact utilities (export a persisted trace to standalone HTML).",
    no_args_is_help=True,
)


@answer_app.command("trace-export")
def answer_trace_export(
    answer_id: str = typer.Argument(
        ..., help="The answer_id to export (matches {answer_id}.trace.json)."
    ),
    project: str = typer.Option(..., "--project", help="Project slug."),
    out: Optional[Path] = typer.Option(
        None, "--out",
        help="Output HTML path (default: {answer_id}.trace.html beside the trace JSON).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Export a persisted AnswerTrace as one self-contained standalone HTML file.

    Renders the SAME ``trace.html`` template the web trace view uses, with
    ``standalone=True``: the vendored subgraph JS is inlined in place of the
    ``/static/vendor/...`` script tags (retrieval-only traces — no neighborhood —
    export without that ~1.4 MB payload, same as the served page's neighborhood
    guard). The output file has no ``/static/`` or external references and opens
    with ``seedgraph serve`` down. Resolves ``answer_id`` under the project's
    ad-hoc ``answers/`` root or a ``runs/{run_id}/answers/`` location, whichever
    has the envelope+trace pair.
    """
    from .errors import SeedgraphError
    from .project.service import open_project
    from .web.ui import render_trace_standalone, resolve_trace_context, trace_answer_paths

    try:
        h = open_project(project, root=root)
    except SeedgraphError as exc:
        typer.echo(f"project error: {exc}")
        raise typer.Exit(2)

    ctx = resolve_trace_context(h, answer_id)
    if ctx is None:
        typer.echo(
            f"trace error: no trace found for answer '{answer_id}' in project '{project}'"
        )
        raise typer.Exit(2)

    html = render_trace_standalone(ctx)
    if out is not None:
        out_path = out
    else:
        _, trace_path = trace_answer_paths(h, answer_id, run_id=ctx["run_id"])
        out_path = trace_path.parent / f"{answer_id}.trace.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    typer.echo(f"wrote {out_path}")


# Grouped sub-apps (group -> verbs), mounted additively onto the root app.
app.add_typer(answer_app, name="answer")  # trace_plans 01 chunk 4: trace-export
app.add_typer(cache_app, name="cache")  # phase_1: real sub-app (NotImplementedError bodies)
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
# phase_4: real `extract notes`; phase_4b: real `extract notes-chunked` (above).
app.add_typer(extract_app, name="extract")
app.add_typer(lens_app, name="lens")  # phase_6: real grouped sub-app
app.add_typer(concepts_app, name="concepts")  # phase_7: real grouped sub-app
app.add_typer(graph_app, name="graph")  # phase_7: real grouped sub-app
app.add_typer(eval_app, name="eval")  # phase_9: real grouped sub-app
app.add_typer(llm_app, name="llm")  # Stage C: LLM backend admin (profiles/keys/usage/estimate)
app.add_typer(secrets_app, name="secrets")  # ADR-0001: machine-global OS-keyring store
app.add_typer(mcp_app, name="mcp")  # MCP server (design 00 / plan 01; optional [mcp] extra)


@app.command("search")
def search(
    query: str = typer.Argument(..., help="Exact-term search query (phrase-safe)."),
    project: str = typer.Option(..., "--project", help="Project slug."),
    work_id: Optional[str] = typer.Option(None, "--work", help="Filter by work id."),
    section_kind: Optional[str] = typer.Option(None, "--section-kind", help="Filter by section_kind."),
    limit: int = typer.Option(20, "--limit", help="Max hits."),
    root: Optional[Path] = _root_option,
) -> None:
    """Exact-term FTS5 span search over a project's evidence spans (no stemming)."""
    from sqlmodel import Session

    from .db.adapter import raw_conn
    from .fts.search import search_spans
    from .project.service import open_project

    h = open_project(project, root=root)
    with Session(h.engine) as session:
        conn = raw_conn(session)
        hits = search_spans(
            conn, query, work_id=work_id, section_kind=section_kind, limit=limit
        )
    if not hits:
        typer.echo("no matches")
        return
    typer.echo("span_id\twork_id\tsection_id\tpage\trank\tquote")
    for hit in hits:
        snippet = hit.quote_text.replace("\n", " ")
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        typer.echo(
            f"{hit.span_id}\t{hit.work_id}\t{hit.section_id or ''}\t"
            f"{hit.page_start or ''}\t{hit.rank:.3f}\t{snippet}"
        )


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="The question to answer over the project corpus."),
    project: str = typer.Option(..., "--project", help="Project slug."),
    mode: str = typer.Option("project_only", "--mode", help="project_only|allow_outside."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Retrieval-only (no LLM call)."),
    no_save: bool = typer.Option(False, "--no-save", help="Skip answer and trace files. With --no-llm, read a checkpointed corpus without writes."),
    json_out: bool = typer.Option(False, "--json", help="Emit the AnswerEnvelope as JSON."),
    save: bool = typer.Option(
        False, "--save",
        help="Deprecated no-op: saving is attempted by default; kept for compatibility.",
    ),
    limit: int = typer.Option(40, "--limit", help="Max ranked candidates (token-budgeted further)."),
    graph_depth: int = typer.Option(
        1, "--graph-depth",
        help="Citation-neighborhood hop bound for citation_search (default 1).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Answer a question over the project corpus with traceable citations (phase_8).

    Deterministic FTS5 retrieval + a single self-declaring LLM call with code-enforced
    faithfulness; ``--no-llm`` (or no resolvable profile/budget block) degrades to a
    ranked, cited evidence list with no prose (decisions 82/58/38).
    """
    import json

    from .answer.delivery import deliver
    from .answer.harness import answer as answer_fn
    from .answer.types import AnswerMode
    from .errors import SeedgraphError
    from .project.service import open_project

    def fail(code, message):
        if json_out:
            typer.echo(json.dumps({"error": {"code": code, "message": message}}))
        else:
            typer.echo(f"{code}: {message}", err=True)
        raise typer.Exit(2)

    if mode not in ("project_only", "allow_outside"):
        fail("invalid_mode", "--mode must be project_only or allow_outside")
    if limit < 1 or graph_depth < 0:
        fail("invalid_request", "--limit must be positive and --graph-depth nonnegative")
    try:
        h = open_project(project, root=root, read_only=no_llm and no_save)
    except SeedgraphError as exc:
        fail(getattr(exc, "code", "project_not_found"), str(exc))
    except (OSError, sqlite3.Error) as exc:
        fail("project_unavailable", str(exc))
    try:
        env, trace = answer_fn(
            question, h, mode=AnswerMode(mode), max_candidates=limit, no_llm=no_llm,
            graph_depth=graph_depth,
        )
    except (SeedgraphError, OSError, sqlite3.Error, ValueError) as exc:
        fail(getattr(exc, "code", "retrieval_failed"), str(exc))
    payload = deliver(env, trace, slug=project, root=h.root, no_save=no_save)
    persistence = payload["persistence"]
    if persistence["status"] == "not_saved":
        typer.echo(f"not saved: {persistence['reason']}", err=True)
    elif persistence["status"] == "saved":
        for key in ("answer_path", "trace_path"):
            typer.echo(f"saved {persistence[key]}", err=True)

    if json_out:
        typer.echo(json.dumps(payload))
        return

    # On abstention the guard guarantees ``answer_text`` is either empty or the
    # canonical not-found banner — never base-model prose. Print only the banner so no
    # model-volunteered prose can reach the user when the answer is insufficient.
    if env.insufficient_evidence:
        typer.echo("[insufficient evidence] the project corpus does not support an answer.")
    elif env.answer_text:
        typer.echo(env.answer_text)
    elif env.mode == AnswerMode.RETRIEVAL_ONLY:
        typer.echo("[retrieval-only] no prose generated; ranked evidence below.")
    from .display import derive_label

    if env.citations:
        typer.echo("\ncitations:")
        for i, c in enumerate(env.citations, start=1):
            quote = (c.quote or "").replace("\n", " ")
            if len(quote) > 100:
                quote = quote[:97] + "..."
            label = derive_label(title=c.title, year=c.year, work_id=c.work_id)
            typer.echo(f"  [{i}] {c.work_id} ({label}, {c.year or '?'}) {quote}")
    if env.recommendations:
        typer.echo("\nrecommendations:")
        for r in env.recommendations:
            label = derive_label(title=r.title, year=r.year, work_id=r.work_id)
            typer.echo(f"  - {r.work_id} ({label}): {r.reason} [{r.status}]")
    if env.warnings:
        typer.echo(f"\nwarnings: {', '.join(env.warnings)}")


from .extraction.agent_cli import agent_extract_app
from .work_read_cli import register_work_read

app.add_typer(agent_extract_app, name="agent-extract")
register_work_read(app)


if __name__ == "__main__":
    app()
