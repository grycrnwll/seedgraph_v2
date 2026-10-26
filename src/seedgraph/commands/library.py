"""`seedgraph library status|search|show|check|attention|remove|delete|merge|refetch-metadata|prune-replaced|textcheck|reconvert`
and `seedgraph library graphs list|show|rename` (design §10).

Thin adapters over :mod:`seedgraph.library` (``search``, ``files``, ``lifecycle``,
``attention``, ``graphs``). ``status``, ``show`` and ``prune-replaced`` have no deep
function yet; they compose public library reads (and, for prune, a contained folder
removal) here.
"""

from __future__ import annotations

import shutil
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer

from ..library import attention as attention_mod
from ..library import files, graphs, lifecycle
from ..library import search as search_mod
from ..library.store import Library, is_dirty, read_meta
from . import _common

library_app = typer.Typer(
    name="library",
    help="The library: every paper you hold or that is cited, shared by all context graphs.",
    no_args_is_help=True,
)
graphs_app = typer.Typer(name="graphs", help="Context graphs built on the library.", no_args_is_help=True)
library_app.add_typer(graphs_app, name="graphs")

ROOT = _common.ROOT_OPTION
JSON = _common.JSON_OPTION


class Kind(str, Enum):
    spans = "spans"
    claims = "claims"
    notes = "notes"


class Audience(str, Enum):
    agent = "agent"
    researcher = "researcher"


class AttentionStatus(str, Enum):
    open = "open"
    resolved = "resolved"
    all = "all"


def _graph_usage(lib: Library, paper_id: str) -> list[dict[str, Any]]:
    out = []
    for info in graphs.list_graphs(lib):
        with graphs.open_graph(lib, info.slug) as g:
            for p in graphs.membership(g):
                if p.paper_id == paper_id:
                    out.append({"slug": info.slug, "display_name": info.display_name, "status": p.status,
                                "is_seed": p.is_seed})
    return out


# --- status -------------------------------------------------------------------------------------

@library_app.command("status")
def status(root: Optional[Path] = ROOT, json_out: bool = JSON) -> None:
    """Counts: papers held and cited-only, context graphs, open attention items, conversion queue."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        conn = lib.conn
        by_status = dict(conn.execute("SELECT status, COUNT(*) FROM paper_status GROUP BY status").fetchall())
        meta = read_meta(conn)
        data = {
            "library": str(lib.root),
            "papers": sum(by_status.values()),
            "held": by_status.get("held", 0),
            "cited_only": by_status.get("cited_only", 0),
            "graphs": len(graphs.list_graphs(lib)),
            "attention_open": len(attention_mod.list_items(lib, status="open")),
            "conversion_queue": len(files.queued_files(lib)),
            "origin": meta.get("origin"),
            "migration_run_id": meta.get("migration_run_id"),
            "dirty": is_dirty(conn),
        }
    if json_out:
        _common.echo_json(data)
        return
    typer.echo(f"library: {data['library']} (origin {data['origin']}"
               f"{', run ' + data['migration_run_id'] if data['migration_run_id'] else ''}"
               f"{', changed since migration' if data['dirty'] else ''})")
    typer.echo(f"papers: {data['papers']} ({data['held']} held, {data['cited_only']} cited-only)")
    typer.echo(f"context graphs: {data['graphs']}")
    typer.echo(f"attention items open: {data['attention_open']}")
    typer.echo(f"conversion queue: {data['conversion_queue']}")


# --- search -------------------------------------------------------------------------------------

@library_app.command("search")
def search(
    query: str = typer.Argument(..., help="Exact-term query (one phrase; no stemming)."),
    kind: Kind = typer.Option(Kind.spans, "--kind", help="What to search."),
    graph: Optional[str] = typer.Option(None, "--graph", help="Only papers in this context graph."),
    frontier: bool = typer.Option(False, "--frontier", help="With --graph, include its frontier."),
    paper: Optional[str] = typer.Option(None, "--paper", help="Only this paper."),
    section_kind: Optional[str] = typer.Option(None, "--section-kind", help="Spans only: section kind."),
    audience: Audience = typer.Option(
        Audience.agent, "--audience",
        help="agent: current versions only, hidden notes dropped; researcher: every version, labelled."),
    limit: int = typer.Option(20, "--limit", min=1, help="Max hits."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Search spans, claims or notes in the library (optionally within one context graph)."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        hits = search_mod.search(lib, query, kind=kind.value, graph=graph, include_frontier=frontier,
                                 paper_id=paper, section_kind=section_kind, audience=audience.value,
                                 limit=limit)
    if json_out:
        _common.echo_json(hits)
        return
    if not hits:
        typer.echo("no matches")
        return
    _common.echo_table(
        ["kind", "paper_id", "id", "page", "rank", "text"],
        [[h.kind, ",".join(h.paper_ids), h.span_id or h.claim_id or h.note_id, h.page_start or "",
          f"{h.rank:.3f}", _common.one_line(h.text)
          + ("" if h.is_current else f" [{h.version_label or 'not current'}]")
          + (" [hidden from agents]" if h.hidden else "")]
         for h in hits])


# --- show ---------------------------------------------------------------------------------------

@library_app.command("show")
def show(
    paper_id: str = typer.Argument(..., help="Library paper id (lp_...)."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """One paper: metadata, identifiers, PDF files and their check, context graphs, attention items."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        conn = lib.conn
        row = conn.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        if row is None:
            _common.fail(f"paper {paper_id!r} is not in the library")
        data = dict(row)
        data["status"] = conn.execute("SELECT status FROM paper_status WHERE paper_id = ?",
                                      (paper_id,)).fetchone()[0]
        data["identifiers"] = [dict(r) for r in conn.execute(
            "SELECT id_type, id_value, source FROM identifiers WHERE paper_id = ? ORDER BY id_type, id_value",
            (paper_id,))]
        data["files"] = [dict(r) for r in conn.execute(
            "SELECT source_file_id, markdown_id, pdf_check_status, attached_at FROM paper_files "
            "WHERE paper_id = ? ORDER BY source_file_id", (paper_id,))]
        data["graphs"] = _graph_usage(lib, paper_id)
        data["attention"] = [{"item_id": i.item_id, "kind": i.kind, "detail": i.detail}
                             for i in attention_mod.list_items(lib, status="open", paper_id=paper_id)]
    if json_out:
        _common.echo_json(data)
        return
    typer.echo(f"{data['paper_id']}  [{data['status']}]")
    typer.echo(f"title: {data.get('title') or '(none)'}")
    typer.echo(f"year: {data.get('year') or ''}  venue: {data.get('venue') or ''}")
    for i in data["identifiers"]:
        typer.echo(f"{i['id_type']}: {i['id_value']}")
    for f in data["files"]:
        typer.echo(f"pdf {f['source_file_id']}: check {f['pdf_check_status']}; "
                   f"{'converted ' + f['markdown_id'] if f['markdown_id'] else 'not converted'}")
    for g in data["graphs"]:
        typer.echo(f"graph {g['slug']}: {g['status']}{' (seed)' if g['is_seed'] else ''}")
    for a in data["attention"]:
        typer.echo(f"attention {a['item_id']}: {a['kind']} {a['detail'] or ''}")


# --- check --------------------------------------------------------------------------------------

@library_app.command("check")
def check(
    paper_id: Optional[str] = typer.Argument(None, help="One paper (default: every held paper)."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Re-run the PDF check (is this PDF this paper?) and store the result."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        if paper_id is not None:
            ids = [paper_id]
        else:
            ids = [r[0] for r in lib.conn.execute(
                "SELECT paper_id FROM paper_status WHERE status = 'held' ORDER BY paper_id")]
        rows = [{"paper_id": pid, "status": c.status, "score": c.score, "evidence": c.evidence}
                for pid in ids for c in files.recheck_pdf(lib, pid)]
    if json_out:
        _common.echo_json(rows)
        return
    _common.echo_table(["paper_id", "check", "score", "evidence"],
                       [[r["paper_id"], r["status"], f"{r['score']:.2f}", _common.one_line(r["evidence"])]
                        for r in rows])


# --- attention ----------------------------------------------------------------------------------

@library_app.command("attention")
def attention(
    status_: AttentionStatus = typer.Option(AttentionStatus.open, "--status", help="Which items to list."),
    kind: Optional[str] = typer.Option(None, "--kind", help="Only this kind."),
    paper: Optional[str] = typer.Option(None, "--paper", help="Only this paper."),
    resolve: Optional[str] = typer.Option(None, "--resolve", help="Resolve this item (needs --resolution)."),
    resolution: Optional[str] = typer.Option(None, "--resolution", help="How it was resolved."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """The attention list: things the researcher should look at. --resolve closes one."""
    if resolve is not None and not (resolution and resolution.strip()):
        raise typer.BadParameter("--resolve needs --resolution TEXT", param_hint="--resolution")
    with _common.cli_errors(), _common.existing_library(root) as lib:
        if resolve is not None:
            item = attention_mod.resolve_item(lib, resolve, resolution)
            if json_out:
                _common.echo_json(item)
            else:
                typer.echo(f"resolved {item.item_id} ({item.kind}): {item.resolution}")
            return
        items = attention_mod.list_items(lib, status=None if status_ is AttentionStatus.all else status_.value,
                                         kind=kind, paper_id=paper)
    if json_out:
        _common.echo_json(items)
        return
    if not items:
        typer.echo("no attention items")
        return
    _common.echo_table(["item_id", "kind", "paper_id", "status", "detail"],
                       [[i.item_id, i.kind, i.paper_id, i.status, _common.one_line(i.detail)] for i in items])


# --- remove, delete -----------------------------------------------------------------------------

@library_app.command("remove")
def remove(
    paper_id: str = typer.Argument(..., help="Library paper id."),
    graph: str = typer.Option(..., "--graph", help="The context graph to take it out of."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Take a paper out of one context graph. The library keeps it."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        res = lifecycle.remove_from_graph(lib, graph, paper_id)
    if json_out:
        _common.echo_json(res)
    elif res.removed:
        typer.echo(f"removed {paper_id} from {graph} (was {res.status}{', seed' if res.was_seed else ''})")
    else:
        typer.echo(f"{paper_id} was not in {graph}; nothing changed")


@library_app.command("delete")
def delete(
    paper_id: str = typer.Argument(..., help="Library paper id."),
    yes: bool = typer.Option(False, "--yes", help="Carry out the delete shown in the plan."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Delete a paper from the library and every context graph. Shows the plan; --yes deletes."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        plan = lifecycle.plan_delete(lib, paper_id)
        result = lifecycle.delete_paper(lib, plan, confirm=True) if yes else None
    if json_out:
        _common.echo_json({"plan": plan, "deleted": result})
        return
    typer.echo(f"delete plan for {paper_id}:")
    if plan.graphs:
        for usage in plan.graphs:
            typer.echo(f"  still used by graph {usage.slug} ({usage.status}{', seed' if usage.is_seed else ''})")
    else:
        typer.echo("  used by no context graph")
    for name, n in sorted(plan.counts.items()):
        typer.echo(f"  {name}: {n}")
    for w in plan.warnings:
        typer.echo(f"  warning: {w}")
    if result is None:
        typer.echo("nothing deleted; re-run with --yes to delete")
    else:
        typer.echo(f"deleted {paper_id}: removed from {len(result.graphs_removed)} graph(s), "
                   f"{len(result.source_files_deleted)} PDF file(s) deleted")


@library_app.command("merge")
def merge_cmd(
    keep_id: str = typer.Argument(..., help="The paper that survives (its id stays)."),
    other_id: str = typer.Argument(..., help="The duplicate merged into it (its id goes)."),
    primary: Optional[str] = typer.Option(None, "--primary", help="Source file id of the PDF to keep as primary."),
    yes: bool = typer.Option(False, "--yes", help="Carry out the merge shown in the plan."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Merge two library papers that are the same paper. Shows the plan; --yes merges."""
    from ..library import merge

    with _common.cli_errors(), _common.existing_library(root) as lib:
        plan = merge.plan_merge(lib, keep_id, other_id, primary_file=primary)
        if yes and plan.blockers:
            _common.fail("the merge is blocked: " + "; ".join(plan.blockers))
        result = merge.merge_papers(lib, plan, confirm=True) if yes else None
    if json_out:
        _common.echo_json({"plan": merge.plan_as_dict(plan), "merged": result})
        return
    typer.echo(f"merge {other_id} into {keep_id}:")
    for t, v in plan.identifiers:
        typer.echo(f"  moves {t} {v}")
    typer.echo(f"  primary PDF: {plan.primary_file or 'none'}"
               + (f"; detached (kept, not deleted): {plan.detached_file}" if plan.detached_file else ""))
    for name, n in sorted(plan.counts.items()):
        typer.echo(f"  {name}: {n}")
    for c in plan.graphs:
        typer.echo(f"  graph {c.slug}: {c.status}{', seed' if c.is_seed else ''}")
    for b in plan.blockers:
        typer.echo(f"  blocked: {b}")
    for w in plan.warnings:
        typer.echo(f"  warning: {w}")
    if result is None:
        typer.echo("nothing merged; re-run with --yes to merge" if plan.ok else "nothing merged")
    else:
        typer.echo(f"merged {other_id} into {keep_id}")


# --- refetch-metadata, prune-replaced -------------------------------------------------------------

def refetch_chain(lib: Library):
    """The provider chain ``refetch-metadata`` uses (the real one). Tests replace this."""
    from ..library import metadata_verify

    return metadata_verify.default_chain(lib)


@library_app.command("refetch-metadata")
def refetch_metadata(
    paper_ids: Optional[list[str]] = typer.Argument(None, help="Papers to look up again."),
    malformed: bool = typer.Option(False, "--malformed",
                                   help="Also every paper whose title is missing, footnote-marked or generic."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Re-run the provider lookups: verify each paper's DOI with OpenAlex / Crossref and prefer the
    verified values; footnote marks are cleaned. Researcher-set metadata is never changed."""
    import sqlite3

    from ..library import metadata_verify

    with _common.cli_errors(), _common.existing_library(root) as lib:
        if not paper_ids and not malformed:
            _common.fail("name papers or pass --malformed")
        chain = refetch_chain(lib)
        try:
            outcomes = metadata_verify.refetch(lib, paper_ids or None, malformed=malformed, chain=chain)
        finally:
            engine = getattr(chain, "cache_engine", None)
            if isinstance(engine, sqlite3.Connection):
                engine.close()
    if json_out:
        _common.echo_json([o.as_dict() for o in outcomes])
    elif not outcomes:
        typer.echo("no papers to look up")
    else:
        _common.echo_table(["paper_id", "status", "doi", "provider", "changed", "detail"],
                           [[o.paper_id, o.status, o.doi or "", o.provider or "", ",".join(o.changed),
                             _common.one_line(o.detail)] for o in outcomes])


@library_app.command("prune-replaced")
def prune_replaced(
    yes: bool = typer.Option(False, "--yes", help="Delete the listed folders."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """List (and with --yes delete) the libraries earlier migrations moved to migration/replaced/."""
    replaced = _common.home(root) / "migration" / "replaced"
    base = replaced.resolve()
    entries = []
    if replaced.is_dir():
        for child in sorted(replaced.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.is_symlink() or child.resolve().parent != base:
                continue
            size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
            entries.append({"name": child.name, "path": str(child), "bytes": size})
    if yes:
        for e in entries:
            shutil.rmtree(e["path"])
    if json_out:
        _common.echo_json({"replaced": entries, "deleted": yes})
        return
    if not entries:
        typer.echo(f"nothing in {replaced}")
        return
    for e in entries:
        typer.echo(f"{e['name']}\t{e['bytes']} bytes")
    typer.echo(f"deleted {len(entries)} folder(s)" if yes else "nothing deleted; re-run with --yes to delete")


# --- textcheck, reconvert ------------------------------------------------------------------------

@library_app.command("textcheck")
def textcheck(
    paper_ids: Optional[list[str]] = typer.Argument(None, help="Papers to check (default: every held paper)."),
    golden: bool = typer.Option(False, "--golden",
                                help="Check the golden set (<home>/golden_set.json): verdict vs category."),
    force: bool = typer.Option(False, "--force", help="Assess again instead of reading the cached verdict."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """The text-quality gate's verdict (pass, garbled, no_text) per paper, cached for batches.
    With --golden, exit 1 when any verdict disagrees with its golden-set category."""
    from ..golden.errors import GoldenError
    from ..golden.manifest import load_manifest
    from ..library import textcheck as textcheck_mod

    with _common.cli_errors(), _common.existing_library(root) as lib:
        if golden:
            try:
                manifest = load_manifest(_common.home(root))
            except GoldenError as exc:
                _common.fail(str(exc))
            rows: list[Any] = textcheck_mod.check_golden(lib, manifest, force=force)
        else:
            rows = textcheck_mod.check_papers(lib, paper_ids or None, force=force)
    if json_out:
        _common.echo_json(rows)
    elif golden:
        _common.echo_table(["golden_id", "category", "verdict", "expected", "agrees", "summary"],
                           [[r.golden_id, r.category, r.status, r.expected, "yes" if r.agrees else "NO",
                             _common.one_line(r.summary)] for r in rows])
        bad = [r.golden_id for r in rows if not r.agrees]
        typer.echo(f"{len(bad)} disagree: {', '.join(bad)}" if bad else "every verdict agrees with its category")
    else:
        _common.echo_table(["paper_id", "markdown_id", "verdict", "cached", "summary"],
                           [[r.paper_id, r.markdown_id or "", r.status, "yes" if r.cached else "",
                             _common.one_line(r.summary)] for r in rows])
    if golden and any(not r.agrees for r in rows):
        raise typer.Exit(_common.EXIT_FAILED)


def reconvert_backend():
    """The converter for ``reconvert``: the shared warm pool (one backend for every paper),
    or the fake converter when ``SEEDGRAPH_FAKE_MARKER`` is set. Tests replace this."""
    from ..cache.convert import _default_backend

    return _default_backend()


@library_app.command("reconvert")
def reconvert(
    paper_ids: list[str] = typer.Argument(..., help="Papers whose PDF to convert again."),
    ocr: bool = typer.Option(False, "--ocr", help="Force OCR (required: the only re-conversion mode)."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Re-convert papers' PDFs with forced OCR. A paper switches to the new text only when it
    passes the gate; its notes' quotes are then re-matched (design §5.5)."""
    from ..library import reconvert as reconvert_mod

    if not ocr:
        raise typer.BadParameter("only forced-OCR re-conversion exists; pass --ocr", param_hint="--ocr")
    with _common.cli_errors(), _common.existing_library(root) as lib:
        backend = reconvert_backend()
        results = [reconvert_mod.reconvert_ocr(lib, pid, backend=backend) for pid in paper_ids]
    if json_out:
        _common.echo_json([r.as_dict() for r in results])
    else:
        _common.echo_table(["paper_id", "status", "old_markdown_id", "new_markdown_id", "detail"],
                           [[r.paper_id, r.status, r.old_markdown_id or "", r.new_markdown_id or "",
                             _common.one_line(r.detail)] for r in results])
    if any(r.status == "conversion_failed" for r in results):
        raise typer.Exit(_common.EXIT_FAILED)


# --- pagecheck ----------------------------------------------------------------------------------

class PagecheckSet(str, Enum):
    proto5 = "proto5"
    proto31 = "proto31"
    current = "current"


class PagecheckMode(str, Enum):
    prototype = "prototype"
    current = "current"


@library_app.command("pagecheck")
def pagecheck(
    set_name: PagecheckSet = typer.Option(..., "--set", help="Which papers: proto5, proto31 or current."),
    mode: PagecheckMode = typer.Option(PagecheckMode.current, "--mode",
                                       help="prototype: the prototype's own claims; current: what you see."),
    verify: bool = typer.Option(True, "--verify/--no-verify",
                                help="Check each quote against the PDF's text layer (pypdfium2)."),
    report: bool = typer.Option(True, "--report/--no-report",
                                help="Write <home>/ui/pagecheck/<timestamp>/report.{md,json}."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """Measure quote → PDF page highlighting (design §3.4): page assigned, passage verified,
    page-only and page-wrong. Reads the library read-only. Exit 1 on a page-wrong quote or,
    in prototype mode, when the prototype's figures in <home>/ui/highlight_expected.json are
    not met."""
    from ..library import pagecheck as pagecheck_mod
    from ..library.store import library_dir

    home = _common.home(root)
    with _common.cli_errors():
        with pagecheck_mod.open_readonly(library_dir(root)) as lib:
            result = pagecheck_mod.run(lib, home, set_name=set_name.value, mode=mode.value, verify=verify)
        out = pagecheck_mod.write_report(result, home) if report else None
    if json_out:
        data = result.as_dict()
        data.pop("quotes")
        data["report_dir"] = str(out) if out else None
        _common.echo_json(data)
    else:
        t = result.totals
        _common.echo_table(["measure", "count"], [
            ["papers", t["papers"]], ["quotes", t["quotes"]], ["page assigned", t["page_assigned"]],
            ["marker / reanchored / none", f"{t['marker']} / {t['reanchored']} / {t['none']}"],
            ["passage verified", t["verified"]], ["page-only", t["page_only"]],
            ["page wrong", t["page_wrong"]], ["not checked", t["not_checked"]],
            ["papers without page markers", t["papers_without_markers"]]])
        for p in result.no_marker_papers:
            typer.echo(f"no page markers: {p['paper_id']} (converter {p['converter_version']}); "
                       "re-convert to get pages")
        if result.unresolved:
            typer.echo(f"{len(result.unresolved)} set entries did not resolve: {', '.join(result.unresolved[:5])}")
        typer.echo(f"verdict: {result.verdict}" + (f" ({'; '.join(result.reasons)})" if result.reasons else ""))
        if out:
            typer.echo(f"report: {out}")
    if result.verdict != "pass":
        raise typer.Exit(_common.EXIT_FAILED)


# --- graphs -------------------------------------------------------------------------------------

@graphs_app.command("list")
def graphs_list(root: Optional[Path] = ROOT, json_out: bool = JSON) -> None:
    """Every context graph, by slug."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        infos = graphs.list_graphs(lib)
    if json_out:
        _common.echo_json(infos)
        return
    if not infos:
        typer.echo("no context graphs")
        return
    _common.echo_table(["slug", "name", "created_at", "origin"],
                       [[i.slug, i.display_name, i.created_at, i.origin] for i in infos])


@graphs_app.command("show")
def graphs_show(
    slug: str = typer.Argument(..., help="Context graph slug."),
    root: Optional[Path] = ROOT,
    json_out: bool = JSON,
) -> None:
    """One context graph: name, origin and its papers by status."""
    with _common.cli_errors(), _common.existing_library(root) as lib, graphs.open_graph(lib, slug) as g:
        info = graphs.graph_info(g)
        papers = graphs.membership(g)
    counts = {s: sum(1 for p in papers if p.status == s) for s in graphs.STATUSES}
    data = {**_common.to_jsonable(info), "counts": counts, "seeds": sum(1 for p in papers if p.is_seed),
            "papers": papers}
    if json_out:
        _common.echo_json(data)
        return
    typer.echo(f"{info.slug}: {info.display_name} (origin {info.origin}, created {info.created_at})")
    typer.echo("papers: " + ", ".join(f"{k}={v}" for k, v in counts.items()) + f"; seeds={data['seeds']}")
    _common.echo_table(["paper_id", "status", "seed", "added_at"],
                       [[p.paper_id, p.status, "yes" if p.is_seed else "", p.added_at] for p in papers])


@graphs_app.command("rename")
def graphs_rename(
    slug: str = typer.Argument(..., help="Context graph slug (fixed)."),
    name: str = typer.Argument(..., help="New display name."),
    root: Optional[Path] = ROOT,
) -> None:
    """Change a context graph's display name. The slug stays."""
    with _common.cli_errors(), _common.existing_library(root) as lib:
        graphs.rename_graph(lib, slug, name)
    typer.echo(f"{slug}: renamed to {name!r}")

