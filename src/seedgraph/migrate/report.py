"""The migration report (design §6.12): ``report.md``, ``report.json``, ``attention.md``.

Written into ``<v3 home>/migration/runs/<run_id>/`` from a
:class:`~seedgraph.migrate.MigrationResult`. It must account for **every v2 row**:

* the plan's ledger gives each v2 row a disposition (carried, merged into another
  row, not carried with a reason, or rebuilt);
* :func:`migration_hook` (run as the migration's ``stage_hook``) also records, from
  the snapshot itself, how many rows each v2 table had and which rows have no
  ledger entry (``v2_rows.json``), so a row the plan forgot cannot pass silently;
* every saved answer must have an exported page.

Anything unaccounted is listed under ``problems``. :func:`write_report` always
writes the three files first and then raises :class:`ReportError`, so the failure is
loud and inspectable.

``report.json`` has a flat ``totals`` section (dotted keys, integers) that the
acceptance check (:mod:`.accept`) compares against ``acceptance/expected.json``.

:func:`run_and_report` is the one call behind ``seedgraph migrate run``.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seedgraph.library.attention import item_id_for

from . import answers as _answers
from .plan import row_key

if TYPE_CHECKING:  # pragma: no cover
    from . import MigrationResult, StageContext

V2_ROWS_FILE = "v2_rows.json"
REPORT_MD, REPORT_JSON, ATTENTION_MD = "report.md", "report.json", "attention.md"

#: Plain-English headings for the attention kinds the migration raises.
ATTENTION_KINDS: dict[str, tuple[str, str]] = {
    "shared_pdf": ("Different papers sharing one PDF",
                   "Two or more papers point at the same PDF. Notes are hidden on a side whose title is not in "
                   "the PDF. Decide which paper the PDF belongs to."),
    "pdf_title_miss": ("Title not found in the paper's own PDF (notes hidden)",
                       "The paper's title and DOI were not found in its PDF or text. The PDF may be a different "
                       "paper. Its notes are kept but hidden from agents until you confirm or fix the paper."),
    "malformed_title": ("Malformed titles",
                        "The title looks like a file name or a hash, and no cached metadata could repair it. "
                        "Run `seedgraph library refetch-metadata --malformed` after acceptance."),
    "identity_check": ("Check these: papers without a strong id",
                       "Papers with no DOI/OpenAlex/arXiv id that could not be merged safely (no match, or "
                       "several matches). They stay separate until you decide."),
    "metadata_needed": ("Cached PDFs with no metadata",
                        "A cached PDF that no v2 project held. It is in the library with an unknown title."),
    "test_converter_markdown": ("Text made by a test converter",
                                "Markdown made by a test/fake converter is carried but not used; the paper is "
                                "queued for real conversion."),
    "concept_contradiction": ("Contradicting concept decisions",
                              "You decided 'merge' in one v2 project and 'never merge' in another for the same "
                              "pair. Both decisions are kept; pick one."),
}

_OUTCOME_WORDS = {"graph": "became a context graph", "folded": "folded into a graph", "library_only":
                  "papers join the library, no graph", "not_carried": "not carried", "cache": "the v2 cache"}

OUTCOMES = ("carried", "merged", "rebuilt", "not_carried", "unaccounted")


class ReportError(RuntimeError):
    """The report could not account for every item (the files are still written).

    ``files`` names the written files; ``result`` is set by :func:`run_and_report`."""

    def __init__(self, message: str, *, files: ReportFiles | None = None) -> None:
        super().__init__(message)
        self.files = files
        self.result: MigrationResult | None = None


@dataclass(frozen=True)
class ReportFiles:
    report_md: Path
    report_json: Path
    attention_md: Path
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass(frozen=True)
class RunOutcome:
    """What ``seedgraph migrate run`` gets back."""

    result: MigrationResult
    files: ReportFiles

    @property
    def ok(self) -> bool:
        return self.result.ok and self.files.ok


# --- the stage hook -----------------------------------------------------------------------

def count_v2_rows(ctx: StageContext) -> dict[str, Any]:
    """Per v2 DB and table: row count, distinct ledger keys, and rows with no ledger entry."""
    snap, ledger = ctx.snapshot, ctx.plan.ledger
    rows: dict[str, dict[str, dict[str, int]]] = {}
    missing: list[list[str]] = []
    for source, db in [("cache", snap.cache), *snap.projects.items()]:
        per: dict[str, dict[str, int]] = {}
        for table, trows in sorted(db.tables.items()):
            pk = db.pk.get(table, ())
            keys = {row_key(pk, r) for r in trows}
            per[table] = {"rows": len(trows), "keys": len(keys)}
            for k in sorted(keys):
                if ledger.get(source, table, k) is None:
                    missing.append([source, table, k])
        rows[source] = per
    return {"run_id": ctx.run_id, "rows": rows, "missing_from_ledger": missing,
            "answer_files": {s: list(v) for s, v in sorted(snap.answer_files.items())},
            "run_answer_files": {s: list(v) for s, v in sorted(snap.run_answer_files.items())}}


def migration_hook(ctx: StageContext) -> None:
    """The ``stage_hook`` for every real run: export v2 answers into staging and record
    the v2 row counts (``runs/<run_id>/v2_rows.json``) the report reconciles against."""
    _answers.stage_hook(ctx)
    data = count_v2_rows(ctx)
    _write(ctx.run_dir / V2_ROWS_FILE, json.dumps(data, sort_keys=True, indent=1, ensure_ascii=False) + "\n")


def run_and_report(v2_home: Path | str, v3_home: Path | str, **run_kwargs: Any) -> RunOutcome:
    """Run the migration with :func:`migration_hook` and write the report.

    ``run_kwargs`` go to :func:`seedgraph.migrate.run_migration` (``graph_names``,
    ``replace_dirty``, ``swap``, ``run_id``, ``now``, ``disk_free``,
    ``process_lister``). Raises :class:`~seedgraph.migrate.MigrationRefused` on a
    refused preflight and :class:`ReportError` when the report cannot account for
    every item (after writing it).
    """
    from . import run_migration
    if "stage_hook" in run_kwargs:
        raise TypeError("run_and_report wires its own stage_hook")
    result = run_migration(v2_home, v3_home, stage_hook=migration_hook, **run_kwargs)
    try:
        return RunOutcome(result, write_report(result))
    except ReportError as exc:
        exc.result = result
        raise


# --- building the report ------------------------------------------------------------------

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def answers_dir(result: MigrationResult) -> Path:
    """Where this run's ``v2_answers`` folder is now (live after a swap, else staging)."""
    return (result.v3_home if result.swapped else result.staging_dir) / _answers.FOLDER


def _exported(folder: Path) -> list[dict[str, str]]:
    out = []
    if folder.is_dir():
        for p in sorted(folder.glob("*/*.md")):
            out.append({"slug": p.parent.name, "answer_id": p.stem,
                        "path": f"{_answers.FOLDER}/{p.parent.name}/{p.name}"})
    return out


def _reconcile(ledger: Any, v2_rows: Mapping[str, Any] | None, problems: list[str]) -> dict[str, Any]:
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for (source, table, _), d in ledger.entries.items():
        r = by.setdefault((source, table), {"carried": 0, "merged": 0, "rebuilt": 0, "not_carried": Counter(),
                                            "unaccounted": 0})
        if d.outcome == "not_carried":
            r["not_carried"][d.reason or "no_reason"] += 1
        elif d.outcome in r:
            r[d.outcome] += 1
        else:
            r["unaccounted"] += 1
    unaccounted = sorted(k for k, d in ledger.entries.items() if d.outcome not in OUTCOMES[:4])
    for k in unaccounted:
        problems.append(f"unaccounted v2 row: {k[0]} {k[1]} {k[2]}")
    counts = (v2_rows or {}).get("rows", {})
    if v2_rows is None:
        problems.append(f"v2 row counts were not recorded ({V2_ROWS_FILE} missing): run the migration through "
                        "seedgraph.migrate.report.run_and_report")
    else:
        for source, table, key in v2_rows.get("missing_from_ledger", []):
            if ledger.get(source, table, key) is None:
                problems.append(f"v2 row with no disposition: {source} {table} {key}")
    out: dict[str, dict[str, Any]] = defaultdict(dict)
    for source, table in sorted(set(by) | {(s, t) for s, ts in counts.items() for t in ts}):
        r = by.get((source, table)) or {"carried": 0, "merged": 0, "rebuilt": 0, "not_carried": Counter(),
                                        "unaccounted": 0}
        accounted = r["carried"] + r["merged"] + r["rebuilt"] + sum(r["not_carried"].values()) + r["unaccounted"]
        c = counts.get(source, {}).get(table)
        if c is not None and c["keys"] != accounted:
            gap = c["keys"] - accounted
            if gap > 0:
                r["unaccounted"] += gap
                accounted += gap
                problems.append(f"{source} {table}: {gap} v2 rows have no disposition")
            else:
                problems.append(f"{source} {table}: the ledger names {-gap} rows the v2 table does not have")
        out[source][table] = {
            "v2_rows": c["keys"] if c is not None else accounted,
            "v2_rows_raw": c["rows"] if c is not None else accounted,
            "carried": r["carried"], "merged": r["merged"], "rebuilt": r["rebuilt"],
            "not_carried": dict(sorted(r["not_carried"].items())), "unaccounted": r["unaccounted"],
        }
    return dict(out)


def build_report(result: MigrationResult, *, v2_rows: Mapping[str, Any] | None = None,
                 answers_folder: Path | None = None) -> dict[str, Any]:
    """The ``report.json`` document (pure apart from listing the exported answers)."""
    plan = result.plan
    rp = plan.report
    problems: list[str] = []
    titles = {p["paper_id"]: p["title"] for p in plan.rows("papers")}
    reconciliation = _reconcile(plan.ledger, v2_rows, problems)

    # graphs ----------------------------------------------------------------------
    graphs: dict[str, Any] = {}
    for g in plan.graphs:
        st = Counter(r["status"] for r in g.papers)
        graphs[g.slug] = {
            "source_project": g.source_project, "display_name": g.display_name,
            "member": st.get("member", 0), "frontier": st.get("frontier", 0), "excluded": st.get("excluded", 0),
            "seeds": sum(1 for r in g.papers if r["is_seed"]),
            "folded_in": rp.get("folded_in", {}).get(g.slug, []),
            "notes": rp.get("graph_notes", {}).get(g.slug, []),
            "counts_in_db": result.graph_counts.get(g.slug, {}),
        }

    # merges ----------------------------------------------------------------------
    by_member = defaultdict(list)
    for m in plan.rows("paper_merge_members"):
        by_member[m["paper_id"]].append(m)
    merges = []
    for m in rp.get("merged_papers", []):
        rows = by_member.get(m["paper_id"], [])
        merges.append({"paper_id": m["paper_id"], "title": m["title"], "members": m["members"],
                       "joined_by": sorted({r["joined_by_key_type"] for r in rows}), "splittable": True})
    ident = rp.get("identity", {})

    # attention -------------------------------------------------------------------
    items = []
    for a in plan.attention:
        items.append({"item_id": item_id_for(a.kind, a.subject), "kind": a.kind, "paper_id": a.paper_id,
                      "title": titles.get(a.paper_id) if a.paper_id else None, "subject": dict(a.subject),
                      "detail": a.detail, "hides_notes": bool(a.hides_notes)})
    by_kind = Counter(a["kind"] for a in items)

    # concepts --------------------------------------------------------------------
    ccounts = Counter(f"{r['kind']}:{r['decided_by']}" for r in plan.rows("concept_decisions"))

    # answers ---------------------------------------------------------------------
    folder = answers_folder if answers_folder is not None else answers_dir(result)
    exported = _exported(folder)
    saved = (v2_rows or {}).get("answer_files") or rp.get("answers", {}).get("saved", {})
    run_scoped = (v2_rows or {}).get("run_answer_files") or rp.get("answers", {}).get("run_scoped_not_carried", {})
    projects = rp.get("projects", {})
    have = {(e["slug"], e["answer_id"]) for e in exported}
    missing_answers = []
    for slug, paths in sorted(saved.items()):
        if projects.get(slug, "not_carried") == "not_carried":
            continue
        for rel in paths:
            aid = Path(rel).name[: -len(".json")]
            if (slug, aid) not in have:
                missing_answers.append(rel)
                problems.append(f"saved v2 answer not exported: {rel}")

    citations = {
        "review_decisions": len(plan.rows("citation_review_decisions")),
        "collapsed_onto_existing": rp.get("citation_decisions_collapsed", []),
        "unmapped_decisions": rp.get("unmapped_decisions", []),
        "reference_lists": len(rp.get("reference_lists", [])),
        "reference_entries_unmapped": rp.get("reference_entries_unmapped", 0),
        "reference_lists_other_markdown": rp.get("reference_lists_other_markdown", []),
        "edges_dropped": rp.get("citation_edges_dropped", {}),
        "citation_links": len(plan.rows("citation_links")),
        "duplicate_decisions": rp.get("duplicate_decisions", []),
    }
    repairs = {"titles": rp.get("titles_repaired", []), "relinks": rp.get("relinks", []),
               "secondary_pdfs": rp.get("secondary_pdfs", []),
               "identifier_conflicts": rp.get("identifier_conflicts", [])}

    total_unacc = sum(r["unaccounted"] for ts in reconciliation.values() for r in ts.values())
    totals: dict[str, int] = {}
    for t, n in sorted(result.counts.items()):
        totals[f"library.{t}"] = int(n)
    for slug, g in graphs.items():
        for k in ("member", "frontier", "excluded", "seeds"):
            totals[f"graph.{slug}.{k}"] = g[k]
        totals[f"graph.{slug}.folded_in"] = len(g["folded_in"])
    for source, ts in reconciliation.items():
        for table, r in ts.items():
            totals[f"v2.{source}.{table}.rows"] = r["v2_rows"]
            for k in ("carried", "merged", "rebuilt"):
                totals[f"v2.{source}.{table}.{k}"] = r[k]
            totals[f"v2.{source}.{table}.not_carried"] = sum(r["not_carried"].values())
    totals["attention.total"] = len(items)
    totals["attention.hides_notes"] = sum(1 for a in items if a["hides_notes"])
    for k, n in sorted(by_kind.items()):
        totals[f"attention.{k}"] = n
    for k, n in sorted(ccounts.items()):
        totals[f"concepts.{k.replace(':', '.')}"] = n
    totals.update({
        "merges.papers": len(merges), "merges.blocked": len(ident.get("blocked", [])),
        "current_elsewhere": len(rp.get("current_elsewhere", [])),
        "repairs.titles": len(repairs["titles"]), "repairs.relinks": len(repairs["relinks"]),
        "not_converted": len(rp.get("not_converted", [])),
        "labelled_versions": len(rp.get("labelled_versions", [])),
        "concepts.contradictions": len(rp.get("concept_contradictions", [])),
        "citations.review_decisions": citations["review_decisions"],
        "citations.unmapped_decisions": len(citations["unmapped_decisions"]),
        "citations.edges_dropped": sum(citations["edges_dropped"].values()),
        "answers.exported": len(exported),
        "answers.run_scoped_not_carried": sum(len(v) for v in run_scoped.values()),
        "errors": len(result.errors), "unaccounted": total_unacc,
    })
    totals = dict(sorted(totals.items()))

    return {
        "run_id": result.run_id, "ok": result.ok and not problems, "swapped": result.swapped,
        "snapshot_time": result.snapshot_time, "v2_home": str(result.v2_home), "v3_home": str(result.v3_home),
        "dump_sha256": result.dump_sha256,
        "v2_guard": {"status": result.guard.status, "diffs": list(result.guard.diffs)},
        "errors": list(result.errors), "warnings": list(result.warnings), "problems": problems,
        "projects": projects, "totals": totals, "reconciliation": reconciliation, "graphs": graphs,
        "merges": {"papers": merges, "rule_merges": ident.get("rule_merges", []),
                   "blocked": ident.get("blocked", []), "skipped_never_merge": ident.get("skipped_never_merge", [])},
        "current_elsewhere": rp.get("current_elsewhere", []),
        "labelled_versions": rp.get("labelled_versions", []),
        "repairs": repairs,
        "not_converted": rp.get("not_converted", []),
        "test_converter_papers": rp.get("test_converter_papers", []),
        "title_check": {"weak": rp.get("title_check_weak", []), "miss": rp.get("title_check_miss", [])},
        "attention": {"by_kind": dict(sorted(by_kind.items())), "items": items},
        "concepts": {"counts": dict(sorted(ccounts.items())), "contradictions": rp.get("concept_contradictions", []),
                     "agent_overridden": rp.get("concept_agent_overridden", [])},
        "citations": citations,
        "answers": {"exported": exported, "missing": missing_answers,
                    "run_scoped_not_carried": sorted(p for v in run_scoped.values() for p in v)},
    }


# --- rendering ----------------------------------------------------------------------------

def _t(title: str | None, pid: str | None = None) -> str:
    if title:
        return title
    return f"(title unknown) `{pid}`" if pid else "(title unknown)"


def _table(head: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(v: Any) -> str:
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return out


def render_report_md(d: Mapping[str, Any]) -> str:
    L: list[str] = [f"# Migration report: {d['run_id']}", ""]
    status = "PASSED" if d["ok"] else "FAILED"
    L += [f"**Result: {status}.** {'The new library was swapped in.' if d['swapped'] else 'The new library was built in staging only (not swapped in).'}",
          "", f"- v2 home (read only): `{d['v2_home']}`", f"- Snapshot taken: {d['snapshot_time']}", ""]
    if d["problems"]:
        L += ["## Problems: items not accounted for", "",
              "The report could not account for these v2 items. This is a migration bug; do not accept the run.", ""]
        L += [f"- {p}" for p in d["problems"]] + [""]
    if d["errors"]:
        L += ["## Errors", ""] + [f"- {e}" for e in d["errors"]] + [""]
    if d["warnings"]:
        L += ["## Warnings", ""] + [f"- {w}" for w in d["warnings"]] + [""]

    L += ["## Counts: every v2 row, project by project", "",
          "Each v2 table's rows are split into: carried over; merged into another carried row (a duplicate); "
          "rebuilt (search indexes); or not carried, with the reason.", ""]
    L += ["Empty tables are left out; `report.json` has every table.", ""]
    for source, tables in d["reconciliation"].items():
        outcome = d["projects"].get(source, "cache" if source == "cache" else "")
        L += [f"### {source} ({_OUTCOME_WORDS.get(outcome.split(':')[0], outcome)})" if outcome
              else f"### {source}", ""]
        total = sum(r["v2_rows"] for r in tables.values())
        if outcome == "not_carried" and not any(r["carried"] or r["merged"] or r["unaccounted"]
                                                for r in tables.values()):
            L += [f"Nothing carried ({total} rows in {sum(1 for r in tables.values() if r['v2_rows'])} tables).", ""]
            continue
        rows, rebuilt = [], 0
        for table, r in tables.items():
            if not r["v2_rows"]:
                continue
            if r["rebuilt"] == r["v2_rows"] and not r["unaccounted"]:
                rebuilt += r["rebuilt"]
                continue
            nc = ", ".join(f"{k.replace('_', ' ')}: {n}" for k, n in r["not_carried"].items())
            rows.append([table, r["v2_rows"], r["carried"], r["merged"], r["rebuilt"] or "", nc or "",
                         r["unaccounted"] or ""])
        if rebuilt:
            rows.append(["search indexes (all tables)", rebuilt, "", "", rebuilt, "", ""])
        L += _table(["table", "v2 rows", "carried", "merged", "rebuilt", "not carried (reason)", "unaccounted"], rows)
        L.append("")

    L += ["## Context graphs", ""]
    for slug, g in d["graphs"].items():
        L += [f"### {g['display_name']} (`{slug}`, from v2 project {g['source_project']})", "",
              f"{g['member']} member papers ({g['seeds']} seeds), {g['frontier']} frontier (cited, no text), "
              f"{g['excluded']} excluded.", ""]
        if g["folded_in"]:
            L += [f"Papers folded in from test_proj ({len(g['folded_in'])}): held in test_proj but not already "
                  f"in this graph, so they joined it as members.", ""]
            L += [f"- {_t(f['title'], f['paper_id'])} (`{f['paper_id']}`)" for f in g["folded_in"]] + [""]
        for n in g["notes"]:
            L.append(f"- Note: `{n['paper_id']}`: {n['note']}")
        if g["notes"]:
            L.append("")

    merges = d["merges"]["papers"]
    L += ["## Papers merged", "",
          f"{len(merges)} library papers were made from more than one v2 record (same DOI, OpenAlex id, arXiv id, "
          "same PDF, or same title and year for papers without an id). Every merge is recorded member by member, "
          "so any of them can be split again.", ""]
    if merges:
        L += _table(["paper", "title", "joined by", "v2 records"],
                    [[f"`{m['paper_id']}`", _t(m["title"]), ", ".join(m["joined_by"]),
                      "; ".join(f"{p}/{w}" for p, w in m["members"])] for m in merges])
        L.append("")
    if d["merges"]["blocked"]:
        L += ["Merges blocked because the records' ids disagree:", ""]
        L += [f"- {b['work_a'][0]}/{b['work_a'][1]} and {b['work_b'][0]}/{b['work_b'][1]} share "
              f"{b['key_type']} but differ on {', '.join(b['conflict_types'])}" for b in d["merges"]["blocked"]]
        L.append("")
    if d["merges"]["skipped_never_merge"]:
        L += ["Merges skipped because you said 'not duplicates' in v2:", ""]
        L += [f"- {m['work_a'][0]}/{m['work_a'][1]} and {m['work_b'][0]}/{m['work_b'][1]} ({m['key_type']})"
              for m in d["merges"]["skipped_never_merge"]] + [""]

    ce = d["current_elsewhere"]
    L += ["## Papers whose current note comes from another project", "",
          "When a paper was held in several v2 projects, the most recent note is now current everywhere. For "
          "these graph papers that note came from a different project than the graph's own. You can switch the "
          "current version by hand.", ""]
    L += ([f"- `{c['paper_id']}` in `{c['graph']}`: note `{c['note_id']}` from {c['from_project']}"
           f"{' (folded-in paper)' if c.get('folded_in') else ''}" for c in ce] or ["None."]) + [""]

    rep = d["repairs"]
    L += ["## Repairs done", ""]
    if rep["titles"]:
        L += ["Malformed titles repaired from cached metadata:", ""]
        L += [f"- `{t['paper_id']}`: {t['old']!r} -> {t['new']!r} ({t['provider']})" for t in rep["titles"]] + [""]
    if rep["relinks"]:
        L += ["Papers that pointed at another PDF's text:", ""]
        for r in rep["relinks"]:
            what = {"relinked": "re-linked to its own converted text",
                    "relink_failed": "its own PDF has no usable text yet; queued for conversion",
                    "pointer_missing": "its text pointer was missing; linked to its own text"}.get(r["repair"], r["repair"])
            L.append(f"- `{r['paper_id']}`: {what}")
        L.append("")
    if d["labelled_versions"]:
        L += [f"{len(d['labelled_versions'])} note versions were made on the wrong text; they are kept as earlier "
              "versions, labelled, and hidden from agents.", ""]
    if not (rep["titles"] or rep["relinks"]):
        L += ["None.", ""]

    nc = d["not_converted"]
    L += ["## Not converted", "", "These papers have a PDF but no usable text. Run `seedgraph convert --queued`.", ""]
    L += ([f"- {_t(r['title'], r['paper_id'])} (`{r['paper_id']}`)" for r in nc] or ["None."]) + [""]

    att = d["attention"]
    L += ["## Needs your attention", "",
          f"{len(att['items'])} items; see `attention.md` for the full list.", ""]
    L += _table(["kind", "items"], [[ATTENTION_KINDS.get(k, (k, ""))[0], n] for k, n in att["by_kind"].items()])
    L.append("")

    c = d["concepts"]
    cc = c["counts"]
    L += ["## Concept decisions carried", "",
          f"- Your merge decisions: {cc.get('merge:researcher', 0)}",
          f"- Your 'never merge' decisions: {cc.get('never_merge:researcher', 0)}",
          f"- Merges made by an agent (bulk pass), applied and shown in the undo list: {cc.get('merge:agent', 0)}",
          f"- Agent merges overridden by your own decision: {len(c['agent_overridden'])}",
          f"- Contradictions (kept, on the attention list): {len(c['contradictions'])}", ""]
    for x in c["contradictions"]:
        L.append(f"  - {x['label_a']!r} / {x['label_b']!r}: newest is {x['newest_kind']} ({x['newest_project']})")
    if c["contradictions"]:
        L.append("")

    ci = d["citations"]
    L += ["## Citation decisions carried", "",
          f"- Rejected citation matches carried: {ci['review_decisions']}",
          f"- Decisions from another project's reference list mapped onto the kept list: "
          f"{len(ci['collapsed_onto_existing'])}",
          f"- Decisions that could not be mapped (listed below): {len(ci['unmapped_decisions'])}",
          f"- Reference entries of other projects' lists with no counterpart: {ci['reference_entries_unmapped']}",
          f"- Citation links carried: {ci['citation_links']}; dropped: "
          + (", ".join(f"{k.replace('_', ' ')} {n}" for k, n in ci["edges_dropped"].items()) or "none"),
          f"- Duplicate decisions: {len(ci['duplicate_decisions'])}", ""]
    for u in ci["unmapped_decisions"]:
        L.append(f"  - {u['project']} item `{u['item_id']}`" + (f": {u['raw']}" if u.get("raw") else ""))
    if ci["unmapped_decisions"]:
        L.append("")

    a = d["answers"]
    L += ["## v2 answers exported", "",
          f"{len(a['exported'])} saved answers were exported as markdown into `v2_answers/` (see its `index.md`). "
          f"{len(a['run_scoped_not_carried'])} answers saved inside run folders were not carried.", ""]
    if a["missing"]:
        L += ["Not exported (a problem):", ""] + [f"- {m}" for m in a["missing"]] + [""]

    g = d["v2_guard"]
    verdict = {"pass": "Passed: the v2 home is byte-for-byte unchanged.",
               "warn_checkpoint": "Passed with a warning: v2's data is unchanged, but another process "
                                  "checkpointed a v2 write-ahead log.",
               "fail": "FAILED: the v2 home changed while the migration ran."}.get(g["status"], g["status"])
    L += ["## v2 home unchanged (guard)", "", verdict, ""] + [f"- {x}" for x in g["diffs"]]
    if g["diffs"]:
        L.append("")
    L += ["## Dump", "", f"Canonical dump sha256: `{d['dump_sha256']}`", ""]
    return "\n".join(L)


def render_attention_md(d: Mapping[str, Any]) -> str:
    items = d["attention"]["items"]
    L = [f"# Needs your attention: {d['run_id']}", "",
         f"{len(items)} items. Items marked 'notes hidden' keep their notes, but agents do not see them until you "
         "resolve the item (`seedgraph library attention`).", ""]
    groups: dict[str, list] = defaultdict(list)
    for a in items:
        groups[a["kind"]].append(a)
    order = [k for k in ATTENTION_KINDS if k in groups] + sorted(k for k in groups if k not in ATTENTION_KINDS)
    for kind in order:
        head, expl = ATTENTION_KINDS.get(kind, (kind.replace("_", " ").capitalize(), ""))
        L += [f"## {head} ({len(groups[kind])})", ""]
        if expl:
            L += [expl, ""]
        for a in groups[kind]:
            flag = " **notes hidden**" if a["hides_notes"] else ""
            who = _t(a["title"], a["paper_id"]) if a["paper_id"] else json.dumps(a["subject"], sort_keys=True)
            L.append(f"- `{a['item_id']}` {who}{flag}")
            if a["detail"]:
                L.append(f"  - {a['detail']}")
        L.append("")
    if not items:
        L.append("Nothing needs your attention.")
    return "\n".join(L) + "\n"


def load_v2_rows(run_dir: Path) -> dict[str, Any] | None:
    p = Path(run_dir) / V2_ROWS_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write_report(result: MigrationResult, *, v2_rows: Mapping[str, Any] | None = None) -> ReportFiles:
    """Write ``report.md``, ``report.json`` and ``attention.md`` into ``result.run_dir``.

    ``v2_rows`` defaults to the ``v2_rows.json`` that :func:`migration_hook` wrote.
    Raises :class:`ReportError` (after writing) when anything is unaccounted.
    """
    run_dir = Path(result.run_dir)
    if v2_rows is None:
        v2_rows = load_v2_rows(run_dir)
    d = build_report(result, v2_rows=v2_rows)
    files = ReportFiles(run_dir / REPORT_MD, run_dir / REPORT_JSON, run_dir / ATTENTION_MD, tuple(d["problems"]))
    _write(files.report_json, json.dumps(d, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    _write(files.report_md, render_report_md(d))
    _write(files.attention_md, render_attention_md(d))
    if d["problems"]:
        shown = "; ".join(d["problems"][:20])
        more = f" (and {len(d['problems']) - 20} more)" if len(d["problems"]) > 20 else ""
        raise ReportError(f"the migration report cannot account for every item: {shown}{more}. "
                          f"See {files.report_md}", files=files)
    return files


def load_report(run_dir: Path | str) -> dict[str, Any]:
    """Read a run's ``report.json``."""
    return json.loads((Path(run_dir) / REPORT_JSON).read_text(encoding="utf-8"))
