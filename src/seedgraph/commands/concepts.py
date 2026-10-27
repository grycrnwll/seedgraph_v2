"""`seedgraph concepts ...`: library concepts from the shell (stage 4 design §5, §12 C4).

Thin adapters over :mod:`seedgraph.concepts`:

    concepts refresh                       recompute the derived concept tables
    concepts search QUERY [--sense] [--family F] [--graph G] [--limit N] [--mode auto|meaning|keyword]
    concepts embed [--model M] [--family F]  embed claim contexts not yet cached (Ollama)
    concepts eval-search --cases PATH [--out DIR] [--model M] [--classifier-model M]
                         [--protocol PATH] [--protocol-only] [--keyword-only] [--record-verdict]
    concepts review-sample [--n 50] [--seed S]   freeze the §11.2 review sample, list it in the home
    concepts acceptance [--inputs PATH] [--sample ID] [--out DIR]   the stage 4 acceptance report

``eval-search`` writes ``protocol.json`` (the frozen claim sets, gloss assignments and
rules) **before** ranking anything, then ``report.md`` and ``report.json``, by default
under ``<home>/concepts/eval/<timestamp>/``. ``--record-verdict`` stores the pass-rule
verdict as the library's adoption flag (``auto`` search uses meaning only when adopted).

Other slices add their commands to :data:`concepts_app`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from . import _common

concepts_app = typer.Typer(name="concepts", help="Library concepts: refresh, search, the embeddings test.",
                           no_args_is_help=True)

ROOT = _common.ROOT_OPTION
JSON = _common.JSON_OPTION


class SearchMode(str, Enum):
    auto = "auto"
    meaning = "meaning"
    keyword = "keyword"


@concepts_app.callback()
def _utf8() -> None:
    _common.force_utf8_stdout()


@concepts_app.command("refresh")
def refresh_cmd(root: Optional[Path] = ROOT, as_json: bool = JSON) -> None:
    """Recompute every derived concept table from the current claims and decisions."""
    from ..concepts import refresh as rf

    with _common.cli_errors(), _common.existing_library(root) as lib:
        report = rf.refresh(lib)
    summary = report.summary()
    if as_json:
        _common.echo_json(summary)
    else:
        for k, v in summary.items():
            typer.echo(f"{k}: {v if not isinstance(v, (list, dict)) else len(v)}")


@concepts_app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Words, or with --sense a handle family/label[~sense]."),
    sense: bool = typer.Option(False, "--sense", help="QUERY is a sense handle: find its neighbours."),
    family: Optional[str] = typer.Option(None, "--family", help="Only this type family."),
    graph: Optional[str] = typer.Option(None, "--graph", help="Only senses supported in this context graph."),
    limit: int = typer.Option(20, "--limit", min=1),
    mode: SearchMode = typer.Option(SearchMode.auto, "--mode"),
    root: Optional[Path] = ROOT,
    as_json: bool = JSON,
) -> None:
    """Search concepts (sense-level hits; the mode marker says which search ran)."""
    from ..concepts import refresh as rf
    from ..concepts import search as cs
    from ..concepts.model import DEFAULT_SENSE, SenseKey, parse_handle

    with _common.cli_errors(), _common.existing_library(root) as lib:
        rf.refresh_if_stale(lib)
        q: str | SenseKey = query
        if sense:
            try:
                fam, key, sid = parse_handle(query)
            except ValueError as exc:
                _common.fail(str(exc), _common.EXIT_USAGE)
            q = SenseKey(fam, key, sid or DEFAULT_SENSE)
        hits = cs.find(lib, q, family=family, graph=graph, k=limit, mode=mode.value)
    if as_json:
        _common.echo_json({"hits": hits, "search_mode": hits[0].mode if hits else None})
        return
    if not hits:
        typer.echo("no hits")
        return
    marker = hits[0].mode + (f" (fallback: {hits[0].fallback})" if hits[0].fallback else "")
    typer.echo(f"search mode: {marker}")
    _common.echo_table(("handle", "score", "papers", "claims", "example"),
                       [(h.handle, f"{h.score:.3f}", h.paper_count, h.claim_count, _common.one_line(h.best_claim_text))
                        for h in hits])


@concepts_app.command("embed")
def embed_cmd(
    model: str = typer.Option("nomic-embed-text", "--model", help="The Ollama embedding model."),
    family: Optional[list[str]] = typer.Option(None, "--family", help="Only these families (repeatable)."),
    root: Optional[Path] = ROOT,
    as_json: bool = JSON,
) -> None:
    """Embed every supported claim context not yet in the vector cache."""
    from ..concepts import refresh as rf
    from ..concepts import vectors as vx
    from ..llm.ollama import OllamaError

    with _common.cli_errors(), _common.existing_library(root) as lib:
        rf.refresh_if_stale(lib)
        try:
            stats = vx.embed_library(lib, vx.OllamaEmbedder(model), families=family or None)
        except OllamaError as exc:
            _common.fail(f"Ollama: {exc}", _common.EXIT_FAILED)
    if as_json:
        _common.echo_json(stats)
    else:
        typer.echo(f"{stats.requested} claim contexts: {stats.embedded} embedded in {stats.seconds:.1f}s, "
                   f"{stats.cached} cached ({stats.model})")


@concepts_app.command("eval-search")
def eval_search_cmd(
    cases: Path = typer.Option(..., "--cases", exists=True, dir_okay=False, help="The frozen cases (JSON)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Report folder (default <home>/concepts/eval/<ts>)."),
    model: str = typer.Option("nomic-embed-text", "--model", help="The Ollama embedding model."),
    classifier_model: str = typer.Option("qwen2.5:7b", "--classifier-model",
                                         help="The Ollama chat model that assigns head-pool claims to glosses."),
    protocol_path: Optional[Path] = typer.Option(None, "--protocol", exists=True, dir_okay=False,
                                                 help="Use this frozen protocol.json instead of building one."),
    protocol_only: bool = typer.Option(False, "--protocol-only", help="Write protocol.json and stop."),
    keyword_only: bool = typer.Option(False, "--keyword-only", help="Rank keyword mode only (no Ollama)."),
    record_verdict: bool = typer.Option(False, "--record-verdict",
                                        help="Store the verdict as the library's embeddings adoption flag."),
    root: Optional[Path] = ROOT,
) -> None:
    """Run the embeddings adoption test (keyword vs meaning) on frozen cases."""
    from ..concepts import refresh as rf
    from ..concepts import search as cs
    from ..concepts import search_eval as ev
    from ..concepts import vectors as vx
    from ..llm.ollama import OllamaError

    raw = cases.read_bytes()
    case_data = json.loads(raw)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out or (_common.home(root) / "concepts" / "eval" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _common.cli_errors(), _common.existing_library(root) as lib:
        rf.refresh_if_stale(lib)
        if protocol_path is not None:
            protocol = ev.Protocol.from_json(json.loads(protocol_path.read_text(encoding="utf-8")))
            if protocol.cases_sha256 not in (None, ev.cases_digest(raw)):
                _common.fail("the protocol was built from a different cases file")
        else:
            pools = ev.gloss_pools(lib.conn, case_data)
            assignments = {}
            if pools:
                typer.echo(f"assigning {sum(len(p) for p in pools.values())} head-pool claims to glosses "
                           f"with {classifier_model} ...", err=True)
                try:
                    assignments = ev.assign_glosses(pools, case_data, ev.ollama_gloss_classifier(classifier_model))
                except OllamaError as exc:
                    _common.fail(f"Ollama (gloss classifier): {exc}", _common.EXIT_FAILED)
            protocol = ev.build_protocol(lib.conn, case_data, assignments=assignments,
                                         cases_sha256=ev.cases_digest(raw), cases_path=str(cases))
        (out_dir / "protocol.json").write_text(
            json.dumps(protocol.to_json(), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        typer.echo(f"protocol: {out_dir / 'protocol.json'}", err=True)
        if protocol_only:
            return
        embedder = None if keyword_only else vx.OllamaEmbedder(model)

        def progress(done: int, total: int) -> None:
            if done == total or done % 1024 == 0:
                typer.echo(f"embedded {done}/{total}", err=True)

        try:
            result = ev.evaluate(lib, protocol, embedder=embedder, progress=progress)
        except OllamaError as exc:
            _common.fail(f"Ollama (embeddings): {exc}", _common.EXIT_FAILED)
        (out_dir / "report.md").write_text(ev.render_markdown(result), encoding="utf-8")
        (out_dir / "report.json").write_text(
            json.dumps(_common.to_jsonable(ev.result_json(result)), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8")
        if record_verdict and result.verdict is not None:
            cs.set_adoption(lib, adopted=result.verdict.passed, model=model, report=str(out_dir / "report.md"))
    typer.echo(f"report: {out_dir / 'report.md'}")
    if result.verdict is not None:
        typer.echo("verdict: " + ("PASS (adopt embeddings)" if result.verdict.passed else "FAIL (keep keyword)"))


@concepts_app.command("review-sample")
def review_sample_cmd(
    n: int = typer.Option(50, "--n", min=1, help="Sample size."),
    seed: int = typer.Option(0, "--seed", help="The seeded draw (stored with the sample)."),
    root: Optional[Path] = ROOT,
    as_json: bool = JSON,
) -> None:
    """Freeze the stratified review sample (§11.2) and write its list under <home>/concepts/review/."""
    from ..concepts import acceptance as acc
    from ..concepts import refresh as rf
    from ..concepts import researcher as rs

    with _common.cli_errors(), _common.existing_library(root) as lib:
        rf.refresh_if_stale(lib)
        try:
            draw = acc.draw_sample(lib, n=n, seed=seed)
        except rs.ResearcherError as exc:
            _common.fail(str(exc))
        items = acc.sample_items(lib.conn, draw.sample)
    out = _common.home(root) / "concepts" / "review" / f"{draw.sample.sample_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(acc.render_sample_list(draw, items), encoding="utf-8")
    comp = acc.composition(items)
    body = {"sample_id": draw.sample.sample_id, "path": draw.path, "seed": draw.sample.seed, "size": comp["size"],
            "eligible": draw.eligible, "topped_up": draw.topped_up, "list": str(out), **comp}
    if as_json:
        _common.echo_json(body)
        return
    typer.echo(f"review sample: {draw.path}")
    typer.echo(f"size {comp['size']} {comp['by_verdict']}; ambiguous-label items {comp['ambiguous']}; "
               f"eligible {draw.eligible}")
    if draw.topped_up:
        typer.echo(f"top-up: {draw.topped_up} migrated agent merges (the batch had only {draw.eligible} eligible)")
    typer.echo(f"list: {out}")


@concepts_app.command("acceptance")
def acceptance_cmd(
    inputs: Optional[Path] = typer.Option(None, "--inputs", help="Hand-entered figures (JSON; default "
                                          "<home>/concepts/acceptance/inputs.json)."),
    sample: Optional[str] = typer.Option(None, "--sample", help="The review sample (default: the newest)."),
    out: Optional[Path] = typer.Option(None, "--out", help="Report folder (default <home>/concepts/acceptance/<ts>)."),
    root: Optional[Path] = ROOT,
) -> None:
    """Write the stage 4 acceptance report (§11), with the researcher review and the wrong-merge gate."""
    from ..concepts import acceptance as acc
    from .concepts_work import readonly_library

    home = _common.home(root)
    try:
        data_in = acc.load_inputs(inputs or home / "concepts" / "acceptance" / "inputs.json")
    except acc.InputsError as exc:
        _common.fail(str(exc))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out or (home / "concepts" / "acceptance" / stamp)
    with _common.cli_errors(), readonly_library(root) as lib:
        try:
            data = acc.gather(lib.conn, home, data_in, sample_id=sample)
        except acc.InputsError as exc:
            _common.fail(str(exc))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(acc.render(data), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False),
                                         encoding="utf-8")
    typer.echo(f"report: {out_dir / 'report.md'}")
    review = data["review"]
    if review is not None:
        gate = review["gate_passed"]
        typer.echo("gate: " + ("pending" if gate is None else "PASS" if gate else "FAIL")
                   + f" ({review['marked']} of {review['size']} marked)")
