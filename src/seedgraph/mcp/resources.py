"""Chunk-4 resources + the ``semantic_query`` prompt (design 00 §4.2/§4.3, plan 01 chunk 4).

Split out of ``server.py`` because the §4.3 prompt text alone clears the plan's
~60-line bar; the plan allows either home.

Like :mod:`tools_read`, this module imports **no** ``mcp`` SDK symbol — the
decorators are called on the ``mcp`` object handed to the registration functions,
so the SDK stays confined to ``server.py`` (plus the deferred ``_tool_error``
imports). Same registration shape as every other chunk: ``register_*(mcp, ctx)``
called from ``build_server``, bodies closing over ``ctx`` lexically.

**The export guard (§4.2, security-relevant).** The ``graph.json`` resource is
produced through :func:`semantic.export.export_graph` with ``allow_private=False``
passed EXPLICITLY. There is deliberately no parameter, argument, or code path on
this resource that can set it True — the URI template's single ``{slug}``
parameter is the whole input surface. (Belt-and-suspenders, found while reading
the seam: ``export_graph`` writes ``runs/{run_id}/graph.json`` through the
public-safe filter *regardless* of ``allow_private`` — the private view goes to a
separate ``exports/graph.private.json``. So the artifact this resource reads is
public-safe twice over. The explicit ``allow_private=False`` is still passed, and
still load-bearing, because it is the guarantee that no private export file is
even written as a side effect of serving this resource.)

**No resource serves cache.db content** (PDF / markdown full text). That non-goal
was reaffirmed in the dogfood reconciliation — a gated section-text read was
considered and deliberately deferred. Do not add one here.

The §5.1 redaction filter is NOT applied to ``graph.json``: the export guard is
strictly stronger on this payload — it DROPS non-shareable concepts and edges
entirely rather than blanking a text field on a row that still ships — so the
filter would have nothing left to withhold. The two docs resources are static
public documentation with no corpus content at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .. import run as run_mod
from ..semantic.export import export_graph

# Repo-relative docs directory: this file is ``<repo>/src/seedgraph/mcp/resources.py``,
# so parents[3] is the repo root. ``ctx.root`` is the *projects home* and never
# contains ``docs/``, so it deliberately is not used here.
_DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"


def _read_doc(name: str) -> str:
    """Return the text of ``docs/{name}``, or the M9 structured error if absent."""
    path = _DOCS_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        from .server import _tool_error

        _tool_error(
            "doc_not_found",
            f"Documentation file {name} is not readable at {path}: {exc}",
            doc=name,
        )


def register_resources(mcp: Any, ctx: Any) -> None:
    """Register the three 00 §4.2 resources on ``mcp``, closing over ``ctx``.

    SDK shape verified against the installed ``mcp`` SDK, and it differs from what
    a reader might assume: a URI containing ``{slug}`` registers as a **resource
    template**, so ``graph.json`` is advertised by ``list_resource_templates`` (as
    ``uriTemplate``) and NOT by ``list_resources``; the two parameterless docs
    resources are advertised by ``list_resources``. All three are read through the
    same ``read_resource(<concrete uri>)`` call.
    """

    @mcp.resource(
        "seedgraph://projects/{slug}/graph.json",
        name="project_graph_json",
        mime_type="application/json",
    )
    def project_graph_json(slug: str) -> str:
        """The project's PUBLIC graph export (node-link JSON) for its latest run.

        Routed through the default-deny export guard
        (``vocab.is_shareable`` / ``field_allowed`` via ``semantic/access.py``):
        non-shareable concepts and edges are DROPPED, never merely thinned, and a
        non-shareable concept's ``definition`` is never emitted. There is no
        parameter on this resource that can relax that — ``allow_private`` is hard
        false. ``run_id`` follows the same latest-run default as the tools
        (``run.latest_run_id``). Citation edges here are REAL ``cites`` edges;
        concept co-anchoring is semantic reach, not citation reach.
        """
        handle = ctx.get_handle(slug)
        run_id = run_mod.latest_run_id(handle.slug, root=ctx.root)
        if run_id is None:
            from .server import _tool_error

            _tool_error(
                "no_graph_run",
                f"Project {slug!r} has no run to export a graph from "
                "(run `cite build` first).",
                project=slug,
            )
        with ctx.project_conn(handle) as conn:
            written = export_graph(
                conn,
                slug=handle.slug,
                run_id=run_id,
                fmt="json",
                # HARD FALSE — the default-deny export guard. Passed explicitly so
                # the guarantee is visible at the call site, and reachable by no
                # caller-supplied value. Do not parameterize this.
                allow_private=False,
                root=ctx.root,
            )
        # export_graph returns every path it wrote; take the public run-dir
        # artifact by NAME rather than by position, so a future added export
        # format cannot silently change which file this resource serves.
        graph_json = next(p for p in written if p.name == "graph.json")
        return graph_json.read_text(encoding="utf-8")

    @mcp.resource(
        "seedgraph://docs/quickstart",
        name="docs_quickstart",
        mime_type="text/markdown",
    )
    def docs_quickstart() -> str:
        """``docs/QUICKSTART.md`` — the short getting-started path. Static, public."""
        return _read_doc("QUICKSTART.md")

    @mcp.resource(
        "seedgraph://docs/user-handbook",
        name="docs_user_handbook",
        mime_type="text/markdown",
    )
    def docs_user_handbook() -> str:
        """``docs/USER_HANDBOOK.md`` — the full command/workflow reference. Static."""
        return _read_doc("USER_HANDBOOK.md")


# ---------------------------------------------------------------------------
# §4.3 — the `semantic_query` prompt.
#
# The workflow text is authored ONCE, here, transcribing design 00 §4.3 as
# reconciled after dogfooding (overview-first; the whole-list load is a
# small-corpus FALLBACK, not the opening move). The nine honesty rules are the
# doc's own wording — tests pin the load-bearing phrases as string presence so a
# future edit has to be deliberate. Weakening a rule into a paraphrase is a
# regression, not a style change.
# ---------------------------------------------------------------------------

_TRIGGER_POSTURE = """\
TRIGGER POSTURE (the shipped, PRECISION-rebalanced consult wording)
Fire proactively, but only when grounding actually changes the answer; skip turns
about the user's own work (method, data, code, plan) that don't hinge on a corpus
fact; don't re-query a thread already grounded."""

_WORKFLOW = """\
WORKFLOW — overview-first, then read a typed slice by meaning

1. `concepts_overview` FIRST — orient by the corpus's *shape*: which
   concept_types are populated, what recurs, what the assumed background is. On a
   real corpus the overlay is large (~12.8k concepts, ~87.5% singletons), so
   dumping the whole `concepts_list` drowns orientation.
2. Scope to the concept_type(s) whose kind matches the term (e.g. `method`,
   `identification_assumption`, `data`).
3. Read the whole *typed* slice by meaning — `concepts_list` with `concept_type`
   set, NO lexical pre-filter (near-synonyms share no words with the term),
   select by MEANING.
4. `concept_show` per selected concept — pull its papers / claims / spans.
5. `ask` with `no_llm=true` — query-driven retrieval + citation-neighborhood
   expansion. The retrieval-only envelope is an honest answer, not a failure.
6. `graph_analyze` — whole-graph structure (god nodes, cross-community bridges).

Loading the whole *untyped* `concepts_list` is an explicit SMALL-CORPUS FALLBACK
only — appropriate when the term genuinely spans types AND the corpus is small
enough to read whole; on a large corpus, lean on the overview + per-type slices
instead."""

_HONESTY_RULES = """\
HONESTY RULES — carry these into the answer, do not soften them

1. The overview ranks by distinct-paper RECURRENCE (`paper_frequency`), not IDF
   `weight` — orientation is not retrieval; do not reuse the weight rule on the
   overview.
2. Concept selection is Claude's own JUDGMENT by meaning, NOT a computed
   similarity score — there is no embedding or similarity number behind it; say
   so plainly.
3. Provenance horizon. Every answer should note the corpus's temporal reach —
   "the seeds span X-Y; N% of works are metadata-only; treat post-Y concurrent
   work as a blind spot" — so the reader knows what the corpus cannot have seen.
4. Rank-and-page, NEVER a pruned view — the overview pages the FULL overlay;
   never present a degree-pruned concept view as if it were the whole (pruning
   was refuted — the singleton tail is real hyper-specific concepts).
5. Semantic reach (a shared concept) is NOT citation reach (a real `cites` edge)
   — never conflate; two papers anchored to one concept are about the same thing,
   not citing it.
6. `weight` is IDF-style DISCRIMINATIVENESS, not importance — high = distinctive
   (few papers), low = ubiquitous; never present it as a relevance or quality
   score.
7. `metadata_only` = THIN EVIDENCE — surface those recommendations as weak
   signals (title / metadata only, no extracted full text), never as fully-read
   works.
8. Ground quotes in EVIDENCE SPANS, never in concept labels — a label is the
   extractor's synthesis across papers, not a verbatim phrase from any one of
   them.
9. A NEGATIVE ANSWER IS A VALID ANSWER — if nothing matches, or the lit-up papers
   sit in one community with no bridge, say so; do not manufacture a match or a
   bridge."""


def render_semantic_query(term: str, project: str | None = None) -> str:
    """Render the §4.3 ``semantic_query`` workflow text for ``term``.

    Split out of the registration closure so the text is testable (and readable)
    without standing up a server. ``project`` is optional: when omitted the prompt
    tells the caller to resolve one via ``project_list`` rather than guessing a
    slug, because every tool but ``project_list`` requires an explicit project.
    """
    if project:
        scope = (
            f"Project: {project} — pass `project=\"{project}\"` to every tool below."
        )
    else:
        scope = (
            "Project: NOT GIVEN — call `project_list` first and pick the project "
            "(or ask the user which one). Every tool below except `project_list` "
            "requires an explicit `project`; do not guess a slug."
        )
    return "\n\n".join(
        [
            f"SEEDGRAPH SEMANTIC QUERY — ground this term in the corpus: {term}",
            scope,
            _TRIGGER_POSTURE,
            _WORKFLOW,
            _HONESTY_RULES,
        ]
    )


def register_prompts(mcp: Any, ctx: Any) -> None:
    """Register the single 00 §4.3 ``semantic_query`` prompt on ``mcp``.

    SDK shape (verified against the installed SDK): ``@mcp.prompt()`` is called
    with parentheses; the function's parameters become the prompt's declared
    arguments, and a parameter with a default is advertised ``required=false``. A
    plain ``str`` return is wrapped by the SDK into a single user message.

    ``ctx`` is accepted for signature parity with the other ``register_*``
    functions; the prompt is pure text and reads no project state (it must render
    identically whether or not the named project exists — a prompt is a workflow
    the client is about to run, not a query result).
    """

    @mcp.prompt()
    def semantic_query(term: str, project: str | None = None) -> str:
        """Ground a term in the seedgraph corpus: overview-first concept workflow.

        Emits the shipped `/seedgraph` workflow against this server's tool catalog
        — `concepts_overview` to orient by corpus shape, scope to the matching
        concept_type(s), read that typed slice by meaning, `concept_show` the
        selections, `ask` (free `no_llm` path), then `graph_analyze` — together
        with the honesty rules the answer must carry (recurrence vs IDF weight,
        judgment vs similarity score, semantic vs citation reach, spans vs labels,
        thin `metadata_only` evidence, the provenance horizon, and that a negative
        answer is a valid answer).
        """
        return render_semantic_query(term, project)
