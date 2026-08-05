"""Grouped Typer sub-apps for the project model (phase_5).

The complete CLI write/read surface (decision 81 — the CLI is the full surface;
the API is a thin read subset). Two grouped sub-apps mounted onto the root app:

* ``seedgraph project new|list|add|show|set-status``
* ``seedgraph review list|resolve``

Every verb is a thin wrapper over the shared :mod:`seedgraph.project.service` /
:mod:`seedgraph.project.review` core (the same core the read-only API consumes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ..errors import SeedgraphError
from ..paths import resolve_home
from . import review as review_mod
from . import service

project_app = typer.Typer(
    name="project",
    help="Project model: create/list/show projects + add/classify works.",
    no_args_is_help=True,
)
review_app = typer.Typer(
    name="review",
    help="Review queue: list open items + resolve them.",
    no_args_is_help=True,
)

_root_option = typer.Option(None, "--root", help="Override the seedgraph home directory.")


def _fail(exc: Exception) -> None:
    typer.echo(f"error: {exc}", err=True)
    raise typer.Exit(code=1)


# --- project ---------------------------------------------------------------

@project_app.command("new")
def project_new(
    slug: str = typer.Argument(..., help="Project slug (^[a-z0-9._-]+$)."),
    name: Optional[str] = typer.Option(None, "--name", help="Display name."),
    description: Optional[str] = typer.Option(None, "--description", help="Description."),
    root: Optional[Path] = _root_option,
) -> None:
    """Scaffold a new project (project.db + project.yaml + runs/)."""
    try:
        handle = service.create_project(slug, name=name, description=description, root=root)
    except SeedgraphError as exc:
        _fail(exc)
    typer.echo(f"created project '{slug}' at {handle.db_path.parent}")


@project_app.command("list")
def project_list(root: Optional[Path] = _root_option) -> None:
    """List project slugs under the seedgraph home."""
    for slug in service.list_projects(root):
        typer.echo(slug)


@project_app.command("add")
def project_add(
    slug: str = typer.Argument(..., help="Project slug."),
    doi: Optional[str] = typer.Option(None, "--doi"),
    arxiv: Optional[str] = typer.Option(None, "--arxiv"),
    openalex: Optional[str] = typer.Option(None, "--openalex"),
    s2: Optional[str] = typer.Option(None, "--s2"),
    ssrn: Optional[str] = typer.Option(None, "--ssrn"),
    title: Optional[str] = typer.Option(None, "--title"),
    year: Optional[int] = typer.Option(None, "--year"),
    author: Optional[list[str]] = typer.Option(None, "--author", help="Repeatable."),
    venue: Optional[str] = typer.Option(None, "--venue"),
    seed: bool = typer.Option(False, "--seed", help="Tag as a seed document."),
    status: str = typer.Option("included", "--status", help="included|metadata_only|excluded."),
    reason: Optional[str] = typer.Option(None, "--reason"),
    note: Optional[str] = typer.Option(None, "--note"),
    root: Optional[Path] = _root_option,
) -> None:
    """Add (or merge) a work by identifier/metadata and set its corpus membership."""
    ids = {
        key: value
        for key, value in {
            "doi": doi,
            "openalex": openalex,
            "arxiv": arxiv,
            "s2": s2,
            "ssrn": ssrn,
        }.items()
        if value
    }
    try:
        handle = service.open_project(slug, root=root)
        work = service.add_work(
            handle,
            ids=ids or None,
            title=title,
            authors=list(author) if author else None,
            year=year,
            venue=venue,
            is_seed=seed,
            inclusion_status=status,
            inclusion_reason=reason,
            user_note=note,
        )
    except SeedgraphError as exc:
        _fail(exc)
    typer.echo(f"{work.work_id}\t{status}{' (seed)' if seed else ''}")


@project_app.command("show")
def project_show(
    slug: str = typer.Argument(..., help="Project slug."),
    status: Optional[list[str]] = typer.Option(
        None, "--status", help="Filter (repeatable); default shows all three."
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Render the corpus as a table (work_id | title | year | status | seed | access)."""
    statuses = tuple(status) if status else ("included", "metadata_only", "excluded")
    try:
        handle = service.open_project(slug, root=root)
        rows = service.list_documents(handle, statuses)
    except SeedgraphError as exc:
        _fail(exc)
    typer.echo("work_id\ttitle\tyear\tinclusion_status\tis_seed\taccess_status")
    for row in rows:
        typer.echo(
            f"{row['work_id']}\t{row['title'] or ''}\t{row['year'] or ''}\t"
            f"{row['inclusion_status']}\t{int(row['is_seed'])}\t{row['access_status'] or ''}"
        )


@project_app.command("set-status")
def project_set_status(
    slug: str = typer.Argument(..., help="Project slug."),
    work_id: str = typer.Argument(..., help="work_ id."),
    status: str = typer.Argument(..., help="included|metadata_only|excluded."),
    reason: Optional[str] = typer.Option(None, "--reason"),
    root: Optional[Path] = _root_option,
) -> None:
    """Transition a work's inclusion_status (access_status stays NULL; decision 30)."""
    try:
        handle = service.open_project(slug, root=root)
        service.set_inclusion_status(handle, work_id, status, reason)
    except SeedgraphError as exc:
        _fail(exc)
    typer.echo(f"{work_id} -> {status}")


@project_app.command("agent-setup")
def project_agent_setup(
    slug: str = typer.Argument(..., help="Project slug (must already exist)."),
    dir: Path = typer.Option(
        Path.cwd, "--dir", help="Target directory for the CLAUDE.md norm block (default: cwd)."
    ),
    skills_dir: Path = typer.Option(
        lambda: Path.home() / ".claude" / "skills",
        "--skills-dir",
        help="Global skills directory to deploy the seedgraph skill into.",
    ),
    no_skill: bool = typer.Option(False, "--no-skill", help="Skip installing the global skill."),
    root: Optional[Path] = _root_option,
) -> None:
    """Install the ambient trigger for a research working directory.

    Writes the per-project CLAUDE.md norm block (Carrier A) and, unless
    ``--no-skill``, deploys the global ``seedgraph`` skill (Carrier B). Both are
    idempotent — re-running updates in place, never duplicates.
    """
    from .. import agent_setup

    try:
        if slug not in service.list_projects(root):
            raise SeedgraphError(
                f"no project '{slug}' under {resolve_home(root)}; "
                f"run `seedgraph project new {slug}` first"
            )
        claude_path = agent_setup.write_claude_block(dir, slug)
        skill_line: Optional[str] = None
        if not no_skill:
            skill_path, changed = agent_setup.install_skill(skills_dir)
            skill_line = f"{skill_path} ({'updated' if changed else 'already current'})"
    except SeedgraphError as exc:
        _fail(exc)
    typer.echo(f"CLAUDE.md: {claude_path}")
    typer.echo(f"skill: {skill_line}" if skill_line else "skill: skipped (--no-skill)")
    typer.echo(f"seedgraph home: {resolve_home(root)}")


# --- review ----------------------------------------------------------------

@review_app.command("list")
def review_list(
    slug: str = typer.Argument(..., help="Project slug."),
    root: Optional[Path] = _root_option,
) -> None:
    """List open review_queue items for the project."""
    try:
        handle = service.open_project(slug, root=root)
        rows = review_mod.review_rows(handle, review_mod.list_open(handle))
    except SeedgraphError as exc:
        _fail(exc)
    # item_id stays as the trailing column — `review resolve` takes it.
    typer.echo("item_type\tdecision\tstatus\titem_id")
    for row in rows:
        typer.echo(
            f"{row['item_type']}\t{row['decision']}\t{row['status']}\t{row['item_id']}"
        )


@review_app.command("resolve")
def review_resolve(
    slug: str = typer.Argument(..., help="Project slug."),
    item_id: str = typer.Argument(..., help="rq_ id."),
    action: str = typer.Argument(
        ..., help="approve|reject|merge|re_resolve|exclude|split."
    ),
    survivor: Optional[str] = typer.Option(
        None, "--survivor",
        help="duplicate_candidate merge: the surviving work_id "
        "(default: the pre-existing canonical work).",
    ),
    candidate: Optional[int] = typer.Option(
        None, "--candidate",
        help="citation_resolution approve/re_resolve: 0-based candidate index "
        "(default: the single candidate).",
    ),
    target_work: Optional[str] = typer.Option(
        None, "--target-work",
        help="citation_resolution approve/re_resolve: an already-known work_id "
        "to point the edge at (bypasses candidates).",
    ),
    root: Optional[Path] = _root_option,
) -> None:
    """Resolve a review item (idempotent; records action + resolved_at).

    Verdicts mutate the graph in the same transaction as the status flip:
    ``merge`` collapses a duplicate pair, ``exclude`` sets the membership row,
    ``approve``/``re_resolve`` on a citation item writes the manual_override edge.
    """
    try:
        handle = service.open_project(slug, root=root)
        review_mod.resolve(
            handle, item_id, action,
            survivor_id=survivor, candidate_index=candidate,
            target_work_id=target_work,
        )
    except SeedgraphError as exc:
        _fail(exc)
    typer.echo(f"{item_id} resolved ({action})")
