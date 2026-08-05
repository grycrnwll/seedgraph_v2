"""Project service core — the single layer the CLI and the read-only API share
(decision 81: one core, two thin surfaces).

Holds the project lifecycle (``create_project`` / ``open_project``), corpus
mutation (``add_work`` / ``set_inclusion_status``), and the project-only retrieval
filter (``corpus_works``). Structural project isolation is the per-project DB file
itself (decision 12): there is no ``project_id`` to filter on — querying
``ProjectHandle.engine`` only ever sees that one project's rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from .. import paths
from ..db.engine import init_project_db, make_project_engine
from ..db.project_models import ProjectDocument, Work
from ..errors import ValidationError
from ..vocab import INCLUSION_REASONS, InclusionStatus  # single source of truth
from . import identity, layout
from .config import AnswerPolicy, ProjectConfig, load_project_config, write_project_config

__all__ = [
    "INCLUSION_REASONS",
    "ProjectHandle",
    "create_project",
    "open_project",
    "list_projects",
    "add_work",
    "set_inclusion_status",
    "corpus_works",
    "list_documents",
    "project_dashboard",
]

# Closed inclusion-status set (the DB also CHECK-enforces it; validate early for a
# clean error rather than an IntegrityError).
_INCLUSION_STATUSES: frozenset[str] = frozenset(s.value for s in InclusionStatus)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_status(status: str) -> None:
    if status not in _INCLUSION_STATUSES:
        raise ValidationError(
            f"invalid inclusion_status {status!r}: must be one of "
            f"{sorted(_INCLUSION_STATUSES)}"
        )


@dataclass(frozen=True)
class ProjectHandle:
    """An opened project: its slug, root, db path, ORM engine, and parsed config.

    ``slug`` IS the project identity (decision 79). The handle is the object every
    service/review function takes; it never stores absolute paths in any row
    (relocatable; decision 17).
    """

    slug: str
    root: Path
    db_path: Path
    engine: Engine
    config: ProjectConfig


def create_project(
    slug: str,
    *,
    name: str | None = None,
    description: str | None = None,
    root: Path | None = None,
) -> ProjectHandle:
    """Scaffold a new project and return its :class:`ProjectHandle`.

    Validates the slug, HARD-ERRORS if the project directory already exists
    (explicit slug, no auto-suffix; decision 17), creates
    ``projects/{slug}/``, writes ``project.yaml`` (``project_id == slug``,
    ``answer_policy`` defaults), runs the ``migrate`` step to apply the numbered
    ``schema/project/*.sql`` (D6 — never ``create_all``), and scaffolds an empty
    ``runs/`` directory.
    """
    paths.validate_slug(slug)
    project_dir = layout.project_dir(slug, root)
    if project_dir.exists():
        raise ValidationError(
            f"project '{slug}' already exists at {project_dir} (explicit slug, no "
            f"auto-suffix; decision 17)"
        )
    project_dir.mkdir(parents=True)
    layout.project_runs_dir(slug, root).mkdir()

    cfg = ProjectConfig(
        project_id=slug,
        project_name=name or slug,
        description=description,
        answer_policy=AnswerPolicy(),
        created_at=_now(),
    )
    write_project_config(cfg, layout.project_yaml_path(slug, root))

    db_path = layout.project_db_path(slug, root)
    engine = make_project_engine(db_path)
    init_project_db(engine)
    return ProjectHandle(
        slug=slug,
        root=paths.resolve_home(root),
        db_path=db_path,
        engine=engine,
        config=cfg,
    )


def open_project(slug: str, *, root: Path | None = None) -> ProjectHandle:
    """Open an existing project, round-tripping ``project.yaml`` into the handle's
    :class:`ProjectConfig`.

    Hard-errors if the project directory is absent. The schema migrate step is
    re-run (idempotent; D6) so an opened project is always at the current version.
    """
    paths.validate_slug(slug)
    project_dir = layout.project_dir(slug, root)
    if not project_dir.exists():
        raise ValidationError(f"project '{slug}' does not exist at {project_dir}")
    cfg = load_project_config(slug, root)
    db_path = layout.project_db_path(slug, root)
    engine = make_project_engine(db_path)
    init_project_db(engine)
    return ProjectHandle(
        slug=slug,
        root=paths.resolve_home(root),
        db_path=db_path,
        engine=engine,
        config=cfg,
    )


def list_projects(root: Path | None = None) -> list[str]:
    """Enumerate project slugs under ``projects/`` (a directory with a ``project.yaml``
    is a project; decision 12 — the directory listing IS the registry)."""
    projects_dir = paths.resolve_home(root) / "projects"
    if not projects_dir.exists():
        return []
    return sorted(
        entry.name
        for entry in projects_dir.iterdir()
        if entry.is_dir() and (entry / "project.yaml").exists()
    )


def add_work(
    h: ProjectHandle,
    *,
    ids: dict[str, str] | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    year: int | None = None,
    venue: str | None = None,
    is_seed: bool = False,
    inclusion_status: str = "included",
    inclusion_reason: str | None = None,
    user_note: str | None = None,
) -> Work:
    """Add (or merge) a work from user-supplied metadata + identifiers and set its
    corpus membership; return the resulting :class:`Work`.

    Normalizes ids, calls ``identity.upsert_work`` (merge-or-create-or-route — on a
    ``duplicate_candidate`` the review item is already enqueued by ``upsert_work``),
    then writes/updates the ``project_documents`` row with ``inclusion_status`` /
    ``is_seed`` / ``inclusion_reason``, leaving ``access_status`` NULL (decision
    11/30 — never derived from membership). All metadata is taken verbatim; there is
    no resolver/network this phase (owned by phase_5b).
    """
    _check_status(inclusion_status)
    if inclusion_reason is None:
        inclusion_reason = "seed_document" if is_seed else "user_added"

    incoming: dict = {}
    for id_type, value in (ids or {}).items():
        if value:
            incoming[id_type] = value
    if title is not None:
        incoming["title"] = title
    if authors is not None:
        incoming["authors"] = authors
    if year is not None:
        incoming["year"] = year
    if venue is not None:
        incoming["venue"] = venue

    with Session(h.engine, expire_on_commit=False) as session:
        work, _outcome = identity.upsert_work(session, incoming)
        now = _now()
        doc = session.get(ProjectDocument, work.work_id)
        if doc is None:
            doc = ProjectDocument(
                work_id=work.work_id,
                inclusion_status=inclusion_status,
                inclusion_reason=inclusion_reason,
                is_seed=1 if is_seed else 0,
                access_status=None,  # decision 30 — NEVER derived from membership
                user_note=user_note,
                created_at=now,
                updated_at=now,
            )
            session.add(doc)
        else:
            doc.inclusion_status = inclusion_status
            doc.inclusion_reason = inclusion_reason
            if is_seed:
                doc.is_seed = 1
            if user_note is not None:
                doc.user_note = user_note
            doc.updated_at = now
            session.add(doc)
        session.commit()
        session.refresh(work)
        session.expunge(work)
    return work


def set_inclusion_status(
    h: ProjectHandle,
    work_id: str,
    status: str,
    reason: str | None = None,
) -> None:
    """Transition a work's ``inclusion_status`` (included|metadata_only|excluded)
    and bump ``updated_at``; ``access_status`` stays NULL across every transition
    (decision 30 — never derived from membership)."""
    _check_status(status)
    with Session(h.engine, expire_on_commit=False) as session:
        doc = session.get(ProjectDocument, work_id)
        if doc is None:
            raise ValidationError(
                f"no membership row for work {work_id!r} in project '{h.slug}'"
            )
        doc.inclusion_status = status
        if reason is not None:
            doc.inclusion_reason = reason
        doc.updated_at = _now()
        # access_status is deliberately left untouched (decision 30).
        session.add(doc)
        session.commit()


def corpus_works(
    h: ProjectHandle, statuses: tuple[str, ...] = ("included",)
) -> list[Work]:
    """Return the project's works whose membership ``inclusion_status`` is in
    ``statuses`` (default: only ``included``).

    The project-only retrieval filter every later retrieval path scopes through —
    isolation is structural (the per-project DB file; decision 12), so two projects
    on one machine never see each other's works."""
    with Session(h.engine, expire_on_commit=False) as session:
        works = list(
            session.exec(
                select(Work)
                .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
                .where(ProjectDocument.inclusion_status.in_(tuple(statuses)))
                .order_by(Work.created_at)
            ).all()
        )
        session.expunge_all()
    return works


def project_dashboard(
    h: ProjectHandle, *, cache_root: Path | str | None = None
) -> dict:
    """Compose the project dashboard read-model shared by the CLI and the UI.

    Pure read over project.db + the run inventory + the runtime LLM/budget config +
    a doctor badge — never opens another project, never dispatches a model, safe
    offline/keyless. Returns:

    * ``status_counts`` — corpus membership tally by ``inclusion_status``.
    * ``extraction`` — included-work extraction coverage (works with >=1 claim / total).
    * ``concepts`` / ``claims`` / ``lenses`` — overlay totals.
    * ``open_reviews`` — open ``review_queue`` items (via :func:`review.list_open`).
    * ``runs`` — :func:`run.list_runs` summaries (most-recent first).
    * ``models`` — preferred profile + availability for each LLM-shaped task.
    * ``budget`` — monthly soft limit + this-month DB-recorded spend.
    * ``doctor`` — ``{ok, n_fail, n_warn}`` badge from :func:`doctor.collect_checks`.
    """
    from ..config.loader import load_project_config as _load_runtime_config
    from ..db.adapter import raw_conn
    from ..llm import cost as _cost
    from ..llm.profiles import is_local_profile, is_profile_available
    from ..run import list_runs
    from . import review

    with Session(h.engine, expire_on_commit=False) as session:
        conn = raw_conn(session)
        status_counts: dict[str, int] = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT inclusion_status, COUNT(*) FROM project_documents "
                "GROUP BY inclusion_status"
            ).fetchall()
        }
        included_ids = {
            row[0]
            for row in conn.execute(
                "SELECT work_id FROM project_documents WHERE inclusion_status='included'"
            ).fetchall()
        }
        extracted_ids = {
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT work_id FROM extracted_claims"
            ).fetchall()
        }
        n_concepts = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
        n_claims = conn.execute("SELECT COUNT(*) FROM extracted_claims").fetchone()[0]
        n_lenses = conn.execute("SELECT COUNT(*) FROM lenses").fetchone()[0]
        year_month = _cost.this_month()
        monthly_spend = _cost.monthly_spend(conn, year_month)

    n_included = len(included_ids)
    covered = len(included_ids & extracted_ids)
    open_reviews = len(review.list_open(h))
    runs = list_runs(h.slug, root=h.root)

    runtime = _load_runtime_config(h.slug, h.root)
    models: list[dict] = []
    for task in (
        "note_extraction",
        "semantic_graph_extraction",
        "project_lens_extraction",
        "answer_generation",
    ):
        route = runtime.llm.routes.get(task)
        if route is None:
            continue
        profile = runtime.llm.profiles.get(route.preferred_profile)
        models.append(
            {
                "task": task,
                "preferred_profile": route.preferred_profile,
                "fallback_profile": route.fallback_profile,
                "local": bool(profile is not None and is_local_profile(profile)),
                "available": bool(profile is not None and is_profile_available(profile)),
            }
        )

    budget = runtime.budget
    doctor_badge = _doctor_badge(h.root, h.slug)

    return {
        "slug": h.slug,
        "project_name": getattr(h.config, "project_name", h.slug),
        "status_counts": status_counts,
        "corpus_total": sum(status_counts.values()),
        "extraction": {
            "works_included": n_included,
            "works_covered": covered,
            "coverage_pct": round(100.0 * covered / n_included, 1) if n_included else 0.0,
        },
        "concepts": n_concepts,
        "claims": n_claims,
        "lenses": n_lenses,
        "open_reviews": open_reviews,
        "runs": runs,
        "latest_run": runs[0] if runs else None,
        "models": models,
        "budget": {
            "monthly_soft_limit_usd": budget.monthly_soft_limit_usd,
            "usd_limit": budget.usd_limit,
            "year_month": year_month,
            "monthly_spend_usd": monthly_spend,
        },
        "doctor": doctor_badge,
    }


def _doctor_badge(root: Path | str | None, slug: str) -> dict:
    """Summarize ``doctor.collect_checks`` into a ``{ok, n_fail, n_warn}`` badge.

    Defensive: a doctor failure becomes a not-ok badge rather than crashing the
    dashboard."""
    try:
        from .. import doctor

        checks = doctor.collect_checks(root, slug)
    except Exception as exc:  # noqa: BLE001 — a doctor crash is a badge, not a 500
        return {"ok": False, "n_fail": 1, "n_warn": 0, "error": str(exc)}
    n_fail = sum(1 for c in checks if (not c.ok) and c.severity == "error")
    n_warn = sum(1 for c in checks if (not c.ok) and c.severity == "warning")
    return {"ok": n_fail == 0, "n_fail": n_fail, "n_warn": n_warn}


def list_documents(
    h: ProjectHandle,
    statuses: tuple[str, ...] = ("included", "metadata_only", "excluded"),
) -> list[dict]:
    """Return serializable corpus rows (work + membership) for ``statuses``.

    The shared read projection behind both ``seedgraph project show`` and the
    read-only API ``GET /projects/{slug}/documents``. ``access_status`` is surfaced
    verbatim (NULL this phase; decision 30). ``label`` is the never-blank display
    label (Build B chunk 9) — ``title`` stays the verbatim ``canonical_title``
    (possibly ``None``); nothing stored is mutated."""
    from ..display import derive_label

    rows: list[dict] = []
    with Session(h.engine, expire_on_commit=False) as session:
        results = session.exec(
            select(Work, ProjectDocument)
            .join(ProjectDocument, ProjectDocument.work_id == Work.work_id)
            .where(ProjectDocument.inclusion_status.in_(tuple(statuses)))
            .order_by(Work.created_at)
        ).all()
        for work, doc in results:
            rows.append(
                {
                    "work_id": work.work_id,
                    "title": work.canonical_title,
                    "label": derive_label(
                        title=work.canonical_title,
                        authors=work.authors,
                        year=work.year,
                        doi=work.doi,
                        openalex_id=work.openalex_id,
                        arxiv_id=work.arxiv_id,
                        work_id=work.work_id,
                    ),
                    "year": work.year,
                    "inclusion_status": doc.inclusion_status,
                    "is_seed": bool(doc.is_seed),
                    "access_status": doc.access_status,
                }
            )
    return rows
