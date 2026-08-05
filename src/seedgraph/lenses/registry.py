"""Lens registry — discover YAML, sync the `lenses` row, freeze-on-active (plan §5/§6.2).

The on-disk ``projects/{slug}/lenses/*.yaml`` files are the source of truth; the
`lenses` table is a thin registry of (hash + path + status) plus a one-time
`definition_yaml` snapshot taken on promotion to ``active`` (decision 50,
immutability-on-use). Status lifecycle (vocab.LensStatus):

    draft -> calibrating -> active -> archived

While ``draft``/``calibrating`` the lens is mutable: `definition_hash` refreshes
in place from the YAML, prior runs go stale (never deleted), `definition_yaml`
stays NULL. On first full run the lens promotes to ``active`` and
`definition_yaml` is snapshotted once; later edits require a new ``lens_id``.

Schema authority is the numbered ``0009_lenses.sql`` migration (D6); this module
only reads/writes rows, never authors schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..db.adapter import raw_conn
from ..errors import ValidationError
from ..vocab import LensStatus
from .schema import LensDefinition

if TYPE_CHECKING:
    from sqlmodel import Session


@dataclass(frozen=True)
class LensRow:
    """Read view of one `lenses` row (mirrors the 0009 migration columns)."""

    lens_id: str
    name: str
    scope: str
    object_type: str
    status: str
    yaml_path: str
    definition_hash: str
    definition_yaml: str | None
    created_at: str
    updated_at: str


_COLUMNS = (
    "lens_id, name, scope, object_type, status, yaml_path, definition_hash, "
    "definition_yaml, created_at, updated_at"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_lens_row(row) -> LensRow:
    return LensRow(*row)


def discover_lens_files(project_dir: Path) -> list[Path]:
    """Return the sorted ``projects/{slug}/lenses/*.yaml`` files (may be empty)."""
    return sorted((Path(project_dir) / "lenses").glob("*.yaml"))


def lens_yaml_path(project_dir: Path, lens_id: str) -> Path:
    """Resolve ``projects/{slug}/lenses/{lens_id}.yaml`` (the authored source)."""
    return Path(project_dir) / "lenses" / f"{lens_id}.yaml"


def get_lens_row(session: "Session", lens_id: str) -> LensRow | None:
    """Return the registry row for ``lens_id`` or ``None``."""
    conn = raw_conn(session)
    row = conn.execute(
        f"SELECT {_COLUMNS} FROM lenses WHERE lens_id = ?", (lens_id,)
    ).fetchone()
    return _row_to_lens_row(tuple(row)) if row is not None else None


def load_lens(project_dir: Path, lens_id: str) -> LensDefinition:
    """Load + validate the on-disk lens definition (raises on bad YAML)."""
    return LensDefinition.from_yaml(lens_yaml_path(project_dir, lens_id))


#: Built-in lens templates shipped with the package (``lens new --from-template``).
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def create_lens_from_template(
    session: "Session",
    project_dir: Path,
    lens_id: str,
    from_template: str,
    templates_dir: Path | None = None,
) -> LensRow:
    """Scaffold ``projects/{slug}/lenses/{lens_id}.yaml`` from a built-in template
    and register it — the shared create path behind both ``lens new`` (CLI) and the
    web lens builder (so neither re-implements scaffolding).

    Copies the named template verbatim (rewriting its ``lens_id:`` line when the new
    id differs) and calls :func:`sync_lens`. Raises
    :class:`~seedgraph.errors.ValidationError` when the template is unknown or the
    destination YAML already exists (no silent overwrite).
    """
    tdir = Path(templates_dir) if templates_dir is not None else TEMPLATES_DIR
    template = tdir / f"{from_template}.yaml"
    if not template.exists():
        raise ValidationError(
            f"unknown template {from_template!r} (only regularity_conditions_v1 ships)"
        )
    lens_dir = Path(project_dir) / "lenses"
    lens_dir.mkdir(parents=True, exist_ok=True)
    dest = lens_dir / f"{lens_id}.yaml"
    if dest.exists():
        raise ValidationError(f"lens YAML already exists: {dest}")
    content = template.read_text(encoding="utf-8")
    if lens_id != from_template:
        content = content.replace(f"lens_id: {from_template}", f"lens_id: {lens_id}")
    dest.write_text(content, encoding="utf-8")
    return sync_lens(session, project_dir, lens_id)


def validate_lens_yaml(yaml_text: str) -> LensDefinition:
    """Parse + validate a lens YAML *document text* into a :class:`LensDefinition`.

    The same validation ``lens validate`` (CLI) runs over the on-disk file, but over
    an in-memory textarea so the web builder can validate before writing. Raises
    :class:`~seedgraph.errors.ValidationError` (message prefixed ``INVALID``) on any
    parse/schema error.
    """
    import yaml as _yaml

    try:
        data = _yaml.safe_load(yaml_text)
    except _yaml.YAMLError as exc:
        raise ValidationError(f"INVALID lens YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("INVALID lens YAML: document did not parse to a mapping")
    try:
        return LensDefinition.from_dict(data)
    except Exception as exc:  # noqa: BLE001 — any schema error is a validation failure
        raise ValidationError(f"INVALID lens YAML: {exc}") from exc


def write_lens_yaml(
    session: "Session", project_dir: Path, lens_id: str, yaml_text: str
) -> tuple[LensDefinition, LensRow]:
    """Validate ``yaml_text``, write it to ``lenses/{lens_id}.yaml``, then resync.

    Validation happens first (nothing is written when the YAML is invalid). On
    success the file is written and :func:`sync_lens` refreshes the registry row.
    """
    lens = validate_lens_yaml(yaml_text)
    path = lens_yaml_path(project_dir, lens_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_text, encoding="utf-8")
    row = sync_lens(session, project_dir, lens_id)
    return lens, row


def sync_lens(session: "Session", project_dir: Path, lens_id: str) -> LensRow:
    """Refresh the `lenses` row for ``lens_id`` from its on-disk YAML.

    Loads + validates ``projects/{slug}/lenses/{lens_id}.yaml``, recomputes
    `definition_hash`, and upserts the registry row. While the lens is
    ``draft``/``calibrating`` the hash refreshes in place (prior runs go stale,
    not deleted) and `definition_yaml` stays NULL. If the lens is already
    ``active``, an on-disk hash change is rejected (immutability-on-use — requires
    a new ``_v2`` lens_id). Promotion to ``active`` (driven by the runner's first
    full run) snapshots `definition_yaml` exactly once. Implements decisions
    22/50/77.
    """
    lens = load_lens(project_dir, lens_id)
    new_hash = lens.definition_hash()
    yaml_rel = f"lenses/{lens_id}.yaml"
    conn = raw_conn(session)
    now = _now()

    existing = get_lens_row(session, lens_id)
    if existing is None:
        conn.execute(
            f"INSERT INTO lenses ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lens.lens_id,
                lens.name,
                lens.scope,
                lens.object_type,
                LensStatus.draft.value,
                yaml_rel,
                new_hash,
                None,  # definition_yaml stays NULL while draft (no duplication)
                now,
                now,
            ),
        )
        session.commit()
        return get_lens_row(session, lens_id)

    if existing.status == LensStatus.active.value:
        # Immutability-on-use: an active lens is frozen. An unchanged re-sync is a
        # no-op; a divergent on-disk edit is rejected (requires a new _v2 lens_id).
        if existing.definition_hash != new_hash:
            raise ValidationError(
                f"lens '{lens_id}' is active (frozen) and its on-disk definition has "
                f"changed; create a new lens_id (e.g. '{lens_id}_v2') instead of "
                f"editing an active lens (immutability-on-use, decision 50)."
            )
        return existing

    # draft / calibrating: mutable — hash refreshes in place, definition_yaml NULL.
    conn.execute(
        "UPDATE lenses SET name = ?, scope = ?, object_type = ?, yaml_path = ?, "
        "definition_hash = ?, updated_at = ? WHERE lens_id = ?",
        (lens.name, lens.scope, lens.object_type, yaml_rel, new_hash, now, lens_id),
    )
    session.commit()
    return get_lens_row(session, lens_id)


def set_status(session: "Session", lens_id: str, status: str) -> LensRow:
    """Set the lens lifecycle ``status`` (draft|calibrating|active|archived)."""
    if status not in {s.value for s in LensStatus}:
        raise ValidationError(f"invalid lens status {status!r}")
    conn = raw_conn(session)
    conn.execute(
        "UPDATE lenses SET status = ?, updated_at = ? WHERE lens_id = ?",
        (status, _now(), lens_id),
    )
    session.commit()
    return get_lens_row(session, lens_id)


def list_lenses(session: "Session") -> list[LensRow]:
    """All registered lenses with status (for ``lens list`` + staleness display)."""
    conn = raw_conn(session)
    rows = conn.execute(f"SELECT {_COLUMNS} FROM lenses ORDER BY lens_id").fetchall()
    return [_row_to_lens_row(tuple(r)) for r in rows]


def lens_staleness(
    session: "Session", project_dir: Path, lens_id: str
) -> dict:
    """Coverage + staleness read-model for ``lens_id`` (the ``lens status`` lift).

    Bundles the registry ``status``, the per-status :func:`lenses.results.coverage`,
    and the pull-based :func:`lenses.results.stale_outputs` run ids into one dict
    shared by the CLI ``lens status`` command and the web lens-detail screen.
    Defensive: an unregistered lens (no row) or a lens whose on-disk YAML is absent
    yields zeroed coverage / no stale list rather than raising, so the UI renders a
    clean "unregistered" state."""
    from .results import coverage as _coverage, stale_outputs as _stale_outputs

    row = get_lens_row(session, lens_id)
    cov = _coverage(session, lens_id)
    stale: list[str] = []
    definition_hash: str | None = row.definition_hash if row else None
    try:
        lens = load_lens(project_dir, lens_id)
        definition_hash = lens.definition_hash()
        stale = _stale_outputs(session, lens)
    except Exception:  # noqa: BLE001 — missing/invalid YAML => unregistered display
        pass
    return {
        "lens_id": lens_id,
        "status": row.status if row else "unregistered",
        "found": cov.found,
        "not_found": cov.not_found,
        "ambiguous": cov.ambiguous,
        "extraction_failed": cov.extraction_failed,
        "not_applicable": cov.not_applicable,
        "skipped_no_markdown": cov.skipped_no_markdown,
        "works_total": cov.works_total,
        "works_covered": cov.works_covered,
        "stale_run_ids": stale,
        "stale_count": len(stale),
        "definition_hash": definition_hash,
    }


def promote_to_active(session: "Session", lens: "LensDefinition") -> LensRow:
    """Promote a ``draft``/``calibrating`` lens to ``active`` and snapshot
    `definition_yaml` once (frozen reproducibility, decision 50). Idempotent if
    already active with a matching hash; raises if the on-disk definition has
    diverged from the snapshot. Called by the runner on first full run (plan §10
    step 10).
    """
    existing = get_lens_row(session, lens.lens_id)
    if existing is None:
        raise ValidationError(
            f"cannot promote unknown lens '{lens.lens_id}' (sync_lens first)"
        )
    new_hash = lens.definition_hash()
    if existing.status == LensStatus.active.value:
        if existing.definition_hash != new_hash:
            raise ValidationError(
                f"active lens '{lens.lens_id}' diverged from its frozen snapshot; "
                f"create a new lens_id (immutability-on-use)."
            )
        return existing  # idempotent

    conn = raw_conn(session)
    conn.execute(
        "UPDATE lenses SET status = ?, definition_hash = ?, definition_yaml = ?, "
        "updated_at = ? WHERE lens_id = ?",
        (
            LensStatus.active.value,
            new_hash,
            lens.definition_snapshot(),
            _now(),
            lens.lens_id,
        ),
    )
    session.commit()
    return get_lens_row(session, lens.lens_id)
