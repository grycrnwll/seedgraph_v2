"""The v2 -> v3 migration (design docs/V3_LIBRARY_DESIGN.md §6).

One deep module; its interface is :func:`run_migration`. Internally a pipeline:
preflight (processes, disk) -> v2 guard (before) -> :mod:`.snapshot` -> :mod:`.plan`
(pure; uses :mod:`.defects`) -> :mod:`.files` (blob copy) -> :mod:`.apply` (staging
library and graphs) -> optional caller hook (S7: answers export) -> :mod:`.dump` ->
swap (:mod:`.files`) -> v2 guard (after, :mod:`.v2guard`).

The v2 home is only ever read (``mode=ro``). Nothing outside ``<v3 home>/migration/``
is written except the owned ``library/``, ``graphs/`` and ``v2_answers/`` folders, and
those are replaced by rename, the old ones moved into ``migration/replaced/``.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import apply as _apply
from . import dump as _dump
from . import files as _files
from . import v2guard
from .files import MigrationRefused
from .plan import MigrationPlan, build_plan
from .snapshot import Snapshot, take_snapshot

__all__ = ["MigrationRefused", "MigrationResult", "StageContext", "run_migration"]


@dataclass(frozen=True)
class StageContext:
    """Handed to ``stage_hook`` after the staging library is built, before the dump.
    A hook may add files under ``staging_dir`` (e.g. ``v2_answers/``); they are
    dumped and swapped in with the library."""

    run_id: str
    v2_home: Path
    v3_home: Path
    staging_dir: Path
    run_dir: Path
    snapshot: Snapshot
    plan: MigrationPlan


@dataclass
class MigrationResult:
    """Everything a report needs to account for the run.

    ``plan`` carries every row written, the attention items, report sections
    (``plan.report``) and the ledger (``plan.ledger``: a disposition for every v2 row).
    ``swapped`` is False when ``swap=False`` was asked or the plan has ``errors``.
    """

    run_id: str
    v2_home: Path
    v3_home: Path
    snapshot_time: str
    staging_dir: Path
    run_dir: Path
    dump_sha256: str
    dump_path: Path
    swapped: bool
    replaced_dir: Path | None
    library_dir: Path               # where the built library is now (live or staging)
    guard_before: v2guard.GuardRecord
    guard_after: v2guard.GuardRecord
    guard: v2guard.GuardResult
    processes: v2guard.ProcessCheck
    counts: dict[str, int]
    graph_counts: dict[str, dict[str, int]]
    attention_count: int
    plan: MigrationPlan
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.guard.status != "fail"

    @property
    def report(self) -> dict[str, Any]:
        return self.plan.report


def new_run_id() -> str:
    return "mig-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def run_migration(
    v2_home: Path | str,
    v3_home: Path | str,
    *,
    graph_names: Mapping[str, str] | None = None,
    replace_dirty: bool = False,
    swap: bool = True,
    run_id: str | None = None,
    now: str | None = None,
    disk_free: Callable[[Path], int] | None = None,
    process_lister: Callable[[], Sequence[v2guard.ProcessInfo]] | None = None,
    stage_hook: Callable[[StageContext], None] | None = None,
) -> MigrationResult:
    """Build a v3 library and graphs from the v2 home and swap them into the v3 home.

    ``graph_names`` maps graph slugs (``did``, ``land-value``) to display names.
    ``swap=False`` builds and dumps into ``migration/staging/<run_id>/`` only (the
    acceptance re-run). ``now`` overrides the snapshot time, ``disk_free`` and
    ``process_lister`` the preflight probes (tests). Raises :class:`MigrationRefused`
    when a preflight check fails or the live library may not be replaced.
    """
    v2_home, v3_home = Path(v2_home), Path(v3_home)
    run_id = run_id or new_run_id()
    warnings: list[str] = []
    if not (v2_home / "cache" / "cache.db").is_file():
        raise MigrationRefused(f"no v2 home at {v2_home} (cache/cache.db missing)")
    if v3_home.resolve() == v2_home.resolve() or v2_home.resolve() in v3_home.resolve().parents:
        raise MigrationRefused("the v3 home must not be the v2 home or inside it")

    # 1. preflight -------------------------------------------------------------------
    half = _files.half_swap(v3_home)
    if half:
        warnings.append(f"a previous swap did not finish: {half}")
    procs = v2guard.check_processes(v2_home, v3_home, lister=process_lister)
    if not procs.ok:
        raise MigrationRefused("; ".join(procs.reasons))
    warnings += procs.warnings
    if swap:
        _files.check_replaceable(v3_home, replace_dirty=replace_dirty)
    _files.check_disk(v3_home, _files.v2_blob_bytes(v2_home), disk_free)

    mig = v3_home / "migration"
    staging = mig / "staging" / run_id
    run_dir = mig / "runs" / run_id
    if staging.exists() or run_dir.exists():
        raise MigrationRefused(f"run {run_id} already exists under {mig}")

    # 2-4. guard, snapshot, plan -------------------------------------------------------
    guard_before = v2guard.guard(v2_home)
    snap = take_snapshot(v2_home, now=now)
    plan = build_plan(snap, graph_names=graph_names)
    warnings += plan.warnings

    # 5-7. files, apply ---------------------------------------------------------------
    staging.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (run_dir / "v2_guard_before.json").write_text(guard_before.to_json(), encoding="utf-8")
    _files.copy_blobs(plan.blobs, snap.cache_root, staging / "library")
    applied = _apply.apply_plan(plan, staging, run_id=run_id, v2_home=v2_home, fingerprint=snap.fingerprint())

    # 8. caller hook (answers export), 9. dump ----------------------------------------------
    if stage_hook is not None:
        stage_hook(StageContext(run_id, v2_home, v3_home, staging, run_dir, snap, plan))
    sha = _dump.write_dump(staging, run_dir)
    _apply.record_dump_sha(staging / "library", run_id, sha)

    # 10. swap --------------------------------------------------------------------------
    errors = list(plan.errors)
    swapped, replaced, library_dir = False, None, staging / "library"
    if swap and not errors:
        res = _files.swap_in(staging, v3_home, run_id=run_id, replace_dirty=replace_dirty)
        swapped, replaced, library_dir = True, res.replaced_dir, v3_home / "library"
    elif swap:
        warnings.append("not swapped: the plan has errors (staging kept for inspection)")

    # 11. guard after -------------------------------------------------------------------
    guard_after = v2guard.guard(v2_home)
    (run_dir / "v2_guard_after.json").write_text(guard_after.to_json(), encoding="utf-8")
    verdict = v2guard.compare(guard_before, guard_after)
    if verdict.status == "fail":
        errors.append("v2 guard FAILED: the v2 home changed during the migration: " + "; ".join(verdict.diffs))
    elif verdict.status == "warn_checkpoint":
        warnings.append("v2 byte hashes changed but logical and tree hashes did not (a WAL checkpoint by "
                        "another process): " + "; ".join(verdict.diffs))

    return MigrationResult(
        run_id=run_id, v2_home=v2_home, v3_home=v3_home, snapshot_time=snap.taken_at, staging_dir=staging,
        run_dir=run_dir, dump_sha256=sha, dump_path=run_dir / "dump.jsonl.gz", swapped=swapped,
        replaced_dir=replaced, library_dir=library_dir, guard_before=guard_before, guard_after=guard_after,
        guard=verdict, processes=procs, counts=applied.counts, graph_counts=applied.graph_counts,
        attention_count=applied.counts.get("attention_items", 0), plan=plan, errors=errors, warnings=warnings,
    )
