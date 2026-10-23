"""Filesystem side of the migration: disk check, blob copy, and the swap (design §1.3, §6.1, §6.5).

* :func:`check_disk` refuses unless the v3 home's volume has at least
  ``2 x (pdfs + markdown) + 1 GB`` free (the free-space probe is injectable).
* :func:`copy_blobs` copies each PDF and markdown blob into the staging library
  through :func:`seedgraph.cache.store.copy_file_atomic` (a byte copy, **never** a
  hardlink, so v3 can never alter a v2 file) and verifies the sha256.
* :func:`swap_in` replaces the live ``library/``, ``graphs/`` and ``v2_answers/`` with
  the staged ones. It never lists, moves or deletes anything else in the home
  (``golden_set.json``, ``golden_reports/``, unknown files). Replaced folders are
  moved to ``migration/replaced/<timestamp>-<run_id>/``, never deleted. A live
  library that is not a clean migration product is refused unless ``replace_dirty``
  - except an *empty, never-written* one (see :attr:`LiveState.empty_unwritten`):
  the library ``seedgraph setup`` creates, which holds nothing to lose.
  A marker file makes a crash between renames visible on the next run.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from seedgraph.cache.store import copy_file_atomic

from .plan import BlobCopy
from .snapshot import connect_v2_ro, sha256_file

GB = 1024 ** 3
OWNED_FOLDERS: tuple[str, ...] = ("library", "graphs", "v2_answers")
SWAP_MARKER = "swap_in_progress.json"


class MigrationRefused(RuntimeError):
    """The migration refused to run or to swap (the message says why)."""


# --- disk ---------------------------------------------------------------------------

def _existing_ancestor(path: Path) -> Path:
    path = Path(path).resolve()
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def default_disk_free(path: Path) -> int:
    return shutil.disk_usage(_existing_ancestor(path)).free


def required_bytes(blob_bytes: int) -> int:
    return 2 * blob_bytes + GB


def check_disk(v3_home: Path, blob_bytes: int, disk_free: Callable[[Path], int] | None = None) -> int:
    """Refuse (:class:`MigrationRefused`) unless enough space is free; return the free bytes."""
    free = (disk_free or default_disk_free)(Path(v3_home))
    need = required_bytes(blob_bytes)
    if free < need:
        raise MigrationRefused(
            f"not enough free disk space on the v3 home's volume: {free} bytes free, {need} needed "
            f"(2 x {blob_bytes} bytes of PDFs and markdown + 1 GB)")
    return free


def v2_blob_bytes(v2_home: Path) -> int:
    total = 0
    for sub in ("pdfs", "markdown"):
        root = Path(v2_home) / "cache" / sub
        if root.is_dir():
            total += sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return total


# --- blobs --------------------------------------------------------------------------

class BlobError(RuntimeError):
    pass


def copy_blobs(blobs: Sequence[BlobCopy], v2_cache_root: Path, lib_dir: Path) -> int:
    """Copy (never hardlink) each blob and verify its sha256; return the bytes copied."""
    total = 0
    for b in blobs:
        src = Path(v2_cache_root) / b.src
        dest = Path(lib_dir) / b.dest
        copy_file_atomic(src, dest)
        got = sha256_file(dest)
        if got != b.sha256 or dest.stem != b.sha256:
            raise BlobError(f"copied blob {b.dest} has sha256 {got}, expected {b.sha256}")
        total += b.size
    return total


# --- swap ---------------------------------------------------------------------------

#: Origins of a library nobody migrated into: ``setup`` (made by ``seedgraph setup``) and
#: ``local`` (any other first open, including an older ``setup``).
_UNMIGRATED_ORIGINS = ("setup", "local")
#: Bookkeeping tables that exist in every library, empty or not.
_SCHEMA_TABLES = ("library_meta", "schema_migrations")
_FTS_SHADOW_SUFFIXES = ("_data", "_idx", "_content", "_docsize", "_config")
_DB_FILES = ("library.db", "library.db-wal", "library.db-shm", "library.db-journal")


@dataclass(frozen=True)
class LiveState:
    present: tuple[str, ...]        # which owned folders exist in the home
    origin: str | None
    dirty: str | None
    migration_run_id: str | None
    empty_unwritten: bool = False   # see the property docs below

    @property
    def clean_migration(self) -> bool:
        return self.origin == "migration" and self.dirty == "0"

    @property
    def replaceable(self) -> bool:
        """A clean migration product, or an empty never-written library: origin
        ``setup``/``local``, ``dirty = '0'`` (every normal write sets it), no row in any
        table but the schema/FTS bookkeeping, and no file in the owned folders but the
        database itself."""
        return self.clean_migration or self.empty_unwritten


def _data_rows(conn) -> int:
    master = conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()
    fts = [n for n, sql in master if (sql or "").upper().startswith("CREATE VIRTUAL TABLE")]
    total = 0
    for name, _ in master:
        if name.startswith("sqlite_") or name in _SCHEMA_TABLES:
            continue
        if any(name.startswith(f) and name[len(f):] in _FTS_SHADOW_SUFFIXES for f in fts):
            continue
        total += conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    return total


def _only_the_database(v3_home: Path, present: Sequence[str]) -> bool:
    for name in present:
        folder = v3_home / name
        for f in folder.rglob("*"):
            if f.is_file() and not (name == "library" and f.parent == folder and f.name in _DB_FILES):
                return False
    return True


def live_state(v3_home: Path) -> LiveState:
    """Read the live library's meta **read-only** (never migrates or creates it)."""
    v3_home = Path(v3_home)
    present = tuple(n for n in OWNED_FOLDERS if (v3_home / n).exists())
    db = v3_home / "library" / "library.db"
    origin = dirty = run = None
    empty = False
    if db.is_file():
        conn = connect_v2_ro(db)  # the generic read-only open (the live library is never written here)
        try:
            meta = {k: v for k, v in conn.execute("SELECT key, value FROM library_meta")}
            origin, dirty, run = meta.get("origin"), meta.get("dirty"), meta.get("migration_run_id")
            if origin in _UNMIGRATED_ORIGINS and dirty == "0":
                empty = _data_rows(conn) == 0 and _only_the_database(v3_home, present)
        finally:
            conn.close()
    return LiveState(present, origin, dirty, run, empty)


def half_swap(v3_home: Path) -> dict | None:
    """The marker of a swap that did not finish, if any."""
    marker = Path(v3_home) / "migration" / SWAP_MARKER
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    return None


def check_replaceable(v3_home: Path, *, replace_dirty: bool) -> LiveState:
    state = live_state(v3_home)
    if state.present and not state.replaceable and not replace_dirty:
        what = ", ".join(f"{n}/" for n in state.present)
        raise MigrationRefused(
            f"the live library is not a clean migration product or an empty, never-written library "
            f"(origin={state.origin!r}, dirty={state.dirty!r}); "
            f"refusing to replace it. --replace-dirty would move {what} into migration/replaced/.")
    return state


@dataclass(frozen=True)
class SwapResult:
    replaced_dir: Path | None
    moved_aside: tuple[str, ...]
    swapped_in: tuple[str, ...]


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def swap_in(staging_dir: Path, v3_home: Path, *, run_id: str, replace_dirty: bool = False) -> SwapResult:
    """Move the live owned folders aside and rename the staged ones into place (§1.3)."""
    v3_home, staging_dir = Path(v3_home), Path(staging_dir)
    check_replaceable(v3_home, replace_dirty=replace_dirty)
    staged = tuple(n for n in OWNED_FOLDERS if (staging_dir / n).exists())
    live = tuple(n for n in OWNED_FOLDERS if (v3_home / n).exists())
    mig = v3_home / "migration"
    replaced_dir = mig / "replaced" / f"{_timestamp()}-{run_id}" if live else None
    marker = mig / SWAP_MARKER
    marker.write_text(json.dumps({"run_id": run_id, "staging": str(staging_dir), "moving_aside": list(live),
                                  "replaced_dir": str(replaced_dir) if replaced_dir else None,
                                  "swapping_in": list(staged)}, indent=2), encoding="utf-8")
    if replaced_dir is not None:
        replaced_dir.mkdir(parents=True, exist_ok=False)
        for name in live:
            os.replace(v3_home / name, replaced_dir / name)
    for name in staged:
        os.replace(staging_dir / name, v3_home / name)
    marker.unlink()
    try:
        staging_dir.rmdir()  # only when empty
    except OSError:
        pass
    return SwapResult(replaced_dir, live, staged)
