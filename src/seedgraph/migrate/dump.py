"""The canonical dump (design §4.5): proof that two runs over unchanged v2 data agree.

For ``library/library.db`` and then each ``graphs/<slug>/graph.db`` (slug order), in
table-name order, one JSON line per row ``ORDER BY`` the primary key (all columns
when there is none). FTS tables are dumped by content (no rowid), their shadow
tables are skipped, and so is everything in :data:`EXCLUDED` (run-specific values:
the snapshot time and wall-clock bookkeeping). Then a manifest of
``library/pdfs``, ``library/markdown`` and ``v2_answers``: relative path, size and
sha256, sorted. ``provider_cache.db`` is a disposable cache and is not dumped.

``dump.sha256`` is the sha256 of the uncompressed canonical bytes; the ``.gz`` is
written with ``mtime=0`` so it is byte-identical too.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

from seedgraph.library.store import connect

from .snapshot import sha256_file

#: Everything left out of the dump, explicitly. Forms: ``table`` (whole table),
#: ``table.column`` (one column of every row), ``table[key=value]`` (one row).
EXCLUDED: tuple[str, ...] = (
    "migration_runs",                       # run bookkeeping (wall clock, run id)
    "schema_migrations.applied_at",         # when the numbered schema was applied
    "library_meta[key=migration_run_id]",   # the run id
    "attention_items.created_at",           # snapshot time
    "conversion_queue.queued_at",           # snapshot time
)

FTS_SHADOW_SUFFIXES = ("_data", "_idx", "_content", "_docsize", "_config")
MANIFEST_DIRS = ("library/pdfs", "library/markdown", "v2_answers")


def _parse_excluded() -> tuple[set[str], dict[str, set[str]], dict[str, set[tuple[str, str]]]]:
    tables: set[str] = set()
    columns: dict[str, set[str]] = {}
    rows: dict[str, set[tuple[str, str]]] = {}
    for item in EXCLUDED:
        if "[" in item:
            t, cond = item[:-1].split("[", 1)
            k, v = cond.split("=", 1)
            rows.setdefault(t, set()).add((k, v))
        elif "." in item:
            t, c = item.split(".", 1)
            columns.setdefault(t, set()).add(c)
        else:
            tables.add(item)
    return tables, columns, rows


def _value(v):
    if isinstance(v, (bytes, bytearray, memoryview)):
        return {"$bytes": bytes(v).hex()}
    return v


def dump_db_lines(db_path: Path, label: str) -> Iterator[str]:
    ex_tables, ex_cols, ex_rows = _parse_excluded()
    conn = connect(db_path)
    try:
        master = conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
        fts = {name for name, sql in master if (sql or "").upper().startswith("CREATE VIRTUAL TABLE")}
        for name, _ in master:
            if name.startswith("sqlite_") or name in ex_tables:
                continue
            if any(name.startswith(f) and name[len(f):] in FTS_SHADOW_SUFFIXES for f in fts):
                continue
            info = conn.execute(f'PRAGMA table_info("{name}")').fetchall()
            cols = [r[1] for r in info if r[1] not in ex_cols.get(name, set())]
            pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
            order = pk if pk and name not in fts else cols
            sql = (f'SELECT {", ".join(chr(34) + c + chr(34) for c in cols)} FROM "{name}" '
                   f'ORDER BY {", ".join(chr(34) + c + chr(34) for c in order)}')
            drop = ex_rows.get(name, set())
            for r in conn.execute(sql):
                row = {c: _value(v) for c, v in zip(cols, r)}
                if any(str(row.get(k)) == v for k, v in drop):
                    continue
                yield json.dumps({"db": label, "table": name, "row": row}, sort_keys=True, ensure_ascii=False)
    finally:
        conn.close()


def manifest_lines(root: Path) -> Iterator[str]:
    for sub in MANIFEST_DIRS:
        base = root / sub
        if not base.is_dir():
            continue
        for p in sorted((p for p in base.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
            yield json.dumps({"manifest": sub, "path": p.relative_to(root).as_posix(), "size": p.stat().st_size,
                              "sha256": sha256_file(p)}, sort_keys=True, ensure_ascii=False)


def canonical_lines(root: Path) -> Iterator[str]:
    """Every dump line for the build rooted at ``root`` (holding ``library/`` and ``graphs/``)."""
    root = Path(root)
    yield from dump_db_lines(root / "library" / "library.db", "library")
    groot = root / "graphs"
    if groot.is_dir():
        for child in sorted(groot.iterdir(), key=lambda p: p.name):
            if (child / "graph.db").is_file():
                yield from dump_db_lines(child / "graph.db", f"graph:{child.name}")
    yield from manifest_lines(root)


def write_dump(root: Path, run_dir: Path) -> str:
    """Write ``dump.jsonl.gz`` and ``dump.sha256`` into ``run_dir``; return the sha."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(run_dir / "dump.jsonl.gz", "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for line in canonical_lines(root):
            data = (line + "\n").encode("utf-8")
            h.update(data)
            gz.write(data)
    sha = h.hexdigest()
    (run_dir / "dump.sha256").write_text(sha + "\n", encoding="utf-8")
    return sha


def first_difference(a: Path, b: Path) -> str | None:
    """The first differing line of two ``dump.jsonl.gz`` files (``None`` when equal)."""
    with gzip.open(a, "rt", encoding="utf-8") as fa, gzip.open(b, "rt", encoding="utf-8") as fb:
        n = 0
        while True:
            la, lb = fa.readline(), fb.readline()
            n += 1
            if la != lb:
                return f"line {n}: {la.strip()[:300]!r} != {lb.strip()[:300]!r}"
            if not la:
                return None

