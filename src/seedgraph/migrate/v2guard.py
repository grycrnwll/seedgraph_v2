"""Proving the v2 home unchanged, and the v2/v3 process preflight (design §6.1, §6.4).

:func:`guard` records, for the cache and every project DB:

* a **byte hash**: sha256 of ``*.db`` and ``*.db-wal`` (not ``-shm``); a missing file
  is recorded as ``"missing"``;
* a **logical hash**: sha256 of a canonical dump of every table, read in one read
  transaction through :func:`seedgraph.migrate.snapshot.connect_v2_ro`;
* a **tree hash**: sha256 over the sorted ``(relative path, size, sha256)`` of
  ``cache/pdfs``, ``cache/markdown`` and the project files the migration reads
  (``project.yaml``, ``lenses/``, ``answers/``). Mtimes are never used.

:func:`compare` gives ``pass`` (all equal), ``warn_checkpoint`` (only byte hashes
differ: a WAL checkpoint by another process) or ``fail`` (a logical or tree change).

:func:`check_processes` is the preflight. It lists processes (an injectable lister;
the default asks Windows' CIM for every process's command line) and classifies
seedgraph processes by command line: a v2 writer (``serve``, extraction, corpus,
build runners ...) refuses the migration, the read-only v2 MCP server only warns,
and a v3 process naming the v3 home (``serve`` / ``mcp serve``) refuses because
Windows will not rename the library folder while it has handles open. A non-empty
rollback ``-journal`` next to a v2 DB is a write in progress and also refuses. No
lock is ever probed: taking a lock needs a write-mode open of a v2 file.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .snapshot import discover_projects, list_tree, read_db, sha256_file

MISSING = "missing"


def _db_files(v2_home: Path) -> list[str]:
    rels = ["cache/cache.db"]
    rels += [f"projects/{slug}/project.db" for slug in discover_projects(v2_home)]
    return rels


@dataclass
class GuardRecord:
    byte: dict[str, str]
    logical: dict[str, str]
    tree: str
    tree_files: int

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> GuardRecord:
        return cls(**json.loads(text))


@dataclass(frozen=True)
class GuardResult:
    status: str  # 'pass' | 'warn_checkpoint' | 'fail'
    diffs: tuple[str, ...]


def _tree_entries(v2_home: Path) -> list[tuple[str, int, str]]:
    out = [(f"cache/{f.relpath}", f.size, f.sha256)
           for sub in ("pdfs", "markdown") for f in list_tree(v2_home / "cache" / sub, v2_home / "cache")]
    for slug in discover_projects(v2_home):
        pdir = v2_home / "projects" / slug
        files = [pdir / "project.yaml", *sorted((pdir / "lenses").glob("*")), *sorted((pdir / "answers").glob("*"))]
        for p in files:
            if p.is_file():
                out.append((p.relative_to(v2_home).as_posix(), p.stat().st_size, sha256_file(p)))
    return sorted(out)


def guard(v2_home: Path | str) -> GuardRecord:
    v2_home = Path(v2_home)
    byte: dict[str, str] = {}
    logical: dict[str, str] = {}
    for rel in _db_files(v2_home):
        for suffix in ("", "-wal"):
            p = v2_home / (rel + suffix)
            byte[rel + suffix] = sha256_file(p) if p.is_file() else MISSING
        logical[rel] = read_db(v2_home / rel, rel).logical_hash
    entries = _tree_entries(v2_home)
    h = hashlib.sha256()
    for rel, size, sha in entries:
        h.update(f"{rel}|{size}|{sha}\n".encode("utf-8"))
    return GuardRecord(byte=byte, logical=logical, tree=h.hexdigest(), tree_files=len(entries))


def compare(before: GuardRecord, after: GuardRecord) -> GuardResult:
    diffs: list[str] = []
    hard = False
    for rel in sorted(set(before.logical) | set(after.logical)):
        if before.logical.get(rel) != after.logical.get(rel):
            diffs.append(f"logical: {rel}")
            hard = True
    if before.tree != after.tree or before.tree_files != after.tree_files:
        diffs.append("tree: cache/pdfs, cache/markdown or project files changed")
        hard = True
    for rel in sorted(set(before.byte) | set(after.byte)):
        if before.byte.get(rel) != after.byte.get(rel):
            diffs.append(f"bytes: {rel}")
    if hard:
        return GuardResult("fail", tuple(diffs))
    if diffs:
        return GuardResult("warn_checkpoint", tuple(diffs))
    return GuardResult("pass", ())


# --- processes ---------------------------------------------------------------------

@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    cmdline: str
    exe: str | None = None
    ppid: int | None = None


@dataclass
class ProcessCheck:
    v2_writers: list[ProcessInfo] = field(default_factory=list)
    v2_readers: list[ProcessInfo] = field(default_factory=list)
    v3_holders: list[ProcessInfo] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.v2_writers or self.v3_holders or self.reasons)


#: v2 subcommands that write a v2 home (anything under these groups may write).
V2_WRITER_COMMANDS = frozenset({
    "serve", "corpus", "extract", "agent-extract", "lens", "concepts", "graph", "cite", "eval",
    "answer", "ask", "project", "review", "cache", "migrate", "run", "build", "llm",
})
#: v3 commands that hold ``library/`` or ``graphs/`` open.
V3_HOLDER_COMMANDS = frozenset({"serve", "mcp"})


def _tokens(cmdline: str) -> list[str]:
    return [t.strip("\"'").replace("\\", "/") for t in cmdline.split()]


def _seedgraph_command(tokens: Sequence[str]) -> tuple[str | None, list[str]]:
    """``(entry, rest)`` where entry is the seedgraph token (or ``None`` if not seedgraph)."""
    for i, tok in enumerate(tokens):
        base = tok.rsplit("/", 1)[-1].lower()
        if base.endswith(".exe"):
            base = base[:-4]
        if base in ("seedgraph", "seedgraph-v3", "seedgraph.cli", "seedgraph.__main__"):
            return base, [t.lower() for t in tokens[i + 1:]]
        if tok == "-m" and i + 1 < len(tokens) and tokens[i + 1].lower().startswith("seedgraph"):
            return tokens[i + 1].lower(), [t.lower() for t in tokens[i + 2:]]
    return None, []


def _subcommands(rest: Sequence[str]) -> list[str]:
    return [t for t in rest if not t.startswith("-")]


def classify(proc: ProcessInfo, v2_home: Path, v3_home: Path) -> str | None:
    """``'v2_writer'``, ``'v2_reader'``, ``'v3_holder'`` or ``None`` (not relevant)."""
    tokens = _tokens(proc.cmdline or "")
    entry, rest = _seedgraph_command(tokens)
    if entry is None:
        return None
    subs = _subcommands(rest)
    low = (proc.cmdline or "").replace("\\", "/").lower()
    v3_marker = str(Path(v3_home)).replace("\\", "/").lower()
    is_v3 = entry == "seedgraph-v3" or (v3_marker and v3_marker in low)
    if is_v3:
        return "v3_holder" if subs and subs[0] in V3_HOLDER_COMMANDS else None
    if not subs:
        return None
    if subs[0] == "mcp":
        return "v2_reader"
    if subs[0] in V2_WRITER_COMMANDS:
        return "v2_writer"
    return None


def default_process_lister() -> list[ProcessInfo]:
    """Every process's command line. Windows: CIM via PowerShell; elsewhere ``ps``."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine,"
                 "ExecutablePath | ConvertTo-Json -Compress"],
                capture_output=True, text=True, timeout=60, check=False).stdout
            data = json.loads(out or "[]")
            if isinstance(data, dict):
                data = [data]
            return [ProcessInfo(int(d["ProcessId"]), d.get("CommandLine") or "", d.get("ExecutablePath"),
                                int(d["ParentProcessId"]) if d.get("ParentProcessId") is not None else None)
                    for d in data]
        out = subprocess.run(["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True,
                             timeout=60, check=False).stdout
        procs = []
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                procs.append(ProcessInfo(int(parts[0]), parts[2] if len(parts) > 2 else "", None, int(parts[1])))
        return procs
    except (OSError, ValueError, subprocess.SubprocessError):
        return []


def check_processes(v2_home: Path | str, v3_home: Path | str, *,
                    lister: Callable[[], Sequence[ProcessInfo]] | None = None) -> ProcessCheck:
    v2_home, v3_home = Path(v2_home), Path(v3_home)
    chk = ProcessCheck()
    procs = list((lister or default_process_lister)())
    if not procs:
        chk.warnings.append("the process list is unavailable: running v2 writers could not be checked "
                            "(the v2 guard hashes remain the gate)")
    # This process and its ancestors (the seedgraph launcher, uv, a shell) are the migration itself.
    parent = {p.pid: p.ppid for p in procs}
    mine = {os.getpid(), os.getppid()}
    cur = os.getppid()
    while cur in parent and parent[cur] is not None and parent[cur] not in mine:
        cur = parent[cur]
        mine.add(cur)
    for proc in procs:
        if proc.pid in mine:
            continue
        kind = classify(proc, v2_home, v3_home)
        if kind == "v2_writer":
            chk.v2_writers.append(proc)
        elif kind == "v2_reader":
            chk.v2_readers.append(proc)
        elif kind == "v3_holder":
            chk.v3_holders.append(proc)
    for p in chk.v2_writers:
        chk.reasons.append(f"a v2 writer is running (pid {p.pid}): {p.cmdline}")
    for p in chk.v3_holders:
        chk.reasons.append(f"a v3 process holds the library (pid {p.pid}): {p.cmdline}")
    for p in chk.v2_readers:
        chk.warnings.append(f"read-only v2 reader running (pid {p.pid}): {p.cmdline}")
    if v2_home.is_dir():
        for rel in _db_files(v2_home):
            journal = v2_home / (rel + "-journal")
            if journal.is_file() and journal.stat().st_size > 0:
                chk.reasons.append(f"hot rollback journal (a v2 write in progress): {rel}-journal")
    return chk
