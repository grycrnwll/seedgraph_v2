"""The final acceptance run (design §11, "Final real-data acceptance run"; gate 2).

:func:`accept` runs, in order, and always shows every step (a step that cannot run
because an earlier one failed is ``skipped``):

1. **preflight** - no v2 writer, no v3 process holding the library, the live library
   replaceable, enough disk;
2. **migrate run** - :func:`seedgraph.migrate.report.run_and_report` (swap in, answers
   export, report);
3. **re-run identical** - a second from-scratch run into a fresh staging folder
   (``swap=False``) must give the same canonical dump sha;
4. **v2 guard** - both runs' guards, plus a reading taken before step 1 compared
   with one taken after step 5 (so the parity capture is covered too);
5. **parity** - :mod:`.parity` against ``migration/parity/queries.json`` (the v2
   capture is taken, or refreshed, when missing or stale);
6. **expected values** - ``report.json`` against the researcher-owned
   ``migration/acceptance/expected.json`` (:func:`compare_expected`).

It prints one pass/fail table (through ``echo``; pass ``echo=None`` to print
nothing) and writes ``runs/<run_id>/acceptance.md``.

``expected.json`` format (all numbers are the researcher's; none live in code)::

    {"totals":    {"<report.json totals key>": {"expected": 489, "tolerance": 5}, ...},
     "attention": {"<attention kind>": {"expected": 7, "tolerance": 0}, ...},
     "allow_other_attention_kinds": false}

A bare number means ``{"expected": n, "tolerance": 0}``; ``tolerance_pct`` is a
percentage alternative. A key missing from the report fails, and so does an
attention kind the report has but ``expected.json`` does not list (unless
``allow_other_attention_kinds``). :func:`write_expected` writes a starter file from
values you give it; :func:`expected_template` lists the keys with empty values.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import files as _files
from . import parity as _parity
from . import report as _report
from . import v2guard
from .files import MigrationRefused

EXPECTED_FILE = "migration/acceptance/expected.json"
ACCEPTANCE_MD = "acceptance.md"
STEPS: tuple[str, ...] = ("preflight", "migrate run", "re-run identical", "v2 guard", "parity", "expected values")

#: ``report.json`` totals the spec's expectations speak about (ticket 13 facts, design
#: §6.2): graph sizes, folded-in papers, every cached PDF in the library, merges,
#: decisions carried, answers exported, not-converted papers. Values are the
#: researcher's to fill in.
STARTER_TOTALS: tuple[str, ...] = (
    "graph.did.member", "graph.did.frontier", "graph.did.folded_in", "graph.land-value.member",
    "graph.land-value.excluded", "library.papers", "library.source_files", "v2.cache.source_files.rows",
    "library.structured_notes", "merges.papers", "current_elsewhere", "not_converted",
    "concepts.merge.researcher", "concepts.never_merge.researcher", "concepts.merge.agent",
    "concepts.contradictions", "citations.review_decisions", "library.duplicate_decisions",
    "library.lens_outputs", "answers.exported", "attention.total", "attention.hides_notes", "errors",
    "unaccounted",
)
STARTER_ATTENTION: tuple[str, ...] = tuple(_report.ATTENTION_KINDS)


class AcceptanceError(RuntimeError):
    """An acceptance input is unusable (e.g. a malformed or existing expected.json)."""


@dataclass(frozen=True)
class StepRow:
    step: str
    status: str      # 'pass' | 'fail' | 'skipped'
    detail: str


@dataclass(frozen=True)
class Check:
    name: str
    expected: Any
    actual: Any
    tolerance: str
    ok: bool


@dataclass
class AcceptanceResult:
    run_id: str
    rows: list[StepRow] = field(default_factory=list)
    path: Path | None = None
    outcome: _report.RunOutcome | None = None
    rerun: Any = None                       # the swap=False MigrationResult
    parity: _parity.ParityResult | None = None
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.rows) and all(r.status == "pass" for r in self.rows)

    def table(self) -> str:
        return render_table(self.rows)


# --- expected.json ----------------------------------------------------------------------

def _spec(value: Any, tolerance: float = 0) -> dict[str, Any]:
    if isinstance(value, Mapping):
        out = dict(value)
        if "tolerance_pct" not in out:
            out.setdefault("tolerance", tolerance)
        return out
    return {"expected": value, "tolerance": tolerance}


def write_expected(v3_home: Path | str, *, totals: Mapping[str, Any], attention: Mapping[str, Any],
                   tolerance: float = 0, allow_other_attention_kinds: bool = False, force: bool = False) -> Path:
    """Write a starter ``expected.json`` from values you supply (none are built in).

    ``totals`` maps ``report.json`` totals keys (see :data:`STARTER_TOTALS`) and
    ``attention`` maps attention kinds to a number or ``{"expected", "tolerance" |
    "tolerance_pct"}``. A bare number gets ``tolerance``. Refuses to overwrite the
    researcher's file unless ``force``.
    """
    path = Path(v3_home) / EXPECTED_FILE
    if path.exists() and not force:
        raise AcceptanceError(f"{path} exists (researcher-owned); pass force=True to replace it")
    data = {"totals": {k: _spec(v, tolerance) for k, v in sorted(totals.items())},
            "attention": {k: _spec(v, tolerance) for k, v in sorted(attention.items())},
            "allow_other_attention_kinds": bool(allow_other_attention_kinds)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def expected_template() -> dict[str, Any]:
    """Every starter key with ``expected: null`` (fill in, then save as expected.json)."""
    return {"totals": {k: {"expected": None, "tolerance": 0} for k in STARTER_TOTALS},
            "attention": {k: {"expected": None, "tolerance": 0} for k in STARTER_ATTENTION},
            "allow_other_attention_kinds": False}


def load_expected(v3_home: Path | str) -> dict[str, Any]:
    path = Path(v3_home) / EXPECTED_FILE
    if not path.is_file():
        raise AcceptanceError(f"no {path}: write it with your expected counts "
                              "(seedgraph.migrate.accept.write_expected) before accepting")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("totals", {}), dict) \
            or not isinstance(data.get("attention", {}), dict):
        raise AcceptanceError(f"{path}: expected an object with 'totals' and 'attention' maps")
    return data


def _one(name: str, spec: Any, actual: Any) -> Check:
    s = _spec(spec)
    exp = s.get("expected")
    if "tolerance_pct" in s:
        tol = abs(float(exp or 0)) * float(s["tolerance_pct"]) / 100.0
        tol_txt = f"±{s['tolerance_pct']}%"
    else:
        tol = float(s.get("tolerance", 0) or 0)
        tol_txt = f"±{s.get('tolerance', 0)}"
    if exp is None or actual is None or not isinstance(actual, (int, float)):
        return Check(name, exp, actual, tol_txt, False)
    return Check(name, exp, actual, tol_txt, abs(actual - float(exp)) <= tol)


def compare_expected(report: Mapping[str, Any], expected: Mapping[str, Any]) -> list[Check]:
    """``report.json`` against ``expected.json``: one :class:`Check` per expectation, plus a
    failing check for every attention kind the report has but ``expected`` omits."""
    checks = []
    totals = report.get("totals", {})
    for key, spec in sorted(expected.get("totals", {}).items()):
        checks.append(_one(key, spec, totals.get(key)))
    by_kind = report.get("attention", {}).get("by_kind", {})
    exp_att = expected.get("attention", {})
    for kind, spec in sorted(exp_att.items()):
        checks.append(_one(f"attention.{kind}", spec, by_kind.get(kind, 0)))
    if not expected.get("allow_other_attention_kinds", False):
        for kind in sorted(set(by_kind) - set(exp_att)):
            checks.append(Check(f"attention.{kind}", "not listed", by_kind[kind], "-", False))
    return checks


# --- rendering ----------------------------------------------------------------------------

def render_table(rows: Sequence[StepRow]) -> str:
    w = max([len(s) for s in STEPS] + [len(r.step) for r in rows])
    lines = [f"{'step'.ljust(w)}  result   detail", f"{'-' * w}  -------  {'-' * 40}"]
    for r in rows:
        lines.append(f"{r.step.ljust(w)}  {r.status.upper().ljust(7)}  {r.detail}")
    ok = bool(rows) and all(r.status == "pass" for r in rows)
    lines.append(f"\nACCEPTANCE {'PASSED' if ok else 'FAILED'}")
    return "\n".join(lines)


def render_acceptance_md(res: AcceptanceResult) -> str:
    L = [f"# Acceptance: {res.run_id}", "", f"**{'PASSED' if res.ok else 'FAILED'}**", "",
         "| step | result | detail |", "|---|---|---|"]
    for r in res.rows:
        L.append(f"| {r.step} | {r.status.upper()} | {r.detail.replace('|', '/')} |")
    if res.checks:
        L += ["", "## Expected values", "", "| check | expected | actual | tolerance | result |",
              "|---|---|---|---|---|"]
        for c in res.checks:
            L.append(f"| {c.name} | {c.expected} | {c.actual} | {c.tolerance} | {'PASS' if c.ok else 'FAIL'} |")
    L += ["", "Files: `report.md`, `attention.md`, `parity.md` in this folder; the expected values are "
          f"`{EXPECTED_FILE}`."]
    return "\n".join(L) + "\n"


# --- the run --------------------------------------------------------------------------------

def accept(
    v2_home: Path | str,
    v3_home: Path | str,
    *,
    graph_names: Mapping[str, str] | None = None,
    replace_dirty: bool = False,
    run_id: str | None = None,
    now: str | None = None,
    disk_free: Callable[[Path], int] | None = None,
    process_lister: Callable[[], Sequence[v2guard.ProcessInfo]] | None = None,
    echo: Callable[[str], Any] | None = print,
) -> AcceptanceResult:
    """Run the six acceptance steps; print the table via ``echo``; write ``acceptance.md``."""
    from . import new_run_id, run_migration

    v2_home, v3_home = Path(v2_home), Path(v3_home)
    run_id = run_id or new_run_id()
    res = AcceptanceResult(run_id=run_id)
    rows: dict[str, StepRow] = {}

    def put(step: str, ok: bool, detail: str) -> None:
        rows[step] = StepRow(step, "pass" if ok else "fail", detail)

    # 1. preflight -------------------------------------------------------------------
    guard_start = None
    try:
        if not (v2_home / "cache" / "cache.db").is_file():
            raise MigrationRefused(f"no v2 home at {v2_home} (cache/cache.db missing)")
        procs = v2guard.check_processes(v2_home, v3_home, lister=process_lister)
        if not procs.ok:
            raise MigrationRefused("; ".join(procs.reasons))
        _files.check_replaceable(v3_home, replace_dirty=replace_dirty)
        _files.check_disk(v3_home, _files.v2_blob_bytes(v2_home), disk_free)
        guard_start = v2guard.guard(v2_home)
        put("preflight", True, "no v2 writer, no v3 process holding the library, disk space ok"
            + (f" ({len(procs.warnings)} warnings)" if procs.warnings else ""))
    except MigrationRefused as exc:
        put("preflight", False, str(exc))

    # 2. migrate run ------------------------------------------------------------------
    run_kw = dict(graph_names=graph_names, now=now, disk_free=disk_free, process_lister=process_lister)
    if rows["preflight"].status == "pass":
        try:
            res.outcome = _report.run_and_report(v2_home, v3_home, replace_dirty=replace_dirty, run_id=run_id,
                                                 **run_kw)
            r = res.outcome.result
            put("migrate run", res.outcome.ok and r.swapped,
                f"swapped in; dump {r.dump_sha256[:12]}; {r.attention_count} attention items"
                if res.outcome.ok and r.swapped else
                f"errors: {'; '.join(r.errors) or 'not swapped'}")
        except _report.ReportError as exc:
            put("migrate run", False, f"report does not account for every item: {exc}")
            if exc.result is not None:
                res.outcome = _report.RunOutcome(exc.result, exc.files) if exc.files else None
        except MigrationRefused as exc:
            put("migrate run", False, f"refused: {exc}")

    # 3. re-run identical --------------------------------------------------------------
    if res.outcome is not None and rows.get("migrate run", StepRow("", "fail", "")).status == "pass":
        rerun_id = f"{run_id}-rerun"
        try:
            res.rerun = run_migration(v2_home, v3_home, swap=False, run_id=rerun_id,
                                      stage_hook=_report.migration_hook, **run_kw)
            a, b = res.outcome.result.dump_sha256, res.rerun.dump_sha256
            same = a == b
            if same:
                shutil.rmtree(res.rerun.staging_dir, ignore_errors=True)  # a full library copy; the dump stays
            put("re-run identical", same, f"dump sha {a[:12]} == {b[:12]}" if same else
                f"dump sha differs: {a[:12]} vs {b[:12]} (`seedgraph migrate verify --against {rerun_id}`; "
                f"staging kept at {res.rerun.staging_dir})")
        except MigrationRefused as exc:
            put("re-run identical", False, f"refused: {exc}")

    # 5. parity -----------------------------------------------------------------------
    if res.outcome is not None and rows.get("migrate run", StepRow("", "fail", "")).status == "pass":
        try:
            queries = _parity.load_queries(v3_home)
            note = ""
            try:
                expected = _parity.load_expected(v3_home)
                fresh = _parity.expected_is_current(v2_home, expected, queries)
            except _parity.ParityError:
                fresh = False
            if not fresh:
                _parity.capture(v2_home, v3_home, queries=queries)
                note = "v2 results captured now; "
            res.parity = _parity.check(v3_home, run_id=run_id)
            p = res.parity
            put("parity", p.ok, note + f"{len(p.differences)} differences, {len(p.unexplained)} unexplained "
                f"({', '.join(f'{k} {n}' for k, n in p.by_class().items()) or 'none'})")
        except _parity.ParityError as exc:
            put("parity", False, str(exc))

    # 4. v2 guard ------------------------------------------------------------------------
    if guard_start is not None and res.outcome is not None:
        statuses = [res.outcome.result.guard.status]
        if res.rerun is not None:
            statuses.append(res.rerun.guard.status)
        overall = v2guard.compare(guard_start, v2guard.guard(v2_home))
        statuses.append(overall.status)
        failed = "fail" in statuses
        warn = "warn_checkpoint" in statuses
        put("v2 guard", not failed, "v2 home changed: " + "; ".join(overall.diffs) if failed else
            ("unchanged (a WAL checkpoint by another process was seen)" if warn else "v2 home unchanged"))

    # 6. expected values -------------------------------------------------------------------
    if res.outcome is not None and rows.get("migrate run", StepRow("", "fail", "")).status == "pass":
        try:
            expected = load_expected(v3_home)
            report = _report.load_report(res.outcome.result.run_dir)
            res.checks = compare_expected(report, expected)
            bad = [c for c in res.checks if not c.ok]
            put("expected values", not bad, f"{len(res.checks)} checks, all within tolerance" if not bad else
                "; ".join(f"{c.name}: expected {c.expected} ({c.tolerance}), got {c.actual}" for c in bad))
        except AcceptanceError as exc:
            put("expected values", False, str(exc))

    for step in STEPS:
        if step not in rows:
            rows[step] = StepRow(step, "skipped", "not run: an earlier step failed")
    res.rows = [rows[s] for s in STEPS]
    run_dir = v3_home / "migration" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    res.path = run_dir / ACCEPTANCE_MD
    res.path.write_bytes(render_acceptance_md(res).encode("utf-8"))
    if echo is not None:
        echo(res.table())
    return res
