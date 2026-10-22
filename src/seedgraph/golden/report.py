"""Terminal table + JSON/markdown report for a golden-set check.

Reports go to ``<v3 home>/golden_reports/<UTC timestamp>/`` (outside the repo).
They contain IDs, counts and rates only, never paper text.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .checks import PdfResult

_COLS = ["page_markers", "references", "quotes", "equations", "tables", "speed", "similarity"]
_MARK = {"pass": "ok", "fail": "FAIL", "warn": "warn"}


def _fmt(name: str, value) -> str:
    if value is None:
        return "-"
    if name == "quotes":
        return f"{value:.1%}"
    if name == "speed":
        return f"{value:.2f}"
    if name == "similarity":
        return f"{value:.4f}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _cell(result: PdfResult, name: str) -> str:
    c = next((x for x in result.checks if x.name == name), None)
    if c is None:  # source failed before any check ran
        return "-"
    if name == "similarity":
        body = _fmt(name, c.new)
    else:
        body = f"{_fmt(name, c.old)}->{_fmt(name, c.new)}"
    return f"{body} {_MARK[c.status]}"


def render_table(results: list[PdfResult]) -> str:
    header = ["id", "status", *_COLS]
    rows = [[r.golden_id, r.status.upper(), *(_cell(r, n) for n in _COLS)] for r in results]
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda cells: " | ".join(str(c).ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    out = [line(header), "-+-".join("-" * w for w in widths)]
    out += [line(r) for r in rows]
    return "\n".join(out)


def _markdown(results: list[PdfResult], meta: dict) -> str:
    lines = ["# Golden-set check", ""]
    lines += [f"- {k}: {v}" for k, v in meta.items()]
    lines += ["", "| id | status | " + " | ".join(_COLS) + " |", "|" + "---|" * (len(_COLS) + 2)]
    for r in results:
        lines.append(f"| {r.golden_id} | {r.status} | " + " | ".join(_cell(r, n) for n in _COLS) + " |")
    lines += ["", "## Reasons", ""]
    for r in results:
        for c in r.checks:
            if c.status != "pass":
                lines.append(f"- {r.golden_id} {c.name} [{c.status}]: {c.reason}")
    lines.append("")
    lines.append("Speed is seconds per PDF page; OLD v2-stored timing includes v2's per-paper model load.")
    return "\n".join(lines) + "\n"


def write_report(results: list[PdfResult], home: Path | str, *, meta: Optional[dict] = None) -> Path:
    meta = dict(meta or {})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = Path(home) / "golden_reports" / stamp
    out.mkdir(parents=True, exist_ok=False)
    meta.setdefault("created_at", stamp)
    payload = {"meta": meta, "overall": "fail" if any(r.status == "fail" for r in results) else "pass",
               "results": [r.to_dict() for r in results]}
    (out / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_markdown(results, meta), encoding="utf-8")
    return out
