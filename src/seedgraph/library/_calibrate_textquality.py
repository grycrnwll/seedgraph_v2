"""Calibration run for :mod:`seedgraph.library.textquality` (design §5.2, slice P3).

Not part of the product and not collected by pytest. Reads a v3 home strictly read-only
(``sqlite3`` ``mode=ro`` URI, markdown blobs opened for reading) and writes only into
``--out``, which must be outside the repo (default ``<home>/work/calibration/textquality``):
the output contains real text excerpts.

    uv run python -m seedgraph.library._calibrate_textquality [--home <v3 home>] [--out <dir>]

Universe: every current markdown (``paper_files.markdown_id``) plus the golden-set papers
by ``v2_markdown_id``. Each is scored reference-stripped, by its
``document_sections.section_kind = 'references'`` ranges.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

from seedgraph.library import textquality as tq

REPO_ROOT = Path(__file__).resolve().parents[3]
LIGATURE_SIGNALS = set("Ö§¢…¤")
GARBAGE_SIGNALS = {"U+FFFD", "(cid:N)", "private_use"}


def _signal_class(hits_by_signal: dict) -> str:
    """'none', 'ligature', 'garbage' or 'mixed', by the strong signals present."""
    sigs = {k for k, v in hits_by_signal.items() if v and k not in ("á", "í")}
    lig, junk = bool(sigs & LIGATURE_SIGNALS), bool(sigs & GARBAGE_SIGNALS)
    return {(False, False): "none", (True, False): "ligature",
            (False, True): "garbage", (True, True): "mixed"}[(lig, junk)]


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _bucket(rate: float) -> str:
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.4, 2.0, 5.0, 10.0]
    if rate == 0:
        return "0"
    for lo, hi in zip(edges, edges[1:]):
        if lo < rate <= hi:
            return f"({lo}, {hi}]"
    return f"> {edges[-1]}"


def run(home: Path, out: Path) -> dict:
    out = out.resolve()
    if REPO_ROOT in out.parents or out == REPO_ROOT:
        raise SystemExit("--out must be outside the repo (output contains real text)")
    out.mkdir(parents=True, exist_ok=True)
    lib = home / "library"
    conn = _ro(lib / "library.db")
    golden = json.loads((home / "golden_set.json").read_text(encoding="utf-8"))
    gs_by_md = {g["v2_markdown_id"]: g for g in golden}

    md_ids = {r[0] for r in conn.execute(
        "SELECT DISTINCT markdown_id FROM paper_files WHERE markdown_id IS NOT NULL")}
    current = set(md_ids)
    md_ids |= set(gs_by_md)

    rows = []
    for md_id in sorted(md_ids):
        row = conn.execute(
            "SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?", (md_id,)
        ).fetchone()
        if row is None:
            rows.append({"markdown_id": md_id, "error": "no markdown row"})
            continue
        text = (lib / row[0]).read_text(encoding="utf-8", errors="replace")
        ranges = conn.execute(
            "SELECT start_char, end_char FROM document_sections "
            "WHERE markdown_id = ? AND section_kind = 'references'", (md_id,)
        ).fetchall()
        result = tq.assess_text(text, reference_ranges=ranges)
        doc_rule = result.doc_rate >= tq.DOC_RATE_MIN and result.strong_hits >= tq.DOC_HITS_MIN
        gs = gs_by_md.get(md_id)
        rows.append({
            "markdown_id": md_id,
            "current": md_id in current,
            "golden_id": gs["id"] if gs else None,
            "golden_category": gs["category"] if gs else None,
            "has_reference_section": bool(ranges),
            "doc_rule": doc_rule,
            **result.as_dict(),
        })
    conn.close()

    ok = [r for r in rows if "error" not in r]
    lib_rows = [r for r in ok if r["current"]]
    flagged = [r for r in lib_rows if r["status"] == "garbled"]
    unflagged = [r for r in lib_rows if r["status"] == "pass"]
    summary = {
        "checker_version": tq.CHECKER_VERSION,
        "constants": {k: getattr(tq, k) for k in (
            "DOC_RATE_MIN", "DOC_HITS_MIN", "PAGE_MIN_WORDS", "PAGE_HITS_MIN",
            "PAGE_RATE_MIN", "BAD_PAGES_MIN", "BAD_PAGES_SHARE", "NO_TEXT_MEDIAN_WORDS")},
        "library_markdowns": len(lib_rows),
        "errors": [r for r in rows if "error" in r],
        "status_counts": dict(collections.Counter(r["status"] for r in lib_rows)),
        "doc_rate_buckets": dict(sorted(collections.Counter(
            _bucket(r["doc_rate"]) for r in lib_rows).items())),
        "papers_with_any_strong_hit": sum(1 for r in lib_rows if r["strong_hits"] > 0),
        "lowest_flagged_doc_rate": min((r["doc_rate"] for r in flagged), default=None),
        "highest_unflagged_doc_rate": max((r["doc_rate"] for r in unflagged), default=None),
        "flagged_by_page_rule_only": sum(1 for r in flagged if not r["doc_rule"]),
        "flagged_without_reference_section": sum(
            1 for r in flagged if not r["has_reference_section"]),
        "by_signal_class": {
            cls: {
                "flagged": sum(1 for r in flagged if _signal_class(r["hits_by_signal"]) == cls),
                "unflagged_with_hits": sum(
                    1 for r in unflagged
                    if r["strong_hits"] and _signal_class(r["hits_by_signal"]) == cls),
                "lowest_flagged_doc_rate": min((r["doc_rate"] for r in flagged
                    if _signal_class(r["hits_by_signal"]) == cls), default=None),
                "highest_unflagged_doc_rate": max((r["doc_rate"] for r in unflagged
                    if _signal_class(r["hits_by_signal"]) == cls), default=None),
            }
            for cls in ("ligature", "garbage", "mixed")
        },
        "flagged_at_exactly_doc_hits_min": sum(
            1 for r in flagged if r["strong_hits"] == tq.DOC_HITS_MIN),
        "top_unflagged": sorted(
            ({"markdown_id": r["markdown_id"], "doc_rate": r["doc_rate"],
              "strong_hits": r["strong_hits"], "bad_pages": r["bad_pages"]}
             for r in unflagged), key=lambda d: -d["doc_rate"])[:10],
        "golden_set": [
            {"id": r["golden_id"], "category": r["golden_category"], "status": r["status"],
             "doc_rate": r["doc_rate"], "strong_hits": r["strong_hits"],
             "bad_pages": r["bad_pages"], "pages": r["pages"], "words": r["words"],
             "expected": "garbled" if "garbled" in r["golden_category"] else "pass",
             "ok": r["status"] == ("garbled" if "garbled" in r["golden_category"] else "pass")}
            for r in sorted((r for r in ok if r["golden_id"]), key=lambda r: r["golden_id"])
        ],
    }
    (out / "per_paper.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False),
                                      encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--home", type=Path, default=Path.home() / ".seedgraph-v3")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.home / "work" / "calibration" / "textquality"
    summary = run(args.home, out)
    brief = {k: v for k, v in summary.items() if k not in ("top_unflagged",)}
    print(json.dumps(brief, indent=1, ensure_ascii=True))


if __name__ == "__main__":
    main()
