"""Calibration run for :mod:`seedgraph.library.titlecheck` (design §7.2, slice S2).

Not part of the product and not collected by pytest. Reads a v2 home strictly read-only
(``sqlite3`` ``mode=ro`` URIs, files opened for reading) and writes only into ``--out``,
which must be outside the repo (the v3 home's ``migration/calibration/``): the output
contains real titles.

    uv run python -m seedgraph.library._calibrate_titlecheck --v2-home <v2 home> --out <dir>

Optional ``<out>/mixups.json``: ``[{"label": "...", "title_contains": "..."}]`` selects
known mix-up works (normalised-title substring) whose scores are reported separately.

For every held work in every project it checks the title against the first pages of the
PDF's **own** latest successful markdown (not ``work_source_files.markdown_hash``, which
can point at another PDF's markdown), then, like the migration, falls back to the PDF
text layer when that is not ``pass``. Non-pass cases are sorted into causes
automatically; the residue is listed for a hand pass.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
from pathlib import Path

from seedgraph.library import titlecheck as tc
from seedgraph.project.identity import normalize_title

FOLD_GLYPHS = ("Ö", "¢", "¤", "…")
BOOK_PAGES = 60  # a markdown with more pages than this is treated as a book / long volume


def _ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _all_pages(md: str) -> list[str]:
    return tc.first_pages_from_markdown(md, n=10_000)


def _check_full(title, doi, md: str) -> tuple[tc.PdfCheck, int | None]:
    """Best check over every page window of the whole document; returns the page index
    where the title was found (by scanning 3-page windows)."""
    pages = _all_pages(md)
    best = tc.check_pdf_text(title, doi, pages)
    found_at = None
    if best.status == "pass" and best.via != "doi":
        for i in range(len(pages)):
            if tc.check_pdf_text(title, None, pages[i : i + 1]).status == "pass":
                found_at = i
                break
    return best, found_at


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2-home", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    home: Path = args.v2_home
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    mixups = []
    if (out / "mixups.json").exists():
        mixups = json.loads((out / "mixups.json").read_text(encoding="utf-8"))

    cache = _ro(home / "cache" / "cache.db")
    sf_by_hash = {h: (sid, uri) for sid, h, uri in cache.execute(
        "SELECT source_file_id, file_hash, storage_uri FROM source_files")}
    own_md: dict[str, tuple[str, str]] = {}
    for sid, mh, uri, _created, _mid in cache.execute(
        "SELECT source_file_id, markdown_hash, storage_uri, created_at, markdown_id FROM markdown_documents "
        "WHERE conversion_status = 'success' ORDER BY source_file_id, created_at, markdown_id"
    ):
        own_md[sid] = (mh, uri)  # last wins = latest
    cache.close()

    projects = sorted(p.name for p in (home / "projects").iterdir() if (p / "project.db").exists())
    rows = []
    hash_works: dict[str, list[tuple[str, str, str | None]]] = collections.defaultdict(list)
    for slug in projects:
        con = _ro(home / "projects" / slug / "project.db")
        con.execute("BEGIN")
        dois = collections.defaultdict(set)
        for wid, doi in con.execute("SELECT work_id, doi FROM works WHERE doi IS NOT NULL"):
            dois[wid].add(doi)
        for wid, val in con.execute("SELECT work_id, id_value FROM identifiers WHERE id_type = 'doi'"):
            dois[wid].add(val)
        q = ("SELECT w.work_id, w.canonical_title, w.year, s.file_hash, s.markdown_hash "
             "FROM works w JOIN work_source_files s ON s.work_id = w.work_id ORDER BY w.work_id, s.file_hash")
        for wid, title, year, fh, wmh in con.execute(q):
            rows.append(dict(project=slug, work_id=wid, title=title, year=year, file_hash=fh,
                             work_md_hash=wmh, dois=sorted(dois.get(wid, ()))))
            hash_works[fh].append((slug, wid, (sorted(dois.get(wid, ())) or [None])[0]))
        con.execute("COMMIT")
        con.close()

    md_cache: dict[str, str | None] = {}
    results = []
    for r in rows:
        sf = sf_by_hash.get(r["file_hash"])
        own = own_md.get(sf[0]) if sf else None
        md = None
        if own:
            if own[0] not in md_cache:
                md_cache[own[0]] = _read(home / "cache" / own[1])
            md = md_cache[own[0]]
        doi = r["dois"][0] if r["dois"] else None
        md_pages = tc.first_pages_from_markdown(md) if md else []
        md_check = tc.check_pdf_text(r["title"], doi, md_pages)
        pdf_check = None
        final = md_check
        if md_check.status != "pass" and sf:
            pdf_pages = tc.first_pages_from_pdf(home / "cache" / sf[1])
            pdf_check = tc.check_pdf_text(r["title"], doi, pdf_pages or [])
            if (pdf_check.status == "pass") or (pdf_check.status != "unknown" and pdf_check.score > final.score) \
                    or final.status == "unknown":
                final = pdf_check if pdf_check.status != "unknown" else final
        cause = None
        detail = {}
        if final.via == "doi":
            # Audit the DOI short-circuit: how well does the title alone match?
            title_only = tc.check_pdf_text(r["title"], None, md_pages)
            detail.update(doi_title_status=title_only.status, doi_title_score=title_only.score)
        if final.status in ("weak", "miss"):
            others = [x for x in hash_works[r["file_hash"]] if (x[0], x[1]) != (r["project"], r["work_id"])]
            other_dois = {x[2] for x in others if x[2]}
            if not own:
                cause = "not_converted_pdf_text_only"
            if md and cause is None:
                full, found_at = _check_full(r["title"], None, md)
                n_pages = len(_all_pages(md))
                detail.update(full_status=full.status, full_score=full.score, found_at_page=found_at,
                              pages=n_pages)
                doi_anywhere = bool(doi) and tc.check_pdf_text("zz", doi, [md]).via == "doi"
                glyphs = any(g in "".join(md_pages) for g in FOLD_GLYPHS)
                if full.status == "pass":
                    cause = "book_or_long_volume" if n_pages > BOOK_PAGES else "title_after_first_pages"
                elif doi_anywhere:
                    cause = "metadata_wrong_title_doi_present"
                elif doi and other_dois and doi not in other_dois:
                    cause = "shared_pdf_conflicting_doi"
                elif glyphs and pdf_check and pdf_check.score > md_check.score:
                    cause = "mojibake_markdown"
                else:
                    cause = "residue"
                detail["first_heading"] = next(
                    (ln.strip()[:160] for p in md_pages for ln in p.splitlines()
                     if len(ln.strip()) > 15), "")
        results.append(dict(
            r, own_md=own[0] if own else None, foreign_md=bool(own and r["work_md_hash"] and r["work_md_hash"] != own[0]),
            md_status=md_check.status, md_score=md_check.score, md_evidence=md_check.evidence,
            pdf_status=pdf_check.status if pdf_check else None, pdf_score=pdf_check.score if pdf_check else None,
            status=final.status, score=final.score, via=final.via, evidence=final.evidence, cause=cause, **detail))

    # Hand classification (``<out>/hand_causes.json``: {"<project>/<work_id prefix>": cause})
    # overrides the automatic cause; it lives with the real data, never in the repo.
    hand = {}
    if (out / "hand_causes.json").exists():
        hand = json.loads((out / "hand_causes.json").read_text(encoding="utf-8"))
    for res in results:
        res["auto_cause"] = res["cause"]
        for key, cause in hand.items():
            slug, prefix = key.split("/", 1)
            if res["project"] == slug and res["work_id"].startswith(prefix) and res["status"] in ("weak", "miss"):
                res["cause"] = cause

    with (out / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for res in results:
            fh.write(json.dumps(res, ensure_ascii=False, sort_keys=True) + "\n")

    # --- report --------------------------------------------------------------------------
    lines = ["# titlecheck calibration", "",
             f"Thresholds: PASS={tc.PASS_THRESHOLD} MISS={tc.MISS_THRESHOLD} short<={tc.SHORT_TITLE_TOKENS}", ""]
    lines.append("## Per project (markdown only → with PDF fallback)")
    lines.append("| project | held | md weak | md miss | md unknown | weak | miss | unknown |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for slug in projects:
        rs = [x for x in results if x["project"] == slug]
        c1 = collections.Counter(x["md_status"] for x in rs)
        c2 = collections.Counter(x["status"] for x in rs)
        lines.append(f"| {slug} | {len(rs)} | {c1['weak']} | {c1['miss']} | {c1['unknown']} | "
                     f"{c2['weak']} | {c2['miss']} | {c2['unknown']} |")
    lines += ["", "## Causes of final weak/miss (per project)"]
    causes = collections.Counter((x["project"], x["status"], x["cause"]) for x in results if x["cause"])
    for (slug, st, cause), n in sorted(causes.items()):
        lines.append(f"- {slug} {st} {cause}: {n}")
    uniq = {x["file_hash"] for x in results if x["status"] in ("weak", "miss")}
    lines.append(f"\nUnique PDFs with a final weak/miss: {len(uniq)}")
    lines.append(f"Held works on a foreign markdown (work_md_hash != own): {sum(x['foreign_md'] for x in results)}")
    for label in ("folded", "hyphen-closed"):
        n = sum(1 for x in results if x["status"] == "pass" and label in x["evidence"])
        lines.append(f"Passes that needed the {label} variant: {n}")
    low = [x for x in results if x["via"] == "doi" and x.get("doi_title_status") in ("weak", "miss")]
    lines.append(f"DOI passes whose title alone would be weak/miss (markdown pages): {len(low)}")
    for x in low:
        lines.append(f"  - {x['project']}/{x['work_id']} title-only {x['doi_title_status']} {x['doi_title_score']}")
    lines.append("Rescued by the PDF-text fallback (md not pass, PDF pass): "
                 f"{sum(1 for x in results if x['md_status'] != 'pass' and x['status'] == 'pass')}")
    lines.append("Final pass via doi / exact / tolerant: " + " / ".join(
        str(sum(1 for x in results if x['status'] == 'pass' and x['via'] == v)) for v in ("doi", "exact", "tolerant")))
    lines += ["", "## Score distribution of non-pass tolerant scores (final)"]
    hist = collections.Counter(round(x["score"], 2) for x in results if x["status"] in ("weak", "miss"))
    for s in sorted(hist):
        lines.append(f"- {s:.2f}: {hist[s]}")
    lines += ["", "## Mix-ups"]
    for m in mixups:
        sel = normalize_title(m["title_contains"])
        hits = [x for x in results if x["title"] and sel in normalize_title(x["title"])]
        for x in hits:
            lines.append(f"- {m['label']}: {x['project']}/{x['work_id']} md={x['md_status']} {x['md_score']} "
                         f"pdf={x['pdf_status']} {x['pdf_score']} final={x['status']} {x['score']} ({x['evidence']})")
    lines += ["", "## Cases scoring 0.25–0.55 (final)", ""]
    for x in sorted(results, key=lambda x: (x["score"], x["project"], x["work_id"])):
        if x["status"] in ("weak", "miss") and 0.25 <= x["score"] <= 0.55:
            lines.append(f"- {x['score']:.2f} {x['status']} {x['project']}/{x['work_id']} [{x['cause']}] "
                         f"title={x['title']!r} | head={x.get('first_heading', '')!r}")
    lines += ["", "## All final weak and miss", ""]
    for x in sorted(results, key=lambda x: (x["project"], x["status"], x["score"], x["work_id"])):
        if x["status"] in ("weak", "miss"):
            lines.append(f"- {x['project']}/{x['work_id']} {x['score']:.2f} [{x['cause']}] title={x['title']!r} "
                         f"| head={x.get('first_heading', '')!r}")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
