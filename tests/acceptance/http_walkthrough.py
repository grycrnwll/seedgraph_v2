"""Scripted HTTP walk-through of everything that needs neither a browser nor an agent
(docs/V3_UI_DESIGN.md §10.1-10.2).

    python -m tests.acceptance.http_walkthrough --root HOME [--paper ID] [--batch ID]
           [--no-ingest] [--convert-timeout 900]

Part A, the real home, GET only, in-process without the lifespan (no JobRunner, so the
home's conversion queue is not resumed): Home, every Library view (held, cited-only,
needs notes, held back, needs a PDF, all; with and without the graph filter), Write
notes, the paper page with its notes pane, text view and PDF, the replay page, batches
(list, detail, items), review (attention, duplicates, citations), graphs, settings and
the stage 5 placeholders. Each page: status, size, time and a check for a server error.

Part B, a temp home with a live ``serve`` (real sockets, real converter pool): add one
synthetic PDF by upload and one by typed folder, poll the job strip until conversion
finishes (or the timeout), then check the library lists the new papers, the paper page
and its PDF route render, a pasted non-DOI is refused politely, Write notes creates a
batch whose page shows the copyable commands, and the review and settings pages render.
Synthetic, invented text only.

Writes ``http_walkthrough.json`` and appends to ``report.md`` in a new run folder.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import harness as h

ERROR_MARKERS = ("Traceback (most recent call last)", "Internal Server Error")


def _page(client: h.PageClient, path: str, *, want: int = 200, contains: str | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    resp = client.get(path)
    ms = round((time.perf_counter() - t0) * 1000)
    text = resp.text if "text" in resp.headers.get("content-type", "") else ""
    ok = resp.status_code == want and not any(m in text for m in ERROR_MARKERS)
    if contains is not None:
        ok = ok and contains in text
    return {"path": path, "status": resp.status_code, "ms": ms, "bytes": len(resp.content),
            "content_type": resp.headers.get("content-type", ""), "ok": ok,
            **({"expects": contains} if contains else {})}


def part_a(home: Path, paper_id: str | None, batch_id: str | None) -> list[dict[str, Any]]:
    from seedgraph.library.store import library_dir, open_library_at

    with open_library_at(library_dir(home)) as lib:
        if paper_id is None:
            row = lib.conn.execute("SELECT paper_id FROM note_current ORDER BY paper_id LIMIT 1").fetchone()
            paper_id = row[0] if row else None
        if batch_id is None:
            row = lib.conn.execute("SELECT batch_id FROM batches ORDER BY created_at DESC LIMIT 1").fetchone()
            batch_id = row[0] if row else None
        graph = "diff_in_diff" if (home / "graphs" / "diff_in_diff").is_dir() else None
    client = h.PageClient.inprocess(home)
    rows: list[dict[str, Any]] = []
    try:
        paths = ["/", "/home/activity", "/library", "/library/jobs"]
        for view in ("held", "cited-only", "needs-notes", "held-back", "needs-pdf", "all"):
            paths.append(f"/library?view={view}")
            if graph:
                paths.append(f"/library?view={view}&graph={graph}")
        if graph:
            paths += [f"/library?view=held&graph={graph}&frontier=1", f"/library?view=held&q=effect&graph={graph}",
                      f"/library/write-notes?view=needs-notes&graph={graph}", "/graphs", f"/graphs/{graph}"]
        paths += ["/library?view=held&page=2", "/batches", "/review/attention", "/review/duplicates",
                  "/review/citations", "/settings", "/records", "/spot-check"]
        rows += [_page(client, p) for p in paths]
        if paper_id:
            rows.append(_page(client, f"/papers/{paper_id}", contains="data-expected-version"))
            rows.append(_page(client, f"/papers/{paper_id}?layout=notes"))
            rows.append(_page(client, f"/papers/{paper_id}/notes?show_hidden=1"))
            rows.append(_page(client, f"/papers/{paper_id}/text?page=1"))
            rows.append(_page(client, f"/library/write-notes?paper={paper_id}"))
            pdf = _page(client, f"/papers/{paper_id}/pdf")
            pdf["ok"] = pdf["ok"] and pdf["content_type"].startswith("application/pdf")
            rows.append(pdf)
            replay = _page(client, f"/papers/{paper_id}/replay", want=404)
            replay["note"] = "404 expected until a newer extraction exists"
            rows.append(replay)
        if batch_id:
            rows += [_page(client, f"/batches/{batch_id}", contains="/seedgraph-v3-work"),
                     _page(client, f"/batches/{batch_id}/items")]
        rows.append(_page(client, "/papers/lp_doesnotexist000000", want=404))
        rows.append(_page(client, "/batches/b_doesnotexist", want=404))
    finally:
        client.close()
    return rows


def wrapped_text_pdf(path: Path, pages: list[str], *, width: int = 85) -> Path:
    """A valid text PDF whose pages carry ``pages`` as wrapped Helvetica lines (a single
    long line per page runs off the page and converts to garbage)."""
    import textwrap

    n = len(pages)
    font_id = 3 + 2 * n
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()]
    for i, text in enumerate(pages):
        lines: list[str] = []
        for para in text.split("\n"):
            lines += textwrap.wrap(para, width) + [""]
        esc = [ln.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") for ln in lines]
        body = " T* ".join(f"({ln}) Tj" for ln in esc)
        stream = f"BT /F1 11 Tf 14 TL 72 740 Td {body} ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


def _synthetic_pdfs(folder: Path) -> tuple[Path, Path]:
    """Two small text PDFs with invented prose: one to upload, one in a folder."""
    from ..web import paper_support as ps

    (folder / "folder").mkdir(parents=True, exist_ok=True)
    title_a = "An Invented Study of Widget Growth"
    title_b = "Invented Notes on Gizmo Prices"
    pages_a = [(f"{title_a}\nAda Inventor and Bo Madeup\n" if i == 0 else "") + p + "\n" + ps.filler(i)
               + "\n" + ps.filler(i + 5) for i, p in enumerate(ps.PAGES)]
    pages_b = [(f"{title_b}\nCy Fictional\n" if i == 0 else "")
               + f"An invented note on gizmo prices, part {i + 1}.\n" + ps.filler(i + 3) + "\n" + ps.filler(i + 7)
               for i in range(3)]
    return (wrapped_text_pdf(folder / "upload-a.pdf", pages_a),
            wrapped_text_pdf(folder / "folder" / "folder-b.pdf", pages_b))


def _wait_jobs(client: h.PageClient, timeout: float) -> tuple[str, float]:
    t0 = time.monotonic()
    html = ""
    while time.monotonic() - t0 < timeout:
        html = client.get("/library/jobs").text
        if 'hx-trigger="every 2s"' not in html:
            return html, time.monotonic() - t0
        time.sleep(2)
    return html, time.monotonic() - t0


def _job_lines(html: str) -> list[str]:
    return [re.sub(r"<[^>]+>", " ", m).strip() for m in re.findall(r"<li[^>]*>(.*?)</li>", html, re.S)]


def part_b(convert_timeout: float) -> dict[str, Any]:
    from seedgraph.library.store import library_dir, open_library_at

    out: dict[str, Any] = {"steps": []}
    steps = out["steps"]
    with tempfile.TemporaryDirectory(prefix="sg-walk-") as tmp:
        root = Path(tmp)
        home = root / "home"
        open_library_at(home / "library").close()
        upload, _ = _synthetic_pdfs(root / "pdfs")
        with h.LiveServer(home) as server:
            client = h.PageClient.live(server.base_url, timeout=120)
            try:
                steps.append(_page(client, "/library"))
                with upload.open("rb") as fh:
                    resp = client.http.post("/library/add", files={"files": ("upload-a.pdf", fh, "application/pdf")},
                                            headers={"X-CSRF-Token": client.token(), "HX-Request": "true"})
                steps.append({"path": "POST /library/add (upload)", "status": resp.status_code,
                              "ok": resp.status_code == 200 and "adding 1 PDF" in resp.text})
                resp = client.post("/library/add-folder", {"folder": str(root / "pdfs" / "folder")}, htmx=True)
                steps.append({"path": "POST /library/add-folder", "status": resp.status_code,
                              "ok": resp.status_code == 200 and "adding the PDFs" in resp.text})
                resp = client.post("/library/add-dois", {"dois": "not-a-doi"}, htmx=True)
                steps.append({"path": "POST /library/add-dois (not a DOI)", "status": resp.status_code,
                              "ok": resp.status_code == 200})
                html, waited = _wait_jobs(client, convert_timeout)
                lines = _job_lines(html)
                converted = sum("converted" in ln.lower() or "done" in ln.lower() for ln in lines)
                steps.append({"path": "job strip polled until idle", "waited_s": round(waited),
                              "idle": 'hx-trigger="every 2s"' not in html, "lines": lines,
                              "ok": 'hx-trigger="every 2s"' not in html})
                with open_library_at(home / "library") as lib:
                    papers = [dict(r) for r in lib.conn.execute(
                        "SELECT p.paper_id, f.markdown_id IS NOT NULL AS converted, f.pdf_check_status "
                        "FROM papers p LEFT JOIN paper_files f ON f.paper_id = p.paper_id ORDER BY p.paper_id")]
                    attention = [dict(r) for r in lib.conn.execute(
                        "SELECT kind, paper_id FROM attention_items WHERE status = 'open'")]
                out["papers"] = papers
                out["attention"] = attention
                out["converted_job_lines"] = converted
                steps.append({"path": "two new papers, both converted", "ok": len(papers) == 2
                              and all(p["converted"] for p in papers), "papers": papers})
                for p in papers:
                    steps.append(_page(client, f"/papers/{p['paper_id']}"))
                    pdf = _page(client, f"/papers/{p['paper_id']}/pdf")
                    pdf["ok"] = pdf["ok"] and pdf["content_type"].startswith("application/pdf")
                    steps.append(pdf)
                for view in ("held", "needs-notes", "held-back", "needs-pdf", "cited-only", "all"):
                    steps.append(_page(client, f"/library?view={view}"))
                if papers and all(p["converted"] for p in papers):
                    pid = papers[0]["paper_id"]
                    steps.append(_page(client, f"/library/write-notes?paper={pid}"))
                    resp = client.post("/library/write-notes", {"paper": pid}, htmx=False)
                    loc = resp.headers.get("location", "")
                    m = re.match(r"^/batches/(b_[0-9a-f]+)", loc)
                    steps.append({"path": "POST /library/write-notes", "status": resp.status_code, "location": loc,
                                  "ok": resp.status_code == 303 and bool(m)})
                    if m:
                        steps.append(_page(client, f"/batches/{m.group(1)}", contains="/seedgraph-v3-work"))
                        steps.append(_page(client, f"/batches/{m.group(1)}/items"))
                        steps.append(_page(client, "/batches"))
                for path in ("/", "/review/attention", "/review/duplicates", "/review/citations", "/settings",
                             "/graphs"):
                    steps.append(_page(client, path))
            finally:
                client.close()
        out["server_output"] = server.output.strip()[-3000:]
    return out


def main(argv: list[str] | None = None) -> int:
    from seedgraph import paths

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--paper", default=None)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--no-ingest", action="store_true", help="skip part B (temp home, live serve, conversion)")
    ap.add_argument("--only-ingest", action="store_true", help="run part B only")
    ap.add_argument("--convert-timeout", type=float, default=900.0)
    args = ap.parse_args(argv)
    home = paths.resolve_home(args.root)
    inputs = h.load_inputs(home).get("edits_survive", {})
    result: dict[str, Any] = {"part_a": [] if args.only_ingest else part_a(
        home, args.paper or inputs.get("paper_id") or None, args.batch or inputs.get("batch_id") or None)}
    if not args.no_ingest:
        result["part_b"] = part_b(args.convert_timeout)
    run = h.new_run(home, f"http-{h.stamp()}")
    h.write_json(run / "http_walkthrough.json", result)
    rows = result["part_a"] + (result.get("part_b", {}).get("steps", []))
    failed = [r for r in rows if not r.get("ok")]
    lines = ["# HTTP walk-through (no browser, no agent)", "",
             f"part A (real home, GET only): {sum(r['ok'] for r in result['part_a'])}/{len(result['part_a'])} ok",
             ]
    if "part_b" in result:
        sb = result["part_b"]["steps"]
        lines.append(f"part B (temp home, live serve, real converter): {sum(bool(r.get('ok')) for r in sb)}/{len(sb)} ok")
    lines += ["", "## Not ok", ""] + [f"- {r.get('path')}: {r.get('status')} {r.get('lines', '')}" for r in failed]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{'ok  ' if r.get('ok') else 'FAIL'} {r.get('status', ''):>4} {r.get('ms', ''):>6} {r.get('path')}")
    print("\n".join(lines))
    print(f"wrote {run}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
