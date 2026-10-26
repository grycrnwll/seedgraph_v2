"""Edits survive re-extraction, setup half (docs/V3_UI_DESIGN.md §10.2 #2, steps 1-3).

    python -m tests.acceptance.edits_survive_setup --root HOME --paper lp_... [--live]

On one paper with a current note and no overlay, over HTTP with the page's CSRF token:

1. re-word, re-label and re-type one anchored agent claim; hide a second anchored one;
   add a researcher assertion; add a quote-backed claim from a text-view selection
   (``/anchor-text`` then ``/notes/add kind=quote``); save researcher notes. Each new
   text carries an invented token (:data:`EDIT_TOKEN`, :data:`NOTES_TOKEN`) so the
   fixed search queries find exactly the researcher content.
2. snapshot the interpretation (agent and researcher audiences), the search result sets
   for three fixed queries and MCP ``paper_show``'s claims to ``<run>/before.json``,
   with the values a re-note is expected to move kept apart (``expected_to_change``).
3. create the ``notes`` batch with replace through the paper page's Write notes button,
   check its item is ``queued``, and print the batch id and the copyable commands.

It stops there: an agent re-notes the paper through the CLI mirror, then
``edits_survive_check`` runs steps 4-6.

The default transport is the app in-process (the full ASGI stack: Host and CSRF checks
included) **without its lifespan**, so the JobRunner never starts and a real home's
conversion queue is not resumed. ``--live`` spawns ``seedgraph serve`` instead and
refuses while the home's conversion queue is not empty (the server would start
converting).
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import harness as h

EDIT_TOKEN = "quillfeather"
NOTES_TOKEN = "lanternwick"
ASSERTION_TEXT = (f"Invented researcher assertion ({EDIT_TOKEN}) written for the acceptance check; "
                  "it is not the paper's words.")
QUOTE_CLAIM_TEXT = f"Researcher claim backed by a selected passage ({EDIT_TOKEN})."
NOTES_TEXT = f"Acceptance check notes ({NOTES_TOKEN}): invented researcher text kept across a re-note."
ANCHOR_FIELD = re.compile(r'name="(markdown_id|start_char|end_char)" value="([^"]*)"')
ANCHOR_STATUS = re.compile(r'data-anchor-status="(\w+)"')
BATCH_LOCATION = re.compile(r"^/batches/(b_[0-9a-f]+)")


class SetupError(RuntimeError):
    pass


def _check_saved(resp, what: str) -> int:
    if resp.status_code != 200:
        raise SetupError(f"{what}: HTTP {resp.status_code}: {resp.text[:300]}")
    msg = h.pane_message(resp.text)
    if msg is None or msg[0] != "success":
        raise SetupError(f"{what}: not saved ({msg})")
    return h.pane_version(resp.text)


def choose_targets(lib, paper_id: str) -> tuple[Any, Any]:
    """Two anchored agent claims of different types in list fields: (edit, hide)."""
    from seedgraph.notes import interpretation

    interp = interpretation.current(lib, paper_id, "researcher")
    if interp is None:
        raise SetupError(f"{paper_id} has no current note")
    if interp.overlay_id is not None:
        raise SetupError(f"{paper_id} already has researcher edits (overlay {interp.overlay_id})")
    found = [c for c in interp.claims if c.kind == "agent" and c.status == "found" and c.span_id
             and "[" in c.field_key and c.claim_text and c.normalized_label]
    if len(found) < 2:
        raise SetupError(f"{paper_id} has fewer than two anchored list claims")
    edit = max(found, key=lambda c: (len(c.claim_text or ""), c.claim_id))
    hide = next((c for c in found if c.claim_type != edit.claim_type), None) or \
        next(c for c in found if c.claim_id != edit.claim_id)
    return edit, hide


def new_type(original: str) -> str:
    from seedgraph.web.views.paper import claim_type_choices

    return next(t for t in claim_type_choices() if t != original)


def passage_candidates(md: str, *, limit: int = 40) -> list[tuple[int, int]]:
    """Prose sentences (60-400 characters, no markup) from the middle of the markdown."""
    out: list[tuple[int, int]] = []
    start = len(md) // 3
    for m in re.finditer(r"[A-Z][^\n.!?$|#*\\_{}\[\]<>]{60,400}[.]", md[start:]):
        out.append((start + m.start(), start + m.end()))
        if len(out) >= limit:
            break
    return out


def add_quote_claim(client: h.PageClient, lib, paper_id: str, version: int, claim_type: str
                    ) -> tuple[int, dict[str, Any]]:
    from seedgraph.library import pages

    md_id, _ = pages.current_markdown(lib, paper_id)
    md = pages.PageAssigner(lib).markdown_text(md_id) if md_id else None
    if not md:
        raise SetupError(f"{paper_id} has no current markdown")
    tried = []
    for s, e in passage_candidates(md):
        resp = client.post(f"/papers/{paper_id}/anchor-text", {"start": s, "end": e})
        status = ANCHOR_STATUS.search(resp.text)
        tried.append(status.group(1) if status else f"http {resp.status_code}")
        if resp.status_code != 200 or not status or status.group(1) != "ok":
            continue
        fields = dict(ANCHOR_FIELD.findall(resp.text))
        data = {"kind": "quote", "claim_type": claim_type, "normalized_label": f"selected passage {EDIT_TOKEN}",
                "claim_text": QUOTE_CLAIM_TEXT, "expected_version": version, **fields}
        version = _check_saved(client.post(f"/papers/{paper_id}/notes/add", data), "add quote-backed claim")
        quote = md[int(fields["start_char"]):int(fields["end_char"])]
        return version, {"claim_type": claim_type, "normalized_label": data["normalized_label"],
                         "claim_text": QUOTE_CLAIM_TEXT, "markdown_id": fields["markdown_id"],
                         "start_char": int(fields["start_char"]), "end_char": int(fields["end_char"]),
                         "quote": quote}
    raise SetupError(f"no selection anchored uniquely ({tried[:10]})")


def _claim_dict(c) -> dict[str, Any]:
    return {"claim_id": c.claim_id, "claim_type": c.claim_type, "normalized_label": c.normalized_label,
            "claim_text": c.claim_text, "field_key": c.field_key, "status": c.status, "span_id": c.span_id}


def run_setup(home: Path, paper_id: str, client: h.PageClient, *, run_name: str | None = None,
              create_batch: bool = True, out=sys.stdout) -> dict[str, Any]:
    from seedgraph.library.store import library_dir, open_library_at
    from seedgraph.notes import interpretation
    from seedgraph.work.commands import work_commands

    lib = open_library_at(library_dir(home))
    try:
        page = client.get(f"/papers/{paper_id}")
        if page.status_code != 200:
            raise SetupError(f"GET /papers/{paper_id}: HTTP {page.status_code}")
        version = h.pane_version(page.text)
        before_fresh = h.freshness(lib, paper_id)
        if before_fresh["newer_extraction"]:
            raise SetupError(f"{paper_id} already has a newer extraction than its current note")
        edit, hide = choose_targets(lib, paper_id)
        edit_set = {"claim_text": f"Researcher re-wording ({EDIT_TOKEN}): {edit.claim_text}",
                    "normalized_label": f"{edit.normalized_label} ({EDIT_TOKEN})",
                    "claim_type": new_type(edit.claim_type)}

        version = _check_saved(client.post(f"/papers/{paper_id}/notes/edit", {
            "claim_id": edit.claim_id, "expected_version": version, **edit_set}), "edit")
        version = _check_saved(client.post(f"/papers/{paper_id}/notes/hide", {
            "claim_id": hide.claim_id, "expected_version": version}), "hide")
        assertion = {"claim_type": edit.claim_type, "normalized_label": f"researcher assertion {EDIT_TOKEN}",
                     "claim_text": ASSERTION_TEXT}
        version = _check_saved(client.post(f"/papers/{paper_id}/notes/add", {
            "kind": "assertion", "expected_version": version, **assertion}), "add assertion")
        version, quote = add_quote_claim(client, lib, paper_id, version, hide.claim_type)
        version = _check_saved(client.post(f"/papers/{paper_id}/notes/researcher-notes", {
            "text": NOTES_TEXT, "expected_version": version}), "researcher notes")

        interp = interpretation.current(lib, paper_id, "researcher", show_hidden=True)
        assert interp is not None
        added = {c.kind: c for c in interp.claims if c.kind in ("assertion", "quote")}
        if set(added) != {"assertion", "quote"}:
            raise SetupError(f"added claims not found in the interpretation: {sorted(added)}")
        hide_label = hide.normalized_label or " ".join((hide.claim_text or "").split()[:4])
        queries = [{"query": EDIT_TOKEN, "kind": "claims", "audience": "agent"},
                   {"query": hide_label, "kind": "claims", "audience": "agent"},
                   {"query": NOTES_TOKEN, "kind": "notes", "audience": "agent"}]
        inv = h.invariant(lib, paper_id, queries)

        # the setup's own assertions (the walk-through's step 6, without a browser)
        q1 = {r[0] for r in inv["search"][0]["results"]}
        q2 = {r[0] for r in inv["search"][1]["results"]}
        show = {c["claim_id"]: c for c in inv["paper_show_claims"] or []}
        checks = {
            "edit_found_by_search": h.canonical(edit.claim_id) in q1,
            "added_found_by_search": all(h.canonical(added[k].claim_id) in q1 for k in added),
            "hidden_not_found_by_search": h.canonical(hide.claim_id) not in q2,
            "notes_found_by_search": bool(inv["search"][2]["results"]),
            "paper_show_edit_trust": show.get(edit.claim_id, {}).get("trust"),
            "paper_show_hidden_absent": hide.claim_id not in show,
            "paper_show_added_trust": sorted({show.get(added[k].claim_id, {}).get("trust") for k in added},
                                             key=str),
        }
        page = client.get(f"/papers/{paper_id}")
        before = {
            "paper_id": paper_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "transport": type(client.http).__name__,
            "queries": queries,
            "ops": {
                "edit": {"claim_id": edit.claim_id, "set": edit_set, "original": _claim_dict(edit)},
                "hide": {"claim_id": hide.claim_id, "original": _claim_dict(hide)},
                "add_assertion": {"claim_id": added["assertion"].claim_id, **assertion},
                "add_quote": {"claim_id": added["quote"].claim_id, **quote},
                "researcher_notes": NOTES_TEXT,
            },
            "overlay": {"overlay_id": interp.overlay_id, "head_version_no": interp.version_no,
                        "base_note_id": interp.note_id},
            "invariant": inv,
            "invariant_sha256": {k: h.sha256(v) for k, v in inv.items()},
            "expected_to_change": {**h.freshness(lib, paper_id),
                                   "page_shows_newer": "data-newer-extraction" in page.text},
            "setup_checks": checks,
        }
        run = h.new_run(home, run_name)
        batch: dict[str, Any] | None = None
        if create_batch:
            resp = client.post(f"/papers/{paper_id}/write-notes", {}, htmx=False)
            m = BATCH_LOCATION.match(resp.headers.get("location", ""))
            if resp.status_code != 303 or not m:
                raise SetupError(f"Write notes: HTTP {resp.status_code} -> {resp.headers.get('location')}")
            batch_id = m.group(1)
            row = lib.conn.execute("SELECT state, expected_current_note_id FROM batch_items WHERE batch_id = ?",
                                   (batch_id,)).fetchall()
            opts = lib.conn.execute("SELECT options_json FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
            detail = client.get(f"/batches/{batch_id}")
            cmds = work_commands(batch_id)
            batch = {"batch_id": batch_id, "items": [dict(zip(("state", "expected_current_note_id"), r)) for r in row],
                     "options_json": opts[0] if opts else None, "detail_status": detail.status_code,
                     "detail_shows_command": cmds.claude_code in detail.text,
                     "commands": cmds.as_dict(),
                     "cli_mirror": f"seedgraph-v3 work claim {batch_id} --worker <name> --client claude"}
            before["batch"] = batch
            # the invariant must not move when the batch is created
            before["invariant_unchanged_by_batch"] = h.canonical(h.invariant(lib, paper_id, queries)) == h.canonical(inv)
        h.write_json(run / "before.json", before)
        h.update_inputs(home, "edits_survive", {"paper_id": paper_id, "run": run.name,
                                                "batch_id": batch["batch_id"] if batch else ""})
        print(f"run folder: {run}", file=out)
        print(f"paper: {paper_id}; overlay head v{interp.version_no}; setup checks: {checks}", file=out)
        if batch:
            print(f"batch: {batch['batch_id']} items={batch['items']}", file=out)
            print(f"  Claude Code: {batch['commands']['claude_code']}", file=out)
            print(f"  CLI mirror:  {batch['cli_mirror']}", file=out)
        before["run"] = str(run)
        return before
    finally:
        lib.close()


def main(argv: list[str] | None = None) -> int:
    from seedgraph import paths
    from seedgraph.library.store import library_dir, open_library_at

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--paper", default=None, help="paper id (default: inputs.json edits_survive.paper_id)")
    ap.add_argument("--run", default=None, help="run folder name (default: a UTC stamp)")
    ap.add_argument("--live", action="store_true", help="spawn `seedgraph serve` instead of in-process")
    ap.add_argument("--no-batch", action="store_true")
    args = ap.parse_args(argv)
    home = paths.resolve_home(args.root)
    paper_id = args.paper or h.load_inputs(home).get("edits_survive", {}).get("paper_id")
    if not paper_id:
        print("error: name the paper with --paper or in inputs.json", file=sys.stderr)
        return 2
    if args.live:
        with open_library_at(library_dir(home)) as lib:
            queued = lib.conn.execute("SELECT count(*) FROM conversion_queue").fetchone()[0]
        if queued:
            print(f"error: {queued} conversions are queued; `serve` would start them. Use the in-process "
                  "transport (the default).", file=sys.stderr)
            return 2
        with h.LiveServer(home) as server:
            client = h.PageClient.live(server.base_url)
            try:
                run_setup(home, paper_id, client, run_name=args.run, create_batch=not args.no_batch)
            finally:
                client.close()
        return 0
    client = h.PageClient.inprocess(home)
    try:
        run_setup(home, paper_id, client, run_name=args.run, create_batch=not args.no_batch)
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
