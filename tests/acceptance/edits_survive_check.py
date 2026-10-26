"""Edits survive re-extraction, check half (docs/V3_UI_DESIGN.md §10.2 #2, steps 4-6).

    python -m tests.acceptance.edits_survive_check --root HOME [--run NAME] [--paper ID]
           [--pre | --accept]

Reads ``<home>/ui/acceptance/<run>/before.json`` (written by ``edits_survive_setup``;
the run and paper default to ``inputs.json``'s ``edits_survive``) and checks:

Step 4, before accept:
- the interpretation snapshot (agent and researcher audiences) is **byte-identical**;
- the search result sets for the three fixed queries are equal;
- the claims MCP ``paper_show`` returns are unchanged;
- the freshness indicators moved: ``note_latest`` is new, the newer-extraction flag is
  set, the paper page shows "Newer extraction available" and ``note_current`` stays
  pinned to the overlay base;
- the replay plan lists the edit as ``kept``, the hide as ``still_hidden`` and the two
  added claims as ``added_kept``.

Steps 5-6 (``--accept``, only when step 4 passed): accept all through the replay page's
POST (CSRF included), then check that the edit (text, label and type), the hide and both
added claims survive, that the new extraction's agent claims are present, that
``note_current`` and the overlay base point to the new note, and report the identity
tiers used.

``--pre`` is the setup's self-test before any re-note: only the invariant is checked
and "no newer extraction yet" is the expected state.

Writes ``check-<stamp>.json`` and ``report.md`` into the run folder; exit 0 when every
check passed, 1 otherwise, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import harness as h


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.info: list[dict[str, Any]] = []  # reported, not gated

    def add(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows)

    def markdown(self) -> list[str]:
        lines = ["| check | result | detail |", "|---|---|---|"]
        for r in self.rows:
            detail = "" if r["detail"] is None else str(r["detail"]).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r['check']} | {'PASS' if r['ok'] else 'FAIL'} | {detail[:300]} |")
        return lines


def _diff_claims(before: list[dict] | None, now: list[dict] | None) -> str:
    b = {c["claim_id"]: c for c in before or []}
    n = {c["claim_id"]: c for c in now or []}
    changed = [k for k in b.keys() & n.keys() if h.canonical(b[k]) != h.canonical(n[k])]
    return f"gone {sorted(b.keys() - n.keys())[:5]}, new {sorted(n.keys() - b.keys())[:5]}, changed {changed[:5]}"


def _own_claim_ids(lib, pid: str, *snapshots: dict[str, Any]) -> set[str]:
    """This paper's claim ids: its extracted claims plus every claim (added ones included)
    in the stored and current interpretations."""
    ids = {r[0] for r in lib.conn.execute("SELECT claim_id FROM extracted_claims WHERE paper_id = ?", (pid,))}
    for snap in snapshots:
        for aud in ("agent", "researcher"):
            interp = snap["interpretation"].get(aud) or {}
            ids.update(c["claim_id"] for c in interp.get("claims", []))
    return ids


def _paper_of(lib, kind: str, key: str) -> str | None:
    if kind == "claims":
        row = lib.conn.execute("SELECT paper_id FROM extracted_claims WHERE claim_id = ?", (key,)).fetchone()
    elif kind == "notes":
        row = lib.conn.execute("SELECT paper_id FROM structured_notes WHERE note_id = ?", (key,)).fetchone()
    else:
        return None
    return row[0] if row else None


def _is_own(lib, pid: str, kind: str, key_json: str, own_claims: set[str]) -> bool:
    key = json.loads(key_json)
    if kind == "claims":
        return key in own_claims
    if kind == "notes":
        return _paper_of(lib, kind, key) == pid
    return True  # spans: not used by the fixed queries; compare whole sets


def check_invariant(checks: Checks, lib, before: dict[str, Any]) -> dict[str, Any]:
    pid = before["paper_id"]
    now = h.invariant(lib, pid, before["queries"])
    inv = before["invariant"]
    same = h.canonical(now["interpretation"]) == h.canonical(inv["interpretation"])
    detail = "byte-identical" if same else _diff_claims(
        (inv["interpretation"]["researcher"] or {}).get("claims"),
        (now["interpretation"]["researcher"] or {}).get("claims"))
    checks.add("interpretation byte-identical (agent + researcher)", same, detail)
    own_claims = _own_claim_ids(lib, pid, inv, now)
    for q_before, q_now in zip(inv["search"], now["search"]):
        mine_b = [r for r in q_before["results"] if _is_own(lib, pid, q_before["kind"], r[0], own_claims)]
        mine_n = [r for r in q_now["results"] if _is_own(lib, pid, q_now["kind"], r[0], own_claims)]
        checks.add(f"search set equal (this paper): {q_before['kind']} {q_before['query']!r}", mine_b == mine_n,
                   f"{len(mine_b)} before, {len(mine_n)} now")
        # the whole-library set is reported, not gated: other papers' notes may change meanwhile
        b, n = {tuple(r) for r in q_before["results"]}, {tuple(r) for r in q_now["results"]}
        if b != n:
            extra = sorted(json.loads(k) for k, _ in n - b)
            gone = sorted(json.loads(k) for k, _ in b - n)
            checks.info.append({"query": q_before["query"], "kind": q_before["kind"],
                                "library_wide_new": extra[:20], "library_wide_gone": gone[:20],
                                "papers": sorted({_paper_of(lib, q_before["kind"], k) or "?" for k in extra + gone})})
    checks.add("MCP paper_show claims unchanged",
               h.canonical(now["paper_show_claims"]) == h.canonical(inv["paper_show_claims"]),
               _diff_claims(inv["paper_show_claims"], now["paper_show_claims"]))
    return now


def check_freshness(checks: Checks, lib, client: h.PageClient, before: dict[str, Any]) -> dict[str, Any]:
    pid = before["paper_id"]
    was = before["expected_to_change"]
    now = h.freshness(lib, pid)
    page = client.get(f"/papers/{pid}")
    now["page_shows_newer"] = "Newer extraction available" in page.text
    checks.add("note_latest is new", now["note_latest_id"] not in (None, was["note_latest_id"]),
               f"{was['note_latest_id']} -> {now['note_latest_id']}")
    checks.add("newer-extraction flag set", now["newer_extraction"] and not was["newer_extraction"])
    checks.add("paper page shows 'Newer extraction available'", now["page_shows_newer"])
    checks.add("note_current pinned to the overlay base",
               now["note_current_id"] == before["overlay"]["base_note_id"], now["note_current_id"])
    return now


def plan_statuses(plan) -> dict[str, Any]:
    return {o.key: {"op": o.op["op"], "status": o.status, "old_claim_id": o.old_claim_id,
                    "new_claim_id": o.new_claim_id, "keep_as": o.keep_as} for o in plan.ops}


def check_plan(checks: Checks, plan, before: dict[str, Any]) -> dict[str, Any]:
    ops = before["ops"]
    st = plan_statuses(plan)
    edit_key, hide_key = f"edit:{ops['edit']['claim_id']}", f"hide:{ops['hide']['claim_id']}"
    checks.add("replay plan: edit kept", st.get(edit_key, {}).get("status") == "kept", st.get(edit_key))
    checks.add("replay plan: hide still hidden", st.get(hide_key, {}).get("status") == "still_hidden",
               st.get(hide_key))
    for name in ("add_assertion", "add_quote"):
        key = ops[name]["claim_id"]
        checks.add(f"replay plan: {name} kept", st.get(key, {}).get("status") == "added_kept", st.get(key))
    return st


def tiers(plan, before: dict[str, Any]) -> dict[str, Any]:
    by_old = {p.old: p for p in plan.matching.pairs}
    out: dict[str, Any] = {"all_pairs": dict(Counter(p.tier for p in plan.matching.pairs)),
                           "ambiguous": len(plan.matching.ambiguous),
                           "unmatched_old": len(plan.matching.unmatched_old),
                           "unmatched_new": len(plan.matching.unmatched_new)}
    for name in ("edit", "hide"):
        p = by_old.get(before["ops"][name]["claim_id"])
        out[name] = None if p is None else {"tier": p.tier, "overlap": p.overlap, "jaccard": p.jaccard,
                                            "new": p.new}
    return out


def accept_all(client: h.PageClient, plan) -> Any:
    return client.post(f"/papers/{plan.paper_id}/replay/accept",
                       {"head_version_no": plan.head_version_no, "new_note_id": plan.new_note_id, "mode": "all"},
                       htmx=False)


def check_after_accept(checks: Checks, lib, plan, before: dict[str, Any]) -> dict[str, Any]:
    from seedgraph.notes import interpretation

    pid = before["paper_id"]
    ops = before["ops"]
    st = plan_statuses(plan)
    res = interpretation.current(lib, pid, "researcher", show_hidden=True)
    agent = interpretation.current(lib, pid, "agent")
    assert res is not None
    by_id = {c.claim_id: c for c in res.claims}
    agent_ids = {c.claim_id for c in (agent.claims if agent else ())}

    new_edit = st.get(f"edit:{ops['edit']['claim_id']}", {}).get("new_claim_id")
    c = by_id.get(new_edit) if new_edit else None
    want = ops["edit"]["set"]
    checks.add("after accept: edit survives (text, label, type)",
               c is not None and c.claim_text == want["claim_text"] and c.normalized_label == want["normalized_label"]
               and c.claim_type == want["claim_type"] and c.trust == "researcher" and c.edited_by_researcher,
               None if c is None else {"claim_id": c.claim_id, "type": c.claim_type, "trust": c.trust})
    new_hide = st.get(f"hide:{ops['hide']['claim_id']}", {}).get("new_claim_id")
    hc = by_id.get(new_hide) if new_hide else None
    checks.add("after accept: hide survives", hc is not None and hc.hidden and new_hide not in agent_ids, new_hide)
    for name, kind in (("add_assertion", "assertion"), ("add_quote", "quote")):
        want_text = ops[name]["claim_text"]
        hits = [x for x in res.claims if x.kind == kind and x.claim_text == want_text and not x.hidden]
        checks.add(f"after accept: {name} survives", bool(hits),
                   [x.claim_id for x in hits] or [x.kind for x in res.claims if x.claim_text == want_text])
    checks.add("after accept: researcher notes survive", res.researcher_notes == ops["researcher_notes"])
    new_ids = [n.claim_id for n in plan.new_claims]
    missing = [i for i in new_ids if i not in agent_ids]
    agent_new_note = [x for x in res.claims if x.kind == "agent"]
    checks.add("after accept: new agent claims present", not missing and bool(agent_new_note),
               f"{len(new_ids)} new claims in the plan, {len(missing)} missing; "
               f"{len(agent_new_note)} agent claims on the new base")
    cur = lib.conn.execute("SELECT note_id FROM note_current WHERE paper_id = ? AND schema_id IS ?",
                           (pid, res.schema_id)).fetchone()
    base = lib.conn.execute("SELECT base_note_id FROM note_overlays WHERE overlay_id = ?",
                            (res.overlay_id,)).fetchone()
    checks.add("after accept: note_current points to the new base",
               cur is not None and cur[0] == plan.new_note_id and res.note_id == plan.new_note_id
               and base is not None and base[0] == plan.new_note_id,
               {"note_current": cur[0] if cur else None, "overlay_base": base[0] if base else None,
                "new_note": plan.new_note_id})
    checks.add("after accept: newer-extraction flag cleared", not res.newer_extraction)
    return {"interpretation_after": h.interpretation_snapshot(lib, pid),
            "search_after": h.search_sets(lib, before["queries"])}


def run_check(home: Path, run: Path, client: h.PageClient, *, pre: bool = False, accept: bool = False,
              out=sys.stdout) -> tuple[Checks, dict[str, Any]]:
    from seedgraph.library.store import library_dir, open_library_at
    from seedgraph.notes import replay

    before = json.loads((run / "before.json").read_text(encoding="utf-8"))
    pid = before["paper_id"]
    checks = Checks()
    result: dict[str, Any] = {"paper_id": pid, "mode": "pre" if pre else ("accept" if accept else "check")}
    lib = open_library_at(library_dir(home))
    try:
        check_invariant(checks, lib, before)
        if pre:
            fresh = h.freshness(lib, pid)
            checks.add("no newer extraction yet (waiting for the re-note)", not fresh["newer_extraction"], fresh)
            try:
                replay.plan_replay(lib, pid)
                checks.add("replay not offered before the re-note", False)
            except replay.ReplayError as exc:
                checks.add("replay not offered before the re-note", True, str(exc))
            result["freshness"] = fresh
        else:
            result["freshness"] = check_freshness(checks, lib, client, before)
            if "batch" in before:
                rows = lib.conn.execute("SELECT state FROM batch_items WHERE batch_id = ?",
                                        (before["batch"]["batch_id"],)).fetchall()
                result["batch_items"] = [r[0] for r in rows]
            try:
                plan = replay.plan_replay(lib, pid)
            except replay.ReplayError as exc:
                checks.add("replay plan available", False, str(exc))
                plan = None
            if plan is not None:
                result["plan"] = {"statuses": check_plan(checks, plan, before), "new_note_id": plan.new_note_id,
                                  "new_claims": len(plan.new_claims), "changed": len(plan.changed),
                                  "gone": len(plan.gone)}
                result["tiers"] = tiers(plan, before)
                if accept:
                    if not checks.ok:
                        checks.add("accept skipped: a step 4 check failed", False)
                    else:
                        resp = accept_all(client, plan)
                        loc = resp.headers.get("location", "")
                        checks.add("accept all (POST /replay/accept)",
                                   resp.status_code == 303 and loc.startswith(f"/papers/{pid}")
                                   and "/replay" not in loc, f"{resp.status_code} -> {loc}")
                        result.update(check_after_accept(checks, lib, plan, before))
    finally:
        lib.close()
    result["checks"] = checks.rows
    result["info"] = checks.info
    result["ok"] = checks.ok
    name = f"check-{result['mode']}-{h.stamp()}"
    h.write_json(run / f"{name}.json", result)
    lines = [f"# Edits survive re-extraction: {result['mode']}", "", f"paper `{pid}`, run `{run.name}`", ""]
    lines += checks.markdown()
    if checks.info:
        lines += ["", "## Library-wide search differences (reported, not gated)", "", "```json",
                  json.dumps(checks.info, indent=1), "```"]
    if "tiers" in result:
        lines += ["", "## Identity tiers", "", "```json", json.dumps(result["tiers"], indent=1), "```"]
    report = run / "report.md"
    with report.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n\n")
    for r in checks.rows:
        print(f"{'PASS' if r['ok'] else 'FAIL'}  {r['check']}" + (f"  ({r['detail']})" if not r["ok"] else ""),
              file=out)
    print(f"{'ALL PASSED' if checks.ok else 'FAILED'}; wrote {run / (name + '.json')} and {report}", file=out)
    return checks, result


def main(argv: list[str] | None = None) -> int:
    from seedgraph import paths

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--run", default=None, help="run folder name (default: inputs.json edits_survive.run)")
    ap.add_argument("--paper", default=None, help="only to confirm the run's paper")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--pre", action="store_true", help="self-test before the re-note")
    mode.add_argument("--accept", action="store_true", help="after step 4 passes, accept all and run steps 5-6")
    args = ap.parse_args(argv)
    home = paths.resolve_home(args.root)
    inputs = h.load_inputs(home).get("edits_survive", {})
    run_name = args.run or inputs.get("run")
    if not run_name:
        print("error: name the run with --run or in inputs.json", file=sys.stderr)
        return 2
    run = h.acceptance_dir(home) / run_name
    if not (run / "before.json").is_file():
        print(f"error: no before.json in {run}", file=sys.stderr)
        return 2
    client = h.PageClient.inprocess(home)
    try:
        checks, result = run_check(home, run, client, pre=args.pre, accept=args.accept)
    finally:
        client.close()
    if args.paper and args.paper != result["paper_id"]:
        print(f"error: the run is for {result['paper_id']}, not {args.paper}", file=sys.stderr)
        return 2
    return 0 if checks.ok else 1


if __name__ == "__main__":
    sys.exit(main())
