"""The stage 4 acceptance report and the review-sample draw (design §11; slice C11).

- :func:`draw_sample` freezes the §11.2 review sample through
  :func:`researcher.create_review_sample` (applied agent merges and links of a batch,
  verdict x family strata, ambiguous labels over-sampled, seeded). When fewer rows are
  eligible than asked for, it takes them all and tops up with a seeded draw of migrated
  agent merges (applied, no batch), and says how many.
- :func:`gather` collects the report's figures from a library (read-only use: it plans
  the concepts in memory and never writes) plus the hand-entered inputs: the dry-run
  estimate, the measured token total, the ticket 11 figures and which workers ran before
  the stricter link rule. :func:`render` turns them into Markdown.

Real labels appear in the rendered text, so reports are written under the home only.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import cleanup as cu
from . import decide
from . import guardrails as g
from . import refresh as rf
from . import researcher as rs

if TYPE_CHECKING:
    from ..library.store import Library

REVIEW_PATH = "/concepts/review/{}"
_PAIR_KINDS = ("merge", "link", "unrelated")


class InputsError(ValueError):
    """The acceptance inputs file is missing or incomplete."""


def load_inputs(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputsError(f"no acceptance inputs at {path} (write the estimate and measured tokens there)")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("batch_id", "estimate", "measured"):
        if key not in data:
            raise InputsError(f"acceptance inputs {path}: missing {key!r}")
    return data


# --- the review sample ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SampleDraw:
    sample: rs.ReviewSample
    eligible: int
    topped_up: int

    @property
    def path(self) -> str:
        return REVIEW_PATH.format(self.sample.sample_id)


def _eligible(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return rs._population(conn)  # the sample's own population rule


def _migrated_merges(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT decision_id FROM concept_decisions WHERE decided_by = 'agent' AND kind = 'merge' "
        "AND state = 'applied' AND batch_id IS NULL ORDER BY decision_id")]


def draw_sample(lib: Library, *, n: int = rs.SAMPLE_SIZE, seed: int = 0, now: str | None = None) -> SampleDraw:
    eligible = len(_eligible(lib.conn))
    if eligible >= n:
        return SampleDraw(rs.create_review_sample(lib, n=n, seed=seed, now=now), eligible, 0)
    now = now or datetime.now(UTC).isoformat()
    base = sorted(d["decision_id"] for d in _eligible(lib.conn))
    pool = [d for d in _migrated_merges(lib.conn) if d not in set(base)]
    extra = random.Random(seed).sample(pool, min(n - len(base), len(pool)))
    ids = sorted(base + extra)
    if not ids:
        raise rs.ResearcherError("no applied agent merges or links to sample yet")
    sid = "rs_" + hashlib.sha256(json.dumps([seed, n, ids]).encode()).hexdigest()[:12]
    body = {"seed": seed, "size": n, "created_at": now, "decision_ids": ids, "topped_up": len(extra)}
    with lib.write() as conn:
        conn.execute("INSERT OR IGNORE INTO library_meta (key, value) VALUES (?, ?)",
                     (rs.SAMPLE_META_PREFIX + sid, json.dumps(body, sort_keys=True)))
    return SampleDraw(rs.review_sample(lib.conn, sid), eligible, len(extra))  # type: ignore[arg-type]


def sample_items(conn: sqlite3.Connection, sample: rs.ReviewSample) -> list[dict[str, Any]]:
    amb = g.effective_ambiguous(conn)
    cols = rf.DECISION_COLS + ("confidence", "batch_id")
    rows = {d["decision_id"]: d for d in rf._rows(conn, "concept_decisions", cols, "decision_id")
            if d["decision_id"] in set(sample.decision_ids)}
    out = []
    for did in sample.decision_ids:
        d = rows.get(did)
        if d is None:
            continue
        fam = d["family"] or "any"
        out.append({**d, "family_label": fam if not d["family_b"] else f"{fam}->{d['family_b']}",
                    "ambiguous": amb.is_ambiguous(d["label_a"]) or amb.is_ambiguous(d["label_b"])})
    return out


def composition(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"size": len(items), "by_verdict": dict(sorted(Counter(i["kind"] for i in items).items())),
            "by_family": dict(sorted(Counter(i["family_label"] for i in items).items())),
            "ambiguous": sum(1 for i in items if i["ambiguous"]),
            "batches": dict(sorted(Counter(i["batch_id"] or "migrated" for i in items).items()))}


def render_sample_list(draw: SampleDraw, items: list[dict[str, Any]]) -> str:
    comp = composition(items)
    s = draw.sample
    lines = [f"# Review sample {s.sample_id}", "",
             f"- review screen: `{draw.path}`", f"- seed {s.seed}, size {s.size}, created {s.created_at}",
             f"- eligible applied agent merges and links: {draw.eligible}",
             f"- top-up from migrated agent merges: {draw.topped_up}" if draw.topped_up else
             "- no top-up needed (enough eligible rows)",
             f"- verdicts: {comp['by_verdict']}; ambiguous-label items: {comp['ambiguous']}",
             f"- families: {comp['by_family']}", "",
             "| # | decision | verdict | family | ambiguous | side a | side b | confidence |",
             "|---|---|---|---|---|---|---|---|"]
    for n, i in enumerate(items, 1):
        sa = i["label_a"] + (f"~{i['sense_a']}" if i["sense_a"] else "")
        sb = i["label_b"] + (f"~{i['sense_b']}" if i["sense_b"] else "")
        lines.append(f"| {n} | {i['decision_id']} | {i['kind']} | {i['family_label']} | "
                     f"{'yes' if i['ambiguous'] else ''} | {sa} | {sb} | {i['confidence'] or ''} |")
    return "\n".join(lines) + "\n"


# --- the report -----------------------------------------------------------------------------------

def _latest_eval(home: Path, given: str | None) -> Path | None:
    if given:
        p = Path(given)
        return p if p.is_file() else None
    root = home / "concepts" / "eval"
    found = sorted(root.glob("*/report.json")) if root.is_dir() else []
    return found[-1] if found else None


def _embeddings(conn: sqlite3.Connection, home: Path, inputs: Mapping[str, Any]) -> dict[str, Any]:
    from .search import adoption
    path = _latest_eval(home, inputs.get("eval_report"))
    out: dict[str, Any] = {"report": str(path.with_name("report.md")) if path else None,
                           "adopted": adoption(conn).adopted}
    if path:
        d = json.loads(path.read_text(encoding="utf-8"))
        v = d.get("verdict") or {}
        out["passed"] = v.get("passed")
        out["checks"] = v.get("checks", {})
        out["metrics"] = {mode: {k: m.get(k) for k in ("recall_at_10", "mrr", "separation")}
                          for mode, m in (d.get("modes") or {}).items()}
    return out


def _cost(conn: sqlite3.Connection, batch: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    q = conn.execute
    packages = q("SELECT COUNT(*), COALESCE(SUM(state = 'saved'), 0) FROM work_packages WHERE batch_id = ?",
                 (batch,)).fetchone()
    subs = dict(q("SELECT s.outcome, COUNT(*) FROM package_submissions s JOIN work_packages p USING (packet_id) "
                  "WHERE p.batch_id = ? GROUP BY s.outcome", (batch,)).fetchall())
    reported = q("SELECT COUNT(*) FROM package_submissions s JOIN work_packages p USING (packet_id) "
                 "WHERE p.batch_id = ? AND s.usage_json IS NOT NULL", (batch,)).fetchone()[0]
    workers = q("SELECT COUNT(DISTINCT c.worker_id) FROM package_submissions s JOIN work_packages p USING (packet_id) "
                "JOIN package_claims c ON c.claim_id = s.claim_id WHERE p.batch_id = ? AND s.outcome = 'accepted'",
                (batch,)).fetchone()[0]
    pairs = q("SELECT COUNT(*) FROM concept_decisions WHERE batch_id = ? AND kind IN ('merge', 'link', 'unrelated')",
              (batch,)).fetchone()[0]
    checks = q("SELECT COUNT(DISTINCT family || '/' || label_key) FROM concept_senses WHERE decision_ref = ?",
               (batch,)).fetchone()[0]
    est = inputs.get("estimate", {})
    measured = inputs.get("measured", {})
    tokens = measured.get("agent_tokens")
    high = est.get("agent_tokens_high")
    return {"packages": packages[0], "packages_saved": packages[1], "submissions": subs, "workers": workers,
            "pair_verdicts": pairs, "sense_checks": checks, "usage_reported": reported,
            "estimate": est, "measured": measured, "ticket11": inputs.get("ticket11", {}),
            "measured_vs_high": (tokens / high) if tokens and high else None}


def guardrail_checks(conn: sqlite3.Connection, batch: str, plan: rf.RefreshPlan) -> dict[str, Any]:
    audit = decide.audit(conn).as_dict()
    concept_of: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cid, _fam, key, _sense in plan.members:
        concept_of[key].add(cid)  # type: ignore[index]
    never = [r for r in conn.execute(
        "SELECT label_a, label_b FROM concept_decisions WHERE kind = 'never_merge' AND decided_by = 'researcher' "
        "AND state IN ('applied', 'pending_apply')")]
    present = [(x, y) for x, y in never if concept_of.get(x) and concept_of.get(y)]
    together = [(x, y) for x, y in present if concept_of[x] & concept_of[y]]
    residual = cu.findings(conn, batch).counts()
    return {"audit": audit, "never_merge_rows": len(never), "never_merge_present": len(present),
            "never_merge_together": len(together), "residual": residual}


def _family_label(d: Mapping[str, Any]) -> str:
    fam = d["family"] or "any"
    return fam if not d["family_b"] or d["family_b"] == fam else f"{fam}->{d['family_b']}"


def _verdict_class(kind: str, state: str) -> str:
    if state == "applied":
        return kind
    return f"{state} {kind}"


def _verdicts(conn: sqlite3.Connection, batch: str, inputs: Mapping[str, Any],
              cleanup: Mapping[str, Any] | None) -> dict[str, Any]:
    cols = rf.DECISION_COLS + ("packet_id", "batch_id")
    rows = [d for d in rf._rows(conn, "concept_decisions", cols, "decision_id")
            if d["batch_id"] == batch and d["kind"] in _PAIR_KINDS]
    old = {c["decision_id"]: c["old_state"] for c in (cleanup or {}).get("changes", [])}
    now_t: dict[str, Counter] = defaultdict(Counter)
    run_t: dict[str, Counter] = defaultdict(Counter)
    for d in rows:
        fam = _family_label(d)
        now_t[fam][_verdict_class(d["kind"], d["state"])] += 1
        run_t[fam][_verdict_class(d["kind"], old.get(d["decision_id"], d["state"]))] += 1
    worker = dict(conn.execute(
        "SELECT s.packet_id, c.worker_id FROM package_submissions s JOIN package_claims c ON c.claim_id = s.claim_id "
        "JOIN work_packages p ON p.packet_id = s.packet_id WHERE p.batch_id = ? AND s.outcome = 'accepted'",
        (batch,)).fetchall())
    rule = inputs.get("link_rule") or {}
    before = set(rule.get("before") or ())
    share: dict[str, dict[str, Any]] = {}
    for side, label in (("before", rule.get("label_before", "before")), ("after", rule.get("label_after", "after"))):
        picked = [d for d in rows if (worker.get(d["packet_id"]) in before) == (side == "before")]
        packets = {d["packet_id"] for d in picked}
        workers = sorted({worker.get(p) or "?" for p in packets}, key=lambda w: (len(w), w))
        links = sum(1 for d in picked if d["kind"] == "link")
        share[side] = {"label": label, "workers": workers, "packages": len(packets), "pair_verdicts": len(picked),
                       "links": links, "share": links / len(picked) if picked else None}

    def table(t: dict[str, Counter]) -> dict[str, dict[str, int]]:
        return {k: dict(sorted(v.items())) for k, v in sorted(t.items())}
    return {"now": table(now_t), "at_run": table(run_t), "link_share": share}


def _queue(conn: sqlite3.Connection) -> dict[str, Any]:
    items = rs.queue(conn)
    return {"size": len(items), "by_reason": dict(sorted(Counter(i.reason or "none" for i in items).items())),
            "by_proposal": dict(sorted(Counter(i.proposal for i in items).items()))}


def _review(conn: sqlite3.Connection, sample_id: str | None) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if sample_id is None:
        samples = rs.review_samples(conn)
        if not samples:
            return None, None
        sample_id = samples[0].sample_id
    sample = rs.review_sample(conn, sample_id)
    if sample is None:
        raise InputsError(f"no review sample {sample_id}")
    items = sample_items(conn, sample)
    raw = conn.execute("SELECT value FROM library_meta WHERE key = ?", (rs.SAMPLE_META_PREFIX + sample_id,)).fetchone()
    amb = g.effective_ambiguous(conn)
    eligible = _eligible(conn)
    info = {"sample_id": sample_id, "path": REVIEW_PATH.format(sample_id), "seed": sample.seed,
            "created_at": sample.created_at, "eligible_now": len(eligible),
            "ambiguous_eligible": sum(1 for d in eligible
                                      if amb.is_ambiguous(d["label_a"]) or amb.is_ambiguous(d["label_b"])),
            "topped_up": json.loads(raw[0]).get("topped_up", 0), **composition(items)}
    s = rs.review_summary(conn, sample_id)
    assert s is not None
    names = {i["decision_id"]: i for i in items}
    got = rs.marks(conn, sample_id)
    review = {"marked": s.marked, "size": s.size, "by_mark": s.by_mark, "by_verdict": s.by_verdict,
              "by_family": s.by_family, "merges_sampled": s.merges_sampled, "gate_limit": s.gate_limit,
              "gate_passed": s.gate_passed,
              "wrong_merges": [{"decision_id": d, "label_a": names.get(d, {}).get("label_a"),
                                "label_b": names.get(d, {}).get("label_b"),
                                "family": names.get(d, {}).get("family_label"), "note": got.get(d, ("", None))[1]}
                               for d in s.wrong_merges]}
    return info, review


def gather(conn: sqlite3.Connection, home: Path, inputs: Mapping[str, Any], *,
           sample_id: str | None = None, now: str | None = None) -> dict[str, Any]:
    batch = inputs["batch_id"]
    plan = rf.plan(conn)
    records = cu.records(conn, batch)
    cleanup = records[-1] if records else None
    sample, review = _review(conn, sample_id)
    return {"generated_at": now or datetime.now(UTC).isoformat(), "batch_id": batch,
            "embeddings": _embeddings(conn, home, inputs), "cost": _cost(conn, batch, inputs),
            "guardrails": guardrail_checks(conn, batch, plan),
            "cleanup": {"key": cleanup["key"], "counts": cleanup["counts"], "before": cleanup["before"],
                        "after": cleanup["after"], "applied_at": cleanup["at"]} if cleanup else None,
            "verdicts": _verdicts(conn, batch, inputs, cleanup), "queue": _queue(conn),
            "sample": sample, "review": review}


# --- rendering ------------------------------------------------------------------------------------

def _n(x: Any) -> str:
    return f"{x:,}" if isinstance(x, int) else ("-" if x is None else str(x))


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _counts_table(t: Mapping[str, Mapping[str, int]]) -> list[str]:
    cols = sorted({c for v in t.values() for c in v})
    lines = ["| family | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    total: Counter = Counter()
    for fam, v in t.items():
        total.update(v)
        lines.append(f"| {fam} | " + " | ".join(str(v.get(c, 0)) for c in cols) + " |")
    lines.append("| **total** | " + " | ".join(str(total.get(c, 0)) for c in cols) + " |")
    return lines


def render(d: Mapping[str, Any]) -> str:
    e, c, gr, v, q = d["embeddings"], d["cost"], d["guardrails"], d["verdicts"], d["queue"]
    out = ["# Stage 4 acceptance", "",
           f"Generated {d['generated_at']} by `seedgraph concepts acceptance` for batch `{d['batch_id']}`. "
           "Real labels: this report stays under the v3 home.", ""]
    out += ["## 1. Embeddings vs keyword", ""]
    if "passed" in e:
        verdict = "PASS: embeddings adopted" if e["passed"] else "FAIL: keyword kept"
        out += [f"- report: `{e['report']}`", f"- verdict: **{verdict}**; library adoption flag: "
                f"{'adopted' if e['adopted'] else 'not adopted (auto search = keyword)'}",
                "- pass-rule checks: " + ", ".join(f"{k} {'yes' if ok else 'no'}" for k, ok in e["checks"].items()),
                "", "| metric | " + " | ".join(e["metrics"]) + " |", "|---|" + "---|" * len(e["metrics"])]
        for k in ("recall_at_10", "mrr", "separation"):
            out.append(f"| {k} | " + " | ".join("-" if m.get(k) is None else f"{m[k]:.3f}"
                                                for m in e["metrics"].values()) + " |")
    else:
        out.append("- no embeddings evaluation report found")
    est, mea, t11 = c["estimate"], c["measured"], c["ticket11"]
    out += ["", "## 2. Cost vs estimate (first bridge pass)", "",
            "| | dry-run estimate | actual | ticket 11 |", "|---|---|---|---|",
            f"| packages | {_n(est.get('packages'))} | {_n(c['packages'])} ({_n(c['packages_saved'])} saved) | "
            f"{t11.get('packages', '-')} |",
            f"| pair verdicts | {_n(est.get('pairs'))} | {_n(c['pair_verdicts'])} | {t11.get('pairs', '-')} |",
            f"| sense checks | {_n(est.get('sense_checks'))} | {_n(c['sense_checks'])} labels | - |",
            f"| agent tokens | {_n(est.get('agent_tokens_low'))} - {_n(est.get('agent_tokens_high'))} | "
            f"{_n(mea.get('agent_tokens'))} (measured) | {t11.get('tokens', '-')} |", "",
            f"- measured tokens: {mea.get('source', 'not stated')}; submissions reporting usage: "
            f"{c['usage_reported']} (so the total is the orchestrator's measurement, not the library's)",
            f"- actual vs the estimate's high end: {_pct(c['measured_vs_high'])}",
            f"- submissions: {c['submissions']}; accepted from {c['workers']} workers"]
    a = gr["audit"]
    out += ["", "## 3. Guardrails (after the cleanup)", "",
            f"- `concepts check` audit: **{'ok' if a['ok'] else 'FAIL'}** "
            + ", ".join(f"{k} {n}" for k, n in a["violations"].items()),
            f"- researcher never-merge rows: {gr['never_merge_rows']}; both labels present: "
            f"{gr['never_merge_present']}; pairs sharing a concept: **{gr['never_merge_together']}**"]
    r = gr["residual"]
    out += ["- bridge-check audit re-run on the batch:",
            f"  - concept pairs with different *applied* verdicts: **{r['conflicts_both_applied']}**",
            f"  - concept pairs with different live verdicts: {r['conflicts']} "
            f"(not waiting for the researcher as a disagreement: **{r['conflicts_unsettled']}**)",
            f"  - live link / unrelated rows inside one concept: **{r['inside']}**; contradicted merges: "
            f"**{r['contradicted']}**", f"  - live rows on a sense that no longer exists: **{r['dead']}**",
            f"  - rows inside one concept that are the researcher's (queued as a disagreement, or joined by a "
            f"researcher merge): {r.get('overridden', 0)}"]
    cl = d["cleanup"]
    if cl:
        out += ["", f"Cleanup `{cl['key']}`: " + ", ".join(f"{k} {n}" for k, n in cl["counts"].items())
                + f" (findings before: {cl['before']})"]
    out += ["", "## 4. Verdicts", "", "### Now (after the cleanup)", ""] + _counts_table(v["now"])
    out += ["", "### As stored by the run (before the cleanup)", ""] + _counts_table(v["at_run"])
    out += ["", "### Link share before / after the stricter link rule", "",
            "| rule | workers | packages with pairs | pair verdicts | links | link share |",
            "|---|---|---|---|---|---|"]
    for side in ("before", "after"):
        s = v["link_share"][side]
        ws = s["workers"]
        span = f"{ws[0]}-{ws[-1]}" if len(ws) > 2 else ", ".join(ws)
        out.append(f"| {s['label']} | {span} | {s['packages']} | {s['pair_verdicts']} | {s['links']} | "
                   f"{_pct(s['share'])} |")
    out += ["", "(Pair verdicts include queued and cleaned-up rows; worker from the accepted submission's claim.)",
            "", "## 5. Queue", "", f"- queue size: **{q['size']}**",
            f"- by reason: {q['by_reason']}", f"- by proposal: {q['by_proposal']}"]
    sm, rv = d["sample"], d["review"]
    out += ["", "## 6. Review sample", ""]
    if sm is None:
        out += ["- no review sample yet (`seedgraph concepts review-sample`)"]
    else:
        out += [f"- sample `{sm['sample_id']}` (seed {sm['seed']}, created {sm['created_at']}): review at "
                f"`{sm['path']}`",
                f"- size {sm['size']}: {sm['by_verdict']}; ambiguous-label items {sm['ambiguous']} "
                f"(eligible ambiguous-label rows now: {sm['ambiguous_eligible']}; the draw over-samples up to "
                f"{rs.AMBIGUOUS_MIN})",
                f"- families: {sm['by_family']}", f"- batches: {sm['batches']}",
                f"- top-up from migrated agent merges: {sm['topped_up']}" if sm["topped_up"] else
                f"- no top-up needed (eligible rows now: {sm['eligible_now']})"]
    out += ["", "## 7. Researcher review", ""]
    if rv is None:
        out += ["- no review sample yet"]
    else:
        gate = rv["gate_passed"]
        state = (f"pending: {rv['marked']} of {rv['size']} marked" if gate is None
                 else ("PASS" if gate else "FAIL"))
        out += [f"- marked: {rv['marked']} of {rv['size']}",
                f"- merges sampled: {rv['merges_sampled']}; allowed wrong merges: {rv['gate_limit']:g} "
                f"({'3' if rv['merges_sampled'] >= 30 else '10% (fewer than 30 merges sampled)'})",
                f"- wrong merges: {len(rv['wrong_merges'])}", f"- gate: {state}",
                f"- by mark: {rv['by_mark']}", f"- by verdict: {rv['by_verdict']}", f"- by family: {rv['by_family']}"]
        if rv["wrong_merges"]:
            out += ["", "| wrong merge | family | side a | side b | note |", "|---|---|---|---|---|"]
            for w in rv["wrong_merges"]:
                out.append(f"| {w['decision_id']} | {w['family']} | {w['label_a']} | {w['label_b']} | "
                           f"{w['note'] or ''} |")
        out += ["", "A gate miss means tightening the unclear thresholds and re-running, not adopting (§11.2)."]
    return "\n".join(out) + "\n"


__all__ = ["InputsError", "SampleDraw", "composition", "draw_sample", "gather", "guardrail_checks", "load_inputs",
           "render",
           "render_sample_list", "sample_items"]

