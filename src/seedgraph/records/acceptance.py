"""The stage 5 acceptance harness (design §10; slice R8).

Mirrors :mod:`seedgraph.concepts.acceptance`: pure functions the ``records`` commands call,
with the library opened read-only.

- the **transcript reader** (:func:`read_transcript`, :func:`logged_calls`) and the
  **cross-check** (:func:`cross_check`): a Claude Code session transcript's successful calls
  of the nine logged seedgraph tools, compared with a record's events by count, key
  arguments and order (§10.1);
- the **per-record checklist** (:func:`sync_checklist`, :func:`parse_ticks`): written once,
  filled by hand, never overwritten (new records are appended);
- the **report** (:func:`gather`, :func:`render`): the five questions, planted citations,
  paths without answer, the identity probe and the spot-check, with the walk-through
  checklist (:func:`sync_walkthrough`).

Real questions, answers and marks appear in what these write, so everything goes under
``<home>/acceptance/stage5/`` and nothing real is committed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import manage
from . import path as paths

if TYPE_CHECKING:
    from ..library.store import Library

STAGE_DIR = ("acceptance", "stage5")
QUESTIONS_FILE = "questions.yaml"
CHECKLIST_FILE = "records_check.md"
WALKTHROUGH_FILE = "walkthrough_check.md"
CROSSCHECK_FILE = "crosscheck.json"
REPORT_FILE = "report.md"
REPORT_JSON = "report.json"
SPOTCHECK_JSON = "spotcheck.json"
SPOTCHECK_MD = "spotcheck.md"

SERVER_NAME = "seedgraph_v3"
#: the nine tools whose successful calls the session log records (§3.2)
LOGGED_TOOLS: tuple[str, ...] = ("search", "paper_list", "lens_results", "concept_search", "literatures",
                                 "concept_hubs", "paper_show", "paper_read", "concept_show")
KEY_ARGS: dict[str, str] = {"search": "query", "paper_list": "query", "concept_search": "query",
                            "paper_show": "paper_id", "paper_read": "paper_id", "concept_show": "handle",
                            "lens_results": "lens_id"}
QUESTION_COUNT = 5
SLACK = timedelta(seconds=60)

#: §10.1: what the researcher checks on each record page.
CHECKLIST_ITEMS: tuple[str, ...] = (
    "the question is right",
    "the answer matches what the agent said",
    "every search appears with its query and hits",
    "every paper or section opened appears",
    "the tags (found / opened / cited) are right",
    "the cited passages are the ones the agent named",
    "the judgement is apart from the answer",
    "the graphs are right",
)

#: §10.2, §10.3, §10.4, §10.6 and Q12: what only the researcher can do.
WALKTHROUGH_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Setup", (
        "`seedgraph-v3 setup agents` shows the \"always allow\" hint for `record_answer`; approved it once in Claude Code",
        "questions.yaml holds five real questions; at least one is asked from Codex; one uses two helper subagents",
    )),
    ("Live: planted unread citation (§10.2)", (
        "asked the agent to cite a named span without opening it; the record shows \"cited without being read\"",
    )),
    ("Live: path without answer (§10.3)", (
        "stopped an agent mid-search; the record shows \"path without answer\" in Records and on Home",
    )),
    ("Live: identity probe (§10.4)", (
        "parent plus two named helper subagents: the parent's record shows helper-tagged reads, nothing crossed over",
        "the same run without names: one connection stream, marked \"session told apart by connection only\"",
        "a server restart mid-question with a name kept one record",
        "the `_meta` keys and client in the report's identity section are noted (open question 1)",
    )),
    ("Walk-through (§10.6)", (
        "filtered the Records list by graph",
        "opened a record that spans two graphs",
        "starred, annotated and exported one record",
        "deleted one record",
        "started a spot-check, marked every item, fixed one with \"fix it now\", read the accuracy table",
    )),
)

_TICK = re.compile(r"^\s*-\s*\[([ xX])\]", re.M)


class AcceptanceError(ValueError):
    """The acceptance inputs (questions, transcript) are missing or malformed."""


def stage_dir(home: Path) -> Path:
    return Path(home).joinpath(*STAGE_DIR)


# --- the transcript reader ----------------------------------------------------------------------

@dataclass(frozen=True)
class TranscriptCall:
    """One seedgraph tool call found in a Claude Code transcript."""

    tool: str
    input: Mapping[str, Any]
    tool_use_id: str
    at: str | None            # when the assistant issued it
    done_at: str | None       # when its result came back
    ok: bool
    result: str               # the result text (carries ``record_id`` for record_answer)
    sidechain: bool
    order: int                # position in the files read, the tiebreak

    @property
    def stream(self) -> str | None:
        s = self.input.get("session")
        return s if isinstance(s, str) and s else None

    @property
    def when(self) -> str:
        return self.done_at or self.at or ""


@dataclass(frozen=True)
class Transcript:
    calls: tuple[TranscriptCall, ...]
    files: tuple[str, ...]     # file names read (main + subagents), no content


def _blocks(message: Any) -> list[Mapping[str, Any]]:
    content = message.get("content") if isinstance(message, Mapping) else None
    return [b for b in content if isinstance(b, Mapping)] if isinstance(content, list) else []


def _result_text(block: Mapping[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(b.get("text", "")) for b in content if isinstance(b, Mapping) and b.get("type") == "text")
    return ""


def _failed(text: str, is_error: Any) -> bool:
    if is_error:
        return True
    try:
        body = json.loads(text)
    except ValueError:
        return False
    err = body.get("error") if isinstance(body, dict) else None
    return isinstance(err, dict) and "code" in err


def transcript_files(path: Path) -> list[Path]:
    """The main transcript and the subagent transcripts beside it (``<stem>/subagents/*.jsonl``)."""
    path = Path(path)
    if not path.is_file():
        raise AcceptanceError(f"no transcript at {path}")
    subs = sorted((path.with_suffix("") / "subagents").glob("*.jsonl"))
    return [path, *subs]


def read_transcript(path: Path, *, server: str = SERVER_NAME) -> Transcript:
    """Every seedgraph tool call in a Claude Code session transcript, with its result.

    Structure (Claude Code JSONL): ``type == "assistant"`` lines hold ``message.content``
    blocks ``tool_use {id, name, input}``; ``type == "user"`` lines hold ``tool_result
    {tool_use_id, is_error, content}``. ``api-request-blob`` lines repeat those blocks and are
    ignored. Subagent transcripts are read too. A tool use is counted once by its id."""
    prefix = f"mcp__{server}__"
    uses: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    files = transcript_files(path)
    n = 0
    for idx, file in enumerate(files):
        for line in file.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict):
                continue
            kind = row.get("type")
            if kind not in ("assistant", "user"):
                continue
            for block in _blocks(row.get("message")):
                if kind == "assistant" and block.get("type") == "tool_use":
                    name, uid = str(block.get("name", "")), block.get("id")
                    if name.startswith(prefix) and isinstance(uid, str) and uid not in uses:
                        inp = block.get("input")
                        n += 1
                        uses[uid] = {"tool": name[len(prefix):], "input": inp if isinstance(inp, dict) else {},
                                     "at": row.get("timestamp"), "sidechain": bool(row.get("isSidechain")) or idx > 0,
                                     "order": n}
                elif kind == "user" and block.get("type") == "tool_result":
                    uid = block.get("tool_use_id")
                    if isinstance(uid, str) and uid not in results:
                        text = _result_text(block)
                        results[uid] = {"done_at": row.get("timestamp"), "text": text,
                                        "failed": _failed(text, block.get("is_error"))}
    calls = []
    for uid, u in uses.items():
        r = results.get(uid)
        calls.append(TranscriptCall(u["tool"], u["input"], uid, u["at"], r["done_at"] if r else None,
                                    bool(r) and not r["failed"], r["text"] if r else "", u["sidechain"], u["order"]))
    calls.sort(key=lambda c: (c.when, c.order))
    return Transcript(tuple(calls), tuple(f.name for f in files))


def logged_calls(calls: Iterable[TranscriptCall]) -> list[TranscriptCall]:
    """The calls the session log should hold: successful calls of the nine logged tools
    (``paper_list`` only with a non-empty ``query``). ``record_answer``, error replies and
    invalid-session refusals are out."""
    out = []
    for c in calls:
        if c.tool not in LOGGED_TOOLS or not c.ok:
            continue
        if c.tool == "paper_list" and not str(c.input.get("query") or "").strip():
            continue
        out.append(c)
    return out


def _record_id(call: TranscriptCall) -> str | None:
    try:
        body = json.loads(call.result)
    except ValueError:
        return None
    rid = body.get("record_id") if isinstance(body, dict) else None
    return rid if isinstance(rid, str) else None


def _root(stream: str | None) -> str | None:
    return stream.partition("/")[0] if stream else None


def record_window(calls: Sequence[TranscriptCall], record_id: str) -> list[TranscriptCall]:
    """The logged calls a record owns: those after the family's previous successful
    ``record_answer`` up to the one that returned ``record_id``, in the answering stream and
    its helpers (a named stream is told apart by its name root, unnamed calls together)."""
    answers = [c for c in calls if c.tool == "record_answer" and c.ok]
    target = next((a for a in answers if _record_id(a) == record_id), None)
    if target is None:
        raise AcceptanceError(f"the transcript holds no successful record_answer for {record_id}")
    root = _root(target.stream)
    earlier = [a for a in answers if _root(a.stream) == root and a.when < target.when]
    lo = max((a.when for a in earlier), default="")
    return [c for c in logged_calls(calls) if _root(c.stream) == root and lo < c.when <= target.when]


# --- the cross-check ----------------------------------------------------------------------------

Sig = tuple[str, str | None, str | None]  # (tool, key argument, stream)


def signature(tool: str, args: Mapping[str, Any], stream: str | None) -> Sig:
    key = KEY_ARGS.get(tool)
    value = args.get(key) if key else None
    return tool, (str(value) if value is not None else None), stream


@dataclass(frozen=True)
class CrossCheck:
    transcript: int
    events: int
    missing: tuple[Sig, ...]      # in the transcript, not in the log
    extra: tuple[Sig, ...]        # in the log, not in the transcript
    order: tuple[str, ...]        # streams whose order differs
    unlogged_calls: int = 0

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.order)


def _sigs(items: Iterable[Sig]) -> Counter[Sig]:
    return Counter(items)


def cross_check(calls: Sequence[TranscriptCall], events: Sequence[Mapping[str, Any]], *,
                unlogged_calls: int = 0) -> CrossCheck:
    """Compare logged transcript calls with events (dicts with ``tool``, ``args``, ``stream``,
    ``seq``): as a multiset of (tool, key argument, stream), then each stream in order."""
    t = [signature(c.tool, c.input, c.stream) for c in calls]
    e = [signature(ev["tool"], ev["args"], ev.get("stream")) for ev in sorted(events, key=lambda ev: ev["seq"])]
    tc, ec = _sigs(t), _sigs(e)
    missing = tuple(sorted((tc - ec).elements(), key=repr))
    extra = tuple(sorted((ec - tc).elements(), key=repr))
    common = tc & ec
    order = []
    for stream in sorted({s[2] for s in common}, key=lambda s: s or ""):
        want = Counter({s: n for s, n in common.items() if s[2] == stream})
        seq_t = _keep(t, want)
        seq_e = _keep(e, want)
        if seq_t != seq_e:
            order.append(stream or "(unnamed)")
    return CrossCheck(len(t), len(e), missing, extra, tuple(order), unlogged_calls)


def _keep(seq: Sequence[Sig], allowed: Counter[Sig]) -> list[Sig]:
    left, out = Counter(allowed), []
    for s in seq:
        if left.get(s, 0) > 0:
            left[s] -= 1
            out.append(s)
    return out


def load_events(conn: sqlite3.Connection, *, record_ids: Sequence[str] = (),
                since: datetime | None = None, until: datetime | None = None) -> list[dict[str, Any]]:
    """Events of the given records, or (with no records) the events in a time window, as
    dicts for :func:`cross_check`. ``stream`` is the session name when the agent named it."""
    sql = ("SELECT e.seq, e.at, e.tool, e.args_json, e.record_id, s.name AS name, s.key_kind AS kind "
           "FROM traversal_events e JOIN record_sessions s ON s.session_id = e.session_id ")
    if record_ids:
        rows = conn.execute(sql + "WHERE e.record_id IN (SELECT value FROM json_each(?)) ORDER BY e.seq",
                            (json.dumps(list(record_ids)),)).fetchall()
    else:
        rows = [r for r in conn.execute(sql + "ORDER BY e.seq")
                if (since is None or datetime.fromisoformat(r["at"]) >= since)
                and (until is None or datetime.fromisoformat(r["at"]) <= until)]
    return [{"seq": r["seq"], "at": r["at"], "tool": r["tool"], "args": json.loads(r["args_json"]),
             "record_id": r["record_id"], "stream": r["name"] if r["kind"] == "named" else None} for r in rows]


def unlogged_for(conn: sqlite3.Connection, record_ids: Sequence[str]) -> int:
    if not record_ids:
        return 0
    return int(conn.execute("SELECT COALESCE(SUM(unlogged_calls), 0) FROM traversal_records "
                            "WHERE record_id IN (SELECT value FROM json_each(?))",
                            (json.dumps(list(record_ids)),)).fetchone()[0])


def _parse_ts(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def check_transcript(conn: sqlite3.Connection, transcript: Transcript, *,
                     record_ids: Sequence[str] = ()) -> dict[str, CrossCheck]:
    """The cross-check per record (``record_ids`` given), or one ``window`` check over the
    transcript's time span (plus a minute each side) when none is."""
    if record_ids:
        out = {}
        for rid in record_ids:
            window = record_window(transcript.calls, rid)
            out[rid] = cross_check(window, load_events(conn, record_ids=[rid]), unlogged_calls=unlogged_for(conn, [rid]))
        return out
    calls = logged_calls(transcript.calls)
    if not calls:
        return {"window": CrossCheck(0, 0, (), (), ())}
    whens = [_parse_ts(c.when) for c in calls if c.when]
    events = load_events(conn, since=min(whens) - SLACK, until=max(whens) + SLACK)
    rids = sorted({e["record_id"] for e in events if e["record_id"]})
    return {"window": cross_check(calls, events, unlogged_calls=unlogged_for(conn, rids))}


def render_cross_check(name: str, cc: CrossCheck) -> str:
    def fmt(s: Sig) -> str:
        return f"{s[0]}({s[1] or ''})" + (f" [{s[2]}]" if s[2] else "")
    lines = [f"{name}: {'OK' if cc.ok else 'FAIL'}: transcript {cc.transcript} calls, log {cc.events} events"]
    lines += [f"  missing from the log: {fmt(s)}" for s in cc.missing]
    lines += [f"  extra in the log: {fmt(s)}" for s in cc.extra]
    lines += [f"  order differs in stream {s}" for s in cc.order]
    if cc.unlogged_calls:
        lines.append(f"  {cc.unlogged_calls} call(s) could not be logged (the record says so)")
    return "\n".join(lines)


def cross_check_json(cc: CrossCheck, now: datetime | None = None) -> dict[str, Any]:
    return {"at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "ok": cc.ok,
            "transcript": cc.transcript, "events": cc.events, "missing": [list(s) for s in cc.missing],
            "extra": [list(s) for s in cc.extra], "order": list(cc.order), "unlogged_calls": cc.unlogged_calls}


def save_cross_checks(home: Path, checks: Mapping[str, CrossCheck], *, now: datetime | None = None) -> Path:
    path = stage_dir(home) / CROSSCHECK_FILE
    old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    old.update({k: cross_check_json(v, now) for k, v in checks.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(old, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return path


# --- questions and the per-record checklist -----------------------------------------------------

QUESTIONS_TEMPLATE = """\
# The five real questions (design §10.1). Write them before the run.
# After each answer, set `record:` to the record id shown on its page (/records/tr_...).
# At least one question is asked from Codex; one uses two helper subagents.
questions:
  - id: q1
    text: ""            # the question, on the migrated DiD or land_value graphs
    client: claude      # claude | codex
    helpers: false      # true when this one uses two helper subagents
    record: ""
  - id: q2
    text: ""
    client: claude
    helpers: false
    record: ""
  - id: q3
    text: ""
    client: claude
    helpers: true
    record: ""
  - id: q4
    text: ""
    client: claude
    helpers: false
    record: ""
  - id: q5
    text: ""
    client: codex
    helpers: false
    record: ""
"""


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    client: str
    helpers: bool
    record: str | None


def load_questions(path: Path) -> list[Question]:
    import yaml

    if not Path(path).is_file():
        raise AcceptanceError(f"no questions file at {path}")
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise AcceptanceError(f"{path} is not valid YAML: {exc}") from None
    rows = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise AcceptanceError(f"{path}: expected a `questions:` list")
    out = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise AcceptanceError(f"{path}: question {i} is not a mapping")
        out.append(Question(str(row.get("id") or f"q{i}"), str(row.get("text") or "").strip(),
                            str(row.get("client") or "claude").strip().lower(), bool(row.get("helpers")),
                            str(row.get("record") or "").strip() or None))
    return out


def _section_heading(record_id: str) -> str:
    return f"## Record {record_id}"


def _sections(text: str) -> dict[str, str]:
    """Heading line -> section text (heading included), for ``## `` sections."""
    parts = re.split(r"(?m)^(?=## )", text)
    return {p.splitlines()[0].strip(): p for p in parts if p.startswith("## ")}


def render_record_section(detail: manage.RecordDetail, label: str | None = None) -> str:
    r = detail.row
    graphs = ", ".join(g.slug for g in r.graphs) or "library-wide"
    lines = [_section_heading(r.record_id) + (f" ({label})" if label else ""), "",
             f"- page: `/records/{r.record_id}`; state: {r.state_label}; client: {r.client or 'unknown'}; "
             f"session: {r.session_name} ({r.key_kind}); graphs: {graphs}",
             f"- counts: {r.counts}; flags: {list(r.flags) or 'none'}; notes: {list(r.notes) or 'none'}", ""]
    lines += [f"- [ ] {item}" for item in CHECKLIST_ITEMS]
    return "\n".join(lines) + "\n\n"


CHECKLIST_HEAD = (
    "# Stage 5 acceptance: per-record checklist (design §10.1)\n\n"
    "Open each record page and tick every line that holds. This file is yours: `records acceptance-report` "
    "never rewrites a section, it only appends records not yet here. A record passes when all its lines "
    "are ticked.\n\n")


def sync_checklist(existing: str | None, details: Sequence[tuple[manage.RecordDetail, str | None]]) -> tuple[str, list[str]]:
    """The checklist file text with a section appended per record not yet in it. Existing
    text is kept byte for byte. Returns (text, record ids added)."""
    text = existing if existing is not None else CHECKLIST_HEAD
    have = {h.split()[2] for h in _sections(text) if h.startswith("## Record ")}
    added = []
    for detail, label in details:
        if detail.row.record_id in have:
            continue
        if text and not text.endswith("\n"):
            text += "\n"
        text += render_record_section(detail, label)
        added.append(detail.row.record_id)
    return text, added


def parse_ticks(text: str) -> dict[str, tuple[int, int]]:
    """Record id -> (ticked, total) per ``## Record`` section."""
    out = {}
    for heading, body in _sections(text).items():
        if heading.startswith("## Record "):
            marks = _TICK.findall(body)
            out[heading.split()[2]] = (sum(1 for m in marks if m != " "), len(marks))
    return out


def sync_walkthrough(existing: str | None) -> str:
    """The walk-through checklist (written once; the ticks are the researcher's)."""
    if existing is not None:
        return existing
    lines = ["# Stage 5 acceptance: live variants and walk-through (design §10.2 to §10.6, §13 Q12)", "",
             "Tick each line once you have done it. `records acceptance-report` never rewrites this file.", ""]
    for title, items in WALKTHROUGH_SECTIONS:
        lines += [f"## {title}", ""] + [f"- [ ] {i}" for i in items] + [""]
    return "\n".join(lines)


def walkthrough_progress(text: str | None) -> dict[str, tuple[int, int]]:
    if text is None:
        return {}
    return {h[3:].strip(): (sum(1 for m in _TICK.findall(b) if m != " "), len(_TICK.findall(b)))
            for h, b in _sections(text).items()}


# --- the spot-check report ----------------------------------------------------------------------

def _row_json(row: Any) -> dict[str, Any]:
    return {"claim_type": row.claim_type, "model_key": row.model_key, "right": row.n_right, "partly": row.n_partly,
            "wrong": row.n_wrong, "marked": row.n_marked, "summary": row.summary}


def spotcheck_data(lib: Library, sample_id: str | None = None, *, previous: Mapping[str, Any] | None = None,
                   now: datetime | None = None) -> dict[str, Any] | None:
    """The §10.5 figures: the sample's composition, marks per cell, the seed, accuracy (type,
    model, overall weighted), which items are fixed, and the snapshots that show the figure
    did not move when a fix landed. ``None`` when no sample exists."""
    from .. import spotcheck
    from ..spotcheck import accuracy as acc_mod
    from ..spotcheck import sample as sample_mod

    got = sample_mod.sample(lib, sample_id) if sample_id else next(iter(spotcheck.samples(lib)), None)
    if got is None:
        return None
    table = acc_mod.accuracy(lib)
    items = got.items
    marks = Counter(i.mark for i in items if i.mark)
    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for i in items:
        cells[(i.claim_type, i.model_key)]["items"] += 1
        if i.mark:
            cells[(i.claim_type, i.model_key)][i.mark] += 1
    figure = {"right": marks.get("right", 0), "partly": marks.get("partly", 0), "wrong": marks.get("wrong", 0)}
    o = table.overall
    snap = {"at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), "sample_id": got.sample_id,
            "n_marked": sum(marks.values()), "n_fixed": sum(1 for i in items if i.fixed), "figure": figure,
            "overall": {"right": o.share_right, "partly": o.share_partly, "wrong": o.share_wrong,
                        "marked": o.n_marked}}
    history = [s for s in (previous or {}).get("snapshots", []) if s.get("sample_id") == got.sample_id]
    key = lambda s: (s["n_marked"], s["n_fixed"], s["figure"], s["overall"])  # noqa: E731
    if not history or key(history[-1]) != key(snap):
        history.append(snap)
    return {
        "sample_id": got.sample_id, "seed": got.seed, "size": got.size, "items": len(items),
        "created_at": got.created_at, "finished_at": got.finished_at, "filters": dict(got.filters),
        "marked": snap["n_marked"], "all_marked": snap["n_marked"] == len(items) > 0, "figure": figure,
        "fixed": [{"claim_id": i.claim_id, "mark": i.mark, "claim_type": i.claim_type} for i in items if i.fixed],
        "cells": [{"claim_type": t, "model_key": m, "items": c["items"], "right": c["right"], "partly": c["partly"],
                   "wrong": c["wrong"]} for (t, m), c in sorted(cells.items())],
        "accuracy": {"cells": [_row_json(r) for r in table.cells], "by_type": [_row_json(r) for r in table.by_type],
                     "by_model": [_row_json(r) for r in table.by_model],
                     "overall": {"summary": o.summary, "marked": o.n_marked, "population": o.population,
                                 "covered": o.covered}},
        "snapshots": history, "unchanged_after_fix": figure_unchanged_after_fix(history),
    }


def figure_unchanged_after_fix(snapshots: Sequence[Mapping[str, Any]]) -> bool | None:
    """Whether the accuracy figure is the same across snapshots with the same marks but more
    fixes. ``None`` when no snapshot pair shows a fix landing; ``False`` on any movement."""
    verdict: bool | None = None
    for a, b in zip(snapshots, snapshots[1:]):
        if b["n_marked"] == a["n_marked"] and b["n_fixed"] > a["n_fixed"]:
            same = a["figure"] == b["figure"] and a["overall"] == b["overall"]
            verdict = same if verdict is None else (verdict and same)
    return verdict


def render_spotcheck(d: Mapping[str, Any] | None) -> str:
    out = ["# Stage 5 acceptance: spot-check (design §10.5)", ""]
    if d is None:
        return "\n".join(out + ["- no spot-check sample yet: start one at `/spot-check`"]) + "\n"
    fig = d["figure"]
    out += [f"- sample `{d['sample_id']}`: seed {d['seed']}, {d['items']} items (asked for {d['size']}), "
            f"filters {d['filters'] or 'none'}",
            f"- marked {d['marked']} of {d['items']}: right {fig['right']}; partly right {fig['partly']}; "
            f"wrong {fig['wrong']}",
            f"- items fixed through \"fix it now\": {len(d['fixed'])}",
            "- figure unchanged after a fix: " + {None: "not shown yet (mark all, run this, fix one, run again)",
                                                  True: "yes", False: "NO (the figure moved)"}[d["unchanged_after_fix"]],
            f"- overall: {d['accuracy']['overall']['summary']}", "",
            "| claim type | model | items | right | partly | wrong |", "|---|---|---|---|---|---|"]
    out += [f"| {c['claim_type']} | {c['model_key']} | {c['items']} | {c['right']} | {c['partly']} | {c['wrong']} |"
            for c in d["cells"]]
    for title, key in (("By claim type", "by_type"), ("By model", "by_model"), ("By type and model", "cells")):
        out += ["", f"### {title}", ""]
        out += [f"- {r['claim_type'] or ''}{' / ' if r['claim_type'] and r['model_key'] else ''}{r['model_key'] or ''}: "
                f"{r['summary']}" for r in d["accuracy"][key]]
    out += ["", "### Snapshots (one per change in marks, fixes or figure)", "",
            "| at | marked | fixed | right | partly | wrong |", "|---|---|---|---|---|---|"]
    out += [f"| {s['at']} | {s['n_marked']} | {s['n_fixed']} | {s['figure']['right']} | {s['figure']['partly']} | "
            f"{s['figure']['wrong']} |" for s in d["snapshots"]]
    return "\n".join(out) + "\n"


# --- the report ---------------------------------------------------------------------------------

def _identity(lib: Library) -> dict[str, Any]:
    q = lib.conn.execute
    conns = [{"connection_id": r[0], "client": r[1], "started_at": r[2], "closed_at": r[3],
              "meta_keys": json.loads(r[4] or "[]")}
             for r in q("SELECT connection_id, client, started_at, closed_at, meta_keys_json FROM record_connections "
                        "ORDER BY started_at")]
    kinds = dict(q("SELECT key_kind, COUNT(*) FROM record_sessions GROUP BY key_kind").fetchall())
    helpers = [{"name": r[0], "parent": r[1]} for r in q(
        "SELECT s.name, p.name FROM record_sessions s JOIN record_sessions p ON p.session_id = s.parent_session_id "
        "ORDER BY s.name")]
    multi = [r[0] for r in q(
        "SELECT s.name FROM record_sessions s WHERE s.key_kind = 'named' AND (SELECT COUNT(DISTINCT e.connection_id) "
        "FROM traversal_events e WHERE e.session_id = s.session_id) > 1 ORDER BY s.name")]
    return {"connections": conns, "sessions_by_kind": kinds, "helpers": helpers, "named_on_several_connections": multi,
            "meta_keys_seen": sorted({k for c in conns for k in c["meta_keys"]}),
            "clients": sorted({c["client"] for c in conns if c["client"]})}


def _record_json(r: manage.RecordRow) -> dict[str, Any]:
    return {"record_id": r.record_id, "state": r.state, "client": r.client, "session": r.session_name,
            "key_kind": r.key_kind, "counts": r.counts, "flags": list(r.flags), "notes": list(r.notes),
            "graphs": [g.slug for g in r.graphs], "unlogged_calls": r.unlogged_calls, "last_event_at": r.last_event_at}


def gather(lib: Library, home: Path, *, now: datetime | None = None, sample_id: str | None = None,
           checklist: str | None = None, walkthrough: str | None = None,
           cross_checks: Mapping[str, Any] | None = None, questions: Sequence[Question] = (),
           previous_spotcheck: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The report's figures from a library (read-only) and the hand-filled files."""
    now = now or datetime.now(timezone.utc)
    page = manage.list_records(lib, limit=100_000, now=now)
    by_id = {r.record_id: r for r in page.rows}
    ticks = parse_ticks(checklist or "")
    cc = dict(cross_checks or {})
    qs = []
    for q in questions:
        r = by_id.get(q.record) if q.record else None
        done, total = ticks.get(q.record or "", (0, 0))
        check = cc.get(q.record or "")
        qs.append({"id": q.id, "text": q.text, "client": q.client, "helpers": q.helpers, "record_id": q.record,
                   "found": r is not None, "record": _record_json(r) if r else None,
                   "client_matches": r is not None and (r.client or "") == q.client,
                   "checklist": {"ticked": done, "total": total, "complete": bool(total) and done == total},
                   "cross_check": check})
    complete = [q for q in qs if q["checklist"]["complete"]]
    clean = [q for q in qs if q["cross_check"] and q["cross_check"]["ok"]]
    claude_qs = [q for q in qs if q["client"] != "codex"]
    claude_clean = [q for q in claude_qs if q["cross_check"] and q["cross_check"]["ok"]]
    passed = (len(qs) >= QUESTION_COUNT and len(complete) >= QUESTION_COUNT
              and len(claude_clean) == len(claude_qs) and any(q["client"] == "codex" for q in qs))
    flagged = [r for r in page.rows if r.flags]
    pwa = [r for r in page.rows if r.state == paths.STATE_NO_ANSWER]
    return {
        "generated_at": now.isoformat(timespec="seconds"), "records_total": page.total,
        "by_state": dict(sorted(Counter(r.state for r in page.rows).items())),
        "by_client": dict(sorted(Counter(r.client or "unknown" for r in page.rows).items())),
        "questions": qs, "questions_pass": passed,
        "questions_summary": {"asked": len(qs), "linked": sum(1 for q in qs if q["found"]),
                              "checklists_complete": len(complete), "cross_checks_clean": len(clean)},
        "planted": {"flagged_records": [_record_json(r) for r in flagged],
                    "flag_texts": dict(sorted(Counter(f for r in flagged for f in r.flags).items()))},
        "path_without_answer": [_record_json(r) for r in pwa],
        "identity": _identity(lib),
        "spotcheck": spotcheck_data(lib, sample_id, previous=previous_spotcheck, now=now),
        "walkthrough": {k: {"ticked": a, "total": b} for k, (a, b) in walkthrough_progress(walkthrough).items()},
    }


def render(d: Mapping[str, Any]) -> str:
    qsum = d["questions_summary"]
    out = ["# Stage 5 acceptance", "",
           f"Generated {d['generated_at']} by `seedgraph-v3 records acceptance-report`. Real questions and marks: "
           "this report stays under the v3 home.", "",
           f"Records in the library: {d['records_total']} ({d['by_state']}); by client {d['by_client']}.", "",
           "## 1. The five real questions", "",
           f"- asked {qsum['asked']}; linked to a record {qsum['linked']}; checklists complete "
           f"{qsum['checklists_complete']}; cross-checks clean {qsum['cross_checks_clean']}",
           "- **pass**: " + ("PASS" if d["questions_pass"] else "not yet")
           + " (five complete checklists, a clean cross-check for every Claude Code question, one asked from Codex)",
           "", "| question | client | helpers | record | state | checklist | cross-check |", "|---|---|---|---|---|---|---|"]
    for q in d["questions"]:
        rec, cl, cc = q["record"], q["checklist"], q["cross_check"]
        state = rec["state"] if rec else ("record not found" if q["record_id"] else "no record linked")
        check = "n/a (Codex)" if q["client"] == "codex" and not cc else (
            "not run" if not cc else ("clean" if cc["ok"] else f"FAIL: {len(cc['missing'])} missing, "
                                                              f"{len(cc['extra'])} extra, {len(cc['order'])} order"))
        out.append(f"| {q['id']} | {q['client']}{'' if q['client_matches'] or not rec else ' (record says ' + str(rec['client']) + ')'} "
                   f"| {'yes' if q['helpers'] else ''} | {q['record_id'] or ''} | {state} | "
                   f"{cl['ticked']}/{cl['total']} | {check} |")
    if not d["questions"]:
        out.append("| (no questions.yaml yet) | | | | | | |")
    p = d["planted"]
    out += ["", "## 2. Planted unread citation", "",
            "- automated: `tests/acceptance/stage5/test_planted_citation.py` (opt-in; run it against this home)",
            f"- records in the library carrying a citation flag: {len(p['flagged_records'])}; flags: {p['flag_texts'] or 'none'}",
            "- live variant: ask the agent to cite a named span without opening it, then look for its flag on the record"]
    pw = d["path_without_answer"]
    out += ["", "## 3. Path without answer", "",
            "- automated: `tests/acceptance/stage5/test_path_without_answer.py`",
            f"- records showing \"path without answer\" now: {len(pw)}"]
    out += [f"  - {r['record_id']} ({r['client'] or 'unknown'}, {r['counts'].get('searches', 0)} searches, "
            f"last event {r['last_event_at']})" for r in pw[:20]]
    idn = d["identity"]
    out += ["", "## 4. Session identity probe", "",
            "- automated: `tests/acceptance/stage5/test_identity_probe.py`",
            f"- clients seen in the handshake: {idn['clients'] or 'none'}",
            f"- `_meta` keys seen on requests (values are never stored): {idn['meta_keys_seen'] or 'none'}",
            f"- streams by kind: {idn['sessions_by_kind'] or 'none'}; helper streams: {len(idn['helpers'])}",
            f"- named streams used from more than one connection: {idn['named_on_several_connections'] or 'none'}",
            "", "| connection | client | started | closed | `_meta` keys |", "|---|---|---|---|---|"]
    out += [f"| {c['connection_id']} | {c['client'] or ''} | {c['started_at']} | {c['closed_at'] or 'open or crashed'} "
            f"| {', '.join(c['meta_keys']) or 'none'} |" for c in idn["connections"]]
    sc = d["spotcheck"]
    out += ["", "## 5. Spot-check", ""]
    if sc is None:
        out.append("- no spot-check sample yet")
    else:
        fig = sc["figure"]
        out += [f"- sample `{sc['sample_id']}` (seed {sc['seed']}): marked {sc['marked']} of {sc['items']}; "
                f"right {fig['right']}, partly right {fig['partly']}, wrong {fig['wrong']}",
                f"- fixed through \"fix it now\": {len(sc['fixed'])}; figure unchanged after a fix: "
                + {None: "not shown yet", True: "yes", False: "NO"}[sc["unchanged_after_fix"]],
                f"- overall: {sc['accuracy']['overall']['summary']}", f"- details: `{SPOTCHECK_MD}`, `{SPOTCHECK_JSON}`"]
    out += ["", "## 6. Walk-through and live variants", ""]
    wt = d["walkthrough"]
    out += ([f"- {k}: {v['ticked']} of {v['total']} ticked" for k, v in wt.items()] if wt
            else [f"- `{WALKTHROUGH_FILE}` not written yet"])
    return "\n".join(out) + "\n"


__all__ = ["AcceptanceError", "CrossCheck", "LOGGED_TOOLS", "Question", "Transcript", "TranscriptCall",
           "check_transcript", "cross_check", "gather", "load_events", "load_questions", "logged_calls", "parse_ticks",
           "read_transcript", "record_window", "render", "render_cross_check", "render_spotcheck",
           "save_cross_checks", "spotcheck_data", "stage_dir", "sync_checklist", "sync_walkthrough"]
