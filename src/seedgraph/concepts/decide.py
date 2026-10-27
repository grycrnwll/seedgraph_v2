"""Concept decisions from agent answers (stage 4 design §2.2-§2.4, §3.4-§3.5).

- :func:`validate` (pure over a frozen package manifest and a :class:`Guard` read from the
  library at check time): every item answered once; the verdict allowed for the item
  kind; refs belong to the item and to the right side; a link has a note and one quote
  per side; a split covers every listed claim exactly once with at least two glossed
  senses; a merge or link on an ambiguous label cites a ref from each side; the
  guardrails (type fence, never-merge / never-connect, transitively over the current
  groups; stoplist; closure of the name). A failure names the item, the reason and the
  answers still allowed.
- :func:`classify`: applied or queued (the "genuinely unclear" rules, in this order):
  a save-time veto; ``unsure``; ``confidence = low``; an ambiguous label without
  ``high`` confidence; **reach** (a merge whose two concepts both have at least
  :data:`REACH_MIN` papers; links and splits are never held back for reach);
  **disagreement** with an existing agent decision between the same two **concepts**
  (any member pair, including decisions saved earlier in the same package), or a link /
  unrelated verdict on two labels that are one concept by now. The same verdict between
  two concepts already decided through other members is dropped (``already_decided``).
- :func:`save_package`: in the protocol's save transaction, writes ``concept_decisions``
  / ``concept_senses`` / ``sense_assignments`` (``decided_by = 'agent'``, batch, packet,
  model), classifies each item, drops pair verdicts on the default sense of a label split
  since the package was frozen (``label_split``), drops proposals a researcher decision already settles
  (with feedback; never queued), refuses items whose cited claims are no longer live,
  invalidates agent decisions on a label's default sense when the label is split, and
  refreshes the derived tables once at the end (applied merges are joined in memory
  meanwhile, so the write transaction stays short).
- :func:`audit`: the guardrail checks of the acceptance plan (§11.4), for ``concepts check``.

**Storage conventions.** ``unsure`` has no decision kind: it is stored as ``kind =
'unrelated'``, ``state = 'queued'`` with ``veto_reason = 'unsure'``, so the pair is not
judged again until the researcher settles it. A queued row's ``veto_reason`` holds the
queue reason (``unsure``, ``low_confidence``, ``ambiguous``, ``reach``, ``disagreement``,
``save_veto:<reason>``). A split writes ``concept_senses`` rows ``s1..sn`` (``confirmed``
when applied, ``queued`` when unclear) and one ``sense_assignments`` row per cited claim;
``same_sense`` writes ``s0`` as ``confirmed`` (or ``queued``).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from . import guardrails as g
from . import refresh as rf
from .model import DEFAULT_SENSE, SenseKey, decision_id, label_key, ordered_pair

REACH_MIN = 5
NOTE_MAX = 200
WHY_MAX = 200
GLOSS_MAX = 120
NAME_WORDS_MAX = 6
DESCRIPTION_MAX = 200
VERDICTS = ("merge", "link", "split", "unrelated", "same_sense", "unsure")
PAIR_VERDICTS = ("merge", "link", "split", "unrelated", "unsure")
CHECK_VERDICTS = ("same_sense", "split", "unsure")
CONFIDENCES = ("high", "medium", "low")
LIVE_STATES = ("applied", "pending_apply", "queued")
REASON_TEXT = {
    "type_fence": "different families ({fa} vs {fb})",
    "never_merge": "a never-merge pair",
    "never_connect": "a never-connect pair",
    "stoplist": "stoplist (the labels carry conflicting version, size or variant words)",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- library state at check / save time -----------------------------------------------------------

def _key(side: Mapping[str, Any]) -> SenseKey:
    return SenseKey(side["family"], side["label_key"], side["sense_id"] or DEFAULT_SENSE)


@dataclass
class Guard:
    """The library state a check or save judges against: current groups (derived
    tables), vetoes, the effective ambiguous list, concept paper counts and the stored
    decisions."""

    members: dict[SenseKey, str]
    groups: dict[str, frozenset[SenseKey]]
    papers: dict[str, int]
    vetoes: g.VetoIndex
    ambiguous: g.AmbiguousList
    decisions: list[dict[str, Any]]
    senses: list[dict[str, Any]]

    @classmethod
    def load(cls, conn: sqlite3.Connection, graph: str | None = None) -> Guard:
        members: dict[SenseKey, str] = {}
        groups: dict[str, set[SenseKey]] = defaultdict(set)
        for cid, fam, key, sense in conn.execute("SELECT concept_id, family, label_key, sense_id FROM concept_members"):
            k = SenseKey(fam, key, sense)
            members[k] = cid
            groups[cid].add(k)
        papers = {r[0]: r[1] for r in conn.execute("SELECT concept_id, paper_count FROM concepts")}
        cols = rf.DECISION_COLS + ("confidence", "batch_id")
        decisions = rf._rows(conn, "concept_decisions", cols, "decision_id")  # noqa: SLF001
        senses = rf._rows(conn, "concept_senses", ("family", "label_key", "sense_id", "decided_by", "state"),  # noqa: SLF001
                          "family, label_key, sense_id")
        vetoes = g.VetoIndex(
            g.Veto(d["kind"], d["label_a"], d["label_b"], d["family"], d["family_b"] or d["family"], d["sense_a"],
                   d["sense_b"], d["decision_id"])
            for d in decisions if d["kind"] in ("never_merge", "never_connect")
            and d["state"] in ("applied", "pending_apply"))
        return cls(members, {c: frozenset(v) for c, v in groups.items()}, papers, vetoes,
                   g.effective_ambiguous(conn, graph), decisions, senses)

    def join(self, a: SenseKey, b: SenseKey) -> None:
        """Record an applied merge in memory, so later items of the same package are judged
        against the reshaped groups without a refresh inside the save transaction."""
        ga, gb = self.group(a), self.group(b)
        cid = self.members.get(a) or self.members.get(b) or f"pending:{a}"
        merged = ga | gb
        self.papers[cid] = self.concept_papers(a) + self.concept_papers(b)
        for k in merged:
            self.members[k] = cid
        self.groups[cid] = frozenset(merged)

    def group(self, key: SenseKey) -> frozenset[SenseKey]:
        cid = self.members.get(key)
        return self.groups.get(cid, frozenset({key})) if cid else frozenset({key})

    def concept_papers(self, key: SenseKey) -> int:
        cid = self.members.get(key)
        return self.papers.get(cid, 0) if cid else 0

    def merge_veto(self, a: SenseKey, b: SenseKey) -> str | None:
        if a.family != b.family:
            return "type_fence"
        ga, gb = self.group(a), self.group(b)
        if ga == gb:
            return None
        return g.merge_veto(ga, gb, self.vetoes)

    def link_veto(self, a: SenseKey, b: SenseKey) -> str | None:
        return g.link_veto(self.group(a), self.group(b), self.vetoes)

    def veto_text(self, why: str, a: SenseKey, b: SenseKey, label_a: str, label_b: str, *,
                  verdict: str = "merge") -> str:
        """The refusal naming what triggered a veto: the stoplist word on each side, both
        families, or the never-merge / never-connect pair (through other members, if so)."""
        if why == "type_fence":
            return (f"different families: {label_a!r} is {_article(a.family)} {a.family}, {label_b!r} is "
                    f"{_article(b.family)} {b.family}, and merges stay within one family")
        ga, gb = self.group(a), self.group(b)
        if why in ("never_merge", "never_connect"):
            v = self.vetoes.between(ga, gb, kinds=(why,))
            if v is None:
                return _reason(why, a, b)
            text = f"{v.label_a!r} and {v.label_b!r} are a {why.replace('_', '-')} pair"
            if {v.label_a, v.label_b} != {a.label_key, b.label_key}:
                text += (", and this merge would put them in one concept" if verdict == "merge"
                         else ", and this link would connect their concepts")
            return text
        if why == "stoplist":
            wa, wb = g.stoplist_triggers([k.label_key for k in ga], [k.label_key for k in gb])
            if wa or wb:
                kind = g.stoplist_kind(wa + wb)
                return (f"the labels differ in {_article(kind)} {kind} word: {_has(label_a, a, wa)}, "
                        f"{_has(label_b, b, wb)}")
        return _reason(why, a, b)

    def is_ambiguous(self, key: SenseKey) -> bool:
        return self.ambiguous.is_ambiguous(key.label_key)

    def pair_rows(self, a: SenseKey, b: SenseKey, *, by: str | None = None) -> list[dict[str, Any]]:
        """Live decisions on the pair (NULL family / sense match any)."""
        return [d for d in self.decisions if d["state"] in LIVE_STATES and not (by and d["decided_by"] != by)
                and _row_matches(d, a, b)]

    def add_decision(self, row: dict[str, Any]) -> None:
        """Record a decision saved in this package, so later items see it."""
        self.decisions.append(row)
        index = self.__dict__.get("_by_label")
        if index is not None:
            index[row["label_a"]].append(row)
            if row["label_b"] != row["label_a"]:
                index[row["label_b"]].append(row)

    def concept_rows(self, a: SenseKey, b: SenseKey, *, by: str = "agent",
                     kinds: tuple[str, ...] = ("merge", "link", "unrelated")) -> list[dict[str, Any]]:
        """Live decisions (of ``kinds``, by ``by``) between **any** member of ``a``'s concept
        and any member of ``b``'s concept: one question per concept pair, however the
        groups were reshaped since the package was frozen."""
        index = self.__dict__.get("_by_label")
        if index is None:
            index = defaultdict(list)
            for d in self.decisions:
                index[d["label_a"]].append(d)
                if d["label_b"] != d["label_a"]:
                    index[d["label_b"]].append(d)
            self.__dict__["_by_label"] = index
        ga, gb = self.group(a), self.group(b)
        b_labels = {k.label_key for k in gb}
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for x in sorted(ga):
            for d in index.get(x.label_key, ()):
                if d["decision_id"] in seen or d["kind"] not in kinds or d["decided_by"] != by \
                        or d["state"] not in LIVE_STATES:
                    continue
                if not {d["label_a"], d["label_b"]} & b_labels:
                    continue
                if any(_row_matches(d, x, y) for y in gb):
                    seen.add(d["decision_id"])
                    out.append(d)
        return out

    def is_split(self, key: SenseKey) -> bool:
        """The label has confirmed senses other than the default (any maker)."""
        return any(s["family"] == key.family and s["label_key"] == key.label_key and s["sense_id"] != DEFAULT_SENSE
                   and s["state"] == "confirmed" for s in self.senses)

    def researcher_senses(self, family: str, key: str) -> list[dict[str, Any]]:
        return [s for s in self.senses if s["family"] == family and s["label_key"] == key
                and s["decided_by"] == "researcher" and s["state"] in ("default", "confirmed", "queued")]


def _row_matches(d: Mapping[str, Any], a: SenseKey, b: SenseKey) -> bool:
    """A decision row is about the pair (either order; NULL family / sense match any)."""
    fa, fb = d["family"], d["family_b"] or d["family"]
    for x, y in ((a, b), (b, a)):
        if (x.label_key, y.label_key) != (d["label_a"], d["label_b"]):
            continue
        if (fa and x.family != fa) or (fb and y.family != fb):
            continue
        if (d["sense_a"] is None or d["sense_a"] == x.sense_id) and \
                (d["sense_b"] is None or d["sense_b"] == y.sense_id):
            return True
    return False


# --- validation -----------------------------------------------------------------------------------

def _reason(why: str, a: SenseKey, b: SenseKey) -> str:
    return REASON_TEXT.get(why, why).format(fa=a.family, fb=b.family)


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _has(label: str, key: SenseKey, words: list[str]) -> str:
    if not words:
        return f"{label!r} has none"
    text = f"{label!r} has " + " and ".join(repr(w) for w in words)
    if not set(words) & set(g.normalize(key.label_key).split()):
        text += " (through another label of its concept)"
    return text


def _refs(side: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(side["refs"]) if side else {}


def same_label(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """Two item sides are one label: one family, and the labels equal after the label
    normalisation (:func:`seedgraph.concepts.model.label_key`), so trivial formatting
    (case, hyphens, punctuation, spacing) never blocks a split."""
    if a["family"] != b["family"]:
        return False
    keys_a = {label_key(a.get("label_key")), label_key(a.get("label"))} - {""}
    keys_b = {label_key(b.get("label_key")), label_key(b.get("label"))} - {""}
    return bool(keys_a & keys_b)


def _check_split(n: int, item: Mapping[str, Any], ans: Mapping[str, Any]) -> list[str]:
    errs = []
    senses = ans.get("senses")
    if not isinstance(senses, list) or len(senses) < 2:
        return [f"item {n}: split needs at least two senses, each with a gloss and its claims"]
    listed = list(_refs(item.get("a"))) + [r for r in _refs(item.get("b")) if r not in _refs(item.get("a"))]
    seen: Counter = Counter()
    for i, s in enumerate(senses, start=1):
        if not isinstance(s, Mapping):
            errs.append(f"item {n}: sense {i} must be an object with gloss and claims")
            continue
        gloss = s.get("gloss")
        if not isinstance(gloss, str) or not gloss.strip():
            errs.append(f"item {n}: sense {i} needs a gloss")
        elif len(gloss) > GLOSS_MAX:
            errs.append(f"item {n}: sense {i} gloss is longer than {GLOSS_MAX} characters")
        claims = s.get("claims")
        if not isinstance(claims, list) or not claims:
            errs.append(f"item {n}: sense {i} needs at least one claim ref")
            continue
        for ref in claims:
            if ref not in listed:
                errs.append(f"item {n}: sense {i} cites {ref!r}, which is not a ref of this item")
            seen[ref] += 1
    errs += [f"item {n}: {ref} is cited twice in the split" for ref, c in sorted(seen.items()) if c > 1]
    errs += [f"item {n}: {ref} is not assigned to a sense" for ref in listed if ref not in seen]
    return errs


def _check_item(n: int, item: Mapping[str, Any], ans: Mapping[str, Any], guard: Guard | None) -> list[str]:
    verdict = ans.get("verdict")
    if verdict not in VERDICTS:
        return [f"item {n}: unknown verdict {verdict!r}; use one of {', '.join(VERDICTS)}"]
    kind = item["kind"]
    allowed = CHECK_VERDICTS if kind == "sense_check" else PAIR_VERDICTS
    if verdict not in allowed:
        what = "a sense check" if kind == "sense_check" else "a pair"
        return [f"item {n}: verdict {verdict!r} is not allowed for {what}; use one of {', '.join(allowed)}"]
    errs = []
    conf = ans.get("confidence")
    if conf not in CONFIDENCES:
        errs.append(f"item {n}: confidence must be high, medium or low")
    why = ans.get("why")
    if why is not None and (not isinstance(why, str) or len(why) > WHY_MAX):
        errs.append(f"item {n}: why must be text of at most {WHY_MAX} characters")
    a_refs, b_refs = _refs(item.get("a")), _refs(item.get("b"))
    for fld, refs, side in (("quote_a", a_refs, "A"), ("quote_b", b_refs, "B")):
        ref = ans.get(fld)
        if ref is not None and ref not in refs:
            errs.append(f"item {n}: {fld} {ref!r} is not a ref of side {side}")
    if verdict == "link":
        note = ans.get("note")
        if not isinstance(note, str) or not note.strip():
            errs.append(f"item {n}: a link needs a note (the difference, or the history for drift)")
        elif len(note) > NOTE_MAX:
            errs.append(f"item {n}: note is longer than {NOTE_MAX} characters")
        if not ans.get("quote_a") or not ans.get("quote_b"):
            errs.append(f"item {n}: a link needs quote_a and quote_b (a ref from each side)")
    elif ans.get("note") is not None and isinstance(ans.get("note"), str) and len(ans["note"]) > NOTE_MAX:
        errs.append(f"item {n}: note is longer than {NOTE_MAX} characters")
    if verdict == "split":
        if kind == "pair":
            if not same_label(item["a"], item["b"]):
                errs.append(f"item {n}: split on a pair needs both sides to share one label; "
                            "answer merge, link, unrelated or unsure")
                return errs
        errs += _check_split(n, item, ans)
    if verdict == "merge":
        labels = [item["a"]["label"], item["b"]["label"], item["a"]["label_key"], item["b"]["label_key"]]
        name = ans.get("name")
        if not isinstance(name, str) or not g.closure_ok(name, labels):
            errs.append(f"item {n}: name must be one of the item's labels "
                        f"({item['a']['label']!r} or {item['b']['label']!r})")
    if errs or kind != "pair" or verdict not in ("merge", "link"):
        return errs
    a, b = _key(item["a"]), _key(item["b"])
    amb = (guard.is_ambiguous(a) or guard.is_ambiguous(b)) if guard else (item["a"]["ambiguous"] or item["b"]["ambiguous"])
    if amb and (not ans.get("quote_a") or not ans.get("quote_b")):
        errs.append(f"item {n}: {verdict} on an ambiguous label must compare claims: give quote_a and quote_b "
                    "(a ref from each side)")
    if guard is None:
        return errs
    if verdict == "merge":
        why_v = guard.merge_veto(a, b)
        if why_v:
            text = guard.veto_text(why_v, a, b, item["a"]["label"], item["b"]["label"])
            errs.append(f"item {n}: merge vetoed: {text}; answer link, unrelated or unsure")
    else:
        why_v = guard.link_veto(a, b)
        if why_v:
            text = guard.veto_text(why_v, a, b, item["a"]["label"], item["b"]["label"], verdict="link")
            errs.append(f"item {n}: link vetoed: {text}; answer unrelated or unsure")
    return errs


def _answers(manifest: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[dict[int, dict], list[str]]:
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        return {}, ['the answer must be {"items": [...]}, one entry per item']
    numbers = {it["item"] for it in manifest["items"]}
    errs: list[str] = []
    out: dict[int, dict] = {}
    for ans in items:
        if not isinstance(ans, Mapping) or not isinstance(ans.get("item"), int):
            errs.append("every answer needs an integer item number")
            continue
        n = ans["item"]
        if n not in numbers:
            errs.append(f"item {n}: there is no such item in this package")
        elif n in out:
            errs.append(f"item {n}: answered twice")
        else:
            out[n] = dict(ans)
    errs += [f"item {n}: not answered" for n in sorted(numbers) if n not in out]
    return out, errs


def validate(manifest: Mapping[str, Any], payload: Mapping[str, Any], guard: Guard | None = None) -> list[str]:
    """Every error of an answer to one package (``[]`` = valid); see the module doc."""
    answers, errs = _answers(manifest, payload)
    for item in manifest["items"]:
        ans = answers.get(item["item"])
        if ans is None:
            continue
        if item["kind"] == "literature":
            errs += _check_literature(item["item"], ans)
        else:
            errs += _check_item(item["item"], item, ans, guard)
    return errs


def warnings(manifest: Mapping[str, Any], payload: Mapping[str, Any], guard: Guard) -> list[dict[str, Any]]:
    """Non-blocking notes: proposals a researcher decision already settles (dropped at save)."""
    answers, _ = _answers(manifest, payload)
    out = []
    for item in manifest["items"]:
        ans = answers.get(item["item"])
        if ans is None or item["kind"] == "literature":
            continue
        if _researcher_conflict(item, ans, guard):
            out.append({"item": item["item"], "reason": "researcher_decided",
                        "message": "a researcher decision already settles this item; it will be dropped at save"})
    return out


def _check_literature(n: int, ans: Mapping[str, Any]) -> list[str]:
    errs = []
    name = ans.get("name")
    if not isinstance(name, str) or not name.strip():
        errs.append(f"item {n}: a literature needs a name")
    elif len(name.split()) > NAME_WORDS_MAX:
        errs.append(f"item {n}: name has more than {NAME_WORDS_MAX} words")
    desc = ans.get("description")
    if desc is not None and (not isinstance(desc, str) or "\n" in desc or len(desc) > DESCRIPTION_MAX):
        errs.append(f"item {n}: description must be one line of at most {DESCRIPTION_MAX} characters")
    return errs


# --- classification -------------------------------------------------------------------------------

def classify(verdict: str, confidence: str | None, *, ambiguous: bool = False, papers_a: int = 0,
             papers_b: int = 0, disagreement: bool = False, save_veto: str | None = None) -> tuple[str, str | None]:
    """``('applied', None)`` or ``('queued', reason)`` (the unclear rules, module doc)."""
    if save_veto:
        return "queued", f"save_veto:{save_veto}"
    if verdict == "unsure":
        return "queued", "unsure"
    if confidence == "low":
        return "queued", "low_confidence"
    if ambiguous and confidence != "high":
        return "queued", "ambiguous"
    if verdict == "merge" and papers_a >= REACH_MIN and papers_b >= REACH_MIN:
        return "queued", "reach"
    if disagreement:
        return "queued", "disagreement"
    return "applied", None


# --- save -----------------------------------------------------------------------------------------

@dataclass
class SaveReport:
    applied: int = 0
    queued: int = 0
    vetoed: int = 0          # save-time vetoes (queued)
    unrelated: int = 0
    dropped: int = 0
    by_verdict: dict[str, int] = field(default_factory=dict)
    queue_reasons: dict[str, int] = field(default_factory=dict)
    feedback: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"applied": self.applied, "queued": self.queued, "vetoed": self.vetoed, "unrelated": self.unrelated,
                "dropped": self.dropped, "by_verdict": dict(sorted(self.by_verdict.items())),
                "queue_reasons": dict(sorted(self.queue_reasons.items())), "feedback": self.feedback}


_CONTRADICTS = {
    "merge": {"never_merge", "never_connect", "link", "unrelated", "merge"},
    "link": {"never_connect", "merge", "unrelated", "link"},
    "unrelated": {"merge", "link", "unrelated", "never_connect"},
    "unsure": {"merge", "link", "unrelated", "never_connect"},
    "split": {"merge", "link", "unrelated", "never_connect"},
}


def _researcher_conflict(item: Mapping[str, Any], ans: Mapping[str, Any], guard: Guard) -> str | None:
    verdict = ans.get("verdict")
    if item["kind"] == "sense_check" or verdict in ("split", "same_sense"):
        a = item["a"]
        rows = guard.researcher_senses(a["family"], a["label_key"])
        return f"the researcher already decided the senses of {a['label']!r}" if rows else None
    a, b = _key(item["a"]), _key(item["b"])
    for d in guard.pair_rows(a, b, by="researcher"):
        if d["kind"] in _CONTRADICTS.get(verdict, set()):
            return f"the researcher decided {d['kind']} for this pair"
    return None


def _live(conn: sqlite3.Connection, ref: Mapping[str, Any]) -> bool:
    if ref.get("span_id"):
        hit = conn.execute("SELECT 1 FROM current_claims WHERE paper_id = ? AND span_id = ? LIMIT 1",
                           (ref["paper_id"], ref["span_id"])).fetchone()
        if hit:
            return True
    return conn.execute("SELECT 1 FROM current_claims WHERE claim_id = ? LIMIT 1", (ref["claim_id"],)).fetchone() \
        is not None


def _quote_json(ref: Mapping[str, Any] | None) -> str | None:
    if not ref:
        return None
    return json.dumps({"paper_id": ref["paper_id"], "span_id": ref["span_id"],
                       "claim_fingerprint": ref["claim_fingerprint"]}, ensure_ascii=False, sort_keys=True)


@dataclass
class _Ctx:
    conn: sqlite3.Connection
    guard: Guard
    batch_id: str | None
    packet_id: str | None
    model: str | None
    now: str
    report: SaveReport
    touched_splits: list[tuple[str, str]] = field(default_factory=list)


def _insert_decision(c: _Ctx, kind: str, a: SenseKey, b: SenseKey, *, state: str, reason: str | None,
                     ans: Mapping[str, Any], qa: Mapping | None, qb: Mapping | None) -> str:
    sa = None if a.sense_id == DEFAULT_SENSE else a.sense_id
    sb = None if b.sense_id == DEFAULT_SENSE else b.sense_id
    la, sa_, fa, lb, sb_, fb = ordered_pair(a.label_key, sa, b.label_key, sb, a.family, b.family)
    if (la, sa_, fa) != (a.label_key, sa, a.family):
        qa, qb = qb, qa
    family, family_b = fa, (fb if fb != fa else None)
    did = decision_id(kind, la, lb, family=family, family_b=family_b, sense_a=sa_, sense_b=sb_)
    note = ans.get("note") if kind == "link" else (ans.get("why") or None)
    existing = c.conn.execute("SELECT decided_by, state FROM concept_decisions WHERE decision_id = ?",
                              (did,)).fetchone()
    if existing is not None and not (existing[0] == "agent" and existing[1] in ("needs_rejudge", "undone")):
        c.report.dropped += 1
        c.report.feedback.append({"item": ans["item"], "reason": "already_decided",
                                  "message": f"{kind} is already stored for this pair"})
        return did
    c.conn.execute(
        "INSERT OR REPLACE INTO concept_decisions (decision_id, kind, family, family_b, label_a, sense_a, label_b, "
        "sense_b, name, decided_by, state, note, quote_a_json, quote_b_json, confidence, veto_reason, batch_id, "
        "packet_id, model, origin, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'agent', ?, ?, ?, ?, ?, ?, ?, ?, ?, "
        "'library', ?)",
        (did, kind, family, family_b, la, sa_, lb, sb_, ans.get("name") if kind == "merge" else None, state, note,
         _quote_json(qa), _quote_json(qb), ans.get("confidence"), reason, c.batch_id, c.packet_id, c.model, c.now))
    c.guard.add_decision({"decision_id": did, "kind": kind, "family": family, "family_b": family_b, "label_a": la,
                          "sense_a": sa_, "label_b": lb, "sense_b": sb_, "decided_by": "agent", "state": state,
                          "veto_reason": reason, "confidence": ans.get("confidence"), "batch_id": c.batch_id})
    return did


def _count(c: _Ctx, verdict: str, state: str, reason: str | None) -> None:
    r = c.report
    r.by_verdict[verdict] = r.by_verdict.get(verdict, 0) + 1
    if state == "applied":
        r.applied += 1
        if verdict == "unrelated":
            r.unrelated += 1
    else:
        r.queued += 1
        r.queue_reasons[(reason or "").split(":")[0]] = r.queue_reasons.get((reason or "").split(":")[0], 0) + 1
        if reason and reason.startswith("save_veto"):
            r.vetoed += 1


def _save_split(c: _Ctx, item: Mapping[str, Any], ans: Mapping[str, Any], state: str) -> None:
    a = item["a"]
    family, key = a["family"], label_key(a["label_key"])
    refs = {**_refs(item.get("a")), **_refs(item.get("b"))}
    sense_state = "confirmed" if state == "applied" else "queued"
    for i, s in enumerate(ans["senses"], start=1):
        sid = f"s{i}"
        c.conn.execute(
            "INSERT OR REPLACE INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, "
            "decision_ref, origin, decided_at) VALUES (?, ?, ?, ?, 'agent', ?, ?, 'library', ?)",
            (family, key, sid, s["gloss"].strip(), sense_state, c.batch_id, c.now))
        for ref in s["claims"]:
            r = refs[ref]
            c.conn.execute(
                "INSERT OR REPLACE INTO sense_assignments (family, label_key, paper_id, claim_fingerprint, sense_id, "
                "decided_by, decided_at) VALUES (?, ?, ?, ?, ?, 'agent', ?)",
                (family, key, r["paper_id"], r["claim_fingerprint"], sid, c.now))
    if state == "applied":
        c.touched_splits.append((family, key))
        c.conn.execute("UPDATE concept_senses SET state = 'undone' WHERE family = ? AND label_key = ? "
                       "AND sense_id = ? AND decided_by = 'agent'", (family, key, DEFAULT_SENSE))
        c.conn.execute(
            "UPDATE concept_decisions SET state = 'needs_rejudge' WHERE decided_by = 'agent' AND state = 'applied' "
            "AND kind IN ('merge', 'link', 'unrelated') AND (family IS NULL OR family = ? OR family_b = ?) AND "
            "((label_a = ? AND sense_a IS NULL) OR (label_b = ? AND sense_b IS NULL))", (family, family, key, key))


def _save_item(c: _Ctx, item: Mapping[str, Any], ans: Mapping[str, Any], graph: str | None) -> None:
    n, verdict, conf = item["item"], ans["verdict"], ans.get("confidence")
    conflict = _researcher_conflict(item, ans, c.guard)
    if conflict:
        c.report.dropped += 1
        c.report.feedback.append({"item": n, "reason": "researcher_decided", "message": conflict})
        return
    refs = {**_refs(item.get("a")), **_refs(item.get("b"))}
    cited = [ans.get("quote_a"), ans.get("quote_b")] + [r for s in (ans.get("senses") or []) for r in s["claims"]] \
        if verdict in ("link", "split", "merge") else []
    stale = [r for r in cited if r and not _live(c.conn, refs[r])]
    if stale:
        c.report.dropped += 1
        c.report.feedback.append({"item": n, "reason": "stale_ref",
                                  "message": f"cited claims are no longer live: {', '.join(stale)}"})
        return
    if item["kind"] == "sense_check" or verdict == "split":
        a = _key(item["a"])
        state, reason = classify("unsure" if verdict == "unsure" else verdict, conf,
                                 ambiguous=c.guard.is_ambiguous(a))
        if verdict == "split":
            _save_split(c, item, ans, state)
        elif verdict == "same_sense":
            c.conn.execute(
                "INSERT OR REPLACE INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, "
                "decision_ref, origin, decided_at) VALUES (?, ?, ?, NULL, 'agent', ?, ?, 'library', ?)",
                (a.family, a.label_key, DEFAULT_SENSE, "confirmed" if state == "applied" else "queued",
                 c.batch_id, c.now))
        else:  # unsure on a sense check: hold the label for the researcher
            c.conn.execute(
                "INSERT OR REPLACE INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, "
                "decision_ref, origin, decided_at) VALUES (?, ?, ?, NULL, 'agent', 'queued', ?, 'library', ?)",
                (a.family, a.label_key, DEFAULT_SENSE, c.batch_id, c.now))
        _count(c, verdict, state, reason)
        return
    a, b = _key(item["a"]), _key(item["b"])
    split = next((k for k in (a, b) if k.sense_id == DEFAULT_SENSE and
                  ((k.family, k.label_key) in c.touched_splits or c.guard.is_split(k))), None)
    if split is not None:  # split after the package was frozen: its default sense is gone
        c.report.dropped += 1
        c.report.feedback.append({"item": n, "reason": "label_split",
                                  "message": f"{split.label_key!r} was split into senses since this package was "
                                             "made; its pairs are judged again per sense"})
        return
    kind = {"merge": "merge", "link": "link", "unrelated": "unrelated", "unsure": "unrelated"}[verdict]
    save_veto = c.guard.merge_veto(a, b) if verdict == "merge" else \
        (c.guard.link_veto(a, b) if verdict == "link" else None)
    same_concept = c.guard.group(a) == c.guard.group(b)
    agent_rows = c.guard.concept_rows(a, b)
    exact = {d["decision_id"] for d in c.guard.pair_rows(a, b, by="agent")}
    if not same_concept and any(d["kind"] == kind and d["decision_id"] not in exact for d in agent_rows):
        c.report.dropped += 1
        c.report.feedback.append({"item": n, "reason": "already_decided",
                                  "message": f"{kind} is already stored between these two concepts "
                                             "(through other member labels)"})
        return
    # a different verdict between the same two concepts (any member pair), or a link /
    # unrelated inside one concept (merged since the package was frozen): the researcher decides
    disagreement = any(d["kind"] != kind for d in agent_rows) or (same_concept and kind != "merge")
    state, reason = classify(verdict, conf, ambiguous=c.guard.is_ambiguous(a) or c.guard.is_ambiguous(b),
                             papers_a=c.guard.concept_papers(a), papers_b=c.guard.concept_papers(b),
                             disagreement=disagreement, save_veto=save_veto)
    qa = refs.get(ans.get("quote_a")) if ans.get("quote_a") else None
    qb = refs.get(ans.get("quote_b")) if ans.get("quote_b") else None
    before = c.report.dropped
    _insert_decision(c, kind, a, b, state=state, reason=reason, ans={**ans, "item": n}, qa=qa, qb=qb)
    if c.report.dropped == before:
        _count(c, verdict, state, reason)
        if kind == "merge" and state == "applied":
            c.guard.join(a, b)


def save_package(conn: sqlite3.Connection, manifest: Mapping[str, Any], payload: Mapping[str, Any], *,
                 batch_id: str | None = None, packet_id: str | None = None, model: str | None = None,
                 graph: str | None = None, now: str | None = None, refresh: bool = True) -> SaveReport:
    """Store one accepted package's answers (caller's transaction); see the module doc."""
    c = _Ctx(conn, Guard.load(conn, graph), batch_id, packet_id, model, now or _now(), SaveReport())
    answers, _ = _answers(manifest, payload)
    for item in manifest["items"]:
        ans = answers.get(item["item"])
        if ans is not None:
            _save_item(c, item, ans, graph)
    if refresh:
        _refresh(conn, c.now)
    return c.report


def _refresh(conn: sqlite3.Connection, now: str) -> None:
    rf.apply_plan(conn, rf.plan(conn, now=now))


# --- audit (concepts check) -----------------------------------------------------------------------

@dataclass(frozen=True)
class AuditResult:
    checks: dict[str, int]          # check name -> violations

    @property
    def ok(self) -> bool:
        return not any(self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "violations": dict(self.checks)}


def audit(conn: sqlite3.Connection) -> AuditResult:
    """The automated guardrail checks (§11.4) over the concepts a refresh would build now
    (in memory: works read-only and on a library whose derived tables are stale). Counts
    only, never labels."""
    plan = rf.plan(conn)
    groups: dict[str, set[SenseKey]] = defaultdict(set)
    for cid, fam, key, sense in plan.members:
        groups[cid].add(SenseKey(fam, key, sense))
    names = {r[0]: r[2] for r in plan.concepts}
    decisions = rf._rows(conn, "concept_decisions", rf.DECISION_COLS, "decision_id")  # noqa: SLF001
    vetoes = [d for d in decisions if d["kind"] in ("never_merge", "never_connect")
              and d["state"] in ("applied", "pending_apply")]
    seed = [tuple(sorted(p)) for p in g.SEED_NEVER_MERGE]
    sense_rows = rf._rows(conn, "concept_senses", ("family", "label_key", "sense_id", "state"),  # noqa: SLF001
                          "family, label_key")
    sense_decided = {(s["family"], s["label_key"]) for s in sense_rows if s["state"] in ("default", "confirmed")}
    ambiguous = g.effective_ambiguous(conn)
    merged_by: dict[tuple[str, str], set[str]] = defaultdict(set)
    for d in decisions + [dict(x) for x in plan.new_rule_decisions]:
        if d["kind"] == "merge" and d["state"] in ("applied", "pending_apply"):
            merged_by[(d["label_a"], d["label_b"])].add(d["decided_by"])
    checks = Counter({"spans_families": 0, "never_merge_inside": 0, "never_connect_inside": 0,
                      "name_outside_members": 0, "stoplist_conflict": 0, "ambiguous_merged_unchecked": 0,
                      "seed_pairs_together": 0})
    for cid, members in groups.items():
        labels = {m.label_key for m in members}
        if len({m.family for m in members}) > 1:
            checks["spans_families"] += 1
        for v in vetoes:
            if v["label_a"] in labels and v["label_b"] in labels:
                checks["never_merge_inside" if v["kind"] == "never_merge" else "never_connect_inside"] += 1
        if cid in names and not g.closure_ok(names[cid], labels):
            checks["name_outside_members"] += 1
        if len(labels) > 1:
            makers = set()
            for x in labels:
                for y in labels:
                    makers |= merged_by.get((x, y), set())
            if makers - {"researcher"} and g.discriminating_conflict(labels):
                checks["stoplist_conflict"] += 1
            for m in members:
                if ambiguous.is_ambiguous(m.label_key) and m.sense_id == DEFAULT_SENSE \
                        and (m.family, m.label_key) not in sense_decided:
                    others = labels - {m.label_key}
                    if any(merged_by.get(tuple(sorted((m.label_key, o)))) - {"researcher"} for o in others):
                        checks["ambiguous_merged_unchecked"] += 1
        for a_, b_ in seed:
            if a_ in labels and b_ in labels:
                checks["seed_pairs_together"] += 1
    return AuditResult(dict(checks))


__all__ = ["Guard", "REACH_MIN", "SaveReport", "audit", "classify", "save_package", "validate", "warnings"]
