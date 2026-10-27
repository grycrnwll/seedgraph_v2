"""The embeddings adoption test (stage 4 design §5.5): keyword vs meaning on frozen cases.

Pure Python (no numpy): a few dozen queries against every sense of the families
involved. Steps, each a function so the protocol is written **before** any result is seen:

1. :func:`gloss_pools` - for a sense probe whose gloss is not itself a library label, the
   claims that could carry it: supported claims whose label matches the probe label as a
   C1 ``head`` pattern (:func:`.guardrails.pattern_matches`), the partner label excluded.
2. :func:`assign_glosses` - each pool claim assigned to gloss ``a``, ``b`` or ``neither``
   by a classifier (the real run uses a local chat model, :func:`ollama_gloss_classifier`;
   every assignment is recorded in the protocol).
3. :func:`build_protocol` - the frozen claim sets. The ladder for a probe side: (a) the
   gloss's ``label_key`` has supported claims -> that label's claims; (b) else the gloss
   assignment over the head pool; (c) else the probe is malformed. A partner case's A set
   is every supported claim of label A; its targets are B plus its ``also`` labels (a hit
   on any counts, best rank). The two sides of a probe never overlap; the partner label
   is never in either side. ``partner_of_b`` makes sense b the query side.
4. :func:`evaluate` - both modes rank, for each query set, the **senses** of the
   families involved (never concepts, so no merge can fake or hide a hit), excluding the
   query's own senses (label A; for a probe every sense sharing a claim with either side)
   and, for a probe, adding the other side as one candidate.
   - keyword: the set's label words + top distinctive terms (:mod:`.search`), claim
     ``bm25`` aggregated to candidates by best score; a candidate with no hit is unranked;
   - meaning: the mean of the set's claim vectors (``search_document:`` contexts) against
     each candidate's mean vector, cosine.
5. :func:`mode_metrics`, :func:`pass_rule`, :func:`render_markdown`.

**Pass rule** (all three): (1) separation >= 0.8; (2) recall@10 >= keyword's and meaning
finds at least ``max(3, ceil(25% of cases))`` partners keyword misses (not in its top 10);
(3) no probe has the other side as the query's top-1 neighbour.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from . import guardrails as g
from . import search as cs
from . import vectors as vx
from .model import SenseKey, handle, label_key

if TYPE_CHECKING:
    import sqlite3

    from ..library.store import Library

TOP_K = 10
SEPARATION_MIN = 0.8
NEW_PARTNERS_MIN = 3
NEW_PARTNERS_SHARE = 0.25
OTHER_ITEM = "<other side>"
EMBED_FORMAT = ("search_document: <family>: <label>. <claim text>. <plain words, when present>. "
                "Quote: <top quote, when it says more than the claim text, first 400 chars>.")
RULES = {
    "separation_min": SEPARATION_MIN, "top_k": TOP_K, "new_partners": f"max({NEW_PARTNERS_MIN}, "
    f"ceil({NEW_PARTNERS_SHARE} x partner cases))", "over_merge": "other side is the query's top-1 neighbour",
    "ranking_level": "senses of the families involved (never concepts)",
    "also": "a hit on B or any 'also' label counts; best rank",
    "exclusions": "partner case: every sense of label A; probe: every sense sharing a claim with either side, "
                  "the other side added as one candidate",
    "control_cases": "included in the metrics",
    "keyword_query": f"label words + top {cs.DISTINCTIVE_TERMS} distinctive terms (tf x idf); bm25 with label "
                     f"weight {cs.LABEL_WEIGHT}; candidates scored by their best claim",
    "meaning_query": "normalised mean of the set's claim vectors vs each candidate's normalised mean; cosine",
}


# --- sets and the protocol ----------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimSet:
    claim_ids: tuple[str, ...]
    labels: tuple[str, ...]  # label keys of the claims
    how: str


@dataclass(frozen=True)
class PartnerCase:
    id: str
    kind: str
    a: ClaimSet
    targets: list[str]
    missing_targets: list[str]
    families: list[str]
    control: bool = False
    note: str | None = None


@dataclass(frozen=True)
class Probe:
    id: str
    label: str
    orientation: str  # 'a' or 'b': the query side
    query_gloss: str
    other_gloss: str
    query: ClaimSet
    other: ClaimSet
    partner: str
    families: list[str]


@dataclass
class Protocol:
    partners: list[PartnerCase]
    probes: list[Probe]
    malformed: list[dict[str, str]]
    assignments: dict[str, dict[str, str]] = field(default_factory=dict)
    cases_sha256: str | None = None
    cases_path: str | None = None
    created_at: str | None = None
    embed_format: str = EMBED_FORMAT
    rules: dict[str, Any] = field(default_factory=lambda: dict(RULES))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Protocol:
        def cset(x: Mapping[str, Any]) -> ClaimSet:
            return ClaimSet(tuple(x["claim_ids"]), tuple(x["labels"]), x["how"])

        partners = [PartnerCase(p["id"], p["kind"], cset(p["a"]), list(p["targets"]), list(p["missing_targets"]),
                                list(p["families"]), p.get("control", False), p.get("note"))
                    for p in d["partners"]]
        probes = [Probe(p["id"], p["label"], p["orientation"], p["query_gloss"], p["other_gloss"], cset(p["query"]),
                        cset(p["other"]), p["partner"], list(p["families"])) for p in d["probes"]]
        return cls(partners, probes, [dict(m) for m in d["malformed"]],
                   {k: dict(v) for k, v in (d.get("assignments") or {}).items()}, d.get("cases_sha256"),
                   d.get("cases_path"), d.get("created_at"), d.get("embed_format", EMBED_FORMAT),
                   dict(d.get("rules") or RULES))


def _label_claims(conn: sqlite3.Connection, key: str) -> list[tuple[str, str]]:
    return [(r[0], r[1]) for r in conn.execute(
        "SELECT claim_id, family FROM claim_concepts WHERE label_key = ? AND supports = 1 ORDER BY claim_id", (key,))]


def _claim_labels(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, tuple[str, str]]:
    ids = sorted(set(ids))
    out: dict[str, tuple[str, str]] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for r in conn.execute(f"SELECT claim_id, label_key, family FROM claim_concepts WHERE claim_id IN "
                              f"({', '.join('?' * len(chunk))})", chunk):
            out[r[0]] = (r[1], r[2])
    return out


def _needs_pool(conn: sqlite3.Connection, probe: Mapping[str, Any]) -> bool:
    return any(not _label_claims(conn, label_key(probe[s])) for s in ("sense_a", "sense_b"))


def _orientation(probe: Mapping[str, Any]) -> str:
    return "b" if probe.get("partner_of_b") and not probe.get("partner_of_a") else "a"


def gloss_pools(conn: sqlite3.Connection, cases: Mapping[str, Any]) -> dict[str, list[vx.ClaimContext]]:
    """For each probe with a gloss that is not a library label: the head-pattern pool."""
    out: dict[str, list[vx.ClaimContext]] = {}
    contexts = None
    for p in cases.get("sense_probes", []):
        if not _needs_pool(conn, p):
            continue
        if contexts is None:
            contexts = vx.claim_contexts(conn)
        head = label_key(p["label"])
        partner = label_key(p.get(f"partner_of_{_orientation(p)}") or "")
        out[p["id"]] = [c for c in contexts if c.label_key != partner and g.pattern_matches(c.label_key, head, "head")]
    return out


Classifier = Callable[[str, str, str, vx.ClaimContext], str]


def assign_glosses(pools: Mapping[str, list[vx.ClaimContext]], cases: Mapping[str, Any],
                   classify: Classifier) -> dict[str, dict[str, str]]:
    """``{probe_id: {claim_id: 'a' | 'b' | 'neither'}}``; ``classify(label, gloss_a, gloss_b, claim)``."""
    by_id = {p["id"]: p for p in cases.get("sense_probes", [])}
    out: dict[str, dict[str, str]] = {}
    for pid, pool in pools.items():
        p = by_id[pid]
        out[pid] = {c.claim_id: classify(p["label"], p["sense_a"], p["sense_b"], c) for c in pool}
    return out


def ollama_gloss_classifier(model: str, *, base_url: str | None = None) -> Classifier:
    """A local chat model as the gloss classifier (temperature 0, JSON answer)."""
    from ..llm.ollama import OllamaClient

    client = OllamaClient(base_url)
    schema = {"type": "object", "properties": {"sense": {"type": "string", "enum": ["A", "B", "neither"]}},
              "required": ["sense"]}
    system = ("You sort claims from economics and econometrics papers by the meaning of one word. "
              "Answer with JSON only.")

    def classify(label: str, gloss_a: str, gloss_b: str, claim: vx.ClaimContext) -> str:
        user = (f'The word "{label}" is used in two different senses.\n'
                f"Sense A: {gloss_a}\nSense B: {gloss_b}\n\n"
                f"Claim (from a paper's notes): {claim.text}\n\n"
                'Which sense does this claim use? Answer {"sense": "A"}, {"sense": "B"}, or {"sense": "neither"} '
                "if it fits neither sense clearly.")
        try:
            got = json.loads(client.complete_json(model, system, user, schema, max_tokens=20).text).get("sense")
        except (ValueError, AttributeError):
            return "neither"
        return {"A": "a", "B": "b"}.get(got, "neither")

    return classify


def build_protocol(conn: sqlite3.Connection, cases: Mapping[str, Any], *,
                   assignments: Mapping[str, Mapping[str, str]] | None = None,
                   cases_sha256: str | None = None, cases_path: str | None = None) -> Protocol:
    assignments = {k: dict(v) for k, v in (assignments or {}).items()}
    partners: list[PartnerCase] = []
    probes: list[Probe] = []
    malformed: list[dict[str, str]] = []

    for c in cases.get("partners", []):
        a_key = label_key(c["a"])
        a_claims = _label_claims(conn, a_key)
        if not a_claims:
            malformed.append({"id": c["id"], "reason": f"label '{c['a']}' has no supported claims"})
            continue
        targets = [label_key(x) for x in [c["b"], *c.get("also", [])]]
        found = {t: _label_claims(conn, t) for t in targets}
        missing = [t for t in targets if not found[t]]
        if len(missing) == len(targets):
            malformed.append({"id": c["id"], "reason": f"no target label ({', '.join(targets)}) has supported claims"})
            continue
        fams = sorted({f for _, f in a_claims} | {f for t in targets for _, f in found[t]})
        partners.append(PartnerCase(c["id"], c.get("kind", ""), ClaimSet(
            tuple(i for i, _ in a_claims), (a_key,), f"label '{a_key}'"), targets, missing, fams,
            bool(c.get("control", False)), c.get("note")))

    pools = None
    for p in cases.get("sense_probes", []):
        orient = _orientation(p)
        partner = label_key(p.get(f"partner_of_{orient}") or "")
        partner_claims = _label_claims(conn, partner)
        if not partner_claims:
            malformed.append({"id": p["id"], "reason": f"partner '{partner}' has no supported claims"})
            continue
        sides: dict[str, list[str]] = {}
        hows: dict[str, str] = {}
        bad = None
        for side in ("a", "b"):
            gloss = p[f"sense_{side}"]
            gk = label_key(gloss)
            own = [i for i, _ in _label_claims(conn, gk)] if gk != partner else []
            if own:
                sides[side], hows[side] = own, f"label '{gk}'"
                continue
            if p["id"] not in assignments:
                bad = f"sense_{side} '{gloss}': no gloss assignment for the head pool"
                break
            if pools is None:
                pools = gloss_pools(conn, cases)
            pool = pools.get(p["id"], [])
            chosen = [x.claim_id for x in pool if assignments[p["id"]].get(x.claim_id) == side]
            if not chosen:
                bad = f"sense_{side} '{gloss}': no claims match the gloss"
                break
            sides[side] = chosen
            hows[side] = (f"gloss assignment over the head pool '{label_key(p['label'])}' "
                          f"({len(chosen)} of {len(pool)} claims)")
        if bad:
            malformed.append({"id": p["id"], "reason": bad})
            continue
        overlap = set(sides["a"]) & set(sides["b"])
        if overlap:
            for side in ("a", "b"):
                sides[side] = [i for i in sides[side] if i not in overlap]
                hows[side] += f"; {len(overlap)} claims in both sides dropped"
        if not sides["a"] or not sides["b"]:
            malformed.append({"id": p["id"], "reason": "the two sides share every claim"})
            continue
        other = "b" if orient == "a" else "a"
        labels = _claim_labels(conn, sides["a"] + sides["b"])

        def cset(side: str) -> ClaimSet:
            ids = tuple(sorted(sides[side]))
            return ClaimSet(ids, tuple(sorted({labels[i][0] for i in ids})), hows[side])

        fams = sorted({labels[i][1] for i in sides["a"] + sides["b"]} | {f for _, f in partner_claims})
        probes.append(Probe(p["id"], p["label"], orient, p[f"sense_{orient}"], p[f"sense_{other}"], cset(orient),
                            cset(other), partner, fams))

    return Protocol(partners, probes, malformed, assignments, cases_sha256, cases_path,
                    datetime.now(timezone.utc).isoformat())


def cases_digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


# --- metrics ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeOutcome:
    partner_rank: int | None
    other_rank: int | None

    @property
    def separated(self) -> bool:
        return self.partner_rank is not None and (self.other_rank is None or self.partner_rank < self.other_rank)

    @property
    def over_merged(self) -> bool:
        return self.other_rank == 1


@dataclass(frozen=True)
class ModeMetrics:
    partner_ranks: dict[str, int | None]
    probes: dict[str, ProbeOutcome]
    recall_at_10: float
    mrr: float
    separation: float
    over_merged: list[str]


def mode_metrics(partner_ranks: Mapping[str, int | None], probes: Mapping[str, ProbeOutcome]) -> ModeMetrics:
    n, m = len(partner_ranks), len(probes)
    recall = sum(1 for r in partner_ranks.values() if r is not None and r <= TOP_K) / n if n else 0.0
    mrr = sum(1 / r for r in partner_ranks.values() if r) / n if n else 0.0
    sep = sum(1 for o in probes.values() if o.separated) / m if m else 0.0
    return ModeMetrics(dict(partner_ranks), dict(probes), recall, mrr, sep,
                       [k for k, o in probes.items() if o.over_merged])


@dataclass(frozen=True)
class Verdict:
    passed: bool
    checks: dict[str, bool]
    new_partners: list[str]
    needed_new: int


def _hit(r: int | None) -> bool:
    return r is not None and r <= TOP_K


def pass_rule(keyword: ModeMetrics, meaning: ModeMetrics) -> Verdict:
    needed = max(NEW_PARTNERS_MIN, math.ceil(NEW_PARTNERS_SHARE * len(meaning.partner_ranks)))
    new = sorted((k for k, r in meaning.partner_ranks.items() if _hit(r) and not _hit(keyword.partner_ranks.get(k))),
                 key=_case_order)
    checks = {
        "separation": meaning.separation >= SEPARATION_MIN,
        "recall": meaning.recall_at_10 >= keyword.recall_at_10,
        "new_partners": len(new) >= needed,
        "no_over_merge": not meaning.over_merged,
    }
    return Verdict(all(checks.values()), checks, new, needed)


def _case_order(cid: str) -> tuple:
    digits = "".join(ch for ch in cid if ch.isdigit())
    return (cid.rstrip("0123456789"), int(digits) if digits else 0, cid)


# --- ranking ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Item:
    id: str
    label_key: str | None  # None for the other side of a probe
    claim_ids: frozenset[str]


def _sense_items(conn: sqlite3.Connection) -> dict[str, list[_Item]]:
    """Every supported sense, grouped by family."""
    groups: dict[SenseKey, set[str]] = defaultdict(set)
    for cid, fam, key, sid in conn.execute(
            "SELECT claim_id, family, label_key, sense_id FROM claim_concepts WHERE supports = 1 "
            "AND sense_id IS NOT NULL"):
        groups[SenseKey(fam, key, sid)].add(cid)
    out: dict[str, list[_Item]] = defaultdict(list)
    for k in sorted(groups):
        out[k.family].append(_Item(handle(k), k.label_key, frozenset(groups[k])))
    return out


def _rank(scores: Mapping[str, float]) -> list[str]:
    return [i for i, _ in sorted(scores.items(), key=lambda t: (-t[1], t[0]))]


def _first(ranked: list[str], items: Mapping[str, _Item], wanted: Callable[[_Item], bool]) -> int | None:
    for pos, iid in enumerate(ranked, 1):
        if wanted(items[iid]):
            return pos
    return None


@dataclass
class EvalResult:
    protocol: Protocol
    modes: dict[str, ModeMetrics]
    tops: dict[str, dict[str, list[tuple[str, float]]]]
    verdict: Verdict | None
    embed_stats: vx.EmbedStats | None
    seconds: dict[str, float]


def evaluate(lib: Library, protocol: Protocol, *, embedder: vx.Embedder | None = None,
             progress: Callable[[int, int], None] | None = None) -> EvalResult:
    """Rank every case in keyword mode and, with an embedder, meaning mode."""
    conn = lib.conn
    by_family = _sense_items(conn)
    queries: list[tuple[str, ClaimSet, dict[str, _Item]]] = []
    for c in protocol.partners:
        items = {it.id: it for f in c.families for it in by_family.get(f, []) if it.label_key != c.a.labels[0]}
        queries.append((c.id, c.a, items))
    for p in protocol.probes:
        both = set(p.query.claim_ids) | set(p.other.claim_ids)
        items = {it.id: it for f in p.families for it in by_family.get(f, []) if not (it.claim_ids & both)}
        items[OTHER_ITEM] = _Item(OTHER_ITEM, None, frozenset(p.other.claim_ids))
        queries.append((p.id, p.query, items))

    seconds: dict[str, float] = {}
    rankings: dict[str, dict[str, list[tuple[str, float]]]] = {"keyword": {}}
    t0 = time.perf_counter()
    for cid, qset, items in queries:
        terms = cs.keyword_terms_for_claims(conn, qset.claim_ids, qset.labels)
        fams = sorted({f for c in protocol.partners + protocol.probes if c.id == cid for f in c.families})
        claim_scores = cs.keyword_claim_scores(conn, terms, families=fams)
        scores = {}
        for iid, it in items.items():
            hits = [claim_scores[x] for x in it.claim_ids if x in claim_scores]
            if hits:
                scores[iid] = max(hits)
        rankings["keyword"][cid] = [(i, scores[i]) for i in _rank(scores)]
    seconds["keyword"] = time.perf_counter() - t0

    stats = None
    if embedder is not None:
        fams_all = sorted({f for c in protocol.partners + protocol.probes for f in c.families})
        contexts = vx.claim_contexts(conn, families=fams_all)
        text_of = {c.claim_id: c.text for c in contexts}
        vecs, stats = vx.vectors_for(lib, list(text_of.values()), embedder, progress=progress)
        t1 = time.perf_counter()
        unit = {cid: vx.normalized(vecs[t]) for cid, t in text_of.items()}
        mean_cache: dict[frozenset[str], list[float]] = {}

        def mean_of(ids: frozenset[str]) -> list[float] | None:
            if ids not in mean_cache:
                vs = [unit[i] for i in ids if i in unit]
                mean_cache[ids] = vx.mean_vector(vs) if vs else []
            return mean_cache[ids] or None

        rankings["meaning"] = {}
        for cid, qset, items in queries:
            qv = mean_of(frozenset(qset.claim_ids))
            scores = {}
            for iid, it in items.items():
                v = mean_of(it.claim_ids)
                if qv is not None and v is not None:
                    scores[iid] = vx.dot(qv, v)
            rankings["meaning"][cid] = [(i, scores[i]) for i in _rank(scores)]
        seconds["meaning_rank"] = time.perf_counter() - t1
        seconds["embed"] = stats.seconds

    modes: dict[str, ModeMetrics] = {}
    items_of = {cid: items for cid, _, items in queries}
    for mode, per_case in rankings.items():
        ranks: dict[str, int | None] = {}
        outcomes: dict[str, ProbeOutcome] = {}
        for c in protocol.partners:
            ranked = [i for i, _ in per_case[c.id]]
            ranks[c.id] = _first(ranked, items_of[c.id], lambda it, c=c: it.label_key in c.targets)
        for p in protocol.probes:
            ranked = [i for i, _ in per_case[p.id]]
            outcomes[p.id] = ProbeOutcome(
                _first(ranked, items_of[p.id], lambda it, p=p: it.label_key == p.partner),
                _first(ranked, items_of[p.id], lambda it: it.id == OTHER_ITEM))
        modes[mode] = mode_metrics(ranks, outcomes)
    tops = {mode: {cid: r[:TOP_K] for cid, r in per_case.items()} for mode, per_case in rankings.items()}
    verdict = pass_rule(modes["keyword"], modes["meaning"]) if "meaning" in modes else None
    return EvalResult(protocol, modes, tops, verdict, stats, seconds)


# --- report -------------------------------------------------------------------------------------

def _r(rank: int | None) -> str:
    return "-" if rank is None else str(rank)


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def render_markdown(result: EvalResult, *, title: str = "Concept search adoption test") -> str:
    p = result.protocol
    kw = result.modes["keyword"]
    mn = result.modes.get("meaning")
    lines = [f"# {title}", ""]
    lines += [f"- Cases: `{p.cases_path or '-'}` (sha256 `{p.cases_sha256 or '-'}`)",
              f"- Protocol written: {p.created_at}",
              f"- Embedding: {result.embed_stats.model if result.embed_stats else 'none'}"]
    if result.embed_stats:
        s = result.embed_stats
        lines.append(f"- Claim contexts: {s.requested} ({s.embedded} embedded now in {s.seconds:.1f}s, "
                     f"{s.cached} from the cache)")
    lines.append(f"- Embed text: `{p.embed_format}`")
    lines += ["", "## Partner cases (rank of the true partner; '-' = not found)", "",
              "| case | kind | A | B (also) | keyword rank | meaning rank |", "|---|---|---|---|---|---|"]
    for c in p.partners:
        b = c.targets[0] + (f" ({', '.join(c.targets[1:])})" if len(c.targets) > 1 else "")
        if c.missing_targets:
            b += f" [no claims: {', '.join(c.missing_targets)}]"
        kind = c.kind + (" (control)" if c.control else "")
        lines.append(f"| {c.id} | {kind} | {c.a.labels[0]} | {b} | {_r(kw.partner_ranks[c.id])} | "
                     f"{_r(mn.partner_ranks[c.id]) if mn else 'n/a'} |")
    lines += ["", "| metric | keyword | meaning |", "|---|---|---|",
              f"| recall@{TOP_K} | {kw.recall_at_10:.2f} | {f'{mn.recall_at_10:.2f}' if mn else 'n/a'} |",
              f"| MRR | {kw.mrr:.3f} | {f'{mn.mrr:.3f}' if mn else 'n/a'} |", ""]
    lines += ["## Sense probes (partner rank vs other-side rank in the query side's neighbour list)", "",
              "| probe | label | query side | other side | partner | keyword p/o | keyword separated | "
              "meaning p/o | meaning separated | meaning top-1 is other side |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for pr in p.probes:
        ko = kw.probes[pr.id]
        mo = mn.probes[pr.id] if mn else None
        lines.append(
            f"| {pr.id} | {pr.label} | {pr.query_gloss} ({len(pr.query.claim_ids)} claims) | {pr.other_gloss} "
            f"({len(pr.other.claim_ids)} claims) | {pr.partner} | {_r(ko.partner_rank)}/{_r(ko.other_rank)} | "
            f"{_yn(ko.separated)} | {f'{_r(mo.partner_rank)}/{_r(mo.other_rank)}' if mo else 'n/a'} | "
            f"{_yn(mo.separated) if mo else 'n/a'} | {_yn(mo.over_merged) if mo else 'n/a'} |")
    lines += ["", "| metric | keyword | meaning |", "|---|---|---|",
              f"| separation | {kw.separation:.2f} | {f'{mn.separation:.2f}' if mn else 'n/a'} |",
              f"| over-merges | {', '.join(kw.over_merged) or 'none'} | "
              f"{(', '.join(mn.over_merged) or 'none') if mn else 'n/a'} |", ""]
    lines += ["## Verdict", ""]
    v = result.verdict
    if v is None:
        lines.append("No meaning run: no verdict.")
    else:
        lines += [f"**{'PASS: adopt embeddings' if v.passed else 'FAIL: keep keyword search'}**", "",
                  f"1. separation >= {SEPARATION_MIN}: {mn.separation:.2f} -> {_yn(v.checks['separation'])}",
                  f"2. recall@{TOP_K} >= keyword's ({mn.recall_at_10:.2f} vs {kw.recall_at_10:.2f}) -> "
                  f"{_yn(v.checks['recall'])}; partners keyword misses: {len(v.new_partners)} "
                  f"({', '.join(v.new_partners) or 'none'}), needed {v.needed_new} -> {_yn(v.checks['new_partners'])}",
                  f"3. no over-merge: {', '.join(mn.over_merged) or 'none'} -> {_yn(v.checks['no_over_merge'])}"]
    if p.malformed:
        lines += ["", "## Malformed cases (not scored; the cases file was not edited)", ""]
        lines += [f"- {m['id']}: {m['reason']}" for m in p.malformed]
    lines += ["", "## How the probe sides were built", ""]
    for pr in p.probes:
        lines.append(f"- {pr.id}: query side: {pr.query.how}; other side: {pr.other.how}. Query labels: "
                     f"{', '.join(pr.query.labels)}; other labels: {', '.join(pr.other.labels)}")
    lines += ["", "## Top neighbours", ""]
    for mode, per_case in result.tops.items():
        lines += [f"### {mode}", ""]
        for cid, ranked in per_case.items():
            lines.append(f"- {cid}: " + "; ".join(f"{i} ({s:.3f})" for i, s in ranked[:5]))
        lines.append("")
    lines += ["## Timing", ""] + [f"- {k}: {v:.1f}s" for k, v in result.seconds.items()]
    return "\n".join(lines) + "\n"


def result_json(result: EvalResult) -> dict[str, Any]:
    return {
        "protocol": result.protocol.to_json(),
        "modes": {m: {**asdict(x), "probes": {k: {**asdict(o), "separated": o.separated,
                                                  "over_merged": o.over_merged} for k, o in x.probes.items()}}
                  for m, x in result.modes.items()},
        "verdict": asdict(result.verdict) if result.verdict else None,
        "embed_stats": asdict(result.embed_stats) if result.embed_stats else None,
        "tops": result.tops,
        "seconds": result.seconds,
    }
