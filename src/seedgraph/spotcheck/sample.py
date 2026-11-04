"""The spot-check sample (design §5.1-5.2, §13 Q5-Q8; slice R4).

**Population** (§5.1): every agent claim of each paper's current note, that is
``extracted_claims`` whose ``structured_note_id`` is a ``note_current`` note. For an
edited paper that is the overlay's base, so edited claims are in (judged on the agent's
original row, which an overlay never changes) and hidden claims are in too: the
population never reads ``current_claims``, which drops hidden claims. Researcher-added
claims (``rc_`` ids) never reach ``extracted_claims``; lens claims have no note. A claim
with a mark in any sample is out, so each claim is marked once; a claim drawn into an
earlier sample but never marked stays in.

**Model key**: ``model_name`` as recorded (a client label included, Q7); else
``"<provider>, model not recorded"``; else ``"model not recorded"``.

**The draw** (§5.2): one claim at a time from the ``(claim type, model key)`` cell with
the fewest marks so far (earlier marks plus this sample's draws); ties go to the model
with fewer marks, then the type with fewer marks, then a seeded choice. Within the cell,
a seeded random claim, skipping papers that already have :data:`PER_PAPER_MAX` claims in
this sample. It stops at ``n`` or when no cell can give a claim. The population is sorted
by claim id and one ``random.Random(seed)`` drives every choice, so the same library
state and seed give the same sample (the stage 4 review-sample pattern,
``concepts.researcher.create_review_sample``); the sample lives in its own tables because
its items carry marks.
"""

from __future__ import annotations

import hashlib
import json
import random
import secrets
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..errors import SeedgraphError
from ..library import graphs
from ..library.store import Library
from ..notes import interpretation, overlay
from ..work.store import fmt_ts, now_utc

DEFAULT_SIZE = 20
#: At most this many claims of one paper in one sample (Q5).
PER_PAPER_MAX = 2
NO_MODEL = "model not recorded"
NO_MODEL_SUFFIX = ", model not recorded"
#: Optional filters: a graph slug (its member papers), a claim type, a model key.
FILTER_KEYS: tuple[str, ...] = ("graph", "claim_type", "model")

Cell = tuple[str, str]  # (claim_type, model_key)


class SpotCheckError(SeedgraphError):
    """A spot-check operation was refused."""


@dataclass(frozen=True)
class Claim:
    """One claim of the population."""

    claim_id: str
    paper_id: str
    extraction_run_id: str
    claim_type: str
    model_key: str

    @property
    def cell(self) -> Cell:
        return (self.claim_type, self.model_key)


@dataclass(frozen=True)
class Item:
    """One claim of a sample: the stored row plus what the researcher sees.

    The claim fields are the agent's **original** row (``extracted_claims``), never the
    overlay's edit. ``edited`` / ``hidden`` / ``overlay_version`` describe the current
    interpretation; ``earlier_extraction`` is set when the paper was re-noted since the
    sample was drawn (the claim is no longer in the current note). ``fixed`` is derived
    (:func:`seedgraph.spotcheck.marks.fixed`).
    """

    sample_id: str
    ordinal: int
    claim_id: str
    paper_id: str
    extraction_run_id: str
    claim_type: str
    model_key: str
    fingerprint: Mapping[str, Any]
    mark: str | None
    reasons: tuple[str, ...]
    note: str | None
    marked_at: str | None
    title: str | None
    year: int | None
    normalized_label: str | None
    claim_text: str | None
    plain_words: str | None
    status: str | None
    quote: str | None
    page_start: int | None
    page_end: int | None
    span_id: str | None
    extracted_at: str | None
    edited: bool
    hidden: bool
    overlay_version: int | None
    earlier_extraction: bool
    fixed: bool


@dataclass(frozen=True)
class Sample:
    sample_id: str
    seed: int
    size: int                       # the size asked for; ``len(items)`` may be smaller
    filters: Mapping[str, str]
    created_at: str
    finished_at: str | None
    items: tuple[Item, ...]

    @property
    def n_marked(self) -> int:
        return sum(1 for i in self.items if i.mark is not None)

    @property
    def counts(self) -> dict[str, int]:
        """Running counts of this sample's marks: right, partly, wrong, unmarked."""
        c = Counter(i.mark or "unmarked" for i in self.items)
        return {k: c.get(k, 0) for k in ("right", "partly", "wrong", "unmarked")}

    def item(self, claim_id: str) -> Item | None:
        return next((i for i in self.items if i.claim_id == claim_id), None)


# --- the population --------------------------------------------------------------------------

def model_key(model_name: str | None, provider: str | None) -> str:
    """The model stratum of a claim's extraction run (§5.1, Q7): kept as recorded."""
    if model_name and model_name.strip():
        return model_name.strip()
    if provider and provider.strip():
        return provider.strip() + NO_MODEL_SUFFIX
    return NO_MODEL


def _conn(lib: Library | sqlite3.Connection) -> sqlite3.Connection:
    return lib if isinstance(lib, sqlite3.Connection) else lib.conn


def marked_claim_ids(lib: Library | sqlite3.Connection) -> set[str]:
    return {r[0] for r in _conn(lib).execute("SELECT DISTINCT claim_id FROM spot_check_items WHERE mark IS NOT NULL")}


def population(lib: Library | sqlite3.Connection, *, include_marked: bool = False,
               paper_ids: Iterable[str] | None = None) -> list[Claim]:
    """Every agent claim of each paper's current note, sorted by claim id; already marked
    claims left out unless ``include_marked`` (the accuracy weights use the whole). ``not_found``
    rows (no text, no quote: nothing to judge) are left out (researcher, 2026-11-04)."""
    conn = _conn(lib)
    rows = conn.execute(
        "SELECT c.claim_id, c.paper_id, c.extraction_run_id, c.claim_type, r.model_name, r.provider "
        "FROM extracted_claims c "
        "JOIN note_current n ON n.note_id = c.structured_note_id "
        "JOIN extraction_runs r ON r.extraction_run_id = c.extraction_run_id "
        "WHERE c.status IS NOT 'not_found' "
        "ORDER BY c.claim_id").fetchall()
    keep = None if paper_ids is None else set(paper_ids)
    marked = set() if include_marked else marked_claim_ids(conn)
    return [Claim(r[0], r[1], r[2], r[3], model_key(r[4], r[5])) for r in rows
            if r[0] not in marked and (keep is None or r[1] in keep)]


def mark_counts(lib: Library | sqlite3.Connection) -> Counter[Cell]:
    """Marked claims per cell (the latest mark per claim, as ``spot_check_accuracy``)."""
    return Counter({(r[0], r[1]): int(r[2]) for r in _conn(lib).execute(
        "SELECT claim_type, model_key, n_marked FROM spot_check_accuracy")})


def _check_filters(filters: Mapping[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in (filters or {}).items():
        if key not in FILTER_KEYS:
            raise SpotCheckError(f"unknown spot-check filter {key!r}; use {', '.join(FILTER_KEYS)}")
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise SpotCheckError(f"the {key} filter must be text")
        out[key] = value
    return out


def _filtered(lib: Library, filters: Mapping[str, str]) -> list[Claim]:
    paper_ids = None
    if "graph" in filters:
        try:
            with graphs.open_graph(lib, filters["graph"]) as g:
                paper_ids = [p.paper_id for p in graphs.membership(g, status="member")]
        except graphs.GraphError as e:
            raise SpotCheckError(f"no graph {filters['graph']!r}: {e}") from e
    pop = population(lib, paper_ids=paper_ids)
    if "claim_type" in filters:
        pop = [c for c in pop if c.claim_type == filters["claim_type"]]
    if "model" in filters:
        pop = [c for c in pop if c.model_key == filters["model"]]
    return pop


# --- the draw --------------------------------------------------------------------------------

def draw(claims: Sequence[Claim], n: int, seed: int, marks: Mapping[Cell, int] | None = None) -> list[Claim]:
    """The §5.2 rule over ``claims`` (unmarked), given earlier ``marks`` per cell. Pure."""
    rng = random.Random(seed)
    pool: dict[Cell, list[Claim]] = {}
    for c in sorted(claims, key=lambda c: c.claim_id):
        pool.setdefault(c.cell, []).append(c)
    cell_marks: Counter[Cell] = Counter(marks or {})
    model_marks: Counter[str] = Counter()
    type_marks: Counter[str] = Counter()
    for (ctype, mkey), k in cell_marks.items():
        model_marks[mkey] += k
        type_marks[ctype] += k
    per_paper: Counter[str] = Counter()
    chosen: list[Claim] = []
    while len(chosen) < n:
        live = sorted(cell for cell, cs in pool.items() if any(per_paper[c.paper_id] < PER_PAPER_MAX for c in cs))
        if not live:
            break
        rank = {cell: (cell_marks[cell], model_marks[cell[1]], type_marks[cell[0]]) for cell in live}
        best = min(rank.values())
        cell = rng.choice([c for c in live if rank[c] == best])
        pick = rng.choice([c for c in pool[cell] if per_paper[c.paper_id] < PER_PAPER_MAX])
        pool[cell].remove(pick)
        chosen.append(pick)
        per_paper[pick.paper_id] += 1
        cell_marks[cell] += 1
        model_marks[cell[1]] += 1
        type_marks[cell[0]] += 1
    return chosen


def _sample_id(seed: int, n: int, ids: Sequence[str]) -> str:
    return "sc_" + hashlib.sha256(json.dumps([seed, n, list(ids)]).encode()).hexdigest()[:12]


def create_sample(lib: Library, *, n: int = DEFAULT_SIZE, seed: int | None = None,
                  filters: Mapping[str, Any] | None = None, now: str | None = None) -> Sample:
    """Draw and store a spot-check sample of up to ``n`` unmarked agent claims. With no
    ``seed`` one is drawn; it is stored either way. The same library state and seed give
    the same sample (and the same ``sample_id``: drawing it again returns the stored one)."""
    if n < 1:
        raise SpotCheckError("a spot-check sample needs at least one claim")
    clean = _check_filters(filters)
    seed = secrets.randbits(31) if seed is None else int(seed)
    now = now or fmt_ts(now_utc())
    chosen = draw(_filtered(lib, clean), n, seed, mark_counts(lib))
    if not chosen:
        raise SpotCheckError("no unmarked agent claims to spot-check" + (" with these filters" if clean else ""))
    sid = _sample_id(seed, n, [c.claim_id for c in chosen])
    with lib.write() as conn:
        if conn.execute("SELECT 1 FROM spot_check_samples WHERE sample_id = ?", (sid,)).fetchone() is None:
            conn.execute("INSERT INTO spot_check_samples (sample_id, seed, size, filters_json, created_at) "
                         "VALUES (?, ?, ?, ?, ?)", (sid, seed, n, json.dumps(clean, sort_keys=True), now))
            for i, c in enumerate(chosen, start=1):
                fp = json.dumps(overlay.fingerprint(conn, c.claim_id), sort_keys=True, ensure_ascii=False)
                conn.execute(
                    "INSERT INTO spot_check_items (sample_id, claim_id, ordinal, paper_id, extraction_run_id, "
                    "claim_type, model_key, claim_fingerprint) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (sid, c.claim_id, i, c.paper_id, c.extraction_run_id, c.claim_type, c.model_key, fp))
    out = sample(lib, sid)
    assert out is not None
    return out


# --- reading ---------------------------------------------------------------------------------

_ITEM_SQL = (
    "SELECT i.sample_id, i.ordinal, i.claim_id, i.paper_id, i.extraction_run_id, i.claim_type, i.model_key, "
    "i.claim_fingerprint, i.mark, i.reasons_json, i.note, i.marked_at, p.title, p.year, "
    "c.normalized_label, c.claim_text, c.plain_words, c.status, c.structured_note_id, r.created_at AS extracted_at "
    "FROM spot_check_items i "
    "LEFT JOIN papers p ON p.paper_id = i.paper_id "
    "LEFT JOIN extracted_claims c ON c.claim_id = i.claim_id "
    "LEFT JOIN extraction_runs r ON r.extraction_run_id = i.extraction_run_id "
    "WHERE i.sample_id = ? ORDER BY i.ordinal")


def _top_quote(conn: sqlite3.Connection, claim_id: str) -> sqlite3.Row | None:
    """The claim's top anchor (the same rank order as the notes' ``_top_span``)."""
    return conn.execute(
        "SELECT e.span_id, e.exact_quote, e.page_start, e.page_end FROM claim_spans cs "
        "JOIN evidence_spans e ON e.span_id = cs.span_id WHERE cs.claim_id = ? "
        "ORDER BY cs.rank, cs.span_id LIMIT 1", (claim_id,)).fetchone()


def _item(conn: sqlite3.Connection, r: sqlite3.Row, interps: dict[str, Any]) -> Item:
    from . import marks  # marks imports this module

    pid = r["paper_id"]
    if pid not in interps:
        interps[pid] = interpretation.current(conn, pid, "researcher", show_hidden=True)
    interp = interps[pid]
    current = next((c for c in interp.claims if c.claim_id == r["claim_id"]), None) if interp else None
    in_current = r["structured_note_id"] is not None and conn.execute(
        "SELECT 1 FROM note_current WHERE note_id = ?", (r["structured_note_id"],)).fetchone() is not None
    quote = _top_quote(conn, r["claim_id"])
    fp = json.loads(r["claim_fingerprint"])
    return Item(
        sample_id=r["sample_id"], ordinal=r["ordinal"], claim_id=r["claim_id"], paper_id=pid,
        extraction_run_id=r["extraction_run_id"], claim_type=r["claim_type"], model_key=r["model_key"],
        fingerprint=fp, mark=r["mark"], reasons=tuple(json.loads(r["reasons_json"] or "[]")), note=r["note"],
        marked_at=r["marked_at"], title=r["title"], year=r["year"], normalized_label=r["normalized_label"],
        claim_text=r["claim_text"], plain_words=r["plain_words"], status=r["status"],
        quote=quote["exact_quote"] if quote else None, page_start=quote["page_start"] if quote else None,
        page_end=quote["page_end"] if quote else None, span_id=quote["span_id"] if quote else None,
        extracted_at=r["extracted_at"],
        edited=bool(current and current.edited_by_researcher), hidden=bool(current and current.hidden),
        overlay_version=interp.version_no if interp else None, earlier_extraction=not in_current,
        fixed=marks.is_fixed(conn, pid, r["claim_id"], fp, r["marked_at"]))


def sample(lib: Library | sqlite3.Connection, sample_id: str) -> Sample | None:
    """A stored sample with its items in draw order, or ``None``."""
    conn = _conn(lib)
    s = conn.execute("SELECT sample_id, seed, size, filters_json, created_at, finished_at "
                     "FROM spot_check_samples WHERE sample_id = ?", (sample_id,)).fetchone()
    if s is None:
        return None
    interps: dict[str, Any] = {}
    items = tuple(_item(conn, r, interps) for r in conn.execute(_ITEM_SQL, (sample_id,)).fetchall())
    return Sample(s[0], s[1], s[2], json.loads(s[3] or "{}"), s[4], s[5], items)


def samples(lib: Library | sqlite3.Connection) -> list[Sample]:
    """Every sample, newest first."""
    ids = [r[0] for r in _conn(lib).execute(
        "SELECT sample_id FROM spot_check_samples ORDER BY created_at DESC, sample_id DESC")]
    return [s for s in (sample(lib, i) for i in ids) if s is not None]
