"""Guided corpus growth for a context graph (docs/V3_CONCEPTS_DESIGN.md §8.2, §12 C6).

Growth beyond the seeds is a convenience. It never runs by itself, nothing joins a graph
without the researcher's approval, and a PDF the researcher supplies is always welcome.
The wizard (C8) calls, in order:

    estimate(seed_count, depth, cap)                   -> dict          (step 2, no network)
    default_seeds(lib, slug)                           -> list[str]     (step 1)
    walk(lib, slug, *, chain, seeds, depth, cap)       -> WalkResult    (step 3, a JobRunner job)
    review(lib, slug)                                  -> list[FoundPaper]   (step 4)
    approve(lib, slug, choices, *, chain, fetch=None)  -> list[Approval]     (step 5)
    ingest.needs_pdf(lib, graph=slug)                  (the fallthrough list)

and, after a newly held paper is converted:

    parse_references_for(lib, paper_id, *, chain)      -> ParseResult

**Walk.** Generation 1 expands the seeds (default: the graph's seed flags, else every
member); generation 2 (``depth=2``) expands the generation-1 papers ranked best, at most
``cap`` of them. A source's outbound references come from
- the library: its ``citation_links`` and resolved ``reference_entries``. A held source
  whose reference list was never parsed is parsed first (:func:`parse_references_for`);
- the provider chain: ``referenced_works`` of the source's ids (cached by the chain).
Each reference is resolved to a library paper by its strong ids (stage 1 identity:
every id is looked up in ``identifiers`` first; an unknown paper is created **cited-only**
under ``derive_key``/``mint_paper_id``, so the same reference always maps to the same id).
At most ``cap`` *new* papers are created per generation, the best ranked first; papers the
library already has are not new. A reference with no strong id is not resolved. New links
go to ``citation_links`` (``provider_reference``). The walk writes nothing to the graph;
its found list is kept in the graph's ``graph_meta`` under :data:`WALK_META_KEY` so the
review step can show it (a re-walk with the same answers rewrites the same list).

**Ranking** (v2's walk ranking, made graph-aware): by the number of distinct graph
**members** citing the paper (every member's library links count, not only the walked
ones), then by the number of walked papers citing it, then by ``paper_id``.

**Approval.** ``choices`` maps paper ids to ``'member'`` or ``'frontier'``. A held paper
joins as the researcher chose; a cited-only paper can only join the frontier (asking for
member refuses the whole approval before anything is written, as does asking to move a
member to the frontier: approval never demotes). Every cited-only paper
approved gets one open-access attempt (``acquisition.fetch.fetch_oa``: open-access copies
only, never a paywalled one); a downloaded copy is attached as ``open_access`` and queued
for conversion. The rest appear in the stage 3 needs-a-PDF list with their DOI link.
Approving never starts a notes batch.

**Reference parsing.** :func:`parse_references_for` parses a held paper's references
(``citation.bib_parser`` over its ``document_sections`` reference ranges) and resolves
them (``citation.resolve``: strong id first, then title + year behind the strong-id gate
and the precision guard). Resolved entries become ``citation_links``
(``parsed_bibliography``) to library papers, created cited-only when new. Ambiguous and
suspect entries (the mid band) get no link; each raises a ``citation_review`` attention
item unless the researcher already rejected that reference.

Rules chosen where the design is silent:
- Papers already in the graph at any status (member, frontier or excluded) are never
  found again; an excluded paper stays excluded.
- ``reference_entries`` ids are ``ref_`` + sha256(citing paper, markdown hash, ordinal),
  so a re-parse of the same markdown writes the same rows. A paper whose entries were
  parsed from the current markdown is skipped; a changed markdown re-parses it, deleting
  its old ``parsed_bibliography`` links first (references with a researcher's citation
  review decision are kept as they are).
- An existing paper found by the walk gains the found record's missing ids (when no other
  paper holds them) and, when empty, its title, year and OA status.
- A found paper whose provider title is generic is created without a title and gets a
  ``metadata_needed`` item, as the DOI paste does; a title + year resemblance to another
  paper raises an ``identity_check`` item; nothing is merged here.
- OA available is read from ``papers.oa_status``: empty is unknown, ``closed`` is no,
  any other value is yes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from . import attention, graphs
from .identity import derive_key, mint_paper_id, normalize_ids, paper_id_for_key
from .metadata_quality import generic_title_reason
from .metadata_verify import provider_of, record_values, run_sync
from .store import Library, LibraryError, is_held, paper_exists

CITATION_REVIEW_KIND = "citation_review"
METADATA_NEEDED = "metadata_needed"
WALK_META_KEY = "growth:last_walk"
MAX_DEPTH = 2
DOI_URL = "https://doi.org/{}"
#: Strong-id lookup order (stage 1 identity precedence, then the carried ids).
_ID_ORDER = ("doi", "openalex", "arxiv", "s2", "ssrn")
_ENRICH_BATCH = 50

Choice = Literal["member", "frontier"]
OaAvailable = Literal["yes", "no", "unknown"]


class GrowthError(LibraryError):
    """A growth request was refused."""


@dataclass(frozen=True)
class GenerationReport:
    generation: int
    sources: tuple[str, ...]
    found: int  # papers found this generation (new and already known)
    new_papers: tuple[str, ...]
    skipped_new: int  # new references left out by the cap
    provider_failures: int


@dataclass(frozen=True)
class ParseResult:
    paper_id: str
    status: Literal["parsed", "already_parsed", "no_markdown", "no_references"]
    entries: int = 0
    resolved: int = 0
    review_items: tuple[str, ...] = ()
    new_papers: tuple[str, ...] = ()


@dataclass(frozen=True)
class WalkResult:
    slug: str
    seeds: tuple[str, ...]
    depth: int
    cap: int
    generations: tuple[GenerationReport, ...]
    new_papers: tuple[str, ...]
    links_added: int
    parsed: tuple[ParseResult, ...] = ()
    attention_item_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FoundPaper:
    paper_id: str
    title: str | None
    year: int | None
    authors: tuple[str, ...]
    doi: str | None
    doi_url: str | None
    generation: int
    cited_by_members: int
    citing_members: tuple[str, ...]
    cited_in_walk: int
    held: bool
    oa: OaAvailable
    reasons: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class Approval:
    paper_id: str
    status: Choice  # the paper's status in the graph now
    held: bool
    oa: Literal["held", "none", "failed", "already_held"]
    oa_source: str | None  # "<provider>: <url>" of a fetched open-access copy
    needs_pdf: bool
    doi_url: str | None
    detail: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- step 1 and 2 -----------------------------------------------------------------------------


def estimate(seed_count: int, depth: int, cap: int) -> dict:
    """The pre-run budget (v2 ``acquisition.preview.preview_acquisition``; no network)."""
    from ..acquisition.preview import preview_acquisition

    return preview_acquisition(seed_count, depth, cap)


def default_seeds(lib: Library, slug: str) -> list[str]:
    """The graph's seed-flagged papers, else every member (sorted)."""
    with graphs.open_graph(lib, slug) as g:
        rows = graphs.membership(g)
    seeds = [r.paper_id for r in rows if r.is_seed]
    return seeds or [r.paper_id for r in rows if r.status == "member"]


# --- provider helpers -------------------------------------------------------------------------


class _SyncChain:
    """The chain's lookups run to completion, so ``citation.resolve`` sees plain dicts."""

    def __init__(self, chain: Any):
        self.chain = chain

    def by_doi(self, doi: str) -> dict | None:
        return run_sync(self.chain.by_doi(doi))

    def by_title(self, title: str, year: int | None = None) -> dict | None:
        return run_sync(self.chain.by_title(title, year))


def _ids_of(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    from ..citation.resolve import record_to_identity

    ident = record_to_identity(dict(record))
    return normalize_ids((t, ident[t]) for t in _ID_ORDER if ident.get(t))


def _owner_of(lib: Library, ids: Iterable[tuple[str, str]]) -> str | None:
    by_type = {t: [] for t in _ID_ORDER}
    for t, v in ids:
        by_type.setdefault(t, []).append(v)
    for t in _ID_ORDER:
        for v in sorted(by_type.get(t, ())):
            row = lib.conn.execute("SELECT paper_id FROM identifiers WHERE id_type = ? AND id_value = ? "
                                   "ORDER BY paper_id LIMIT 1", (t, v)).fetchone()
            if row is not None:
                return row[0]
    return None


def _source_record(lib: Library, paper_id: str) -> dict:
    """A provider lookup record for a library paper (its ids, title and year)."""
    rec: dict[str, Any] = {}
    for t, v in lib.conn.execute("SELECT id_type, id_value FROM identifiers WHERE paper_id = ? "
                                 "ORDER BY id_type, id_value", (paper_id,)):
        key = {"doi": "doi", "openalex": "openalex_id", "arxiv": "arxiv_id", "s2": "semantic_scholar_id",
               "ssrn": "ssrn_id"}[t]
        rec.setdefault(key, v)
    row = lib.conn.execute("SELECT title, year FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
    if row is not None:
        if row[0]:
            rec["title"] = row[0]
        if row[1] is not None:
            rec["year"] = row[1]
    return rec


def _enrich(chain: Any, records: list[dict]) -> None:
    """Fill title/DOI of bare OpenAlex references from batched ``by_openalex_ids`` (v2 backfill)."""
    if not hasattr(chain, "by_openalex_ids"):
        return
    need: dict[str, list[dict]] = {}
    for r in records:
        if r.get("title"):
            continue
        wid = dict(_ids_of(r)).get("openalex")
        if wid:
            need.setdefault(wid, []).append(r)
    wids = sorted(need)
    for i in range(0, len(wids), _ENRICH_BATCH):
        try:
            batch = run_sync(chain.by_openalex_ids(wids[i:i + _ENRICH_BATCH])) or []
        except Exception:  # noqa: BLE001 - a failed batch leaves the references bare
            continue
        for got in batch:
            if not isinstance(got, dict):
                continue
            wid = dict(_ids_of(got)).get("openalex")
            for r in need.get(wid, ()):
                for k, v in got.items():
                    r.setdefault(k, v)


# --- library writes ---------------------------------------------------------------------------


def _add_free_ids(conn, paper_id: str, ids: Iterable[tuple[str, str]], source: str) -> None:
    for t, v in ids:
        if conn.execute("SELECT 1 FROM identifiers WHERE id_type = ? AND id_value = ?", (t, v)).fetchone() is None:
            conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value, source) VALUES (?, ?, ?, ?)",
                         (paper_id, t, v, source))


def _fill_existing(lib: Library, paper_id: str, ids, record: Mapping[str, Any]) -> None:
    provider = provider_of(dict(record))
    values = record_values(dict(record))
    oa_status = record.get("oa_status") if isinstance(record.get("oa_status"), str) else None
    with lib.write() as conn:
        _add_free_ids(conn, paper_id, ids, provider)
        row = conn.execute("SELECT title, year, oa_status FROM papers WHERE paper_id = ?", (paper_id,)).fetchone()
        title = values["title"]
        if row[0] is None and title and generic_title_reason(title, venue=values["venue"]) is None:
            conn.execute("UPDATE papers SET title = ?, title_source = ? WHERE paper_id = ?", (title, provider, paper_id))
        if row[1] is None and values["year"] is not None:
            conn.execute("UPDATE papers SET year = ? WHERE paper_id = ?", (values["year"], paper_id))
        if row[2] is None and oa_status:
            conn.execute("UPDATE papers SET oa_status = ? WHERE paper_id = ?", (oa_status, paper_id))


def _create_cited_only(lib: Library, ids: tuple[tuple[str, str], ...], record: Mapping[str, Any]) -> tuple[str, list[str]]:
    """A new cited-only paper keyed on its strong ids; returns ``(paper_id, attention ids)``."""
    from .ingest import raise_identity_check, similar_papers

    provider = provider_of(dict(record))
    values = record_values(dict(record))
    if values["title"] and generic_title_reason(values["title"], venue=values["venue"]) is not None:
        values["title"] = None
    taken = {r[0] for r in lib.conn.execute("SELECT paper_id FROM papers")}
    pid = mint_paper_id(derive_key(ids=ids), taken).paper_id
    item_ids: list[str] = []
    abstract = record.get("abstract") if isinstance(record.get("abstract"), str) else None
    oa_status = record.get("oa_status") if isinstance(record.get("oa_status"), str) else None
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO papers (paper_id, title, title_source, authors_json, year, venue, abstract, oa_status, "
            "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, values["title"], provider if values["title"] else None,
             json.dumps(values["authors"], ensure_ascii=False) if values["authors"] else None,
             values["year"], values["venue"], abstract, oa_status, _now()))
        _add_free_ids(conn, pid, ids, provider)
        if not values["title"]:
            item_ids.append(attention.raise_item(
                lib, METADATA_NEEDED, {"paper_id": pid}, paper_id=pid,
                detail="found by guided growth; its provider gave no usable title").item_id)
        similar = similar_papers(lib, pid, values["title"], values["year"])
        if similar:
            item = raise_identity_check(lib, pid, similar, "a paper found by guided growth may be the same paper as")
            if item:
                item_ids.append(item)
    return pid, item_ids


def _resolve_record(lib: Library, record: Mapping[str, Any]) -> tuple[str | None, bool, list[str]]:
    """The library paper for a provider record: ``(paper_id, created, attention ids)``.
    ``(None, False, [])`` when the record carries no strong id."""
    ids = _ids_of(record)
    if not ids:
        return None, False, []
    owner = _owner_of(lib, ids)
    if owner is not None:
        _fill_existing(lib, owner, ids, record)
        return owner, False, []
    pid, items = _create_cited_only(lib, ids, record)
    return pid, True, items


def _link(conn, source: str, target: str, provenance: str, confidence: float,
          reference_id: str | None = None) -> int:
    if source == target:
        return 0
    cur = conn.execute(
        "INSERT OR IGNORE INTO citation_links (source_paper_id, target_paper_id, provenance, confidence, "
        "reference_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (source, target, provenance, confidence, reference_id, _now()))
    return cur.rowcount


def _library_targets(lib: Library, paper_id: str) -> set[str]:
    rows = lib.conn.execute(
        "SELECT target_paper_id FROM citation_links WHERE source_paper_id = ? "
        "UNION SELECT resolved_paper_id FROM reference_entries WHERE citing_paper_id = ? "
        "AND resolution_status = 'resolved' AND resolved_paper_id IS NOT NULL", (paper_id, paper_id))
    return {r[0] for r in rows if r[0] != paper_id}


def _citing_members(lib: Library, paper_id: str, members: set[str]) -> set[str]:
    rows = lib.conn.execute(
        "SELECT source_paper_id FROM citation_links WHERE target_paper_id = ? "
        "UNION SELECT citing_paper_id FROM reference_entries WHERE resolved_paper_id = ? "
        "AND resolution_status = 'resolved'", (paper_id, paper_id))
    return {r[0] for r in rows} & members


# --- step 3: the walk -------------------------------------------------------------------------


def walk(lib: Library, slug: str, *, chain: Any, seeds: Sequence[str] | None = None, depth: int = 1,
         cap: int = 50, progress: Callable[[str], None] | None = None) -> WalkResult:
    """Walk the outbound references of ``seeds`` (module doc). Writes library papers and
    links, and the graph's found list; never the graph's membership."""
    if depth < 1 or depth > MAX_DEPTH:
        raise GrowthError(f"depth must be 1 to {MAX_DEPTH}, not {depth}")
    if cap < 1:
        raise GrowthError(f"cap must be at least 1, not {cap}")
    with graphs.open_graph(lib, slug) as g:
        rows = graphs.membership(g)
    in_graph = {r.paper_id for r in rows}
    members = {r.paper_id for r in rows if r.status == "member"}
    seeds = list(dict.fromkeys(seeds)) if seeds is not None else default_seeds(lib, slug)
    for pid in seeds:
        if not paper_exists(lib.conn, pid):
            raise GrowthError(f"seed {pid!r} is not in the library")

    seen = set(in_graph) | set(seeds)
    found: dict[str, dict[str, int]] = {}
    reports: list[GenerationReport] = []
    all_new: list[str] = []
    parsed: list[ParseResult] = []
    items: list[str] = []
    links_added = 0
    sources = seeds
    for gen in range(1, depth + 1):
        if not sources:
            break
        citers: dict[str, set[str]] = {}  # paper id (existing or to-be-minted) -> walked citers
        from_provider: dict[str, set[str]] = {}  # the citers a provider reported (links to write)
        pending: dict[str, tuple[tuple[tuple[str, str], ...], dict]] = {}  # minted id -> (ids, record)
        failures = 0
        for src in sources:
            if progress is not None:
                progress(f"generation {gen}: {src}")
            if is_held(lib.conn, src):
                res = parse_references_for(lib, src, chain=chain)
                if res.status == "parsed":
                    parsed.append(res)
                    all_new.extend(res.new_papers)
                    items.extend(res.review_items)
            for tgt in _library_targets(lib, src):
                citers.setdefault(tgt, set()).add(src)
            record = _source_record(lib, src)
            if not _ids_of(record):
                continue
            try:
                refs = [dict(r) for r in (run_sync(chain.referenced_works(record)) or []) if isinstance(r, dict)]
            except Exception:  # noqa: BLE001 - one source's provider failure is counted, not fatal
                failures += 1
                continue
            for ref in refs:
                ids = _ids_of(ref)
                if not ids:
                    continue
                owner = _owner_of(lib, ids)
                if owner is None:
                    owner = paper_id_for_key(derive_key(ids=ids))
                    if owner in pending:
                        for k, v in ref.items():
                            pending[owner][1].setdefault(k, v)
                    else:
                        pending[owner] = (ids, ref)
                else:
                    _fill_existing(lib, owner, ids, ref)
                citers.setdefault(owner, set()).add(src)
                from_provider.setdefault(owner, set()).add(src)

        def member_count(pid: str) -> int:
            known = _citing_members(lib, pid, members) if pid not in pending else set()
            return len(known | (citers[pid] & members))

        ranked = sorted((p for p in citers if p not in seen),
                        key=lambda p: (-member_count(p), -len(citers[p]), p))
        chosen_new = [p for p in ranked if p in pending][:cap]
        skipped = {p for p in ranked if p in pending} - set(chosen_new)
        _enrich(chain, [pending[p][1] for p in chosen_new])
        minted: dict[str, str] = {}
        gen_new: list[str] = []
        for p in chosen_new:
            ids, record = pending[p]
            pid, created, new_items = _resolve_record(lib, record)
            if pid is None:
                continue
            minted[p] = pid
            items.extend(new_items)
            if created:
                gen_new.append(pid)
        gen_found: list[str] = []
        with lib.write() as conn:
            for p in ranked:
                if p in skipped:
                    continue
                if p in pending and p not in minted:
                    continue
                pid = minted.get(p, p)
                for src in sorted(from_provider.get(p, ())):
                    links_added += _link(conn, src, pid, "provider_reference", 1.0)
                if pid in seen or pid in gen_found:
                    continue  # a found record that turned out to be a paper already placed
                gen_found.append(pid)
                found[pid] = {"generation": gen, "cited_in_walk": len(citers[p])}
        all_new.extend(gen_new)
        reports.append(GenerationReport(gen, tuple(sources), len(gen_found), tuple(gen_new), len(skipped), failures))
        seen |= set(gen_found)
        sources = gen_found[:cap]

    _store_found(lib, slug, {"seeds": list(seeds), "depth": depth, "cap": cap,
                             "found": [{"paper_id": p, **v} for p, v in sorted(found.items())]})
    return WalkResult(slug, tuple(seeds), depth, cap, tuple(reports), tuple(dict.fromkeys(all_new)), links_added,
                      tuple(parsed), tuple(dict.fromkeys(items)))


def _store_found(lib: Library, slug: str, payload: dict) -> None:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    with graphs.open_graph(lib, slug) as g:
        stored = g.conn.execute("SELECT value FROM graph_meta WHERE key = ?", (WALK_META_KEY,)).fetchone()
        if stored is not None and stored[0] == text:
            return
        with graphs.graph_write(g) as conn:
            graphs.set_graph_meta(conn, WALK_META_KEY, text)


def _load_found(lib: Library, slug: str) -> dict:
    with graphs.open_graph(lib, slug) as g:
        row = g.conn.execute("SELECT value FROM graph_meta WHERE key = ?", (WALK_META_KEY,)).fetchone()
    return json.loads(row[0]) if row and row[0] else {}


# --- step 4: review ---------------------------------------------------------------------------


def _oa_available(oa_status: str | None) -> OaAvailable:
    if not oa_status:
        return "unknown"
    return "no" if oa_status.strip().lower() == "closed" else "yes"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def review(lib: Library, slug: str) -> list[FoundPaper]:
    """The last walk's found papers not yet in the graph, ranked (module doc), with reasons."""
    payload = _load_found(lib, slug)
    with graphs.open_graph(lib, slug) as g:
        rows = graphs.membership(g)
    in_graph = {r.paper_id for r in rows}
    members = {r.paper_id for r in rows if r.status == "member"}
    out: list[FoundPaper] = []
    for entry in payload.get("found", ()):
        pid = entry["paper_id"]
        if pid in in_graph or not paper_exists(lib.conn, pid):
            continue
        row = lib.conn.execute(
            "SELECT title, year, authors_json, oa_status, "
            "(SELECT MIN(id_value) FROM identifiers WHERE paper_id = papers.paper_id AND id_type = 'doi') "
            "FROM papers WHERE paper_id = ?", (pid,)).fetchone()
        try:
            authors = tuple(json.loads(row[2])) if row[2] else ()
        except (TypeError, ValueError):
            authors = ()
        citing = tuple(sorted(_citing_members(lib, pid, members)))
        is_held_now = is_held(lib.conn, pid)
        oa = _oa_available(row[3])
        reasons = [f"cited by {_plural(len(citing), 'member')}"]
        if entry["generation"] > 1:
            reasons.append(f"cited by {_plural(entry['cited_in_walk'], 'paper')} found in generation "
                           f"{entry['generation'] - 1}")
        if is_held_now:
            reasons.append("already held (the library has its PDF)")
        elif oa == "yes":
            reasons.append("an open-access copy is reported")
        out.append(FoundPaper(pid, row[0], row[1], authors, row[4], DOI_URL.format(row[4]) if row[4] else None,
                              entry["generation"], len(citing), citing, entry["cited_in_walk"], is_held_now, oa,
                              tuple(reasons)))
    out.sort(key=lambda f: (f.generation, -f.cited_by_members, -f.cited_in_walk, f.paper_id))
    return out


# --- step 5: approve --------------------------------------------------------------------------


def approve(lib: Library, slug: str, choices: Mapping[str, Choice], *, chain: Any,
            fetch: Callable[[dict], Any] | None = None) -> list[Approval]:
    """Add the researcher's ticked papers to the graph (module doc). Every choice is checked
    before anything is written; then each paper joins, and each cited-only one gets one
    open-access attempt. Results follow ``choices`` order."""
    from .ingest import fetch_open_access

    if fetch is None:
        from ..acquisition.fetch import fetch_oa

        def fetch(record: dict) -> Any:  # noqa: F811 - the default open-access fetch
            return fetch_oa(record, chain=chain)

    with graphs.open_graph(lib, slug) as g:
        current = {r.paper_id: r.status for r in graphs.membership(g)}
    for pid, choice in choices.items():
        if current.get(pid) == "member" and choice != "member":
            raise GrowthError(f"paper {pid!r} is already a member; approval never demotes a member")
        if choice not in ("member", "frontier"):
            raise GrowthError(f"choose 'member' or 'frontier' for {pid}, not {choice!r}")
        if not paper_exists(lib.conn, pid):
            raise GrowthError(f"paper {pid!r} is not in the library")
        if choice == "member" and not is_held(lib.conn, pid):
            raise graphs.NotHeldError(f"paper {pid!r} is cited-only (no PDF); it can join the frontier, "
                                      "and becomes a member once its PDF is added")
    with graphs.open_graph(lib, slug) as g:
        existing = {r.paper_id: r for r in graphs.membership(g)}
        for pid, choice in choices.items():
            prev = existing.get(pid)
            graphs.set_membership(g, pid, choice, is_seed=bool(prev and prev.is_seed),
                                  reason="approved from guided growth")
    out: list[Approval] = []
    for pid, choice in choices.items():
        doi = lib.conn.execute("SELECT MIN(id_value) FROM identifiers WHERE paper_id = ? AND id_type = 'doi'",
                               (pid,)).fetchone()[0]
        doi_url = DOI_URL.format(doi) if doi else None
        if is_held(lib.conn, pid):
            out.append(Approval(pid, choice, True, "already_held", None, False, doi_url, f"joined as {choice}"))
            continue
        oa, source = fetch_open_access(lib, pid, _source_record(lib, pid), fetch)
        now_held = oa == "held"
        detail = (f"joined as {choice}; open-access PDF from {source}, queued for conversion" if now_held
                  else f"joined as {choice}; no open-access copy, so it is on the needs-a-PDF list")
        out.append(Approval(pid, choice, now_held, oa, source, not now_held, doi_url, detail))
    return out


# --- reference parsing ------------------------------------------------------------------------


def _parsed_fields_json(entry: Any) -> str:
    return json.dumps({"first_author": entry.first_author, "year": entry.year, "title": entry.title,
                       "doi": entry.doi, "arxiv": entry.arxiv, "ordinal": entry.ordinal,
                       "continued_author": entry.continued_author}, ensure_ascii=False)


def _reference_ranges(lib: Library, markdown_id: str) -> list[tuple[int, int, str]]:
    return [(int(s), int(e), h or "References") for s, e, h in lib.conn.execute(
        "SELECT start_char, end_char, heading_text FROM document_sections WHERE markdown_id = ? "
        "AND section_kind = 'references' AND end_char > start_char ORDER BY start_char, ordinal", (markdown_id,))]


def parse_references_for(lib: Library, paper_id: str, *, chain: Any) -> ParseResult:
    """Parse and resolve a held paper's reference list (module doc). Call it after the
    paper's PDF is converted; the walk calls it for every held source. No transaction may
    be open (the provider chain is consulted)."""
    from ..citation.bib_parser import parse_references
    from ..citation.resolve import resolve_references
    from ..notes.spans import read_markdown
    from ..project import identity as project_identity
    from ..work.packages import primary_markdown
    from .files import index_sections

    md = primary_markdown(lib.conn, paper_id)
    if md is None:
        return ParseResult(paper_id, "no_markdown")
    markdown_id, markdown_hash = md
    stored = {r[0] for r in lib.conn.execute(
        "SELECT DISTINCT markdown_hash FROM reference_entries WHERE citing_paper_id = ?", (paper_id,))}
    if stored and (markdown_hash in stored or None in stored):
        return ParseResult(paper_id, "already_parsed")
    index_sections(lib, markdown_id)
    ranges = _reference_ranges(lib, markdown_id)
    text = read_markdown(lib.conn, markdown_id) if ranges else None
    if not ranges or text is None:
        return ParseResult(paper_id, "no_references")

    parsed = parse_references(text, ranges)
    resolved = resolve_references(parsed, _SyncChain(chain), project_identity, citing_work_id=paper_id)
    rows: list[tuple[str, Any, str, str | None, str | None, float | None]] = []
    new_papers: list[str] = []
    for ordinal, rr in enumerate(resolved, start=1):
        rid = _reference_id(paper_id, markdown_hash, ordinal)
        status, source, target, conf = rr.status, rr.resolution_source, None, None
        if status == "resolved":
            record = dict(rr.candidates[0]) if rr.candidates else {}
            for k, v in rr.entry.to_record().items():
                record.setdefault(k, v)
            target, created, _items = _resolve_record(lib, record)
            if target is None or target == paper_id:
                status, source, target = "ambiguous", None, None
            else:
                conf = float(rr.confidence)
                if created:
                    new_papers.append(target)
        rows.append((rid, rr, status, source, target, conf))

    review: list[str] = []
    now = _now()
    with lib.write() as conn:
        if stored:  # the markdown changed: drop the old parse, keep what the researcher decided on
            conn.execute("DELETE FROM citation_links WHERE source_paper_id = ? AND provenance = 'parsed_bibliography'",
                         (paper_id,))
            conn.execute(
                "DELETE FROM reference_entries WHERE citing_paper_id = ? AND reference_id NOT IN "
                "(SELECT reference_id FROM citation_review_decisions) AND reference_id NOT IN "
                "(SELECT reference_id FROM citation_links WHERE reference_id IS NOT NULL)", (paper_id,))
        for ordinal, (rid, rr, status, source, target, conf) in enumerate(rows, start=1):
            conn.execute(
                "INSERT INTO reference_entries (reference_id, citing_paper_id, raw_reference_text, parsed_fields_json, "
                "resolved_paper_id, resolution_status, resolution_source, confidence, markdown_id, markdown_hash, "
                "section_label, ordinal, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(reference_id) DO UPDATE SET markdown_id = excluded.markdown_id, "
                "markdown_hash = excluded.markdown_hash",
                (rid, paper_id, rr.entry.raw, _parsed_fields_json(rr.entry), target, status, source, conf,
                 markdown_id, markdown_hash, rr.entry.section_label, ordinal, now))
            if target is not None:
                _link(conn, paper_id, target, "parsed_bibliography", conf or 0.0, rid)
            elif status in ("ambiguous", "suspect") and (rr.candidates or rr.rejected_record):
                decided = conn.execute("SELECT 1 FROM citation_review_decisions WHERE reference_id = ?",
                                       (rid,)).fetchone()
                if decided is None:
                    review.append(attention.raise_item(
                        lib, CITATION_REVIEW_KIND, {"reference_id": rid}, paper_id=paper_id,
                        detail=_review_detail(ordinal, rr, status)).item_id)
    return ParseResult(paper_id, "parsed", len(rows), sum(1 for r in rows if r[4] is not None), tuple(review),
                       tuple(dict.fromkeys(new_papers)))


def _review_detail(ordinal: int, rr: Any, status: str) -> str:
    raw = rr.entry.raw if len(rr.entry.raw) <= 160 else rr.entry.raw[:157] + "..."
    cand = (rr.rejected_record or (rr.candidates[0] if rr.candidates else None)) or {}
    why = (f"the closest provider match was rejected ({rr.reject_reason.replace('_', ' ')})"
           if status == "suspect" and rr.reject_reason else "no provider match carries a strong id")
    seen = f"; candidate: '{cand.get('title')}'" if cand.get("title") else ""
    return f"reference {ordinal} could not be matched with confidence ({why}): '{raw}'{seen}"


def _reference_id(paper_id: str, markdown_hash: str, ordinal: int) -> str:
    return "ref_" + hashlib.sha256(f"{paper_id}|{markdown_hash}|{ordinal}".encode()).hexdigest()[:24]
