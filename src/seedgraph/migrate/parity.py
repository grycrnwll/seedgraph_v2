"""Search parity between v2 and the migrated v3 library (design §8.2-§8.4).

Parity is defined on match **sets**, not ranks. For every query of the fixed set
(``migration/parity/queries.json``, researcher-owned, ``{graph slug: [query, ...]}``):

* **Capture** (:func:`capture`): run v2's search against the read-only v2 project
  that became the graph (``second_test`` -> ``diff_in_diff``, ``land_value`` ->
  ``land_value``), every query inside one read transaction, with no limit, and
  store ``migration/parity/v2_expected.json``. Matching uses v2's phrase-safe
  :func:`seedgraph.fts.search.build_match_query` for all three kinds (so the
  query semantics equal v3's ``search``). Spans: every ``span_fts`` hit. Claims and
  notes: v2's ``answer/retrieve`` baseline, i.e. ``CURRENT_CLAIMS_CTE`` plus the
  not-excluded filter (frozen here as :data:`V2_CURRENT_CLAIMS_CTE` and
  :data:`V2_NOT_EXCLUDED`, since that v2 code goes away in a later stage). Each hit
  keeps its attribution (v2 work ids, markdown id) for the classifier.
* **Check** (:func:`check`): v3 :func:`seedgraph.library.search.match_set` on the
  graph (agent audience), keyed like v2 (spans by ``(markdown_hash, start_char,
  end_char)``, claims and notes by their carried ids). Every difference is
  classified by a rule over the migration's own data into one of the six named
  classes (:data:`CLASSES`); anything unclassified is *unexplained* and fails.
  ``runs/<run_id>/parity.md`` and ``parity.json`` record, per query and kind, the
  matched, allowed (by class) and unexplained counts, plus top-20 overlap and
  Kendall tau (reported, never gated).

Classes (design §8.3), with the rule used:

* ``folded_in`` - an extra hit on a graph member that joined from test_proj
  (``graph_papers.origin`` is not the source project's).
* ``hidden`` - a missing note/claim hit on a paper with an open ``hides_notes``
  attention item, or on a labelled version.
* ``merged`` - a missing note/claim hit whose paper merged several v2 works, the
  current version coming from the same project; or a missing span whose paper's
  current markdown is another merged member's.
* ``current_elsewhere`` - the current version is another project's: the old one
  goes missing, the other project's appears.
* ``relinked`` - hits that changed because the paper was re-linked from a foreign
  markdown (the report's repair list): notes labelled ``other_markdown`` and spans
  on the markdown it pointed at.
* ``spans_from_other_projects`` - an extra span hit on a member's markdown,
  created by another v2 project.

The live library is only read; reading it never sets ``dirty``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from seedgraph.fts.search import build_match_query
from seedgraph.library import graphs as _graphs
from seedgraph.library.attention import hidden_paper_ids
from seedgraph.library.search import match_set, search
from seedgraph.library.store import Library, open_library_at, read_meta

from .plan import GRAPH_PROJECTS
from .snapshot import connect_v2_ro, read_db

QUERIES_FILE = "migration/parity/queries.json"
EXPECTED_FILE = "migration/parity/v2_expected.json"
CLASSES: tuple[str, ...] = ("folded_in", "hidden", "merged", "current_elsewhere", "relinked",
                            "spans_from_other_projects")
KINDS: tuple[str, ...] = ("spans", "claims", "notes")
TOP_K = 20

#: v2's read-time "current claims" CTE (``seedgraph.semantic.current``), frozen.
V2_CURRENT_CLAIMS_CTE = """
WITH current_notes AS (
    SELECT note_id FROM (
        SELECT note_id, ROW_NUMBER() OVER (
            PARTITION BY work_id, schema_id
            ORDER BY created_at DESC, rowid DESC
        ) AS rn FROM structured_notes
    ) WHERE rn = 1
),
current_lens_runs AS (
    SELECT extraction_run_id FROM (
        SELECT extraction_run_id, ROW_NUMBER() OVER (
            PARTITION BY work_id, lens_id
            ORDER BY created_at DESC, rowid DESC
        ) AS rn FROM extraction_runs WHERE lens_id IS NOT NULL
    ) WHERE rn = 1
),
current_claims AS (
    SELECT ec.*
    FROM extracted_claims ec
    LEFT JOIN extraction_runs er ON er.extraction_run_id = ec.extraction_run_id
    WHERE (ec.structured_note_id IS NOT NULL
           AND ec.structured_note_id IN (SELECT note_id FROM current_notes))
       OR (ec.structured_note_id IS NULL AND er.lens_id IS NOT NULL
           AND ec.extraction_run_id
               IN (SELECT extraction_run_id FROM current_lens_runs))
       OR (ec.structured_note_id IS NULL AND er.lens_id IS NULL)
)
"""
#: v2's ``answer/retrieve._NOT_EXCLUDED`` filter, frozen.
V2_NOT_EXCLUDED = (
    " NOT IN (SELECT pd.work_id FROM project_documents pd "
    "WHERE pd.inclusion_status = 'excluded')"
)

_V2_SPANS = ("SELECT e.markdown_hash, e.start_char, e.end_char, e.span_id, e.work_id, e.markdown_id, "
             "bm25(span_fts) AS score FROM span_fts JOIN evidence_spans e ON e.span_id = span_fts.span_id "
             "WHERE span_fts MATCH ? ORDER BY score, e.span_id")
_V2_CLAIMS = (V2_CURRENT_CLAIMS_CTE + "SELECT c.claim_id, c.work_id, bm25(claim_fts) AS score "
              "FROM claim_fts JOIN current_claims c ON c.claim_id = claim_fts.claim_id "
              "WHERE claim_fts MATCH ? AND c.work_id" + V2_NOT_EXCLUDED + " ORDER BY score, c.claim_id")
_V2_NOTES = (V2_CURRENT_CLAIMS_CTE + "SELECT n.note_id, n.work_id, bm25(note_fts) AS score "
             "FROM note_fts JOIN structured_notes n ON n.note_id = note_fts.note_id "
             "WHERE note_fts MATCH ? AND n.note_id IN (SELECT note_id FROM current_notes) "
             "AND n.work_id" + V2_NOT_EXCLUDED + " ORDER BY score, n.note_id")

STOPWORDS = frozenset("""
a about above after again against all also although among an and any are as at be because been before being
below between both but by can could did do does doing down during each either et etc few for from further
had has have having here how however if in into is it its itself just less may more most much must no nor
not of off on once only or other our out over own per same shall should since so some such than that the
their them then there these they this those through thus to too under until up upon very via was we were
what when where whether which while who whom why will with within without would yet you your paper papers
study studies find finds found show shows shown using used use result results effect effects table figure
section data model models analysis approach based first second third one two three new also
""".split())


class ParityError(RuntimeError):
    """Parity cannot run (missing queries, capture or library), or the inputs disagree."""


# --- files ----------------------------------------------------------------------------------

def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _validate_queries(data: Any, where: str) -> dict[str, list[str]]:
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, list) for k, v in data.items()):
        raise ParityError(f"{where}: expected a JSON object {{graph slug: [query, ...]}}")
    out = {}
    for slug, qs in data.items():
        clean = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
        out[slug] = list(dict.fromkeys(clean))
    return out


def load_queries(v3_home: Path | str) -> dict[str, list[str]]:
    """The researcher's query set, ``{graph slug: [query, ...]}``."""
    path = Path(v3_home) / QUERIES_FILE
    if not path.is_file():
        raise ParityError(f"no parity queries at {path}: seed them with `seedgraph migrate parity --seed-queries` "
                          "(seedgraph.migrate.parity.seed_queries) and edit the list")
    return _validate_queries(json.loads(path.read_text(encoding="utf-8")), str(path))


def write_queries(v3_home: Path | str, queries: Mapping[str, Sequence[str]], *, force: bool = False) -> Path:
    """Write ``queries.json``; refuses to overwrite the researcher's file unless ``force``."""
    path = Path(v3_home) / QUERIES_FILE
    if path.exists() and not force:
        raise ParityError(f"{path} exists (researcher-owned); pass force=True to replace it")
    data = _validate_queries({k: list(v) for k, v in queries.items()}, "queries")
    _write(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path


def load_expected(v3_home: Path | str) -> dict[str, Any]:
    path = Path(v3_home) / EXPECTED_FILE
    if not path.is_file():
        raise ParityError(f"no captured v2 results at {path}: run the parity capture first "
                          "(`seedgraph migrate parity --capture`)")
    return json.loads(path.read_text(encoding="utf-8"))


# --- default query set ----------------------------------------------------------------------

_WORD = re.compile(r"\w+", re.UNICODE)


def _content(tok: str) -> bool:
    return len(tok) >= 4 and tok.isalpha() and tok not in STOPWORDS


def default_queries(v2_home: Path | str, *, per_graph: int = 40) -> dict[str, list[str]]:
    """A starting query set per graph, from the v2 project's own text, deterministically.

    Roughly a third each: the most frequent claim labels, the most frequent adjacent
    content-word pairs, and the most frequent content words of span quotes and
    claim texts (ties broken alphabetically). The researcher edits the result.
    """
    v2_home = Path(v2_home)
    out: dict[str, list[str]] = {}
    for project, slug in GRAPH_PROJECTS.items():
        db = v2_home / "projects" / project / "project.db"
        if not db.is_file():
            continue
        conn = connect_v2_ro(db)
        try:
            conn.execute("BEGIN")
            labels = [r[0] for r in conn.execute("SELECT normalized_label FROM extracted_claims "
                                                 "WHERE normalized_label IS NOT NULL")]
            texts = [r[0] for r in conn.execute("SELECT exact_quote FROM evidence_spans WHERE exact_quote IS NOT NULL")]
            texts += [r[0] for r in conn.execute("SELECT claim_text FROM extracted_claims WHERE claim_text IS NOT NULL")]
            conn.execute("COMMIT")
        finally:
            conn.close()
        words: Counter[str] = Counter()
        pairs: Counter[str] = Counter()
        for t in texts:
            toks = [w.lower() for w in _WORD.findall(t)]
            words.update(w for w in toks if _content(w))
            pairs.update(f"{a} {b}" for a, b in zip(toks, toks[1:]) if _content(a) and _content(b))
        lab = Counter(" ".join(_WORD.findall(x.lower())) for x in labels)

        def top(c: Counter[str], n: int, min_count: int = 2) -> list[str]:
            return [k for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])) if k and v >= min_count][:n]

        third = max(1, per_graph // 3)
        picked: list[str] = []
        for cand in (top(lab, third), top(pairs, third), top(words, per_graph)):
            for q in cand:
                if q not in picked and len(picked) < per_graph:
                    picked.append(q)
        out[slug] = picked
    return out


def seed_queries(v2_home: Path | str, v3_home: Path | str, *, per_graph: int = 40, force: bool = False) -> Path:
    """Write :func:`default_queries` to ``queries.json`` (never over the researcher's file
    unless ``force``)."""
    path = Path(v3_home) / QUERIES_FILE
    if path.exists() and not force:
        raise ParityError(f"{path} exists (researcher-owned); pass force=True to replace it")
    return write_queries(v3_home, default_queries(v2_home, per_graph=per_graph), force=True)


# --- capture ----------------------------------------------------------------------------------

def capture_v2(v2_home: Path | str, queries: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    """v2 match sets (unlimited, in v2 rank order) for every graph's queries."""
    v2_home = Path(v2_home)
    graphs_out: dict[str, Any] = {}
    for project, slug in GRAPH_PROJECTS.items():
        qs = list(queries.get(slug, []))
        db = v2_home / "projects" / project / "project.db"
        if not qs or not db.is_file():
            continue
        rel = f"projects/{project}/project.db"
        per_q: dict[str, Any] = {}
        conn = connect_v2_ro(db)
        try:
            conn.execute("BEGIN")
            for q in qs:
                if not q.strip():
                    continue
                m = build_match_query(q)
                spans: dict[tuple, dict[str, Any]] = {}
                for mh, s, e, sid, wid, mid, _ in conn.execute(_V2_SPANS, (m,)):
                    ent = spans.setdefault((mh, s, e), {"key": [mh, s, e], "markdown_id": mid, "span_ids": [],
                                                        "work_ids": []})
                    ent["span_ids"].append(sid)
                    if wid not in ent["work_ids"]:
                        ent["work_ids"].append(wid)
                claims = [{"id": cid, "work_id": wid} for cid, wid, _ in conn.execute(_V2_CLAIMS, (m,))]
                notes = [{"id": nid, "work_id": wid} for nid, wid, _ in conn.execute(_V2_NOTES, (m,))]
                per_q[q] = {"spans": list(spans.values()), "claims": claims, "notes": notes}
            conn.execute("COMMIT")
        finally:
            conn.close()
        graphs_out[slug] = {"project": project, "v2_logical_hash": read_db(db, rel).logical_hash,
                            "queries": per_q}
    return {"version": 1, "match": "seedgraph.fts.search.build_match_query (phrase-safe, all kinds)",
            "claims_notes_baseline": "v2 answer/retrieve CURRENT_CLAIMS_CTE + not-excluded",
            "graphs": graphs_out}


def capture(v2_home: Path | str, v3_home: Path | str, *, queries: Mapping[str, Sequence[str]] | None = None) -> Path:
    """Capture v2 results for ``queries`` (default: ``queries.json``) into ``v2_expected.json``."""
    qs = queries if queries is not None else load_queries(v3_home)
    data = capture_v2(v2_home, qs)
    path = Path(v3_home) / EXPECTED_FILE
    _write(path, json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return path


def expected_is_current(v2_home: Path | str, expected: Mapping[str, Any],
                        queries: Mapping[str, Sequence[str]] | None = None) -> bool:
    """True when the capture was taken from the v2 data as it is now (and for ``queries``)."""
    v2_home = Path(v2_home)
    for slug, g in expected.get("graphs", {}).items():
        db = v2_home / "projects" / g["project"] / "project.db"
        if not db.is_file() or read_db(db, f"projects/{g['project']}/project.db").logical_hash != g["v2_logical_hash"]:
            return False
    if queries is not None:
        captured = {slug: sorted(g["queries"]) for slug, g in expected.get("graphs", {}).items()}
        wanted = {slug: sorted(q) for slug, q in queries.items() if q and slug in GRAPH_PROJECTS.values()}
        if captured != wanted:
            return False
    return True


# --- check -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Difference:
    graph: str
    query: str
    kind: str
    key: Any                  # (markdown_hash, start, end) for spans, else the claim/note id
    direction: str            # 'extra' (v3 only) | 'missing' (v2 only)
    cls: str | None           # one of CLASSES, or None when unexplained
    detail: str


@dataclass
class ParityResult:
    run_id: str | None
    differences: list[Difference] = field(default_factory=list)
    queries: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def unexplained(self) -> list[Difference]:
        return [d for d in self.differences if d.cls is None]

    @property
    def ok(self) -> bool:
        return not self.unexplained

    def by_class(self) -> dict[str, int]:
        return dict(sorted(Counter(d.cls or "unexplained" for d in self.differences).items()))


class _Context:
    """The migration's own data the classifier reads, for one graph."""

    def __init__(self, lib: Library, slug: str, project: str, relinks: Sequence[Mapping[str, Any]]):
        self.lib, self.slug, self.source = lib, slug, f"v2:{project}"
        self.project = project
        conn = lib.conn
        with _graphs.open_graph(lib, slug) as g:
            everyone = _graphs.membership(g)
        self.members = {p.paper_id: p for p in everyone if p.status == "member"}
        self.statuses = {p.paper_id: p.status for p in everyone}
        self.work_paper = {r[0]: r[1] for r in conn.execute(
            "SELECT v2_id, v3_id FROM v2_id_map WHERE v2_project = ? AND v2_table = 'works'", (project,))}
        self.md_hash = {r[0]: r[1] for r in conn.execute("SELECT markdown_id, markdown_hash FROM markdown_documents")}
        self.current_md: dict[str, str] = {}
        for pid, mid in conn.execute("SELECT paper_id, markdown_id FROM paper_files WHERE markdown_id IS NOT NULL "
                                     "ORDER BY paper_id, role, source_file_id"):
            self.current_md.setdefault(pid, mid)
        self.on_hash: dict[str, set[str]] = {}
        for pid, mid in self.current_md.items():
            if pid in self.members:
                self.on_hash.setdefault(self.md_hash.get(mid, ""), set()).add(pid)
        self.merged = {r[0] for r in conn.execute("SELECT DISTINCT paper_id FROM paper_merge_members")}
        self.relinks = {r["paper_id"]: r for r in relinks}
        self.hidden = hidden_paper_ids(conn)

    def folded(self, pid: str) -> bool:
        p = self.members.get(pid)
        return p is not None and p.origin != self.source

    def spans_at(self, key: Sequence[Any]) -> list[tuple[str, str, str]]:
        return [tuple(r) for r in self.lib.conn.execute(
            "SELECT span_id, origin, markdown_id FROM evidence_spans WHERE markdown_hash = ? AND start_char = ? "
            "AND end_char = ? ORDER BY span_id", tuple(key))]

    def item(self, kind: str, item_id: str) -> dict[str, Any] | None:
        conn = self.lib.conn
        if kind == "notes":
            r = conn.execute("SELECT n.paper_id, n.note_id, n.schema_id, n.version_label, n.origin, "
                             "n.note_id IN (SELECT note_id FROM note_current) FROM structured_notes n "
                             "WHERE n.note_id = ?", (item_id,)).fetchone()
        else:
            r = conn.execute(
                "SELECT c.paper_id, c.structured_note_id, n.schema_id, n.version_label, c.origin, "
                "CASE WHEN c.structured_note_id IS NOT NULL "
                "     THEN c.structured_note_id IN (SELECT note_id FROM note_current) "
                "     WHEN er.lens_id IS NOT NULL THEN c.extraction_run_id = ("
                "       SELECT o.extraction_run_id FROM extraction_runs o WHERE o.paper_id = er.paper_id "
                "       AND o.lens_id = er.lens_id ORDER BY o.created_at DESC, o.extraction_run_id DESC LIMIT 1) "
                "     ELSE 1 END, er.lens_id "
                "FROM extracted_claims c LEFT JOIN structured_notes n ON n.note_id = c.structured_note_id "
                "LEFT JOIN extraction_runs er ON er.extraction_run_id = c.extraction_run_id "
                "WHERE c.claim_id = ?", (item_id,)).fetchone()
        if r is None:
            return None
        return {"paper_id": r[0], "note_id": r[1], "schema_id": r[2], "label": r[3], "origin": r[4],
                "current": bool(r[5]), "lens_id": r[6] if kind == "claims" and r[1] is None else None}

    def current_lens_run(self, pid: str, lens_id: str) -> tuple[str, str] | None:
        """v3's current lens run for ``(paper, lens)`` (as search picks it): id and origin."""
        r = self.lib.conn.execute("SELECT extraction_run_id, origin FROM extraction_runs WHERE paper_id = ? "
                                  "AND lens_id = ? ORDER BY created_at DESC, extraction_run_id DESC LIMIT 1",
                                  (pid, lens_id)).fetchone()
        return None if r is None else (r[0], r[1])

    def current_note(self, pid: str, schema_id: Any) -> tuple[str, str] | None:
        r = self.lib.conn.execute("SELECT note_id, origin FROM note_current WHERE paper_id = ? AND schema_id IS ? "
                                  "ORDER BY note_id LIMIT 1", (pid, schema_id)).fetchone()
        return None if r is None else (r[0], r[1])

    # rules ------------------------------------------------------------------------------
    def span_extra(self, key: Sequence[Any]) -> tuple[str | None, str]:
        papers = self.on_hash.get(key[0], set())
        if papers and all(self.folded(p) for p in papers):
            return "folded_in", f"on folded-in paper(s) {', '.join(sorted(papers))}"
        for p in sorted(papers):
            rl = self.relinks.get(p)
            if rl and rl.get("to_markdown_id") and self.md_hash.get(rl["to_markdown_id"]) == key[0]:
                return "relinked", f"paper {p} was re-linked to this markdown"
        spans = self.spans_at(key)
        if spans and all(origin != self.source for _, origin, _ in spans):
            return "spans_from_other_projects", f"span {spans[0][0]} created in {spans[0][1]}"
        if not spans:
            return None, "no such span in the library"
        return None, f"span {spans[0][0]} ({spans[0][1]}) on paper(s) {', '.join(sorted(papers)) or 'none'}"

    def span_missing(self, key: Sequence[Any], entry: Mapping[str, Any]) -> tuple[str | None, str]:
        pids = [self.work_paper.get(w) for w in entry.get("work_ids", [])]
        for p in pids:
            rl = self.relinks.get(p) if p else None
            if rl and rl.get("from_markdown_id") == entry.get("markdown_id"):
                return "relinked", f"paper {p} pointed at this markdown in v2 and was re-linked ({rl.get('repair')})"
        for p in pids:
            if p and p in self.members and p in self.merged and \
                    self.md_hash.get(self.current_md.get(p, ""), None) != key[0]:
                return "merged", f"paper {p} merged several v2 works; its current markdown is another member's"
        def why(p: str | None) -> str:
            if p is None:
                return "not carried"
            if p not in self.members:
                status = self.statuses.get(p)
                return f"{p} ({status} in {self.slug}, not a member)" if status else f"{p} (not in {self.slug})"
            cur = self.current_md.get(p)
            if cur is None:
                return f"{p} (member with no current markdown)"
            return f"{p} (v2 markdown {entry.get('markdown_id')} != current {cur})"
        where = ", ".join(f"{w} -> {why(p)}" for w, p in zip(entry.get("work_ids", []), pids))
        return None, f"v2 works {where}"

    def item_extra(self, kind: str, item_id: str) -> tuple[str | None, str]:
        it = self.item(kind, item_id)
        if it is None:
            return None, "not in the library"
        if self.folded(it["paper_id"]):
            return "folded_in", f"paper {it['paper_id']} folded in ({self.members[it['paper_id']].origin})"
        if it["origin"] != self.source and it["current"]:
            return "current_elsewhere", f"current version comes from {it['origin']}"
        return None, f"paper {it['paper_id']}, origin {it['origin']}, current={it['current']}"

    def item_missing(self, kind: str, item_id: str) -> tuple[str | None, str]:
        it = self.item(kind, item_id)
        if it is None:
            return None, "not carried into the library"
        p = it["paper_id"]
        if it["label"] == "other_markdown" and p in self.relinks:
            return "relinked", f"made on another PDF's text; paper {p} re-linked"
        if p in self.hidden or it["label"] is not None:
            why = f"labelled version ({it['label']})" if it["label"] else "paper is on the attention list (notes hidden)"
            return "hidden", why
        if p not in self.members:
            status = self.statuses.get(p)
            return None, (f"paper {p} is {status} in {self.slug}, not a member" if status
                          else f"paper {p} is not in {self.slug}")
        if not it["current"] and it.get("lens_id"):
            run = self.current_lens_run(p, it["lens_id"])
            if run is not None and run[1] != self.source:
                return "current_elsewhere", f"current {it['lens_id']} run {run[0]} comes from {run[1]}"
            if run is not None and p in self.merged:
                return "merged", f"paper {p} merged several v2 works; current {it['lens_id']} run is {run[0]}"
            return None, f"lens claim of a superseded run on paper {p}"
        if not it["current"]:
            cur = self.current_note(p, it["schema_id"])
            if cur is None:
                return "hidden", "no unlabelled version is current"
            if cur[1] != self.source:
                return "current_elsewhere", f"current version {cur[0]} comes from {cur[1]}"
            if p in self.merged:
                return "merged", f"paper {p} merged several v2 works; current version is {cur[0]}"
        return None, f"paper {p}, current={it['current']}"


def _kendall_tau(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    common = [x for x in a if x in set(b)]
    if len(common) < 2:
        return None
    pos = {x: i for i, x in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            if pos[common[i]] < pos[common[j]]:
                conc += 1
            else:
                disc += 1
    return round((conc - disc) / (conc + disc), 4)


def _norm_key(kind: str, key: Any) -> Any:
    return tuple(key) if kind == "spans" else key


def compare(lib: Library, expected: Mapping[str, Any], *, relinks: Sequence[Mapping[str, Any]] = (),
            run_id: str | None = None) -> ParityResult:
    """Compare v3 graph-scoped match sets with the captured v2 ones (pure over ``lib``)."""
    result = ParityResult(run_id=run_id)
    for slug in sorted(expected.get("graphs", {})):
        g = expected["graphs"][slug]
        if not _graphs.graph_db_path(lib, slug).is_file():
            raise ParityError(f"graph {slug!r} is not in the library")
        ctx = _Context(lib, slug, g["project"], relinks)
        for q in sorted(g["queries"]):
            row: dict[str, Any] = {"graph": slug, "query": q}
            for kind in KINDS:
                v2_entries = g["queries"][q].get(kind, [])
                if kind == "spans":
                    v2 = {tuple(e["key"]): e for e in v2_entries}
                    v2_order = [tuple(e["key"]) for e in v2_entries]
                else:
                    v2 = {e["id"]: e for e in v2_entries}
                    v2_order = [e["id"] for e in v2_entries]
                v3 = set(match_set(lib, q, kind=kind, graph=slug))
                v3_top = [_norm_key(kind, h.key) for h in search(lib, q, kind=kind, graph=slug, limit=TOP_K)]
                allowed: Counter[str] = Counter()
                unexplained = 0
                for key in sorted(v3 - set(v2), key=str):
                    cls, detail = ctx.span_extra(key) if kind == "spans" else ctx.item_extra(kind, key)
                    result.differences.append(Difference(slug, q, kind, key, "extra", cls, detail))
                    if cls is None:
                        unexplained += 1
                    else:
                        allowed[cls] += 1
                for key in sorted(set(v2) - v3, key=str):
                    cls, detail = (ctx.span_missing(key, v2[key]) if kind == "spans"
                                   else ctx.item_missing(kind, key))
                    result.differences.append(Difference(slug, q, kind, key, "missing", cls, detail))
                    if cls is None:
                        unexplained += 1
                    else:
                        allowed[cls] += 1
                top_v2 = v2_order[:TOP_K]
                row[kind] = {"v2": len(v2), "v3": len(v3), "matched": len(v3 & set(v2)),
                             "allowed": dict(sorted(allowed.items())),
                             "unexplained": unexplained,
                             "top20_overlap": len(set(top_v2) & set(v3_top)),
                             "kendall_tau": _kendall_tau(top_v2, v3_top)}
            result.queries.append(row)
    return result


def _relinks_for(v3_home: Path, run_id: str | None, notes: list[str]) -> list[dict[str, Any]]:
    if run_id:
        p = v3_home / "migration" / "runs" / run_id / "report.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")).get("repairs", {}).get("relinks", [])
    notes.append("no report.json for the live library's run: re-linked papers are not known, so 'relinked' "
                 "differences cannot be classified")
    return []


def render_parity_md(res: ParityResult) -> str:
    L = [f"# Search parity: {res.run_id or 'live library'}", "",
         f"**Result: {'PASSED' if res.ok else 'FAILED'}.** {len(res.differences)} differences between v2 and v3 "
         f"match sets; {len(res.unexplained)} unexplained.", "",
         "Parity compares the full sets of matches (not their order). Each difference must belong to a named "
         "known-difference class; anything else is unexplained and fails. Top-20 overlap and Kendall tau are for "
         "information only.", ""]
    L += [f"- {n}" for n in res.notes]
    if res.notes:
        L.append("")
    L += ["## Differences by class", ""]
    L += [f"- {k}: {n}" for k, n in res.by_class().items()] or ["None."]
    L += ["", "## Per query", "",
          "| graph | query | kind | v2 | v3 | matched | allowed (by class) | unexplained | top-20 overlap | tau |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for row in res.queries:
        for kind in KINDS:
            k = row[kind]
            allowed = ", ".join(f"{c} {n}" for c, n in k["allowed"].items()) or "-"
            q = row["query"].replace("|", "\\|")
            L.append(f"| {row['graph']} | {q} | {kind} | {k['v2']} | {k['v3']} | {k['matched']} | {allowed} | "
                     f"{k['unexplained']} | {k['top20_overlap']} | {'' if k['kendall_tau'] is None else k['kendall_tau']} |")
    if res.unexplained:
        L += ["", "## Unexplained differences", ""]
        for d in res.unexplained:
            L.append(f"- {d.graph} / {d.query!r} / {d.kind}: {d.direction} `{d.key}`: {d.detail}")
    return "\n".join(L) + "\n"


def check(v3_home: Path | str, *, run_id: str | None = None, expected: Mapping[str, Any] | None = None,
          write: bool = True) -> ParityResult:
    """Run the parity check on the live library of ``v3_home``.

    ``expected`` defaults to ``v2_expected.json``; ``run_id`` to the live library's
    migration run. Writes ``runs/<run_id>/parity.md`` and ``parity.json`` unless
    ``write=False``. Raises :class:`ParityError` when inputs are missing or the
    query set changed since the capture.
    """
    v3_home = Path(v3_home)
    if expected is None:
        expected = load_expected(v3_home)
        qpath = v3_home / QUERIES_FILE
        if qpath.is_file():
            queries = load_queries(v3_home)
            captured = {s: sorted(g["queries"]) for s, g in expected.get("graphs", {}).items()}
            wanted = {s: sorted(q) for s, q in queries.items() if q and s in GRAPH_PROJECTS.values()}
            if captured != wanted:
                raise ParityError("queries.json changed since the v2 capture: run the capture again")
    if not (v3_home / "library" / "library.db").is_file():
        raise ParityError(f"no live library under {v3_home}: run the migration first")
    lib = open_library_at(v3_home / "library")
    try:
        run_id = run_id or read_meta(lib.conn).get("migration_run_id")
        notes: list[str] = []
        relinks = _relinks_for(v3_home, run_id, notes)
        res = compare(lib, expected, relinks=relinks, run_id=run_id)
        res.notes += notes
    finally:
        lib.close()
    if write and run_id:
        run_dir = v3_home / "migration" / "runs" / run_id
        res.path = run_dir / "parity.md"
        _write(res.path, render_parity_md(res))
        doc = {"run_id": run_id, "ok": res.ok, "by_class": res.by_class(), "notes": res.notes,
               "queries": res.queries,
               "differences": [{"graph": d.graph, "query": d.query, "kind": d.kind,
                                "key": list(d.key) if isinstance(d.key, tuple) else d.key,
                                "direction": d.direction, "class": d.cls, "detail": d.detail}
                               for d in res.differences]}
        _write(run_dir / "parity.json", json.dumps(doc, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return res


__all__ = ["CLASSES", "EXPECTED_FILE", "QUERIES_FILE", "Difference", "ParityError", "ParityResult", "capture",
           "capture_v2", "check", "compare", "default_queries", "expected_is_current", "load_expected",
           "load_queries", "seed_queries", "write_queries"]
