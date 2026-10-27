"""Concept search (stage 4 design §5.3-§5.4): one interface, two modes, sense-level hits.

    find(lib, query: str | SenseKey, *, family=None, graph=None, k=20,
         mode='auto' | 'meaning' | 'keyword', embedder=None) -> list[SenseHit]

- **keyword** (always present; the fallback): FTS5 ``claim_fts`` (label, claim text,
  plain words). A text query becomes an OR of its content words, each quoted, so FTS5
  syntax in the query is inert. A sense query (or a claim set, :func:`keyword_terms_for_claims`)
  is its label's words plus the top distinctive terms of its claims (tf over the set x
  idf over every supported claim). Claim hits (``bm25`` with the label column weighted
  :data:`LABEL_WEIGHT`) are aggregated to senses by **best** score. Only supporting
  claims count.
- **meaning**: cosine of the query vector against sense vectors (:mod:`.vectors`), built
  from the cached claim vectors of the embedder's model. A text query is embedded with
  the ``search_query:`` prefix; a sense query uses its own sense vector.
- **auto** uses meaning only when embeddings are adopted (:func:`adoption`, a
  ``library_meta`` row written after a passing adoption test) and it works; otherwise
  keyword. A meaning request that cannot run (Ollama down, no vectors yet) falls back to
  keyword silently: every hit says ``mode='keyword'`` and ``fallback`` says why.

The lower-level helpers (:func:`keyword_claim_scores`, :func:`keyword_terms_for_claims`,
:func:`text_terms`) are what the adoption test (:mod:`.search_eval`) ranks claim sets with.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from ..semantic.merge import normalize
from . import vectors as vx
from .model import DEFAULT_SENSE, SenseKey, handle, label_key

if TYPE_CHECKING:
    from ..library.store import Library

Mode = Literal["auto", "meaning", "keyword"]
ADOPTION_META_KEY = "concept_search_embeddings"
LABEL_WEIGHT = 2.0
DISTINCTIVE_TERMS = 8
#: Function words dropped from keyword queries (FTS5 has no stoplist of its own).
STOPWORDS: frozenset[str] = frozenset("""
a an and are as at be been but by can do does for from has have if in into is it its may more most no not
of on or our so such than that the their then there these they this those to under was we were when where
which while with without would also using use used via each other between both all any some only over
""".split())


# --- adoption -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Adoption:
    adopted: bool = False
    model: str = vx.DEFAULT_MODEL
    report: str | None = None
    decided_at: str | None = None


def adoption(conn: sqlite3.Connection) -> Adoption:
    row = conn.execute("SELECT value FROM library_meta WHERE key = ?", (ADOPTION_META_KEY,)).fetchone()
    if row is None or not row[0]:
        return Adoption()
    d = json.loads(row[0])
    return Adoption(bool(d.get("adopted")), d.get("model") or vx.DEFAULT_MODEL, d.get("report"), d.get("decided_at"))


def set_adoption(lib: Library, *, adopted: bool, model: str = vx.DEFAULT_MODEL, report: str | None = None) -> None:
    """Record the adoption verdict (only a passing adoption test should set ``adopted``)."""
    value = json.dumps({"adopted": adopted, "model": model, "report": report,
                        "decided_at": datetime.now(timezone.utc).isoformat()}, sort_keys=True)
    with lib.write() as conn:
        conn.execute("INSERT OR REPLACE INTO library_meta (key, value) VALUES (?, ?)", (ADOPTION_META_KEY, value))


# --- hits ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SenseHit:
    key: SenseKey
    handle: str
    concept_id: str | None
    name: str | None
    gloss: str | None
    paper_count: int
    claim_count: int
    score: float
    mode: str  # 'meaning' | 'keyword'
    best_claim_id: str | None
    best_claim_text: str | None
    best_paper_id: str | None
    fallback: str | None = None  # why a meaning request ran as keyword


# --- keyword ------------------------------------------------------------------------------------

def text_terms(text: str) -> list[str]:
    """The content words of ``text`` (normalised, stopwords and 1-char tokens dropped), in order."""
    out: list[str] = []
    for t in normalize(text or "").split():
        if len(t) > 1 and t not in STOPWORDS and t not in out:
            out.append(t)
    return out


def match_or(terms: Iterable[str]) -> str:
    """An FTS5 MATCH string: each term quoted, OR'ed."""
    return " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)


def _in(col: str, values: list[str], params: list) -> str:
    params.extend(values)
    return f"{col} IN ({', '.join('?' * len(values))})"


def keyword_claim_scores(conn: sqlite3.Connection, terms: Iterable[str], *, families: Iterable[str] | None = None,
                         papers: Iterable[str] | None = None) -> dict[str, float]:
    """``claim_id -> score`` (higher is better: ``-bm25``) over supporting claims."""
    terms = list(terms)
    if not terms:
        return {}
    params: list = [match_or(terms)]
    where = ["claim_fts MATCH ?", "k.supports = 1"]
    if families is not None:
        where.append(_in("k.family", sorted(set(families)), params))
    if papers is not None:
        where.append(_in("k.paper_id", sorted(set(papers)), params))
    sql = (f"SELECT claim_fts.claim_id, bm25(claim_fts, 0.0, {LABEL_WEIGHT}, 1.0, 1.0) FROM claim_fts "
           "JOIN claim_concepts k ON k.claim_id = claim_fts.claim_id "
           f"WHERE {' AND '.join(where)}")
    return {cid: -float(score) for cid, score in conn.execute(sql, params)}


_DF_CACHE: dict[tuple, tuple[int, Counter]] = {}


def _fingerprint(conn: sqlite3.Connection) -> tuple:
    db = conn.execute("PRAGMA database_list").fetchone()[2]
    built = conn.execute("SELECT MAX(built_at), COUNT(*) FROM concept_build_state").fetchone()
    return (db, built[0], built[1], conn.execute("SELECT COUNT(*) FROM claim_concepts").fetchone()[0])


def _document_frequency(conn: sqlite3.Connection) -> tuple[int, Counter]:
    key = _fingerprint(conn)
    hit = _DF_CACHE.get(key)
    if hit is None:
        df: Counter = Counter()
        n = 0
        for row in conn.execute("SELECT f.normalized_label, f.claim_text, f.plain_words FROM claim_fts f "
                                "JOIN claim_concepts k ON k.claim_id = f.claim_id WHERE k.supports = 1"):
            n += 1
            df.update(set(text_terms(" ".join(x or "" for x in row))))
        hit = _DF_CACHE[key] = (n, df)
    return hit


def keyword_terms_for_claims(conn: sqlite3.Connection, claim_ids: Iterable[str], labels: Iterable[str] = (),
                             *, n: int = DISTINCTIVE_TERMS) -> list[str]:
    """A claim set's keyword query: its labels' words, then its ``n`` most distinctive
    terms (tf in the set's label + text + plain words x idf over every supported claim;
    ties by term)."""
    ids = sorted(set(claim_ids))
    terms: list[str] = []
    for lab in sorted(set(labels)):
        terms += [t for t in text_terms(lab) if t not in terms]
    if not ids:
        return terms
    tf: Counter = Counter()
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for row in conn.execute(f"SELECT normalized_label, claim_text, plain_words FROM claim_fts WHERE claim_id IN "
                                f"({', '.join('?' * len(chunk))})", chunk):
            tf.update(text_terms(" ".join(x or "" for x in row)))
    total, df = _document_frequency(conn)
    scored = sorted(((-c * math.log((total + 1) / (df.get(t, 0) + 1)), t) for t, c in tf.items() if t not in terms))
    return terms + [t for _, t in scored[:n]]


# --- senses -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _SenseInfo:
    concept_id: str | None
    name: str | None
    gloss: str | None
    paper_count: int
    claim_count: int


def _sense_info(conn: sqlite3.Connection, keys: Iterable[SenseKey]) -> dict[SenseKey, _SenseInfo]:
    keys = set(keys)
    out: dict[SenseKey, _SenseInfo] = {}
    if not keys:
        return out
    counts = {(r[0], r[1], r[2]): (r[3], r[4]) for r in conn.execute(
        "SELECT family, label_key, sense_id, COUNT(DISTINCT paper_id), COUNT(*) FROM claim_concepts "
        "WHERE supports = 1 AND sense_id IS NOT NULL GROUP BY family, label_key, sense_id")}
    rows = conn.execute(
        "SELECT m.family, m.label_key, m.sense_id, m.concept_id, c.name, s.gloss FROM concept_members m "
        "LEFT JOIN concepts c ON c.concept_id = m.concept_id LEFT JOIN concept_senses s ON s.family = m.family "
        "AND s.label_key = m.label_key AND s.sense_id = m.sense_id").fetchall()
    meta = {SenseKey(r[0], r[1], r[2]): (r[3], r[4], r[5]) for r in rows}
    for k in keys:
        cid, name, gloss = meta.get(k, (None, None, None))
        pc, cc = counts.get(tuple(k), (0, 0))
        out[k] = _SenseInfo(cid, name, gloss, pc, cc)
    return out


def _claim_senses(conn: sqlite3.Connection, *, families: Iterable[str] | None = None,
                  papers: Iterable[str] | None = None) -> dict[str, SenseKey]:
    params: list = []
    where = ["supports = 1", "sense_id IS NOT NULL"]
    if families is not None:
        where.append(_in("family", sorted(set(families)), params))
    if papers is not None:
        where.append(_in("paper_id", sorted(set(papers)), params))
    return {r[0]: SenseKey(r[1], r[2], r[3]) for r in conn.execute(
        f"SELECT claim_id, family, label_key, sense_id FROM claim_concepts WHERE {' AND '.join(where)}", params)}


def _claim_texts(conn: sqlite3.Connection, ids: Iterable[str]) -> dict[str, tuple[str, str]]:
    ids = sorted(set(ids))
    out: dict[str, tuple[str, str]] = {}
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        for r in conn.execute(f"SELECT claim_id, claim_text, paper_id FROM current_claims WHERE claim_id IN "
                              f"({', '.join('?' * len(chunk))})", chunk):
            out[r[0]] = (r[1], r[2])
    return out


def _sense_claims(conn: sqlite3.Connection, key: SenseKey) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT claim_id FROM claim_concepts WHERE family = ? AND label_key = ? AND sense_id = ? AND supports = 1",
        tuple(key))]


def _surface(conn: sqlite3.Connection, key: SenseKey) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT cc.normalized_label FROM claim_concepts k JOIN current_claims cc ON cc.claim_id = k.claim_id "
        "WHERE k.family = ? AND k.label_key = ? AND k.sense_id = ?", tuple(key))] or [key.label_key]


# --- meaning ------------------------------------------------------------------------------------

_SENSE_CACHE: dict[tuple, dict[SenseKey, list[float]]] = {}


class _NoVectors(Exception):
    pass


def library_sense_vectors(conn: sqlite3.Connection, model: str) -> dict[SenseKey, list[float]]:
    """Sense vectors from the cached claim vectors of ``model`` (loaded once per refresh
    fingerprint and vector count)."""
    n_vec = conn.execute("SELECT COUNT(*) FROM concept_vectors WHERE model = ?", (model,)).fetchone()[0]
    key = (*_fingerprint(conn), model, n_vec)
    hit = _SENSE_CACHE.get(key)
    if hit is None:
        contexts = vx.claim_contexts(conn)
        vecs = vx.cached_vectors(conn, [c.text for c in contexts], model)
        hit = _SENSE_CACHE[key] = vx.sense_vectors(contexts, vecs)
    return hit


def _meaning(lib: Library, query: str | SenseKey, embedder: vx.Embedder,
             allowed: set[SenseKey] | None) -> list[tuple[SenseKey, float]]:
    senses = library_sense_vectors(lib.conn, embedder.model)
    if not senses:
        raise _NoVectors()
    if isinstance(query, SenseKey):
        qv = senses.get(query)
        if qv is None:
            raise _NoVectors()
    else:
        vecs, _ = vx.vectors_for(lib, [query], embedder, prefix=vx.QUERY_PREFIX)
        qv = vx.normalized(vecs[query])
    return [(k, vx.dot(qv, v)) for k, v in senses.items()
            if k != query and (allowed is None or k in allowed)]


# --- find ---------------------------------------------------------------------------------------

def _graph_papers(lib: Library, graph: str) -> list[str]:
    from ..library import graphs

    with graphs.open_graph(lib, graph) as g:
        return [p.paper_id for p in graphs.membership(g) if p.status == "member"]


def _as_key(query: str | SenseKey) -> str | SenseKey:
    if isinstance(query, SenseKey):
        return SenseKey(query.family, label_key(query.label_key), query.sense_id or DEFAULT_SENSE)
    return query


def find(lib: Library, query: str | SenseKey, *, family: str | None = None, graph: str | None = None,
         k: int = 20, mode: Mode = "auto", embedder: vx.Embedder | None = None) -> list[SenseHit]:
    """Sense-level hits, best first (see the module docstring)."""
    from ..llm.ollama import OllamaError

    query = _as_key(query)
    conn = lib.conn
    families = [family] if family else None
    papers = _graph_papers(lib, graph) if graph else None
    claim_sense = _claim_senses(conn, families=families, papers=papers)
    fallback: str | None = None
    ranked: list[tuple[SenseKey, float, str | None]] = []
    ran = "keyword"

    want_meaning = mode == "meaning" or (mode == "auto" and adoption(conn).adopted)
    if want_meaning:
        if embedder is None:
            embedder = vx.OllamaEmbedder(adoption(conn).model)
        try:
            scored = _meaning(lib, query, embedder, set(claim_sense.values()))
            ranked = [(s, score, None) for s, score in scored]
            ran = "meaning"
        except _NoVectors:
            fallback = "no_vectors"
        except OllamaError as exc:
            fallback = exc.code

    if ran == "keyword":
        if isinstance(query, SenseKey):
            terms = keyword_terms_for_claims(conn, _sense_claims(conn, query), _surface(conn, query))
        else:
            terms = text_terms(query)
        best: dict[SenseKey, tuple[float, str]] = {}
        for cid, score in keyword_claim_scores(conn, terms, families=families, papers=papers).items():
            s = claim_sense.get(cid)
            if s is None or s == query:
                continue
            if s not in best or (score, _neg(cid)) > (best[s][0], _neg(best[s][1])):
                best[s] = (score, cid)
        ranked = [(s, sc, cid) for s, (sc, cid) in best.items()]

    ranked.sort(key=lambda t: (-t[1], tuple(t[0])))
    ranked = ranked[:k]
    if ran == "meaning":  # the best example claim: the sense's first supporting claim with a quote
        firsts: dict[SenseKey, str] = {}
        for cid, s in sorted(claim_sense.items()):
            firsts.setdefault(s, cid)
        ranked = [(s, sc, firsts.get(s)) for s, sc, _ in ranked]
    info = _sense_info(conn, [s for s, _, _ in ranked])
    texts = _claim_texts(conn, [c for _, _, c in ranked if c])
    out = []
    for s, sc, cid in ranked:
        i = info[s]
        text, paper = texts.get(cid, (None, None)) if cid else (None, None)
        out.append(SenseHit(s, handle(s), i.concept_id, i.name, i.gloss, i.paper_count, i.claim_count,
                            round(sc, 6), ran, cid, text, paper, fallback))
    return out


def _neg(text: str) -> tuple[int, ...]:
    return tuple(-ord(c) for c in text)
