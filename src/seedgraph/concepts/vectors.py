"""Claim-context vectors (stage 4 design §5.2-§5.3). Pure Python: no numpy.

- **What is embedded**: each supported claim as its *context*,
  ``"<family>: <label>. <claim text>. <plain words>. Quote: <quote>"`` (:func:`claim_context`;
  plain words where present, the top quote when it says more than the claim text).
  Bare labels are never embedded.
- **Cache**: ``concept_vectors`` keyed by ``sha256(exact text sent)`` + model, never by
  ``claim_id``. The task prefix is part of the hashed text (nomic needs
  ``search_document:`` when indexing and ``search_query:`` when searching), so the two
  never collide. A re-note that keeps a claim's text reuses its vector.
- **Sense vectors**: the normalised mean of their supporting claims' (unit) vectors
  (:func:`sense_vectors`).
- **Embedder**: anything with a ``model`` attribute that maps a list of texts to a list
  of vectors; :class:`OllamaEmbedder` is the real one. Tests inject a fake.
"""

from __future__ import annotations

import hashlib
import math
import sqlite3
import sys
import time
from array import array
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

from ..semantic.merge import normalize
from .model import SenseKey

if TYPE_CHECKING:
    from ..library.store import Library

DEFAULT_MODEL = "nomic-embed-text"
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
DEFAULT_BATCH = 64
QUOTE_CHARS = 400


class Embedder(Protocol):
    model: str

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class OllamaEmbedder:
    """``llm.ollama.embed`` as an :class:`Embedder`."""

    model: str = DEFAULT_MODEL
    base_url: str | None = None
    timeout: float = 300.0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        from ..llm.ollama import OllamaClient

        return OllamaClient(self.base_url, timeout=self.timeout).embed(self.model, texts)


# --- text ---------------------------------------------------------------------------------------

def _sentence(text: str | None) -> str:
    flat = " ".join((text or "").split()).strip()
    return flat if not flat or flat[-1] in ".!?" else flat + "."


def claim_context(family: str, label: str, text: str | None, *, plain_words: str | None = None,
                  quote: str | None = None) -> str:
    """The embed text of one claim (no task prefix)."""
    parts = [_sentence(f"{family}: {' '.join((label or '').split())}"), _sentence(text)]
    if plain_words and plain_words.strip():
        parts.append(_sentence(plain_words))
    if quote and quote.strip():
        q, t = normalize(quote), normalize(text or "")
        if q and q not in t and t not in q:
            flat = " ".join(quote.split())
            parts.append("Quote: " + _sentence(flat[:QUOTE_CHARS]))
    return " ".join(p for p in parts if p)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- vector arithmetic --------------------------------------------------------------------------

def pack(vec: Sequence[float]) -> bytes:
    arr = array("f", vec)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def unpack(blob: bytes) -> list[float]:
    arr = array("f")
    arr.frombytes(blob)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tolist()


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def normalized(v: Sequence[float]) -> list[float]:
    n = math.sqrt(math.fsum(x * x for x in v))
    return [x / n for x in v] if n > 0 else list(v)


def mean_vector(vectors: Iterable[Sequence[float]]) -> list[float]:
    """The normalised mean of the unit vectors."""
    vs = [normalized(v) for v in vectors]
    if not vs:
        raise ValueError("mean of no vectors")
    dims = len(vs[0])
    return normalized([math.fsum(v[i] for v in vs) / len(vs) for i in range(dims)])


# --- claims -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimContext:
    claim_id: str
    paper_id: str
    family: str
    label_key: str
    sense_id: str | None  # None = needs sense
    agent_visible: bool
    text: str

    @property
    def key(self) -> SenseKey | None:
        return SenseKey(self.family, self.label_key, self.sense_id) if self.sense_id else None


def claim_contexts(conn: sqlite3.Connection, *, families: Iterable[str] | None = None,
                   claim_ids: Iterable[str] | None = None) -> list[ClaimContext]:
    """Every supported claim's context (after a refresh), ordered by claim id."""
    where, params = ["k.supports = 1"], []
    if families is not None:
        fams = sorted(set(families))
        where.append(f"k.family IN ({', '.join('?' * len(fams))})")
        params += fams
    if claim_ids is not None:
        ids = sorted(set(claim_ids))
        where.append(f"k.claim_id IN ({', '.join('?' * len(ids))})")
        params += ids
    sql = ("SELECT k.claim_id, k.paper_id, k.family, k.label_key, k.sense_id, k.agent_visible, "
           "cc.normalized_label, cc.claim_text, cc.plain_words, e.exact_quote "
           "FROM claim_concepts k JOIN current_claims cc ON cc.claim_id = k.claim_id "
           "LEFT JOIN evidence_spans e ON e.span_id = cc.span_id "
           f"WHERE {' AND '.join(where)} ORDER BY k.claim_id")
    out = []
    for r in conn.execute(sql, params):
        out.append(ClaimContext(r[0], r[1], r[2], r[3], r[4], bool(r[5]),
                                claim_context(r[2], r[6] or r[3], r[7], plain_words=r[8], quote=r[9])))
    return out


# --- cache --------------------------------------------------------------------------------------

@dataclass
class EmbedStats:
    model: str
    requested: int = 0
    cached: int = 0
    embedded: int = 0
    seconds: float = 0.0


def cached_vectors(conn: sqlite3.Connection, texts: Iterable[str], model: str, *,
                   prefix: str = DOC_PREFIX) -> dict[str, list[float]]:
    """The stored vectors of ``texts`` (keyed by the un-prefixed text); a read only."""
    by_sha = {text_sha256(prefix + t): t for t in texts}
    out: dict[str, list[float]] = {}
    shas = list(by_sha)
    for i in range(0, len(shas), 500):
        chunk = shas[i:i + 500]
        for sha, blob in conn.execute(
                f"SELECT text_sha256, vector FROM concept_vectors WHERE model = ? AND text_sha256 IN "
                f"({', '.join('?' * len(chunk))})", [model, *chunk]):
            out[by_sha[sha]] = unpack(blob)
    return out


def vectors_for(lib: Library, texts: Iterable[str], embedder: Embedder, *, prefix: str = DOC_PREFIX,
                batch_size: int = DEFAULT_BATCH,
                progress: Callable[[int, int], None] | None = None) -> tuple[dict[str, list[float]], EmbedStats]:
    """Vectors for ``texts`` (keyed by the un-prefixed text): cached ones read, the rest
    embedded in batches and stored (one write per batch). Raises the embedder's error
    (``OllamaError``) when the daemon fails; vectors stored before the failure stay."""
    unique = list(dict.fromkeys(texts))
    stats = EmbedStats(embedder.model, requested=len(unique))
    out = cached_vectors(lib.conn, unique, embedder.model, prefix=prefix)
    stats.cached = len(out)
    todo = [t for t in unique if t not in out]
    start = time.perf_counter()
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        vecs = embedder([prefix + t for t in chunk])
        now = datetime.now(timezone.utc).isoformat()
        with lib.write() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO concept_vectors (text_sha256, model, dims, vector, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(text_sha256(prefix + t), embedder.model, len(v), pack(v), now) for t, v in zip(chunk, vecs)])
        for t, v in zip(chunk, vecs):
            out[t] = unpack(pack(v))  # what a later read returns (float32)
        stats.embedded += len(chunk)
        if progress is not None:
            progress(stats.embedded, len(todo))
    stats.seconds = time.perf_counter() - start
    return out, stats


def embed_library(lib: Library, embedder: Embedder, *, families: Iterable[str] | None = None,
                  batch_size: int = DEFAULT_BATCH,
                  progress: Callable[[int, int], None] | None = None) -> EmbedStats:
    """Embed every supported claim context not yet cached (the after-refresh job)."""
    contexts = claim_contexts(lib.conn, families=families)
    _, stats = vectors_for(lib, [c.text for c in contexts], embedder, batch_size=batch_size, progress=progress)
    return stats


def sense_vectors(contexts: Iterable[ClaimContext], vectors: dict[str, list[float]]) -> dict[SenseKey, list[float]]:
    """Each sense's normalised mean claim vector (claims needing a sense are skipped)."""
    groups: dict[SenseKey, list[list[float]]] = defaultdict(list)
    for c in contexts:
        if c.key is not None and c.text in vectors:
            groups[c.key].append(vectors[c.text])
    return {k: mean_vector(vs) for k, vs in sorted(groups.items())}
