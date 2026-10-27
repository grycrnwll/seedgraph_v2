"""A deterministic fake embedder for the concept search tests: a bag of hashed content
words (the task prefix and a few function words dropped), so texts that share words
land close and texts that do not land apart. Records every call."""

from __future__ import annotations

import hashlib
import re

_STOP = {"search_document", "search_query", "the", "a", "an", "of", "is", "to", "and", "in", "on", "for",
         "invented", "holds", "used", "method", "assumption", "result", "estimand"}
_WORD = re.compile(r"[a-z_]+")


class FakeEmbedder:
    def __init__(self, model: str = "fake-embed", dims: int = 64) -> None:
        self.model = model
        self.dims = dims
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for t in texts:
            v = [0.0] * self.dims
            for w in _WORD.findall(t.lower()):
                if w in _STOP:
                    continue
                v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % self.dims] += 1.0
            v[0] += 0.01  # never the zero vector
            out.append(v)
        return out

    @property
    def embedded(self) -> list[str]:
        return [t for call in self.calls for t in call]


class DownEmbedder:
    """An embedder whose daemon is unreachable."""

    model = "fake-embed"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        from seedgraph.llm.ollama import OllamaError

        raise OllamaError("unreachable", "cannot reach Ollama (invented)")


def index_fts(lib) -> None:
    """Fill ``claim_fts`` for the World's claims, as a notes save does (World writes
    ``extracted_claims`` directly)."""
    with lib.write() as conn:
        conn.execute("DELETE FROM claim_fts")
        conn.execute("INSERT INTO claim_fts (claim_id, normalized_label, claim_text, plain_words) "
                     "SELECT claim_id, normalized_label, claim_text, plain_words FROM extracted_claims "
                     "ORDER BY claim_id")
