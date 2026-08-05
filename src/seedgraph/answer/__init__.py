"""Seedgraph answer harness (phase_8).

Deterministic FTS5 retrieval + a single self-declaring LLM call with code-enforced
faithfulness and number→id citation mapping; no-LLM mode degrades to retrieval
only (decisions 82/57/58/38/62). The package exposes exactly the public entry
``answer()`` and the contract type ``AnswerEnvelope`` that later phases (eval,
FastAPI read view, agent surfaces) consume.
"""

from __future__ import annotations

from .harness import answer, save_answer
from .trace import AnswerTrace, save_trace
from .types import AnswerEnvelope

__all__ = ["answer", "save_answer", "save_trace", "AnswerEnvelope", "AnswerTrace"]
