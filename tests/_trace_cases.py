"""Shared answer-PATH case definitions for the AnswerTrace parity + trace tests.

One place defines the questions that exercise every answer path (retrieval-only, LLM,
weak/empty short-circuit abstain, gap_finding, comparative, citation_search) so the
pre-change parity generator and the post-change tests drive byte-identical inputs — the
C9 envelope-parity guard (00 §6, plan criterion 1) is only meaningful if both sides ask
the same questions the same way.

Each case yields ``(name, envelope)`` through a caller-supplied ``invoke(question,
**kwargs)`` that returns an :class:`AnswerEnvelope`: the pre-change generator binds
``invoke = lambda q, **kw: answer(q, project, **kw)``; the post-change tests bind
``... answer(q, project, **kw)[0]``. This module is therefore agnostic to the
``(envelope, trace)`` tuple return that chunk 0 introduces.
"""

from __future__ import annotations

from _phase8_helpers import canned_envelope_json, make_answer_config

from seedgraph.answer import compose
from seedgraph.llm.backend import FakeLLMBackend

# The zeroed placeholder every parity snapshot carries in place of the random
# ``answer_id`` (decisions 29/44/65) so two runs of the same question compare equal.
_ZERO_ANSWER_ID = "ans_" + "0" * 32


def normalize_envelope(env_dict: dict) -> dict:
    """Normalize an envelope dump for parity comparison.

    The only non-deterministic field on an :class:`AnswerEnvelope` is the random
    ``answer_id`` — zero it. The envelope carries no ``created_at`` (only the trace
    does), so nothing else varies run to run for a fixed corpus + config.
    """
    out = dict(env_dict)
    out["answer_id"] = _ZERO_ANSWER_ID
    return out


def parity_cases(invoke) -> list[tuple[str, object]]:
    """Return ``(name, envelope)`` for every answer path against the ph8 fixture corpus.

    ``invoke(question, **kwargs)`` must build and return an :class:`AnswerEnvelope`.
    Sets the offline fake-backend override per case (the LLM-path case needs a canned
    reply; the no-LLM / abstain cases clear it), and always clears it on exit so the
    override never leaks into a following test.
    """
    cfg = make_answer_config()
    cases: list[tuple[str, object]] = []
    try:
        # 1. retrieval-only degrade (no_llm) — standard path, grounded question.
        compose._BACKEND_OVERRIDE = None
        cases.append((
            "retrieval_only",
            invoke('According to Paper A, what holds "across groups"?',
                   no_llm=True, config=cfg),
        ))
        # 2. full LLM path via the fake backend — standard path: prose + a citation.
        compose._BACKEND_OVERRIDE = FakeLLMBackend(
            response=canned_envelope_json(
                answer_text="It holds across groups [E1].", cited_markers=[1])
        )
        cases.append((
            "llm",
            invoke('According to Paper A, what holds "across groups"?', config=cfg),
        ))
        # 3. weak/empty short-circuit abstain — a no-evidence question.
        compose._BACKEND_OVERRIDE = FakeLLMBackend(
            response=canned_envelope_json(answer_text="x", cited_markers=[1])
        )
        cases.append((
            "abstain_empty",
            invoke("What is the capital of France?", config=cfg),
        ))
        # 4. gap_finding early return — co-citation recommendations, no LLM call.
        compose._BACKEND_OVERRIDE = None
        cases.append((
            "gap_finding",
            invoke("What research gaps remain in the corpus?", config=cfg),
        ))
        # 5. comparative — factual retrieval run once per named work (no_llm keeps the
        #    snapshot backend-independent; the comparative RETRIEVAL branch is the point).
        compose._BACKEND_OVERRIDE = None
        cases.append((
            "comparative",
            invoke('Compare "Paper A" and "Paper B" on identification.',
                   no_llm=True, config=cfg),
        ))
        # 6. citation_search — depth-1 neighborhood traversal (neighborhood present).
        compose._BACKEND_OVERRIDE = None
        cases.append((
            "citation_search",
            invoke("Which works are cited by Paper A?", no_llm=True, config=cfg),
        ))
    finally:
        compose._BACKEND_OVERRIDE = None
    return cases
