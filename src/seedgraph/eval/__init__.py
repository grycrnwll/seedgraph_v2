"""Phase 9 — Evaluation & Validation Harness (governing decision 62).

Cross-cutting last phase: it audits the outputs of every prior phase. It adds
measurement (deterministic, stdlib metrics), seeded sampling, a human validation
surface (the ``seedgraph eval`` CLI writing ``audit_records`` + per-run
``runs/{run_id}/eval/`` artifacts), and CI safety gates that pass WITHOUT an API
key by replaying recorded ``AnswerEnvelope`` cassettes (decisions 39/73).

This package is intentionally a SCAFFOLD: the public interfaces (phase_9 §6) are
present with real signatures + docstrings, but bodies raise ``NotImplementedError``
pending the full build. Nothing here mutates graph state — that is
``review_queue``'s job (decisions 56/71/80; phase_9 §4 boundary rule). The
``AnswerEnvelope`` consumed by ``runners``/``fixtures`` is imported from phase-8
(``seedgraph.answer.envelope``), never redefined here (decisions 62/82; §9).

Submodules are imported lazily by callers; this package marker pulls in nothing
so that importing :mod:`seedgraph.eval` stays free of upstream-phase dependencies.
"""

from __future__ import annotations

__all__ = [
    "audit",
    "boundary",
    "fixtures",
    "goldsets",
    "metrics",
    "report",
    "runners",
]
