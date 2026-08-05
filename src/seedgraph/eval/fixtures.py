"""Recorded AnswerEnvelope cassettes for the key-free CI gate (decisions 39/73).

The CI safety invariants (``test_phase_9_acceptance``) MUST pass with NO API key
present (doc 10 §3): they never call a live model — they replay hand-authored
``AnswerEnvelope`` JSON from ``evals/fixtures/envelopes/``. Those cassettes are
authored to EXACTLY the phase-8 envelope shape (§9) and double as the executable
spec phase-8 must satisfy. Repo-level fixtures hold NO restricted full text.

``AnswerEnvelope`` is imported from phase-8 (``seedgraph.answer.types`` — the plan
names ``answer.envelope`` but the actual module is ``answer.types``), never defined
here (decisions 62/82; §9).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported only for type-checkers; never executed at runtime
    from seedgraph.answer.types import AnswerEnvelope  # phase_8 owns (plan §5/§9)


def evals_dir() -> Path:
    """Resolve the repo-level ``evals/`` directory (version-controlled fixtures).

    Honors ``$SEEDGRAPH_EVALS_DIR`` (explicit override); otherwise computes it
    relative to the installed package — for the editable dev/CI install
    ``src/seedgraph/eval/fixtures.py`` → repo root → ``evals``.
    """
    override = os.environ.get("SEEDGRAPH_EVALS_DIR")
    if override:
        return Path(override)
    # parents: [eval] -> [seedgraph] -> [src] -> [repo root]
    return Path(__file__).resolve().parents[3] / "evals"


def envelopes_dir() -> Path:
    """``evals/fixtures/envelopes/`` — the recorded AnswerEnvelope cassettes."""
    return evals_dir() / "fixtures" / "envelopes"


def load_envelope(fixture_name: str) -> "AnswerEnvelope":
    """Load + deserialize ``evals/fixtures/envelopes/{fixture_name}.json`` into a
    phase-8 ``AnswerEnvelope`` (no LLM). Raises ``FileNotFoundError`` when the named
    cassette is absent. The sole envelope source for the recorded-fixture CI gate
    (§11/§12) — the live harness path lives only in ``runners.run_question``.
    """
    from ..answer.types import AnswerEnvelope  # imported, NEVER redefined (§9)

    path = envelopes_dir() / f"{fixture_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"recorded envelope cassette not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return AnswerEnvelope.model_validate(data)


def load_envelopes(prefix: str) -> list["AnswerEnvelope"]:
    """Load every cassette whose filename starts with ``prefix`` (sorted by name).

    Used by the leakage gate (``leak_*``) and the answer-faithfulness gate
    (``answer_*``) to replay the canned envelope set deterministically.
    """
    out: list = []
    for path in sorted(envelopes_dir().glob(f"{prefix}*.json")):
        out.append(load_envelope(path.stem))
    return out
