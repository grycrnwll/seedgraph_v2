"""Shared token estimator (chars/4 heuristic).

THE single estimator phase_4's context-window gate (``extraction/runner.py``) and
phase_4b's chunk planner (``extraction/chunker.py``) must both use, so the
oversize verdict and the chunk-sizing budget agree on one coordinate system (D9).
"""

from __future__ import annotations


def estimate_tokens(text: str) -> int:
    # ponytail: chars/4 heuristic; swap for a real tokenizer if oversize verdicts need precision
    return max(1, len(text) // 4)


if __name__ == "__main__":
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 400) == 100
    print("ok")
