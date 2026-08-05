"""No-network closed-form acquisition budget preview (Build D ch13, D-13).

Port of v1 ``web/state.py:104-177`` (``preview_budget``) MINUS the LLM lines —
v2's planner already previews LLM token/cost separately through the executor
``preflight`` seam, so this helper covers only the acquisition fan-out: the one
phase that hits rate-limited public APIs. Computed PURELY from ``seed_count`` +
``depth`` + ``cap`` with NO network and NO I/O, so it is safe to render on page
load (the whole point — network-based estimation was the rejected alternative;
it defeats the preview's reason to exist).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Estimated bytes per fetched OA PDF + converted markdown output (for the disk
# estimate). Order-of-magnitude only — this is a pre-run BUDGET, not an
# accounting. (v1 web/state.py:106-107.)
# --------------------------------------------------------------------------- #
_EST_PDF_BYTES = 1_500_000  # ~1.5 MB per OA PDF
_EST_MARKDOWN_BYTES = 400_000  # ~0.4 MB markdown + meta per paper

#: Provider calls per paper: resolve identity + pull its reference list, plus the
#: OA-URL lookup the explicit fetch pass adds (resolve + referenced_works +
#: OA-location), roughly. (v1 web/state.py:108-111 — bumped from 2 for the
#: two-phase flow.)
_API_CALLS_PER_PAPER = 3


def preview_acquisition(seed_count: int, depth: int, cap: int) -> dict:
    """Compute a pre-run acquisition BUDGET from depth + cap with NO network.

    The walk resolves ``seed_count`` seeds (generation 0), then expands up to
    ``depth`` further generations, each frontier capped at ``cap``
    (``per_gen_cap``). The worst case is therefore::

        expected_papers = seed_count + cap * depth

    From that closed form: ``api_calls`` is ``~3`` provider calls per paper and
    ``est_disk_bytes`` is ``~1.9 MB`` per paper (OA PDF + converted markdown).
    Returns a plain dict so the planner template and the CLI ``--dry-run`` can
    render it directly. Negative inputs clamp to 0 (v1 behavior).
    """
    seed_count = max(0, int(seed_count))
    depth = max(0, int(depth))
    cap = max(0, int(cap))

    expected_papers = seed_count + cap * depth
    est_disk = expected_papers * (_EST_PDF_BYTES + _EST_MARKDOWN_BYTES)

    return {
        "seed_count": seed_count,
        "depth": depth,
        "cap": cap,
        "expected_papers": expected_papers,
        "api_calls": expected_papers * _API_CALLS_PER_PAPER,
        "est_disk_bytes": est_disk,
        "est_disk_mb": round(est_disk / 1_000_000, 1),
    }


__all__ = ["preview_acquisition"]
