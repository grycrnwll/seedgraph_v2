"""The spot-check (v3 stage 5, docs/V3_RECORDS_DESIGN.md §5; slice R4).

On-demand samples of about 20 agent claims, spread across claim type and model; the
researcher marks each right, partly right or wrong; accuracy per type and per model is
read from the marks, always with its count.

- :mod:`.sample`: the population (every agent claim of each paper's current note, hidden
  and edited included), model keys, the least-marked-cell-first draw, reading a sample;
- :mod:`.marks`: validation, the last-mark rule, the derived "fixed" check;
- :mod:`.accuracy`: raw rows per (type, model), per type and per model, the weighted
  overall line and the "too few marks" rule.

The functions ``sample.sample`` and ``accuracy.accuracy`` share their module's name, so
they are reached through the module (``from seedgraph.spotcheck import sample``).
"""

from .accuracy import AccuracyRow, AccuracyTable, OverallLine
from .marks import MARK_LABELS, MARKS, REASON_LABELS, REASONS, fixed, mark
from .sample import (
    DEFAULT_SIZE,
    NO_MODEL,
    PER_PAPER_MAX,
    Item,
    Sample,
    SpotCheckError,
    create_sample,
    model_key,
    samples,
)

__all__ = [
    "DEFAULT_SIZE", "MARKS", "MARK_LABELS", "NO_MODEL", "PER_PAPER_MAX", "REASONS", "REASON_LABELS",
    "AccuracyRow", "AccuracyTable", "Item", "OverallLine", "Sample", "SpotCheckError",
    "create_sample", "fixed", "mark", "model_key", "samples",
]
