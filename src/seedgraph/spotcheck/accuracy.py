"""Spot-check accuracy (design §5.4, §13 Q4; slice R4).

The library stores raw counts per (claim type, model key) as the view
``spot_check_accuracy`` (latest mark per claim). This module adds the per-type and
per-model roll-ups (raw shares of the marks in that row), the overall line and the
"too few" rule:

- the headline is the share marked **right**; partly right is shown beside it, never
  counted as half a point;
- every figure carries its count ("right 14 of 20 (70%); partly right 4; wrong 2");
- a row with fewer than :data:`MIN_MARKS` marks says "too few marks to say";
- the overall line is **weighted to the library's mix**: each marked cell's shares count
  by that cell's share of the whole current population (marked claims included, so the
  weights do not drift as marks come in), renormalised over the cells that have marks.
  The sampling rule over-draws thin cells, so a raw overall share would mislead.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..library.store import Library
from .sample import Cell, population

MIN_MARKS = 5
TOO_FEW = "too few marks to say"
WEIGHTED_NOTE = "weighted to the library's mix of types and models"
BY: tuple[str, ...] = ("type", "model")


def _pct(k: int | float, n: int | float) -> str:
    return f"{round(100 * k / n)}%"


@dataclass(frozen=True)
class AccuracyRow:
    """Counts for one cell, claim type or model (``None`` in a roll-up's other column)."""

    claim_type: str | None
    model_key: str | None
    n_right: int
    n_partly: int
    n_wrong: int
    n_marked: int

    @property
    def too_few(self) -> bool:
        return self.n_marked < MIN_MARKS

    def share(self, which: str) -> float | None:
        """``right``, ``partly`` or ``wrong`` as a share of the marks; ``None`` when too few."""
        if self.too_few:
            return None
        return {"right": self.n_right, "partly": self.n_partly, "wrong": self.n_wrong}[which] / self.n_marked

    @property
    def share_right(self) -> float | None:
        return self.share("right")

    @property
    def summary(self) -> str:
        if self.too_few:
            return f"{TOO_FEW} ({self.n_marked} marked)"
        return (f"right {self.n_right} of {self.n_marked} ({_pct(self.n_right, self.n_marked)}); "
                f"partly right {self.n_partly}; wrong {self.n_wrong}")


@dataclass(frozen=True)
class OverallLine:
    """The library-wide figure, weighted to the population's mix of cells."""

    n_marked: int
    share_right: float | None
    share_partly: float | None
    share_wrong: float | None
    population: int            # claims in the current population
    covered: float | None      # share of the population in cells that have marks

    @property
    def too_few(self) -> bool:
        return self.share_right is None

    @property
    def summary(self) -> str:
        if self.too_few:
            return f"{TOO_FEW} ({self.n_marked} marked)"
        return (f"right {_pct(self.share_right or 0, 1)}; partly right {_pct(self.share_partly or 0, 1)}; "
                f"wrong {_pct(self.share_wrong or 0, 1)} — {WEIGHTED_NOTE} ({self.n_marked} marked)")


@dataclass(frozen=True)
class AccuracyTable:
    cells: tuple[AccuracyRow, ...]          # per (type, model)
    by_type: tuple[AccuracyRow, ...]
    by_model: tuple[AccuracyRow, ...]
    overall: OverallLine


def _rollup(cells: Iterable[AccuracyRow], key: str) -> tuple[AccuracyRow, ...]:
    sums: dict[str, list[int]] = {}
    for c in cells:
        k = c.claim_type if key == "type" else c.model_key
        assert k is not None
        s = sums.setdefault(k, [0, 0, 0, 0])
        for i, v in enumerate((c.n_right, c.n_partly, c.n_wrong, c.n_marked)):
            s[i] += v
    return tuple(AccuracyRow(k if key == "type" else None, k if key == "model" else None, *v)
                 for k, v in sorted(sums.items()))


def overall(cells: Iterable[AccuracyRow], weights: Mapping[Cell, int]) -> OverallLine:
    """The weighted overall line from cell rows and population counts per cell. Pure."""
    cells = list(cells)
    n_marked = sum(c.n_marked for c in cells)
    pop = sum(weights.values())

    def weight(c: AccuracyRow) -> int:
        return weights.get((c.claim_type or "", c.model_key or ""), 0)

    marked = [c for c in cells if c.n_marked > 0 and weight(c) > 0]
    total = sum(weight(c) for c in marked)
    if n_marked < MIN_MARKS or total == 0:
        return OverallLine(n_marked, None, None, None, pop, total / pop if pop else None)

    def weighted(count: str) -> float:
        return sum(weight(c) / total * getattr(c, count) / c.n_marked for c in marked)

    return OverallLine(n_marked, weighted("n_right"), weighted("n_partly"), weighted("n_wrong"), pop, total / pop)


def accuracy(lib: Library | sqlite3.Connection, *, by: Iterable[str] = BY) -> AccuracyTable:
    """Accuracy from the marks: cells always, the roll-ups named in ``by``, the overall line."""
    want = tuple(by)
    bad = [b for b in want if b not in BY]
    if bad:
        raise ValueError(f"unknown accuracy grouping {bad[0]!r}; use {', '.join(BY)}")
    conn = lib if isinstance(lib, sqlite3.Connection) else lib.conn
    cells = tuple(AccuracyRow(r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in conn.execute(
        "SELECT claim_type, model_key, n_right, n_partly, n_wrong, n_marked FROM spot_check_accuracy "
        "ORDER BY claim_type, model_key"))
    weights: Counter[Cell] = Counter(c.cell for c in population(conn, include_marked=True))
    return AccuracyTable(
        cells=cells,
        by_type=_rollup(cells, "type") if "type" in want else (),
        by_model=_rollup(cells, "model") if "model" in want else (),
        overall=overall(cells, weights))
