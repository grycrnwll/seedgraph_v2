"""Spot-check accuracy (design §5.4, §13 Q4; slice R4): raw rows with counts, roll-ups per
type and per model, partly right never counted as half, the "too few" rule and the
overall line weighted to the library's mix."""

from __future__ import annotations

import pytest

from seedgraph.spotcheck import accuracy as acc
from seedgraph.spotcheck import marks
from seedgraph.spotcheck import sample as sp


def row(ctype, model, r, p, w):
    return acc.AccuracyRow(ctype, model, r, p, w, r + p + w)


def test_every_figure_carries_its_count():
    assert row("result", "m1", 14, 4, 2).summary == "right 14 of 20 (70%); partly right 4; wrong 2"
    assert row("result", "m1", 14, 4, 2).share_right == pytest.approx(0.7)


def test_partly_right_is_never_half_a_point():
    r = row("method", "m1", 0, 10, 0)
    assert r.share_right == 0 and r.share("partly") == 1


def test_too_few_marks_to_say():
    r = row("method", "m1", 3, 1, 0)
    assert r.too_few and r.share_right is None
    assert r.summary == "too few marks to say (4 marked)"
    assert not row("method", "m1", 3, 1, 1).too_few


def test_overall_is_weighted_to_the_population_mix():
    cells = [row("method", "m1", 10, 0, 0), row("result", "m2", 0, 5, 5)]
    # the thin cell was over-drawn: 90 of 100 claims are methods
    line = acc.overall(cells, {("method", "m1"): 90, ("result", "m2"): 10})
    assert line.share_right == pytest.approx(0.9)
    assert line.share_partly == pytest.approx(0.05) and line.share_wrong == pytest.approx(0.05)
    assert line.covered == pytest.approx(1.0) and line.population == 100 and line.n_marked == 20
    assert acc.WEIGHTED_NOTE in line.summary and "right 90%" in line.summary
    # a cell with no marks yet: weights renormalise over marked cells, coverage says so
    line = acc.overall(cells, {("method", "m1"): 90, ("result", "m2"): 10, ("data", "m1"): 100})
    assert line.share_right == pytest.approx(0.9) and line.covered == pytest.approx(0.5)


def test_overall_with_too_few_marks():
    line = acc.overall([row("method", "m1", 2, 1, 1)], {("method", "m1"): 10})
    assert line.too_few and line.share_right is None
    assert line.summary == "too few marks to say (4 marked)"
    assert acc.overall([], {}).too_few


def test_accuracy_from_the_library(world, lib):
    for _ in range(7):
        world.noted_paper(model="invented-model-a")
    for _ in range(3):
        world.noted_paper(model=None, client="codex")
    s = sp.create_sample(lib, n=20, seed=9)
    values = ["right", "partly", "wrong"]
    for k, item in enumerate(s.items):
        marks.mark(lib, s.sample_id, item.claim_id, values[k % 3])

    table = acc.accuracy(lib)
    view = {(r[0], r[1]): tuple(r[2:]) for r in lib.conn.execute("SELECT * FROM spot_check_accuracy")}
    assert {(c.claim_type, c.model_key): (c.n_right, c.n_partly, c.n_wrong, c.n_marked) for c in table.cells} == view
    for rollup, key in ((table.by_type, 0), (table.by_model, 1)):
        expected: dict[str, list[int]] = {}
        for cell, counts in view.items():
            e = expected.setdefault(cell[key], [0, 0, 0, 0])
            for i, v in enumerate(counts):
                e[i] += v
        got = {(r.claim_type if key == 0 else r.model_key): [r.n_right, r.n_partly, r.n_wrong, r.n_marked]
               for r in rollup}
        assert got == expected
    assert {r.model_key for r in table.by_model} == {"invented-model-a", "codex, model not recorded"}
    assert all(r.model_key is None for r in table.by_type) and all(r.claim_type is None for r in table.by_model)
    assert sum(r.n_marked for r in table.by_model) == 20
    assert table.overall.n_marked == 20 and not table.overall.too_few
    assert table.overall.population == 10 * 5  # 12 claims per note, 7 of them not_found
    weights = {}
    for c in sp.population(lib, include_marked=True):
        weights[c.cell] = weights.get(c.cell, 0) + 1
    assert table.overall == acc.overall(table.cells, weights)


def test_by_selects_the_rollups(world, lib):
    world.noted_paper()
    t = acc.accuracy(lib, by=("type",))
    assert t.by_model == () and t.cells == () and t.overall.too_few
    with pytest.raises(ValueError, match="unknown accuracy grouping"):
        acc.accuracy(lib, by=("paper",))
