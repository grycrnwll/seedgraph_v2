"""Lens choice for a context graph (stage 4 design §8.4): choose and unchoose."""

from __future__ import annotations

import pytest

from seedgraph.library import graphs, store


@pytest.fixture
def lib(tmp_path):
    library = store.open_library(tmp_path)
    yield library
    library.close()


def test_unchoose_lens_removes_only_that_lens_and_reports_whether_it_was_chosen(lib):
    with graphs.create_graph(lib, "g1") as g:
        graphs.choose_lens(g, "widget_lens_v1")
        graphs.choose_lens(g, "gadget_lens_v2")

        assert graphs.unchoose_lens(g, "widget_lens_v1") is True
        assert graphs.chosen_lenses(g) == ["gadget_lens_v2"]
        assert graphs.unchoose_lens(g, "widget_lens_v1") is False  # already gone
        assert graphs.chosen_lenses(g) == ["gadget_lens_v2"]


def test_a_lens_can_be_chosen_again_after_unchoosing(lib):
    with graphs.create_graph(lib, "g1") as g:
        graphs.choose_lens(g, "widget_lens_v1")
        graphs.unchoose_lens(g, "widget_lens_v1")
        graphs.choose_lens(g, "widget_lens_v1")
        assert graphs.chosen_lenses(g) == ["widget_lens_v1"]
