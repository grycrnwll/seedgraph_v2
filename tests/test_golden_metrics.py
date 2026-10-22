"""Golden-set metric counters on synthetic markdown (no paper text)."""

from __future__ import annotations

from seedgraph.golden import metrics


def _paged(*pages: str) -> str:
    return "\n\n".join(f"{{{i}}}------------------------------------------------\n\n{p}" for i, p in enumerate(pages))


def test_page_markers_lists_marker_indices_in_order():
    md = _paged("alpha", "beta", "gamma")
    assert metrics.page_markers(md) == [0, 1, 2]


def test_page_markers_empty_when_no_pagination():
    assert metrics.page_markers("# Title\n\nbody") == []


def test_math_counts_display_and_inline_spans():
    md = "Text $a+b$ and $c$.\n\n$$\n\\frac{1}{2}\n$$\n\nMore $$x^{2}$$ end."
    stats = metrics.math_stats(md)
    assert stats.display == 2
    assert stats.inline == 2
    assert stats.total == 4
    assert stats.non_parsing == 0


def test_math_unbalanced_braces_do_not_parse():
    md = "Bad $\\frac{1}{2$ good $x$.\n\n$$\n\\left( x \\right.\n$$\n\n$$\\left( y$$"
    stats = metrics.math_stats(md)
    assert stats.total == 4
    # \frac{1}{2  -> brace imbalance; \left( y with no \right -> mismatch.
    assert stats.non_parsing == 2


def test_math_escaped_braces_and_dollars_are_ignored():
    md = "Cost \\$5 and set $\\{x\\}$ here."
    stats = metrics.math_stats(md)
    assert stats.inline == 1
    assert stats.non_parsing == 0


def test_math_unpaired_dollar_is_one_non_parsing_span():
    md = "Price $x+y and nothing closes it."
    stats = metrics.math_stats(md)
    assert stats.total == 1
    assert stats.non_parsing == 1


def test_math_unclosed_display_block_is_non_parsing():
    md = "Start\n\n$$\nx = 1\n\nno close"
    stats = metrics.math_stats(md)
    assert stats.display == 1
    assert stats.non_parsing == 1


def test_count_tables_counts_blocks_with_separator_rows():
    md = (
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "Para\n\n"
        "| x | y | z |\n| :--- | :---: | ---: |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n\n"
        "| not | a table |\n| just | pipes |\n"
    )
    assert metrics.count_tables(md) == 2


def test_has_references_uses_section_kind():
    assert metrics.has_references("# Intro\n\nx\n\n## References\n\n- A. 2001.") is True
    assert metrics.has_references("# Intro\n\nx\n\n## Comparison with previous references\n") is False


def test_similarity_identical_is_exactly_one_and_ignores_page_markers():
    a = _paged("one two three", "four five")
    b = "one two three\n\nfour five"
    assert metrics.similarity(a, a) == 1.0
    assert metrics.similarity(a, b) == 1.0


def test_similarity_drops_for_different_text():
    a = "the quick brown fox jumps over the lazy dog " * 20
    b = "lorem ipsum dolor sit amet consectetur " * 20
    assert metrics.similarity(a, b) < 0.5
