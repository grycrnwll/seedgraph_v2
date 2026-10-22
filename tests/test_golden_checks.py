"""Golden-set per-PDF comparison: OLD vs NEW markdown -> structured result."""

from __future__ import annotations

from seedgraph.golden.checks import Rendition, compare


def _doc(n_pages: int = 3, *, refs: bool = True, extra: str = "", tables: int = 1, math: str = "$a$ $b$") -> str:
    pages = []
    for i in range(n_pages):
        body = f"Paragraph number {i} with quote-{i} in it."
        if i == 0:
            body += "\n\n" + math + "\n\n" + "\n\n".join("| h |\n|---|\n| v |" for _ in range(tables))
        if i == n_pages - 1 and refs:
            body += "\n\n## References\n\n- Someone. 1999."
        pages.append(f"{{{i}}}------------------------------------------------\n\n{body}")
    return "\n\n".join(pages) + extra


QUOTES = ["quote-0", "quote-1", "quote-2"]


def _by_name(result):
    return {c.name: c for c in result.checks}


def test_identical_renditions_pass_everything():
    md = _doc()
    old = Rendition(md, seconds=30.0)
    new = Rendition(md, seconds=30.0)
    result = compare("GS99", old, new, page_count=3, quotes=QUOTES)
    checks = _by_name(result)
    assert result.status == "pass", [(c.name, c.status, c.reason) for c in result.checks]
    assert set(checks) == {"page_markers", "references", "quotes", "equations", "tables", "speed", "similarity"}
    assert checks["quotes"].old == 1.0 and checks["quotes"].new == 1.0
    assert checks["similarity"].new == 1.0
    assert checks["page_markers"].new == 3


def test_page_marker_count_mismatch_fails():
    result = compare("GS99", Rendition(_doc(3)), Rendition(_doc(2)), page_count=3, quotes=QUOTES)
    assert _by_name(result)["page_markers"].status == "fail"
    assert result.status == "fail"


def test_non_contiguous_markers_fail():
    bad = _doc(3).replace("{1}---", "{5}---")
    result = compare("GS99", Rendition(_doc(3)), Rendition(bad), page_count=3, quotes=QUOTES)
    assert _by_name(result)["page_markers"].status == "fail"


def test_unknown_page_count_warns_not_fails():
    result = compare("GS99", Rendition(_doc(3)), Rendition(_doc(3)), page_count=None, quotes=QUOTES)
    assert _by_name(result)["page_markers"].status == "warn"


def test_missing_references_in_new_fails():
    result = compare("GS99", Rendition(_doc()), Rendition(_doc(refs=False)), page_count=3, quotes=QUOTES)
    assert _by_name(result)["references"].status == "fail"


def test_references_missing_in_both_only_warns():
    result = compare("GS99", Rendition(_doc(refs=False)), Rendition(_doc(refs=False)), page_count=3, quotes=QUOTES)
    assert _by_name(result)["references"].status == "warn"


def test_quote_rate_drop_over_one_point_fails():
    quotes = [f"quote-{i}" for i in range(3)]
    new_md = _doc().replace("quote-2", "qu0te-2")
    result = compare("GS99", Rendition(_doc()), Rendition(new_md), page_count=3, quotes=quotes)
    q = _by_name(result)["quotes"]
    assert q.status == "fail"
    assert q.old == 1.0
    assert abs(q.new - 2 / 3) < 1e-9


def test_quote_ambiguous_in_new_counts_as_unmatched():
    new_md = _doc(extra="\n\nquote-1 again")
    result = compare("GS99", Rendition(_doc()), Rendition(new_md), page_count=3, quotes=QUOTES)
    assert _by_name(result)["quotes"].status == "fail"


def test_no_quotes_warns():
    result = compare("GS99", Rendition(_doc()), Rendition(_doc()), page_count=3, quotes=[])
    assert _by_name(result)["quotes"].status == "warn"


def test_more_non_parsing_math_fails():
    result = compare(
        "GS99", Rendition(_doc()), Rendition(_doc(math="$a$ $\\frac{b$")), page_count=3, quotes=QUOTES
    )
    assert _by_name(result)["equations"].status == "fail"


def test_math_total_falling_over_two_percent_fails():
    old = _doc(math=" ".join(["$x$"] * 100))
    new = _doc(math=" ".join(["$x$"] * 97))
    result = compare("GS99", Rendition(old), Rendition(new), page_count=3, quotes=QUOTES)
    assert _by_name(result)["equations"].status == "fail"
    within = _doc(math=" ".join(["$x$"] * 98))
    result = compare("GS99", Rendition(old), Rendition(within), page_count=3, quotes=QUOTES)
    assert _by_name(result)["equations"].status == "pass"


def test_fewer_tables_fails():
    result = compare("GS99", Rendition(_doc(tables=2)), Rendition(_doc(tables=1)), page_count=3, quotes=QUOTES)
    t = _by_name(result)["tables"]
    assert (t.old, t.new, t.status) == (2, 1, "fail")


def test_speed_over_125_percent_fails_and_reports_ratio():
    result = compare(
        "GS99", Rendition(_doc(), seconds=30.0), Rendition(_doc(), seconds=39.0), page_count=3, quotes=QUOTES
    )
    s = _by_name(result)["speed"]
    assert s.status == "fail"
    assert s.old == 10.0 and s.new == 13.0
    assert abs(s.extra["ratio"] - 1.3) < 1e-9


def test_speed_within_bar_passes():
    result = compare(
        "GS99", Rendition(_doc(), seconds=30.0), Rendition(_doc(), seconds=37.5), page_count=3, quotes=QUOTES
    )
    assert _by_name(result)["speed"].status == "pass"


def test_missing_timing_warns():
    result = compare("GS99", Rendition(_doc(), seconds=None), Rendition(_doc(), seconds=5.0), page_count=3, quotes=QUOTES)
    assert _by_name(result)["speed"].status == "warn"


def test_low_similarity_only_warns():
    old = _doc()
    new = _doc(extra="\n\n" + "entirely different filler text line\n" * 200)
    result = compare("GS99", Rendition(old), Rendition(new), page_count=3, quotes=QUOTES)
    sim = _by_name(result)["similarity"]
    assert sim.status == "warn"
    assert result.status == "warn"


def test_result_serializes_to_plain_dict():
    result = compare("GS99", Rendition(_doc(), seconds=3.0), Rendition(_doc(), seconds=3.0), page_count=3, quotes=QUOTES)
    d = result.to_dict()
    assert d["id"] == "GS99" and d["status"] == "pass"
    assert {c["name"] for c in d["checks"]} >= {"quotes", "speed"}
    assert all({"old", "new", "status", "reason"} <= set(c) for c in d["checks"])
