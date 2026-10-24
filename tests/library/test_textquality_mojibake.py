"""The shared mojibake table and fold (design §5.2). All text is invented."""

from __future__ import annotations

from seedgraph.library.mojibake import FAMILIES, find_hits, fold, fold_with_map


def test_fold_repairs_family_a():
    assert fold("identiÖed di§erent e¢ ciency reáects the authorís view") == (
        "identified different efficiency reflects the author's view"
    )
    assert fold("The Örst step") == "The first step"


def test_fold_repairs_family_b():
    assert fold("the …rst di¤erence coe¢cient in…nity") == (
        "the first difference coefficient infinity"
    )


def test_fold_leaves_names_and_symbols_alone():
    text = "Sánchez, Martínez, Galán's work; § 3; 5¢ each; Weiß; ratio Þ; wait… then"
    folded = fold(text)
    assert "Martínez" in folded and "§ 3" in folded and "5¢ each" in folded
    assert "wait… then" in folded and "Weiß" in folded and "Þ" in folded


def test_fold_map_points_back_to_the_source():
    text = "an e¢ cient model"
    folded, spans = fold_with_map(text)
    i = folded.index("efficient")
    j = i + len("efficient")
    assert text[spans[i][0] : spans[j - 1][1]] == "e¢ cient"


def test_table_names_both_families_and_strengths():
    by_glyph = {g.glyph: g for g in FAMILIES}
    assert by_glyph["á"].strength == by_glyph["í"].strength == "corroborating"
    assert {g.glyph for g in FAMILIES if g.strength == "strong"} == set("Ö§¢…¤")


def test_ordinary_ellipsis_and_section_sign_are_not_hits():
    assert find_hits("see § 2 and so on… and on; costs 5¢ now") == []
