"""Claim identity across extractions (stage 3 design §4.4): the tiers, one-to-one
deterministic matching, ambiguous ties and the original-type rule. Pure: invented
fingerprints, no library."""

from __future__ import annotations

from seedgraph.notes import identity
from seedgraph.notes.identity import ClaimFingerprint

MD_A = "a" * 64
MD_B = "b" * 64


def fp(key, claim_type="method", *, field="method_or_model[0]", label="widget estimator",
       text="a two step widget estimator", span=None, md=MD_A):
    root = field.split("[")[0]
    tokens = frozenset(text.split())
    if span is None:
        return ClaimFingerprint(key=key, claim_type=claim_type, field_root=root, label_fold=label,
                                text_tokens=tokens, field_key=field)
    return ClaimFingerprint(key=key, claim_type=claim_type, field_root=root, label_fold=label,
                            text_tokens=tokens, field_key=field, quote_hash=f"q{span}",
                            markdown_hash=md, start_char=span[0], end_char=span[1])


def pairs(m):
    return {p.old: (p.new, p.tier) for p in m.pairs}


# --- the tiers -------------------------------------------------------------------------------

def test_tier_s_matches_a_scalar_by_its_field_whatever_its_wording():
    old = [fp("o", "research_question", field="research_question", label="", text="do widgets grow")]
    new = [fp("n", "research_question", field="research_question", label="", text="entirely new words")]
    assert pairs(identity.match(old, new)) == {"o": ("n", "S")}


def test_tier_1_overlapping_quotes_and_the_same_label():
    old = [fp("o", span=(100, 200), text="old words here")]
    new = [fp("n", span=(120, 210), text="quite different phrasing now")]
    assert pairs(identity.match(old, new)) == {"o": ("n", "1")}


def test_tier_2_overlapping_quotes_and_similar_text_with_a_new_label():
    old = [fp("o", span=(100, 200), label="widget estimator")]
    new = [fp("n", span=(100, 190), label="double count estimator")]
    assert pairs(identity.match(old, new)) == {"o": ("n", "2")}


def test_quotes_that_overlap_too_little_do_not_match():
    old = [fp("o", span=(100, 200))]
    new = [fp("n", span=(180, 300))]  # IoU 20/200
    assert identity.tier(old[0], new[0]) is None  # (the same unique label is tier 4)


def test_quotes_on_different_markdowns_never_overlap():
    old = [fp("o", span=(100, 200), md=MD_A)]
    new = [fp("n", span=(100, 200), md=MD_B)]
    assert identity.tier(old[0], new[0]) is None  # (the same unique label is tier 4)


def test_tier_3_unanchored_claims_with_the_same_label_and_similar_text():
    old = [fp("o", "general_assumption", field="assumptions[1]", label="exact counting",
              text="widgets are counted without error")]
    new = [fp("n", "general_assumption", field="assumptions[4]", label="exact counting",
              text="widgets are counted without any error")]
    assert pairs(identity.match(old, new)) == {"o": ("n", "3")}


def test_tier_3_needs_the_same_label():
    old = [fp("o", label="exact counting", text="widgets are counted without error")]
    new = [fp("n", label="error free counts", text="widgets are counted without error")]
    assert identity.match(old, new).pairs == ()


def test_tier_3_needs_similar_text():
    old = [fp("o", text="widgets are counted without error")]
    new = [fp("n", text="gadgets shrink in winter")]
    assert identity.tier(old[0], new[0]) is None  # (the same unique label is tier 4)


def test_an_anchored_claim_does_not_match_an_unanchored_one_by_tier_3():
    old = [fp("o", span=(100, 200))]
    new = [fp("n")]
    assert identity.tier(old[0], new[0]) is None  # (the same unique label is tier 4)


def test_a_different_claim_type_never_matches():
    old = [fp("o", "method", span=(100, 200))]
    new = [fp("n", "limitation", field="limitations[0]", span=(100, 200))]
    assert identity.match(old, new).pairs == ()


def test_jaccard_and_iou_helpers():
    assert identity.iou((0, 10), (5, 15)) == 5 / 15
    assert identity.jaccard(frozenset("ab"), frozenset("bc")) == 1 / 3
    assert identity.jaccard(frozenset(), frozenset()) == 1.0


# --- matching --------------------------------------------------------------------------------

def test_matching_is_one_to_one_and_prefers_the_better_tier():
    old = [fp("o1", span=(100, 200)), fp("o2", span=(100, 200), label="something else")]
    new = [fp("n1", span=(100, 200))]
    m = identity.match(old, new)
    assert pairs(m) == {"o1": ("n1", "1")}
    assert m.unmatched_old == ("o2",)
    assert m.unmatched_new == ()


def test_matching_prefers_the_larger_overlap():
    old = [fp("o", span=(100, 200))]
    new = [fp("n_small", span=(140, 220), field="method_or_model[0]"),
           fp("n_big", span=(105, 200), field="method_or_model[1]")]
    m = identity.match(old, new)
    assert pairs(m) == {"o": ("n_big", "1")}
    assert m.unmatched_new == ("n_small",)


def test_matching_is_deterministic_whatever_the_input_order():
    old = [fp(f"o{i}", span=(100 * i, 100 * i + 80), label=f"l{i}") for i in range(1, 6)]
    new = [fp(f"n{i}", span=(100 * i + 5, 100 * i + 85), label=f"l{i}", field=f"method_or_model[{i}]")
           for i in range(1, 6)]
    a = identity.match(old, new)
    b = identity.match(list(reversed(old)), list(reversed(new)))
    assert a == b
    assert pairs(a) == {f"o{i}": (f"n{i}", "1") for i in range(1, 6)}


def test_field_order_breaks_ties_numerically_not_as_strings():
    # equal scores against two new claims would be ambiguous; distinct scores are not
    old = [fp("o", span=(0, 100))]
    new = [fp("n10", span=(0, 90), field="method_or_model[10]"),
           fp("n2", span=(10, 100), field="method_or_model[2]")]
    assert identity.new_order(new[0]) > identity.new_order(new[1])
    m = identity.match(old, new)
    assert m.ambiguous and set(m.ambiguous[0].new) == {"n10", "n2"}


def test_a_true_tie_is_flagged_ambiguous_and_never_picked():
    old = [fp("o", span=(100, 200))]
    new = [fp("n1", span=(100, 200), field="method_or_model[0]", md=MD_A),
           fp("n2", span=(100, 200), field="method_or_model[1]", md=MD_A)]
    m = identity.match(old, new)
    assert m.pairs == ()
    [amb] = m.ambiguous
    assert amb.old == ("o",) and amb.new == ("n1", "n2") and amb.tier == "1"
    assert m.unmatched_old == () and m.unmatched_new == ()


def test_two_old_claims_tied_on_one_new_claim_are_ambiguous():
    old = [fp("o1", "general_assumption", field="assumptions[0]", label="exact counting",
              text="widgets are counted"),
           fp("o2", "general_assumption", field="assumptions[1]", label="exact counting",
              text="widgets are counted")]
    new = [fp("n", "general_assumption", field="assumptions[0]", label="exact counting",
              text="widgets are counted")]
    [amb] = identity.match(old, new).ambiguous
    assert amb.old == ("o1", "o2") and amb.new == ("n",)


def test_a_retyped_claim_matches_on_its_original_type():
    """The fingerprint holds the agent's original type: the researcher's re-type is in the op."""
    old = [fp("o", "general_assumption", field="assumptions[0]", span=(100, 200), label="no anticipation")]
    new = [fp("n_orig", "general_assumption", field="assumptions[2]", span=(100, 200), label="no anticipation"),
           fp("n_retyped", "identification_assumption", field="assumptions[0]", span=(100, 200),
              label="no anticipation")]
    assert pairs(identity.match(old, new)) == {"o": ("n_orig", "1")}


def test_from_dict_reads_an_overlay_fingerprint():
    d = {"claim_type": "method", "field_root": "method_or_model", "label_fold": "widget estimator",
         "text_tokens": ["a", "estimator"], "quote_hash": "q", "markdown_hash": MD_A,
         "start_char": 3, "end_char": 9}
    f = ClaimFingerprint.from_dict("k", d, field_key="method_or_model[0]")
    assert f.anchored and f.text_tokens == frozenset({"a", "estimator"}) and f.span == (3, 9)
    assert not ClaimFingerprint.from_dict("k", {**d, "quote_hash": None}).anchored


def test_relocated_moves_a_fingerprint_onto_another_markdown():
    f = fp("o", span=(100, 200), md=MD_A)
    g = f.relocated(MD_B, (40, 140))
    assert (g.markdown_hash, g.span, g.key) == (MD_B, (40, 140), "o")
    lost = f.relocated(MD_B, None)
    assert not lost.anchored  # an old quote that no longer locates counts as unanchored


# --- tiers 4 and 5: label alone, then strong text overlap (real-data misses) ------------------

def words(text):
    """Folded tokens as ``overlay.fingerprint`` stores them (``semantic.merge.normalize``)."""
    from seedgraph.semantic.merge import normalize
    return " ".join(normalize(text).split())


def test_tier_4_the_same_label_matches_when_the_quotes_are_different_sentences():
    # invented analogue of a re-worded model claim quoting a different sentence
    old = [fp("o", "model", label=words("Quux-Fringe Widget Function"), span=(100, 200),
              text=words("The paper builds a modified quux-fringe widget function in which delays act like "
                         "a tax on widget inputs and output falls"))]
    new = [fp("n", "model", label=words("quux fringe widget function"), span=(900, 980),
              text=words("A modified daily quux-fringe model where outages behave as a varying input levy"))]
    assert pairs(identity.match(old, new)) == {"o": ("n", "4")}


def test_tier_4_matches_an_anchored_claim_to_an_unanchored_one():
    old = [fp("o", label="widget function", span=(100, 200), text="one thing")]
    new = [fp("n", label="widget function", text="something else entirely")]
    assert pairs(identity.match(old, new)) == {"o": ("n", "4")}


def test_tier_4_needs_the_label_unique_on_both_sides():
    old = [fp("o", label="widget function", span=(100, 200), text="one thing"),
           fp("o2", label="widget function", span=(300, 400), text="other thing")]
    new = [fp("n", label="widget function", span=(900, 980), text="something else entirely")]
    m = identity.match(old, new)
    assert m.pairs == () and m.ambiguous == ()


def test_tier_5_a_strong_text_overlap_matches_whatever_the_label():
    # invented analogue of a shorter re-wording with a new label, a hyphen and fixed/constant
    old = [fp("o", "limitation", field="limitations[0]", label=words("Simulation Scope Limitation"),
              span=(100, 300),
              text=words("The pause-contract simulation holds constant the gadget sector's total quux "
                         "allotment, though saved quux could be allotted to other sectors"))]
    new = [fp("n", "limitation", field="limitations[2]", label=words("partial equilibrium counterfactual"),
              span=(700, 790),
              text=words("The pause contract simulation holds the gadget sector's total quux allotment fixed."))]
    m = identity.match(old, new)
    assert pairs(m) == {"o": ("n", "5")}
    assert m.pairs[0].text_overlap == 1.0


def test_overlap_coefficient_normalises_stopwords_hyphens_and_fixed_constant():
    a = identity.content_words(frozenset(words("holds constant the pause-contract allotment").split()))
    b = identity.content_words(frozenset(words("holds the pause contract allotment fixed").split()))
    assert a == b == frozenset({"holds", "constant", "pause", "contract", "allotment"})
    assert identity.overlap_coefficient(a, frozenset({"holds", "pause", "zebra"})) == 2 / 3


def test_two_different_limitations_sharing_some_words_do_not_match():
    old = [fp("o", "limitation", field="limitations[0]", label="sample coverage", span=(100, 200),
              text=words("The invented sample covers only urban widget firms in one region"))]
    new = [fp("n", "limitation", field="limitations[0]", label="measurement error", span=(500, 600),
              text=words("Invented widget prices in the sample are measured with error"))]
    assert identity.match(old, new).pairs == ()


def test_tier_5_without_a_clear_margin_is_ambiguous():
    old = [fp("o", "limitation", field="limitations[0]", label="a", span=(0, 10),
              text=words("the gadget simulation holds total quux allotment constant across sectors"))]
    new = [fp("n1", "limitation", field="limitations[0]", label="b", span=(50, 60),
              text=words("gadget simulation holds total quux allotment constant")),       # 7/7
           fp("n2", "limitation", field="limitations[1]", label="c", span=(70, 80),
              text=words("gadget simulation holds total quux allotment fixed widely"))]  # 7/8
    m = identity.match(old, new)
    assert m.pairs == ()
    [amb] = m.ambiguous
    assert amb.tier == "5" and set(amb.new) == {"n1", "n2"}


def test_tier_5_with_a_clear_margin_takes_the_best():
    old = [fp("o", "limitation", field="limitations[0]", label="a", span=(0, 10),
              text=words("the gadget simulation holds total quux allotment constant across sectors"))]
    new = [fp("n1", "limitation", field="limitations[0]", label="b", span=(50, 60),
              text=words("gadget simulation holds total quux allotment constant")),           # 7/7
           fp("n2", "limitation", field="limitations[1]", label="c", span=(70, 80),
              text=words("gadget simulation holds total quux allotment zebra yak"))]          # 6/8
    assert pairs(identity.match(old, new)) == {"o": ("n1", "5")}


def test_tier_5_ignores_very_short_texts():
    old = [fp("o", "limitation", field="limitations[0]", label="a", span=(0, 10), text="widget bias")]
    new = [fp("n", "limitation", field="limitations[0]", label="b", span=(50, 60),
              text="widget bias appears in many invented places")]
    assert identity.match(old, new).pairs == ()


def test_earlier_tiers_still_win_over_tiers_4_and_5():
    old = [fp("o", span=(100, 200))]
    new = [fp("n_label", span=(900, 990), field="method_or_model[0]"),
           fp("n_quote", span=(100, 200), label="other label", field="method_or_model[1]")]
    assert pairs(identity.match(old, new)) == {"o": ("n_quote", "2")}


def test_tier_1_label_comparison_is_casefolded_by_the_fingerprint():
    from seedgraph.semantic.merge import normalize
    assert normalize("Cobb-Douglas Production Function") == normalize("cobb douglas  production function")
