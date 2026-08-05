"""Build F chunk 3 — references-heading detection completions (§5.8), pure-function.

Chunk 0 pinned the §5.8 gaps as executable evidence (Check B as a plain behavior
pin; Checks C/D as ``xfail(strict=True)`` reproductions). Chunk 3 lands the fix in
``sections.parser`` (Part 1: the ``reference list`` / generic ``… cited`` cue; Part 2:
prefix-tolerant, start-anchored cue matching) and ``citation.parsed_bib`` (Part 3: the
non-ATX fallback, exercised end-to-end in ``tests/test_phase_3b.py``), so this module
now:

* keeps **Check B** (§5.8a) as a plain behavior pin — the sections parser stays
  ATX-only (decisions 31/52), so a bold-only ``**REFERENCES**`` or bare ``References``
  line still yields no ``section_kind='references'`` section; the recovery lives in
  ``citation/parsed_bib.py``, not here;
* FLIPS **Check C** (§5.8b) and **Check D** (§5.8c) to plain passing assertions — the
  ``reference list`` cue and the start-anchored, prefix-stripped matcher are in;
* adds the **positive real-marker battery** (mirroring v1
  ``test_real_marker_heading_shapes_are_sectioned``, ``tests/test_bib_parser.py:163``):
  the numbered / bold-wrapped / anchor-span heading shapes marker actually emits MUST
  still classify ``references`` under the anchored match. These pass at HEAD under the
  old substring search and MUST NOT regress under anchoring — this battery is the ONLY
  gate against that regression (the chunk-4 corpus gate structurally cannot see it, per
  the plan's risk 4, because it synthesizes its own clean ``## References`` heading);
* adds the **false-positive battery** (mirroring v1 §6.3): prose headings that merely
  mention a references cue mid-line stay ``body``.

All checks are pure functions (``parse_sections`` / ``_classify_kind``) with no DB
scaffolding. The deferred end-to-end Check-B assertions (a bold-heading document parses
entries via the ``parsed_bib`` fallback) live in ``tests/test_phase_3b.py``, where the
project-DB scaffolding already exists.
"""

from __future__ import annotations

import pytest

from seedgraph.sections.parser import _classify_kind, parse_sections


def _parse(markdown: str) -> list:
    """``parse_sections`` with dummy ids — pure function, never touches a store."""
    return parse_sections(
        markdown,
        markdown_id="md_x",
        markdown_hash="h_x",
        source_file_id="src_x",
        source_file_hash="sh_x",
        work_id="w_x",
    )


# --------------------------------------------------------------------------
# Check B (§5.8a) — plain behavior pin (never flips; sections parser is ATX-only).
# --------------------------------------------------------------------------

def test_check_b_bold_only_references_heading_yields_no_references_section():
    """A bold-only ``**REFERENCES**`` heading produces no ``references`` section.

    ``_ATX`` matches only ``#``-prefixed lines, so a bold-emphasis line is never a
    heading and the whole doc falls into one level-0 preamble. Documents why the
    citations path needs the ``parsed_bib`` fallback for such docs; the sections
    parser stays ATX-only (decisions 31/52), so this assertion never flips.
    """
    md = "Some intro text.\n\n**REFERENCES**\n\n[1] Smith, J. (2020). Title. Journal.\n"
    secs = _parse(md)
    assert not any(s.section_kind == "references" for s in secs)


def test_check_b_bare_title_line_references_heading_yields_no_references_section():
    """A bare ``References`` title line (no ``#``) produces no ``references`` section.

    Same ATX-only rationale as the bold-only variant: a plain title line is not an ATX
    heading, so no ``section_kind='references'`` range is created.
    """
    md = "Some intro text.\n\nReferences\n\n[1] Smith, J. (2020). Title. Journal.\n"
    secs = _parse(md)
    assert not any(s.section_kind == "references" for s in secs)


# --------------------------------------------------------------------------
# Check C (§5.8b) — missing-cue gap, now FIXED (Part 1 adds the `reference list` cue).
# --------------------------------------------------------------------------

def test_check_c_reference_list_classifies_references():
    """``_classify_kind('Reference List')`` classifies ``references`` (the singular
    ``Reference`` + ``reference list`` cue, absent from the old ``references`` word)."""
    assert _classify_kind("Reference List") == "references"


# --------------------------------------------------------------------------
# Check D (§5.8c) — false-positive gap, now FIXED (Part 2's start-anchored cue).
# --------------------------------------------------------------------------

def test_check_d_comparison_with_previous_references_classifies_body():
    """``_classify_kind('Comparison with previous references')`` classifies ``body``:
    the cue is mid-line, not at the start (after ≤1 qualifier word)."""
    assert _classify_kind("Comparison with previous references") == "body"


# --------------------------------------------------------------------------
# POSITIVE real-marker battery — mirrors v1 test_real_marker_heading_shapes_are_sectioned
# (tests/test_bib_parser.py:163). The numbered / bold-wrapped / anchor-span shapes marker
# actually emits MUST still classify 'references' under the anchored match. THIS is the
# only regression gate for the prefix-strip-then-anchor design (risk 4).
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "heading",
    [
        "# **REFERENCES**",  # bold-wrapped ATX (most common live shape)
        "### 6. REFERENCES",  # section-numbered heading
        "#### 7. References",  # section-numbered, mixed case
        '# <span id="page-8-0"></span>**References**',  # anchor span + bold
        '### <span id="page-13-11"></span>References',  # anchor span, no bold
    ],
)
def test_real_marker_heading_shapes_classify_references(heading):
    """Each real-marker references heading shape yields exactly one ``references``
    section (prefix-stripped, then start-anchored cue)."""
    md = (
        "# Body\n\nSome prose that must not be sectioned as references.\n\n"
        f"{heading}\n\n"
        "Smith, J. (1999). A study of widgets. Journal of Widgets, 12(3), 134-141.\n"
    )
    secs = _parse(md)
    refs = [s for s in secs if s.section_kind == "references"]
    assert len(refs) == 1


@pytest.mark.parametrize(
    "heading",
    [
        "References",
        "Selected References",  # one qualifier word before the cue
        "Rejoinder References",
        "Bibliography",
        "Works Cited",
        "Literature Cited",
        "Reference List",
        "References Cited",  # the generic '… cited' cue
    ],
)
def test_qualified_and_plain_references_headings_classify_references(heading):
    """Plain and single-qualifier references headings classify ``references``."""
    assert _classify_kind(heading) == "references"


# --------------------------------------------------------------------------
# FALSE-POSITIVE battery — mirrors v1 §6.3. A prose heading whose references cue sits
# mid-line (not at the start, after ≤1 qualifier word) must stay 'body'.
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "heading",
    [
        "Comparison with previous references",
        "A note on the references and sources we used",
        "How our results relate to the bibliography",
        "Reply to comments and discussion",
        "Discussion",
        "Notes",
    ],
)
def test_prose_headings_with_reference_cue_midline_stay_body(heading):
    """A prose heading mentioning a references cue mid-line stays ``body`` — the cue is
    start-anchored, so it never opens a bibliography here."""
    assert _classify_kind(heading) == "body"


def test_prefix_stripped_numbered_heading_still_needs_cue_at_start():
    """A numbered heading whose text is NOT a references cue after the number is stripped
    stays ``body`` (the number strip must not manufacture a false positive)."""
    assert _classify_kind("5.1 Introduction to the references we cite") == "body"
