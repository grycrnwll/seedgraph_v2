"""A synthetic library for the routine tests: held papers with sectioned markdown on disk.
Invented text only. The markdown has no page markers and enough words per section that
the text-quality gate passes it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from seedgraph.library import store

NOW = "2026-01-01T00:00:00+00:00"

INTRO_QUOTE = "We ask whether invented gadget subsidies raise gadget output in a made-up valley."
METHOD_QUOTE = "Our two-step gadget estimator compares subsidised and unsubsidised workshops."
ASSUMPTION_QUOTE = "Workshops do not anticipate the invented programme before it starts."
RESULT_QUOTE = "Gadget output rises by an invented eleven percent after the subsidy."
MU_QUOTE = "The dispersion parameter μ equals one half in the invented calibration."
DUPLICATE_SENTENCE = "Gadgets are counted twice."
REFERENCE_QUOTE = "Inventor, A. (2001). A made-up treatise on gadgets. Imaginary Press."


def filler(tag: str, n: int) -> str:
    """``n`` distinct invented sentences (about ten words each)."""
    return " ".join(f"Sentence {tag} number {i} describes an invented detail about gadgets here." for i in range(n))


def sections_markdown(body_sentences: int = 12, tag: int = 0) -> tuple[str, list[tuple[str, str, int, int]]]:
    """The markdown and its sections ``(heading, kind, start, end)``."""
    blocks = [
        ("Introduction", "body", f"# Introduction ({tag})\n\n{INTRO_QUOTE} {filler('intro', body_sentences)}\n\n"),
        ("Method", "body", f"## Method\n\n{METHOD_QUOTE} {ASSUMPTION_QUOTE} {DUPLICATE_SENTENCE} "
                           f"{filler('method', body_sentences)}\n\n"),
        ("Results", "body", f"## Results\n\n{RESULT_QUOTE} {MU_QUOTE} {DUPLICATE_SENTENCE} "
                            f"{filler('results', body_sentences)}\n\n"),
        ("References", "references", f"## References\n\n{REFERENCE_QUOTE}\n"),
    ]
    text, sections, pos = "", [], 0
    for heading, kind, block in blocks:
        sections.append((heading, kind, pos, pos + len(block)))
        text += block
        pos += len(block)
    return text, sections


@dataclass
class Paper:
    paper_id: str
    markdown_id: str
    markdown_hash: str
    source_file_id: str
    text: str


def _add_markdown(lib, conn, sf_id: str, text: str, tag: str) -> tuple[str, str]:
    raw = text.encode("utf-8")
    mh = hashlib.sha256(raw).hexdigest()
    rel = f"markdown/{mh}.md"
    (lib.root / "markdown").mkdir(parents=True, exist_ok=True)
    (lib.root / rel).write_bytes(raw)
    conn.execute(
        "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, "
        "converter_name, converter_package, converter_version, python_version, config_json, "
        "conversion_fingerprint, conversion_epistemic_type, run_status, markdown_hash, created_at) "
        "VALUES (?, ?, ?, 'marker', 'marker-pdf', '9.9.9', '3.13', '{}', 'fp', "
        "'local_deterministic_conversion', 'success', ?, ?)",
        (f"conv_{tag}_{mh[:8]}", sf_id, sf_id[3:], mh, NOW))
    conn.execute(
        "INSERT OR IGNORE INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
        "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'success', ?, ?)",
        (f"md_{mh}", f"conv_{tag}_{mh[:8]}", sf_id, mh, rel, len(raw), NOW))
    return f"md_{mh}", mh


def _add_sections(conn, md_id, mh, sf_id, sections) -> None:
    for ordinal, (heading, kind, start, end) in enumerate(sections, start=1):
        conn.execute(
            "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, source_file_id, "
            "source_file_hash, level, ordinal, heading_text, section_kind, start_char, end_char, "
            "section_parser_version, created_at) VALUES (?, ?, ?, ?, ?, 2, ?, ?, ?, ?, ?, 'v1', ?)",
            (f"sec_{mh[:12]}_{ordinal}", md_id, mh, sf_id, sf_id[3:], ordinal, heading, kind, start, end, NOW))


def make_paper(lib, n: int, *, text: str | None = None, sections=None, title: str | None = None,
               year: int | None = 2020) -> Paper:
    """A held paper ``lp_test<n>`` with its markdown and sections."""
    if text is None:
        text, sections = sections_markdown(tag=n)
    paper_id = f"lp_test{n:04d}"
    sf_hash = f"{n:064x}"
    sf_id = f"sf_{sf_hash}"
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, year, created_at) VALUES (?, ?, ?, ?)",
                     (paper_id, title or f"An invented gadget paper {n}", year, NOW))
        conn.execute("INSERT INTO source_files (source_file_id, file_hash, storage_uri, created_at) "
                     "VALUES (?, ?, ?, ?)", (sf_id, sf_hash, f"pdfs/{sf_hash}.pdf", NOW))
        md_id, mh = _add_markdown(lib, conn, sf_id, text, f"p{n}")
        conn.execute("INSERT INTO paper_files (paper_id, source_file_id, markdown_id, attached_at) "
                     "VALUES (?, ?, ?, ?)", (paper_id, sf_id, md_id, NOW))
        _add_sections(conn, md_id, mh, sf_id, sections or [])
    return Paper(paper_id, md_id, mh, sf_id, text)


def reconvert(lib, paper: Paper, new_text: str) -> str:
    """Point the paper at a new markdown (a re-conversion); return its markdown id."""
    with lib.write() as conn:
        md_id, _ = _add_markdown(lib, conn, paper.source_file_id, new_text, "re")
        conn.execute("UPDATE paper_files SET markdown_id = ? WHERE paper_id = ?", (md_id, paper.paper_id))
    return md_id


def claim(text: str, quote: str, label: str, **extra) -> dict:
    return {"claim_text": text, "normalized_label": label, "status": "found",
            "assertion_status": "stated", "confidence": 0.8, "exact_quote": quote, **extra}


def good_note(*, mu_quote: str = MU_QUOTE) -> dict:
    """A note that passes validate_note + check_note and whose quotes all anchor."""
    return {
        "research_question": claim("Do gadget subsidies raise output?", INTRO_QUOTE, "gadget subsidy effect"),
        "main_contribution": None,
        "setting": None,
        "estimand_or_target_object": None,
        "method_or_model": [claim("A two-step estimator.", METHOD_QUOTE, "two-step estimator",
                                  method_type="estimator", plain_words="Compare workshops with and without it.")],
        "data_sources": [],
        "assumptions": [claim("No anticipation.", ASSUMPTION_QUOTE, "no anticipation",
                              assumption_type="identification",
                              plain_words="Workshops do not change before the programme.")],
        "main_results": [claim("Output rises.", RESULT_QUOTE, "output rises", result_type="empirical"),
                         claim("The parameter is one half.", mu_quote, "dispersion one half",
                               result_type="descriptive")],
        "limitations": [],
        "robustness_checks": [],
        "open_questions": [],
    }


def empty_note(**arrays) -> dict:
    note = {"research_question": None, "main_contribution": None, "setting": None,
            "estimand_or_target_object": None, "method_or_model": [], "data_sources": [],
            "assumptions": [], "main_results": [], "limitations": [], "robustness_checks": [],
            "open_questions": []}
    note.update(arrays)
    return note


def open_lib(tmp_path):
    return store.open_library_at(tmp_path / "library")
