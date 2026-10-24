"""A real (temporary) library with one held paper whose markdown is on disk. Invented text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pytest

from seedgraph.library import store

NOW = "2026-01-01T00:00:00+00:00"

MARKDOWN = (
    "# An invented paper on widgets\n\n"
    "{0}------------------------------------------------\n\n"
    "We study whether invented widgets grow in a made-up market.\n\n"
    "## Method\n\n"
    "Our two-step widget estimator counts widgets twice and averages.\n"
    "Units do not anticipate the invented programme before it starts.\n\n"
    "{1}------------------------------------------------\n\n"
    "## Results\n\n"
    "Widgets grow by an invented amount. Widgets grow by an invented amount.\n"
    "The micro-sign µ appears once in this invented text.\n"
)


@dataclass
class PaperFixture:
    lib: store.Library
    paper_id: str
    markdown_id: str
    markdown_hash: str
    source_file_id: str
    text: str


def add_markdown_file(lib, conn, sf_id, text, n):
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
        (f"conv_{n}", sf_id, sf_id[3:], mh, NOW),
    )
    conn.execute(
        "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
        "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
        "VALUES (?, ?, ?, ?, ?, 'success', ?, ?)",
        (f"md_{mh}", f"conv_{n}", sf_id, mh, rel, len(raw), NOW),
    )
    return f"md_{mh}", mh


def make_paper(lib, *, paper_id="lp_invented0000000001", n=1, text=MARKDOWN):
    sf_hash = f"{n:064x}"
    sf_id = f"sf_{sf_hash}"
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES (?, ?, ?)",
                     (paper_id, "An invented paper on widgets", NOW))
        conn.execute(
            "INSERT INTO source_files (source_file_id, file_hash, storage_uri, created_at) "
            "VALUES (?, ?, ?, ?)", (sf_id, sf_hash, f"pdfs/{sf_hash}.pdf", NOW))
        md_id, mh = add_markdown_file(lib, conn, sf_id, text, n)
        conn.execute(
            "INSERT INTO paper_files (paper_id, source_file_id, markdown_id, attached_at) "
            "VALUES (?, ?, ?, ?)", (paper_id, sf_id, md_id, NOW))
        if "## Method" in text:
          conn.execute(
            "INSERT INTO document_sections (section_id, markdown_id, markdown_hash, source_file_id, "
            "source_file_hash, level, ordinal, heading_text, section_kind, start_char, end_char, "
            "section_parser_version, created_at) VALUES (?, ?, ?, ?, ?, 2, 1, 'Method', 'body', ?, ?, 'v1', ?)",
            (f"sec_{n}_method", md_id, mh, sf_id, sf_hash, text.index("## Method"),
             text.index("{1}"), NOW),
        )
    return PaperFixture(lib=lib, paper_id=paper_id, markdown_id=md_id, markdown_hash=mh,
                        source_file_id=sf_id, text=text)


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def paper(lib):
    return make_paper(lib)
