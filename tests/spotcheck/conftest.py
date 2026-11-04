"""Spot-check fixtures: a real temporary library with invented papers and agent notes.

Each paper gets its own markdown (``markdown_id`` is the text's hash) holding the same
textbook sentences, so every quote anchors. The model is set per note through the save's
provenance (``client`` is stored as the run's provider)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import pytest

from seedgraph.library import graphs, store
from seedgraph.notes import format as fmt
from seedgraph.notes import save

NOW = "2026-01-01T00:00:00+00:00"

RQ_QUOTE = "We estimate the ATE of an invented training programme on earnings."
METHOD_QUOTE = "We use a difference-in-differences design with two periods."
TRENDS_QUOTE = "Treated and control units follow parallel trends before the programme."
RESULT_QUOTE = "Earnings rise after the programme in the treated group."

MARKDOWN = (
    "# An invented study {n}\n\n"
    "{rq}\n\n"
    "## Method\n\n"
    "{method}\n"
    "{trends}\n\n"
    "## Results\n\n"
    "{result}\n"
    "Serial correlation in the errors is handled by clustering.\n"
)


def raw_note(*, results: int = 1) -> dict:
    """A note with a research question, a method, two assumptions and ``results`` results."""
    return {
        "research_question": {"claim_text": "What is the ATE of the invented programme?", "status": "found",
                              "exact_quote": RQ_QUOTE},
        "method_or_model": [{"claim_text": "A two-period difference-in-differences design.",
                             "normalized_label": "difference in differences", "status": "found",
                             "method_type": "estimator",
                             "plain_words": "Compare the change in treated units with the change in controls.",
                             "exact_quote": METHOD_QUOTE}],
        "assumptions": [{"claim_text": "Parallel trends between treated and control units.",
                         "normalized_label": "parallel trends", "status": "found",
                         "assumption_type": "identification", "exact_quote": TRENDS_QUOTE},
                        {"claim_text": "No serial correlation remains after clustering.",
                         "normalized_label": "no serial correlation", "status": "inferred",
                         "assumption_type": "identification",
                         "inferred_explanation": "invented reason"}],
        "main_results": [{"claim_text": f"Earnings rise after the programme (result {i}).",
                          "normalized_label": f"earnings rise {i}", "status": "found",
                          "result_type": "main", "exact_quote": RESULT_QUOTE}
                         for i in range(results)],
    }


@dataclass
class World:
    lib: store.Library
    n: int = 0

    def paper(self, *, title: str | None = None, year: int = 2020) -> str:
        self.n += 1
        n = self.n
        pid = f"lp_spot{n:016d}"
        text = MARKDOWN.format(n=n, rq=RQ_QUOTE, method=METHOD_QUOTE, trends=TRENDS_QUOTE, result=RESULT_QUOTE)
        raw = text.encode("utf-8")
        mh = hashlib.sha256(raw).hexdigest()
        rel = f"markdown/{mh}.md"
        (self.lib.root / "markdown").mkdir(parents=True, exist_ok=True)
        (self.lib.root / rel).write_bytes(raw)
        sf_hash = f"{n:064x}"
        sf_id = f"sf_{sf_hash}"
        with self.lib.write() as conn:
            conn.execute("INSERT INTO papers (paper_id, title, year, created_at) VALUES (?, ?, ?, ?)",
                         (pid, title or f"An invented study {n}", year, NOW))
            conn.execute("INSERT INTO source_files (source_file_id, file_hash, storage_uri, created_at) "
                         "VALUES (?, ?, ?, ?)", (sf_id, sf_hash, f"pdfs/{sf_hash}.pdf", NOW))
            conn.execute(
                "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, "
                "converter_name, converter_package, converter_version, python_version, config_json, "
                "conversion_fingerprint, conversion_epistemic_type, run_status, markdown_hash, created_at) "
                "VALUES (?, ?, ?, 'marker', 'marker-pdf', '9.9.9', '3.13', '{}', 'fp', "
                "'local_deterministic_conversion', 'success', ?, ?)", (f"conv_{n}", sf_id, sf_hash, mh, NOW))
            conn.execute(
                "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
                "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'success', ?, ?)",
                (f"md_{mh}", f"conv_{n}", sf_id, mh, rel, len(raw), NOW))
            conn.execute("INSERT INTO paper_files (paper_id, source_file_id, markdown_id, attached_at) "
                         "VALUES (?, ?, ?, ?)", (pid, sf_id, f"md_{mh}", NOW))
        return pid

    def note(self, pid: str, *, model: str | None = "invented-model-a", client: str | None = "claude",
             raw: dict | None = None) -> save.SavedNote:
        raw = raw or raw_note()
        note, errors = fmt.validate_note(raw)
        assert errors == []
        norm = fmt.normalize_parts([note])
        md = self.lib.conn.execute("SELECT f.markdown_id, m.markdown_hash FROM paper_files f "
                                   "JOIN markdown_documents m ON m.markdown_id = f.markdown_id "
                                   "WHERE f.paper_id = ?", (pid,)).fetchone()
        with self.lib.write() as conn:
            return save.save_note(
                conn, paper_id=pid, markdown_id=md[0], markdown_hash=md[1], drafts=norm.drafts,
                note_text=norm.note_text, archetype=norm.archetype, raw_note_json=json.dumps(raw),
                provenance=save.NoteProvenance(client=client, model=model))

    def noted_paper(self, **kw) -> str:
        pid = self.paper()
        self.note(pid, **kw)
        return pid

    def claims(self, note_id: str) -> dict[str, str]:
        """field_key -> claim_id of one note."""
        return {r[0]: r[1] for r in self.lib.conn.execute(
            "SELECT field_key, claim_id FROM extracted_claims WHERE structured_note_id = ?", (note_id,))}

    def graph(self, slug: str, members: list[str]) -> None:
        with graphs.create_graph(self.lib, slug) as g:
            for pid in members:
                graphs.set_membership(g, pid, "member")


@pytest.fixture
def lib(tmp_path):
    library = store.open_library_at(tmp_path / "library")
    yield library
    library.close()


@pytest.fixture
def world(lib):
    return World(lib)
