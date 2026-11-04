"""Invented libraries for the R2 record tests: held papers with markdown on disk, notes saved
through the stage 3 path (so spans, claim spans and overlays are real), graphs, and a
logging helper over :func:`seedgraph.records.log.log_event`. Textbook terms only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from seedgraph.library import graphs
from seedgraph.notes import format as fmt
from seedgraph.notes import overlay, save
from seedgraph.records import log
from seedgraph.records.sessions import CallInfo

NOW = "2026-01-01T00:00:00+00:00"
T0 = datetime(2026, 11, 4, 14, 32, 5, tzinfo=timezone.utc)
UTC = timezone.utc
PAGE = "------------------------------------------------"

DID_TEXT = (
    "# Difference in differences with invented counties\n\n"
    "{0}" + PAGE + "\n\n"
    "We estimate the effect of an invented minimum wage on teen employment.\n\n"
    "## Identification\n\n"
    "Treated and control counties would have followed parallel trends without the policy.\n"
    "We cluster standard errors by county to allow for serial correlation.\n\n"
    "{1}" + PAGE + "\n\n"
    "## Results\n\n"
    "The estimated ATE on teen employment is small and not significant.\n"
)
ATE_TEXT = (
    "# Matching estimators for an invented training programme\n\n"
    "{0}" + PAGE + "\n\n"
    "We ask whether an invented training programme raises earnings.\n\n"
    "## Method\n\n"
    "Nearest-neighbour matching on the propensity score estimates the ATE.\n"
    "Selection on observables makes the comparison credible.\n"
)

QUOTES = {
    "did_question": "We estimate the effect of an invented minimum wage on teen employment.",
    "parallel": "Treated and control counties would have followed parallel trends without the policy.",
    "serial": "We cluster standard errors by county to allow for serial correlation.",
    "did_result": "The estimated ATE on teen employment is small and not significant.",
    "ate_question": "We ask whether an invented training programme raises earnings.",
    "matching": "Nearest-neighbour matching on the propensity score estimates the ATE.",
    "unconfounded": "Selection on observables makes the comparison credible.",
}


def did_note():
    return {
        "research_question": {"claim_text": "Does an invented minimum wage change teen employment?",
                              "status": "found", "exact_quote": QUOTES["did_question"]},
        "method_or_model": [{"claim_text": "A difference-in-differences design over counties.",
                             "normalized_label": "difference in differences", "status": "found",
                             "method_type": "estimator", "exact_quote": QUOTES["serial"]}],
        "assumptions": [{"claim_text": "Parallel trends between treated and control counties.",
                         "normalized_label": "parallel trends", "status": "found",
                         "assumption_type": "identification", "exact_quote": QUOTES["parallel"]},
                        {"claim_text": "No spillovers across county lines.",
                         "normalized_label": "no spillovers", "status": "inferred",
                         "assumption_type": "identification", "inferred_explanation": "invented reason"}],
    }


def ate_note():
    return {
        "research_question": {"claim_text": "Does an invented training programme raise earnings?",
                              "status": "found", "exact_quote": QUOTES["ate_question"]},
        "method_or_model": [{"claim_text": "Propensity-score matching estimates the ATE.",
                             "normalized_label": "propensity score matching", "status": "found",
                             "method_type": "estimator", "exact_quote": QUOTES["matching"]}],
        "assumptions": [{"claim_text": "Selection on observables.",
                         "normalized_label": "unconfoundedness", "status": "found",
                         "assumption_type": "identification", "exact_quote": QUOTES["unconfounded"]}],
    }


@dataclass
class Paper:
    paper_id: str
    markdown_id: str
    markdown_hash: str
    text: str
    note_id: str | None = None
    claims: dict[str, str] = field(default_factory=dict)  # field_key -> claim_id

    def at(self, key: str) -> tuple[int, int]:
        """The character range of quote ``key`` in this paper's markdown."""
        start = self.text.index(QUOTES[key])
        return start, start + len(QUOTES[key])


class World:
    def __init__(self, lib) -> None:
        self.lib = lib
        self.n = 0

    def paper(self, paper_id: str, text: str, title: str, year: int) -> Paper:
        self.n += 1
        n = self.n
        sf_hash = f"{n:064x}"
        sf_id = f"sf_{sf_hash}"
        raw = text.encode("utf-8")
        mh = hashlib.sha256(raw).hexdigest()
        rel = f"markdown/{mh}.md"
        (self.lib.root / "markdown").mkdir(parents=True, exist_ok=True)
        (self.lib.root / rel).write_bytes(raw)
        with self.lib.write() as conn:
            conn.execute("INSERT INTO papers (paper_id, title, year, created_at) VALUES (?, ?, ?, ?)",
                         (paper_id, title, year, NOW))
            conn.execute("INSERT INTO source_files (source_file_id, file_hash, storage_uri, created_at) "
                         "VALUES (?, ?, ?, ?)", (sf_id, sf_hash, f"pdfs/{sf_hash}.pdf", NOW))
            conn.execute(
                "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, "
                "converter_name, converter_package, converter_version, python_version, config_json, "
                "conversion_fingerprint, conversion_epistemic_type, run_status, markdown_hash, created_at) "
                "VALUES (?, ?, ?, 'marker', 'marker-pdf', '9.9.9', '3.13', '{}', 'fp', "
                "'local_deterministic_conversion', 'success', ?, ?)", (f"conv_{n}", sf_id, sf_hash, mh, NOW))
            conn.execute(
                "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, markdown_hash, "
                "storage_uri, conversion_status, byte_size, created_at) VALUES (?, ?, ?, ?, ?, 'success', ?, ?)",
                (f"md_{mh}", f"conv_{n}", sf_id, mh, rel, len(raw), NOW))
            conn.execute("INSERT INTO paper_files (paper_id, source_file_id, markdown_id, attached_at) "
                         "VALUES (?, ?, ?, ?)", (paper_id, sf_id, f"md_{mh}", NOW))
        return Paper(paper_id, f"md_{mh}", mh, text)

    def note(self, paper: Paper, raw: dict) -> Paper:
        note, errors = fmt.validate_note(raw)
        assert errors == []
        norm = fmt.normalize_parts([note])
        with self.lib.write() as conn:
            saved = save.save_note(
                conn, paper_id=paper.paper_id, markdown_id=paper.markdown_id, markdown_hash=paper.markdown_hash,
                drafts=norm.drafts, note_text=norm.note_text, archetype=norm.archetype,
                raw_note_json=json.dumps(raw), provenance=save.NoteProvenance(client="claude", model="m-1"))
        paper.note_id = saved.note_id
        paper.claims = {r[0]: r[1] for r in self.lib.conn.execute(
            "SELECT field_key, claim_id FROM extracted_claims WHERE structured_note_id = ?", (saved.note_id,))}
        return paper

    def span(self, paper: Paper, key: str) -> str:
        """The span id of quote ``key`` (saved by the note, else created)."""
        start, end = paper.at(key)
        row = self.lib.conn.execute("SELECT span_id FROM evidence_spans WHERE markdown_id = ? AND start_char = ? "
                                    "AND end_char = ?", (paper.markdown_id, start, end)).fetchone()
        if row:
            return row[0]
        from seedgraph.notes.spans import ensure_span
        with self.lib.write() as conn:
            return ensure_span(conn, markdown_id=paper.markdown_id, exact_quote=QUOTES[key])

    def edit(self, paper: Paper, claim_id: str, expected: int, **set_):
        return overlay.save_edit(self.lib, paper.paper_id, {"op": "edit", "target": {"claim_id": claim_id},
                                                            "set": set_}, expected_version=expected)

    def graph(self, slug: str, *members: Paper) -> None:
        g = graphs.create_graph(self.lib, slug)
        try:
            for p in members:
                graphs.set_membership(g, p.paper_id, "member")
        finally:
            g.close()


def standard(lib) -> tuple[World, Paper, Paper]:
    """Two noted papers: a DiD paper and an ATE matching paper."""
    w = World(lib)
    did = w.note(w.paper("lp_did0000000000000001", DID_TEXT, "Invented county minimum wages", 2019), did_note())
    ate = w.note(w.paper("lp_ate0000000000000001", ATE_TEXT, "Invented matching for training", 2015), ate_note())
    return w, did, ate


class Agent:
    """One agent's calls: a connection, a client and an optional session name, on a clock."""

    def __init__(self, lib, conn_id: str = "mcp-aaaa0001", name: str | None = None, client: str = "claude",
                 start: datetime = T0) -> None:
        self.lib, self.conn_id, self.name, self.client, self.clock = lib, conn_id, name, client, start

    @property
    def call(self) -> CallInfo:
        return CallInfo(self.conn_id, self.client, self.name)

    def tick(self, minutes: float = 1) -> datetime:
        self.clock += timedelta(minutes=minutes)
        return self.clock

    def _log(self, tool, kind, args, items):
        log.log_event(self.lib, self.call, tool, kind, args, items, now=self.tick(), tz=UTC)

    def search(self, query="parallel trends", *, hits=(), graph=None, kind="spans"):
        args = {"query": query, "kind": kind}
        if graph:
            args["graph"] = graph
        self._log("search", "search", args, list(hits))

    def paper_show(self, paper: Paper, claim_ids=None, *, anchors=None):
        anchors = anchors or {}
        ids = claim_ids if claim_ids is not None else list(paper.claims.values())
        self._log("paper_show", "open_paper", {"paper_id": paper.paper_id},
                  [{"claim_id": c, "span_ids": list(anchors.get(c, ()))} for c in ids])

    def paper_read(self, paper: Paper, start: int, end: int, *, section_id=None):
        self._log("paper_read", "open_text", {"paper_id": paper.paper_id, "start": start, "end": end},
                  [{"paper_id": paper.paper_id, "markdown_id": paper.markdown_id, "start": start, "end": end,
                    "section_id": section_id}])

    def concept_show(self, handle: str, claims: dict[str, list[str]], *, graph=None):
        args = {"handle": handle}
        if graph:
            args["graph"] = graph
        self._log("concept_show", "open_concept", args,
                  [{"claim_id": c, "span_ids": s} for c, s in claims.items()])

    def answer(self, evidence, *, question="Is parallel trends plausible here?", answer="The paper says so.",
               judgement=None):
        from seedgraph.records import answer as answer_mod
        return answer_mod.record_answer(self.lib, self.call, question, answer, evidence, judgement,
                                        now=self.tick(), tz=UTC)


def span_hit(span_id: str, paper: Paper, preview: str = "an invented preview") -> dict:
    return {"kind": "span", "id": span_id, "paper_id": paper.paper_id, "preview": preview}


def claim_hit(claim_id: str, paper: Paper) -> dict:
    return {"kind": "claim", "id": claim_id, "paper_id": paper.paper_id, "preview": "an invented claim"}
