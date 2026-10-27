"""Invented libraries for the concept tests: papers with notes and claims written straight
into a temporary library (stage 1 schema), so tests can shape claims freely."""

from __future__ import annotations

import hashlib
import itertools

from seedgraph.library import store

NOW = "2026-01-01T00:00:00+00:00"
_seq = itertools.count(1)
_FIELDS = {"identification_assumption": "assumptions", "general_assumption": "assumptions",
           "regularity_condition": "assumptions", "model": "method_or_model", "method": "method_or_model",
           "result": "main_results", "data": "data_sources", "limitation": "limitations"}


class World:
    def __init__(self, lib: store.Library) -> None:
        self.lib = lib
        self.md: dict[str, tuple[str, str, str, str]] = {}  # paper -> (md_id, md_hash, sf_id, sf_hash)

    def paper(self, pid: str) -> str:
        n = next(_seq)
        sf_hash = f"{n:064x}"
        md_hash = hashlib.sha256(f"md-{pid}-{n}".encode()).hexdigest()
        sf_id, md_id = f"sf_{sf_hash}", f"md_{md_hash}"
        with self.lib.write() as conn:
            conn.execute("INSERT INTO papers (paper_id, title, created_at) VALUES (?, ?, ?)",
                         (pid, f"An invented paper {pid}", NOW))
            conn.execute("INSERT INTO source_files (source_file_id, file_hash, storage_uri, created_at) "
                         "VALUES (?, ?, ?, ?)", (sf_id, sf_hash, f"pdfs/{sf_hash}.pdf", NOW))
            conn.execute(
                "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, converter_name, "
                "converter_package, converter_version, python_version, config_json, conversion_fingerprint, "
                "conversion_epistemic_type, run_status, markdown_hash, created_at) VALUES (?, ?, ?, 'marker', "
                "'marker-pdf', '9.9.9', '3.13', '{}', 'fp', 'local_deterministic_conversion', 'success', ?, ?)",
                (f"conv_{n}", sf_id, sf_hash, md_hash, NOW))
            conn.execute(
                "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, markdown_hash, "
                "storage_uri, conversion_status, byte_size, created_at) VALUES (?, ?, ?, ?, ?, 'success', 10, ?)",
                (md_id, f"conv_{n}", sf_id, md_hash, f"markdown/{md_hash}.md", NOW))
            conn.execute("INSERT INTO paper_files (paper_id, source_file_id, markdown_id, attached_at) "
                         "VALUES (?, ?, ?, ?)", (pid, sf_id, md_id, NOW))
        self.md[pid] = (md_id, md_hash, sf_id, sf_hash)
        return pid

    def note(self, pid: str, claims: list[dict]) -> dict[str, str]:
        """A new note version for ``pid``. Each claim: ``type``, ``label``, ``text`` and
        optionally ``status`` ('found'), ``quote`` (default True: anchored) and ``start``
        (the quote's offset). Returns label -> claim_id (last wins)."""
        if pid not in self.md:
            self.paper(pid)
        md_id, md_hash, sf_id, sf_hash = self.md[pid]
        n = next(_seq)
        run, note = f"run_{n}", f"note_{n}"
        out = {}
        with self.lib.write() as conn:
            vno = conn.execute("SELECT COALESCE(MAX(version_no), 0) + 1 FROM structured_notes WHERE paper_id = ?",
                               (pid,)).fetchone()[0]
            conn.execute(
                "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, schema_id, "
                "schema_version, prompt_version, run_status, created_at) VALUES (?, ?, ?, ?, "
                "'default_research_note_v1', '2.0.0', 'p1', 'success', ?)", (run, pid, md_id, md_hash, NOW))
            conn.execute(
                "INSERT INTO structured_notes (note_id, extraction_run_id, paper_id, markdown_id, markdown_hash, "
                "schema_id, schema_version, prompt_version, raw_note_json, note_text, created_at, version_no) "
                "VALUES (?, ?, ?, ?, ?, 'default_research_note_v1', '2.0.0', 'p1', '{}', 'invented', ?, ?)",
                (note, run, pid, md_id, md_hash, NOW, vno))
            counts: dict[str, int] = {}
            for i, c in enumerate(claims):
                field = _FIELDS.get(c["type"], c["type"])
                k = counts.get(field, 0)
                counts[field] = k + 1
                cid = f"cl_{n}_{i}"
                conn.execute(
                    "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, paper_id, "
                    "claim_type, field_key, normalized_label, claim_text, status, epistemic_type, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'paper_stated', ?)",
                    (cid, note, run, pid, c["type"], f"{field}[{k}]", c["label"], c["text"],
                     c.get("status", "found"), NOW))
                if c.get("quote", True):
                    start = c.get("start", 100 * i)
                    span = f"sp_{n}_{i}"
                    conn.execute(
                        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
                        "source_file_hash, start_char, end_char, exact_quote, quote_hash, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (span, md_id, md_hash, sf_id, sf_hash, start, start + 50, c["text"],
                         hashlib.sha256(c["text"].encode()).hexdigest(), NOW))
                    conn.execute("INSERT INTO claim_spans (claim_id, span_id, rank, created_at) VALUES (?, ?, 0, ?)",
                                 (cid, span, NOW))
                out[c["label"]] = cid
        return out

    def decide(self, kind: str, a: str, b: str, *, by: str = "researcher", state: str = "applied",
               family: str | None = None, at: str = NOW, name: str | None = None,
               sense_a: str | None = None, sense_b: str | None = None, origin: str = "library") -> str:
        from seedgraph.concepts.model import decision_id
        if (a, sense_a or "") > (b, sense_b or ""):
            a, b, sense_a, sense_b = b, a, sense_b, sense_a
        did = decision_id(kind, a, b, family=family, sense_a=sense_a, sense_b=sense_b)
        with self.lib.write() as conn:
            conn.execute(
                "INSERT INTO concept_decisions (decision_id, kind, family, label_a, sense_a, label_b, sense_b, "
                "name, decided_by, state, origin, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (did, kind, family, a, sense_a, b, sense_b, name, by, state, origin, at))
        return did

    def split(self, family: str, label: str, senses: dict[str, str], *, by: str = "researcher") -> None:
        """Split a label into senses ``{sense_id: gloss}``."""
        with self.lib.write() as conn:
            for sid, gl in senses.items():
                conn.execute("INSERT INTO concept_senses (family, label_key, sense_id, gloss, decided_by, state, "
                             "decided_at) VALUES (?, ?, ?, ?, ?, 'confirmed', ?)", (family, label, sid, gl, by, NOW))

    def assign(self, family: str, label: str, claim_id: str, sense: str, *, by: str = "researcher") -> None:
        """Assign a current claim to a sense (keyed by its fingerprint, as C5/C7 will)."""
        from seedgraph.concepts.model import claim_fingerprint
        row = self.lib.conn.execute(
            "SELECT cc.*, e.quote_hash, e.markdown_hash, e.start_char, e.end_char FROM current_claims cc "
            "LEFT JOIN evidence_spans e ON e.span_id = cc.span_id WHERE cc.claim_id = ?", (claim_id,)).fetchone()
        span = row if row["quote_hash"] is not None else None
        with self.lib.write() as conn:
            conn.execute("INSERT INTO sense_assignments (family, label_key, paper_id, claim_fingerprint, sense_id, "
                         "decided_by, decided_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (family, label, row["paper_id"], claim_fingerprint(row, span), sense, by, NOW))

    def concept_of(self, family: str, label: str, sense: str = "s0") -> str | None:
        row = self.lib.conn.execute(
            "SELECT concept_id FROM concept_members WHERE family = ? AND label_key = ? AND sense_id = ?",
            (family, label, sense)).fetchone()
        return row[0] if row else None

    def concept(self, cid: str):
        return self.lib.conn.execute("SELECT * FROM concepts WHERE concept_id = ?", (cid,)).fetchone()

    def state(self, did: str) -> tuple[str, str | None]:
        return tuple(self.lib.conn.execute(
            "SELECT state, veto_reason FROM concept_decisions WHERE decision_id = ?", (did,)).fetchone())


def a(label: str, text: str | None = None, **kw) -> dict:
    """An identification-assumption claim."""
    return {"type": "identification_assumption", "label": label, "text": text or f"Invented: {label} holds.", **kw}


def m(label: str, text: str | None = None, **kw) -> dict:
    """A method claim."""
    return {"type": "method", "label": label, "text": text or f"Invented method: {label} is used.", **kw}
