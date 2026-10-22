"""Synthetic v2 home for golden-set tests (no paper text; schema via the repo's own migrations)."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from seedgraph.db.bootstrap import ensure_cache_db, ensure_project_db

PAGE = "{%d}------------------------------------------------"


def synthetic_markdown(n_pages: int, quotes: list[str]) -> str:
    pages = []
    for i in range(n_pages):
        body = f"Synthetic page {i}. " + " ".join(q for j, q in enumerate(quotes) if j % n_pages == i)
        if i == 0:
            body += "\n\nInline $x_{1}$ and display:\n\n$$\n\\frac{a}{b}\n$$\n\n| h1 | h2 |\n|---|---|\n| 1 | 2 |"
        if i == n_pages - 1:
            body += "\n\n## References\n\n- Synthetic, A. 2000."
        pages.append(PAGE % i + "\n\n" + body)
    return "\n\n".join(pages)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeV2:
    """Builds a minimal v2 home: cache.db + markdown files + project.db notes/spans."""

    def __init__(self, home: Path):
        self.home = Path(home)
        ensure_cache_db(self.home)
        self._n = 0

    def _cache(self) -> sqlite3.Connection:
        return sqlite3.connect(self.home / "cache" / "cache.db")

    def add_paper(
        self,
        *,
        markdown: str,
        pdf_hash: str | None = None,
        slug: str = "proj",
        quotes: list[str] = (),
        created_at: str = "2026-07-01T10:00:00+00:00",
        completed_at: str = "2026-07-01T10:01:00+00:00",
        older_note_quotes: list[str] | None = None,
    ) -> dict:
        self._n += 1
        n = self._n
        md_hash = _sha(markdown)
        pdf_hash = pdf_hash or _sha(f"pdf-{n}")
        sf = "sf_" + pdf_hash
        conv = f"conv_{n:032d}"
        md_id = "md_" + md_hash
        rel = f"markdown/{md_hash}.md"
        (self.home / "cache" / "markdown").mkdir(parents=True, exist_ok=True)
        (self.home / "cache" / rel).write_text(markdown, encoding="utf-8", newline="")
        c = self._cache()
        with c:
            c.execute(
                "INSERT INTO source_files (source_file_id, file_hash, file_type, storage_uri, created_at) "
                "VALUES (?,?,?,?,?)",
                (sf, pdf_hash, "pdf", f"pdfs/{pdf_hash}.pdf", created_at),
            )
            c.execute(
                "INSERT INTO conversion_runs (conversion_run_id, source_file_id, source_file_hash, "
                "converter_name, converter_package, converter_version, python_version, config_json, "
                "conversion_fingerprint, conversion_epistemic_type, run_status, markdown_hash, "
                "created_at, completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (conv, sf, pdf_hash, "marker", "marker-pdf", "1.10.2", "3.13", "{}", "fp",
                 "local_deterministic_conversion", "success", md_hash, created_at, completed_at),
            )
            c.execute(
                "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
                "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (md_id, conv, sf, md_hash, rel, "success", len(markdown), completed_at),
            )
        c.close()
        work_id = f"work_{n:04d}"
        ensure_project_db(slug, self.home)
        p = sqlite3.connect(self.home / "projects" / slug / "project.db")
        with p:
            p.execute("INSERT INTO works (work_id, canonical_title, created_at) VALUES (?,?,?)",
                      (work_id, "synthetic", created_at))
            generations = []
            if older_note_quotes is not None:
                generations.append(("2026-07-02T00:00:00+00:00", older_note_quotes))
            generations.append(("2026-10-20T00:00:00+00:00", list(quotes)))
            note_id = None
            for g, (ts, qs) in enumerate(generations):
                run = f"run_{n}_{g}"
                note_id = f"note_{n}_{g}"
                p.execute(
                    "INSERT INTO extraction_runs (extraction_run_id, work_id, markdown_id, markdown_hash, "
                    "schema_id, schema_version, prompt_version, access_class, run_status, created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (run, work_id, md_id, md_hash, "default_research_note_v1", "1", "1",
                     "user_supplied_private", "success", ts),
                )
                p.execute(
                    "INSERT INTO structured_notes (note_id, extraction_run_id, work_id, markdown_id, "
                    "markdown_hash, schema_id, schema_version, prompt_version, raw_note_json, note_text, "
                    "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (note_id, run, work_id, md_id, md_hash, "default_research_note_v1", "1", "1", "{}", "",
                     ts),
                )
                for k, q in enumerate(qs):
                    claim = f"claim_{n}_{g}_{k}"
                    span = f"span_{n}_{g}_{k}"
                    p.execute(
                        "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, "
                        "work_id, claim_type, field_key, status, epistemic_type, created_at) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (claim, note_id, run, work_id, "result", f"f[{k}]", "found", "llm_extracted", ts),
                    )
                    p.execute(
                        "INSERT INTO evidence_spans (span_id, markdown_id, markdown_hash, source_file_id, "
                        "source_file_hash, work_id, start_char, end_char, exact_quote, quote_hash, "
                        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (span, md_id, md_hash, sf, pdf_hash, work_id, 0, len(q), q, _sha(q), ts),
                    )
                    p.execute(
                        "INSERT INTO claim_spans (claim_id, span_id, created_at) VALUES (?,?,?)",
                        (claim, span, ts),
                    )
        p.close()
        return {"markdown_id": md_id, "pdf_hash": pdf_hash, "work_id": work_id, "note_id": note_id,
                "slug": slug, "md_prefix": md_hash[:12], "pdf_prefix": pdf_hash[:12]}
