"""The synthetic v2 home builder (design §8.4, §11 S3).

Each test queries the built v2 databases and files directly: the fixture is the
seam the migration tests (S6/S7) stand on, so what it plants must really be there.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from fixtures import build_v2_home
from seedgraph.migrate import v2_schema

SLUGS = ["second_test", "land_value", "test_proj", "oct2026test", "dbg_probe", "dbg_probe2"]
PAGE_RE = re.compile(r"^\{(\d+)\}-{10,}$", re.M)


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    return build_v2_home(tmp_path_factory.mktemp("v2") / "home")


def ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)


def proj(fx, slug) -> sqlite3.Connection:
    return ro(fx.home / "projects" / slug / "project.db")


def cache(fx) -> sqlite3.Connection:
    return ro(fx.home / "cache" / "cache.db")


def content_words(text: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9]+", " ", text.lower()).split() if len(t) > 3}


def one(conn, sql, *args):
    row = conn.execute(sql, args).fetchone()
    assert row is not None, sql
    return row


def work_row(fx, key):
    ref = fx.works[key]
    conn = proj(fx, ref.slug)
    row = conn.execute(
        "SELECT w.*, d.inclusion_status FROM works w LEFT JOIN project_documents d USING(work_id) "
        "WHERE work_id = ?", (ref.work_id,)
    ).fetchone()
    names = [c[0] for c in conn.execute(
        "SELECT w.*, d.inclusion_status FROM works w LEFT JOIN project_documents d USING(work_id) LIMIT 0"
    ).description]
    assert row is not None, key
    return dict(zip(names, row))


def wsf(fx, key):
    ref = fx.works[key]
    return proj(fx, ref.slug).execute(
        "SELECT source_file_id, markdown_id FROM work_source_files WHERE work_id = ?", (ref.work_id,)
    ).fetchone()


def markdown_text(fx, markdown_id):
    uri = one(cache(fx), "SELECT storage_uri FROM markdown_documents WHERE markdown_id = ?", markdown_id)[0]
    return (fx.home / "cache" / uri).read_bytes().decode("utf-8")


def pdf_text(fx, source_file_id):
    uri = one(cache(fx), "SELECT storage_uri FROM source_files WHERE source_file_id = ?", source_file_id)[0]
    doc = pdfium.PdfDocument(str(fx.home / "cache" / uri))
    try:
        texts = []
        for i in range(len(doc)):
            tp = doc[i].get_textpage()
            texts.append(tp.get_text_range())
            tp.close()
        return texts
    finally:
        doc.close()


# --- layout and schema -------------------------------------------------------

def test_home_layout(fx):
    assert (fx.home / "cache" / "cache.db").is_file()
    for slug in SLUGS:
        assert (fx.home / "projects" / slug / "project.db").is_file()
        assert (fx.home / "projects" / slug / "project.yaml").is_file()
    assert (fx.home / "projects" / "second_test" / "lenses" / "plot_layout_v1.yaml").is_file()


def test_databases_use_the_frozen_schema_and_are_valid(fx):
    dbs = [("cache", fx.home / "cache" / "cache.db")] + [
        ("project", fx.home / "projects" / s / "project.db") for s in SLUGS
    ]
    for scope, path in dbs:
        conn = ro(path)
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == \
            v2_schema.latest_version(scope)
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert not list(fx.home.rglob("*.db-wal"))


def test_blobs_are_content_addressed_lf_only(fx):
    c = cache(fx)
    for sf, file_hash, uri, size in c.execute(
        "SELECT source_file_id, file_hash, storage_uri, byte_size FROM source_files"
    ):
        data = (fx.home / "cache" / uri).read_bytes()
        assert sf == "sf_" + file_hash == "sf_" + hashlib.sha256(data).hexdigest()
        assert uri == f"pdfs/{file_hash}.pdf" and size == len(data)
        assert len(pdfium.PdfDocument(data)) == 3
    for md, md_hash, uri, size in c.execute(
        "SELECT markdown_id, markdown_hash, storage_uri, byte_size FROM markdown_documents"
    ):
        data = (fx.home / "cache" / uri).read_bytes()
        assert md == "md_" + md_hash == "md_" + hashlib.sha256(data).hexdigest()
        assert uri == f"markdown/{md_hash}.md" and size == len(data)
        assert b"\r" not in data


def test_marker_markdown_has_page_markers(fx):
    md = wsf(fx, "overlap@second_test")[1]
    text = markdown_text(fx, md)
    assert [int(n) for n in PAGE_RE.findall(text)] == [0, 1, 2]


def test_spans_are_anchored(fx):
    for slug in ["second_test", "land_value", "test_proj", "oct2026test"]:
        conn = proj(fx, slug)
        rows = conn.execute("SELECT markdown_id, start_char, end_char, exact_quote, quote_hash FROM evidence_spans").fetchall()
        assert rows
        for md, start, end, quote, qh in rows:
            assert markdown_text(fx, md)[start:end] == quote
            assert qh == hashlib.sha256(quote.encode("utf-8")).hexdigest()


def test_manifest_lists_every_planted_class(fx):
    required = {
        "overlap_diverging_notes", "double_import", "shared_pdf_conflicting_dois", "shared_pdf_one_side_no_id",
        "foreign_markdown", "foreign_markdown_unconverted", "noid_title_year_match", "noid_ambiguous",
        "noid_unmatched", "mojibake_markdown", "mojibake_title", "wrong_pdf", "malformed_title_cached",
        "malformed_title_uncached", "cited_only", "metadata_only_unreferenced", "duplicate_verdict_different",
        "duplicate_verdict_same", "citation_review_decided", "citation_review_open", "concept_llm_merge",
        "concept_user_must_link", "concept_user_cannot_link", "concept_contradiction",
        "concept_agent_and_researcher_same_pair", "lens_with_outputs", "conversion_failed",
        "conversion_pending", "fake_converter_markdown", "unheld_pdf", "external_markdown_no_markers",
        "pdf_no_text_layer", "saved_answer", "run_scoped_answer", "excluded_in_graph", "dbg_project",
        "open_review_items", "dropped_tables",
    }
    assert required <= set(fx.kinds)
    for item in fx.items:
        assert item.expect, item.kind


# --- planted classes ---------------------------------------------------------

def test_overlap_two_projects_diverging_notes(fx):
    item = fx.one("overlap_diverging_notes")
    st, tp = fx.works["overlap@second_test"], fx.works["overlap@test_proj"]
    assert work_row(fx, "overlap@second_test")["doi"] == work_row(fx, "overlap@test_proj")["doi"]
    assert st.work_id != tp.work_id
    assert wsf(fx, "overlap@second_test") == wsf(fx, "overlap@test_proj")
    notes = {}
    for ref in (st, tp):
        notes[ref.slug] = one(proj(fx, ref.slug),
                              "SELECT note_id, note_text, created_at FROM structured_notes WHERE work_id = ?",
                              ref.work_id)
    assert notes["second_test"][1] != notes["test_proj"][1]
    assert notes["test_proj"][2] > notes["second_test"][2]  # the lower-precedence project's note is newer
    assert set(item.ids["note_ids"]) == {notes["second_test"][0], notes["test_proj"][0]}
    # Manual spans on the same quote in both projects, with different random ids.
    keys = {}
    for ref in (st, tp):
        keys[ref.slug] = sorted(
            one(proj(fx, ref.slug), "SELECT span_id, start_char, end_char FROM evidence_spans "
                "WHERE span_id = ? AND work_id = ? AND span_kind = 'manual'", sid, ref.work_id)
            for sid in item.ids["span_ids"][ref.slug])
        keys[ref.slug].sort(key=lambda k: k[1])
    assert [k[1:] for k in keys["second_test"]] == [k[1:] for k in keys["test_proj"]]
    assert {k[0] for k in keys["second_test"]}.isdisjoint({k[0] for k in keys["test_proj"]})
    # Deterministic section ids repeat across projects for the same markdown.
    secs = [set(r[0] for r in proj(fx, s).execute(
        "SELECT section_id FROM document_sections WHERE markdown_id = ?", (item.ids["markdown_id"],)))
        for s in ("second_test", "test_proj")]
    assert secs[0] and secs[0] == secs[1]


def test_double_import_within_one_project(fx):
    item = fx.one("double_import")
    a, b = (work_row(fx, k) for k in item.ids["work_keys"])
    assert fx.works[item.ids["work_keys"][0]].slug == fx.works[item.ids["work_keys"][1]].slug
    assert a["doi"] and not b["doi"] and b["openalex_id"] and not a["openalex_id"]
    assert wsf(fx, item.ids["work_keys"][0])[0] == wsf(fx, item.ids["work_keys"][1])[0]


def test_shared_pdf_conflicting_dois_is_a_true_mixup(fx):
    item = fx.one("shared_pdf_conflicting_dois")
    ka, kb = item.ids["work_keys"]
    a, b = work_row(fx, ka), work_row(fx, kb)
    assert a["doi"] and b["doi"] and a["doi"] != b["doi"]
    assert wsf(fx, ka)[0] == wsf(fx, kb)[0] == item.ids["source_file_id"]
    text = " ".join(pdf_text(fx, item.ids["source_file_id"]))
    assert a["canonical_title"] in text and b["doi"] not in text
    assert not content_words(b["canonical_title"]) & content_words(text)
    assert item.expect["miss_side"] == kb


def test_shared_pdf_one_side_without_id(fx):
    item = fx.one("shared_pdf_one_side_no_id")
    ka, kb = item.ids["work_keys"]
    b = work_row(fx, kb)
    assert work_row(fx, ka)["doi"] and not (b["doi"] or b["openalex_id"] or b["arxiv_id"])
    assert wsf(fx, ka)[0] == wsf(fx, kb)[0]
    assert proj(fx, fx.works[kb].slug).execute(
        "SELECT COUNT(*) FROM identifiers WHERE work_id = ?", (fx.works[kb].work_id,)).fetchone()[0] == 0


def test_work_pointing_at_another_pdfs_markdown(fx):
    for kind, queued in [("foreign_markdown", False), ("foreign_markdown_unconverted", True)]:
        item = fx.one(kind)
        sf, md = wsf(fx, item.ids["work_key"])
        md_source = one(cache(fx), "SELECT source_file_id FROM markdown_documents WHERE markdown_id = ?", md)[0]
        assert md_source != sf
        own = cache(fx).execute(
            "SELECT markdown_id FROM markdown_documents WHERE source_file_id = ?", (sf,)).fetchall()
        assert (own == []) is queued
        ref = fx.works[item.ids["work_key"]]
        assert one(proj(fx, ref.slug), "SELECT markdown_id FROM structured_notes WHERE note_id = ?",
                   item.ids["note_id"])[0] == md
        assert item.expect["version_label"] == "other_markdown"


def test_noid_papers(fx):
    match = fx.one("noid_title_year_match")
    w = work_row(fx, match.ids["work_key"])
    target = work_row(fx, match.ids["matches"])
    assert not (w["doi"] or w["openalex_id"] or w["arxiv_id"]) and target["doi"]
    assert w["canonical_title"] != target["canonical_title"] and w["year"] == target["year"]
    norm = lambda t: " ".join(re.sub(r"[^a-z0-9]+", " ", t.lower()).split())  # noqa: E731
    assert norm(w["canonical_title"]) == norm(target["canonical_title"])

    amb = fx.one("noid_ambiguous")
    w = work_row(fx, amb.ids["work_key"])
    cands = [work_row(fx, k) for k in amb.ids["candidates"]]
    assert len(cands) == 2 and len({c["doi"] for c in cands}) == 2
    assert all(c["canonical_title"] == w["canonical_title"] and c["year"] == w["year"] for c in cands)

    none = fx.one("noid_unmatched")
    w = work_row(fx, none.ids["work_key"])
    near = work_row(fx, none.ids["near_miss"])
    assert wsf(fx, none.ids["work_key"]) is not None
    assert norm(near["canonical_title"]) != norm(w["canonical_title"])


def test_mojibake(fx):
    item = fx.one("mojibake_markdown")
    text = markdown_text(fx, wsf(fx, item.ids["work_key"])[1])
    assert "SpeciÖcation" in text and "E¢ cient" in text and "Di¤erence" in text
    assert "Specification" in work_row(fx, item.ids["work_key"])["canonical_title"]
    item = fx.one("mojibake_title")
    assert "Ö" in work_row(fx, item.ids["work_key"])["canonical_title"]
    assert "Identification" in markdown_text(fx, wsf(fx, item.ids["work_key"])[1])


def test_wrong_pdf(fx):
    item = fx.one("wrong_pdf")
    w = work_row(fx, item.ids["work_key"])
    sf, md = wsf(fx, item.ids["work_key"])
    words = content_words(w["canonical_title"])
    assert words and not words & content_words(" ".join(pdf_text(fx, sf)))
    assert not words & content_words(markdown_text(fx, md))
    assert item.expect["attention"] == "pdf_title_miss" and item.expect["hides_notes"] == 1


def test_malformed_titles_and_provider_cache(fx):
    cached = fx.one("malformed_title_cached")
    w = work_row(fx, cached.ids["work_key"])
    resp = one(cache(fx), "SELECT response FROM provider_cache WHERE request_key = ?",
               f"by_doi:doi={w['doi']}")[0]
    assert json.loads(resp)["title"] == cached.expect["title"]
    assert w["canonical_title"].endswith(".pdf")
    unc = fx.one("malformed_title_uncached")
    w = work_row(fx, unc.ids["work_key"])
    assert cache(fx).execute("SELECT COUNT(*) FROM provider_cache WHERE request_key LIKE ?",
                             (f"%{w['doi']}%",)).fetchone()[0] == 0


def test_cited_only_and_unreferenced(fx):
    item = fx.one("cited_only")
    for key in item.ids["work_keys"]:
        assert work_row(fx, key)["inclusion_status"] == "metadata_only"
        assert wsf(fx, key) is None
        ref = fx.works[key]
        assert proj(fx, ref.slug).execute(
            "SELECT COUNT(*) FROM citation_edges WHERE target_work_id = ?", (ref.work_id,)).fetchone()[0] >= 1
    unref = fx.one("metadata_only_unreferenced")
    ref = fx.works[unref.ids["work_key"]]
    conn = proj(fx, ref.slug)
    assert conn.execute("SELECT COUNT(*) FROM citation_edges WHERE ? IN (source_work_id, target_work_id)",
                        (ref.work_id,)).fetchone()[0] == 0


def test_duplicate_verdicts(fx):
    diff = fx.one("duplicate_verdict_different")
    conn = proj(fx, "test_proj")
    status, action, payload = one(conn, "SELECT status, action, payload FROM review_queue WHERE item_id = ?",
                                  diff.ids["item_id"])
    assert (status, action) == ("resolved", "dismissed")
    payload = json.loads(payload)
    assert payload["conflicting_work_ids"] == [fx.works[diff.ids["work_keys"][0]].work_id]
    b = work_row(fx, diff.ids["work_keys"][1])
    a = work_row(fx, diff.ids["work_keys"][0])
    assert not b["doi"] and a["canonical_title"] == b["canonical_title"] and a["year"] == b["year"]
    same = fx.one("duplicate_verdict_same")
    status, action = one(conn, "SELECT status, action FROM review_queue WHERE item_id = ?", same.ids["item_id"])
    assert (status, action) == ("resolved", "merge")


def test_citation_reviews(fx):
    decided = fx.one("citation_review_decided")
    for slug, item_id, action in decided.ids["items"]:
        status, act, ttype = one(proj(fx, slug),
                                 "SELECT status, action, target_type FROM review_queue WHERE item_id = ?", item_id)
        assert (status, act, ttype) == ("resolved", action, "reference_entry")
    assert decided.expect["unmapped"]
    opened = fx.one("citation_review_open")
    slug, item_id = opened.ids["item"]
    assert one(proj(fx, slug), "SELECT status FROM review_queue WHERE item_id = ?", item_id)[0] == "open"


def test_concept_constraints(fx):
    rows = {}
    for slug in ("second_test", "test_proj"):
        rows[slug] = set(proj(fx, slug).execute("SELECT kind, label_a, label_b, source FROM concept_constraints"))
    assert ("must_link", "difference in differences", "diff-in-diff", "llm_merge") in rows["second_test"]
    assert any(r[0] == "must_link" and r[3] == "user" for r in rows["second_test"])
    assert any(r[0] == "cannot_link" and r[3] == "user" for r in rows["test_proj"])
    contra = fx.one("concept_contradiction")
    assert contra.ids["pair"] == ["event study", "staggered adoption"]


def test_lens_with_outputs(fx):
    item = fx.one("lens_with_outputs")
    conn = proj(fx, "second_test")
    lens_id, yaml_path, dhash = one(conn, "SELECT lens_id, yaml_path, definition_hash FROM lenses")
    assert (fx.home / "projects" / "second_test" / yaml_path).is_file()
    assert dhash == item.ids["definition_hash"]
    statuses = {r[0] for r in conn.execute("SELECT status FROM lens_outputs")}
    assert {"found", "not_found"} <= statuses
    assert conn.execute("SELECT COUNT(*) FROM extraction_runs WHERE lens_id = ? AND lens_definition_hash = ?",
                        (lens_id, dhash)).fetchone()[0] >= 1


def test_lens_hash_matches_v2_definition_hash(fx):
    schema = pytest.importorskip("seedgraph.lenses.schema")
    path = fx.home / "projects" / "second_test" / "lenses" / "plot_layout_v1.yaml"
    assert schema.LensDefinition.from_yaml(path).definition_hash() == fx.one("lens_with_outputs").ids["definition_hash"]


def test_conversion_runs(fx):
    c = cache(fx)
    assert c.execute("SELECT COUNT(*) FROM conversion_runs WHERE run_status='failed'").fetchone()[0] >= 1
    pend = fx.one("conversion_pending")
    assert one(c, "SELECT run_status, completed_at FROM conversion_runs WHERE conversion_run_id = ?",
               pend.ids["conversion_run_id"]) == ("pending", None)
    failed = fx.one("conversion_failed")
    sf = failed.ids["source_file_id"]
    assert c.execute("SELECT COUNT(*) FROM markdown_documents WHERE source_file_id = ?", (sf,)).fetchone()[0] == 0
    fake = fx.one("fake_converter_markdown")
    assert one(c, "SELECT r.converter_version FROM markdown_documents m JOIN conversion_runs r "
                  "USING(conversion_run_id) WHERE m.markdown_id = ?", wsf(fx, fake.ids["work_key"])[1])[0] == "fake-0.0.0"


def test_unheld_pdf(fx):
    item = fx.one("unheld_pdf")
    sf = item.ids["source_file_id"]
    for slug in SLUGS:
        assert proj(fx, slug).execute("SELECT COUNT(*) FROM work_source_files WHERE source_file_id = ?",
                                      (sf,)).fetchone()[0] == 0
    assert one(cache(fx), "SELECT COUNT(*) FROM source_files WHERE source_file_id = ?", sf)[0] == 1


def test_external_markdown_and_scan(fx):
    ext = fx.one("external_markdown_no_markers")
    md = wsf(fx, ext.ids["work_key"])[1]
    assert not PAGE_RE.search(markdown_text(fx, md))
    assert one(cache(fx), "SELECT r.converter_name FROM markdown_documents m JOIN conversion_runs r "
                          "USING(conversion_run_id) WHERE m.markdown_id = ?", md)[0] == "external_import"
    scan = fx.one("pdf_no_text_layer")
    sf, md = wsf(fx, scan.ids["work_key"])
    assert all(t.strip() == "" for t in pdf_text(fx, sf))
    assert work_row(fx, scan.ids["work_key"])["canonical_title"] in markdown_text(fx, md)


def test_saved_answers(fx):
    items = fx.by_kind("saved_answer")
    assert {i.ids["slug"] for i in items} == {"second_test", "test_proj"}
    for item in items:
        base = fx.home / "projects" / item.ids["slug"] / "answers"
        env = json.loads((base / f"{item.ids['answer_id']}.json").read_bytes())
        trace = json.loads((base / f"{item.ids['answer_id']}.trace.json").read_bytes())
        assert env["answer_id"] == trace["answer_id"] == item.ids["answer_id"]
        conn = proj(fx, item.ids["slug"])
        for wid in env["cited_work_ids"]:
            assert conn.execute("SELECT 1 FROM works WHERE work_id = ?", (wid,)).fetchone()
        for sid in env["cited_span_ids"]:
            assert conn.execute("SELECT 1 FROM evidence_spans WHERE span_id = ?", (sid,)).fetchone()
    run = fx.one("run_scoped_answer")
    assert (fx.home / run.ids["path"]).is_file()


def test_answers_validate_against_v2_models(fx):
    types = pytest.importorskip("seedgraph.answer.types")
    trace = pytest.importorskip("seedgraph.answer.trace")
    for item in fx.by_kind("saved_answer"):
        base = fx.home / "projects" / item.ids["slug"] / "answers"
        types.AnswerEnvelope.model_validate_json((base / f"{item.ids['answer_id']}.json").read_bytes())
        trace.AnswerTrace.model_validate_json((base / f"{item.ids['answer_id']}.trace.json").read_bytes())


def test_graph_shapes_and_not_carried_items(fx):
    excl = fx.one("excluded_in_graph")
    assert work_row(fx, excl.ids["work_key"])["inclusion_status"] == "excluded"
    for slug in ("dbg_probe", "dbg_probe2"):
        conn = proj(fx, slug)
        assert conn.execute("SELECT COUNT(*) FROM work_source_files").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] >= 1
    item = fx.one("open_review_items")
    for slug, item_id in item.ids["items"]:
        assert one(proj(fx, slug), "SELECT status FROM review_queue WHERE item_id = ?", item_id)[0] == "open"
    conn = proj(fx, "second_test")
    for table in fx.one("dropped_tables").ids["tables"]:
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] >= 1, table


def test_pdf_titles_pass_where_expected(fx):
    """Pass-side papers print their title contiguously on page 1 of the PDF."""
    for key in ("overlap@second_test", "lattice@second_test", "victim@second_test"):
        title = work_row(fx, key)["canonical_title"]
        assert title in pdf_text(fx, wsf(fx, key)[0])[0]


def test_title_check_outcomes_with_the_shared_matcher(fx):
    """Integration with S2's matcher: each planted title-check case gives one result.

    Mirrors design §7.2: check the markdown's first pages, fall back to the PDF text
    when that is not a pass, best result wins."""
    tc = pytest.importorskip("seedgraph.library.titlecheck")
    rank = {"pass": 3, "weak": 2, "miss": 1, "unknown": 0}

    def best(key, title=None):
        w = work_row(fx, key)
        sf, md = wsf(fx, key)
        title = title or w["canonical_title"]
        pdf_path = fx.home / "cache" / "pdfs" / f"{sf[3:]}.pdf"
        pdf_pages = tc.first_pages_from_pdf(pdf_path) or []
        on_pdf = tc.check_pdf_text(title, w["doi"], pdf_pages).status
        if md is None:
            return on_pdf, on_pdf
        on_md = tc.check_pdf_text(title, w["doi"], tc.first_pages_from_markdown(markdown_text(fx, md))).status
        overall = on_md if on_md == "pass" else max(on_md, on_pdf, key=rank.get)
        return overall, on_pdf

    mix = fx.one("shared_pdf_conflicting_dois")
    for key, status in mix.expect["pdf_check"].items():
        assert best(key)[0] == status, key
    assert best(fx.one("wrong_pdf").ids["work_key"])[0] == "miss"
    assert best(fx.one("mojibake_markdown").ids["work_key"])[0] == "pass"
    assert best(fx.one("mojibake_title").ids["work_key"])[0] == "pass"
    assert best(fx.one("external_markdown_no_markers").ids["work_key"])[0] == "pass"
    scan = best(fx.one("pdf_no_text_layer").ids["work_key"])
    assert scan == ("pass", "unknown")
    cached = fx.one("malformed_title_cached")
    assert best(cached.ids["work_key"], title=cached.expect["title"])[0] == "pass"
    for key in ("overlap@second_test", "lattice@second_test", "noid_none@second_test", "port_b@test_proj"):
        assert best(key)[0] == "pass", key


# --- determinism -------------------------------------------------------------

def _snapshot(home: Path) -> dict[str, object]:
    out: dict[str, object] = {}
    for path in sorted(home.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(home).as_posix()
        if path.suffix == ".db":
            conn = ro(path)
            out[rel] = list(conn.iterdump())
            conn.close()
        else:
            out[rel] = path.read_bytes()
    return out


def test_builds_are_deterministic(tmp_path):
    a = build_v2_home(tmp_path / "a" / "home")
    b = build_v2_home(tmp_path / "elsewhere" / "nested" / "home")
    snap_a, snap_b = _snapshot(a.home), _snapshot(b.home)
    assert sorted(snap_a) == sorted(snap_b)
    for rel in snap_a:
        assert snap_a[rel] == snap_b[rel], rel
    assert [(i.kind, i.ids) for i in a.items] == [(i.kind, i.ids) for i in b.items]
    assert str(tmp_path) not in json.dumps([i.ids for i in a.items])


def test_refuses_a_non_empty_target(tmp_path):
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "something").write_text("x")
    with pytest.raises(FileExistsError):
        build_v2_home(tmp_path / "home")
