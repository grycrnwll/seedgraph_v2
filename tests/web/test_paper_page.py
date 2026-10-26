"""The paper page (docs/V3_UI_DESIGN.md §2.4): header, badges, layout toggle, notes pane with
quote locators, PDF pane hooks. Synthetic PDFs, invented text."""

from __future__ import annotations

import re

import pytest

from . import paper_support as ps


@pytest.fixture
def fx(home, tmp_path):
    return ps.build(home, tmp_path)


@pytest.fixture
def client(fx, make_app, client_for):
    return client_for(make_app())


def test_the_header_shows_metadata_provenance_and_held_badges(client, fx):
    r = client.get(f"/papers/{fx.held}")
    assert r.status_code == 200
    html = r.text
    assert ps.TITLE in html
    assert "Ada Inventor" in html and "Bo Madeup" in html
    assert "2031" in html and "Journal of Invented Results" in html
    assert "10.9999/invented.2031.001" in html
    assert "Title from: v2" in html
    assert 'data-badge="held"' in html
    assert 'data-badge="cited-only"' not in html
    assert f'href="/papers/{fx.held}/pdf"' in html  # Open PDF
    assert 'data-badge="pdf-check"' in html
    assert "fake-fixed-1" in html  # converter version


def test_an_unknown_paper_is_404(client):
    assert client.get("/papers/lp_nope").status_code == 404
    assert client.get("/papers/x").status_code == 404


def test_a_cited_only_paper_offers_add_pdf_and_no_open_pdf(client, fx):
    html = client.get(f"/papers/{fx.cited}").text
    assert 'data-badge="cited-only"' in html
    assert "Title from: OpenAlex (verified by DOI)" in html
    assert f'href="/papers/{fx.cited}/pdf"' not in html
    m = re.search(r'<form method="post" action="/papers/%s/add-pdf"[^>]*>\s*(<input[^>]*>)' % fx.cited, html)
    assert m and 'name="csrf_token"' in m.group(1)
    assert "/write-notes" not in html


@pytest.mark.parametrize("layout", ["split", "pdf", "notes"])
def test_the_layout_toggle_comes_from_the_query(client, fx, layout):
    html = client.get(f"/papers/{fx.held}?layout={layout}").text
    assert f'data-layout="{layout}"' in html
    assert re.search(r'data-layout-set="%s" aria-pressed="true"' % layout, html)


def test_an_unknown_layout_falls_back_to_side_by_side(client, fx):
    assert 'class="sg-layout sg-paper-layout" data-layout="split"' in client.get(
        f"/papers/{fx.held}?layout=bogus").text


def test_quotes_carry_their_page_locators_and_trust_labels(client, fx):
    html = client.get(f"/papers/{fx.held}").text
    # the research question quote is on page 1, the method quote on page 2, the result on page 3
    assert re.search(r'data-page="1"[^>]*data-how="marker" data-needle="westudywhetherinventedwidgets', html)
    assert ">p. 2<" in html and ">p. 3<" in html
    assert ps.METHOD_QUOTE in html
    assert 'data-trust="paper_states"' in html and "paper states" in html
    assert 'data-trust="agent_inferred"' in html  # the inferred assumption
    assert 'data-pdf-url="/papers/%s/pdf"' % fx.held in html
    # claim groups in the fixed order
    order = [html.index(f">{t}</h2>") for t in ("Research question", "Methods", "Identification assumptions",
                                                 "Results")]
    assert order == sorted(order)


def test_a_paper_without_page_markers_says_so_and_never_guesses_a_page(home, tmp_path, make_app, client_for):
    fx = ps.build(home, tmp_path)
    with fx.open() as lib:
        md = lib.conn.execute("SELECT markdown_id FROM paper_files WHERE paper_id = ?", (fx.plain,)).fetchone()[0]
        ps.save_note(lib, fx.plain, md)
    html = client_for(make_app()).get(f"/papers/{fx.plain}").text
    assert 'data-badge="no-pages"' in html
    assert "data-no-pages" in html
    assert "data-no-page>no page<" in html
    assert ">p. 1<" not in html


def test_the_write_notes_action_shows_the_token_estimate(client, fx):
    html = client.get(f"/papers/{fx.held}").text
    assert "Write notes (replace)" in html
    assert re.search(r"data-estimate>about [\d,]+ paper text tokens; agent [\d,]+–[\d,]+ tokens", html)


def test_write_notes_creates_a_one_paper_batch_and_goes_to_its_page(client, fx):
    r = ps.post(client, f"/papers/{fx.held}/write-notes", htmx=False)
    assert r.status_code == 303
    loc = r.headers["location"]
    assert loc.startswith("/batches/")
    batch_id = loc.split("?")[0].rsplit("/", 1)[1]
    with fx.open() as lib:
        row = lib.conn.execute("SELECT task_type, selection_json, options_json FROM batches WHERE batch_id = ?",
                               (batch_id,)).fetchone()
    assert row[0] == "notes" and fx.held in row[1] and '"replace":true' in row[2]


def test_metadata_edit_sets_the_user_source(client, fx):
    r = ps.post(client, f"/papers/{fx.held}/metadata", {
        "title": "  A Researcher-Corrected   Invented Title ", "year": "2030", "venue": "Invented Letters",
        "authors": "Cy Fictional\n\n Di Imaginary "}, htmx=False)
    assert r.status_code == 303
    html = client.get(r.headers["location"]).text
    assert "A Researcher-Corrected Invented Title" in html
    assert "Title from: you" in html
    assert "Cy Fictional, Di Imaginary" in html
    assert "metadata saved" in html


def test_metadata_edit_refuses_an_empty_title(client, fx):
    r = ps.post(client, f"/papers/{fx.held}/metadata", {"title": "  ", "year": ""}, htmx=False)
    html = client.get(r.headers["location"]).text
    assert "metadata not saved" in html and ps.TITLE in html


def test_add_to_graph_adds_a_member_and_refuses_a_cited_only_member(client, fx):
    from seedgraph.library import graphs

    with fx.open() as lib:
        graphs.create_graph(lib, "widgets", display_name="Invented widgets").close()
    html = client.get(f"/papers/{fx.held}").text
    assert '<option value="widgets">Invented widgets</option>' in html
    r = ps.post(client, f"/papers/{fx.held}/graphs", {"slug": "widgets", "status": "member"}, htmx=False)
    assert "added to widgets" in client.get(r.headers["location"]).text
    html = client.get(f"/papers/{fx.held}").text
    assert 'href="/graphs/widgets"' in html and "(member)" in html

    r = ps.post(client, f"/papers/{fx.cited}/graphs", {"slug": "widgets", "status": "member"}, htmx=False)
    assert "cited-only" in client.get(r.headers["location"]).text
    with fx.open() as lib, graphs.open_graph(lib, "widgets") as g:
        assert [m.paper_id for m in graphs.membership(g)] == [fx.held]


def test_add_pdf_attaches_an_upload_to_the_cited_only_paper(client, fx):
    pdf = ps.text_pdf(fx.lib_dir.parent / "upload-src.pdf", ["A cited-only invented paper."], marker="up")
    tok = ps.token(client)
    r = client.post(f"/papers/{fx.cited}/add-pdf", headers={"X-CSRF-Token": tok, "HX-Request": "true"},
                    data={"csrf_token": tok},
                    files={"pdf": ("../../evil name.pdf", pdf.read_bytes(), "application/pdf")})
    assert r.status_code == 200
    assert r.headers["hx-redirect"].startswith(f"/papers/{fx.cited}?flash=")
    html = client.get(f"/papers/{fx.cited}").text
    assert 'data-badge="held"' in html
    assert client.get(f"/papers/{fx.cited}/pdf").content.startswith(b"%PDF")
    with fx.open() as lib:
        assert lib.conn.execute("SELECT 1 FROM conversion_queue").fetchone() is not None
    assert not any((fx.lib_dir.parent / "tmp" / "upload").glob("*/*"))


def test_add_pdf_refuses_a_file_that_is_not_a_pdf(client, fx):
    tok = ps.token(client)
    r = client.post(f"/papers/{fx.cited}/add-pdf", headers={"X-CSRF-Token": tok},
                    files={"pdf": ("x.pdf", b"not a pdf at all", "application/pdf")})
    assert r.status_code == 303
    assert "not a PDF" in client.get(r.headers["location"]).text
    assert 'data-badge="cited-only"' in client.get(f"/papers/{fx.cited}").text


def test_get_on_the_paper_pages_leaves_the_library_unchanged(client, fx):
    import sqlite3

    def dump():
        conn = sqlite3.connect(fx.lib_dir / "library.db")
        try:
            return list(conn.iterdump())
        finally:
            conn.close()

    before = dump()
    for path in (f"/papers/{fx.held}", f"/papers/{fx.held}?layout=notes&show_hidden=1", f"/papers/{fx.cited}",
                 f"/papers/{fx.plain}", f"/papers/{fx.held}/notes", f"/papers/{fx.held}/text?page=2",
                 f"/papers/{fx.held}/replay", f"/papers/{fx.held}/pdf"):
        assert client.get(path).status_code in (200, 404), path
    assert dump() == before
