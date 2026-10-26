"""The PDF route and the paper-page security rules (docs/V3_UI_DESIGN.md §8): blobs served by
``paper_id`` only, no path parameters, no inline script or style, CSRF first in every POST
form, routes never run SQL. Synthetic PDFs only."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.routing import Route

from . import paper_support as ps

ROUTE_MODULES = ("papers", "notes", "files")
WEB = Path(__file__).resolve().parents[2] / "src" / "seedgraph" / "web"


@pytest.fixture
def fx(home, tmp_path):
    return ps.build(home, tmp_path)


@pytest.fixture
def client(fx, make_app, client_for):
    return client_for(make_app())


def test_the_pdf_is_served_by_paper_id_as_application_pdf(client, fx):
    r = client.get(f"/papers/{fx.held}/pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.headers["content-disposition"].startswith("inline")
    assert r.content.startswith(b"%PDF")
    with fx.open() as lib:
        uri = lib.conn.execute("SELECT storage_uri FROM source_files WHERE source_file_id = ?",
                               (fx.source_file_id,)).fetchone()[0]
    assert r.content == (fx.lib_dir / uri).read_bytes()


def test_range_requests_are_honoured_for_pdfjs(client, fx):
    r = client.get(f"/papers/{fx.held}/pdf", headers={"Range": "bytes=0-4"})
    assert r.status_code == 206 and r.content == b"%PDF-"


@pytest.mark.parametrize("path", [
    "/papers/lp_00000000000000000803/pdf",       # cited-only
    "/papers/lp_unknown/pdf",                    # no such paper
    "/papers/x/pdf",
    "/papers/..%2f..%2flibrary.db/pdf",
    "/papers/lp_00000000000000000801/pdf/../../library.db",
    "/papers/lp_00000000000000000801/pdf?path=library.db",
])
def test_anything_but_a_held_paper_id_is_404(client, fx, path):
    r = client.get(path)
    if path.endswith("?path=library.db"):
        assert r.content.startswith(b"%PDF")  # the query is ignored: still the paper's own PDF
    else:
        assert r.status_code == 404, path


def test_a_missing_blob_is_404(client, fx):
    with fx.open() as lib:
        uri = lib.conn.execute("SELECT storage_uri FROM source_files WHERE source_file_id = ?",
                               (fx.source_file_id,)).fetchone()[0]
    (fx.lib_dir / uri).unlink()
    assert client.get(f"/papers/{fx.held}/pdf").status_code == 404


def test_no_paper_route_takes_a_path_parameter(make_app):
    import importlib

    for name in ROUTE_MODULES:
        mod = importlib.import_module(f"seedgraph.web.routes.{name}")
        for r in mod.ROUTES:
            assert isinstance(r, Route)
            assert set(r.param_convertors) <= {"paper_id"}, r.path
            assert ":path" not in r.path
            assert "{paper_id}" in r.path


def test_route_modules_never_run_sql():
    for name in (*ROUTE_MODULES, "_paperkit"):
        src = (WEB / "routes" / f"{name}.py").read_text(encoding="utf-8")
        assert "sqlite3" not in src and "execute(" not in src, name


def test_every_mutating_paper_route_is_post_only(make_app):
    import importlib

    for name in ROUTE_MODULES:
        for r in importlib.import_module(f"seedgraph.web.routes.{name}").ROUTES:
            assert r.methods in ({"GET", "HEAD"}, {"POST"}), r.path


# --- CSP: what the pages and partials render --------------------------------------------


def _pages(client, fx) -> dict[str, str]:
    from seedgraph.notes import overlay

    with fx.open() as lib:
        overlay.save_edit(lib, fx.held, {"op": "hide", "target": {"claim_id": fx.claim("assumptions[1]")}},
                          expected_version=0)
        ps.save_note(lib, fx.held, fx.markdown_id, ps.raw_note(method_text="Newer invented wording."))
    out = {p: client.get(p).text for p in (
        f"/papers/{fx.held}", f"/papers/{fx.cited}", f"/papers/{fx.plain}", f"/papers/{fx.held}/notes?show_hidden=1",
        f"/papers/{fx.held}/text?page=1", f"/papers/{fx.held}/replay")}
    out["anchor"] = ps.post(client, f"/papers/{fx.held}/anchor", {"page": 3, "text": "frumious index"}).text
    return out


def test_no_inline_script_style_or_event_handlers(client, fx):
    for path, html in _pages(client, fx).items():
        for tag in re.findall(r"<script\b[^>]*>", html):
            assert "src=" in tag, (path, tag)
        assert not re.search(r"<style\b", html), path
        assert not re.search(r"\sstyle=", html), path
        assert not re.search(r"\shx-on", html), path
        assert not re.search(r"\son[a-z]+=", html), path


def test_every_post_form_carries_the_csrf_field_first(client, fx):
    seen = 0
    for path, html in _pages(client, fx).items():
        for m in re.finditer(r"<form\b([^>]*)>(.*?)</form>", html, re.S):
            if 'method="post"' not in m.group(1):
                continue
            seen += 1
            first = re.search(r"<(input|select|textarea|button)\b[^>]*>", m.group(2))
            assert first and 'name="csrf_token"' in first.group(0), (path, m.group(1))
    assert seen >= 10


def test_static_scripts_use_no_eval_and_pass_is_eval_supported_false():
    for name in ("pdfpane.js", "select.js"):
        src = (WEB / "static" / "js" / name).read_text(encoding="utf-8")
        assert "eval(" not in src and "new Function" not in src and "innerHTML =" not in src, name
    pane = (WEB / "static" / "js" / "pdfpane.js").read_text(encoding="utf-8")
    assert "isEvalSupported: false" in pane
    assert "/static/vendor/pdfjs/pdf.worker.min.mjs" in pane


# --- the reusable title macros (for U7's lists) -------------------------------------------


def test_the_pdf_button_macro_renders_only_for_held_papers(make_app):
    env = make_app().state.web.templates.env
    tpl = env.from_string('{% from "_pdf_button.html" import paper_title, pdf_button %}'
                          '{{ paper_title("lp_1", "A <b> title", True) }}|{{ pdf_button("lp_2", False) }}')
    html = tpl.render()
    held, cited = html.split("|")
    assert 'href="/papers/lp_1"' in held and "A &lt;b&gt; title" in held
    assert 'href="/papers/lp_1/pdf" target="_blank"' in held
    assert cited == ""
