"""Chunk 4 — standalone AnswerTrace HTML export (trace_plans 00 §6.1, 01 chunk 4).

Offline / keyless, over the phase_8 fixture corpus with ``no_llm`` asks (no LLM call
needed to exercise both the retrieval-only and citation_search/neighborhood paths).
Exercises the CLI ``seedgraph answer trace-export`` command end to end via typer's
``CliRunner``, plus the self-containment bar (criterion 4): the export must carry
zero live references. The self-containment check is anchored to ``src="/static/``,
``src="http``, and ``href="http`` — NOT a bare ``"http"`` substring scan, because the
inlined minified vendor JS legitimately contains ``http`` in SVG/XML namespace URIs
and license-header comments.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner
from _phase8_helpers import build_fixture_project, make_answer_config

from seedgraph.answer import answer, save_answer, save_trace
from seedgraph.cli import app as cli_app

GROUNDED_Q = 'According to Paper A, what holds "across groups"?'
CITE_Q = "Which works are cited by Paper A?"

runner = CliRunner()


def _persist(handle, env, tr, *, run_id=None):
    save_answer(env, slug=handle.slug, root=handle.root, run_id=run_id)
    save_trace(tr, slug=handle.slug, root=handle.root, run_id=run_id)


def _export(answer_id: str, slug: str, *, out=None):
    args = ["answer", "trace-export", answer_id, "--project", slug]
    if out is not None:
        args += ["--out", str(out)]
    return runner.invoke(cli_app, args)


# --------------------------------------------------------------------------
# self-containment (criterion 4): a neighborhood trace inlines the vendor payload
# --------------------------------------------------------------------------

def test_export_citation_search_trace_is_self_contained(tmp_path):
    h = build_fixture_project("trace_export_cite")
    env, tr = answer(CITE_Q, h, no_llm=True, config=make_answer_config())
    assert tr.neighborhood is not None
    _persist(h, env, tr)

    out_path = tmp_path / "out.trace.html"
    result = _export(env.answer_id, h.slug, out=out_path)
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    assert f"wrote {out_path}" in result.output
    html = out_path.read_text(encoding="utf-8")

    # question, disposition strings, and the embedded subgraph payload are present.
    assert CITE_Q in html
    assert "shown" in html
    assert 'id="trace-subgraph-data"' in html
    assert "work_c" in html                      # neighborhood membership id

    # self-containment: no live/external references (anchored, not a bare substring
    # scan — the inlined minified JS itself legitimately contains "http" inside SVG/
    # XML namespace URIs and license-header comments).
    assert 'src="/static/' not in html
    assert 'src="http' not in html
    assert 'href="http' not in html

    # positively prove the vendor payload was actually inlined, not merely absent.
    assert len(html) > 1_000_000


# --------------------------------------------------------------------------
# retrieval-only traces (no neighborhood) export WITHOUT the vendor payload
# --------------------------------------------------------------------------

def test_export_retrieval_only_trace_has_no_vendor_payload(tmp_path):
    h = build_fixture_project("trace_export_ro")
    env, tr = answer(GROUNDED_Q, h, no_llm=True, config=make_answer_config())
    assert tr.neighborhood is None
    _persist(h, env, tr)

    out_path = tmp_path / "out.trace.html"
    result = _export(env.answer_id, h.slug, out=out_path)
    assert result.exit_code == 0, result.output
    html = out_path.read_text(encoding="utf-8")

    assert "According to Paper A" in html
    assert "across groups" in html
    assert "shown" in html
    assert "<script src=" not in html
    assert 'id="trace-subgraph-data"' not in html
    # small: no ~1.4 MB vendor payload inlined.
    assert len(html) < 100_000


# --------------------------------------------------------------------------
# default --out path: beside the trace JSON, ad-hoc and run-scoped
# --------------------------------------------------------------------------

def test_export_default_out_path_adhoc():
    h = build_fixture_project("trace_export_default")
    env, tr = answer(GROUNDED_Q, h, no_llm=True, config=make_answer_config())
    _persist(h, env, tr)

    result = _export(env.answer_id, h.slug)
    assert result.exit_code == 0, result.output
    expected = h.root / "projects" / h.slug / "answers" / f"{env.answer_id}.trace.html"
    assert expected.exists()
    assert f"wrote {expected}" in result.output


def test_export_default_out_path_run_scoped_and_resolves_nested_path():
    """A run-scoped answer resolves without the CLI being told the run_id up front —
    ``resolve_trace_context`` searches ``runs/*/answers/`` after the ad-hoc miss."""
    h = build_fixture_project("trace_export_run_default")
    env, tr = answer(CITE_Q, h, no_llm=True, config=make_answer_config())
    _persist(h, env, tr, run_id="run_x")

    result = _export(env.answer_id, h.slug)
    assert result.exit_code == 0, result.output
    expected = (
        h.root / "projects" / h.slug / "runs" / "run_x" / "answers"
        / f"{env.answer_id}.trace.html"
    )
    assert expected.exists()
    html = expected.read_text(encoding="utf-8")
    assert 'id="trace-subgraph-data"' in html   # citation_search neighborhood present


# --------------------------------------------------------------------------
# unknown answer_id -> clean error, not a traceback
# --------------------------------------------------------------------------

def test_export_unknown_answer_id_is_clean_error():
    h = build_fixture_project("trace_export_missing")
    result = _export("ans_doesnotexist", h.slug)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "no trace found" in result.output


def test_export_unknown_project_is_clean_error():
    result = _export("ans_whatever", "trace_export_no_such_project")
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_export_traversal_answer_id_is_clean_error():
    """A traversal-shaped answer_id exits cleanly (non-zero, no traceback): the choke-point
    guard makes ``resolve_trace_context`` return None, so the CLI prints its normal
    "no trace found" error instead of interpolating the id into a filesystem path."""
    h = build_fixture_project("trace_export_traversal")
    for bad in ["../../secret", "a..b", "a/b"]:
        result = _export(bad, h.slug)
        assert result.exit_code != 0, result.output
        assert "Traceback" not in result.output
        assert "no trace found" in result.output


# --------------------------------------------------------------------------
# the served (non-standalone) route is unaffected by the template's new conditional
# --------------------------------------------------------------------------

def test_served_trace_route_still_uses_src_refs():
    from fastapi.testclient import TestClient

    from seedgraph.api.app import app
    from seedgraph.web import serve

    h = build_fixture_project("trace_export_served")
    env, tr = answer(CITE_Q, h, no_llm=True, config=make_answer_config())
    _persist(h, env, tr)

    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(app)
        resp = client.get(f"/ui/projects/{h.slug}/answers/{env.answer_id}/trace")
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)
    assert resp.status_code == 200, resp.text[:300]
    assert 'src="/static/vendor/three.min.js"' in resp.text
    assert 'src="/static/vendor/3d-force-graph.min.js"' in resp.text
    assert "<script>" not in resp.text.split("BEGIN vendored graph assets")[1].split(
        "END vendored graph assets"
    )[0]


# --------------------------------------------------------------------------
# template divergence risk (#5): the export's embedded payload equals the served page's
# --------------------------------------------------------------------------

def test_export_and_served_embed_same_subgraph_payload():
    from fastapi.testclient import TestClient

    from seedgraph.api.app import app
    from seedgraph.project.service import open_project
    from seedgraph.web import serve
    from seedgraph.web.ui import render_trace_standalone, resolve_trace_context

    h = build_fixture_project("trace_export_parity")
    env, tr = answer(CITE_Q, h, no_llm=True, config=make_answer_config())
    _persist(h, env, tr)

    app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        client = TestClient(app)
        served_html = client.get(f"/ui/projects/{h.slug}/answers/{env.answer_id}/trace").text
    finally:
        app.dependency_overrides.pop(serve.require_local_session, None)

    handle = open_project(h.slug, root=h.root)
    ctx = resolve_trace_context(handle, env.answer_id)
    exported_html = render_trace_standalone(ctx)

    def _payload(html: str) -> str:
        m = re.search(
            r'<script id="trace-subgraph-data" type="application/json">(.*?)</script>',
            html, re.S,
        )
        assert m, "embedded subgraph payload script tag not found"
        return m.group(1)

    assert _payload(served_html) == _payload(exported_html)
