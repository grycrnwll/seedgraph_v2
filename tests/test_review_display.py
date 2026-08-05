"""Review-list display: human decisions instead of opaque ids (TODO UX gap,
found on the first live semantic run).

``seedgraph review list`` and the web review screen share one rendering seam
(:func:`seedgraph.project.review.review_rows`): a ``concept_merge_candidate`` row
shows the merge PAIR (``member -> canonical``, surfaces preferred over the
normalized merge-key labels); a ``citation_resolution`` row shows the payload's
human fields (``first_author (year) — title`` truncated sanely, the citing
work's label, ``status, N candidates``) instead of ``rq_``/``ref_`` ids. The
``rq_`` item_id stays as a trailing column because ``review resolve`` takes it.

Offline / keyless: CliRunner over the Typer app + Starlette TestClient over the
FastAPI app, both against the ``isolated_home`` autouse fixture.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from seedgraph.api.app import app as web_app
from seedgraph.cli import app as cli_app
from seedgraph.project import review as review_mod
from seedgraph.project import service
from seedgraph.web import serve

runner = CliRunner()

LONG_TITLE = (
    "Difference-in-Differences with Continuous Treatments: "
    "Identification, Estimation, and Inference Under Staggered Adoption"
)


def _merge_item(h) -> str:
    return review_mod.enqueue(
        h,
        "concept_merge_candidate",
        target_type="concept",
        target_id="c_canon",
        payload={
            "kind": "concept_merge_candidate",
            # normalized merge-keys differ from the human surfaces on purpose:
            # the row must render the SURFACES.
            "canonical_label": "continuous_treatment_extension",
            "member_label": "continuous_treatment",
            "canonical_concept_id": "c_canon",
            "member_concept_id": "c_member",
            "canonical_surface": "continuous treatment extension",
            "member_surface": "continuous treatment",
            "confidence": 0.61,
            "run_id": "R1",
        },
    )


def _citation_item(h, citing_work_id: str) -> str:
    return review_mod.enqueue(
        h,
        "citation_resolution",
        target_type="reference_entry",
        target_id="ref_disp",
        payload={
            "kind": "citation_resolution",
            "status": "ambiguous",
            "citing_work_id": citing_work_id,
            "reference_id": "ref_disp",
            "raw": "Doe, J. (2020). Some raw reference text.",
            "title": LONG_TITLE,
            "year": 2020,
            "first_author": "Doe",
            "candidates": [{"doi": "10.9/c0"}, {"doi": "10.9/c1"}],
            "run_id": "R1",
        },
    )


def test_review_list_renders_concept_merge_pair():
    h = service.create_project("dispmerge")
    item_id = _merge_item(h)
    result = runner.invoke(cli_app, ["review", "list", "dispmerge"])
    assert result.exit_code == 0, result.output
    # the merge PAIR, member first — what merges into what.
    assert "continuous treatment -> continuous treatment extension" in result.output
    # surfaces render, not the normalized merge-key labels.
    assert "continuous_treatment" not in result.output
    # item_id stays available (trailing column) — resolve takes it.
    assert item_id in result.output


def test_review_list_renders_citation_human_fields():
    h = service.create_project("dispcite")
    citing = service.add_work(
        h, ids={"doi": "10.1/citing"}, title="Citing Paper"
    ).work_id
    item_id = _citation_item(h, citing)
    result = runner.invoke(cli_app, ["review", "list", "dispcite"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Doe (2020)" in out
    # the title renders truncated — head visible, full string never dumped.
    assert "Difference-in-Differences with Continuous Treatments" in out
    assert LONG_TITLE not in out
    assert "Citing Paper" in out  # citing work label via display.derive_label
    assert "ambiguous, 2 candidates" in out  # status + candidate count
    assert "ref_disp" not in out  # the opaque ref_ id is gone from the row
    assert item_id in out  # … but the rq_ id stays (resolve takes it)


def test_web_review_screen_renders_decisions():
    h = service.create_project("dispweb")
    citing = service.add_work(
        h, ids={"doi": "10.1/wciting"}, title="Citing Paper"
    ).work_id
    _merge_item(h)
    _citation_item(h, citing)
    web_app.dependency_overrides[serve.require_local_session] = lambda: None
    try:
        resp = TestClient(web_app).get("/ui/projects/dispweb/review")
    finally:
        web_app.dependency_overrides.pop(serve.require_local_session, None)
    assert resp.status_code == 200
    text = resp.text
    # Jinja autoescape turns the ASCII arrow's '>' into '&gt;'.
    assert "continuous treatment -&gt; continuous treatment extension" in text
    assert "Doe (2020)" in text and "Citing Paper" in text
    assert "ambiguous, 2 candidates" in text
