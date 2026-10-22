"""Golden-set manifest init from a seed against a synthetic, read-only v2 home."""

from __future__ import annotations

import json

import pytest

from _golden_fixtures import FakeV2, synthetic_markdown
from seedgraph.golden import manifest, v2
from seedgraph.golden.errors import GoldenError


@pytest.fixture
def fake_v2(tmp_path):
    return FakeV2(tmp_path / "v2home")


def _seed(paper, gid="GS01", **over):
    entry = {"id": gid, "category": "math", "md": paper["md_prefix"], "pdf": paper["pdf_prefix"],
             "project": paper["slug"]}
    entry.update(over)
    return entry


def test_build_manifest_resolves_full_hashes_and_current_note(fake_v2):
    p = fake_v2.add_paper(markdown=synthetic_markdown(2, ["alpha beta"]), quotes=["alpha beta"],
                          older_note_quotes=["stale quote"])
    entries = manifest.build_manifest([_seed(p)], default_v2_home=fake_v2.home)
    assert entries == [{
        "id": "GS01",
        "category": "math",
        "pdf_sha256": p["pdf_hash"],
        "v2_markdown_id": p["markdown_id"],
        "quote_source": {
            "v2_home": str(fake_v2.home),
            "project_slug": "proj",
            "work_id": p["work_id"],
            "schema_id": "default_research_note_v1",
            "note_id": p["note_id"],
            "note_created_at": "2026-10-20T00:00:00+00:00",
        },
    }]


def test_pdf_prefix_optional_resolved_from_markdown(fake_v2):
    p = fake_v2.add_paper(markdown=synthetic_markdown(2, []), quotes=["x"])
    seed = _seed(p)
    del seed["pdf"]
    [entry] = manifest.build_manifest([seed], default_v2_home=fake_v2.home)
    assert entry["pdf_sha256"] == p["pdf_hash"]


def test_zero_match_fails_loudly(fake_v2):
    p = fake_v2.add_paper(markdown=synthetic_markdown(2, []), quotes=["x"])
    with pytest.raises(GoldenError, match="no markdown"):
        manifest.build_manifest([_seed(p, md="ffffffffffff")], default_v2_home=fake_v2.home)


def test_ambiguous_prefix_fails_loudly(fake_v2):
    # Two docs; find a shared 1-char prefix by brute force over a few synthetic texts.
    papers = [fake_v2.add_paper(markdown=synthetic_markdown(2, [f"q{i}"]), quotes=[f"q{i}"]) for i in range(20)]
    firsts = {}
    for p in papers:
        firsts.setdefault(p["md_prefix"][0], []).append(p)
    shared = next(v for v in firsts.values() if len(v) > 1)
    with pytest.raises(GoldenError, match="ambiguous"):
        manifest.build_manifest([_seed(shared[0], md=shared[0]["md_prefix"][0], pdf=None)],
                                default_v2_home=fake_v2.home)


def test_pdf_prefix_mismatch_fails(fake_v2):
    a = fake_v2.add_paper(markdown=synthetic_markdown(2, ["a"]), quotes=["a"])
    b = fake_v2.add_paper(markdown=synthetic_markdown(3, ["b"]), quotes=["b"])
    with pytest.raises(GoldenError, match="does not match"):
        manifest.build_manifest([_seed(a, pdf=b["pdf_prefix"])], default_v2_home=fake_v2.home)


def test_write_and_load_round_trip(tmp_path, fake_v2):
    p = fake_v2.add_paper(markdown=synthetic_markdown(2, ["a"]), quotes=["a"])
    entries = manifest.build_manifest([_seed(p)], default_v2_home=fake_v2.home)
    home = tmp_path / "v3home"
    path = manifest.write_manifest(entries, home)
    assert path == home / "golden_set.json"
    assert json.loads(path.read_text(encoding="utf-8")) == entries
    assert manifest.load_manifest(home) == entries


def test_refuses_to_write_into_a_v2_home(fake_v2):
    p = fake_v2.add_paper(markdown=synthetic_markdown(2, ["a"]), quotes=["a"])
    entries = manifest.build_manifest([_seed(p)], default_v2_home=fake_v2.home)
    with pytest.raises(GoldenError, match="v2 home"):
        manifest.write_manifest(entries, fake_v2.home)


def test_v2_reads_are_read_only(fake_v2):
    p = fake_v2.add_paper(markdown=synthetic_markdown(2, ["a"]), quotes=["a"])
    conn = v2.connect_ro(fake_v2.home / "cache" / "cache.db")
    with pytest.raises(Exception, match="readonly"):
        conn.execute("CREATE TABLE t (x)")
    conn.close()


def test_stored_rendition_and_quotes(fake_v2):
    md = synthetic_markdown(2, ["alpha beta", "gamma"])
    p = fake_v2.add_paper(markdown=md, quotes=["alpha beta", "gamma"], older_note_quotes=["old one"],
                          created_at="2026-07-01T10:00:00+00:00", completed_at="2026-07-01T10:01:30+00:00")
    [entry] = manifest.build_manifest([_seed(p)], default_v2_home=fake_v2.home)
    r = v2.stored_rendition(fake_v2.home, entry["v2_markdown_id"])
    assert r.markdown == md
    assert r.seconds == 90.0
    assert r.converter_version == "1.10.2"
    assert "model load" in r.timing_label
    assert sorted(v2.note_quotes(entry["quote_source"])) == ["alpha beta", "gamma"]
