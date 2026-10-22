"""Golden-set OLD/NEW sources, PDF page count, the runner and the report writer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from _golden_fixtures import FakeV2, synthetic_markdown
from seedgraph.golden import manifest, pdf, report, sources
from seedgraph.golden.errors import GoldenError
from seedgraph.golden.run import run_check

QUOTES = ["alpha beta", "gamma delta"]


@pytest.fixture
def setup(tmp_path):
    fake = FakeV2(tmp_path / "v2home")
    md = synthetic_markdown(3, QUOTES)
    p = fake.add_paper(markdown=md, quotes=QUOTES)
    seed = [{"id": "GS01", "category": "math", "md": p["md_prefix"], "project": p["slug"]}]
    entries = manifest.build_manifest(seed, default_v2_home=fake.home)
    return fake, md, entries


def test_pdf_page_count_of_synthetic_pdf(tmp_path):
    pdfium = pytest.importorskip("pypdfium2")
    doc = pdfium.PdfDocument.new()
    for _ in range(4):
        doc.new_page(200, 200)
    out = tmp_path / "four.pdf"
    doc.save(str(out))
    doc.close()
    assert pdf.page_count(out) == 4


def test_pdf_page_count_missing_file_is_none(tmp_path):
    assert pdf.page_count(tmp_path / "nope.pdf") is None


def test_v2_stored_source(setup):
    fake, md, [entry] = setup
    r = sources.V2StoredSource().rendition(entry)
    assert r.markdown == md and r.seconds == 60.0


def test_directory_source_reads_markdown_and_timings(setup, tmp_path):
    _, md, [entry] = setup
    d = tmp_path / "newmd"
    d.mkdir()
    (d / "GS01.md").write_text(md, encoding="utf-8")
    (d / "timings.json").write_text(json.dumps({"GS01": {"seconds": 12.5, "converter_version": "9.9"}}))
    r = sources.DirectorySource(d).rendition(entry)
    assert (r.markdown, r.seconds, r.converter_version) == (md, 12.5, "9.9")


def test_directory_source_plain_number_timing_and_missing_file(setup, tmp_path):
    _, md, [entry] = setup
    d = tmp_path / "newmd"
    d.mkdir()
    with pytest.raises(GoldenError, match="GS01.md"):
        sources.DirectorySource(d).rendition(entry)
    (d / "GS01.md").write_text(md, encoding="utf-8")
    (d / "timings.json").write_text(json.dumps({"GS01": 7}))
    assert sources.DirectorySource(d).rendition(entry).seconds == 7.0


@dataclass
class _Timings:
    convert_seconds: float


@dataclass
class _Result:
    markdown: str
    page_count: int
    timings: _Timings
    converter_version: str


@dataclass
class FakeConverter:
    markdown: str
    calls: list = field(default_factory=list)

    def convert(self, pdf_path, options):
        self.calls.append((pdf_path, options))
        return _Result(self.markdown, 3, _Timings(4.5), "1.10.2")


def test_convert_source_uses_injected_converter(setup):
    fake, md, [entry] = setup
    conv = FakeConverter(md)
    r = sources.ConvertSource(conv).rendition(entry)
    assert r.markdown == md and r.seconds == 4.5 and r.page_count == 3
    assert r.converter_version == "1.10.2"
    [(path, options)] = conv.calls
    assert str(path).endswith(entry["pdf_sha256"] + ".pdf")
    assert options["paginate_output"] is True


def test_convert_source_accepts_mapping_timings(setup):
    _, md, [entry] = setup

    class DictConverter:
        def convert(self, pdf_path, options):
            return _Result(md, 3, {"convert_seconds": 2.0, "wall_seconds": 2.5}, "1.10.2")

    assert sources.ConvertSource(DictConverter()).rendition(entry).seconds == 2.0


def test_make_source_convert_without_factory_raises(monkeypatch):
    monkeypatch.setattr(sources, "converter_factory", None)
    with pytest.raises(GoldenError, match="not wired"):
        sources.make_source("convert")


def test_make_source_convert_with_factory(monkeypatch):
    monkeypatch.setattr(sources, "converter_factory", lambda: FakeConverter("x"))
    assert isinstance(sources.make_source("convert"), sources.ConvertSource)


def test_run_check_identity_passes_and_report_written(setup, tmp_path):
    _, md, entries = setup
    results = run_check(entries, sources.V2StoredSource(), sources.V2StoredSource(), page_counter=lambda e: 3)
    [res] = results
    assert res.status == "pass"
    assert res.check("quotes").old == res.check("quotes").new == 1.0
    out = report.write_report(results, tmp_path / "v3home", meta={"old": "v2-stored", "new": "v2-stored"})
    assert out.parent == tmp_path / "v3home" / "golden_reports"
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert data["results"][0]["id"] == "GS01" and data["meta"]["old"] == "v2-stored"
    assert "GS01" in (out / "report.md").read_text(encoding="utf-8")
    assert "GS01" in report.render_table(results)


def test_run_check_marks_crashed_conversion_failed_and_continues(setup, tmp_path):
    fake, md, entries = setup
    p2 = fake.add_paper(markdown=synthetic_markdown(2, ["zeta"]), quotes=["zeta"])
    entries = entries + manifest.build_manifest(
        [{"id": "GS02", "category": "x", "md": p2["md_prefix"], "project": p2["slug"]}], default_v2_home=fake.home)

    class Crashy:
        def __init__(self):
            self.n = 0

        def convert(self, pdf_path, options):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("converter crashed")
            return _Result(synthetic_markdown(2, ["zeta"]), 2, _Timings(1.0), "1.10.2")

    results = run_check(entries, sources.V2StoredSource(), sources.ConvertSource(Crashy()),
                        page_counter=lambda e: None)
    assert [r.status for r in results][0] == "fail"
    assert "crashed" in results[0].check("source").reason
    assert results[1].check("quotes").new == 1.0
    assert "GS01" in report.render_table(results)


def test_run_check_falls_back_to_rendition_page_count(setup):
    _, md, entries = setup
    new = sources.ConvertSource(FakeConverter(md))
    [res] = run_check(entries, sources.V2StoredSource(), new, page_counter=lambda e: None)
    assert res.meta["page_count"] == 3
    assert res.check("page_markers").status == "pass"


def test_run_check_converts_on_every_pool_converter_at_once(setup):
    """A pool of N converters (``size``) gets N PDFs at a time, so both GPUs work."""
    import threading

    fake, md, entries = setup
    p2 = fake.add_paper(markdown=synthetic_markdown(3, ["zeta"]), quotes=["zeta"])
    entries = entries + manifest.build_manifest(
        [{"id": "GS02", "category": "x", "md": p2["md_prefix"], "project": p2["slug"]}],
        default_v2_home=fake.home)

    class TwoSlotPool:
        size = 2

        def __init__(self):
            self.barrier = threading.Barrier(2, timeout=10)

        def convert(self, pdf_path, options):
            self.barrier.wait()  # only returns if both PDFs are in flight together
            return _Result(md, 3, _Timings(1.0), "1.10.2")

    results = run_check(entries, sources.V2StoredSource(), sources.ConvertSource(TwoSlotPool()),
                        page_counter=lambda e: 3)
    assert [r.golden_id for r in results] == ["GS01", "GS02"]
    for r in results:
        assert "source" not in [c.name for c in r.checks], r.checks
