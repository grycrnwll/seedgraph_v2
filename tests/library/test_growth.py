"""Guided corpus growth (docs/V3_CONCEPTS_DESIGN.md §8.2, §12 C6): seeds -> walk -> found
papers ranked by member citations -> researcher approval -> open-access fetch or the
needs-a-PDF list. A fake provider chain, synthetic PDFs, invented titles; no network."""

from __future__ import annotations

import httpx
import pytest

from seedgraph.acquisition.fetch import FetchResult, OACandidate, fetch_oa
from seedgraph.library import files, graphs, growth, ingest
from seedgraph.library.identity import derive_key, paper_id_for_key
from seedgraph.library.store import open_library_at

from .test_files import text_pdf


@pytest.fixture
def lib(tmp_path):
    library = open_library_at(tmp_path / "home" / "library")
    try:
        yield library
    finally:
        library.close()


def rec(doi: str, title: str, year: int = 2030, **extra) -> dict:
    return {"doi": doi, "title": title, "year": year, "authors": ["Ada Inventor"], **extra}


class GrowthChain:
    """A provider chain from dicts: ``records`` by DOI, ``refs`` (DOI -> referenced records),
    ``oa`` (DOI -> open-access candidates). Records every call by verb."""

    def __init__(self, records=None, refs=None, oa=None, titles=None, openalex=None):
        self.records = {k.lower(): v for k, v in (records or {}).items()}
        self.refs = {k.lower(): v for k, v in (refs or {}).items()}
        self.oa = {k.lower(): v for k, v in (oa or {}).items()}
        self.titles = titles or {}
        self.openalex = openalex or {}
        self.calls: list[tuple[str, str]] = []

    async def by_doi(self, doi):
        self.calls.append(("by_doi", doi))
        r = self.records.get(str(doi).lower())
        return dict(r) if r else None

    async def by_title(self, title, year=None):
        self.calls.append(("by_title", title))
        r = self.titles.get(title)
        return dict(r) if r else None

    async def by_openalex_ids(self, ids):
        self.calls.append(("by_openalex_ids", ",".join(ids)))
        return [dict(self.openalex[i]) for i in ids if i in self.openalex]

    async def referenced_works(self, record):
        key = str(record.get("doi") or "").lower()
        self.calls.append(("referenced_works", key))
        return [dict(r) for r in self.refs.get(key, [])]

    async def oa_pdf_candidates(self, record):
        key = str(record.get("doi") or "").lower()
        self.calls.append(("oa_pdf_candidates", key))
        return list(self.oa.get(key, []))


def held(lib, tmp_path, paper_id: str, title: str, doi: str) -> str:
    now = "2030-01-01T00:00:00+00:00"
    with lib.write() as conn:
        conn.execute("INSERT INTO papers (paper_id, title, year, created_at) VALUES (?, ?, 2030, ?)",
                     (paper_id, title, now))
        conn.execute("INSERT INTO identifiers (paper_id, id_type, id_value) VALUES (?, 'doi', ?)", (paper_id, doi))
    files.attach_pdf(lib, paper_id, text_pdf(tmp_path / f"{paper_id}.pdf", [f"{title} doi:{doi}", "Body."]))
    return paper_id


def make_graph(lib, slug: str, members: list[str], seeds: list[str] | None = None) -> None:
    seeds = members if seeds is None else seeds
    with graphs.create_graph(lib, slug) as g:
        for pid in members:
            graphs.set_membership(g, pid, "member", is_seed=pid in seeds)


def pid_for(doi: str) -> str:
    return paper_id_for_key(derive_key(ids=[("doi", doi)]))


X, Y, Z = "10.9999/x.1", "10.9999/y.2", "10.9999/z.3"


@pytest.fixture
def three_members(lib, tmp_path):
    a = held(lib, tmp_path, "lp_a", "Invented Ovens at Dawn", "10.9999/a")
    b = held(lib, tmp_path, "lp_b", "Invented Bread Prices", "10.9999/b")
    c = held(lib, tmp_path, "lp_c", "Invented Flour Mills", "10.9999/c")
    make_graph(lib, "ovens", [a, b, c])
    chain = GrowthChain(refs={
        "10.9999/a": [rec(X, "Invented Yeast Ledger"), rec(Y, "Invented Crust Census")],
        "10.9999/b": [rec(X, "Invented Yeast Ledger"), rec(Z, "Invented Salt Roads")],
        "10.9999/c": [rec(X, "Invented Yeast Ledger"), rec(Y, "Invented Crust Census")],
    })
    return chain


# --- seeds and the estimate ------------------------------------------------------------------


def test_seeds_default_to_the_seed_flags_and_the_estimate_needs_no_network(lib, tmp_path):
    a = held(lib, tmp_path, "lp_a", "Invented Ovens at Dawn", "10.9999/a")
    b = held(lib, tmp_path, "lp_b", "Invented Bread Prices", "10.9999/b")
    make_graph(lib, "ovens", [a, b], seeds=[b])
    make_graph(lib, "loaves", [a, b], seeds=[])

    assert growth.default_seeds(lib, "ovens") == [b]
    assert growth.default_seeds(lib, "loaves") == [a, b]
    est = growth.estimate(seed_count=2, depth=2, cap=10)
    assert (est["expected_papers"], est["api_calls"]) == (22, 66)
    with pytest.raises(growth.GrowthError):
        growth.walk(lib, "ovens", chain=GrowthChain(), depth=3)


# --- walk and ranking ------------------------------------------------------------------------


def test_found_papers_are_ranked_by_how_many_members_cite_them(lib, three_members):
    growth.walk(lib, "ovens", chain=three_members, depth=1, cap=10)

    found = growth.review(lib, "ovens")

    assert [(f.paper_id, f.cited_by_members) for f in found] == [(pid_for(X), 3), (pid_for(Y), 2), (pid_for(Z), 1)]
    top = found[0]
    assert top.title == "Invented Yeast Ledger" and top.held is False and top.generation == 1
    assert any("cited by 3 members" in r for r in top.reasons)


def counts(lib) -> dict[str, int]:
    return {t: lib.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("papers", "identifiers", "citation_links", "reference_entries", "attention_items")}


def status_of(lib, paper_id):
    return lib.conn.execute("SELECT status FROM paper_status WHERE paper_id = ?", (paper_id,)).fetchone()[0]


def test_found_papers_resolve_to_library_papers_by_stage_one_identity(lib, tmp_path):
    a = held(lib, tmp_path, "lp_a", "Invented Ovens at Dawn", "10.9999/a")
    known = held(lib, tmp_path, "lp_known", "Invented Yeast Ledger", X)  # held, not in the graph
    make_graph(lib, "ovens", [a])
    chain = GrowthChain(
        refs={"10.9999/a": [rec(X.upper(), "Invented Yeast Ledger"), rec(Y, "Invented Crust Census"),
                            {"openalex_id": "W77"}, {"title": "A Reference With No Id"}]},
        openalex={"W77": {"openalex_id": "W77", "doi": Z, "title": "Invented Salt Roads", "year": 2029}})

    result = growth.walk(lib, "ovens", chain=chain, depth=1, cap=10)

    # the DOI already in the library maps to that paper; new references are cited-only papers
    # whose ids are derived from their strong ids (a bare OpenAlex reference gains its DOI from the
    # batched lookup, and the DOI keys it); a reference with no id is not resolved
    assert set(result.new_papers) == {pid_for(Y), pid_for(Z)}
    assert status_of(lib, pid_for(Y)) == "cited_only"
    salt = lib.conn.execute("SELECT title, year FROM papers WHERE paper_id = ?",
                            (pid_for(Z),)).fetchone()
    assert tuple(salt) == ("Invented Salt Roads", 2029)
    found = growth.review(lib, "ovens")
    assert {f.paper_id for f in found} == {known, pid_for(Y), pid_for(Z)}
    links = lib.conn.execute("SELECT target_paper_id, provenance FROM citation_links WHERE source_paper_id = ?",
                             (a,)).fetchall()
    assert {tuple(r) for r in links} == {(known, "provider_reference"), (pid_for(Y), "provider_reference"),
                                         (pid_for(Z), "provider_reference")}
    (row,) = [f for f in found if f.paper_id == known]
    assert row.held and any("already held" in r for r in row.reasons)

    # walking again maps the bare OpenAlex reference to the paper it became (its stored id)
    before, lookups = counts(lib), [c for c in chain.calls if c[0] == "by_openalex_ids"]
    again = growth.walk(lib, "ovens", chain=chain, depth=1, cap=10)
    assert again.new_papers == () and counts(lib) == before
    assert [c for c in chain.calls if c[0] == "by_openalex_ids"] == lookups


def test_nothing_joins_the_graph_without_approval(lib, three_members):
    with graphs.open_graph(lib, "ovens") as g:
        before = graphs.membership(g)

    growth.walk(lib, "ovens", chain=three_members, depth=2, cap=10)

    with graphs.open_graph(lib, "ovens") as g:
        assert graphs.membership(g) == before


def test_a_rewalk_with_the_same_answers_changes_nothing(lib, three_members):
    first = growth.walk(lib, "ovens", chain=three_members, depth=2, cap=10)
    before, found = counts(lib), growth.review(lib, "ovens")

    again = growth.walk(lib, "ovens", chain=three_members, depth=2, cap=10)

    assert first.new_papers and again.new_papers == () and again.links_added == 0
    assert counts(lib) == before
    assert growth.review(lib, "ovens") == found


def test_the_cap_bounds_new_papers_per_generation_best_ranked_first(lib, three_members):
    result = growth.walk(lib, "ovens", chain=three_members, depth=1, cap=2)

    assert set(result.new_papers) == {pid_for(X), pid_for(Y)}
    assert result.generations[0].skipped_new == 1
    assert not lib.conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (pid_for(Z),)).fetchone()


def test_depth_two_expands_the_best_found_papers(lib, three_members):
    three_members.refs[X] = [rec("10.9999/deep.1", "Invented Grain Treaty")]

    result = growth.walk(lib, "ovens", chain=three_members, depth=2, cap=1)

    assert [g.sources for g in result.generations] == [("lp_a", "lp_b", "lp_c"), (pid_for(X),)]
    deep = [f for f in growth.review(lib, "ovens") if f.generation == 2]
    assert [(f.paper_id, f.cited_by_members, f.cited_in_walk) for f in deep] == [(pid_for("10.9999/deep.1"), 0, 1)]


def test_excluded_papers_are_never_found_again(lib, three_members):
    growth.walk(lib, "ovens", chain=three_members, depth=1, cap=10)
    with graphs.open_graph(lib, "ovens") as g:
        graphs.set_membership(g, pid_for(X), "excluded")

    growth.walk(lib, "ovens", chain=three_members, depth=1, cap=10)

    assert pid_for(X) not in {f.paper_id for f in growth.review(lib, "ovens")}


# --- approval, open access and the needs-a-PDF list ------------------------------------------


def membership_of(lib, slug="ovens") -> dict[str, str]:
    with graphs.open_graph(lib, slug) as g:
        return {r.paper_id: r.status for r in graphs.membership(g)}


def mock_client(requests: list[str], body: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_the_walk_never_looks_for_or_fetches_pdfs(lib, three_members):
    growth.walk(lib, "ovens", chain=three_members, depth=2, cap=10)
    assert not [c for c in three_members.calls if c[0] == "oa_pdf_candidates"]


def test_approved_open_access_papers_are_fetched_and_queued_for_conversion(lib, three_members, tmp_path):
    pdf = text_pdf(tmp_path / "oa.pdf", ["Invented Yeast Ledger doi:" + X, "Body."]).read_bytes()
    three_members.oa[X] = [OACandidate(url="https://repo.invalid/x.pdf", provider="unpaywall", is_oa_asserted=True)]
    growth.walk(lib, "ovens", chain=three_members, depth=1, cap=10)
    requests: list[str] = []

    async def fetch(record):
        async with mock_client(requests, pdf) as client:
            return await fetch_oa(record, chain=three_members, client=client)

    (res,) = growth.approve(lib, "ovens", {pid_for(X): "frontier"}, chain=three_members, fetch=fetch)

    assert (res.status, res.oa, res.needs_pdf) == ("frontier", "held", False)
    assert res.oa_source == "unpaywall: https://repo.invalid/x.pdf"
    assert requests == ["https://repo.invalid/x.pdf"]
    row = lib.conn.execute(
        "SELECT s.access_class, s.acquisition_method, q.source_file_id IS NOT NULL FROM paper_files f "
        "JOIN source_files s ON s.source_file_id = f.source_file_id "
        "LEFT JOIN conversion_queue q ON q.source_file_id = f.source_file_id WHERE f.paper_id = ?",
        (pid_for(X),)).fetchone()
    assert tuple(row) == ("open_access", "open_access_fetch", 1)
    assert membership_of(lib)[pid_for(X)] == "frontier"
    # papers that were not ticked were neither fetched nor added
    assert [c for c in three_members.calls if c[0] == "oa_pdf_candidates"] == [("oa_pdf_candidates", X)]
    assert pid_for(Y) not in membership_of(lib)


def test_a_paywalled_copy_is_never_fetched_and_the_paper_needs_a_pdf(lib, three_members):
    three_members.oa[X] = [OACandidate(url="https://publisher.invalid/paywalled.pdf", provider="crossref",
                                       is_oa_asserted=False)]
    growth.walk(lib, "ovens", chain=three_members, depth=1, cap=10)
    requests: list[str] = []

    async def fetch(record):
        async with mock_client(requests, b"%PDF-1.4 should never be fetched") as client:
            return await fetch_oa(record, chain=three_members, client=client)

    (res,) = growth.approve(lib, "ovens", {pid_for(X): "frontier"}, chain=three_members, fetch=fetch)

    assert requests == []
    assert (res.status, res.oa, res.needs_pdf) == ("frontier", "none", True)
    assert res.doi_url == f"https://doi.org/{X}"
    assert status_of(lib, pid_for(X)) == "cited_only"
    (row,) = ingest.needs_pdf(lib, graph="ovens")
    assert (row.paper_id, row.doi_url) == (pid_for(X), f"https://doi.org/{X}")


def test_held_papers_join_as_the_researcher_chooses_and_cited_only_ones_only_as_frontier(lib, three_members, tmp_path):
    held(lib, tmp_path, "lp_known", "Invented Yeast Ledger", X)
    three_members.refs["10.9999/a"].append(rec("10.9999/q", "Invented Quay Tolls"))
    growth.walk(lib, "ovens", chain=three_members, depth=1, cap=10)
    no_fetch = []

    async def fetch(record):
        no_fetch.append(record)
        return FetchResult(status="stub")

    with pytest.raises(graphs.NotHeldError):
        growth.approve(lib, "ovens", {"lp_known": "member", pid_for(Y): "member"}, chain=three_members, fetch=fetch)
    assert "lp_known" not in membership_of(lib) and no_fetch == []  # refused before anything was written
    with pytest.raises(growth.GrowthError):  # approval never demotes a member
        growth.approve(lib, "ovens", {"lp_a": "frontier"}, chain=three_members, fetch=fetch)

    results = growth.approve(lib, "ovens", {"lp_known": "member", pid_for(Y): "frontier",
                                             pid_for("10.9999/q"): "frontier"}, chain=three_members, fetch=fetch)

    by_id = {r.paper_id: r for r in results}
    assert (by_id["lp_known"].status, by_id["lp_known"].oa) == ("member", "already_held")
    assert by_id[pid_for(Y)].needs_pdf and by_id[pid_for(Y)].oa == "none"
    assert membership_of(lib) == {"lp_a": "member", "lp_b": "member", "lp_c": "member", "lp_known": "member",
                                  pid_for(Y): "frontier", pid_for("10.9999/q"): "frontier"}
    assert len(no_fetch) == 2
    assert {f.paper_id for f in growth.review(lib, "ovens")} == {pid_for(Z)}
