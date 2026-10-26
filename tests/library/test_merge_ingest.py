"""A researcher's "different papers" verdict stops ingest from raising that pair again."""

from __future__ import annotations

import json

from seedgraph.library import ingest, merge
from seedgraph.library.identity import derive_key, mint_paper_id

from .metadata_support import FakeChain, insert_paper, items, openalex_record
from .test_ingest import DOI, TITLE, lib, oa_fetch  # noqa: F401 - the fixture


def test_a_pair_decided_different_is_not_raised_again_at_doi_paste(lib):  # noqa: F811
    existing = insert_paper(lib, "lp_similar", title=TITLE, year=2031)
    # the pasted DOI's paper was added, judged different, deleted, and is now pasted again
    new_id = mint_paper_id(derive_key(ids=[("doi", DOI)]), set()).paper_id
    merge.record_different(lib, new_id, existing)
    chain = FakeChain({DOI: openalex_record(DOI, TITLE + ".", 2031)})

    (res,) = ingest.add_dois(lib, [DOI], chain=chain, fetch=oa_fetch(None))

    assert res.status == "created" and res.paper_id == new_id
    assert items(lib, "identity_check") == []


def test_other_candidates_are_still_raised(lib):  # noqa: F811
    first = insert_paper(lib, "lp_similar", title=TITLE, year=2031)
    second = insert_paper(lib, "lp_similar2", title=TITLE, year=2031)
    new_id = mint_paper_id(derive_key(ids=[("doi", DOI)]), set()).paper_id
    merge.record_different(lib, new_id, first)
    chain = FakeChain({DOI: openalex_record(DOI, TITLE + ".", 2031)})

    ingest.add_dois(lib, [DOI], chain=chain, fetch=oa_fetch(None))

    (item,) = items(lib, "identity_check")
    assert json.loads(item["subject_json"])["candidates"] == [second]
