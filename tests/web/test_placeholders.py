"""Spot-check is a static placeholder until slice R6 (stage 5 design §11); Records is a real
page since slice R5 (tests/web/test_records.py)."""

from __future__ import annotations


def test_the_spot_check_placeholder_still_says_it_arrives_in_stage_5(make_app, client_for):
    r = client_for(make_app()).get("/spot-check")
    assert r.status_code == 200
    assert "<h1>Spot-check</h1>" in r.text
    assert "arrives in stage 5" in r.text
    assert 'href="/spot-check" aria-current="page"' in r.text


def test_records_is_no_longer_a_placeholder(make_app, client_for):
    r = client_for(make_app()).get("/records")
    assert r.status_code == 200
    assert "arrives in stage 5" not in r.text
    assert 'href="/records" aria-current="page"' in r.text
