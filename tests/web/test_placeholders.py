"""Records and Spot-check are static placeholders until stage 5 (design §2)."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("path,title,nav", [("/records", "Records", "/records"),
                                             ("/spot-check", "Spot-check", "/spot-check")])
def test_placeholder_pages_say_they_arrive_in_stage_5(make_app, client_for, path, title, nav):
    r = client_for(make_app()).get(path)
    assert r.status_code == 200
    assert f"<h1>{title}</h1>" in r.text
    assert "arrives in stage 5" in r.text
    assert f'href="{nav}" aria-current="page"' in r.text
