"""The spot-check sample (design §5.1-5.2, §13 Q5-Q8; slice R4): the population, model
keys, the least-marked-first draw with its ties, the 2-per-paper cap, determinism,
filters and what an item shows. Invented fixtures only."""

from __future__ import annotations

import json

import pytest

from seedgraph.notes import overlay
from seedgraph.spotcheck import sample as sp
from seedgraph.spotcheck import marks

NOW = "2026-01-01T00:00:00+00:00"


def edit(lib, pid, cid, version, **set_):
    return overlay.save_edit(lib, pid, {"op": "edit", "target": {"claim_id": cid}, "set": set_},
                             expected_version=version)


def hide(lib, pid, cid, version):
    return overlay.save_edit(lib, pid, {"op": "hide", "target": {"claim_id": cid}}, expected_version=version)


def add_lens_claim(lib, pid):
    md = lib.conn.execute("SELECT markdown_id FROM paper_files WHERE paper_id = ?", (pid,)).fetchone()[0]
    with lib.write() as conn:
        conn.execute(
            "INSERT INTO extraction_runs (extraction_run_id, paper_id, markdown_id, markdown_hash, lens_id, "
            "schema_version, prompt_version, model_name, run_status, created_at) "
            "VALUES ('extr_lens1', ?, ?, 'h', 'lens_x', 'v1', 'v1', 'invented-model-a', 'success', ?)",
            (pid, md, NOW))
        conn.execute(
            "INSERT INTO extracted_claims (claim_id, structured_note_id, extraction_run_id, paper_id, claim_type, "
            "field_key, claim_text, status, epistemic_type, created_at) VALUES ('claim_lens1', NULL, 'extr_lens1', "
            "?, 'result', 'lens[0]', 'A lens claim about serial correlation.', 'found', 'x', ?)", (pid, NOW))
    return "claim_lens1"


# --- the population --------------------------------------------------------------------------

def test_population_includes_hidden_and_edited_excludes_added_lens_and_older_notes(world, lib):
    pid = world.paper()
    old = world.note(pid)
    cur = world.note(pid)  # a re-note: the first note is no longer current
    claims = world.claims(cur.note_id)
    edited, hidden = claims["method_or_model[0]"], claims["assumptions[0]"]
    edit(lib, pid, edited, 0, claim_text="An edited design with two periods.")
    hide(lib, pid, hidden, 1)
    overlay.save_edit(lib, pid, {"op": "add", "kind": "assertion", "claim_type": "result",
                                 "normalized_label": "added", "claim_text": "A researcher-added result."},
                      expected_version=2)
    lens = add_lens_claim(lib, pid)

    ids = {c.claim_id for c in sp.population(lib)}
    assert ids == set(claims.values())
    assert edited in ids and hidden in ids
    assert hidden not in {r[0] for r in lib.conn.execute("SELECT claim_id FROM current_claims")}
    assert not any(i.startswith("rc_") for i in ids)
    assert lens not in ids
    assert ids.isdisjoint(world.claims(old.note_id).values())
    # the edited claim keeps its original type in the population
    method = next(c for c in sp.population(lib) if c.claim_id == edited)
    assert method.claim_type == "method"


def test_marked_claims_leave_the_population_and_unmarked_draws_stay(world, lib):
    for _ in range(2):
        world.noted_paper()
    s = sp.create_sample(lib, n=2, seed=1, filters={"claim_type": "identification_assumption"})
    first, second = s.items
    marks.mark(lib, s.sample_id, first.claim_id, "right")
    ids = {c.claim_id for c in sp.population(lib)}
    assert first.claim_id not in ids and second.claim_id in ids
    later = sp.create_sample(lib, n=50, seed=2, filters={"claim_type": "identification_assumption"})
    assert first.claim_id not in {i.claim_id for i in later.items}
    assert second.claim_id in {i.claim_id for i in later.items}  # drawn before but unmarked: still in
    assert len(later.items) == 3


def test_model_key_fallbacks():
    assert sp.model_key("invented-model-a", "claude") == "invented-model-a"
    assert sp.model_key(None, "claude") == "claude, model not recorded"
    assert sp.model_key("  ", "claude") == "claude, model not recorded"
    assert sp.model_key(None, None) == "model not recorded"
    assert sp.model_key("", " ") == sp.NO_MODEL


def test_model_keys_from_runs_including_a_client_label(world, lib):
    world.noted_paper(model="invented-model-a", client="claude")
    world.noted_paper(model=None, client="claude")
    world.noted_paper(model=None, client=None)
    world.noted_paper(model="codex", client="codex")  # a client recorded as the model: kept as recorded
    keys = {c.model_key for c in sp.population(lib)}
    assert keys == {"invented-model-a", "claude, model not recorded", "model not recorded", "codex"}


# --- the draw (pure) -------------------------------------------------------------------------

def C(cid, ctype, model, paper=None):
    return sp.Claim(cid, paper or f"p_{cid}", "extr_x", ctype, model)


def test_draw_takes_the_least_marked_cell_first():
    pop = [C("a1", "method", "m1"), C("b1", "result", "m1")]
    got = sp.draw(pop, 1, seed=0, marks={("method", "m1"): 3})
    assert [c.claim_id for c in got] == ["b1"]


def test_ties_go_to_the_model_with_fewer_marks():
    pop = [C("a1", "method", "m1"), C("b1", "method", "m2")]
    for seed in range(10):
        got = sp.draw(pop, 1, seed=seed, marks={("result", "m1"): 2})
        assert [c.claim_id for c in got] == ["b1"]


def test_then_the_type_with_fewer_marks():
    pop = [C("a1", "method", "m1"), C("b1", "result", "m1")]
    for seed in range(10):
        got = sp.draw(pop, 1, seed=seed, marks={("method", "m2"): 2})
        assert [c.claim_id for c in got] == ["b1"]


def test_then_a_seeded_choice():
    pop = [C("a1", "method", "m1"), C("b1", "result", "m2")]
    picks = {sp.draw(pop, 1, seed=s)[0].claim_id for s in range(30)}
    assert picks == {"a1", "b1"}
    assert sp.draw(pop, 1, seed=4) == sp.draw(list(reversed(pop)), 1, seed=4)


def test_draws_spread_evenly_and_fill_the_thinnest_cells():
    cells = [("method", "m1"), ("method", "m2"), ("result", "m1"), ("result", "m2")]
    pop = [C(f"{t}_{m}_{i}", t, m) for t, m in cells for i in range(5)]
    got = sp.draw(pop, 8, seed=3)
    assert sorted((c.claim_type, c.model_key) for c in got) == sorted(cells * 2)
    # with earlier marks in one cell, it gets fewer draws
    got = sp.draw(pop, 6, seed=3, marks={("method", "m1"): 2})
    per = {cell: sum(1 for c in got if (c.claim_type, c.model_key) == cell) for cell in cells}
    assert per == {("method", "m1"): 0, ("method", "m2"): 2, ("result", "m1"): 2, ("result", "m2"): 2}


def test_at_most_two_claims_per_paper():
    pop = [C(f"x{i}", f"type{i}", "m1", paper="p_big") for i in range(6)] + [C("y", "type9", "m1", paper="p_small")]
    got = sp.draw(pop, 20, seed=1)
    assert sum(1 for c in got if c.paper_id == "p_big") == 2
    assert len(got) == 3  # stops when no cell can give a claim


def test_stops_at_n():
    pop = [C(f"c{i}", "method", "m1") for i in range(30)]
    assert len(sp.draw(pop, 20, seed=0)) == 20


# --- creating and reading samples ------------------------------------------------------------

def test_same_state_and_seed_give_the_same_sample(world, lib):
    for _ in range(3):
        world.noted_paper()
    a = sp.create_sample(lib, n=6, seed=5)
    expected = sp.draw(sp.population(lib), 6, 5, sp.mark_counts(lib))
    assert [i.claim_id for i in a.items] == [c.claim_id for c in expected]
    b = sp.create_sample(lib, n=6, seed=5)
    assert b.sample_id == a.sample_id and [i.claim_id for i in b.items] == [i.claim_id for i in a.items]
    assert lib.conn.execute("SELECT COUNT(*) FROM spot_check_samples").fetchone()[0] == 1
    others = {tuple(i.claim_id for i in sp.create_sample(lib, n=6, seed=s).items) for s in range(6, 12)}
    assert len(others | {tuple(i.claim_id for i in a.items)}) > 1
    assert a.sample_id.startswith("sc_")
    assert len({i.paper_id for i in a.items}) == 3  # 2 per paper at most


def test_a_seed_is_drawn_and_stored_when_none_is_given(world, lib):
    world.noted_paper()
    s = sp.create_sample(lib, n=2)
    assert isinstance(s.seed, int)
    assert lib.conn.execute("SELECT seed FROM spot_check_samples WHERE sample_id = ?",
                            (s.sample_id,)).fetchone()[0] == s.seed


def test_default_size_is_twenty(world, lib):
    for _ in range(12):
        world.noted_paper()
    s = sp.create_sample(lib, seed=1)
    assert s.size == sp.DEFAULT_SIZE == 20 and len(s.items) == 20
    assert [i.ordinal for i in s.items] == list(range(1, 21))


def test_filters_by_graph_type_and_model(world, lib):
    a = world.noted_paper(model="invented-model-a")
    b = world.noted_paper(model="invented-model-b")
    world.graph("did", [a])
    s = sp.create_sample(lib, n=20, seed=1, filters={"graph": "did"})
    assert {i.paper_id for i in s.items} == {a}
    assert json.loads(lib.conn.execute("SELECT filters_json FROM spot_check_samples WHERE sample_id = ?",
                                       (s.sample_id,)).fetchone()[0]) == {"graph": "did"}
    assert s.filters == {"graph": "did"}
    s = sp.create_sample(lib, n=20, seed=1, filters={"claim_type": "identification_assumption"})
    assert {i.claim_type for i in s.items} == {"identification_assumption"} and len(s.items) == 4
    s = sp.create_sample(lib, n=20, seed=1, filters={"model": "invented-model-b", "graph": None})
    assert {i.paper_id for i in s.items} == {b}
    with pytest.raises(sp.SpotCheckError, match="unknown spot-check filter"):
        sp.create_sample(lib, filters={"lens": "x"})
    with pytest.raises(sp.SpotCheckError, match="no graph"):
        sp.create_sample(lib, filters={"graph": "nowhere"})
    with pytest.raises(sp.SpotCheckError, match="with these filters"):
        sp.create_sample(lib, filters={"model": "no-such-model"})


def test_refusals(world, lib):
    with pytest.raises(sp.SpotCheckError, match="no unmarked agent claims"):
        sp.create_sample(lib)
    world.noted_paper()
    with pytest.raises(sp.SpotCheckError, match="at least one"):
        sp.create_sample(lib, n=0)
    assert sp.sample(lib, "sc_missing") is None


def test_items_show_the_original_claim_and_its_state(world, lib):
    pid = world.paper(title="An invented DiD study", year=2019)
    saved = world.note(pid)
    claims = world.claims(saved.note_id)
    method = claims["method_or_model[0]"]
    edit(lib, pid, method, 0, claim_text="Edited: an event-study design.", claim_type="result")
    trends, serial = claims["assumptions[0]"], claims["assumptions[1]"]
    hide(lib, pid, trends, 1)

    s = sp.create_sample(lib, n=20, seed=2, filters={"claim_type": "method"})
    assumptions = sp.create_sample(lib, n=20, seed=2, filters={"claim_type": "identification_assumption"})
    assert len(sp.create_sample(lib, n=20, seed=2).items) == sp.PER_PAPER_MAX
    by_id = {i.claim_id: i for i in s.items + assumptions.items}
    assert set(by_id) == {method, trends, serial}
    m = by_id[method]
    assert m.claim_type == "method"  # the agent's original type, not the re-type
    assert m.claim_text == "A two-period difference-in-differences design."
    assert m.plain_words == "Compare the change in treated units with the change in controls."
    assert m.normalized_label == "difference in differences"
    assert m.edited and not m.hidden and m.overlay_version == 2
    assert m.quote == "We use a difference-in-differences design with two periods." and m.span_id
    assert m.title == "An invented DiD study" and m.year == 2019
    assert m.model_key == "invented-model-a" and m.extracted_at
    assert m.fingerprint == overlay.fingerprint(lib.conn, method)
    assert m.mark is None and not m.fixed and not m.earlier_extraction
    assert by_id[trends].hidden and not by_id[trends].edited
    assert not by_id[serial].hidden and by_id[serial].quote is None
    assert assumptions.counts == {"right": 0, "partly": 0, "wrong": 0, "unmarked": 2}

    # a re-note after the draw (the overlay is accepted onto it elsewhere): earlier extraction
    with lib.write() as conn:
        overlay.delete_overlays(conn, pid)
    world.note(pid)
    again = sp.sample(lib, s.sample_id)
    assert again is not None and all(i.earlier_extraction for i in again.items)
    assert again.item(method).claim_text == "A two-period difference-in-differences design."


def test_samples_list_newest_first(world, lib):
    world.noted_paper()
    a = sp.create_sample(lib, n=1, seed=1, now="2026-01-01T00:00:00.000000+00:00")
    b = sp.create_sample(lib, n=1, seed=2, now="2026-02-01T00:00:00.000000+00:00")
    assert [s.sample_id for s in sp.samples(lib)] == [b.sample_id, a.sample_id]
