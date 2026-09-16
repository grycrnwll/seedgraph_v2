"""Offline host round trips, durability, policy and coverage boundaries."""
import json
from pathlib import Path

import pytest
from sqlmodel import Session

from seedgraph import run
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.marker_backend import FakeMarkerBackend
from seedgraph.db.adapter import raw_conn
from seedgraph.extraction import agent
from seedgraph.extraction import runner
from seedgraph.project import service
from seedgraph.vocab import AccessClass
from test_phase_4 import _make_doc, _add_doc, _dbconn, _run_extract, good_note_dict, MD


def _one(h, wid, **kwargs):
    batch = agent.prepare_batch(h, [wid], **kwargs)
    packet = agent.next_packet(h, batch["batch_id"])
    assert packet["status"] == "ready", batch
    return batch["batch_id"], packet


def test_roundtrip_replay_replacement_and_unknown_usage():
    h, wid, _ = _make_doc("host_roundtrip")
    batch, packet = _one(h, wid, client="codex")
    assert packet == agent.next_packet(h, batch)
    note = good_note_dict()
    note["main_contribution"]["exact_quote"] = "invented quotation"
    out = agent.submit_packet(h, batch, packet["packet_id"], result=note)
    assert out["status"] == out["paper_status"] == "accepted"
    conn = _dbconn(h)
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM extracted_claims WHERE field_key='main_contribution'").fetchone()[0] == "ambiguous"
    usage = conn.execute("SELECT model_name,input_tokens,output_tokens,estimated_cost,access_mode FROM extraction_runs").fetchone()
    assert tuple(usage) == (None, None, None, None, "subscription_host_reported")
    assert conn.execute("SELECT COUNT(*) FROM llm_usage_events").fetchone()[0] == 0
    assert agent.submit_packet(h, batch, packet["packet_id"], result=note)["status"] == "duplicate"
    assert agent.submit_packet(h, batch, packet["packet_id"], result={})["status"] == "conflict"
    preserved = agent.prepare_batch(h, [wid])
    assert preserved["papers"][0]["status"] == "preserved"
    replace_batch, replacement = _one(h, wid, replace=True)
    other_batch, competing = _one(h, wid, replace=True)
    assert agent.submit_packet(h, replace_batch, replacement["packet_id"], result={})["status"] == "accepted"
    assert agent.submit_packet(h, other_batch, competing["packet_id"], result={})["status"] == "conflict"
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 2
    conn.close()


def test_incomplete_batch_failure_pause_retry_and_resume():
    text = "\n\n".join(f"## Section {i}\n" + ("Text with useful facts. " * 70) for i in range(12))
    h, wid, _ = _make_doc("host_long", markdown=text)
    second, _ = _add_doc(h, "second", MD, doi="10.1/second")
    plan = agent.prepare_batch(h, [wid, second], input_budget_tokens=6000)
    batch = plan["batch_id"]
    packet = agent.next_packet(h, batch)
    assert packet["binding"]["chunk_count"] > 1
    assert packet["estimated_input_tokens"] <= packet["input_budget_tokens"]
    agent.submit_packet(h, batch, packet["packet_id"], result={})
    conn = _dbconn(h)
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
    packet2 = agent.next_packet(h, batch)
    assert packet2["binding"]["chunk_index"] == 1
    agent.submit_packet(h, batch, packet2["packet_id"], failure="quota")
    assert agent.next_packet(h, batch)["status"] == "paused"
    assert agent.next_packet(h, batch, resume=True)["packet_id"] == packet2["packet_id"]
    agent.submit_packet(h, batch, packet2["packet_id"], failure="extraction_failed")
    next_work = agent.next_packet(h, batch)
    assert next_work["binding"]["work_id"] == second
    agent.submit_packet(h, batch, next_work["packet_id"], result={})
    assert agent.next_packet(h, batch)["status"] == "finished"
    retry = agent.next_packet(h, batch, retry_failed=True)
    assert retry["binding"]["chunk_index"] == 1
    assert retry["packet_id"] != packet2["packet_id"]
    while retry["status"] == "ready":
        agent.submit_packet(h, batch, retry["packet_id"], result={})
        retry = agent.next_packet(h, batch)
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 2
    assert agent.batch_status(h, batch)["counts"] == {"accepted": 2}
    conn.close()


def test_stale_source_retained_and_content_gate(monkeypatch):
    h, wid, md = _make_doc("host_stale")
    batch, packet = _one(h, wid)
    with agent.closing(agent.cache_access.open_cache_ro(h.root)) as cache:
        row = agent.cache_access.read_markdown(cache, h.root, md.markdown_id)
    convert_source_file(row.source_file_id, backend=FakeMarkerBackend(markdown=MD + "\nNew version."), force=True, root=h.root)
    outcome = agent.submit_packet(h, batch, packet["packet_id"], result={})
    assert outcome["status"] == "stale"
    directory = run._run_dir(h.slug, batch, h.root)
    assert list((directory / "submissions").glob("*.json"))
    conn = _dbconn(h)
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
    conn.close()
    private, pwork, _ = _make_doc("host_private", access_class=AccessClass.user_supplied_private)
    blocked = agent.prepare_batch(private, [pwork])
    assert blocked["papers"][0]["status"] == "failed"
    assert "PAPER MARKDOWN" not in json.dumps(blocked)
    allowed, private_packet = _one(private, pwork, confirm_external=True)
    assert private_packet["status"] == "ready"


def test_authoritative_policy_tightening_after_prepare(monkeypatch):
    h, wid, md = _make_doc("host_policy")
    batch, packet = _one(h, wid)
    import sqlite3
    from seedgraph.db.connection import cache_db_path
    cache = sqlite3.connect(cache_db_path(h.root))
    cache.execute("UPDATE source_files SET access_class='user_supplied_private'")
    cache.commit()
    cache.close()
    result = agent.next_packet(h, batch)
    assert result["status"] == "paused"
    assert "system_prompt" not in result and "user_prompt" not in result


def test_commit_and_failure_checkpoint_recovery(monkeypatch):
    h, wid, _ = _make_doc("host_recover")
    batch, packet = _one(h, wid)
    checkpoint = agent._checkpoint
    monkeypatch.setattr(agent, "_checkpoint", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        agent.submit_packet(h, batch, packet["packet_id"], result={})
    monkeypatch.setattr(agent, "_checkpoint", checkpoint)
    assert agent.next_packet(h, batch)["status"] == "finished"
    conn = _dbconn(h)
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 1
    conn.close()
    h2, wid2, _ = _make_doc("host_pause_recover")
    batch2, packet2 = _one(h2, wid2)
    monkeypatch.setattr(agent, "_checkpoint", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        agent.submit_packet(h2, batch2, packet2["packet_id"], failure="quota")
    monkeypatch.setattr(agent, "_checkpoint", checkpoint)
    assert agent.next_packet(h2, batch2)["status"] == "paused"
    assert agent.next_packet(h2, batch2, resume=True)["packet_id"] == packet2["packet_id"]


def test_batch_lock_run_ids_restart_and_atomic_writer(monkeypatch):
    h, wid, _ = _make_doc("host_lock")
    batch, packet = _one(h, wid)
    with run.batch_writer(h.slug, batch, root=h.root):
        with pytest.raises(Exception, match="another process"):
            agent.next_packet(h, batch)
    for value in ["../x", "a/b", "a\\b", "D:\\x", ".", "a.json"]:
        with pytest.raises(Exception, match="invalid run"):
            agent.next_packet(h, value)
    assert batch not in run.mark_interrupted_runs(h.slug, root=h.root)
    monkeypatch.setattr(runner, "ensure_span", lambda *a, **k: (_ for _ in ()).throw(ValueError("anchor failed")))
    failed = agent.submit_packet(h, batch, packet["packet_id"], result=good_note_dict())
    assert failed["status"] == "stale"
    conn = _dbconn(h)
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == 0
    conn.close()


def test_same_text_replaced_source_binding_is_stale():
    h, wid, _ = _make_doc("host_replaced_source")
    batch, packet = _one(h, wid)
    conn = _dbconn(h)
    conn.execute("UPDATE work_source_files SET source_file_id='sf_changed',file_hash='changed' WHERE work_id=?", (wid,))
    conn.commit()
    outcome = agent.submit_packet(h, batch, packet["packet_id"], result={})
    assert outcome["status"] == "stale"
    assert conn.execute("SELECT COUNT(*) FROM structured_notes").fetchone()[0] == 0
    conn.close()


def test_live_policy_reload_redaction_and_invalid_binding():
    from seedgraph.config.loader import write_global_config
    h, wid, _ = _make_doc("host_policy_reload", access_class=AccessClass.user_supplied_private)
    write_global_config({"content_policy": {"external_llm_for_private_full_text": True}}, root=h.root)
    batch, packet = _one(h, wid)
    redacted = agent.next_packet(h, batch, redact_private=True)
    assert redacted["status"] == "policy_withheld"
    assert "system_prompt" not in redacted
    invalid = agent.submit_packet(h, batch, packet["packet_id"], result={"binding": {}, "note": {}})
    assert invalid["status"] == "invalid"
    assert agent.next_packet(h, batch)["packet_id"] == packet["packet_id"]
    write_global_config({"content_policy": {"external_llm_for_private_full_text": False}}, root=h.root)
    blocked = agent.next_packet(h, batch)
    assert blocked["status"] == "paused" and "user_prompt" not in blocked


def _probe_batch_lock(root, slug, batch, result):
    try:
        with run.batch_writer(slug, batch, root=root):
            result.put("acquired")
    except Exception as exc:
        result.put(str(exc))


def test_batch_ownership_is_cross_process():
    import multiprocessing
    h, wid, _ = _make_doc("host_process_lock")
    batch, _ = _one(h, wid)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    with run.batch_writer(h.slug, batch, root=h.root):
        child = context.Process(target=_probe_batch_lock, args=(h.root, h.slug, batch, result))
        child.start()
        try:
            assert "another process" in result.get(timeout=20)
            child.join(20)
            assert child.exitcode == 0
        finally:
            if child.is_alive():
                child.terminate()
                child.join(10)
            result.close()
    assert agent.next_packet(h, batch)["status"] == "ready"


def test_external_disable_and_route_access_are_not_confirm_overrides():
    from seedgraph.config.loader import write_global_config
    from seedgraph.llm.routing import assert_external_content_allowed
    from seedgraph.config.models import GlobalConfig
    from seedgraph.errors import ConfigError
    cfg = GlobalConfig()
    cfg.content_policy.allow_external_llm = False
    cfg.content_policy.external_llm_for_private_full_text = True
    with pytest.raises(ConfigError, match="allow_external_llm=false"):
        assert_external_content_allowed("open_access", cfg)
    cfg.content_policy.allow_external_llm = True
    cfg.llm.routes["note_extraction"].allowed_access_classes = ["open_access"]
    with pytest.raises(ConfigError, match="does not permit"):
        assert_external_content_allowed("user_supplied_private", cfg)
    h, wid, _ = _make_doc("host_global_disable")
    write_global_config({"content_policy": {"allow_external_llm": False}}, root=h.root)
    result = agent.prepare_batch(h, [wid], confirm_external=True)
    assert result["papers"][0]["status"] == "failed"
    assert "allow_external_llm=false" in result["papers"][0]["reason"]


@pytest.mark.parametrize("corrupt_bytes", [b"corrupted but valid utf8", b"\xff\xfe"])
def test_corrupt_source_fails_paper_and_continues_batch(corrupt_bytes):
    from seedgraph.cache.store import resolve_uri
    h, first, first_md = _make_doc("host_corrupt")
    second, _ = _add_doc(h, "uncorrupted", MD + "\nDistinct paper", doi="10.1/uncorrupted")
    batch = agent.prepare_batch(h, [first, second])["batch_id"]
    with agent.closing(agent.cache_access.open_cache_ro(h.root)) as cache:
        row = agent.cache_access.read_markdown(cache, h.root, first_md.markdown_id)
    blob = resolve_uri(row.storage_uri, h.root)
    original = blob.read_bytes()
    blob.write_bytes(corrupt_bytes)
    packet = agent.next_packet(h, batch)
    assert packet["status"] == "ready" and packet["binding"]["work_id"] == second
    status = agent.batch_status(h, batch)
    assert status["papers"][0]["status"] == "failed"
    assert "validation" in status["papers"][0]["reason"]
    agent.submit_packet(h, batch, packet["packet_id"], result={})
    assert agent.next_packet(h, batch)["status"] == "finished"
    blob.write_bytes(original)
    retry = agent.next_packet(h, batch, retry_failed=True)
    assert retry["binding"]["work_id"] == first


def test_new_source_without_conversion_never_prepares_old_text():
    from seedgraph.acquisition.bridge import write_bridge
    from seedgraph.cache.ingest import ingest_file
    from seedgraph.vocab import AcquisitionMethod
    h, wid, old_md = _make_doc("host_bridge_transition")
    replacement = h.root / "replacement.pdf"
    replacement.write_bytes(b"%PDF-1.4 replacement source for the same work")
    source = ingest_file(replacement, access_class=AccessClass.open_access,
                         acquisition_method=AcquisitionMethod.open_access_fetch, root=h.root)
    with Session(h.engine) as session:
        write_bridge(session, work_id=wid, source_file_id=source.source_file_id,
                     file_hash=source.file_hash, acquisition_method="open_access_fetch")
        session.commit()
    batch = agent.prepare_batch(h, [wid])
    assert batch["papers"][0]["status"] == "failed"
    assert "convert the current source" in batch["papers"][0]["reason"]
    assert agent.next_packet(h, batch["batch_id"])["status"] == "finished"
    converted = convert_source_file(source.source_file_id,
        backend=FakeMarkerBackend(markdown="# Replacement\nDifferent source text."), root=h.root)
    with Session(h.engine) as session:
        write_bridge(session, work_id=wid, source_file_id=source.source_file_id,
                     file_hash=source.file_hash, markdown_id=converted.markdown_id,
                     markdown_hash=converted.markdown_hash, acquisition_method="open_access_fetch")
        session.commit()
    _, packet = _one(h, wid)
    assert packet["binding"]["source_file_id"] == source.source_file_id
    assert packet["binding"]["markdown_id"] != old_md.markdown_id


def test_deduplicated_text_proves_each_producer_and_repoint_only_tightens_access():
    h, first, first_md = _make_doc("host_shared_text")
    batch, packet = _one(h, first, confirm_external=True)
    actual_source = packet["binding"]["source_file_id"]
    second, second_md = _add_doc(h, "private_shared_text", MD,
        access_class=AccessClass.user_supplied_private, doi="10.1/private-shared")
    assert second_md.markdown_id == first_md.markdown_id
    returned = agent.next_packet(h, batch)
    assert returned["packet_id"] == packet["packet_id"]
    assert returned["access_class"] == "user_supplied_private"
    assert returned["binding"]["source_file_id"] == actual_source
    # The other producer also legitimately prepares the shared content.
    _, second_packet = _one(h, second, confirm_external=True)
    assert second_packet["binding"]["source_file_id"] != actual_source
    assert agent.submit_packet(h, batch, packet["packet_id"], result={})["status"] == "accepted"
    replacement, replacement_packet = _one(h, first, replace=True, confirm_external=True)
    # Latest-version lookup must follow the actual producer, not the repointed row.
    convert_source_file(actual_source, backend=FakeMarkerBackend(markdown=MD + "\nProducer one revision."),
                        force=True, root=h.root)
    assert agent.submit_packet(h, replacement, replacement_packet["packet_id"], result={})["status"] == "stale"


def test_pruned_latest_conversion_does_not_fall_back_and_doctor_reports_missing():
    from seedgraph import doctor_spans
    from seedgraph.db.connection import open_cache_db
    h, wid, md = _make_doc("host_pruned_latest")
    _run_extract(h, wid, note=json.dumps(good_note_dict()))
    with agent.closing(agent.cache_access.open_cache_ro(h.root)) as cache:
        source = agent.cache_access.read_markdown(cache, h.root, md.markdown_id)
    newer = convert_source_file(source.source_file_id,
        backend=FakeMarkerBackend(markdown=MD + "\nLatest unavailable version."), force=True, root=h.root)
    with agent.closing(open_cache_db(h.root)) as cache:
        cache.execute("DELETE FROM markdown_documents WHERE markdown_id=?", (newer.markdown_id,))
        cache.commit()
    with agent.closing(agent.cache_access.open_cache_ro(h.root)) as cache:
        assert agent.cache_access.current_markdown_for_source(cache, source.source_file_id,
            source.source_file_hash) == (newer.markdown_id, newer.markdown_hash)
    failed = agent.prepare_batch(h, [wid], replace=True)
    assert failed["papers"][0]["status"] == "failed"
    assert "newest converted text is unavailable" in failed["papers"][0]["reason"]
    with agent.closing(open_cache_db(h.root)) as cache:
        cache.execute("DELETE FROM markdown_documents WHERE markdown_id=?", (md.markdown_id,))
        cache.commit()
    with agent.closing(agent.cache_access.open_cache_ro(h.root)) as cache, agent.closing(_dbconn(h)) as conn:
        report = doctor_spans.run(conn, cache, h.root)
        assert report.spans_missing > 0 and not report.ok
