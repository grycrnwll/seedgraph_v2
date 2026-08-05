"""Phase 1 (Corpus Cache MVP) acceptance tests — plan §11.

Every unit test uses :class:`FakeMarkerBackend` (no GPU / network / marker-pdf).
The opt-in real-Marker smoke test is marked ``marker`` and deselected by default
(``-m 'not marker'`` in pyproject) AND self-skips on ImportError, so it never runs
in CI. Each test isolates into a throwaway ``$SEEDGRAPH_HOME`` via the autouse
``isolated_home`` fixture (tests/conftest.py), so the env-default cache root is a
fresh per-test directory.
"""

from __future__ import annotations

import json
import re
import sqlite3

import pytest

# Import-clean smoke: the whole cache package (and its public surface) imports.
import seedgraph.cache  # noqa: F401
from seedgraph.cache import (
    CacheError,
    ConversionError,
    FakeMarkerBackend,
    MarkerConfig,
    add_document,
    conversion_fingerprint,
    markdown_path,
    markdown_text,
    resolve_markdown,
    source_access_class,
    validate_markdown,
)
from seedgraph.cache import store
from seedgraph.cache.convert import convert_source_file
from seedgraph.cache.db import init_cache_db
from seedgraph.cache.hashing import sha256_bytes, sha256_file
from seedgraph.cache.ingest import ingest_file
from seedgraph.cache.models import CacheEvent, ConversionRun, MarkdownDocument, SourceFile
from seedgraph.db.connection import open_cache_db
from seedgraph.db.migrations import current_version, latest_version
from seedgraph.vocab import AccessClass, AcquisitionMethod

_GOOD_MD = "# Title\n\n## Section\n\nBody text with several real words here.\n\n## References\n\n[1] x.\n"


# --- helpers ----------------------------------------------------------------

def _write_pdf(tmp_path, data: bytes = b"%PDF-1.4 synthetic content words words words", name="paper.pdf"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _query(sql, params=()):
    conn = open_cache_db()
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _event_types():
    return [r[0] for r in _query("SELECT event_type FROM cache_events ORDER BY cache_event_id")]


# --- tests ------------------------------------------------------------------

def test_hashing(tmp_path):
    """sha256 determinism: identical bytes -> identical hex; chunk boundary is
    irrelevant (streamed sha256_file == sha256_bytes of the same content)."""
    # >1 MiB so the streamed reader crosses its 1 MiB chunk boundary.
    data = b"seedgraph" * 200_000
    assert sha256_bytes(data) == sha256_bytes(bytes(data))  # identical bytes -> identical hex
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert sha256_file(path) == sha256_bytes(data)  # streamed == one-shot (boundary irrelevant)
    assert sha256_bytes(b"a") != sha256_bytes(b"b")


def test_store(tmp_path):
    """Content-addressed flat path derivation; atomic write leaves no temp file;
    a copy survives a different-root relocation (relative storage_uri re-resolves)."""
    root_a = tmp_path / "a"
    cache_a = store.cache_root(root_a)  # == root_a/cache (root is the seedgraph HOME)
    # Flat content-addressed derivation.
    assert store.markdown_blob_path("deadbeef", root_a) == cache_a / "markdown" / "deadbeef.md"
    assert store.pdf_path("cafe", root_a) == cache_a / "pdfs" / "cafe.pdf"

    dest = store.markdown_blob_path("abc123", root_a)
    store.write_bytes_atomic(b"# hi\n", dest)
    assert dest.read_bytes() == b"# hi\n"
    # No temp file left behind on success.
    assert not list(dest.parent.glob("*.tmp"))

    uri = store.relative_uri(dest, root_a)
    assert uri == "markdown/abc123.md"  # POSIX, relative to the cache root

    # Relocate the cache to a different root: the relative URI re-resolves there.
    root_b = tmp_path / "b"
    new_dest = store.markdown_blob_path("abc123", root_b)
    store.copy_file_atomic(dest, new_dest)
    assert store.read_uri(uri, root_b) == b"# hi\n"
    assert store.resolve_uri(uri, root_b) == new_dest


def test_ingest_dedup(tmp_path):
    """Ingest the same bytes twice -> one source_files row; the second emits
    source_file_deduped and writes no second blob."""
    pdf = _write_pdf(tmp_path, b"%PDF dedup target bytes")
    s1 = ingest_file(pdf, access_class=AccessClass.user_supplied_private,
                     acquisition_method=AcquisitionMethod.upload)
    # Copy the same bytes to a different path/name -> same hash.
    pdf2 = _write_pdf(tmp_path, b"%PDF dedup target bytes", name="other.pdf")
    s2 = ingest_file(pdf2, access_class=AccessClass.user_supplied_private,
                     acquisition_method=AcquisitionMethod.upload)

    assert s1.source_file_id == s2.source_file_id
    assert _query("SELECT COUNT(*) FROM source_files")[0][0] == 1
    # Exactly one stored blob.
    assert len(list((store.cache_root() / "pdfs").glob("*.pdf"))) == 1
    assert "source_file_deduped" in _event_types()


def test_access_class_default(tmp_path):
    """upload -> access_class user_supplied_private (fail-closed); open_access_fetch
    (+ explicit open_access) -> open_access."""
    p1 = _write_pdf(tmp_path, b"%PDF upload default", name="up.pdf")
    s1 = ingest_file(p1, access_class=AccessClass.user_supplied_private,
                     acquisition_method=AcquisitionMethod.upload)
    assert s1.access_class == AccessClass.user_supplied_private.value

    p2 = _write_pdf(tmp_path, b"%PDF oa fetch", name="oa.pdf")
    s2 = ingest_file(p2, access_class=AccessClass.open_access,
                     acquisition_method=AcquisitionMethod.open_access_fetch)
    assert s2.access_class == AccessClass.open_access.value

    # Fail-closed: open_access requested but acquisition is upload -> stays private.
    p3 = _write_pdf(tmp_path, b"%PDF oa requested but uploaded", name="up2.pdf")
    s3 = ingest_file(p3, access_class=AccessClass.open_access,
                     acquisition_method=AcquisitionMethod.upload)
    assert s3.access_class == AccessClass.user_supplied_private.value


def test_access_class_reconcile(tmp_path):
    """Ingest open_access then re-ingest same bytes as user_supplied_private -> row
    tightened + access_class_reconciled emitted; reverse order (private then
    open_access) stays user_supplied_private (never downgraded)."""
    # Tighten: open_access -> private.
    pdf = _write_pdf(tmp_path, b"%PDF reconcile A", name="ra.pdf")
    ingest_file(pdf, access_class=AccessClass.open_access,
                acquisition_method=AcquisitionMethod.open_access_fetch)
    same = _write_pdf(tmp_path, b"%PDF reconcile A", name="ra2.pdf")
    s2 = ingest_file(same, access_class=AccessClass.user_supplied_private,
                     acquisition_method=AcquisitionMethod.upload)
    assert s2.access_class == AccessClass.user_supplied_private.value
    assert "access_class_reconciled" in _event_types()

    # Never downgrade: private first, open_access second -> stays private.
    pdf_b = _write_pdf(tmp_path, b"%PDF reconcile B", name="rb.pdf")
    ingest_file(pdf_b, access_class=AccessClass.user_supplied_private,
                acquisition_method=AcquisitionMethod.upload)
    same_b = _write_pdf(tmp_path, b"%PDF reconcile B", name="rb2.pdf")
    s_b2 = ingest_file(same_b, access_class=AccessClass.open_access,
                       acquisition_method=AcquisitionMethod.open_access_fetch)
    assert s_b2.access_class == AccessClass.user_supplied_private.value


def test_conversion_fingerprint():
    """Fingerprint is stable under key reordering; changes when converter_version or
    any MarkerConfig field changes — incl. a paginate_output flip yielding a distinct
    fingerprint (paginated never deduped against non-paginated)."""
    base = MarkerConfig()
    assert conversion_fingerprint("v1", base) == conversion_fingerprint("v1", MarkerConfig())
    assert conversion_fingerprint("v1", base) != conversion_fingerprint("v2", base)
    # Every MarkerConfig field perturbs the fingerprint.
    assert conversion_fingerprint("v1", base) != conversion_fingerprint(
        "v1", MarkerConfig(paginate_output=False))
    assert conversion_fingerprint("v1", base) != conversion_fingerprint(
        "v1", MarkerConfig(use_llm=True))
    assert conversion_fingerprint("v1", base) != conversion_fingerprint(
        "v1", MarkerConfig(force_ocr=True))
    assert conversion_fingerprint("v1", base) != conversion_fingerprint(
        "v1", MarkerConfig(output_format="json"))
    assert conversion_fingerprint("v1", base) != conversion_fingerprint(
        "v1", MarkerConfig(llm_service="anthropic"))


def test_validate_markdown():
    """empty/whitespace/garbage -> ok=False; good markdown w/ title+sections+refs ->
    ok=True with no warnings; markdown missing a references section -> ok=True with a
    references warning."""
    assert validate_markdown("").ok is False
    assert validate_markdown("   \n\t  ").ok is False
    assert validate_markdown("@#$%^&*()_+=~`<>{}[]|\\;:").ok is False

    good = validate_markdown(_GOOD_MD)
    assert good.ok is True
    assert good.warnings == []

    no_refs = validate_markdown("# Title\n\n## Section\n\nBody text with real words.\n")
    assert no_refs.ok is True
    assert any("references" in w.lower() for w in no_refs.warnings)


def test_validate_markdown_math_dense_not_false_rejected():
    """A symbol/math-dense paper (few 2+-letter word runs) trips the LEGACY word-like
    -only gate but the math-aware gate accepts it; a truly-empty/garbled one still
    fails."""
    math_md = (
        "$$\\sum_{i=1}^{n} x_i^2 \\leq \\left( \\sum_{i=1}^{n} x_i \\right)^2$$\n\n"
        "$$1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36$$\n\n"
        "$$2^{10} = 1024, \\quad 3^5 = 243$$\n\n"
        "$a_1 + a_2 = b$, $c^2 = d^2 + e^2$, $1/2 + 3/4 = 5/4$.\n\n"
        "$$\\int_0^1 \\, dx = 1$$; $6 \\cdot 7 = 42$; $8 - 9 = -1$; $x \\to \\infty$.\n"
    )
    stripped = math_md.strip()
    legacy_wordlike = sum(len(m) for m in re.findall(r"[A-Za-z]{2,}", stripped))
    legacy_ratio = legacy_wordlike / len(stripped)
    assert legacy_ratio < 0.20  # WOULD trip the old word-like-only 0.20 garbage floor
    assert validate_markdown(math_md).ok is True  # math-aware gate accepts it

    # Genuinely empty / garbled output is still rejected.
    assert validate_markdown("").ok is False
    assert validate_markdown("@#$%^&*()_+=~`<>{}[]|;:").ok is False


def test_validate_markdown_ratio_configurable(monkeypatch):
    """The garbage-gate floor is configurable: explicit param > SEEDGRAPH_GARBAGE_RATIO
    env > default 0.20."""
    garbage = "@#$%^&*()_+=~`<>{}[]|;:"  # no word-like, no math -> content ratio 0
    assert validate_markdown(garbage).ok is False  # default 0.20 floor rejects

    # An env floor of 0.0 is the escape hatch: nothing above min-length is rejected.
    monkeypatch.setenv("SEEDGRAPH_GARBAGE_RATIO", "0.0")
    assert validate_markdown(garbage).ok is True
    # An explicit param overrides the env.
    assert validate_markdown(garbage, min_content_ratio=0.5).ok is False
    monkeypatch.delenv("SEEDGRAPH_GARBAGE_RATIO", raising=False)

    # A high floor rejects even ordinary prose.
    prose = "# Title\n\nplain english body text here.\n"
    assert validate_markdown(prose).ok is True
    assert validate_markdown(prose, min_content_ratio=0.99).ok is False


def test_convert_failed_on_empty(tmp_path):
    """FakeMarkerBackend returning empty markdown -> run_status='failed', error set,
    NO markdown_documents row, NO .md written, conversion_failed emitted."""
    pdf = _write_pdf(tmp_path, b"%PDF will produce empty markdown")
    src = ingest_file(pdf, access_class=AccessClass.user_supplied_private,
                      acquisition_method=AcquisitionMethod.upload)
    with pytest.raises(ConversionError):
        convert_source_file(src.source_file_id, backend=FakeMarkerBackend(markdown=""))

    run = _query("SELECT run_status, error FROM conversion_runs")[0]
    assert run[0] == "failed"
    assert run[1] and "validation failed" in run[1]
    assert _query("SELECT COUNT(*) FROM markdown_documents")[0][0] == 0
    assert not list((store.cache_root() / "markdown").glob("*.md"))
    assert "conversion_failed" in _event_types()


def test_convert_dedup(tmp_path):
    """Convert a file twice with the same config -> FakeMarkerBackend.call_count == 1,
    same markdown_id, conversion_reused event on the 2nd call (milestone core)."""
    pdf = _write_pdf(tmp_path, b"%PDF convert dedup")
    src = ingest_file(pdf, access_class=AccessClass.user_supplied_private,
                      acquisition_method=AcquisitionMethod.upload)
    fake = FakeMarkerBackend()
    md1 = convert_source_file(src.source_file_id, backend=fake)
    md2 = convert_source_file(src.source_file_id, backend=fake)

    assert fake.call_count == 1  # Marker ran zero additional times
    assert md1.markdown_id == md2.markdown_id
    assert _event_types().count("conversion_reused") == 1
    assert _query("SELECT COUNT(*) FROM conversion_runs")[0][0] == 1


def test_convert_version_bump(tmp_path):
    """Bump the backend version -> new conversion_run, backend invoked again, prior
    run retained (append-only)."""
    pdf = _write_pdf(tmp_path, b"%PDF version bump")
    src = ingest_file(pdf, access_class=AccessClass.user_supplied_private,
                      acquisition_method=AcquisitionMethod.upload)
    convert_source_file(src.source_file_id, backend=FakeMarkerBackend(version="fake-0.0.0"))
    fake_new = FakeMarkerBackend(version="fake-1.0.0")
    convert_source_file(src.source_file_id, backend=fake_new)

    assert fake_new.call_count == 1  # the bumped version was NOT a cache hit
    versions = {r[0] for r in _query("SELECT converter_version FROM conversion_runs")}
    assert versions == {"fake-0.0.0", "fake-1.0.0"}
    assert _query("SELECT COUNT(*) FROM conversion_runs")[0][0] == 2  # prior retained


def test_force(tmp_path):
    """--force mints a SECOND success run with the same (source_file_hash,
    conversion_fingerprint) — no IntegrityError (no unique index); both rows coexist;
    resolve/dedup picks the latest by created_at (must-fix)."""
    pdf = _write_pdf(tmp_path, b"%PDF force run")
    src = ingest_file(pdf, access_class=AccessClass.user_supplied_private,
                      acquisition_method=AcquisitionMethod.upload)
    fake = FakeMarkerBackend()
    md1 = convert_source_file(src.source_file_id, backend=fake)
    md2 = convert_source_file(src.source_file_id, backend=fake, force=True)  # must NOT raise

    assert fake.call_count == 2  # --force always re-ran
    assert md1.markdown_id == md2.markdown_id  # identical bytes -> same content-addressed id
    runs = _query(
        "SELECT conversion_fingerprint FROM conversion_runs WHERE run_status='success'"
    )
    assert len(runs) == 2  # both success rows coexist (append-only)
    assert len({r[0] for r in runs}) == 1  # same fingerprint
    # A subsequent non-force convert resolves to the latest success by recency -> reused.
    convert_source_file(src.source_file_id, backend=fake)
    assert fake.call_count == 2  # not re-run


def test_manifest(tmp_path):
    """manifest.json exists, round-trips, and its hashes are sha256:-prefixed and
    match the conversion/markdown rows."""
    pdf = _write_pdf(tmp_path, b"%PDF manifest")
    md = add_document(pdf, backend=FakeMarkerBackend())
    run_id = md.conversion_run_id
    manifest_path = store.marker_dir(run_id) / "manifest.json"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))  # round-trips
    assert manifest["markdown_hash"] == f"sha256:{md.markdown_hash}"
    src_hash = _query(
        "SELECT source_file_hash FROM conversion_runs WHERE conversion_run_id=?", (run_id,)
    )[0][0]
    assert manifest["source_file_hash"] == f"sha256:{src_hash}"
    assert manifest["converter"]["version"] == "fake-0.0.0"
    assert manifest["config"]["paginate_output"] is True


def test_fk_and_pragmas(tmp_path):
    """foreign_keys=ON; inserting a markdown_documents row with a bogus
    conversion_run_id fails the FK."""
    init_cache_db()
    conn = open_cache_db()
    try:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO markdown_documents (markdown_id, conversion_run_id, source_file_id, "
                "markdown_hash, storage_uri, conversion_status, byte_size, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("md_x", "conv_bogus", "sf_bogus", "h", "markdown/x.md", "success", 1, "2026"),
            )
            conn.commit()
    finally:
        conn.close()


def test_content_addressed_ids(tmp_path):
    """Identical file bytes -> source_file_id == 'sf_' + file_hash; identical markdown
    bytes -> markdown_id == 'md_' + markdown_hash; re-ingest/re-convert yields the SAME
    ids. conversion_run_id stays an opaque 'conv_' uuid (D1)."""
    pdf = _write_pdf(tmp_path, b"%PDF content addressed")
    fake = FakeMarkerBackend()
    md = add_document(pdf, backend=fake)
    assert md.source_file_id == "sf_" + sha256_file(pdf)
    assert md.markdown_id == "md_" + md.markdown_hash
    assert md.conversion_run_id.startswith("conv_")
    assert len(md.conversion_run_id) == len("conv_") + 32  # uuid4 hex

    # Re-ingest / re-convert identical bytes -> identical content-addressed ids.
    md2 = add_document(pdf, backend=fake)
    assert md2.source_file_id == md.source_file_id
    assert md2.markdown_id == md.markdown_id


def test_migrations_are_source_of_truth(tmp_path):
    """init_cache_db applies the numbered db/schema/cache/ .sql files; the resulting
    schema matches SQLModel.metadata (tables/columns); create_all is never the
    authoring path; an applied-version check is readable for doctor (D6)."""
    init_cache_db()
    conn = open_cache_db()
    try:
        # The migrate runner — not create_all — populated schema_migrations.
        assert current_version(conn) == latest_version("cache")
        applied = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        assert {1, 2} <= applied

        # ORM mapping mirrors the migrated schema, column-for-column.
        for cls in (SourceFile, ConversionRun, MarkdownDocument, CacheEvent):
            orm_cols = {c.name for c in cls.__table__.columns}
            db_cols = {r[1] for r in conn.execute(f"PRAGMA table_info({cls.__tablename__})")}
            assert orm_cols == db_cols, f"{cls.__tablename__}: {orm_cols ^ db_cols}"
    finally:
        conn.close()


def test_html_ingest_then_convert_rejected(tmp_path):
    """HTML ingests/dedups normally; convert raises a clear 'deferred' error."""
    html = tmp_path / "page.html"
    html.write_bytes(b"<html><body>some words here</body></html>")
    src = ingest_file(html, access_class=AccessClass.user_supplied_private,
                      acquisition_method=AcquisitionMethod.upload)
    assert src.file_type == "html"
    # Re-ingest dedups.
    html2 = tmp_path / "page2.html"
    html2.write_bytes(b"<html><body>some words here</body></html>")
    src2 = ingest_file(html2, access_class=AccessClass.user_supplied_private,
                       acquisition_method=AcquisitionMethod.upload)
    assert src2.source_file_id == src.source_file_id

    with pytest.raises(CacheError) as excinfo:
        convert_source_file(src.source_file_id, backend=FakeMarkerBackend())
    message = str(excinfo.value).lower()
    assert "html" in message and "deferred" in message


def test_read_apis(tmp_path):
    """resolve_markdown by hash and by id; source_access_class walks md -> source_file;
    markdown_text/markdown_path round-trip the stored blob."""
    pdf = _write_pdf(tmp_path, b"%PDF read apis")
    md = add_document(pdf, backend=FakeMarkerBackend())

    by_id = resolve_markdown(markdown_id=md.markdown_id)
    by_hash = resolve_markdown(markdown_hash=md.markdown_hash)
    assert by_id is not None and by_hash is not None
    assert by_id.markdown_id == by_hash.markdown_id == md.markdown_id
    assert resolve_markdown(markdown_id="md_does_not_exist") is None

    # Exactly one selector kwarg required.
    with pytest.raises(ValueError):
        resolve_markdown()

    assert markdown_text(by_id).startswith("# Title")
    assert markdown_path(by_id) == store.markdown_blob_path(md.markdown_hash)
    assert source_access_class(md.markdown_id) == AccessClass.user_supplied_private


def test_milestone_no_reprocess(tmp_path):
    """Milestone (doc 10 §4 + §5): add_document(pdf) records markdown_id_1 with
    call_count==1; add_document(same bytes) creates no new source_files row,
    call_count stays 1, and resolves to markdown_id_1; manifest.json traces
    markdown_hash -> source_file_hash + converter_version + config."""
    pdf = _write_pdf(tmp_path, b"%PDF milestone end to end content words")
    fake = FakeMarkerBackend()

    md1 = add_document(pdf, backend=fake)
    assert fake.call_count == 1

    # Same bytes again -> no new source_files row, Marker not re-run, same markdown_id.
    md2 = add_document(pdf, backend=fake)
    assert fake.call_count == 1
    assert md2.markdown_id == md1.markdown_id
    assert _query("SELECT COUNT(*) FROM source_files")[0][0] == 1

    # Phase-3 traceability: manifest traces markdown -> source + converter + config.
    manifest = json.loads((store.marker_dir(md1.conversion_run_id) / "manifest.json").read_text())
    src_hash = _query(
        "SELECT source_file_hash FROM conversion_runs WHERE conversion_run_id=?",
        (md1.conversion_run_id,),
    )[0][0]
    assert manifest["markdown_hash"] == f"sha256:{md1.markdown_hash}"
    assert manifest["source_file_hash"] == f"sha256:{src_hash}"
    assert manifest["converter"]["version"] == "fake-0.0.0"
    assert "config" in manifest and "paginate_output" in manifest["config"]


@pytest.mark.marker
def test_marker_smoke():
    """Opt-in/slow: confirm the real LocalMarkerBackend adapter wires up against the
    installed marker-pdf (version capture). Deselected by default (-m 'not marker')
    AND self-skips when marker-pdf is not installed; never runs in CI."""
    pytest.importorskip("marker", reason="marker-pdf optional extra not installed")
    from seedgraph.cache.marker_backend import LocalMarkerBackend

    version = LocalMarkerBackend().version
    assert isinstance(version, str) and version
