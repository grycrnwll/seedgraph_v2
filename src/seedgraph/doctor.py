"""``seedgraph doctor`` — validate the whole foundation with actionable messages.

Bootstraps + migrates both scopes (cache always; project when ``--project`` is
given and the slug is valid), then runs the doc 13 §13 checks. Any error-severity
failure makes the command exit non-zero; warnings (e.g. Ollama not reachable with
``--probe-ollama``) do not. Scholarly-API keys are listed by environment variable name.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


from . import paths
from .config.loader import load_global_config, load_project_config
from .db.bootstrap import ensure_cache_db, ensure_project_db
from .db.connection import cache_db_path, project_db_path
from .db.migrations import current_version, latest_version
from .errors import SeedgraphError, ValidationError

_SQLITE_BASELINE = (3, 35, 0)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    severity: str = "error"  # "error" | "warning"


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".") if part.isdigit())


def _fts5_available() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def _check_sqlite_and_fts5() -> list[CheckResult]:
    results: list[CheckResult] = []
    version_ok = _version_tuple(sqlite3.sqlite_version) >= _SQLITE_BASELINE
    results.append(
        CheckResult(
            "sqlite_version",
            version_ok,
            f"sqlite {sqlite3.sqlite_version} "
            f"(baseline {'.'.join(map(str, _SQLITE_BASELINE))})",
        )
    )
    fts5 = _fts5_available()
    results.append(
        CheckResult(
            "fts5_available",
            fts5,
            "FTS5 compiled in" if fts5 else "FTS5 NOT compiled in — required by later phases",
        )
    )
    return results


def _check_home_writable(root: Path | str | None) -> CheckResult:
    home = paths.resolve_home(root)
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".doctor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return CheckResult("home_writable", True, f"{home} is writable")
    except OSError as exc:
        return CheckResult("home_writable", False, f"{home} is not writable: {exc}")


def _check_schema_version(scope: str, db_path: Path) -> CheckResult:
    name = f"schema_version_{scope}"
    expected = latest_version(scope)  # type: ignore[arg-type]
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            applied = current_version(conn)
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"{scope}: cannot read schema version: {exc}")
    if applied == expected:
        return CheckResult(name, True, f"{scope} schema at version {applied} (current)")
    return CheckResult(
        name,
        False,
        f"{scope} schema version mismatch: applied={applied}, expected={expected}. "
        "v3 does not upgrade v2 databases in place: open this one with the v2 `seedgraph` "
        "install that wrote it, or move the v2 home into v3 with `seedgraph-v3 migrate run`.",
    )


def _check_foreign_keys(scope: str, db_path: Path, opener) -> CheckResult:
    name = f"foreign_keys_{scope}"
    try:
        conn = opener()
        try:
            value = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"{scope}: cannot read PRAGMA foreign_keys: {exc}")
    ok = value == 1
    return CheckResult(
        name,
        ok,
        f"{scope}: foreign_keys={'ON' if ok else 'OFF'}",
    )


def _check_foreign_key_integrity(scope: str, opener) -> CheckResult:
    """``PRAGMA foreign_key_check`` over a scope (phase_5b). Reports violating rows.
    Cheap real check — empty result on a clean DB."""
    name = f"foreign_key_check_{scope}"
    try:
        conn = opener()
        try:
            violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"{scope}: foreign_key_check failed: {exc}")
    if not violations:
        return CheckResult(name, True, f"{scope}: no foreign-key violations")
    return CheckResult(
        name,
        False,
        f"{scope}: {len(violations)} foreign-key violation(s): {violations[:5]}",
    )


# --- Per-phase reconcile check stubs (additive; no-op presence checks) -------
# Each later phase requested a doctor sub-check in its needs_wiring. They are
# registered here as no-op PASS stubs so the command surface is complete and the
# gate stays green until the owning phase implements the real reconcile.

def _stub_reconcile_checks(proceed_project: bool) -> list[CheckResult]:
    stubs = [
        ("span_section_fts_reconcile",
         "phase_3 span/section/FTS reconcile not yet wired (no-op stub)"),
        ("cross_db_bridge_reconcile",
         "phase_5b work_source_files bridge reconcile not yet wired (no-op stub)"),
    ]
    # Project-only checks degrade to a benign note when no --project is given.
    return [CheckResult(name, True, detail) for name, detail in stubs]


def semantic_graph_findings(conn: sqlite3.Connection) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Scan the phase_7 overlay: ``(dangling_edges, orphaned_concepts)``.

    A dangling edge is a ``project_graph_edges`` row whose ``(node_type, node_id)``
    endpoint does not resolve to a real row (the polymorphic-FK substitute, doc 07
    §4.2). An orphaned concept is a ``concepts`` row with zero ``claim_concepts``
    mentions (a concept must always be evidence-backed, doc 07 §1). Both are
    project.db-local reads — no cache.db.
    """
    home = {"Work": ("works", "work_id"), "Concept": ("concepts", "concept_id"),
            "Claim": ("extracted_claims", "claim_id"),
            "EvidenceSpan": ("evidence_spans", "span_id")}

    def _exists(table: str, pk: str, value: str) -> bool:
        return conn.execute(
            f"SELECT 1 FROM {table} WHERE {pk}=? LIMIT 1", (value,)
        ).fetchone() is not None

    dangling: list[tuple[str, str, str]] = []
    try:
        edges = conn.execute(
            "SELECT edge_id, source_node_type, source_node_id, target_node_type, "
            "target_node_id FROM project_graph_edges"
        ).fetchall()
    except sqlite3.OperationalError:
        return [], []  # overlay not migrated yet
    for edge_id, st, sid, tt, tid in edges:
        for ntype, nid in ((st, sid), (tt, tid)):
            entry = home.get(ntype)
            if entry is None or not _exists(entry[0], entry[1], nid):
                dangling.append((edge_id, ntype, nid))

    orphans = [
        r[0]
        for r in conn.execute(
            "SELECT c.concept_id FROM concepts c "
            "LEFT JOIN claim_concepts cc ON cc.concept_id = c.concept_id "
            "WHERE cc.concept_id IS NULL ORDER BY c.concept_id"
        ).fetchall()
    ]
    return dangling, orphans


def _semantic_graph_reconcile(root: Path | str | None, slug: str) -> list[CheckResult]:
    """The phase_7 dangling-edge + orphaned-concept scan, as a doctor check.

    PASS when the overlay is clean. A dangling polymorphic endpoint is an ERROR
    (integrity violation — the insert path should have rejected it); an orphaned
    concept is a WARNING (run ``concepts build`` to recompute / prune)."""
    try:
        conn = _open_project(slug, root)
        try:
            dangling, orphans = semantic_graph_findings(conn)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - a reconcile error is a check failure
        return [CheckResult("semantic_graph_reconcile", False, f"scan failed: {exc}")]

    results: list[CheckResult] = []
    results.append(
        CheckResult(
            "semantic_graph_dangling_edges",
            not dangling,
            "no dangling project_graph_edges endpoints"
            if not dangling
            else f"{len(dangling)} dangling endpoint(s): {dangling[:5]}",
        )
    )
    results.append(
        CheckResult(
            "semantic_graph_orphaned_concepts",
            True,
            "no orphaned concepts"
            if not orphans
            else f"{len(orphans)} orphaned concept(s) with no claim mention "
            f"(re-run `concepts build` to prune): {orphans[:5]}",
            severity="warning",
        )
    )
    return results


def reference_entries_findings(engine, cache_root: Path | str | None) -> list[tuple[str, str]]:
    """Classify every ``reference_entries`` soft-ref ``ok | stale | missing`` (§4.5; phase_3b).

    Routes EACH distinct ``(citing_work_id, markdown_hash)`` through phase_5b's
    shared ``ok/stale/missing`` content-hash primitive
    (``doctor_reconcile.bridge_row_status``), anchored on the citing work's
    ``work_source_files`` lineage — never re-implementing ATTACH/staleness logic
    (decision r2-10). Returns ``[(citing_work_id, status)]``.
    """
    from sqlmodel import Session, select

    from .acquisition.doctor_reconcile import bridge_row_status
    from .db.project_models import ReferenceEntry, WorkSourceFile

    with Session(engine) as session:
        pairs = {
            (cw, mh)
            for cw, mh in session.exec(
                select(ReferenceEntry.citing_work_id, ReferenceEntry.markdown_hash)
            ).all()
        }
        bridges = {b.work_id: b for b in session.exec(select(WorkSourceFile)).all()}

    findings: list[tuple[str, str]] = []
    for citing_work_id, markdown_hash in pairs:
        bridge = bridges.get(citing_work_id)
        if bridge is None:
            findings.append((citing_work_id, "missing"))
            continue
        status = bridge_row_status(
            cache_root, file_hash=bridge.file_hash, markdown_hash=markdown_hash
        )
        findings.append((citing_work_id, status))
    return findings


def _reference_entries_reconcile(root: Path | str | None, slug: str) -> list[CheckResult]:
    """The phase_3b ``reference_entries`` soft-ref reconcile, as a doctor check.

    PASS when every soft-ref is current; a stale anchor (reconversion) is a warning;
    a missing anchor (cache GC'd / no bridge) is a warning so a local-first cache
    prune does not fail the gate (the rows fall back to ``metadata_only`` on read).
    """
    try:
        from .project.service import open_project

        h = open_project(slug, root=root)
        findings = reference_entries_findings(h.engine, root)
    except Exception as exc:  # noqa: BLE001 - a reconcile error is a check failure
        return [CheckResult("reference_entries_reconcile", False, f"reconcile failed: {exc}")]

    ok = sum(1 for _w, s in findings if s == "ok")
    stale = [w for w, s in findings if s == "stale"]
    missing = [w for w, s in findings if s == "missing"]
    if stale or missing:
        return [
            CheckResult(
                "reference_entries_reconcile",
                True,
                f"{ok} ok, {len(stale)} stale, {len(missing)} missing "
                f"(re-run `cite parse --force` to re-link; reconversion is expected staleness)",
                severity="warning",
            )
        ]
    return [
        CheckResult(
            "reference_entries_reconcile", True,
            f"{ok} reference_entries soft-ref(s) ok ({len(findings)} citing-work anchors)",
        )
    ]


def _check_config(root: Path | str | None, slug: str | None) -> CheckResult:
    try:
        if slug is not None:
            load_project_config(slug, root)
            label = f"project '{slug}'"
        else:
            load_global_config(root)
            label = "global"
        return CheckResult("config", True, f"{label} config parses")
    except SeedgraphError as exc:
        return CheckResult("config", False, f"config invalid: {exc}")


def _check_scholarly_keys() -> CheckResult:
    """Which scholarly-API environment variables are set (names only, never values).
    Keys come from the environment only; every one is optional, so this never fails."""
    from .providers import SCHOLARLY_ENV_VARS, env_key

    parts = [f"{name}={'set' if env_key(name) else 'unset'}" for name in SCHOLARLY_ENV_VARS]
    return CheckResult("scholarly_api_keys", True,
                       "from environment variables only: " + ", ".join(parts))


def _check_ollama() -> CheckResult:
    """Opt-in (``--probe-ollama``) reachability of the local Ollama server used by
    ``work ollama``: ``GET /api/tags``, no model call. Unreachable is a warning (Ollama
    is optional)."""
    from .llm.ollama import ENV_URL, check_reachable

    r = check_reachable(timeout=1.0)
    if r.ok:
        models = ", ".join(r.models[:8]) or "no models pulled"
        return CheckResult("ollama", True, f"Ollama reachable at {r.base_url} ({models})")
    return CheckResult("ollama", True,
                       f"Ollama not reachable at {r.base_url} ({r.error}); `work ollama` needs it "
                       f"(set {ENV_URL} or OLLAMA_HOST for another address)", severity="warning")


def _check_gpu_probe() -> CheckResult:
    """Informational GPU probe: GPUs visible to ``nvidia-smi`` and the converter version.

    The app environment has no torch (PDF conversion runs in the separate converter
    environment), so GPUs are counted with ``nvidia-smi -L``. ALWAYS ``ok=True`` --
    the probe informs, it never gates anything (Build E design 8). The recommended
    marker batch profile stays hard-coded ``conservative``: a misleading probe must
    degrade to *slow*, never to an OOM-prone tier.
    """
    from .conversion import pool as conversion_pool

    try:
        gpus = str(conversion_pool._nvidia_smi_gpu_count())
    except Exception:  # noqa: BLE001 - informational only
        gpus = "unknown"
    version = conversion_pool.locked_marker_version() or "unknown"
    return CheckResult(
        "gpu_probe",
        True,
        f"gpus={gpus} (nvidia-smi); converter: marker-pdf {version} (converter/uv.lock); "
        f"one warm converter per GPU; recommended batch profile: conservative "
        f"(hard-coded — probe never gates)",
    )


def collect_checks(
    root: Path | str | None = None,
    slug: str | None = None,
    *,
    probe_ollama: bool = False,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_check_sqlite_and_fts5())
    results.append(_check_home_writable(root))

    proceed_project = False
    if slug is not None:
        try:
            paths.validate_slug(slug)
            results.append(CheckResult("project_slug", True, f"slug '{slug}' is valid"))
            proceed_project = True
        except ValidationError as exc:
            results.append(CheckResult("project_slug", False, str(exc)))

    # Bootstrap (create + migrate). Wrapped so a tampered/broken DB yields a failed
    # check rather than a traceback.
    try:
        ensure_cache_db(root)
        results.append(CheckResult("cache_db_reachable", True, f"{cache_db_path(root)} ready"))
    except (sqlite3.Error, OSError, SeedgraphError) as exc:
        results.append(CheckResult("cache_db_reachable", False, f"cache.db bootstrap failed: {exc}"))

    if proceed_project:
        try:
            ensure_project_db(slug, root)  # type: ignore[arg-type]
            results.append(
                CheckResult("project_db_reachable", True, f"{project_db_path(slug, root)} ready")
            )
        except (sqlite3.Error, OSError, SeedgraphError) as exc:
            results.append(
                CheckResult("project_db_reachable", False, f"project.db bootstrap failed: {exc}")
            )

    results.append(_check_schema_version("cache", cache_db_path(root)))
    results.append(
        _check_foreign_keys("cache", cache_db_path(root), lambda: _open_cache(root))
    )
    results.append(_check_foreign_key_integrity("cache", lambda: _open_cache(root)))
    if proceed_project:
        results.append(_check_schema_version("project", project_db_path(slug, root)))
        results.append(
            _check_foreign_keys(
                "project", project_db_path(slug, root), lambda: _open_project(slug, root)
            )
        )
        results.append(
            _check_foreign_key_integrity("project", lambda: _open_project(slug, root))
        )

    results.append(_check_config(root, slug))
    results.append(_check_scholarly_keys())
    if probe_ollama:
        results.append(_check_ollama())
    # Build E ch5a: informational GPU probe — reports, never gates (design 8).
    results.append(_check_gpu_probe())
    # Cross-DB reference scan hook — no-op in Phase 0, structure in place for later phases.
    results.append(CheckResult("cross_db_reference_scan", True, "no cross-DB references yet (no-op)"))
    # Per-phase reconcile sub-checks (additive no-op stubs until each phase lands).
    results.extend(_stub_reconcile_checks(proceed_project))
    # phase_3b: real reference_entries soft-ref reconcile (replaces its stub) when a
    # valid project is in scope; routes through phase_5b's shared ok/stale/missing
    # primitive (decision r2-10).
    if proceed_project:
        results.extend(_reference_entries_reconcile(root, slug))  # type: ignore[arg-type]
        # phase_7: real dangling-edge + orphaned-concept scan (replaces its stub).
        results.extend(_semantic_graph_reconcile(root, slug))  # type: ignore[arg-type]
    return results


def _open_cache(root):
    from .db.connection import open_cache_db

    return open_cache_db(root)


def _open_project(slug, root):
    from .db.connection import open_project_db

    return open_project_db(slug, root)


def has_failure(results: list[CheckResult]) -> bool:
    return any((not r.ok) and r.severity == "error" for r in results)


def render(results: list[CheckResult]) -> str:
    lines = []
    for r in results:
        if r.ok:
            mark = "PASS"
        elif r.severity == "warning":
            mark = "WARN"
        else:
            mark = "FAIL"
        lines.append(f"[{mark}] {r.name}: {r.detail}")
    lines.append("")
    lines.append("RESULT: " + ("FAIL" if has_failure(results) else "OK"))
    return "\n".join(lines)


def render_json(results: list[CheckResult]) -> str:
    """One machine-readable JSON object over the SAME results ``render`` shows.

    Shape contract (Build F ch6): ``{"checks": [{name, ok, severity, detail}...],
    "ok": bool}`` — the per-check KEYS are pinned, the check LIST is not, so
    later builds' additive checks flow through order-free. ``ok`` mirrors the
    exit-code rule (:func:`has_failure`): warnings never fail it.
    """
    return json.dumps(
        {
            "checks": [
                {"name": r.name, "ok": r.ok, "severity": r.severity, "detail": r.detail}
                for r in results
            ],
            "ok": not has_failure(results),
        },
        sort_keys=True,
    )
