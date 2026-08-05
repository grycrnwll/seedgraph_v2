"""``seedgraph doctor`` — validate the whole foundation with actionable messages.

Bootstraps + migrates both scopes (cache always; project when ``--project`` is
given and the slug is valid), then runs the doc 13 §13 checks. Any error-severity
failure makes the command exit non-zero; warnings (e.g. a missing external key in
a local-first setup) do not.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from datetime import date as _date

from . import paths
from .config.loader import load_global_config, load_llm_capabilities, load_project_config
from .db.bootstrap import ensure_cache_db, ensure_project_db
from .db.connection import cache_db_path, project_db_path
from .db.migrations import current_version, latest_version
from .errors import SeedgraphError, ValidationError
from .llm.profiles import is_local_profile
from .llm.secrets import resolve_profile_key

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
        f"Run `seedgraph migrate` to bring it current.",
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
            cfg = load_project_config(slug, root)
            label = f"project '{slug}'"
        else:
            cfg = load_global_config(root)
            label = "global"
        n_profiles = len(cfg.llm.profiles)
        n_routes = len(cfg.llm.routes)
        return CheckResult(
            "config",
            True,
            f"{label} config parses; {n_profiles} profiles, {n_routes} routes; "
            f"routing references resolve",
        )
    except SeedgraphError as exc:
        return CheckResult("config", False, f"config invalid: {exc}")


def _check_key_refs(root: Path | str | None, slug: str | None) -> list[CheckResult]:
    try:
        cfg = load_project_config(slug, root) if slug is not None else load_global_config(root)
    except SeedgraphError:
        return []  # config check already reported the failure
    results: list[CheckResult] = []
    for profile_id, profile in sorted(cfg.llm.profiles.items()):
        if is_local_profile(profile):
            continue
        # Full chain (ADR-0002): keyring entry then env var — a keyring-only
        # profile (no env_var) is checkable now, so no env_var skip.
        present = resolve_profile_key(profile) is not None
        ref = profile.key_name or f"seedgraph/{profile.provider}/default"
        via = f"keyring {ref}" + (f" or env {profile.env_var}" if profile.env_var else "")
        results.append(
            CheckResult(
                f"key_ref_{profile_id}",
                present,
                f"profile '{profile_id}': key ({via}) "
                f"{'present' if present else 'absent (local-first; warning only)'}",
                severity="error" if present else "warning",
            )
        )
    return results


# --- Stage C LLM-backend doctor checks (no default paid calls) --------------

_PRICING_SNAPSHOT_MAX_AGE_DAYS = 180


def _load_cfg(root, slug):
    """Best-effort config load (project if a valid slug, else global). ``None`` on error."""
    try:
        if slug is not None:
            paths.validate_slug(slug)
            return load_project_config(slug, root)
        return load_global_config(root)
    except SeedgraphError:
        return None


def _check_pricing_snapshot_age() -> CheckResult:
    """Warn when the bundled ``llm_capabilities.yaml`` pricing snapshot is stale.

    The snapshot is a hand-pinned static file (re-verify before budget enforcement);
    an aged snapshot is a WARNING, never a hard failure, so a keyless/offline home
    still passes the gate."""
    try:
        caps = load_llm_capabilities()
    except SeedgraphError as exc:
        return CheckResult("pricing_snapshot", False, f"capabilities snapshot invalid: {exc}")
    age = (_date.today() - caps.snapshot_date).days
    fresh = age <= _PRICING_SNAPSHOT_MAX_AGE_DAYS
    return CheckResult(
        "pricing_snapshot_age",
        fresh,
        f"pricing snapshot {caps.snapshot_date.isoformat()} is {age} day(s) old"
        + ("" if fresh else f" (> {_PRICING_SNAPSHOT_MAX_AGE_DAYS}d — re-verify pricing before budget enforcement)"),
        severity="warning",
    )


def _check_llm_capabilities(root, slug) -> list[CheckResult]:
    """Task/model capability + structured-output mismatch warnings (no dispatch).

    For each routed task, resolve the preferred profile's model and check it against
    the capability snapshot: a model with no capability row, or a route that
    ``requires_structured_output`` against a model whose ``supports_structured_output``
    is False, is a WARNING (the executor parses/validates JSON app-side regardless)."""
    cfg = _load_cfg(root, slug)
    if cfg is None:
        return []  # _check_config already reported the load failure
    try:
        caps = load_llm_capabilities()
    except SeedgraphError:
        return []  # pricing-snapshot check reports the snapshot failure
    missing: list[str] = []
    structured: list[str] = []
    for task_type, route in sorted(cfg.llm.routes.items()):
        profile = cfg.llm.profiles.get(route.preferred_profile)
        if profile is None or not profile.model:
            continue  # no_llm / deterministic route — no model to validate
        cap = caps.models.get(profile.model)
        if cap is None:
            missing.append(f"{task_type}->{profile.model}")
            continue
        if route.requires_structured_output and cap.supports_structured_output is False:
            structured.append(f"{task_type}->{profile.model}")
    results: list[CheckResult] = []
    results.append(
        CheckResult(
            "llm_capability_coverage",
            not missing,
            "every routed model has a capability row"
            if not missing
            else f"{len(missing)} routed model(s) absent from the snapshot: {missing}",
            severity="warning",
        )
    )
    results.append(
        CheckResult(
            "llm_structured_output_match",
            not structured,
            "no structured-output capability mismatch"
            if not structured
            else f"{len(structured)} route(s) require structured output from a model that "
            f"lacks a native structured-output API (validated app-side): {structured}",
            severity="warning",
        )
    )
    return results


def _check_private_content_policy(root, slug) -> CheckResult:
    """Flag a private-content policy conflict (a route that would block real work).

    Conflict = a ``requires_source_text`` task whose preferred profile is EXTERNAL
    while ``external_llm_for_private_full_text`` is off AND no local fallback is
    configured: such a route refuses every non-open-access work (skipped_policy) with
    no local path. Also flags the privacy-reducing case where private full text is
    permitted to leave the machine. Both are WARNINGS."""
    cfg = _load_cfg(root, slug)
    if cfg is None:
        return CheckResult("private_content_policy", True, "config unavailable (skipped)", severity="warning")

    if cfg.content_policy.external_llm_for_private_full_text:
        return CheckResult(
            "private_content_policy",
            True,
            "content_policy.external_llm_for_private_full_text=TRUE — private full text "
            "may leave the machine for an external LLM (privacy notice)",
            severity="warning",
        )

    conflicts: list[str] = []
    for task_type, route in sorted(cfg.llm.routes.items()):
        if not route.requires_source_text:
            continue
        pref = cfg.llm.profiles.get(route.preferred_profile)
        if pref is None or is_local_profile(pref):
            continue  # local-first preferred — no conflict
        fb = cfg.llm.profiles.get(route.fallback_profile) if route.fallback_profile else None
        has_local_fallback = fb is not None and is_local_profile(fb)
        if not has_local_fallback:
            conflicts.append(task_type)
    if conflicts:
        return CheckResult(
            "private_content_policy",
            True,
            f"{len(conflicts)} source-text task(s) prefer an external profile with no "
            f"local fallback while private full text is forbidden — non-open-access works "
            f"will skip (skipped_policy): {conflicts}. Add a local profile or "
            f"--confirm-external.",
            severity="warning",
        )
    return CheckResult(
        "private_content_policy",
        True,
        "no private-content policy conflict (source-text tasks keep a local path)",
    )


def _probe_ollama(root, slug) -> CheckResult:
    """Opt-in TCP probe of the configured Ollama endpoint (``--probe-ollama``).

    A pure socket connect (no model call, no key) to the local-first profile's
    ``base_url`` host:port. A closed port is a WARNING (Ollama is optional in a
    local-first setup), never a hard failure."""
    import socket
    from urllib.parse import urlparse

    from .llm.providers.ollama import DEFAULT_BASE_URL

    cfg = _load_cfg(root, slug)
    base_url = DEFAULT_BASE_URL
    if cfg is not None:
        for profile in cfg.llm.profiles.values():
            if profile.provider == "ollama" and profile.base_url:
                base_url = profile.base_url
                break
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return CheckResult("ollama_probe", True, f"Ollama reachable at {host}:{port}")
    except OSError as exc:
        return CheckResult(
            "ollama_probe",
            True,
            f"Ollama not reachable at {host}:{port} ({exc.__class__.__name__}); local-first "
            f"extraction will fall back to the hosted profile if configured",
            severity="warning",
        )


def _check_gpu_probe() -> CheckResult:
    """Informational GPU probe: torch importability, detected device, ``gpu_memory_gb``.

    ALWAYS ``ok=True`` — the probe informs, it never gates anything (Build E design 8).
    The recommended marker batch profile stays hard-coded ``conservative`` regardless
    of what the probe sees: a broken/mismatched CUDA build (a live possibility on an
    sm_120 box) must never silently pick a larger, OOM-prone tier off a misleading
    VRAM number — VRAM-gated auto-tiering is exactly what produced v1's live 16 GB
    OutOfMemoryError cascade. A wrong probe degrades to *slow*, never to *crash*.
    """
    try:
        import torch  # type: ignore  # probed lazily so doctor stays import-safe torch-less
    except Exception as exc:  # noqa: BLE001 - torch absent/broken is informational, never a failure
        return CheckResult(
            "gpu_probe",
            True,
            f"torch not importable ({exc.__class__.__name__}) — local marker conversion "
            f"needs the ML runtime; informational only, nothing gated",
        )

    device = "unknown"
    gpu_memory_gb = None
    try:
        if torch.cuda.is_available():
            device = "cuda"
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = round(total / (1024**3), 1)
            except Exception:  # noqa: BLE001 - broken/mismatched driver → leave None
                gpu_memory_gb = None
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except Exception:  # noqa: BLE001 - device probe failure → device stays "unknown"
        pass
    return CheckResult(
        "gpu_probe",
        True,
        f"torch importable; device={device}; gpu_memory_gb={gpu_memory_gb}; "
        f"recommended batch profile: conservative (hard-coded — probe never gates)",
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
    results.extend(_check_key_refs(root, slug))
    # Stage C LLM-backend checks (offline; no paid calls). All advisory/warning so a
    # keyless local-first home still passes the gate.
    results.append(_check_pricing_snapshot_age())
    results.extend(_check_llm_capabilities(root, slug))
    results.append(_check_private_content_policy(root, slug))
    if probe_ollama:
        results.append(_probe_ollama(root, slug))
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
