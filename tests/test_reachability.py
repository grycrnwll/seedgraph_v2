"""Every module under ``src/seedgraph`` is reachable from an entry point, or is on a short,
reasoned allowlist (docs/V3_UI_DESIGN.md §9, U9).

The scan is static: it reads every ``import`` / ``from ... import`` in a module's source,
including imports inside functions and ``TYPE_CHECKING`` blocks, and importing
``a.b.c`` also reaches the packages ``a`` and ``a.b`` (their ``__init__`` runs).

Entry points are the console script (``seedgraph.cli``, ``python -m seedgraph``), the
MCP server, the web app, the app side of the converter worker (``seedgraph.conversion``),
and every module in ``seedgraph.web.routes`` (the web app discovers those with
``pkgutil``, which a static scan cannot see).

Reserved roots are v2 modules kept for stage 4 (the orchestrator's build rule: code is
deleted in the stage that replaces it). They count as roots, so what they import stays
too, and their own unit tests keep running.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PKG = SRC / "seedgraph"

ENTRY_POINTS = (
    "seedgraph.cli",
    "seedgraph.__main__",
    "seedgraph.mcp.server",
    "seedgraph.web.app",
    "seedgraph.conversion",
    "seedgraph.web.routes.*",
)

STAGE4 = "reserved for stage 4 (concepts/graphs/growth)"

# module or package (``.*`` = every module under it) -> reason. Reserved roots: kept and
# followed (their imports stay reachable).
RESERVED = {
    "seedgraph.concepts.*": STAGE4 + ": concept model (C1); wired to the CLI, MCP and web by C4/C5/C7",
    "seedgraph.library.literatures": STAGE4 + ": literatures (C2); wired to the CLI, MCP and web by C7/C8/C9",
    "seedgraph.semantic.*": STAGE4 + ": concept guardrails, closure, stoplist, type fence (ADR-0013)",
    "seedgraph.graph.*": STAGE4 + ": citation graph and communities (keeps networkx)",
    "seedgraph.citation.*": STAGE4 + ": reference parsing and resolution, citation edges",
    "seedgraph.acquisition.resolve": STAGE4 + ": corpus growth (resolve)",
    "seedgraph.acquisition.walk": STAGE4 + ": corpus growth (citation walk)",
    "seedgraph.acquisition.backfill": STAGE4 + ": corpus growth (title backfill)",
    "seedgraph.acquisition.bridge": STAGE4 + ": corpus growth (work/markdown bridge)",
    "seedgraph.acquisition.preview": STAGE4 + ": corpus growth (walk preview)",
    "seedgraph.extraction.metadata": STAGE4 + ": deterministic identifier scan for growth",
}

# Unreachable modules that stay on purpose (not followed further).
KEEP = {
    "seedgraph.lenses.registry": "lens definitions for the stage 4 lens choice (lenses/{schema,registry,prompt})",
    "seedgraph.migrate.v2_schema": "frozen v2 schema, package data read by the migrate test fixtures",
    "seedgraph.library._calibrate_textquality": "calibration script, run by hand",
    "seedgraph.library._calibrate_titlecheck": "calibration script, run by hand",
    "seedgraph.db.models_project": STAGE4 + ": ORM mapping the kept lens-registry (test_phase_6) and reserved semantic (test_phase_7) tests build data with",
}


def _modules() -> dict[str, Path]:
    found = {}
    for path in PKG.rglob("*.py"):
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        found[".".join(parts)] = path
    return found


MODULES = _modules()


def _expand(pattern: str) -> set[str]:
    if pattern.endswith(".*"):
        base = pattern[:-2]
        return {m for m in MODULES if m == base or m.startswith(base + ".")}
    return {pattern} if pattern in MODULES else set()


def _imports(name: str) -> set[str]:
    path = MODULES[name]
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    targets: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                if node.level > 1:
                    base = base[: -(node.level - 1)]
                module = ".".join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            targets.add(module)
            targets.update(f"{module}.{alias.name}" for alias in node.names)  # `from pkg import mod`
    reached = set()
    for target in targets:
        parts = target.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix in MODULES:
                reached.add(prefix)
    return reached


def _closure(roots: set[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen or name not in MODULES:
            continue
        seen.add(name)
        stack.extend(_imports(name))
    return seen


def _roots(patterns) -> set[str]:
    return set().union(*(_expand(p) for p in patterns))


def test_every_module_is_reachable_or_allowlisted():
    reachable = _closure(_roots(ENTRY_POINTS) | _roots(RESERVED))
    unreachable = sorted(set(MODULES) - reachable - set(KEEP))
    assert not unreachable, (
        "unreachable from every entry point; delete them or allowlist them with a reason:\n  "
        + "\n  ".join(unreachable))


def test_every_allowlist_entry_names_an_existing_module():
    for pattern in [*ENTRY_POINTS, *RESERVED, *KEEP]:
        assert _expand(pattern), f"stale allowlist entry: {pattern}"


def test_kept_modules_are_really_unreachable():
    # A KEEP entry that the core already reaches is dead weight on the list.
    reachable = _closure(_roots(ENTRY_POINTS) | _roots(RESERVED))
    assert not (set(KEEP) & reachable), sorted(set(KEEP) & reachable)


def test_the_scan_sees_lazy_and_relative_imports():
    # Guards the scanner itself: the CLI reaches the library commands through
    # `commands.register`, which imports its modules inside the function body.
    reached = _closure({"seedgraph.cli"})
    assert "seedgraph.commands.library" in reached
    assert "seedgraph.library.store" in reached
