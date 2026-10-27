"""Every module under ``src/seedgraph`` is reachable from an entry point, or is on a short,
reasoned allowlist (docs/V3_UI_DESIGN.md §9, U9).

The scan is static: it reads every ``import`` / ``from ... import`` in a module's source,
including imports inside functions and ``TYPE_CHECKING`` blocks, and importing
``a.b.c`` also reaches the packages ``a`` and ``a.b`` (their ``__init__`` runs).

Entry points are the console script (``seedgraph.cli``, ``python -m seedgraph``), the
MCP server, the web app, the app side of the converter worker (``seedgraph.conversion``),
and every module in ``seedgraph.web.routes`` (the web app discovers those with
``pkgutil``, which a static scan cannot see).

Reserved roots were v2 modules kept for a later stage (the orchestrator's build rule: code
is deleted in the stage that replaces it). Stage 4 (C10) deleted or wired the last of them,
so the reserved list is empty and must stay so.
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

# module or package (``.*`` = every module under it) -> reason. Reserved roots are kept and
# followed (their imports stay reachable). Stage 4 (C10) emptied it; it must stay empty.
RESERVED: dict[str, str] = {}

# Unreachable modules that stay on purpose (not followed further).
KEEP = {
    "seedgraph.migrate.v2_schema": "frozen v2 schema, package data read by the migrate test fixtures",
    "seedgraph.library._calibrate_textquality": "calibration script, run by hand",
    "seedgraph.library._calibrate_titlecheck": "calibration script, run by hand",
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


def test_nothing_is_reserved_for_a_later_stage():
    # Stage 4 was the last stage with reserved v2 code (docs/V3_CONCEPTS_DESIGN.md §10).
    assert RESERVED == {}
    assert not [name for name, why in KEEP.items() if "reserved" in why.lower()]


# v2 code the stage 4 design deletes (docs/V3_CONCEPTS_DESIGN.md §10): stored co-occurrence
# and broader/narrower edges, the labels-only merge proposer and its review-every-merge
# flow, the v2 citation graph, and the ORM-bound corpus-growth pipeline.
DELETED_IN_STAGE4 = (
    "seedgraph.semantic.access", "seedgraph.semantic.audit", "seedgraph.semantic.canon",
    "seedgraph.semantic.concepts", "seedgraph.semantic.current", "seedgraph.semantic.edges",
    "seedgraph.semantic.export", "seedgraph.semantic.graph_build", "seedgraph.semantic.proposer",
    "seedgraph.semantic.query", "seedgraph.graph",
    "seedgraph.citation.edges", "seedgraph.citation.parsed_bib", "seedgraph.citation.project_edges",
    "seedgraph.acquisition.walk", "seedgraph.acquisition.resolve", "seedgraph.acquisition.backfill",
    "seedgraph.acquisition.bridge", "seedgraph.acquisition.service",
    "seedgraph.acquisition.doctor_reconcile", "seedgraph.extraction", "seedgraph.db.models_project",
    "seedgraph.project.review", "seedgraph.project.merge", "seedgraph.project.service",
)


def test_stage4_deletions_are_gone():
    present = sorted(m for m in DELETED_IN_STAGE4 if _expand(m + ".*"))
    assert not present, present


def test_no_stored_concept_edge_types_remain():
    # Co-occurrence and broader/narrower concept edges were v2 only (§10).
    words = ("co_occurs_with", "broader_than", "narrower_than")
    hits = sorted(m for m, p in MODULES.items()
                  if not m.startswith("seedgraph.migrate.v2_schema")
                  and any(w in p.read_text(encoding="utf-8") for w in words))
    assert not hits, hits


def _private_names_used_across_modules(name: str) -> list[str]:
    """``from X import _y`` and ``mod._y`` accesses in ``name`` that reach a sibling
    seedgraph module's private helper."""
    tree = ast.parse(MODULES[name].read_text(encoding="utf-8"))
    seedgraph_aliases = set()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("seedgraph")):
            for alias in node.names:
                if alias.name.startswith("_") and not alias.name.startswith("__"):
                    found.append(f"{node.module}.{alias.name}")
                else:
                    seedgraph_aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in seedgraph_aliases
                and node.attr.startswith("_") and not node.attr.startswith("__")):
            found.append(f"{node.value.id}.{node.attr}")
    return sorted(set(found))


def test_growth_uses_only_public_library_helpers():
    # C6 reached into ingest/graphs privates; C10 made them public (§12 C10).
    assert _private_names_used_across_modules("seedgraph.library.growth") == []
    assert _private_names_used_across_modules("seedgraph.library.literatures") == []


def test_the_scan_sees_lazy_and_relative_imports():
    # Guards the scanner itself: the CLI reaches the library commands through
    # `commands.register`, which imports its modules inside the function body.
    reached = _closure({"seedgraph.cli"})
    assert "seedgraph.commands.library" in reached
    assert "seedgraph.library.store" in reached
