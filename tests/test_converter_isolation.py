"""The converter boundary: the app never imports Marker / torch / Surya, and the
converter never imports seedgraph (ticket 12 #1, docs/DEPENDENCY_POLICY.md)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEAVY = {"torch", "marker", "surya", "torchvision", "transformers"}


def _imported_roots(py: Path) -> set[str]:
    tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_app_module_imports_marker_torch_or_surya():
    offenders = {
        str(py.relative_to(REPO)): sorted(_imported_roots(py) & HEAVY)
        for py in (REPO / "src" / "seedgraph").rglob("*.py")
        if _imported_roots(py) & HEAVY
    }
    assert offenders == {}


def test_converter_package_never_imports_seedgraph():
    offenders = [
        str(py.relative_to(REPO))
        for py in (REPO / "converter" / "src").rglob("*.py")
        if "seedgraph" in _imported_roots(py)
    ]
    assert offenders == []


def test_app_cli_and_mcp_server_import_without_attempting_heavy_imports():
    """Import the package, CLI and MCP server in a fresh interpreter with an import hook
    that records any attempt to import a heavy module (even a guarded one)."""
    code = f"""
import json, sys
HEAVY = {sorted(HEAVY)!r}
attempts = []
class Spy:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in HEAVY:
            attempts.append(name)
        return None
sys.meta_path.insert(0, Spy())
import seedgraph, seedgraph.cli, seedgraph.commands.work, seedgraph.mcp.server
import seedgraph.conversion, seedgraph.cache.convert, seedgraph.library.files
print(json.dumps(sorted(set(attempts))))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


def test_convert_worker_subprocess_entrypoint_is_gone():
    assert not (REPO / "src" / "seedgraph" / "web" / "_convert_worker.py").exists()
