"""Packaging / environment invariants for v3 (Stage 0).

Pure file reads: no subprocess, no network, nothing written.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONVERTER_PACKAGES = {"torch", "torchvision", "marker-pdf", "surya-ocr"}


def _toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _dep_name(spec: str) -> str:
    return re.split(r"[\s\[<>=!~;]", spec, maxsplit=1)[0].lower()


def test_console_command_is_interim_seedgraph_v3():
    # Renamed back to `seedgraph` at stage 7 (v2's seedgraph.exe is on PATH until then).
    scripts = _toml(REPO / "pyproject.toml")["project"]["scripts"]
    assert scripts == {"seedgraph-v3": "seedgraph.cli:app"}


def test_mcp_is_a_core_dependency_and_marker_extra_is_gone():
    project = _toml(REPO / "pyproject.toml")["project"]
    assert "mcp" in {_dep_name(d) for d in project["dependencies"]}
    extras = project.get("optional-dependencies", {})
    assert "marker" not in extras
    assert "mcp" not in extras


def test_no_uv_workspace():
    assert "workspace" not in _toml(REPO / "pyproject.toml").get("tool", {}).get("uv", {})


def test_python_version_files_pin_313():
    assert (REPO / ".python-version").read_text().strip() == "3.13"
    assert (REPO / "converter" / ".python-version").read_text().strip() == "3.13"


def test_app_lock_has_no_converter_packages():
    names = {p["name"] for p in _toml(REPO / "uv.lock")["package"]}
    assert not names & CONVERTER_PACKAGES


def test_converter_does_not_depend_on_seedgraph():
    conv = _toml(REPO / "converter" / "pyproject.toml")["project"]
    assert "seedgraph" not in {_dep_name(d) for d in conv["dependencies"]}
    names = {p["name"] for p in _toml(REPO / "converter" / "uv.lock")["package"]}
    assert "seedgraph" not in names
    assert {"torch", "marker-pdf", "surya-ocr"} <= names
