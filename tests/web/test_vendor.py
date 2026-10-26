"""Vendored front-end assets match web/static/vendor/VERSIONS.json (design §1.1).

Every file on disk is listed, every listed file exists, and every sha256 matches, so a
vendored asset can never drift silently. No CDN is referenced by our own templates.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import seedgraph.web

WEB = Path(seedgraph.web.__file__).parent
VENDOR = WEB / "static" / "vendor"


def _manifest() -> dict:
    return json.loads((VENDOR / "VERSIONS.json").read_text(encoding="utf-8"))


def _listed() -> dict[str, str]:
    return {f["path"]: f["sha256"] for pkg in _manifest()["packages"] for f in pkg["files"]}


def _sha256(path: Path) -> set[str]:
    """The file's hash, plus its LF-normalised hash: a Windows checkout with
    ``core.autocrlf`` rewrites text files to CRLF without changing their content."""
    data = path.read_bytes()
    return {hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}


def test_every_vendored_file_is_listed_with_a_matching_hash():
    listed = _listed()
    on_disk = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*")
               if p.is_file() and p.name != "VERSIONS.json"}
    assert on_disk == set(listed), (on_disk ^ set(listed))
    bad = [rel for rel, sha in listed.items() if sha not in _sha256(VENDOR / rel)]
    assert bad == []


def test_the_manifest_pins_htmx_and_pdfjs_with_their_licences():
    pkgs = {p["name"]: p for p in _manifest()["packages"]}
    assert set(pkgs) == {"htmx", "pdfjs"}
    for pkg in pkgs.values():
        assert pkg["version"] and pkg["source"].startswith("https://registry.npmjs.org/")
        assert f"{pkg['name']}/LICENSE" in {f["path"] for f in pkg["files"]}
    paths = {f["path"] for f in pkgs["pdfjs"]["files"]}
    for needed in ("pdfjs/pdf.min.mjs", "pdfjs/pdf.worker.min.mjs", "pdfjs/pdf_viewer.mjs",
                   "pdfjs/pdf_viewer.css"):
        assert needed in paths
    assert any(p.startswith("pdfjs/cmaps/") for p in paths)
    assert any(p.startswith("pdfjs/standard_fonts/") for p in paths)


def test_vendored_scripts_are_javascript_not_error_pages():
    for rel in ("htmx/htmx.min.js", "pdfjs/pdf.min.mjs", "pdfjs/pdf.worker.min.mjs"):
        head = (VENDOR / rel).read_bytes()[:512].lstrip().lower()
        assert not head.startswith((b"<!doctype", b"<html")), rel


def test_our_templates_and_scripts_load_nothing_from_a_cdn():
    ours = [p for p in (WEB / "templates").rglob("*.html")]
    ours += [p for p in (WEB / "static").rglob("*") if p.is_file() and "vendor" not in p.parts
             and p.suffix in {".js", ".css", ".mjs"}]
    assert ours
    loads = re.compile(r"""(?:\b(?:src|href|action)\s*=\s*["']?|url\(\s*["']?|\bimport\b[^;]*?["']|"""
                       r"""\bfrom\s*["']|\bfetch\(\s*["'])(?:https?:)?//""", re.I)
    for p in ours:
        assert not loads.search(p.read_text(encoding="utf-8")), p
