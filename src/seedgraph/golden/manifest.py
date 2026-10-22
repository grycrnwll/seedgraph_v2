"""Golden-set manifest: ``<v3 home>/golden_set.json`` (machine-local, never in the repo).

The seed (machine-specific data, supplied by the researcher) is a JSON list of::

    {"id": "GS01", "category": "math", "md": "<markdown-id hex prefix>",
     "pdf": "<pdf sha256 prefix, optional>", "project": "<v2 project slug>",
     "v2_home": "<optional>", "work_id": "<optional>"}

:func:`build_manifest` resolves every prefix against the v2 cache read-only and
fails loudly on zero or several matches. Each manifest entry is::

    {"id", "category", "pdf_sha256", "v2_markdown_id",
     "quote_source": {"v2_home", "project_slug", "work_id", "schema_id",
                      "note_id", "note_created_at"}}

``note_id`` pins the paper's current note at init time, so OLD and NEW are
always scored against the same quote list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from . import v2
from .errors import GoldenError

MANIFEST_NAME = "golden_set.json"


def load_seed(path: Path | str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(e, dict) for e in data):
        raise GoldenError("seed must be a JSON list of objects")
    return data


def build_manifest(seed: Iterable[dict], *, default_v2_home: Path | str = v2.DEFAULT_V2_HOME) -> list[dict]:
    """Resolve a seed into manifest entries (read-only against each v2 home)."""
    entries = []
    seen = set()
    for item in seed:
        try:
            gid, category, md, slug = item["id"], item["category"], item["md"], item["project"]
        except KeyError as exc:
            raise GoldenError(f"seed entry {item!r} missing {exc}") from None
        if gid in seen:
            raise GoldenError(f"duplicate golden id {gid}")
        seen.add(gid)
        home = Path(item.get("v2_home") or default_v2_home)
        try:
            doc = v2.resolve_markdown(home, md)
            if item.get("pdf"):
                pdf = v2.resolve_pdf(home, item["pdf"])
                if pdf != doc["pdf_sha256"]:
                    raise GoldenError(f"pdf prefix {item['pdf']!r} does not match the markdown's source file")
            note = v2.current_note(home, slug, doc["markdown_id"], item.get("work_id"))
        except GoldenError as exc:
            raise GoldenError(f"{gid}: {exc}") from None
        entries.append({
            "id": gid,
            "category": category,
            "pdf_sha256": doc["pdf_sha256"],
            "v2_markdown_id": doc["markdown_id"],
            "quote_source": {"v2_home": str(home), "project_slug": slug, **note},
        })
    return entries


def guard_v3_home(home: Path | str, v2_homes: Iterable[Path | str]) -> None:
    """Refuse to write anything when the v3 home is (or is inside) a v2 home."""
    h = Path(home).resolve()
    for other in {Path(x) for x in v2_homes} | {v2.DEFAULT_V2_HOME}:
        o = other.resolve()
        if h == o or o in h.parents:
            raise GoldenError(
                f"refusing to write under a v2 home ({o}); set SEEDGRAPH_HOME/--root to the v3 home"
            )


def manifest_path(home: Path | str) -> Path:
    return Path(home) / MANIFEST_NAME


def write_manifest(entries: list[dict], home: Path | str) -> Path:
    home = Path(home)
    guard_v3_home(home, [e["quote_source"]["v2_home"] for e in entries])
    home.mkdir(parents=True, exist_ok=True)
    path = manifest_path(home)
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    return path


def load_manifest(home: Path | str) -> list[dict]:
    path = manifest_path(home)
    if not path.exists():
        raise GoldenError(f"no golden-set manifest at {path}; run `seedgraph-v3 golden init <seed>`")
    return json.loads(path.read_text(encoding="utf-8"))
