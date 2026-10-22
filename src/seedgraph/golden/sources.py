"""Where OLD and NEW markdown come from for a golden-set check.

Three sources, each turning one manifest entry into a :class:`Rendition`:

* ``v2-stored`` (:class:`V2StoredSource`) - v2's stored markdown and conversion
  wall time, read-only from the v2 home recorded in the entry.
* a directory (:class:`DirectorySource`) - ``<dir>/<GOLDEN_ID>.md`` plus an optional
  ``<dir>/timings.json`` mapping ID -> seconds, or ID -> ``{"seconds": s,
  "converter_version": v}``.
* ``convert`` (:class:`ConvertSource`) - runs a :class:`Converter` on the cached
  PDF. The converter is injected; the CLI obtains it from
  :data:`converter_factory`.

TODO(stage 0 part B): wire the warm converter pool. Set
``seedgraph.golden.sources.converter_factory`` to a zero-argument callable that
returns an object satisfying :class:`Converter` (e.g. a context-managed pool
started on first use). Until then ``--new convert`` exits with "converter pool
not wired". Nothing here imports part B's modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from . import v2
from .checks import Rendition
from .errors import GoldenError

#: Options passed to ``Converter.convert``; mirrors v2's stored Marker config
#: (v2 also had use_llm/redo_inline_math off, which the converter fixes itself).
DEFAULT_CONVERT_OPTIONS: dict[str, Any] = {
    "output_format": "markdown",
    "paginate_output": True,
    "force_ocr": False,
}

CONVERT_TIMING_LABEL = "warm converter timings.convert_seconds (models already loaded)"


class ConvertTimings(Protocol):
    convert_seconds: float


class ConvertResult(Protocol):
    markdown: str
    page_count: int
    timings: ConvertTimings  # also accepted: a mapping with "convert_seconds"
    converter_version: str


@runtime_checkable
class Converter(Protocol):
    """Minimal converter boundary the golden check needs."""

    def convert(self, pdf_path: Path, options: Mapping[str, Any]) -> ConvertResult: ...


#: Hook for the real warm converter pool (see module TODO). ``None`` = not wired.
converter_factory: Optional[Callable[[], Converter]] = None


class Source(Protocol):
    label: str

    def rendition(self, entry: dict) -> Rendition: ...


class V2StoredSource:
    label = "v2-stored"

    def rendition(self, entry: dict) -> Rendition:
        return v2.stored_rendition(entry["quote_source"]["v2_home"], entry["v2_markdown_id"])


class DirectorySource:
    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.label = f"dir:{self.directory}"
        tpath = self.directory / "timings.json"
        self._timings = json.loads(tpath.read_text(encoding="utf-8")) if tpath.exists() else {}

    def rendition(self, entry: dict) -> Rendition:
        gid = entry["id"]
        path = self.directory / f"{gid}.md"
        if not path.is_file():
            raise GoldenError(f"missing {path.name} in {self.directory}")
        t = self._timings.get(gid)
        seconds = version = None
        if isinstance(t, (int, float)):
            seconds = float(t)
        elif isinstance(t, dict):
            seconds = float(t["seconds"]) if t.get("seconds") is not None else None
            version = t.get("converter_version")
        return Rendition(
            markdown=path.read_text(encoding="utf-8"),
            seconds=seconds,
            timing_label="timings.json" if seconds is not None else "",
            converter_version=version,
        )


class ConvertSource:
    label = "convert"

    def __init__(self, converter: Converter, options: Optional[Mapping[str, Any]] = None):
        self.converter = converter
        self.options = dict(DEFAULT_CONVERT_OPTIONS if options is None else options)

    def rendition(self, entry: dict) -> Rendition:
        path = v2.pdf_path(entry["quote_source"]["v2_home"], entry["pdf_sha256"])
        result = self.converter.convert(path, self.options)
        t = result.timings
        seconds = t.get("convert_seconds") if isinstance(t, Mapping) else getattr(t, "convert_seconds", None)
        return Rendition(
            markdown=result.markdown,
            seconds=float(seconds) if seconds is not None else None,
            timing_label=CONVERT_TIMING_LABEL,
            converter_version=result.converter_version,
            page_count=result.page_count,
        )


def make_source(spec: str) -> Source:
    """``v2-stored`` | ``convert`` | a directory path."""
    if spec == "v2-stored":
        return V2StoredSource()
    if spec == "convert":
        if converter_factory is None:
            raise GoldenError("converter pool not wired (TODO: stage 0 part B sets sources.converter_factory)")
        return ConvertSource(converter_factory())
    path = Path(spec)
    if not path.is_dir():
        raise GoldenError(f"source {spec!r} is not 'v2-stored', 'convert' or an existing directory")
    return DirectorySource(path)
