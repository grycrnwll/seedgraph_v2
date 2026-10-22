"""Marker conversion backend (doc 11 §3 — versioned backend behind a Protocol).

Marker never runs in the app process. The real adapter is
:class:`seedgraph.conversion.WarmMarkerBackend`, which sends the PDF to a warm
converter in the separate converter environment (docs/CONVERTER_PROTOCOL.md). Tests
use :class:`FakeMarkerBackend` and never need a GPU, network, or Marker.

Pagination (gap 10 / decision 83): the backend passes ``cfg.paginate_output`` to
the marker call so the markdown carries **in-markdown page-delimiter markers**
that phase_3 parses for real page citations. The datalab/remote markdown-only path
also honors it (block JSON unavailable there) — reinforcing that page fidelity
rides on the delimiters, NOT on block-JSON page extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids convert<->backend import cycle)
    from .convert import MarkerConfig


@dataclass
class MarkerResult:
    """Raw output of a single backend conversion call.

    ``markdown`` is the converted text (carrying page delimiters when
    ``paginate_output`` is set). ``block_json`` is Marker's raw, UNVALIDATED
    block/metadata dump (or ``None``); this phase dumps it verbatim to
    ``marker/{conv}/meta.json`` and never parses it (doc 11 §7). ``warnings`` are
    backend-level signals merged into the run's ``warnings_json``.
    """

    markdown: str
    block_json: dict | None
    warnings: list[str]
    #: Pages in the source PDF (from the PDF itself), when the backend knows it.
    page_count: int | None = None
    #: Backend timings, e.g. ``convert_seconds`` / ``wall_seconds``.
    timings: dict = field(default_factory=dict)
    #: Converter version that actually produced this result (worker's ready message).
    converter_version: str | None = None


@runtime_checkable
class MarkerBackend(Protocol):
    """A pluggable PDF->markdown converter with a captured version string."""

    def __call__(self, pdf_path: Path, cfg: "MarkerConfig") -> MarkerResult:
        """Convert ``pdf_path`` to markdown under ``cfg`` (incl. ``paginate_output``)."""
        ...

    @property
    def version(self) -> str:
        """Backend version string captured for the conversion fingerprint."""
        ...


class FakeMarkerBackend:
    """Deterministic in-memory test double (no GPU/network/marker dep).

    Returns canned markdown and counts invocations, so tests can assert the
    milestone "same bytes -> Marker runs zero additional times" (``call_count``).
    Implemented in full because it is a trivial test fixture the skipped phase_1
    tests construct at import/collection time.
    """

    def __init__(
        self,
        markdown: str = "# Title\n\n## Section\n\nBody.\n\n## References\n\n[1] x.\n",
        *,
        version: str = "fake-0.0.0",
        block_json: dict | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self._markdown = markdown
        self._version = version
        self._block_json = block_json
        self._warnings = list(warnings or [])
        self.call_count = 0

    @property
    def version(self) -> str:
        return self._version

    def __call__(self, pdf_path: Path, cfg: "MarkerConfig") -> MarkerResult:
        self.call_count += 1
        return MarkerResult(
            markdown=self._markdown,
            block_json=self._block_json,
            warnings=list(self._warnings),
        )
