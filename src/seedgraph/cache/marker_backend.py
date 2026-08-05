"""Marker conversion backend (doc 11 §3 — versioned backend behind a Protocol).

Marker is a heavy / optionally-GPU dependency, so it sits behind a
:class:`MarkerBackend` ``Protocol``. ``marker-pdf`` is an install **extra**; all
tests use :class:`FakeMarkerBackend` and never require a GPU, network, or the real
package. The real :class:`LocalMarkerBackend` lazy-imports ``marker`` only inside
its ``__call__`` so importing this module never pulls the heavy dep.

Pagination (gap 10 / decision 83): the backend passes ``cfg.paginate_output`` to
the marker call so the markdown carries **in-markdown page-delimiter markers**
that phase_3 parses for real page citations. The datalab/remote markdown-only path
also honors it (block JSON unavailable there) — reinforcing that page fidelity
rides on the delimiters, NOT on block-JSON page extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
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


class LocalMarkerBackend:
    """Real backend over ``marker-pdf`` (lazy-imported inside ``__call__``).

    Captures ``importlib.metadata.version('marker-pdf')`` as :attr:`version` and
    ``platform.python_version()`` onto the run. Defaults to no-LLM (local,
    deterministic); passes ``cfg.paginate_output`` to the marker call; returns
    best-effort ``block_json`` (``None`` on the markdown-only/remote path).
    """

    @property
    def version(self) -> str:
        """``marker-pdf`` version via ``importlib.metadata`` (captured at runtime).

        Lazy: ``importlib.metadata.version`` raises ``PackageNotFoundError`` (a
        subclass of ``ModuleNotFoundError``) when the optional ``marker-pdf`` extra
        is not installed, so importing this module never requires the dep.
        """
        from importlib.metadata import version as _pkg_version

        return _pkg_version("marker-pdf")

    def __call__(self, pdf_path: Path, cfg: "MarkerConfig") -> MarkerResult:
        """Run Marker on ``pdf_path``; lazy-import the dep so module import is cheap.

        Passes ``cfg.paginate_output`` to the marker call so the markdown carries
        in-markdown page-delimiter markers (gap 10 / decision 83). No text leaves the
        machine on the no-LLM default path. ``block_json`` is best-effort (``None``
        on the markdown-only/remote path). This path is exercised only by the opt-in
        ``-m marker`` smoke test; all unit tests use :class:`FakeMarkerBackend`.
        """
        # Lazy imports — keep marker-pdf strictly optional at package-import time.
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        config: dict = {
            "output_format": cfg.output_format,
            "paginate_output": cfg.paginate_output,
            "force_ocr": cfg.force_ocr,
            "use_llm": cfg.use_llm,
            "redo_inline_math": cfg.redo_inline_math,
        }
        if cfg.use_llm and cfg.llm_service:
            config["llm_service"] = cfg.llm_service

        converter = PdfConverter(artifact_dict=create_model_dict(), config=config)
        rendered = converter(str(pdf_path))
        markdown, _metadata, _images = text_from_rendered(rendered)

        block_json: dict | None = None
        meta = getattr(rendered, "metadata", None)
        if isinstance(meta, dict):
            block_json = meta

        return MarkerResult(markdown=markdown, block_json=block_json, warnings=[])


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
