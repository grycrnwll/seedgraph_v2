"""``WarmMarkerBackend``: the converter pool behind the cache's ``MarkerBackend`` seam.

``convert_source_file`` keeps doing dedup, validation and storage in the app; this
adapter only turns a PDF into a :class:`~seedgraph.cache.marker_backend.MarkerResult`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..cache.marker_backend import MarkerResult
from .pool import ConverterPool, get_pool
from .warm import ConverterError

if TYPE_CHECKING:  # pragma: no cover
    from ..cache.convert import MarkerConfig


class WarmMarkerBackend:
    """Converts through a :class:`ConverterPool` (the shared one by default)."""

    def __init__(self, pool: ConverterPool | None = None) -> None:
        self._pool = pool

    @property
    def pool(self) -> ConverterPool:
        return self._pool if self._pool is not None else get_pool()

    @property
    def version(self) -> str:
        """The marker-pdf version (from converter/uv.lock; no child is started, so a
        dedup hit never wakes a GPU). Workers that report anything else are refused."""
        return self.pool.converter_version

    def __call__(self, pdf_path: Path, cfg: "MarkerConfig") -> MarkerResult:
        if cfg.use_llm:
            raise ConverterError(
                "use_llm is not supported by the warm converter (local, no-LLM only)")
        out = self.pool.convert(pdf_path, {
            "paginate_output": cfg.paginate_output,
            "force_ocr": cfg.force_ocr,
            "output_format": cfg.output_format,
        })
        return MarkerResult(
            markdown=out.markdown,
            block_json=out.metadata,
            warnings=out.warnings,
            page_count=out.page_count,
            timings=out.timings,
            converter_version=out.converter_version,
        )
