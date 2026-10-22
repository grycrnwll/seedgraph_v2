"""PDF conversion through warm converters in the separate converter environment.

The app never imports Marker or torch. It sends a PDF path to a long-lived worker
(``converter/``, ``sg_converter.worker``) over a pipe and gets markdown + page info
back; the app saves the result. See docs/CONVERTER_PROTOCOL.md.

Interface::

    from seedgraph.conversion import get_pool
    out = get_pool().convert(pdf_path, {"force_ocr": False}, timeout=None)
    out.markdown, out.page_count, out.timings, out.converter_version, out.gpu

:class:`WarmMarkerBackend` adapts the pool to the cache's ``MarkerBackend`` seam
(``convert_source_file``'s default backend).
"""

from .backend import WarmMarkerBackend
from .pool import (
    ConverterPool,
    discover_gpus,
    get_pool,
    locked_marker_version,
    set_pool,
)
from .warm import (
    DEFAULT_OPTIONS,
    ConverterCrashed,
    ConverterError,
    ConvertOutput,
    WarmConverter,
)

__all__ = [
    "DEFAULT_OPTIONS",
    "ConvertOutput",
    "ConverterCrashed",
    "ConverterError",
    "ConverterPool",
    "WarmConverter",
    "WarmMarkerBackend",
    "discover_gpus",
    "get_pool",
    "locked_marker_version",
    "set_pool",
]
