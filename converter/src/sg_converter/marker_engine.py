"""The real engine: Marker 1.x with its models loaded once per process."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version


class MarkerEngine:
    def __init__(self) -> None:
        from marker.models import create_model_dict

        self._artifacts = create_model_dict()

    def info(self) -> dict:
        import torch

        device = None
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
        return {
            "version": _pkg_version("marker-pdf"),
            "torch": torch.__version__,
            "cuda_device": device,
        }

    def convert(self, pdf_path: str, options: dict) -> dict:
        from marker.converters.pdf import PdfConverter
        from marker.output import text_from_rendered

        # Same config keys v2's in-process backend passed, so output matches v2.
        config = {
            "output_format": options.get("output_format", "markdown"),
            "paginate_output": bool(options.get("paginate_output", True)),
            "force_ocr": bool(options.get("force_ocr", False)),
            "use_llm": False,
            "redo_inline_math": False,
        }
        converter = PdfConverter(artifact_dict=self._artifacts, config=config)
        rendered = converter(pdf_path)
        markdown, _ext, _images = text_from_rendered(rendered)
        meta = getattr(rendered, "metadata", None)
        return {
            "markdown": markdown,
            "page_count": _page_count(pdf_path),
            "metadata": meta if isinstance(meta, dict) else {},
            "warnings": [],
        }


def _page_count(pdf_path: str) -> int:
    """Pages in the PDF itself (not inferred from the markdown)."""
    import pypdfium2

    doc = pypdfium2.PdfDocument(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()
