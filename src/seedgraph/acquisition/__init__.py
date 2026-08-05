"""``seedgraph.acquisition`` — resolution, citation walk, OA acquisition + bridge.

This is the SOLE network-touching, ``provider_cache``-writing, corpus-growth phase
(binding r2-1; D3). Public surface (§5):

* :func:`resolve_corpus`  — metadata resolution + confidence rubric (``resolve``).
* :func:`walk_corpus`     — N-generation outbound corpus-growth walk (``walk``).
* :func:`acquire_corpus`  — OA fetch + bridge write (``acquire``).
* :func:`manual_upload`   — lawful user-supplied-PDF upload (``upload``).
* :func:`run_corpus`      — the ``corpus run`` umbrella (resolve→walk→acquire).
* :func:`resolve_work_markdown` — the ``work_source_files`` read accessor, OWNED
  and DECLARED here (NEW-B); imported by phase_3b and phase_6 from this package.
"""

from __future__ import annotations

from .bridge import resolve_work_markdown
from .resolve import resolve_corpus
from .service import acquire_corpus, import_markdown_and_bridge, manual_upload, run_corpus
from .walk import walk_corpus

__all__ = [
    "resolve_corpus",
    "walk_corpus",
    "acquire_corpus",
    "manual_upload",
    "import_markdown_and_bridge",
    "run_corpus",
    "resolve_work_markdown",
]
