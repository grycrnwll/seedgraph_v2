"""Project Model MVP (phase_5) — the per-project relational store.

Modules:

* :mod:`seedgraph.project.layout`   — slug-safe path helpers for a project dir.
* :mod:`seedgraph.project.config`   — ``project.yaml`` schema + load/write.
* :mod:`seedgraph.project.identity` — identifier normalization + conservative
  strong-id merge, incl. the cross-work identifier-collision path.
* :mod:`seedgraph.project.review`   — polymorphic ``review_queue`` surface +
  per-item_type validated payloads (decision 80).
* :mod:`seedgraph.project.service`  — ``ProjectHandle`` + create/open/add_work/
  set_inclusion_status/corpus_works (the single core both the CLI and the
  read-only API consume; decision 81).
"""

from __future__ import annotations
