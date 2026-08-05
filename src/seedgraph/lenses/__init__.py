"""Project-specific extraction lenses (phase_6).

A *lens* is a user-added, project-scoped reading module: a field-specific
extraction instruction + typed output schema + positive/negative anchors +
evidence policy, run as a *sibling extraction schema* over the corpus
(decision 22). It produces typed, evidence-linked, project-scoped claims plus
explicit not-found records, reusing the phase_4 extraction plumbing and the
phase_3/phase_5 span-anchoring path — no bespoke storage machinery (decision 49).

Module map (plan §5):
  schema     -- LensDefinition / FieldSpec / EvidencePolicy; YAML load + validate.
  registry   -- discover YAML, compute definition_hash, sync the `lenses` row,
                status transitions + snapshot-on-freeze.
  prompt     -- render the strict-JSON LLM extraction prompt from a LensDefinition.
  runner     -- one lens pass: resolve work->markdown -> idempotency guard ->
                extraction_run -> records -> spans -> claims -> lens_outputs.
  fallback   -- deterministic (no-LLM) anchor/FTS candidate-span matcher.
  calibrate  -- sample run + FP/FN review report assembly (the revise loop).
  results    -- query helpers: all outputs for a lens, stale set, coverage.
  templates/ -- the single built-in template, regularity_conditions_v1.yaml.

Submodules are imported explicitly by callers; this package init intentionally
performs no heavy imports so the package stays cheap and import-clean.
"""

from __future__ import annotations

__all__ = [
    "schema",
    "registry",
    "prompt",
    "runner",
    "fallback",
    "calibrate",
    "results",
]
