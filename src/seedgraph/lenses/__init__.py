"""Lens definitions (v3 lens choice).

A *lens* is a field-specific extraction instruction with a typed output schema.

Module map:
  schema     -- LensDefinition / FieldSpec / EvidencePolicy; YAML load + validate.
  prompt     -- render the strict-JSON extraction prompt from a LensDefinition.
  templates/ -- the single built-in template, regularity_conditions_v1.yaml.

The v2 project-db lens registry was deleted in stage 4.
"""

from __future__ import annotations

__all__ = ["schema", "prompt"]
