"""Library concepts as label + sense (v3 stage 4, docs/V3_CONCEPTS_DESIGN.md).

- :mod:`.families`: the type families (the type fence's unit);
- :mod:`.model`: label keys, sense keys, handles, ids, claim fingerprints, grounding,
  :func:`.model.resolve`;
- :mod:`.guardrails`: vetoes, seed lists, the ambiguous-label list with per-graph overrides;
- :mod:`.refresh`: the deterministic rebuild of the derived tables.
"""
