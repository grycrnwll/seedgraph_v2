"""Phase 7 — Semantic graph overlay (public package entry).

Normalizes concepts across the project corpus, connects them to the
already-extracted claims/papers/evidence, persists durable relational rows in
``project.db``, and exposes a computed NetworkX view + file export. The graph is
a retrieval/organization layer, never a truth source (doc 07 §1, §13).

Primary success criterion (doc 10 §11): *select a concept → see its linked
papers, claims, and evidence spans* — resolved by the relational path
``concepts → claim_concepts → extracted_claims → claim_spans → evidence_spans``
plus ``concepts → project_graph_edges (Work --discusses--> Concept)``.

Binding decisions implemented by this package:
  * **D2** — the origin field is named ``epistemic_type`` on every edge/concept
    row; ``assertion_status`` is claim-only and absent here.
  * **D6** — schema authority is ``db/schema/project/0010_semantic_overlay.sql``;
    nothing here authors or evolves schema.
  * **D8** — export routes through the phase_0 default-deny gate
    (``vocab.is_shareable`` / ``vocab.field_allowed``); see :mod:`.access`.
  * **D10** — concept canonicalization is an AUTO tier: exact-label + acronym
    folds auto-apply, borderline LLM folds route to ``review_queue``, and the
    anti-overmerge guard (``BERT-base`` ≠ ``BERT-large``, ``SQuAD v1.1`` ≠
    ``SQuAD v2.0``) always wins. Perfect merges are NOT required.

Scaffold status: the public surface and signatures are real; bodies raise
``NotImplementedError`` pending the full build.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing config at runtime; annotation only.
    from ..config.models import LLMProfile, ProjectConfig
    from ..llm.backend import LLMBackend

__all__ = ["SemanticBuildReport", "build_semantic_overlay"]


@dataclass(frozen=True)
class SemanticBuildReport:
    """Outcome of one ``concepts build`` run.

    ``concept_mode`` is one of ``'llm'`` (LLM proposer participated),
    ``'deterministic'`` (exact-key + acronym + co-occurrence only, no synonym
    folds, no ``provisional`` rows), or ``'none'`` (zero extracted claims → the
    overlay is honestly empty). ``access_class_summary`` counts written concepts
    by their stamped ``access_class`` (for the run manifest, doc 07 §4.5).
    """

    run_id: str
    concept_mode: str
    concepts_written: int = 0
    aliases_written: int = 0
    claim_concepts_written: int = 0
    discusses_edges: int = 0
    interpretive_edges: int = 0
    co_occurs_edges: int = 0
    review_items_enqueued: int = 0
    access_class_summary: dict[str, int] = field(default_factory=dict)


def build_semantic_overlay(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    profile: "LLMProfile | None",
    tau: float = 0.6,
    min_shared: int = 2,
    config: "ProjectConfig | None" = None,
    backend: "LLMBackend | None" = None,
) -> SemanticBuildReport:
    """Build (or idempotently rebuild) the semantic overlay for one project.

    Orchestrates the full pipeline over the current ``extracted_claims`` set:
    gather type-scoped ``normalized_label``s → apply sticky
    ``concept_constraints`` (must/cannot-link) BEFORE clustering → deterministic
    candidate clustering (token-Jaccard ≥ ``tau`` ∨ acronym) → optional LLM
    synonym proposal (``profile`` is ``None`` or a no-LLM profile ⇒ deterministic
    mode, no fuzzy folds) → three deterministic guardrails dispose →
    relabel-then-one-merge → upsert ``concepts`` (preserving ``user_confirmed`` /
    ``user_split`` status and any ``concept_constraints``) + ``concept_aliases``
    → write ``claim_concepts`` → build ``Work --discusses--> Concept`` and
    ``Concept--Concept`` interpretive / ``co_occurs_with`` edges → enqueue
    borderline folds and interpretive edges to ``review_queue`` → stamp
    ``run_id`` and the ``access_class`` resolved from contributing claims.

    Idempotent and append-only-recomputable (doc 07 §7): unchanged labels keep
    their deterministic ``concept_id`` and any user status survives a rebuild.

    Implements D2 (``epistemic_type``), D8 (export gate), and D10 (anti-overmerge
    + borderline→review). Fails closed if ``extracted_claims.access_class`` is
    absent (no cache.db fallback walk; doc 07 §7, §9).
    """
    import dataclasses

    from . import concepts as concepts_mod
    from . import edges as edges_mod
    from .llm_propose import make_proposer

    proposer = make_proposer(profile, config=config, backend=backend)
    report = concepts_mod.build_concepts(
        conn, run_id=run_id, proposer=proposer, tau=tau
    )
    if report.concept_mode == "none":
        conn.commit()
        return report

    # Always build the deterministic structural overlay: Work--discusses-->Concept
    # (aggregated from claim_concepts) + Concept--co_occurs_with-->Concept. The
    # interpretive Concept--Concept edges are built only when an edge proposer is
    # supplied (not wired offline); they NEVER touch citation_edges.
    discusses = edges_mod.build_discusses_edges(conn, run_id=run_id)
    co_occurs = edges_mod.build_co_occurs_edges(
        conn, run_id=run_id, min_shared=min_shared
    )
    conn.commit()
    return dataclasses.replace(
        report, discusses_edges=discusses, co_occurs_edges=co_occurs
    )
