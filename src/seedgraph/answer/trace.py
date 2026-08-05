"""AnswerTrace — the versioned capture artifact beside every AnswerEnvelope
(design 00 §3 / §4.2, decisions T1/T3/T7, C1–C5).

``harness.answer()`` computes a query's intermediate state — the deterministic
:class:`~seedgraph.answer.types.QuerySpec`, every ranked candidate with its per-boost
breakdown, which candidates the token budget showed to or dropped from the model, and
the citation-graph neighborhood — and today discards it as locals. The
:class:`AnswerTrace` persists exactly that, as ``answers/{answer_id}.trace.json`` next
to the envelope, joined to it by ``answer_id`` and never duplicating envelope content
(citations / warnings / recommendations already live there — the trace joins, it does
not copy).

**This is a measurement instrument, not free-form debug output (T1).** The schema is
versioned (:data:`TRACE_VERSION`) with a stable core and an **additive-only** evolution
rule: a new field arrives only with a version bump, existing fields never change meaning
or type, and every reader (the views, the future diff harness) MUST tolerate fields it
does not know. Writer-side models use ``extra="forbid"`` (house style) so a typo is a
loud failure here, while readers stay lenient — that asymmetry is what keeps an old
trace interpretable after the schema grows.

Dispositions are exact (:attr:`AnswerTrace.dispositions_exact` ``True``): on the LLM path
``build_prompt``'s report splits ``shown`` from ``cut_budget`` by the token-budget break
index, and ``shown_evidence`` / ``prompt_version`` / ``prompt_sha256`` are populated from
compose; off the LLM path no budget loop runs, so the labeling is unambiguous and those
fields are ``None`` (no prompt was built).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..project import layout

# Bumped only on an additive schema change (T1). Readers switch on it; writers stamp it.
TRACE_VERSION = 1

# ``text_preview`` cap (C2): a dropped candidate keeps every score/boost but truncates
# its text — the full quote lives only in ``shown_evidence`` on the LLM path.
TEXT_PREVIEW_CHARS = 160


class TraceCandidate(BaseModel):
    """One ranked candidate, disposition-tagged (C2): every candidate in the ranked
    list appears exactly once with ``disposition ∈ {shown, cut_rank, cut_budget}``.

    Scores and boosts are recorded verbatim (``boosts`` is the
    :class:`~seedgraph.answer.types.RankedCandidate` breakdown) so a trace stays
    interpretable against the ``boost_constants`` snapshot on the parent trace even
    after the weights change (C5). ``text_preview`` is truncated to
    :data:`TEXT_PREVIEW_CHARS`.
    """

    model_config = ConfigDict(extra="forbid")

    item_id: str
    kind: str                              # "span" | "claim" | "note" | "concept"
    work_id: str
    span_id: str | None = None
    concept_id: str | None = None
    section: str | None = None
    # Which truncation dropped this row (C2 — exactly three, no fourth): "cut_rank" =
    # dropped by the ``[:max_candidates]`` rank slice (position past it); "cut_budget" =
    # survived the slice but fell past the token-budget break when a prompt was built;
    # "shown" = survived both and was surfaced (into the prompt, or the retrieval-only
    # envelope's candidate surface). Off the prompt path an abstain survivor is "shown"
    # too — "nothing reached a model" is recorded by ``shown_evidence``/``outcome``, not
    # here. See ``harness._disposition``.
    disposition: str                       # "shown" | "cut_rank" | "cut_budget"
    rank_position: int                     # 1-based position in the ranked list
    rank_score: float
    bm25_score: float
    boosts: dict[str, float] = Field(default_factory=dict)
    access_class: str
    epistemic_type: str | None = None
    concept_weight: float | None = None
    concept_paper_frequency: int | None = None
    text_preview: str


class ShownEvidence(BaseModel):
    """One numbered ``[E{n}]`` evidence block as actually rendered into the prompt (T3:
    the structured shown-list, never prompt bodies). ``n`` joins to the envelope's
    citation markers; ``quote`` is the fragment post ``max_fragment_chars`` cut.

    Populated from ``build_prompt``'s report (via ``compose.generate``) on the LLM path
    and the failed-dispatch degrade; the parent's ``shown_evidence`` is ``None`` on paths
    that never build a prompt.
    """

    model_config = ConfigDict(extra="forbid")

    n: int                                 # the [E{n}] marker
    item_id: str
    work_id: str
    span_ids: list[str] = Field(default_factory=list)
    section: str | None = None
    quote: str
    token_estimate: int
    truncated: bool


class NeighborhoodTrace(BaseModel):
    """Citation-graph neighborhood membership for a ``citation_search`` ask (T7/C4):
    inputs + summary + membership only. ``nodes`` / ``edges`` are ids and id-pairs —
    titles/years are enriched at view time, never stored. Mirrors the
    ``traverse.neighborhood`` return dict field-for-field.
    """

    model_config = ConfigDict(extra="forbid")

    seeds: list[str] = Field(default_factory=list)
    unknown_seeds: list[str] = Field(default_factory=list)
    depth: int
    nodes: list[str] = Field(default_factory=list)
    edges: list[list[str]] = Field(default_factory=list)   # membership only (T7)
    coverage_ratio: float
    engulfed: bool


class AnswerTrace(BaseModel):
    """The versioned capture artifact for one answer (design 00 §3).

    Joined to its :class:`~seedgraph.answer.types.AnswerEnvelope` by ``answer_id``; the
    filename ``answers/{answer_id}.trace.json`` IS the join key (C1). ``path`` +
    ``outcome`` name which branch ran so a degrade is identifiable without inferring
    from absent fields (C6).
    """

    model_config = ConfigDict(extra="forbid")

    trace_version: int = TRACE_VERSION
    answer_id: str                         # join key to the envelope (never duplicated)
    question: str
    created_at: str
    seedgraph_version: str                 # C5 — self-describing scoring
    boost_constants: dict[str, float] = Field(default_factory=dict)   # C5
    spec: dict                             # QuerySpec.model_dump(mode="json") whole (C3)
    path: str                              # "gap_finding" | "comparative" | "standard"
    # "llm" | "retrieval_only" | "abstain_empty" | "abstain_gap_finding" (C6).
    outcome: str
    degrade_reasons: list[str] = Field(default_factory=list)
    retrieved_count: int                   # raw retrieval size (pre rank-truncation)
    # Whether shown/cut_budget/cut_rank are exact. ``harness._build_trace`` sets this True
    # on every real path (build_prompt's break index splits cut_budget from cut_rank on
    # the LLM path; off it no budget loop runs, so the labeling is unambiguous). The
    # default stays the conservative False for a bare/hand-built trace.
    dispositions_exact: bool = False
    candidates: list[TraceCandidate] = Field(default_factory=list)   # ONE table (C2)
    shown_evidence: list[ShownEvidence] | None = None   # None when no prompt was built
    prompt_version: str | None = None      # T3; None off the LLM path
    prompt_sha256: str | None = None       # T3; None off the LLM path
    neighborhood: NeighborhoodTrace | None = None       # None when traversal didn't run


def save_trace(
    trace: AnswerTrace,
    *,
    slug: str,
    root: Path | str | None = None,
    run_id: str | None = None,
) -> Path:
    """Persist ``trace`` beside its envelope (design 00 §4.2).

    Path logic mirrors ``harness.save_answer`` exactly — ad-hoc ``ask`` →
    ``projects/{slug}/answers/{answer_id}.trace.json``; inside a build run (``run_id``
    set) → ``projects/{slug}/runs/{run_id}/answers/{answer_id}.trace.json``. The
    filename is the join key; there is no DB row and no index. Written atomically
    (tempfile + ``os.replace``, the ``run.py`` durability discipline).
    """
    if run_id:
        base = layout.project_runs_dir(slug, root) / run_id / "answers"
    else:
        base = layout.project_dir(slug, root) / "answers"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{trace.answer_id}.trace.json"
    fd, tmp = tempfile.mkstemp(dir=str(base), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(trace.model_dump_json(indent=2))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path
