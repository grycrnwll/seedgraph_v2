"""Code-enforced answer guarantees (plan §5 / §6.5 / §10 step 7, 08 §8;
decisions 82/62).

The guard is the deterministic backstop: nothing about faithfulness, abstention,
or leakage is left to the prompt. It runs after ``compose.generate`` and has the
final say over the envelope. Ordered duties (plan §10 step 7):

(a) **membership** — drop any citation whose ``work_id`` is not in the allowed set,
    and strip any ``span_id`` not in it; record a warning.
(b) **post-drop abstention (must-fix #1)** — if, after (a), a prose answer has an
    empty citation set (or fewer than the support floor), force
    ``insufficient_evidence=True``, ``answer_category=unresolved``, replace
    ``answer_text`` with the not-found message, and warn
    ``unsupported_prose_suppressed``. This closes the 09 §8 "all citations drop →
    unsupported prose" hole.
(c) **empty/weak short-circuit** — empty retrieval or all scores below the
    ``absent_floor`` → ``insufficient_evidence=True``, ``unresolved``, no prose, no
    LLM call (decided pre-LLM by :func:`short_circuit_if_empty`). Below
    ``weak_evidence_floor`` but present → proceed with a ``weak_evidence`` warning.
(d) **project-only leakage** — ``project_only`` + declared ``outside_corpus`` →
    coerce to ``unresolved`` (09 §9).
(e) **category reconciliation** — evidence has final say over the LLM-declared
    category (≥1 supporting span ⇒ ``source_grounded``; spans across ≥2 works with
    no single stating span ⇒ ``corpus_synthesis``; only graph support ⇒
    ``project_graph_inference``).

Floors are config constants under ``answer_generation`` (plan §8/§11); the scaffold
defaults below are the fallback the constants override.
"""

from __future__ import annotations

import sqlite3

from .types import AllowedSet, AnswerCategory, AnswerEnvelope, AnswerMode, RankedCandidate

# inclusion_status values a cited work MUST resolve to (plan §11 faithfulness
# invariant). Anything else — ``excluded`` or a missing project_documents row —
# is dropped by the guard.
_CITABLE_INCLUSION = ("included", "metadata_only")

# Scaffold defaults — real values are config constants (``cfg.answer.*``, plan §8/§11).
DEFAULT_WEAK_EVIDENCE_FLOOR = 0.15   # normalized top-rank score: below → weak_evidence warning
DEFAULT_ABSENT_FLOOR = 0.0           # at/below → treat as absent → short-circuit
DEFAULT_SUPPORT_FLOOR = 1            # min supporting citations for non-abstaining prose

# The honest not-found message substituted on forced abstention.
NOT_FOUND_MESSAGE = (
    "The project corpus does not contain evidence sufficient to answer this question."
)


def short_circuit_if_empty(
    candidates: list[RankedCandidate],
    *,
    weak_evidence_floor: float = DEFAULT_WEAK_EVIDENCE_FLOOR,
    absent_floor: float = DEFAULT_ABSENT_FLOOR,
) -> tuple[bool, list[str]]:
    """Plan §10 step 7(c) — decide the empty/weak pre-LLM short-circuit.

    Returns ``(should_short_circuit, warnings)``. ``True`` when ``candidates`` is
    empty or every score is at/below ``absent_floor`` (the harness then abstains
    with ``empty_corpus`` and makes **no** LLM call). A non-empty set whose top
    score is below ``weak_evidence_floor`` does **not** short-circuit but yields a
    ``weak_evidence`` warning to carry forward.
    """
    if not candidates:
        return True, ["empty_corpus"]
    top = max(c.rank_score for c in candidates)
    if top <= absent_floor:
        return True, ["empty_corpus"]
    warnings: list[str] = []
    if top < weak_evidence_floor:
        warnings.append("weak_evidence")
    return False, warnings


def _recompute_cited_ids(envelope: AnswerEnvelope) -> None:
    work_ids: list[str] = []
    span_ids: list[str] = []
    for citation in envelope.citations:
        if citation.work_id not in work_ids:
            work_ids.append(citation.work_id)
        for span_id in citation.span_ids:
            if span_id not in span_ids:
                span_ids.append(span_id)
    envelope.cited_work_ids = work_ids
    envelope.cited_span_ids = span_ids


def reconcile(envelope: AnswerEnvelope, allowed: AllowedSet) -> AnswerEnvelope:
    """Plan §6.1 step 11 / §10 step 7(e) — category reconciliation from evidence.

    The guard, not the LLM, has the final say on ``answer_category``: ≥1 supporting
    span ⇒ ``source_grounded``; supporting spans across ≥2 works with no single
    *stating* span (no ``assertion_status='stated'`` citation) ⇒ ``corpus_synthesis``;
    support only via work-level graph citations (no spans) ⇒ ``project_graph_inference``.
    Returns the (mutated) envelope with a reconciled category.
    """
    supporting = [c for c in envelope.citations if c.span_ids]
    if supporting:
        stated = any(c.assertion_status == "stated" for c in supporting)
        works = {c.work_id for c in supporting}
        if len(works) >= 2 and not stated:
            envelope.answer_category = AnswerCategory.corpus_synthesis
        else:
            envelope.answer_category = AnswerCategory.source_grounded
    elif envelope.citations:
        # citations exist but none carry a span -> graph/metadata-level support only.
        envelope.answer_category = AnswerCategory.project_graph_inference
    return envelope


def _drop_excluded_work_citations(
    envelope: AnswerEnvelope, conn: sqlite3.Connection, warnings: list[str]
) -> None:
    """Plan §11 faithfulness invariant, code-enforced (decision 82 — the guard has
    FINAL SAY). Drop any surviving citation whose ``work_id`` does not resolve to a
    ``project_documents`` row with ``inclusion_status`` ∈ (``included``,
    ``metadata_only``).

    This is the belt-and-suspenders layer behind the ``retrieve.py`` inclusion filter:
    a work the project marked ``excluded`` (or one with no membership row) can never
    appear in ``cited_work_ids`` — even if a stale retrieval surfaced its spans (e.g. a
    work excluded *after* extraction that still carries citable spans). Enforced from
    the evidence in code, never trusted to retrieval or the prompt.
    """
    if not envelope.citations:
        return
    kept = []
    changed = False
    for citation in envelope.citations:
        row = conn.execute(
            "SELECT inclusion_status FROM project_documents WHERE work_id = ?",
            (citation.work_id,),
        ).fetchone()
        if row is None or row[0] not in _CITABLE_INCLUSION:
            warnings.append(f"excluded_work_citation_dropped:{citation.work_id}")
            changed = True
            continue
        kept.append(citation)
    if changed:
        envelope.citations = kept
        _recompute_cited_ids(envelope)


def enforce(
    envelope: AnswerEnvelope,
    allowed: AllowedSet,
    mode: AnswerMode,
    *,
    support_floor: int = DEFAULT_SUPPORT_FLOOR,
    conn: sqlite3.Connection | None = None,
) -> AnswerEnvelope:
    """The single public guard entry (plan §6.5) — apply duties (a)-(e) in order.

    Drops out-of-set citations (membership), drops citations to ``excluded`` works
    (inclusion-status invariant, when ``conn`` is supplied), coerces ``outside_corpus``
    → ``unresolved`` under ``project_only`` (leakage), reconciles the
    ``insufficient_evidence`` flag and category **from the surviving evidence** (not the
    model's self-declaration — guard has final say, decision 82), and forces abstention
    when prose is left unsupported (must-fix #1). Recomputes ``cited_work_ids`` /
    ``cited_span_ids`` from the surviving citations and appends every action to
    ``warnings``. Returns the final, contract-safe :class:`AnswerEnvelope`.

    ``conn`` is the project connection used only to resolve ``inclusion_status`` for the
    inclusion-status invariant; omitting it (e.g. pure in-process guard unit tests)
    skips that belt-and-suspenders layer (the retrieval filter still applies upstream).
    """
    warnings = list(envelope.warnings)

    # (a) membership: drop any citation whose work is not in the allowed set; strip
    # any span_id not in it. A citation that loses ALL of its (originally present)
    # spans is dropped (it can no longer be traced to a verbatim quote).
    survivors = []
    for citation in envelope.citations:
        if citation.work_id not in allowed.work_ids:
            warnings.append(f"dropped_citation_work_not_retrieved:{citation.work_id}")
            continue
        valid_spans = [s for s in citation.span_ids if s in allowed.span_ids]
        if citation.span_ids and not valid_spans:
            warnings.append(f"dropped_citation_no_valid_span:{citation.work_id}")
            continue
        if len(valid_spans) != len(citation.span_ids):
            warnings.append(f"dropped_invented_span:{citation.work_id}")
            citation.span_ids = valid_spans
        survivors.append(citation)
    envelope.citations = survivors
    _recompute_cited_ids(envelope)

    # (a2) inclusion-status invariant (plan §11): every cited work must be
    # included/metadata_only — code-enforced, never prompt-trusted. Runs before the
    # grounding/abstention normalization so an excluded-work-only citation set correctly
    # triggers abstention below.
    if conn is not None:
        _drop_excluded_work_citations(envelope, conn, warnings)

    is_retrieval_only = mode == AnswerMode.RETRIEVAL_ONLY

    # (d) project-only leakage: an outside-corpus declaration is suppressed when the
    # project is private/closed-world. This is a terminal abstention (the normalization
    # below does not re-grade it).
    leakage_suppressed = False
    if mode == AnswerMode.PROJECT_ONLY and envelope.answer_category == AnswerCategory.outside_corpus:
        envelope.answer_category = AnswerCategory.unresolved
        envelope.insufficient_evidence = True
        envelope.answer_text = ""
        warnings.append("outside_corpus_suppressed")
        leakage_suppressed = True

    # A deliberately-labeled outside_corpus answer that survived the leakage guard
    # (i.e. mode == allow_outside) is exempt from corpus-grounding abstention and
    # reconciliation — it explicitly asserts it rests on knowledge outside the corpus.
    is_labeled_outside = envelope.answer_category == AnswerCategory.outside_corpus

    # (b)+(e) evidence-driven flag normalization (must-fix #1 + the self-contradiction
    # fix): the guard reconciles ``insufficient_evidence`` FROM the surviving evidence,
    # NOT the model's self-declaration (decision 82 — guard has final say). The envelope
    # can therefore never be internally contradictory (e.g. insufficient=true WHILE
    # carrying supporting citations + prose, or a model laundering base-model prose by
    # setting the flag). Retrieval-only, a deliberately-labeled outside_corpus answer
    # (allow_outside), and a leakage-suppressed abstention are exempt.
    if not is_retrieval_only and not is_labeled_outside and not leakage_suppressed:
        if len(envelope.citations) >= support_floor:
            # Grounded: ≥ support_floor surviving citations ⇒ the answer rests on the
            # corpus. Override any self-declared insufficient flag and reconcile the
            # category from the evidence — prose + citations are kept, flag is False.
            if envelope.insufficient_evidence:
                warnings.append("insufficient_flag_overridden")
            envelope.insufficient_evidence = False
            reconcile(envelope, allowed)
        else:
            # Below the support floor ⇒ not grounded ⇒ abstain. Suppress any unsupported
            # prose with the honest not-found banner (closes the 09 §8 hole and the
            # insufficient-flag laundering path); no banner needed when there is no prose.
            envelope.insufficient_evidence = True
            envelope.answer_category = AnswerCategory.unresolved
            if envelope.answer_text.strip():
                envelope.answer_text = NOT_FOUND_MESSAGE
                warnings.append("unsupported_prose_suppressed")

    _recompute_cited_ids(envelope)
    envelope.warnings = warnings
    return envelope
