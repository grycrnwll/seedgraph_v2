"""Phase 4b — chunk planner over ``document_sections`` (plan §4.0 / §4.3 / §5).

The chunk planner is the *map* half of phase_4b's map-reduce. It partitions a
paper's full markdown into context-fitting chunks so that each chunk's prompt
fits the routed model's window read from ``llm_capabilities.yaml`` (decision D9),
then the runner extracts a phase-4 default note per chunk and the reducer merges
them into ONE note (decision D4 — oversize papers are extracted, not dropped).

Determinism is the whole contract: a chunk plan is a pure function of
``(markdown_hash, section_parser_version, model_id, prompt_version, overlap)``
and is **recomputed**, never persisted as its own table — its persisted shadow is
the set of ``chunked_map`` ``extraction_runs`` rows (``chunk_index`` /
``chunk_count`` / ``chunk_section_ids``). Re-planning identical markdown therefore
yields byte-identical chunks (plan §4.3, §11 determinism test).

Sizing (plan §8 / D9): the per-chunk **input budget** is
``context_window_tokens - prompt_overhead_tokens - reserved_output_tokens`` read
from the model capability snapshot. Token counts (``est_prompt_tokens``) use the
shared ``llm.tokens.estimate_tokens`` (chars/4) — THE same estimator phase_4's
context-window gate uses, so chunk sizing and the oversize verdict agree.
``plan_chunks`` greedily packs whole
``document_sections`` under that budget; a single section that exceeds it is
paragraph-sub-split on phase-3 ``segment.paragraphs`` boundaries; a single
paragraph that *still* exceeds it is emitted as its own (over-budget) chunk so
the runner records ``run_status='skipped_oversize_section'`` for that span and
processes the rest of the paper (D4 — residual recorded, never silently dropped).

Owns no persistence, no LLM call, no second pass. Decisions implemented: D4
(oversize follow-up), D9 (capability-snapshot sizing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ..llm.tokens import estimate_tokens
from ..segment import paragraphs

if TYPE_CHECKING:  # import-only types; never evaluated at runtime (import-clean)
    from ..config.models import ModelCapability

#: chars/token used to translate the token-denominated overlap into a character
#: offset — matches ``llm.tokens.estimate_tokens`` (chars/4) so overlap and sizing
#: share one coordinate system.
_CHARS_PER_TOKEN = 4

#: sentinel for a required ``_attr`` lookup (distinguishes "missing" from a real
#: ``None`` default).
_MISSING = object()


@dataclass(frozen=True)
class Chunk:
    """One context-fitting unit of a deterministic chunk plan (plan §4.3).

    The chunk is the *read scope* for one map extraction; spans are still anchored
    against the FULL markdown (plan §4.5), so ``start_char``/``end_char`` are a
    half-open ``[start, end)`` slice into the canonical document — not a private
    coordinate system. ``est_prompt_tokens`` is the planner's estimate used for
    the per-chunk context gate (plan §8); the runner re-checks it against the live
    model budget immediately before dispatch (defense in depth against estimator
    drift). A chunk whose ``est_prompt_tokens`` exceeds the budget is an
    irreducible oversized paragraph → the runner records ``skipped_oversize_section``.
    """

    index: int
    """0-based position of this chunk in the plan."""
    section_ids: list[str]
    """``document_sections.section_id`` covered, in document order."""
    start_char: int
    """Half-open start offset into the FULL markdown."""
    end_char: int
    """Half-open end offset into the FULL markdown."""
    est_prompt_tokens: int
    """Estimated prompt tokens for this chunk (sizing + per-chunk gate input)."""


def plan_chunks(
    sections: Sequence[object],
    full_markdown: str,
    *,
    model_caps: "ModelCapability",
    prompt_overhead_tokens: int,
    reserved_output_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Partition ``full_markdown`` into a deterministic list of context-fitting chunks.

    Greedily packs whole ``document_sections`` (in document order) under the
    per-chunk input budget
    ``model_caps.context_window_tokens - prompt_overhead_tokens - reserved_output_tokens``
    (plan §8 / D9). A section larger than the budget is sub-split on phase-3
    ``segment.paragraphs`` boundaries; a single paragraph still larger than the
    budget is emitted as its own over-budget chunk (the runner then records
    ``skipped_oversize_section`` for that span and continues — D4). A fixed
    ``overlap_tokens`` of trailing text from the previous chunk is prepended to the
    next to mitigate chunk-boundary context loss (plan §12).

    Pure and deterministic: identical ``(full_markdown, sections, model_caps,
    overheads, overlap)`` yields identical chunks (plan §4.3 / §11). ``start_char``
    /``end_char`` index into ``full_markdown``; the planner never mutates state and
    never calls an LLM.

    Args:
        sections: ordered ``document_sections`` rows for the work (phase 3 owns the
            shape); each contributes its ``[start_char, end_char)`` extent.
        full_markdown: the canonical cached markdown bytes decoded to ``str``.
        model_caps: routed model's capability snapshot row (D9), providing
            ``context_window_tokens`` (and ``max_output_tokens`` for the default
            output reservation when ``reserved_output_tokens`` derives from it).
        prompt_overhead_tokens: fixed schema + instruction overhead reserved.
        reserved_output_tokens: output budget reserved from the window.
        overlap_tokens: fixed inter-chunk overlap (plan §6 default 128).

    Returns:
        The ordered chunk plan. An empty document yields ``[]``.

    Implements: D4 (chunked oversize follow-up), D9 (capability-snapshot sizing).
    """
    input_budget = (
        int(model_caps.context_window_tokens)
        - int(prompt_overhead_tokens)
        - int(reserved_output_tokens)
    )
    if input_budget < 1:
        # Degenerate window (overhead+output reserve >= the whole window): nothing
        # can fit, so every unit becomes an over-budget chunk (the runner then
        # records skipped_oversize_section, never silently dropping — D4).
        input_budget = 1

    # 1. Flatten the ordered sections into packable units (half-open offsets into
    #    the FULL markdown). A section that fits the budget is one unit; an
    #    oversized section is paragraph-sub-split (phase-3 segment.paragraphs); a
    #    paragraph that *still* exceeds the budget is its own (over-budget) unit.
    units: list[tuple[int, int, str | None]] = []  # (start, end, section_id)
    for section in _ordered(sections):
        sect_start = int(_attr(section, "start_char"))
        sect_end = int(_attr(section, "end_char"))
        section_id = _attr(section, "section_id", default=None)
        if sect_end <= sect_start:
            continue
        sect_text = full_markdown[sect_start:sect_end]
        if estimate_tokens(sect_text) <= input_budget:
            units.append((sect_start, sect_end, section_id))
            continue
        # Oversized section -> sub-split on paragraph boundaries (offsets are
        # relative to the section slice; shift back into the full document).
        paras = paragraphs(sect_text)
        if not paras:
            units.append((sect_start, sect_end, section_id))
            continue
        for p_start, p_end in paras:
            units.append((sect_start + p_start, sect_start + p_end, section_id))

    # 2. Greedily pack units (in document order) under input_budget. Token counts
    #    are taken over the contiguous [cur_start, end) slice of the FULL markdown
    #    (conservative — includes any inter-unit whitespace), so sizing is one
    #    coordinate system with the runner's per-chunk gate.
    raw_chunks: list[tuple[int, int, list[str]]] = []  # (start, end, section_ids)
    cur_start: int | None = None
    cur_end: int | None = None
    cur_secs: list[str] = []

    def _flush() -> None:
        nonlocal cur_start, cur_end, cur_secs
        if cur_start is not None and cur_end is not None:
            raw_chunks.append((cur_start, cur_end, cur_secs))
        cur_start, cur_end, cur_secs = None, None, []

    for u_start, u_end, section_id in units:
        unit_tokens = estimate_tokens(full_markdown[u_start:u_end])
        if unit_tokens > input_budget:
            # Irreducible over-budget unit (a single paragraph still too large):
            # flush the current chunk, then emit this unit as its own over-budget
            # chunk so the runner records skipped_oversize_section for that span and
            # processes the rest of the paper (D4 — residual recorded, never dropped).
            _flush()
            raw_chunks.append(
                (u_start, u_end, [section_id] if section_id is not None else [])
            )
            continue
        if cur_start is None:
            cur_start, cur_end, cur_secs = u_start, u_end, (
                [section_id] if section_id is not None else []
            )
            continue
        combined_tokens = estimate_tokens(full_markdown[cur_start:u_end])
        if combined_tokens <= input_budget:
            cur_end = u_end
            if section_id is not None and section_id not in cur_secs:
                cur_secs.append(section_id)
        else:
            _flush()
            cur_start, cur_end, cur_secs = u_start, u_end, (
                [section_id] if section_id is not None else []
            )
    _flush()

    # 3. Apply the fixed inter-chunk overlap: prepend ~``overlap_tokens`` of the
    #    previous chunk's trailing text to each subsequent chunk's read scope
    #    (mitigates chunk-boundary context loss — plan §12). Deterministic: a pure
    #    function of overlap_tokens. The overlap can never run before document
    #    start nor before the previous chunk's own start.
    overlap_chars = max(0, int(overlap_tokens)) * _CHARS_PER_TOKEN
    chunks: list[Chunk] = []
    for index, (start, end, section_ids) in enumerate(raw_chunks):
        read_start = start
        if index > 0 and overlap_chars > 0:
            prev_start = raw_chunks[index - 1][0]
            read_start = max(0, prev_start, start - overlap_chars)
        est_prompt_tokens = (
            estimate_tokens(full_markdown[read_start:end]) + int(prompt_overhead_tokens)
        )
        chunks.append(
            Chunk(
                index=index,
                section_ids=list(section_ids),
                start_char=read_start,
                end_char=end,
                est_prompt_tokens=est_prompt_tokens,
            )
        )
    return chunks


def _ordered(sections: "Sequence[object]") -> list[object]:
    """Return ``sections`` in document order (by ``ordinal`` when present)."""
    items = list(sections)
    if items and _attr(items[0], "ordinal", default=None) is not None:
        return sorted(items, key=lambda s: int(_attr(s, "ordinal")))
    return items


def _attr(obj: object, name: str, *, default: object = _MISSING):
    """Read ``name`` from a Section dataclass / row-like object (attr or index/key)."""
    if hasattr(obj, name):
        return getattr(obj, name)
    try:  # mapping-like / sqlite3.Row
        return obj[name]  # type: ignore[index]
    except (KeyError, TypeError, IndexError):
        if default is not _MISSING:
            return default
        raise AttributeError(name)
