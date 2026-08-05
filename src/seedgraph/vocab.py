"""Controlled vocabularies — the single source of truth (decisions 14/30/48).

Closed ``StrEnum`` sets for values the system fully governs (access class,
acquisition/inclusion/resolution state, provenance/epistemic origin). Open
``frozenset`` vocabularies (``CLAIM_TYPES``/``EDGE_TYPES``) for extensible kinds
with a catch-all via :func:`normalize`.

The export gate lives here too: :func:`is_shareable` (the default-deny access
allowlist), :func:`field_allowed` (the thin wrapper every export path routes
through), and :data:`PROVIDER_SHAREABLE_FIELDS` (the D8 provider-metadata
field allowlist). The SQL ``access_class IN (...)`` CHECK list is kept in sync
with :class:`AccessClass` by ``test_vocab_sql_parity``.
"""

from __future__ import annotations

from enum import StrEnum


# --- Closed StrEnums -------------------------------------------------------

class AccessClass(StrEnum):
    """Content-access boundary (doc 03 §11). Fail-closed default is the most
    restrictive value, ``user_supplied_private``."""

    open_access = "open_access"
    user_supplied_private = "user_supplied_private"
    metadata_only = "metadata_only"
    licensed_future = "licensed_future"
    unknown = "unknown"

    @classmethod
    def most_restrictive(cls, *classes) -> "AccessClass":
        """Fail-closed reconcile: return the MOST restrictive of ``classes``.

        Restrictiveness order (most -> least): ``user_supplied_private`` >
        ``unknown`` > ``licensed_future`` > ``metadata_only`` > ``open_access``
        (decisions 11/76). Any ``None`` is ignored; any value not in the enum
        fails closed to the most-restrictive ``user_supplied_private``; an empty
        / all-``None`` call also defaults to ``user_supplied_private``. Shared
        contract consumed by ``cache.ingest.reconcile_access_class`` and the
        phase_3/phase_4 span/claim access floor.
        """
        best: "AccessClass | None" = None
        best_rank = -1
        for raw in classes:
            if raw is None:
                continue
            try:
                value = cls(raw)
            except (ValueError, TypeError):
                return cls.user_supplied_private  # unknown -> fail closed
            rank = _ACCESS_RESTRICTIVENESS[value]
            if rank > best_rank:
                best_rank = rank
                best = value
        return best if best is not None else cls.user_supplied_private


# Restrictiveness ranking (most -> least). Higher number == more restrictive.
# Consumed by AccessClass.most_restrictive and cache.ingest.reconcile_access_class.
_ACCESS_RESTRICTIVENESS: dict[AccessClass, int] = {
    AccessClass.user_supplied_private: 5,
    AccessClass.unknown: 4,
    AccessClass.licensed_future: 3,
    AccessClass.metadata_only: 2,
    AccessClass.open_access: 1,
}


# ponytail: no AcquisitionState enum — phase_5b owns acquisition status as a
# derived display string from a plain ACQUISITION_STATES set (decision 1;
# phase_5b §6.3), not an enum. It defines that frozenset when built; nothing
# in Phase 0 reads it.


class InclusionStatus(StrEnum):
    """Settled membership vocabulary (doc 01 §6 / decisions 13/59). `candidate`
    is deliberately absent — only these three values are sanctioned (phase_5)."""

    included = "included"
    metadata_only = "metadata_only"
    excluded = "excluded"


class ResolutionStatus(StrEnum):
    """Reference-resolution outcome (phase_5b §10 / phase_3b)."""

    resolved = "resolved"
    unresolved = "unresolved"
    ambiguous = "ambiguous"
    suspect = "suspect"


class Provenance(StrEnum):
    """Epistemic origin of a claim/edge (D2 ``epistemic_type``, 6-tier)."""

    deterministic = "deterministic"
    metadata_resolved = "metadata_resolved"
    llm_extracted = "llm_extracted"
    llm_inferred = "llm_inferred"
    user_supplied = "user_supplied"
    user_validated = "user_validated"


# Authority / precedence for conflict resolution: higher wins.
PROVENANCE_AUTHORITY: dict[Provenance, int] = {
    Provenance.user_validated: 6,
    Provenance.user_supplied: 5,
    Provenance.deterministic: 4,
    Provenance.metadata_resolved: 3,
    Provenance.llm_extracted: 2,
    Provenance.llm_inferred: 1,
}


# --- Open vocabularies (extensible, catch-all via normalize) ---------------

CLAIM_TYPE_OTHER = "other"
CLAIM_TYPES: frozenset[str] = frozenset(
    {
        "finding",
        "method",
        "definition",
        "assumption",
        "hypothesis",
        "result",
        "limitation",
        "background",
        CLAIM_TYPE_OTHER,
    }
)

EDGE_TYPE_OTHER = "related"
EDGE_TYPES: frozenset[str] = frozenset(
    {
        "cites",
        "supports",
        "contradicts",
        "extends",
        "uses",
        "compares",
        EDGE_TYPE_OTHER,
    }
)


def normalize(value: str | None, vocabulary: frozenset[str], *, fallback: str) -> str:
    """Lower/strip ``value`` and return it if in ``vocabulary``, else ``fallback``.

    Used to map an unknown ``claim_type``/``edge_type`` onto the catch-all.
    """
    if value is None:
        return fallback
    candidate = value.strip().lower()
    return candidate if candidate in vocabulary else fallback


# --- Export gate (default-deny) --------------------------------------------

_SHAREABLE_ACCESS: frozenset[AccessClass] = frozenset(
    {AccessClass.open_access, AccessClass.metadata_only}
)


def is_shareable(access_class) -> bool:
    """Default-deny access allowlist: True only for ``open_access`` /
    ``metadata_only``. Unknown / NULL / new values fail closed to False."""
    try:
        return AccessClass(access_class) in _SHAREABLE_ACCESS
    except (ValueError, TypeError):
        return False


def field_allowed(access_class) -> bool:
    """The one export gate every export path routes through. Thin default-deny
    wrapper over :func:`is_shareable` — unknown/NULL/new → withheld (False)."""
    return is_shareable(access_class)


# D8 — provider-metadata field allowlist. Exactly the shareable provider fields;
# abstracts, snippets, TDM/licensed fields, and raw payload blobs are excluded
# by default and never appear here.
PROVIDER_SHAREABLE_FIELDS: frozenset[str] = frozenset(
    {
        # external ids
        "doi",
        "openalex_id",
        "arxiv_id",
        "s2_id",
        "ssrn_id",
        # normalized bibliographic metadata
        "title",
        "authors",
        "year",
        "venue",
        "oa_url",
        # citation graph + resolution
        "referenced_works",
        "resolution_metadata",
    }
)


def project_provider_fields(row: dict) -> dict:
    """Project a provider-cache metadata row through :data:`PROVIDER_SHAREABLE_FIELDS`,
    dropping excluded fields (abstracts/snippets/licensed/raw payload)."""
    return {key: value for key, value in row.items() if key in PROVIDER_SHAREABLE_FIELDS}


# ===========================================================================
# Wiring-stage additive extensions — phase-private vocab promoted to the shared
# foundation module (per each phase's needs_wiring). Nothing above is removed;
# AccessClass membership is unchanged (the SQL parity test still holds).
# ===========================================================================

# --- phase_1: cache identity/conversion vocab ------------------------------

class AcquisitionMethod(StrEnum):
    """How a source file entered the cache (phase_1 §7). ``open_access_fetch``
    is the only method that may set ``access_class = open_access``."""

    upload = "upload"
    folder_import = "folder_import"
    open_access_fetch = "open_access_fetch"


class FileType(StrEnum):
    """Cache source-file content type (phase_1). HTML conversion is deferred."""

    pdf = "pdf"
    html = "html"


class RunStatus(StrEnum):
    """Conversion-run lifecycle — matches ``conversion_runs.run_status`` CHECK in
    cache/0002_cache.sql."""

    pending = "pending"
    success = "success"
    failed = "failed"


class ConversionEpistemicType(StrEnum):
    """Marker-conversion epistemic tier — matches
    ``conversion_runs.conversion_epistemic_type`` CHECK in cache/0002_cache.sql."""

    local_deterministic_conversion = "local_deterministic_conversion"
    local_conversion_with_external_llm_assist = "local_conversion_with_external_llm_assist"
    local_conversion_with_local_llm_assist = "local_conversion_with_local_llm_assist"


# --- phase_5 / phase_5b: identity, inclusion, acquisition, review ----------

class IdType(StrEnum):
    """External identifier kinds (phase_5 ``identifiers.id_type``). Precedence
    ``doi > openalex > arxiv > s2 > ssrn`` lives in ``project/identity.py``."""

    doi = "doi"
    openalex = "openalex"
    arxiv = "arxiv"
    s2 = "s2"
    ssrn = "ssrn"


# Open vocabulary of inclusion reasons (decision 14 — open frozenset). Includes
# ``citation_walk`` (phase_5b). Validation is advisory; no DB CHECK.
INCLUSION_REASONS: frozenset[str] = frozenset(
    {
        "seed_document",
        "citation_walk",
        "user_added",
        "unavailable_full_text",
        "user_excluded",
    }
)

# Derived acquisition display states (phase_5b §6.3 — a plain set, NOT an enum or
# stored column; computed from bridge-row presence + reconcile state).
ACQUISITION_STATES: frozenset[str] = frozenset(
    {
        "already_cached_local",
        "available_open_access",
        "requires_user_upload",
        "metadata_only",
        "failed",
        "excluded",
    }
)

# Resolution review routing kinds (phase_5b §10 / phase_3b).
RESOLUTION_REVIEW_KINDS: frozenset[str] = frozenset(
    {
        "title_year_ambiguous",
        "cross_id_collision",
        "title_collision",
    }
)


class ReviewItemType(StrEnum):
    """``review_queue.item_type`` governance enum (free TEXT in DDL — enum-level
    governance only). Grows one member per reviewer (phases 5/3b/7)."""

    duplicate_candidate = "duplicate_candidate"
    citation_resolution = "citation_resolution"
    concept_merge_candidate = "concept_merge_candidate"
    concept_edge_candidate = "concept_edge_candidate"
    # phase_6 (decision 80): ambiguous / extraction_failed lens records + unanchorable
    # lens quotes route here as a sanctioned polymorphic-queue extension.
    lens_record = "lens_record"
    # bulk folder ingest (corpus ingest-folder): a dropped PDF that matched no pending
    # work (or matched ambiguously) routes here; approve -> new work + bridge.
    unmatched_upload = "unmatched_upload"


class ReviewAction(StrEnum):
    """Resolution action recorded on a ``review_queue`` row (phase_5/7).

    The full six-action surface the plan enumerates. ``merge`` and ``exclude`` are
    the two load-bearing resolutions for a ``duplicate_candidate`` item (the
    identity-merge flow routes cross-id / title collisions here); ``re_resolve``
    re-runs resolution. ``approve`` / ``reject`` / ``split`` serve later reviewers
    (citation resolution, concept merge). Additive only — ``AccessClass`` and the
    export gate are untouched, so ``test_vocab_sql_parity`` stays green.
    """

    approve = "approve"
    reject = "reject"
    merge = "merge"
    re_resolve = "re_resolve"
    exclude = "exclude"
    split = "split"


class ReviewStatus(StrEnum):
    """``review_queue.status`` — matches the CHECK in project/0001_foundation.sql."""

    open = "open"
    resolved = "resolved"


# --- phase_2: citation-edge provenance (distinct from D2 epistemic_type) ----

class EdgeProvenance(StrEnum):
    """Citation-edge provenance (phase_2). Deliberately named distinctly from the
    D2 ``Provenance``/``epistemic_type`` 6-tier vocabulary they do not share.

    The provenance authority ladder is the SINGLE ``citation.edges.PROVENANCE_AUTHORITY``
    (manual_override > parsed_bibliography > provider_reference); it is NOT
    duplicated here (the prior ``EDGE_PROVENANCE_AUTHORITY`` mirror was removed)."""

    provider_reference = "provider_reference"
    parsed_bibliography = "parsed_bibliography"
    manual_override = "manual_override"


class CitationEdgeType(StrEnum):
    """``citation_edges.edge_type`` governance (phase_2). Only ``cites`` is written
    this phase; the interpretive context edge types (uses/extends/criticizes) live
    on ``project_graph_edges`` in a later phase, not on ``citation_edges``."""

    cites = "cites"


# --- phase_3: section / span / anchor kinds --------------------------------

class SectionKind(StrEnum):
    """``document_sections.section_kind`` (phase_3, 0005 column comments)."""

    body = "body"
    abstract = "abstract"
    references = "references"
    appendix = "appendix"
    acknowledgments = "acknowledgments"


class SpanKind(StrEnum):
    """``evidence_spans.span_kind`` (phase_3, paragraph-only MVP)."""

    paragraph = "paragraph"
    section = "section"
    manual = "manual"


class AnchorStatus(StrEnum):
    """Verbatim re-anchor status of a stored span (phase_3, 0005)."""

    anchored = "anchored"
    stale = "stale"
    orphaned = "orphaned"


# --- phase_4 / phase_6: claim status + assertion + epistemic alias ----------

class StatusValue(StrEnum):
    """Extraction field outcome (phase_4 §6 / phase_6 lens ``FieldStatus``)."""

    found = "found"
    not_found = "not_found"
    ambiguous = "ambiguous"
    not_applicable = "not_applicable"
    extraction_failed = "extraction_failed"


# phase_6 lens_outputs.status uses the same value set (0009 comment: FieldStatus).
FieldStatus = StatusValue


class AssertionStatus(StrEnum):
    """D2 author-assertion field on claims: stated vs inferred (NULL = N/A)."""

    stated = "stated"
    inferred = "inferred"


# D2 epistemic_type origin alias — the 6-tier set already ships as ``Provenance``.
# phase_4/7/8 reference it under the name ``EpistemicType``; bind the alias rather
# than rename (non-breaking; ``Provenance`` stays valid).
EpistemicType = Provenance
EPISTEMIC_TYPE_AUTHORITY = PROVENANCE_AUTHORITY


# phase_4: extend the open CLAIM_TYPES vocabulary to the doc-05 §6 set (additive —
# the phase_0 members above are retained). Catch-all for normalization is 'claim'.
CLAIM_TYPE_CATCH_ALL = "claim"
CLAIM_TYPES = CLAIM_TYPES | frozenset(
    {
        "research_question",
        "main_contribution",
        "model",
        "data",
        "setting",
        "estimand",
        "identification_assumption",
        "regularity_condition",
        "general_assumption",
        "robustness_check",
        "open_question",
        "result",
        "limitation",
        "method",
        CLAIM_TYPE_CATCH_ALL,
    }
)


def normalize_claim_type(value: str | None) -> str:
    """Map an arbitrary claim_type onto a canonical CLAIM_TYPES member, else the
    ``'claim'`` catch-all. Passthrough for known values (NO data->dataset rename)."""
    return normalize(value, CLAIM_TYPES, fallback=CLAIM_TYPE_CATCH_ALL)


# --- phase_6: lens lifecycle -----------------------------------------------

class LensStatus(StrEnum):
    """Lens lifecycle (phase_6 §5): draft -> calibrating -> active -> archived."""

    draft = "draft"
    calibrating = "calibrating"
    active = "active"
    archived = "archived"


# object_type -> canonical claim_type aliases (decision 14/49, plan §4.4). A lens
# object_type that has a canonical CLAIM_TYPES home maps to it; the finer typed
# refinement (e.g. condition_type[]) stays only in lens_outputs.fields_json, NOT
# in claim_type. ``assumption`` is the one pinned example: it maps to the canonical
# ``general_assumption`` (NOT the bare ``assumption`` member).
_LENS_OBJECT_TYPE_ALIASES: dict[str, str] = {
    "assumption": "general_assumption",
}


def lens_claim_type(object_type: str | None, lens_id: str) -> str:
    """Map a lens ``object_type`` onto a canonical ``claim_type`` if one exists,
    else the project-scoped namespace ``lens:{lens_id}`` (plan §4.4, decision 14/49).

    The pinned rule: ``assumption`` -> ``general_assumption`` (the canonical home);
    any other object_type already present in :data:`CLAIM_TYPES` passes through; an
    object_type with no canonical home is namespaced ``lens:{lens_id}`` so the claim
    spine never collides with the default-extraction vocabulary. The finer typed
    refinement (``condition_type[]`` etc.) is NEVER folded into ``claim_type`` — it
    lives only in ``lens_outputs.fields_json``.
    """
    ot = (object_type or "").strip().lower()
    if ot in _LENS_OBJECT_TYPE_ALIASES:
        return _LENS_OBJECT_TYPE_ALIASES[ot]
    if ot in CLAIM_TYPES:
        return ot
    return f"lens:{lens_id}"


# --- phase_7: semantic graph node/edge vocab -------------------------------

class NodeType(StrEnum):
    """Closed node-type vocabulary for project_graph_edges endpoints (phase_7).
    Values mirror ``semantic.edges.NODE_HOME_TABLES`` keys."""

    Work = "Work"
    Concept = "Concept"
    Claim = "Claim"
    EvidenceSpan = "EvidenceSpan"


# Open semantic graph edge-type vocabulary (phase_7) + catch-all normalize.
GRAPH_EDGE_TYPE_OTHER = "related_to"
GRAPH_EDGE_TYPES: frozenset[str] = frozenset(
    {
        "discusses",
        "related_to",
        "broader_than",
        "narrower_than",
        "contrasts_with",
        "co_occurs_with",
        "has_alias",
        "mentions_concept",
    }
)


def normalize_graph_edge_type(value: str | None) -> str:
    """Map an arbitrary semantic edge_type onto a GRAPH_EDGE_TYPES member, else the
    ``'related_to'`` catch-all."""
    return normalize(value, GRAPH_EDGE_TYPES, fallback=GRAPH_EDGE_TYPE_OTHER)


# --- phase_8: answer-harness closed enums (relocated from answer/types.py) ---

class QueryType(StrEnum):
    """Closed query-type vocabulary (phase_8 §4). Values equal those scaffolded in
    ``answer/types.py``."""

    factual = "factual"
    comparative = "comparative"
    synthesis = "synthesis"
    gap_finding = "gap_finding"
    citation_search = "citation_search"
    concept_explanation = "concept_explanation"
    assumption_search = "assumption_search"
    evidence_request = "evidence_request"


class AnswerCategory(StrEnum):
    """Closed epistemic-category vocabulary (phase_8 §7)."""

    source_grounded = "source_grounded"
    corpus_synthesis = "corpus_synthesis"
    project_graph_inference = "project_graph_inference"
    outside_corpus = "outside_corpus"
    unresolved = "unresolved"


class AnswerMode(StrEnum):
    """Answer scope mode (phase_8 §2 / §6). Member casing follows the §6 signature
    literally (``AnswerMode.PROJECT_ONLY``)."""

    PROJECT_ONLY = "project_only"
    ALLOW_OUTSIDE = "allow_outside"
    RETRIEVAL_ONLY = "retrieval_only"
