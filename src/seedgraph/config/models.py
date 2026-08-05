"""Typed configuration models (Pydantic v2).

Phase 0 validates config only — it calls no model. The defaults here give a
clean, keyless environment a working, validatable config (the five profiles from
doc 13 §4 plus sensible per-task routes), so ``doctor`` and ``resolve_route``
work out of the box; YAML files merge over these.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# --- LLM access profiles (doc 13 §4) ---------------------------------------

class LLMProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profile_id: str
    provider: str = "none"           # anthropic | openai | ollama | none
    access_mode: str = "api_key"     # api_key | local | none
    model: Optional[str] = None
    env_var: Optional[str] = None    # name of the env var holding the key (external only)
    is_local: bool = False
    # Track 1 (LLM backend): provider endpoint override (e.g. a non-default Ollama
    # host) and the secret source. ``key_source`` is the resolve_profile_key
    # discriminator (ADR-0002): ``auto`` chains OS keyring → env var; an explicit
    # source pins one backend. ``key_name`` overrides the conventional keyring
    # entry name ``seedgraph/{provider}/default``.
    base_url: Optional[str] = None
    key_source: str = "auto"         # auto | environment | system_keyring
    key_name: Optional[str] = None   # keyring entry (default seedgraph/{provider}/default)
    # doc 13 §4/§6/§13: task classes this profile is authorized for. Config
    # validation ("profile supports requested task") checks task_type membership.
    # ``None`` means unconstrained (legacy/permissive); an explicit list that
    # omits a task makes the profile unusable for that task in resolve_route.
    allowed_tasks: Optional[list[str]] = None


class TaskRoute(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_type: str
    preferred_profile: str
    fallback_profile: Optional[str] = None
    requires_source_text: bool = False
    # whether a genuine rule-based path exists when no LLM is available (decision 58)
    deterministic_fallback: bool = False
    # decision 51: external embedding fallback permitted ONLY for this access class
    external_fallback_access_class: Optional[str] = None
    # Track 1 (LLM backend) — additive policy fields. These may only NARROW the
    # content gate, never widen it (cross-cutting decision #4); resolve_route stays
    # the authoritative external-dispatch gate.
    # ``None`` ⇒ unconstrained (no per-route access-class narrowing).
    allowed_access_classes: Optional[list[str]] = None
    # may bounded private fragments leave the machine for this task (answer-boundary
    # mirror of content_policy.external_llm_for_answer_generation; defense-in-depth).
    allow_external_fragments: bool = False
    requires_structured_output: bool = False
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    # what the executor does when no usable LLM is available: "degrade" (honest
    # retrieval_only / deterministic) is the only behavior wired in the MVP.
    fallback_behavior: str = "degrade"


class ContentPolicy(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # may full source text leave the local machine to an external LLM?
    external_llm_for_private_full_text: bool = False
    allow_external_llm: bool = True
    # phase_8 (plan §7): may bounded private (``user_supplied_private``) evidence
    # FRAGMENTS leave the machine for external answer generation? Private-by-default
    # (decisions 76/30/60): False → a private feeding fragment routes to a local
    # profile if available, else the answer degrades to retrieval_only. This is
    # orthogonal to ``external_llm_for_private_full_text`` (that gates whole-document
    # source text; this gates short retrieved snippets at the answer boundary).
    external_llm_for_answer_generation: bool = False


class AnswerConfig(BaseModel):
    """phase_8 answer-harness tunables (plan §8/§11) — config constants under the
    ``answer_generation`` task. Floors make the weak-vs-absent distinction testable;
    the token budget keeps evidence a bounded set of fragments."""

    model_config = ConfigDict(extra="ignore")

    # token budgeting (plan §8): the effective evidence cap is
    # ``min(max_evidence_tokens, context_window - prompt_overhead - reserved_output)``.
    max_evidence_tokens: int = 6000
    max_fragment_chars: int = 1200
    prompt_overhead_tokens: int = 800
    reserved_output_tokens: int = 1024
    # rank-score floors over the normalized top candidate (plan §10 step 7c).
    weak_evidence_floor: float = 0.15
    absent_floor: float = 0.0
    # minimum supporting citations a non-abstaining prose answer must keep (must-fix #1).
    support_floor: int = 1


class AnswerPolicy(BaseModel):
    """Global ANCHOR for the answer-harness policy so a project can inherit it live.

    Mirrors the three declarative fields (and defaults) of
    :class:`seedgraph.project.config.AnswerPolicy`: the on-disk ``project.yaml``
    carries a single ``answer_policy`` block that both the declarative project model
    and this runtime anchor interpret. Anchoring it on :class:`GlobalConfig` lets an
    unset project field resolve to the current global default via the loader deep-
    merge (settings inheritance). ``allow_external_search_by_default`` is a
    tighten-only ceiling (see :data:`CEILING_SPEC`); the other two are free.
    """

    model_config = ConfigDict(extra="ignore")

    require_project_sources: bool = True
    allow_external_search_by_default: bool = False
    distinguish_source_claims_from_synthesis: bool = True


class LlmBudget(BaseModel):
    model_config = ConfigDict(extra="ignore")

    usd_limit: Optional[float] = None
    max_tokens: Optional[int] = None
    # fail-closed default (§8): a USD-limited task refuses to run against a model
    # whose pricing_status != "verified" unless this is explicitly set.
    allow_unverified_pricing: bool = False
    # Multi-work budget controls (phase_4 §8 BudgetState). Monthly spend accrues
    # from llm_usage_events (no new table).
    per_run_soft_limit_usd: Optional[float] = None
    monthly_soft_limit_usd: Optional[float] = None
    require_confirmation_above_usd: Optional[float] = None
    stop_on_budget_exceeded: bool = False


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    profiles: dict[str, LLMProfile] = Field(default_factory=dict)
    routes: dict[str, TaskRoute] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _routes_reference_known_profiles(self) -> "LLMConfig":
        for task_type, route in self.routes.items():
            for profile_id in (route.preferred_profile, route.fallback_profile):
                if profile_id is not None and profile_id not in self.profiles:
                    raise ValueError(
                        f"route '{task_type}' references unknown profile "
                        f"'{profile_id}' (known profiles: {sorted(self.profiles)})"
                    )
        return self


def default_profiles() -> dict[str, LLMProfile]:
    return {
        "anthropic_api_default": LLMProfile(
            profile_id="anthropic_api_default",
            provider="anthropic",
            access_mode="api_key",
            model="claude-sonnet-4-6",
            env_var="ANTHROPIC_API_KEY",
            allowed_tasks=[
                "note_extraction",
                "semantic_graph_extraction",
                "project_lens_extraction",
                "metadata_extraction",
                "answer_generation",
            ],
        ),
        # Build C D11 (gap scan §5.3): Haiku-class sibling profile for concept
        # canonicalization. Rationale ported from prototype concept_canon.py:652-657:
        # canon is a small controlled-vocab subset-selection task over pre-clustered
        # labels — Haiku-class work. The deterministic guardrails (closure, veto
        # stoplist, type fence), not the model, carry correctness, so the cheaper/
        # faster model ($1/$5 per MTok vs sonnet's $3/$15) is strictly better here;
        # every other task keeps the sonnet default. Not referenced by any default
        # route (local-first, ADR-0006) — users opt in per project via
        # ``seedgraph llm route set --task semantic_graph_extraction
        # --preferred anthropic_api_cheap``.
        "anthropic_api_cheap": LLMProfile(
            profile_id="anthropic_api_cheap",
            provider="anthropic",
            access_mode="api_key",
            model="claude-haiku-4-5",
            env_var="ANTHROPIC_API_KEY",
            allowed_tasks=["semantic_graph_extraction"],
        ),
        # ADR-0003/ADR-0005: hosted OpenAI profile, cheap-tier default model. Not
        # referenced by any default route (local-first, ADR-0006) — users opt in
        # per project via ``seedgraph llm route set``.
        "openai_api_default": LLMProfile(
            profile_id="openai_api_default",
            provider="openai",
            access_mode="api_key",
            model="gpt-5.4-mini",
            env_var="OPENAI_API_KEY",
            allowed_tasks=[
                "note_extraction",
                "semantic_graph_extraction",
                "project_lens_extraction",
                "metadata_extraction",
                "answer_generation",
            ],
        ),
        # ADR-0004/ADR-0005: hosted Gemini profile, cheap-tier default model. Like
        # openai_api_default, never referenced by a default route (ADR-0006).
        "gemini_api_default": LLMProfile(
            profile_id="gemini_api_default",
            provider="gemini",
            access_mode="api_key",
            model="gemini-3.5-flash",
            env_var="GEMINI_API_KEY",
            allowed_tasks=[
                "note_extraction",
                "semantic_graph_extraction",
                "project_lens_extraction",
                "metadata_extraction",
                "answer_generation",
            ],
        ),
        "local_ollama_default": LLMProfile(
            profile_id="local_ollama_default",
            provider="ollama",
            access_mode="local",
            model="llama3",
            is_local=True,
            allowed_tasks=[
                "note_extraction",
                "semantic_graph_extraction",
                "project_lens_extraction",
                "metadata_extraction",
                "answer_generation",
            ],
        ),
        "local_embedding_model": LLMProfile(
            profile_id="local_embedding_model",
            provider="ollama",
            access_mode="local",
            model="nomic-embed-text",
            is_local=True,
            allowed_tasks=["embeddings"],
        ),
        "no_llm": LLMProfile(
            profile_id="no_llm",
            provider="none",
            access_mode="none",
            is_local=True,
            allowed_tasks=[],
        ),
    }


def default_routes() -> dict[str, TaskRoute]:
    return {
        # genuine no-LLM capabilities (decision 58): deterministic path exists.
        "reference_parsing": TaskRoute(
            task_type="reference_parsing",
            preferred_profile="no_llm",
            requires_source_text=False,
            deterministic_fallback=True,
        ),
        # LLM-shaped task: local-first (review/decision) so private full text never
        # leaves the machine by default; the hosted Anthropic profile is the runtime
        # fallback when Ollama is down (the content gate is re-applied on that hop).
        "note_extraction": TaskRoute(
            task_type="note_extraction",
            preferred_profile="local_ollama_default",
            fallback_profile="anthropic_api_default",
            requires_source_text=True,
            deterministic_fallback=False,
        ),
        # semantic graph (concept) extraction (review #3): local-first, with a
        # genuine deterministic fallback (anchor/embedding clustering) when no LLM
        # is available. Reuses the existing task_type — no rename to concept_proposal.
        "semantic_graph_extraction": TaskRoute(
            task_type="semantic_graph_extraction",
            preferred_profile="local_ollama_default",
            fallback_profile="no_llm",
            requires_source_text=True,
            deterministic_fallback=True,
        ),
        # embeddings: local-preferred; external fallback only for open_access (decision 51).
        "embeddings": TaskRoute(
            task_type="embeddings",
            preferred_profile="local_embedding_model",
            requires_source_text=True,
            deterministic_fallback=False,
            external_fallback_access_class="open_access",
        ),
        # post-conversion bibliographic metadata backfill: local-first so a private
        # paper's first-page text never leaves the machine; degrades cleanly to the
        # deterministic regex-only id path (no_llm) when no LLM is available.
        "metadata_extraction": TaskRoute(
            task_type="metadata_extraction",
            preferred_profile="local_ollama_default",
            fallback_profile="no_llm",
            requires_source_text=True,
            deterministic_fallback=True,
            # A thinking model (qwen3) needs headroom beyond the JSON answer or it
            # spends the whole budget on hidden reasoning and emits empty content
            # (must-fix #3c). 1024 was too tight; 2048 leaves room for the object.
            max_output_tokens=2048,
        ),
        # phase_6 project lenses: a sibling extraction task (decision 22). Local-
        # preferred so private full text never leaves the machine; the no-LLM
        # deterministic anchor/FTS fallback is honest (decision 38).
        "project_lens_extraction": TaskRoute(
            task_type="project_lens_extraction",
            preferred_profile="local_ollama_default",
            fallback_profile="no_llm",
            requires_source_text=True,
            deterministic_fallback=True,
        ),
        # phase_8 answer composer (plan §8): external-preferred, no genuine
        # deterministic prose path (no_llm => retrieval_only honest degrade, 58/38).
        # requires_source_text=False — only bounded FRAGMENTS are sent, never full
        # text, so the resolve_route full-text gate does not fire; the per-fragment
        # private-content gate lives in answer/compose.py (plan §7).
        "answer_generation": TaskRoute(
            task_type="answer_generation",
            preferred_profile="anthropic_api_default",
            fallback_profile="no_llm",
            requires_source_text=False,
            deterministic_fallback=False,
        ),
        # phase_8 deterministic reranking (decision 57): zero LLM calls. Mapped to
        # no_llm with a genuine deterministic path so the doc 13 §5/§6 routing table
        # is complete; resolve_route is never actually consulted for it.
        "reranking": TaskRoute(
            task_type="reranking",
            preferred_profile="no_llm",
            requires_source_text=False,
            deterministic_fallback=True,
        ),
    }


def default_llm_config() -> LLMConfig:
    return LLMConfig(profiles=default_profiles(), routes=default_routes())


# --- Top-level config ------------------------------------------------------

class GlobalConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    home: Optional[str] = None
    log_level: str = "INFO"
    # Polite-pool identifier consumed by the provider chain (OpenAlex / Crossref /
    # Unpaywall; Unpaywall requires it). Plain str — no email validation.
    contact_email: Optional[str] = None
    llm: LLMConfig = Field(default_factory=default_llm_config)
    content_policy: ContentPolicy = Field(default_factory=ContentPolicy)
    budget: LlmBudget = Field(default_factory=LlmBudget)
    answer: AnswerConfig = Field(default_factory=AnswerConfig)
    # Global anchor for the answer-harness policy so a project inherits it live
    # (settings inheritance). ``project.yaml`` carries the same ``answer_policy``
    # block that the declarative project model owns; here it is the inheritable
    # default with a tighten-only ceiling on ``allow_external_search_by_default``.
    answer_policy: AnswerPolicy = Field(default_factory=AnswerPolicy)
    # Build D chunk 11 (gap scan §4.3): institutional lawful-access LINK config,
    # consumed ONLY by acquisition/links.py (links only — never a fetch path; the
    # separation invariant is asserted in tests/test_links.py). Both are NON-secret
    # public URLs, so they live here in config.yaml rather than env vars (v1's
    # choice) or the keyring (ADR-0001 — keyring is for secrets only). Plain
    # strings, no URL validation (decision-1 style):
    #   openurl_resolver — the institution's OpenURL resolver base URL.
    #   ezproxy_host     — a user-pasted EZproxy template carrying ``{url}`` or
    #                      ``{doi}``, or a bare host for the ``login?url=`` form.
    openurl_resolver: Optional[str] = None
    ezproxy_host: Optional[str] = None


class ProjectConfig(GlobalConfig):
    slug: str


# --- Settings-inheritance ceilings (single source of truth) ----------------
#
# Global settings act as project DEFAULTS with LIVE inheritance; a project may
# override a field, but a subset of fields are CEILINGS: an override may only make
# the setting MORE restrictive than the global default, never looser. This
# declarative map (dotted runtime-config field -> rule) is the ONE source of truth
# consumed by both ``config.loader.clamp_project_ceiling`` (runtime defense-in-depth
# on every load) and ``config.loader.write_project_overrides`` (pre-write
# validation). Dotted paths address the already-deep-merged runtime dict; ``*``
# matches every key at that level (per route). Rules:
#
#   * ``"and"``    — boolean permission gate (True == more permissive): effective =
#                    global AND project, so a project can never flip a global-False
#                    gate back to True (privacy bools; allow_unverified_pricing).
#   * ``"min"``    — numeric ceiling (``None`` == unlimited/no cap): effective =
#                    min(global, project); a project value above the global cap is
#                    clamped down. A global ``None`` imposes no ceiling.
#   * ``"subset"`` — list allow-set (``None`` == unconstrained/universal): effective
#                    = project ∩ global, so a project may only NARROW the set, never
#                    widen it; a global ``None`` lets the project narrow freely.
#
# Everything NOT listed here is a free override (routing profile CHOICE,
# ``answer.*`` tunables, ``answer_policy.require_project_sources`` /
# ``distinguish_source_claims_from_synthesis``): the plain deep-merge wins.
CEILING_SPEC: dict[str, str] = {
    # privacy gates (content policy) — project may only tighten
    "content_policy.external_llm_for_private_full_text": "and",
    "content_policy.allow_external_llm": "and",
    "content_policy.external_llm_for_answer_generation": "and",
    # budget ceilings — project may only lower a cap (or keep global's)
    "budget.usd_limit": "min",
    "budget.max_tokens": "min",
    "budget.per_run_soft_limit_usd": "min",
    "budget.monthly_soft_limit_usd": "min",
    "budget.require_confirmation_above_usd": "min",
    "budget.allow_unverified_pricing": "and",
    # answer policy ceiling — external search may only be narrowed, never widened
    "answer_policy.allow_external_search_by_default": "and",
    # per-route access-class allow-set — project must be a subset (intersection)
    "llm.routes.*.allowed_access_classes": "subset",
}


# --- LLM capability / pricing snapshot (D9) --------------------------------

class ModelCapability(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str
    context_window_tokens: int
    max_output_tokens: Optional[int] = None
    input_usd_per_mtok: Optional[float] = None
    output_usd_per_mtok: Optional[float] = None
    supports_structured_output: Optional[bool] = None
    access_modes: list[str] = Field(default_factory=list)
    pricing_status: str


class LlmCapabilities(BaseModel):
    model_config = ConfigDict(extra="ignore")

    snapshot_date: date
    schema_version: int = 1
    models: dict[str, ModelCapability]
