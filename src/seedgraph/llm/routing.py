"""Per-task routing + content-access gate (doc 13 §5/§7/§13).

``resolve_route`` picks ``preferred_profile`` (falling back to
``fallback_profile`` if the preferred is unavailable), then enforces:

* the content-access gate — source text must not leave the machine for private
  full text when policy forbids it (raises :class:`ConfigError`);
* the embeddings open-access-only external-fallback gate (decision 51), a
  symmetric mirror making ``allow_external_llm: false`` truthful.

If the resolved profile is ``no_llm`` (or nothing is available) it returns a
:class:`NoLlmRoute` carrying the task's ``deterministic_fallback`` flag — the
router never fabricates output.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config.models import ProjectConfig
from ..errors import ConfigError
from ..vocab import AccessClass
from .profiles import is_local_profile, is_no_llm_profile, is_profile_available, resolve_profile

# doc 13 §13 actionable message.
_CONTENT_GATE_MESSAGE = (
    "Task '{task_type}' requires source text to leave the local machine (external "
    "profile '{profile_id}'), but this project forbids external LLM calls for "
    "restricted full text (access_class='{access_class}', "
    "content_policy.external_llm_for_private_full_text=false). Disable the external "
    "route, use a local LLM profile, pass --confirm-external, or change the project "
    "policy."
)

_EMBEDDING_GATE_MESSAGE = (
    "Task '{task_type}' would send source text to an external profile "
    "('{profile_id}'), but external embedding fallback is permitted only for "
    "access_class='{allowed}' (got '{access_class}'). Use the local embedding "
    "profile or restrict to open-access content."
)


@dataclass
class Route:
    profile_id: str
    provider: str
    access_mode: str
    external_full_text: bool
    fallback: str | None


@dataclass
class NoLlmRoute:
    task_type: str
    deterministic_fallback: bool


def resolve_route(task_type: str, access_class: str, cfg: ProjectConfig) -> Route | NoLlmRoute:
    route = cfg.llm.routes.get(task_type)
    if route is None:
        # No route configured: no LLM, and no claim of a deterministic path.
        return NoLlmRoute(task_type=task_type, deterministic_fallback=False)

    chosen_id: str | None = None
    chosen = None
    for profile_id in (route.preferred_profile, route.fallback_profile):
        if profile_id is None:
            continue
        candidate = resolve_profile(profile_id, cfg)  # raises ConfigError if unknown
        if is_no_llm_profile(candidate):
            chosen_id, chosen = profile_id, candidate
            break
        # Config validation (doc 13 §13): the profile must be authorized for this
        # task class. ``allowed_tasks is None`` means unconstrained (permissive);
        # an explicit list that omits the task makes the profile unusable here, so
        # we skip to the fallback (ultimately a NoLlmRoute if nothing is authorized).
        if candidate.allowed_tasks is not None and task_type not in candidate.allowed_tasks:
            continue
        if is_profile_available(candidate):
            chosen_id, chosen = profile_id, candidate
            break

    if chosen is None or is_no_llm_profile(chosen):
        return NoLlmRoute(task_type=task_type, deterministic_fallback=route.deterministic_fallback)

    external = not is_local_profile(chosen)
    sends_source_text = external and route.requires_source_text

    if sends_source_text:
        # Content-access gate (plan §8; decisions 30/60/76): restricted full text
        # must not leave the machine. Default-deny — EVERY class other than
        # ``open_access`` is restricted (user_supplied_private / metadata_only /
        # licensed_future / unknown, plus any unrecognized value), so an external
        # dispatch of it requires an explicit per-project confirmation
        # (``external_llm_for_private_full_text``, set by ``--confirm-external``).
        if (
            access_class != AccessClass.open_access
            and not cfg.content_policy.external_llm_for_private_full_text
        ):
            raise ConfigError(
                _CONTENT_GATE_MESSAGE.format(
                    task_type=task_type, profile_id=chosen_id, access_class=access_class
                )
            )
        # Embeddings open-access-only external-fallback gate (decision 51).
        if (
            route.external_fallback_access_class is not None
            and access_class != route.external_fallback_access_class
        ):
            raise ConfigError(
                _EMBEDDING_GATE_MESSAGE.format(
                    task_type=task_type,
                    profile_id=chosen_id,
                    allowed=route.external_fallback_access_class,
                    access_class=access_class,
                )
            )

    return Route(
        profile_id=chosen_id,
        provider=chosen.provider,
        access_mode=chosen.access_mode,
        external_full_text=sends_source_text,
        fallback=route.fallback_profile,
    )


def local_fallback_route(cfg: ProjectConfig, task_type: str) -> Route | None:
    """First usable local, non-no-LLM profile authorized for ``task_type``.

    Consolidates the previously-triplicated ``_local_fallback_route`` (extraction
    runner / chunked runner / answer composer). Used by the content-access gate to
    keep restricted source text on the machine: when the external route is refused,
    fall back to a local profile so the work proceeds locally rather than leaking.
    Returns a local :class:`Route` (``external_full_text=False``) or ``None`` (the
    caller then records the honest skipped_policy / retrieval_only degradation).
    """
    route = cfg.llm.routes.get(task_type)
    candidate_ids: list[str | None] = []
    if route is not None:
        candidate_ids = [route.fallback_profile, route.preferred_profile]
    candidate_ids += list(cfg.llm.profiles)
    seen: set[str] = set()
    for profile_id in candidate_ids:
        if profile_id is None or profile_id in seen:
            continue
        seen.add(profile_id)
        profile = cfg.llm.profiles.get(profile_id)
        if profile is None:
            continue
        if is_no_llm_profile(profile) or not is_local_profile(profile):
            continue
        if profile.allowed_tasks is not None and task_type not in profile.allowed_tasks:
            continue
        if not is_profile_available(profile):
            continue
        return Route(
            profile_id=profile_id,
            provider=profile.provider,
            access_mode=profile.access_mode,
            external_full_text=False,
            fallback=None,
        )
    return None
