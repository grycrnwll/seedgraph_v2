"""Profile resolution + availability (doc 13 §4).

Profiles are config objects (no secrets stored). A *local* profile (Ollama /
no-LLM) is always available — ``doctor`` does not ping it. An *external* profile
is available iff its key resolves through the keyring → env chain (ADR-0002).
"""

from __future__ import annotations

from ..config.models import LLMProfile, ProjectConfig
from ..errors import ConfigError
from .backend import SUPPORTED_PROVIDERS
from .secrets import resolve_profile_key

# Providers / access modes that need no external key to be "available".
_LOCAL_PROVIDERS = {"ollama", "mailbox", "none", "local"}
_NO_BACKEND_MODES = {"local", "none"}

# ``SUPPORTED_PROVIDERS`` (review #1) is owned by ``llm.backend`` (the lazy provider
# registry) and re-exported here for callers that import it from ``profiles``. An
# external profile is "available" only if its provider is in this set AND its key
# is present — so a user-added OpenAI profile (no shipped adapter) stays
# unavailable and is never silently selected by resolve_route.
__all__ = [
    "SUPPORTED_PROVIDERS",
    "is_local_profile",
    "is_no_llm_profile",
    "is_profile_available",
    "resolve_profile",
]


def is_local_profile(profile: LLMProfile) -> bool:
    return (
        profile.is_local
        or profile.access_mode in _NO_BACKEND_MODES
        or profile.provider in _LOCAL_PROVIDERS
    )


def is_no_llm_profile(profile: LLMProfile) -> bool:
    return profile.provider == "none" or profile.access_mode == "none"


def resolve_profile(profile_id: str, cfg: ProjectConfig) -> LLMProfile:
    profile = cfg.llm.profiles.get(profile_id)
    if profile is None:
        raise ConfigError(
            f"unknown profile {profile_id!r} (known: {sorted(cfg.llm.profiles)})"
        )
    return profile


def is_profile_available(profile: LLMProfile) -> bool:
    """True if the profile can be used right now.

    Local profiles are always available (no endpoint validation in Phase 0).
    External profiles are available iff (a) their provider has a shipped backend
    adapter (``SUPPORTED_PROVIDERS``, review #1) AND (b) their key resolves
    through the keyring → env chain (ADR-0002). An unimplemented provider is
    never "available" even with a key set, so it can never be silently selected.
    """
    if is_local_profile(profile):
        return True
    # Registry-aware: refuse a provider whose adapter has not shipped.
    if profile.provider not in SUPPORTED_PROVIDERS:
        return False
    return resolve_profile_key(profile) is not None
