"""``seedgraph.providers`` — the citation/metadata provider chain (phase_5b §5).

Public surface: :class:`CitationProvider` (Protocol), :class:`CircuitBreaker`,
:class:`ProviderChain` (from :mod:`seedgraph.providers.base`), and
:func:`build_default_providers` (lazy chain construction). Concrete providers
(OpenAlex / Crossref / Unpaywall / Semantic Scholar / CORE) are imported lazily
inside :func:`build_default_providers` so ``httpx`` is touched only when a chain is
actually built (decisions 39/73). Ported wholesale from v1.

Scholarly-API keys come from environment variables only (stage 2; the keyring store
is gone): see :data:`SCHOLARLY_ENV_VARS`. ``seedgraph doctor`` lists them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from .base import CircuitBreaker, CitationProvider, ProviderChain

if TYPE_CHECKING:
    from ..config.models import GlobalConfig

#: Environment variable -> what it is for. Keys are read from the environment only.
SCHOLARLY_ENV_VARS: dict[str, str] = {
    "SEEDGRAPH_CONTACT_EMAIL": "polite-pool contact email for OpenAlex, Crossref and Unpaywall "
                               "(falls back to contact_email in config.yaml; not a secret)",
    "S2_API_KEY": "Semantic Scholar API key (optional; raises the rate limit)",
    "OPENALEX_API_KEY": "OpenAlex API key (optional)",
    "CORE_API_KEY": "CORE API key (CORE joins the chain only when set; legacy name SEEDGRAPH_CORE_KEY)",
}


def env_key(name: str) -> str | None:
    """The value of environment variable ``name``, stripped, or ``None`` when unset/blank."""
    value = os.environ.get(name)
    return value.strip() or None if value else None


def _cfg(config, key: str):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(key)
    val = getattr(config, key, None)
    if val is None:
        cp = getattr(config, "content_policy", None)
        val = getattr(cp, key, None) if cp is not None else None
    return val or None


def build_default_providers(config: "GlobalConfig") -> list[CitationProvider]:
    """Construct the default ordered provider chain (lazy-import; ported v1).

    Order: OpenAlex -> Crossref -> Unpaywall -> Semantic Scholar -> CORE (BYO-key,
    last; added only when a CORE key is set). SSRN is link-only and arXiv is handled
    via id-normalize + a deterministic PDF URL inside the OA gather. Keys come from
    the environment only (:data:`SCHOLARLY_ENV_VARS`); the contact email from
    ``SEEDGRAPH_CONTACT_EMAIL``, else ``contact_email`` in ``config``. This is the single
    acquisition resolution seam — provider classes stay key-ignorant. Concrete
    providers are imported INSIDE this function so module import never pulls
    ``httpx``. A provider whose module is not importable is skipped so the chain
    still constructs.
    """
    if os.environ.get("SEEDGRAPH_FAKE_PROVIDERS"):
        # Offline demo/verify path (mirrors SEEDGRAPH_FAKE_MARKER); never active in
        # tests, which inject their own mocks.
        from .fake import FakeProvider

        return [FakeProvider()]

    contact_email = env_key("SEEDGRAPH_CONTACT_EMAIL") or _cfg(config, "contact_email")
    s2_api_key = env_key("S2_API_KEY")
    # SEEDGRAPH_CORE_KEY: the legacy v1 spelling, kept as a fallback.
    core_key = env_key("CORE_API_KEY") or env_key("SEEDGRAPH_CORE_KEY")
    openalex_key = env_key("OPENALEX_API_KEY")

    providers: list[CitationProvider] = []
    try:
        from .openalex import OpenAlexProvider

        providers.append(OpenAlexProvider(contact_email=contact_email, api_key=openalex_key))
    except Exception:  # noqa: BLE001
        pass
    try:
        from .crossref import CrossrefProvider

        providers.append(CrossrefProvider(contact_email=contact_email))
    except Exception:  # noqa: BLE001
        pass
    try:
        from .unpaywall import UnpaywallProvider

        providers.append(UnpaywallProvider(contact_email=contact_email))
    except Exception:  # noqa: BLE001
        pass
    try:
        from .semantic_scholar import SemanticScholarProvider

        providers.append(SemanticScholarProvider(s2_api_key=s2_api_key, contact_email=contact_email))
    except Exception:  # noqa: BLE001
        pass
    if core_key:
        try:
            from .core import CoreProvider

            providers.append(CoreProvider(core_key=core_key, contact_email=contact_email))
        except Exception:  # noqa: BLE001
            pass
    return providers


__all__ = [
    "CitationProvider",
    "CircuitBreaker",
    "ProviderChain",
    "SCHOLARLY_ENV_VARS",
    "build_default_providers",
    "env_key",
]
