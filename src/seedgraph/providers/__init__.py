"""``seedgraph.providers`` — the citation/metadata provider chain (phase_5b §5).

Public surface: :class:`CitationProvider` (Protocol), :class:`CircuitBreaker`,
:class:`ProviderChain` (from :mod:`seedgraph.providers.base`), and
:func:`build_default_providers` (lazy chain construction). Concrete providers
(OpenAlex / Crossref / Unpaywall / Semantic Scholar / CORE) are imported lazily
inside :func:`build_default_providers` so ``httpx`` is touched only when a chain is
actually built (decisions 39/73). Ported wholesale from v1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import CircuitBreaker, CitationProvider, ProviderChain

if TYPE_CHECKING:
    from ..config.models import GlobalConfig


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
    last; added only when a ``core_key`` is configured). SSRN is link-only and arXiv
    is handled via id-normalize + a deterministic PDF URL inside the OA gather.
    ``contact_email`` (polite pool) comes from ``config``; acquisition keys resolve
    through the one shared chain (ADR-0001): config attr -> keyring
    ``seedgraph/{service}/default`` -> env var. This is the single acquisition
    resolution seam — provider classes stay BYO-key/secret-ignorant. Concrete
    providers are imported INSIDE this function so module import never pulls
    ``httpx``. A provider whose module is not importable is skipped so the chain
    still constructs.
    """
    import os

    from ..llm.secrets import resolve_named_secret

    if os.environ.get("SEEDGRAPH_FAKE_PROVIDERS"):
        # Offline demo/verify path (mirrors SEEDGRAPH_FAKE_MARKER); never active in
        # tests, which inject their own mocks.
        from .fake import FakeProvider

        return [FakeProvider()]

    contact_email = _cfg(config, "contact_email")
    s2_api_key = _cfg(config, "s2_api_key") or resolve_named_secret(
        "seedgraph/s2/default", env_var="S2_API_KEY"
    )
    core_key = (
        _cfg(config, "core_key")
        or resolve_named_secret("seedgraph/core/default", env_var="CORE_API_KEY")
        # Legacy v1 spelling, kept as a last-resort fallback.
        or os.environ.get("SEEDGRAPH_CORE_KEY")
    )
    openalex_key = _cfg(config, "openalex_key") or resolve_named_secret(
        "seedgraph/openalex/default", env_var="OPENALEX_API_KEY"
    )

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
    "build_default_providers",
]
