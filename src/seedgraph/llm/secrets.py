"""Secret resolution — references only (doc 13 §12, ADR-0001/0002).

Resolution is a chain: ``key_source='auto'`` (the default) tries the OS keyring
then the env var; explicit ``environment`` / ``system_keyring`` sources pin one
backend. Keyring entries are named ``seedgraph/{service}/default`` and map onto
the OS store as ``service='seedgraph', username='{service}/default'``. A missing
OS keyring backend (headless CI) behaves exactly like a key miss. Secret *values*
are never logged, returned to a manifest, or written to any DB.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..errors import ConfigError

if TYPE_CHECKING:  # pragma: no cover
    from ..config.models import LLMProfile

_ENV_SOURCES = {"environment", "env"}
_KEYRING_SOURCES = {"system_keyring", "keyring"}
_KEYRING_SERVICE = "seedgraph"
_NAME_PREFIX = _KEYRING_SERVICE + "/"


def _keyring_get(name: str) -> str | None:
    """Read ``name`` (``seedgraph/{service}/default``) from the OS keyring.

    Returns ``None`` on a miss OR when no usable keyring backend exists
    (``NoKeyringError`` on headless CI) — both fall through to the env leg of the
    chain. The ``seedgraph/`` prefix is a namespace guard: a reference outside it
    is a misconfiguration, not a miss. Lazy import (httpx convention)."""
    username = _split_keyring_name(name)
    try:
        import keyring

        return keyring.get_password(_KEYRING_SERVICE, username)
    except Exception:  # noqa: BLE001 — NoKeyringError / backend faults == miss
        return None


def _split_keyring_name(name: str) -> str:
    """Validate the ``seedgraph/`` namespace guard and return the username part."""
    if not name.startswith(_NAME_PREFIX) or len(name) <= len(_NAME_PREFIX):
        raise ConfigError(
            f"invalid keyring name {name!r}: expected 'seedgraph/<service>/...' "
            "(e.g. 'seedgraph/openai/default')"
        )
    return name[len(_NAME_PREFIX):]


def _keyring_set(name: str, value: str) -> None:
    """Write ``name`` to the OS keyring. Unlike reads, a missing backend is an
    error here — a silently dropped write would report success for a secret
    that was never stored."""
    username = _split_keyring_name(name)
    try:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, username, value)
    except Exception as exc:  # noqa: BLE001 — surface backend faults actionably
        raise ConfigError(
            f"could not write {name!r} to the OS keyring: {exc} "
            "(is an OS secret store backend available on this machine?)"
        ) from exc


def _keyring_delete(name: str) -> bool:
    """Delete ``name`` from the OS keyring. Returns False when there was
    nothing to delete (absent entry or no backend) — not an error."""
    username = _split_keyring_name(name)
    try:
        import keyring

        keyring.delete_password(_KEYRING_SERVICE, username)
        return True
    except Exception:  # noqa: BLE001 — PasswordDeleteError et al: already absent
        return False


def resolve_named_secret(name: str, *, env_var: str | None = None) -> str | None:
    """Resolve a named secret: keyring ``name`` → ``env_var`` → ``None`` (ADR-0001).

    The one shared chain for non-profile secrets (acquisition provider keys:
    OpenAlex premium, S2, CORE). Never logs the value."""
    value = _keyring_get(name)
    if value is not None:
        return value
    if env_var:
        return os.environ.get(env_var)
    return None


def resolve_secret(key_source: str, reference: str) -> str | None:
    """Resolve a secret by explicit source. Returns the value or ``None`` (miss).

    ``reference`` is an env-var name for ``environment`` and a
    ``seedgraph/{service}/default`` entry name for ``system_keyring``. Profile
    resolution (including the ``auto`` chain) lives in
    :func:`resolve_profile_key`; ``'auto'`` is not valid here because a single
    reference string cannot name both legs of the chain."""
    if key_source in _ENV_SOURCES:
        return os.environ.get(reference)
    if key_source in _KEYRING_SOURCES:
        return _keyring_get(reference)
    raise ConfigError(
        f"unknown key_source {key_source!r}: expected 'environment' or 'system_keyring'"
    )


def resolve_profile_key(profile: "LLMProfile") -> str | None:
    """Resolve an LLM profile's API key. Returns the value or ``None`` (total miss).

    ``key_source='auto'`` chains keyring → env; explicit sources are
    single-backend (a pinned ``environment`` profile never touches the keyring,
    and vice versa). The keyring entry is ``profile.key_name`` or the convention
    name ``seedgraph/{provider}/default``. A total miss returns ``None`` — the
    availability machinery treats that as "key absent" (unknown ``key_source``
    still raises)."""
    source = (profile.key_source or "auto").lower()
    name = profile.key_name or f"seedgraph/{profile.provider}/default"
    if source == "auto":
        value = _keyring_get(name)
        if value is not None:
            return value
        if profile.env_var:
            return os.environ.get(profile.env_var)
        return None
    if source in _ENV_SOURCES:
        return os.environ.get(profile.env_var) if profile.env_var else None
    if source in _KEYRING_SOURCES:
        return _keyring_get(name)
    raise ConfigError(
        f"unknown key_source {source!r}: expected 'auto', 'environment', or 'system_keyring'"
    )
