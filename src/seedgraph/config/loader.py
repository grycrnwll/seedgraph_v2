"""Load + merge + validate config and the bundled capability snapshot.

Merge order (project overrides win): built-in defaults → global ``config.yaml``
→ project ``project.yaml``. Each is a shallow-recursive dict merge; validation
runs once on the merged result and any failure is surfaced as a
:class:`ConfigError` with field-level detail.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError as PydanticValidationError

from .. import paths
from ..errors import ConfigError
from .models import CEILING_SPEC, GlobalConfig, LlmCapabilities, ProjectConfig

_BUNDLED_CAPABILITIES = Path(__file__).resolve().parent.parent / "llm_capabilities.yaml"

#: Key NAMES that would carry a RAW secret value if present in a config patch. We
#: store only env-var *references* (``env_var`` / ``key_source``), never the secret
#: itself (plan cross-cutting #5 / Track 2 setup); a patch carrying any of these is
#: rejected so a raw key can never be written into ``config.yaml``.
_FORBIDDEN_SECRET_KEYS: frozenset[str] = frozenset(
    {"api_key", "apikey", "secret", "secret_key", "password", "access_token", "token"}
)

#: Profile-DEFINITION field NAMES (endpoint + credential routing). Per the decided
#: settings-inheritance semantics, API keys and endpoints stay MACHINE-GLOBAL: a
#: project may only CHOOSE a profile via ``llm.routes.<task>.preferred_profile`` /
#: ``fallback_profile`` — it may never DEFINE or REDIRECT one. ``profiles`` is the
#: whole profile registry; ``provider`` / ``env_var`` / ``base_url`` / ``key_source``
#: / ``key_name`` are the per-profile endpoint + credential-routing fields. Any of
#: these appearing under a project override's ``llm`` subtree would let a project
#: bind the machine's real API key to an attacker endpoint (SSRF / key
#: exfiltration), so they are rejected on the write path AND stripped on the live
#: load path (defense-in-depth).
_FORBIDDEN_PROFILE_FIELDS: frozenset[str] = frozenset(
    {"profiles", "provider", "env_var", "base_url", "key_source", "key_name"}
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` over ``base`` (dicts merge, others replace)."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _defaults_dict() -> dict[str, Any]:
    return GlobalConfig().model_dump()


def _ceiling_targets(
    global_node: Any, merged_node: Any, parts: list[str]
) -> "Iterator[tuple[Any, dict[str, Any], str]]":
    """Yield ``(global_leaf, merged_parent, key)`` for every concrete leaf the dotted
    ``parts`` path selects, descending ``global_node`` and ``merged_node`` in
    lockstep. ``*`` matches every key present in the merged mapping at that level
    (the per-route wildcard). Robust to a missing/absent section on either side —
    it simply yields nothing where the merged path does not resolve to a mapping."""
    if not parts:
        return
    head, rest = parts[0], parts[1:]
    if head == "*":
        if isinstance(merged_node, dict):
            for k, mv in merged_node.items():
                gv = global_node.get(k) if isinstance(global_node, dict) else None
                if rest:
                    yield from _ceiling_targets(gv, mv, rest)
                else:
                    yield (gv, merged_node, k)
        return
    if not rest:
        if isinstance(merged_node, dict) and head in merged_node:
            gv = global_node.get(head) if isinstance(global_node, dict) else None
            yield (gv, merged_node, head)
        return
    mv = merged_node.get(head) if isinstance(merged_node, dict) else None
    if isinstance(mv, dict):
        gv = global_node.get(head) if isinstance(global_node, dict) else None
        yield from _ceiling_targets(gv if isinstance(gv, dict) else {}, mv, rest)


def _apply_ceiling_rule(rule: str, global_val: Any, merged_val: Any) -> Any:
    """Return the tighten-only clamped leaf value for one ``CEILING_SPEC`` rule."""
    if rule == "and":
        # boolean permission gate: effective = global AND project (True == looser).
        # Fail CLOSED: bool(None) is False, so a None/absent global ceiling (only
        # reachable via a malformed config.yaml that load_global_config would itself
        # reject) clamps the project to the most-restrictive value — never loosened.
        return bool(global_val) and bool(merged_val)
    if rule == "min":
        # numeric ceiling; None == unlimited / no cap.
        if global_val is None:
            return merged_val  # global imposes no ceiling; any project cap is tighter
        if merged_val is None:
            return global_val  # project 'unlimited' would loosen -> clamp to global
        if isinstance(merged_val, (int, float)) and isinstance(global_val, (int, float)):
            return global_val if merged_val > global_val else merged_val
        return merged_val
    if rule == "subset":
        # list allow-set; None == unconstrained / universal.
        if global_val is None:
            return merged_val  # global unconstrained; project may narrow freely
        if merged_val is None:
            return list(global_val)  # project 'unconstrained' would widen -> clamp
        if isinstance(merged_val, list):
            allowed = list(global_val)
            return [x for x in merged_val if x in allowed]
        return merged_val
    return merged_val


def clamp_project_ceiling(
    global_dict: dict[str, Any], merged_dict: dict[str, Any]
) -> dict[str, Any]:
    """Enforce the tighten-only ceilings (:data:`models.CEILING_SPEC`) on an already
    deep-merged runtime config.

    ``merged_dict`` is ``defaults ⊕ global ⊕ project`` (what the project WOULD get);
    ``global_dict`` is the effective GLOBAL config ``defaults ⊕ global`` that serves
    as the ceiling. A project override may only make a setting MORE restrictive; any
    value that would loosen the global default is clamped back to the global value
    IN PLACE (privacy bools -> AND, budget caps -> min, access-class lists ->
    intersection). Operates on plain dicts (pre-validation), mutates + returns
    ``merged_dict``, and is robust to missing sections. Free overrides (anything not
    in ``CEILING_SPEC``) are left exactly as the deep-merge produced them.
    """
    for dotted, rule in CEILING_SPEC.items():
        for global_val, parent, key in _ceiling_targets(
            global_dict, merged_dict, dotted.split(".")
        ):
            parent[key] = _apply_ceiling_rule(rule, global_val, parent[key])
    return merged_dict


def load_global_config(root: Path | str | None = None) -> GlobalConfig:
    raw = _read_yaml(paths.resolve_home(root) / "config.yaml")
    merged = _deep_merge(_defaults_dict(), raw)
    try:
        return GlobalConfig(**merged)
    except PydanticValidationError as exc:
        raise ConfigError(f"invalid global config: {exc}") from exc


def load_project_config(slug: str, root: Path | str | None = None) -> ProjectConfig:
    paths.validate_slug(slug)
    global_raw = _read_yaml(paths.resolve_home(root) / "config.yaml")
    project_raw = _read_yaml(paths.project_dir(slug, root) / "project.yaml")
    # LIVE inheritance: defaults ⊕ global ⊕ project (project overrides win per field,
    # an unset project field resolves to the CURRENT global). ``global_effective`` is
    # the ceiling reference; clamp BEFORE constructing the runtime config so a hand-
    # edited project.yaml can never LOOSEN a ceiling (runtime defense-in-depth).
    global_effective = _deep_merge(_defaults_dict(), global_raw)
    merged = _deep_merge(global_effective, project_raw)
    merged = clamp_project_ceiling(global_effective, merged)
    # Runtime credential-boundary defense: a hand-edited project.yaml can never
    # redirect an LLM endpoint or rebind the machine-global key — force llm.profiles
    # back to the global registry (the project keeps only its route profile CHOICE).
    merged = _strip_project_profile_overrides(global_effective, merged)
    merged["slug"] = slug
    try:
        return ProjectConfig(**merged)
    except PydanticValidationError as exc:
        raise ConfigError(f"invalid project config for '{slug}': {exc}") from exc


def _strip_secret_named_keys(node: Any) -> Any:
    """Return a copy of ``node`` with every mapping key named in
    :data:`_FORBIDDEN_SECRET_KEYS` dropped, recursively (lists descended)."""
    if isinstance(node, dict):
        return {
            key: _strip_secret_named_keys(value)
            for key, value in node.items()
            if not (isinstance(key, str) and key.lower() in _FORBIDDEN_SECRET_KEYS)
        }
    if isinstance(node, list):
        return [_strip_secret_named_keys(item) for item in node]
    return node


def effective_config_snapshot(cfg: Any) -> dict[str, Any]:
    """The merged EFFECTIVE config as a plain JSON-safe dict for the run manifest
    (R2-6 / Build F ch5; v1 ``export.config_snapshot_yaml`` counterpart).

    Accepts a validated config model (:class:`GlobalConfig` / :class:`ProjectConfig`)
    or an already-plain mapping. Every key NAMED like a raw secret
    (:data:`_FORBIDDEN_SECRET_KEYS`) is stripped recursively — defense-in-depth:
    config stores env-var/keyring *references*, never raw secrets, but a snapshot
    persisted into a run manifest must hold even against a hostile merged patch.
    """
    if hasattr(cfg, "model_dump"):
        raw = cfg.model_dump(mode="json")
    else:
        raw = copy.deepcopy(dict(cfg))
    return _strip_secret_named_keys(raw)


def config_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """64-char sha256 hex digest over ``snapshot``'s canonical key-sorted JSON
    (v1 ``export.fingerprint`` port — dict ordering never changes the digest).

    Round-trip contract (v1 criterion 9): a manifest section is reproducible iff
    ``section["config_fingerprint"] == config_fingerprint(section["config_snapshot"])``.
    """
    payload = json.dumps(
        snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assert_no_raw_secrets(node: Any, *, path: str = "") -> None:
    """Recursively reject any mapping key whose NAME implies a raw secret value.

    Only env-var *names* (``env_var``) and the ``key_source`` discriminator are
    permitted near credentials; an actual key/secret/token must never be persisted
    to ``config.yaml`` (env-ref names only)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_SECRET_KEYS:
                where = f"{path}.{key}" if path else key
                raise ConfigError(
                    f"refusing to write a raw secret into config.yaml at {where!r}: "
                    f"store only the env-var NAME (e.g. profile.env_var), never the key value"
                )
            _assert_no_raw_secrets(value, path=f"{path}.{key}" if path else str(key))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _assert_no_raw_secrets(item, path=f"{path}[{i}]")


def _assert_no_profile_routing(llm_node: Any, *, path: str = "llm") -> None:
    """Reject any profile-DEFINITION field anywhere under a project override's ``llm``
    subtree (write-path enforcement of the machine-global credential boundary).

    A project may only CHOOSE a profile (``llm.routes.<task>.preferred_profile`` /
    ``fallback_profile`` — plain string references to an existing machine-global
    profile). It may NEVER carry ``profiles`` (define/redirect a profile) or the
    per-profile endpoint / credential-routing fields
    (:data:`_FORBIDDEN_PROFILE_FIELDS`: ``provider`` / ``env_var`` / ``base_url`` /
    ``key_source``) anywhere under ``llm``; doing so would let a project bind the
    machine's real API key to an attacker endpoint. ``llm_node`` is the ``llm``
    subtree (or ``None`` when absent); scan is a no-op for a non-mapping."""
    if isinstance(llm_node, dict):
        for key, value in llm_node.items():
            if isinstance(key, str) and key.lower() in _FORBIDDEN_PROFILE_FIELDS:
                where = f"{path}.{key}"
                raise ConfigError(
                    f"refusing to write LLM profile-definition field {where!r} into "
                    f"project.yaml: API endpoints and credentials stay machine-global; "
                    f"a project may only CHOOSE a profile "
                    f"(llm.routes.<task>.preferred_profile / fallback_profile)"
                )
            _assert_no_profile_routing(value, path=f"{path}.{key}")
    elif isinstance(llm_node, list):
        for i, item in enumerate(llm_node):
            _assert_no_profile_routing(item, path=f"{path}[{i}]")


def _strip_project_profile_overrides(
    global_dict: dict[str, Any], merged_dict: dict[str, Any]
) -> dict[str, Any]:
    """Runtime defense-in-depth mirror of :func:`_assert_no_profile_routing`: force
    the effective ``llm.profiles`` back to the MACHINE-GLOBAL registry after the
    deep-merge, discarding any project-supplied profile additions or endpoint /
    credential-routing field overrides (``base_url`` / ``env_var`` / ``key_source`` /
    ``provider``).

    ``merged_dict`` is ``defaults ⊕ global ⊕ project``; a hand-edited ``project.yaml``
    could deep-merge a redirected ``base_url`` (or a brand-new profile) into
    ``llm.profiles``. Replacing that subtree wholesale with the global's profiles
    means a project keeps its route profile CHOICE but can never redirect an
    endpoint or rebind the machine-global key. Mutates + returns ``merged_dict``;
    robust to a missing ``llm`` / ``profiles`` section on either side."""
    merged_llm = merged_dict.get("llm")
    if not isinstance(merged_llm, dict):
        return merged_dict
    global_llm = global_dict.get("llm")
    global_profiles = global_llm.get("profiles") if isinstance(global_llm, dict) else None
    if isinstance(global_profiles, dict):
        merged_llm["profiles"] = copy.deepcopy(global_profiles)
    else:
        merged_llm.pop("profiles", None)
    return merged_dict


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic ``tempfile + os.replace`` write of ``text`` (same durability pattern
    as the run manifest writer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_global_config(
    patch: dict[str, Any], root: Path | str | None = None
) -> GlobalConfig:
    """Deep-merge ``patch`` into ``~/.seedgraph/config.yaml``, validate, atomic-write.

    The shared write path behind both the first-run setup screen and the settings
    screen (lifted so the UI never re-implements config writing). ``patch`` carries
    only the user overrides (deep-merged over the existing file, not the built-in
    defaults), so ``config.yaml`` stays a thin overlay that ``load_global_config``
    re-merges over the defaults. The MERGED overlay is validated against the full
    default-backed :class:`GlobalConfig` (a bad patch raises :class:`ConfigError`
    and nothing is written) and is scanned for raw secrets (env-ref names only —
    never a key value). Returns the validated effective config.
    """
    if not isinstance(patch, dict):
        raise ConfigError(f"config patch must be a mapping, got {type(patch).__name__}")
    _assert_no_raw_secrets(patch)
    path = paths.resolve_home(root) / "config.yaml"
    existing = _read_yaml(path)
    merged = _deep_merge(existing, patch)
    # Validate the EFFECTIVE config (defaults ⊕ overlay); raise before writing.
    candidate = _deep_merge(_defaults_dict(), merged)
    try:
        cfg = GlobalConfig(**candidate)
    except PydanticValidationError as exc:
        raise ConfigError(f"invalid global config after patch: {exc}") from exc
    _atomic_write_text(
        path, yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
    )
    return cfg


def write_project_overrides(
    slug: str, patch: dict[str, Any], *, root: Path | str | None = None
) -> None:
    """Deep-merge a thin project OVERRIDE ``patch`` into ``projects/{slug}/project.yaml``.

    The settings-inheritance write path (mirror of :func:`write_global_config`): the
    ``patch`` carries ONLY the four override groups — routing profile CHOICE
    (``llm.routes.*.preferred_profile`` / ``fallback_profile``), ``content_policy``,
    ``budget``, ``answer`` tunables, and ``answer_policy`` — never project identity
    fields. It is deep-merged (a thin overlay) over the existing ``project.yaml`` so
    an omitted key means INHERIT (the current global default is never materialized
    into the file). The merged project dict is scanned for raw secrets — a project
    may CHOOSE a profile but may never persist a raw key / provider endpoint /
    ``key_source`` (API keys stay machine-global). Before writing, the EFFECTIVE
    runtime config (global defaults ⊕ project, then clamped to the tighten-only
    ceilings) is validated as a :class:`ProjectConfig`; a bad patch raises
    :class:`ConfigError` and NOTHING is written. On success the raw (un-clamped)
    project overlay is atomic-written — the ceiling is re-applied live at load time.
    """
    if not isinstance(patch, dict):
        raise ConfigError(
            f"project override patch must be a mapping, got {type(patch).__name__}"
        )
    paths.validate_slug(slug)
    _assert_no_raw_secrets(patch)
    # Credential-boundary: a project may CHOOSE a profile but may never DEFINE or
    # REDIRECT one (no llm.profiles / provider / env_var / base_url / key_source).
    _assert_no_profile_routing(patch.get("llm") if isinstance(patch, dict) else None)
    path = paths.project_dir(slug, root) / "project.yaml"
    existing = _read_yaml(path)
    merged_project = _deep_merge(existing, patch)
    # Belt-and-suspenders: the persisted project file must never carry a raw key
    # either (a project may only choose a profile, not ship credentials).
    _assert_no_raw_secrets(merged_project)
    _assert_no_profile_routing(merged_project.get("llm"))
    # Validate the EFFECTIVE runtime config (defaults ⊕ global ⊕ project, clamped)
    # BEFORE writing; raise on failure so a bad patch never touches the file.
    global_raw = _read_yaml(paths.resolve_home(root) / "config.yaml")
    global_effective = _deep_merge(_defaults_dict(), global_raw)
    candidate = clamp_project_ceiling(
        global_effective, _deep_merge(global_effective, merged_project)
    )
    candidate["slug"] = slug
    try:
        ProjectConfig(**candidate)
    except PydanticValidationError as exc:
        raise ConfigError(f"invalid project override for '{slug}': {exc}") from exc
    _atomic_write_text(
        path, yaml.safe_dump(merged_project, sort_keys=False, allow_unicode=True)
    )


def load_llm_capabilities(path: Path | str | None = None) -> LlmCapabilities:
    """Load + validate the bundled (or provided) ``llm_capabilities.yaml`` (D9)."""
    cap_path = Path(path) if path is not None else _BUNDLED_CAPABILITIES
    if not cap_path.exists():
        raise ConfigError(f"llm_capabilities snapshot not found at {cap_path}")
    data = yaml.safe_load(cap_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"llm_capabilities snapshot {cap_path} must be a YAML mapping")
    try:
        return LlmCapabilities(**data)
    except PydanticValidationError as exc:
        raise ConfigError(f"malformed llm_capabilities snapshot at {cap_path}: {exc}") from exc
