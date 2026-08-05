"""``project.yaml`` schema + load/write (decision 36/79).

``project.yaml`` IS the project's settings store — there is deliberately no
``project_settings`` DB table (a mirror would be two sources of truth). The
project identity is the slug: ``project_id == slug`` (decision 79); there is no
separate UUID project id.

NOTE: this ``ProjectConfig`` is the *declarative project file* model and is
distinct from :class:`seedgraph.config.models.ProjectConfig`, which is the phase_0
*runtime/LLM* config (a ``GlobalConfig`` subclass carrying a slug). They model
different concerns and intentionally do not share a class.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError as PydanticValidationError

from ..errors import ValidationError
from . import layout


class AnswerPolicy(BaseModel):
    """The answer-harness policy block (read by the phase_8/10 harness).

    Defaults encode the local-first, source-grounded posture: answers must be
    backed by project sources and external search is off unless explicitly enabled.
    """

    model_config = ConfigDict(extra="ignore")

    require_project_sources: bool = True
    allow_external_search_by_default: bool = False
    distinguish_source_claims_from_synthesis: bool = True


class ProjectConfig(BaseModel):
    """Validated contents of ``project.yaml`` (the project's settings; decision 36/79).

    ``project_id`` MUST equal the directory slug (the sole project identity;
    decision 79). ``storage_mode`` defaults to ``local_first``.
    """

    model_config = ConfigDict(extra="ignore")

    project_id: str
    project_name: str
    description: str | None = None
    schema_version: int = 1
    storage_mode: str = "local_first"
    answer_policy: AnswerPolicy
    created_at: str


def load_project_config(slug: str, root: Path | str | None = None) -> ProjectConfig:
    """Load + validate ``projects/{slug}/project.yaml`` into a :class:`ProjectConfig`.

    Reads via PyYAML ``safe_load`` and validates with Pydantic. Raises a seedgraph
    :class:`~seedgraph.errors.ValidationError` if the file is missing, malformed, or
    its ``project_id != slug`` (single-project-per-DB invariant; decision 12/79).
    """
    path = layout.project_yaml_path(slug, root)
    if not path.exists():
        raise ValidationError(f"project '{slug}': no project.yaml at {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"project '{slug}': malformed project.yaml: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError(f"project '{slug}': project.yaml is not a mapping")
    try:
        cfg = ProjectConfig(**data)
    except PydanticValidationError as exc:
        raise ValidationError(f"project '{slug}': invalid project.yaml: {exc}") from exc
    if cfg.project_id != slug:
        raise ValidationError(
            f"project '{slug}': project.yaml project_id={cfg.project_id!r} does not "
            f"match the directory slug {slug!r} (decision 12/79: slug IS the identity)"
        )
    return cfg


def write_project_config(cfg: ProjectConfig, path: Path) -> None:
    """Persist the DECLARATIVE fields this model owns (identity + ``answer_policy``)
    to ``project.yaml`` WITHOUT clobbering settings-inheritance override groups.

    ``project.yaml`` is a SHARED file: this writer owns the declarative identity +
    ``answer_policy`` block, while the settings-inheritance layer
    (:func:`seedgraph.config.loader.write_project_overrides`) may write ``llm`` /
    ``content_policy`` / ``budget`` / ``answer`` override groups to the same file. A
    full ``safe_dump`` of this ``extra='ignore'`` model would DELETE those groups
    (the clobber hazard), so we read the existing raw file, overlay ONLY the keys
    this model owns, and preserve every other top-level key. Atomic (reuses the
    loader's ``tempfile + os.replace`` writer). Used by ``service.create_project``
    (fresh file) and the project settings screen (existing file, maybe with groups).
    """
    from ..config.loader import _atomic_write_text

    path = Path(path)
    existing: dict = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    # Overlay only the declarative fields this model owns; preserve any override
    # groups (llm/content_policy/budget/answer) already on disk.
    existing.update(cfg.model_dump())
    _atomic_write_text(
        path, yaml.safe_dump(existing, sort_keys=False, allow_unicode=True)
    )
