"""LLM-assisted post-conversion bibliographic metadata backfill.

The converted markdown is the authoritative identity of a paper; the PDF filename
is just the user's acquisition convention (``2023_santanna.pdf`` -> the junk title
``2023 santanna``). This module reads a converted work's markdown and backfills
``works.canonical_title`` / ``authors`` / ``venue`` / ``year`` + the strong-id
scalar columns:

* :func:`extract_biblio_metadata` calls the LLM through the ``llm.executor`` seam
  (structured schema + one repair + hashes-only usage row), over the first
  ~2600 chars of markdown, and returns a typed :class:`BiblioMeta` (or ``None`` when
  no LLM backend is available / extraction failed — the caller degrades).
* :func:`deterministic_identifiers` scans ONLY the cover/head region (the same
  first ~2600 chars the LLM sees) with regexes for a DOI / arXiv id — this catches
  the paper's OWN DOI (incl. a JSTOR stable-URL ``10.2307/...`` cover DOI) while
  NEVER picking up a CITED work's DOI/arXiv id from a later reference list. LLM
  fields win for title/authors/year/venue; the regex fills doi/arxiv only when the
  LLM returned ``null``.
* :func:`backfill_work_metadata` is the idempotent runner: resolve the work's
  markdown via the bridge, extract, and apply ``identity.update_work_bibliography``.

Tests MOCK the extractor — the live LLM (local Ollama) is used only at runtime.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .. import cache_access
from ..acquisition.bridge import resolve_work_markdown
from ..llm.parse import parse_json_object
from ..project import identity
from ..project.identity import normalize_id

if TYPE_CHECKING:
    from pathlib import Path

    from sqlmodel import Session

    from ..config.models import LlmCapabilities, ProjectConfig
    from ..llm.backend import LLMBackend
    from ..project.service import ProjectHandle

__all__ = [
    "BiblioMeta",
    "BackfillResult",
    "TASK_CLASS",
    "extract_biblio_metadata",
    "deterministic_identifiers",
    "parse_biblio_json",
    "backfill_work_metadata",
    "best_effort_backfill",
]

TASK_CLASS = "metadata_extraction"

# Only the first ~2600 chars (title block / author line / abstract head) carry the
# paper's own metadata; the LLM ignores publisher/JSTOR cover boilerplate + refs.
HEAD_CHARS = 2600


@dataclass
class BiblioMeta:
    """Typed bibliographic metadata for one paper (the extractor's output)."""

    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    venue: Optional[str] = None


@dataclass
class BackfillResult:
    """Per-work outcome of :func:`backfill_work_metadata` (the CLI summary line)."""

    work_id: str
    status: str
    old_title: Optional[str] = None
    new_title: Optional[str] = None
    title_changed: bool = False
    ids_attached: list[tuple[str, str]] = field(default_factory=list)
    ids_collided: list[tuple[str, str]] = field(default_factory=list)
    fields_filled: list[str] = field(default_factory=list)
    used_llm: bool = False
    message: Optional[str] = None


# --- deterministic identifiers (regex over the cover/HEAD region only) -------

# Crossref DOI shape (catches the JSTOR stable DOI 10.2307/<int>). Trailing
# sentence punctuation is stripped after the match.
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
# An arXiv id is trusted ONLY when EXPLICITLY labeled: an ``arxiv.org/abs/<id>``
# URL or an ``arXiv:<id>`` prefix. A bare ``NNNN.NNNNN`` numeric token is NOT an
# arXiv id (it collides with coefficients / table values / cited-paper ids), so no
# bare-number pattern exists — dropping it is deliberate (must-fix #2). The URL is
# tried first so ``arXiv: https://arxiv.org/abs/<id>`` yields the id, not "https".
_ARXIV_URL_RE = re.compile(r"arxiv\.org/abs/([A-Za-z0-9][A-Za-z0-9.\-/]*)", re.IGNORECASE)
_ARXIV_LABELED_RE = re.compile(r"arxiv:\s*([A-Za-z0-9][A-Za-z0-9.\-/]*)", re.IGNORECASE)


def _clean_tail(text: str) -> str:
    return text.rstrip(".,;)]}>\"'")


def deterministic_identifiers(markdown: str) -> dict[str, Optional[str]]:
    """Regex-scan the paper's cover/HEAD region for its OWN DOI / arXiv id (no LLM).

    Returns ``{"doi": <normalized doi | None>, "arxiv": <normalized arxiv | None>}``.

    The scan is bounded to the first :data:`HEAD_CHARS` characters — the same
    cover/title-block region the LLM sees. This is deliberate identity hygiene: a
    paper's OWN DOI (including a JSTOR ``Stable URL`` cover DOI) lives on the cover,
    whereas the DOIs / arXiv ids in a later reference list belong to CITED works and
    must NEVER be attached to the citing paper (must-fix #1). arXiv ids are accepted
    ONLY when explicitly labeled (``arXiv:<id>`` or an ``arxiv.org/abs/<id>`` URL);
    a bare ``NNNN.NNNNN`` numeric token is never treated as an arXiv id (must-fix #2).
    """
    text = (markdown or "")[:HEAD_CHARS]
    doi = None
    m = _DOI_RE.search(text)
    if m is not None:
        doi = normalize_id("doi", _clean_tail(m.group(0)))

    arxiv = None
    um = _ARXIV_URL_RE.search(text)
    if um is not None:
        arxiv = normalize_id("arxiv", _clean_tail(um.group(1)))
    if not arxiv:
        lm = _ARXIV_LABELED_RE.search(text)
        if lm is not None:
            arxiv = normalize_id("arxiv", _clean_tail(lm.group(1)))

    return {"doi": doi, "arxiv": arxiv}


# --- tolerant JSON parse ----------------------------------------------------

def parse_biblio_json(raw_text: str) -> Optional[dict]:
    """Tolerantly parse the model's JSON object, or ``None``.

    Delegates to the shared brace-depth parser (:func:`seedgraph.llm.parse.parse_json_object`):
    first balanced ``{...}`` extracted (fences / prose / trailing tokens tolerated),
    decoded, dict-checked. Garbage / no object -> ``None``; a valid object with missing
    keys still parses (defaults fill in :func:`_meta_from_dict`).
    """
    return parse_json_object(raw_text)


def _clean_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _meta_from_dict(data: dict) -> BiblioMeta:
    """Build a :class:`BiblioMeta` from a parsed JSON object (tolerant coercion)."""
    raw_authors = data.get("authors") or []
    if isinstance(raw_authors, str):
        raw_authors = [raw_authors]
    authors: list[str] = []
    if isinstance(raw_authors, (list, tuple)):
        for a in raw_authors:
            name = _clean_str(a)
            if name:
                authors.append(name)

    year = data.get("year")
    try:
        year = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year = None

    return BiblioMeta(
        title=_clean_str(data.get("title")),
        authors=authors,
        year=year,
        doi=normalize_id("doi", data.get("doi")),
        arxiv_id=normalize_id("arxiv", data.get("arxiv_id")),
        venue=_clean_str(data.get("venue")),
    )


def _parse_result(raw_text: str) -> tuple[Optional[BiblioMeta], list[str]]:
    """``parse`` callback for the executor: raw text -> (BiblioMeta | None, errors)."""
    data = parse_biblio_json(raw_text)
    if data is None:
        return None, ["no JSON object found in model output"]
    return _meta_from_dict(data), []


# --- prompt -----------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a meticulous bibliographic-metadata extractor. You are given the FIRST "
    "PAGE of one paper's converted markdown. Extract THIS paper's OWN metadata and "
    "emit EXACTLY ONE JSON object. Output JSON ONLY — no prose, no commentary.\n"
    "\n"
    "Rules:\n"
    "- Ignore publisher / JSTOR / repository cover-page boilerplate, stable-URL "
    "notices, download timestamps, running headers, and any cited references — those "
    "are NOT this paper's metadata.\n"
    "- Keep a two-part 'Main Title: Subtitle' whole (do not drop the subtitle).\n"
    "- authors is an array of full author-name strings in order (empty array if none "
    "are legible).\n"
    "- Use null for any of year / doi / arxiv_id / venue you cannot find; year is a "
    "4-digit integer.\n"
    "\n"
    'Return: {"title": string|null, "authors": [string], "year": integer|null, '
    '"doi": string|null, "arxiv_id": string|null, "venue": string|null}\n'
)


def _build_prompt(markdown_head: str) -> tuple[str, str]:
    user_prompt = (
        "Extract the bibliographic metadata for THIS paper from its first page.\n"
        "\n"
        "=== PAPER FIRST PAGE START ===\n"
        f"{markdown_head}\n"
        "=== PAPER FIRST PAGE END ===\n"
    )
    return _SYSTEM_PROMPT, user_prompt


def _build_repair(prior_response: str, errors: list[str]) -> str:
    error_lines = "\n".join(f"- {e}" for e in errors) or "- the output was not valid JSON"
    return (
        "Your previous response could not be parsed. Return ONE corrected JSON object "
        "ONLY (no prose, no fences) with keys title, authors, year, doi, arxiv_id, "
        "venue.\n"
        "\n"
        "Errors:\n"
        f"{error_lines}\n"
        "\n"
        "Previous response (for reference):\n"
        f"{prior_response}\n"
    )


# --- LLM extraction (through the executor seam) -----------------------------

def extract_biblio_metadata(
    markdown_head: str,
    *,
    access_class: str,
    config: "ProjectConfig | None" = None,
    capabilities: "LlmCapabilities | None" = None,
    conn=None,
    backend: "LLMBackend | None" = None,
    run_id: Optional[str] = None,
    profile_id_override: Optional[str] = None,
    confirm_external: bool = False,
) -> Optional[BiblioMeta]:
    """Extract bibliographic metadata from the markdown HEAD via the executor seam.

    Mirrors ``extraction.runner`` — the routed content-access gate + one repair
    re-prompt + hashes-only usage row all live in ``executor.run_llm``. Returns a
    :class:`BiblioMeta` on success, or ``None`` when no usable LLM backend is
    available (routed ``no_llm`` / policy block) or the output failed validation
    after one repair. The caller then degrades to deterministic-only.
    """
    from ..llm import executor

    system_prompt, user_prompt = _build_prompt(markdown_head)
    # Output budget: honor the route's ``max_output_tokens`` (config-driven), default
    # 2048. A thinking model (e.g. qwen3) needs headroom beyond the JSON answer so it
    # does not exhaust the budget mid-object and emit empty content (must-fix #3c).
    max_out = 2048
    try:
        if config is not None:
            route = config.llm.routes.get(TASK_CLASS)
            if route is not None and route.max_output_tokens:
                max_out = int(route.max_output_tokens)
    except Exception:  # noqa: BLE001 - never let a config quirk break extraction
        max_out = 2048
    result = executor.run_llm(
        TASK_CLASS,
        system_prompt,
        user_prompt,
        access_class=access_class,
        config=config,
        capabilities=capabilities,
        conn=conn,
        run_id=run_id,
        profile_id_override=profile_id_override,
        confirm_external=confirm_external,
        parse=_parse_result,
        repair_prompt=_build_repair,
        temperature=0.0,
        max_tokens=max_out,
        backend=backend,
        log=conn is not None,
    )
    if not result.ok or result.parsed is None:
        return None
    return result.parsed  # a BiblioMeta (the _parse_result output)


def _merge(llm_meta: Optional[BiblioMeta], det: dict[str, Optional[str]]) -> BiblioMeta:
    """LLM wins for title/authors/year/venue; regex fills doi/arxiv when LLM is null.

    ``llm_meta is None`` (no LLM available) -> deterministic-only: regex ids, no title.
    """
    if llm_meta is None:
        return BiblioMeta(doi=det.get("doi"), arxiv_id=det.get("arxiv"))
    return BiblioMeta(
        title=llm_meta.title,
        authors=list(llm_meta.authors),
        year=llm_meta.year,
        doi=llm_meta.doi or det.get("doi"),
        arxiv_id=llm_meta.arxiv_id or det.get("arxiv"),
        venue=llm_meta.venue,
    )


# --- runner -----------------------------------------------------------------

def backfill_work_metadata(
    h: "ProjectHandle",
    work_id: str,
    *,
    cache_root: "Path | str | None" = None,
    force: bool = False,
    config: "ProjectConfig | None" = None,
    capabilities: "LlmCapabilities | None" = None,
    backend: "LLMBackend | None" = None,
    cache_conn=None,
) -> BackfillResult:
    """Backfill one work's bibliographic metadata from its converted markdown.

    Idempotent: a work whose current title is NOT filename-derived is skipped unless
    ``force``. Resolves the work's markdown via the bridge, extracts (LLM head +
    deterministic full-doc ids, merged), and applies ``identity.update_work_bibliography``
    (which never creates a work and routes id collisions to review). Degrades safely:
    no LLM backend -> deterministic-only ids, no crash.
    """
    from sqlmodel import Session

    from ..config.loader import load_project_config
    from ..db.project_models import Work
    from ..extraction.runner import resolve_source_access_class

    if config is None:
        try:
            config = load_project_config(h.slug, h.root)
        except Exception:  # noqa: BLE001 - fall back to keyless defaults
            from ..config.models import GlobalConfig

            config = GlobalConfig()

    # 1. Resolve work -> markdown; short-circuit idempotently on a good title.
    with Session(h.engine, expire_on_commit=False) as session:
        resolved = resolve_work_markdown(session, work_id=work_id)
        work = session.get(Work, work_id)
        if work is None:
            return BackfillResult(work_id=work_id, status="skipped_no_work",
                                  message=f"no such work {work_id}")
        old_title = work.canonical_title
        if resolved is None:
            return BackfillResult(work_id=work_id, status="skipped_no_markdown",
                                  old_title=old_title, new_title=old_title,
                                  message=f"work {work_id} has no resolvable markdown")
        if not force and not identity.is_filename_derived_title(session, work):
            return BackfillResult(work_id=work_id, status="skipped_idempotent",
                                  old_title=old_title, new_title=old_title,
                                  message="title already authoritative (use --force)")
    markdown_id, _markdown_hash = resolved

    # 2. Read markdown text + resolve its access_class (fail-closed private default).
    own_cache = cache_conn is None
    if own_cache:
        cache_conn = cache_access.open_cache_ro(cache_root)
    try:
        md = cache_access.read_markdown(cache_conn, cache_root, markdown_id)
        if md is None:
            return BackfillResult(work_id=work_id, status="skipped_no_markdown",
                                  old_title=old_title, new_title=old_title,
                                  message=f"markdown {markdown_id} unresolvable in cache")
        access_class = resolve_source_access_class(cache_conn, markdown_id)
    finally:
        if own_cache:
            cache_conn.close()

    # 3. Extract: LLM over the head (may be None), deterministic ids over the SAME
    #    head region so a cited work's reference-list DOI/arXiv is never attached.
    head = md.text[:HEAD_CHARS]
    llm_meta = extract_biblio_metadata(
        head, access_class=access_class, config=config,
        capabilities=capabilities, backend=backend,
    )
    det = deterministic_identifiers(head)
    meta = _merge(llm_meta, det)

    # 4. Apply to the work (never creates one; id collisions -> review).
    with Session(h.engine, expire_on_commit=False) as session:
        upd = identity.update_work_bibliography(
            session, work_id, meta, allow_title_overwrite=force
        )
        session.commit()

    return BackfillResult(
        work_id=work_id,
        status="ok",
        old_title=upd.old_title,
        new_title=upd.new_title,
        title_changed=upd.title_changed,
        ids_attached=upd.ids_attached,
        ids_collided=upd.ids_collided,
        fields_filled=upd.fields_filled,
        used_llm=llm_meta is not None,
    )


def best_effort_backfill(
    h: "ProjectHandle",
    work_id: str,
    *,
    cache_root: "Path | str | None" = None,
    force: bool = False,
    backend: "LLMBackend | None" = None,
) -> Optional[BackfillResult]:
    """Run :func:`backfill_work_metadata`, swallowing ANY error (log to stderr).

    The convert-worker post-hook: a metadata-backfill failure must NEVER fail the
    conversion, so every exception is caught and reported to stderr, returning
    ``None``.
    """
    try:
        return backfill_work_metadata(
            h, work_id, cache_root=cache_root, force=force, backend=backend
        )
    except Exception as exc:  # noqa: BLE001 - best-effort; conversion must not fail
        print(
            f"metadata backfill failed for {work_id}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return None
