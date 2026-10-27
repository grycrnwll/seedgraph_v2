"""The task-type seam (design §2.1): what varies per task sits behind :class:`TaskType`,
registered by name. The protocol (``routine``) owns batches, packages, package claims,
feedback assembly and the save transaction; anchor preview is a protocol service run on
the quotes a task's :meth:`TaskType.check` declares.

Registered here: ``notes``, ``metadata`` and ``lens:<lens_id>`` (a prefix factory).
**Stage 4**: ``concepts:*`` task types (``subject_kind = 'concept_package'``) register
with :func:`register_prefix`. Subjects that are not papers use the optional hooks below
(stage 4 design §3.1); a task without them keeps the paper behaviour unchanged.

**Optional hooks** (not part of :class:`TaskType`; looked up with ``getattr``):

- ``task_options: Mapping[str, Any]`` - the task's own batch options and their defaults
  (e.g. ``{"family": None, "k": 5, "budget_tokens": None}``). :func:`batch_options` merges
  them into the stored option map, so ``eligibility`` / ``plan`` / ``save`` (via
  ``ctx.options``) and the hooks below all see the same values; a key the task does not
  declare and :class:`~seedgraph.work.routine.BatchOptions` does not know is refused.
- ``select(lib, options) -> list[str]`` (:data:`SELECT`) - the task chooses its subjects.
  ``create_batch`` calls it instead of the paper selection and refuses ``paper_ids`` and a
  ``query``. ``options`` is the merged map plus ``"graph"`` (the ``graph`` argument or
  ``None``). The ids must be distinct.
- ``subject_label(lib, subject_id) -> str`` (:data:`SUBJECT_LABEL`) - the human label of a
  subject (``"assumption pairs 41-90"``) for package messages, batch views and the batch
  page. Default: the paper's title.
- ``estimate(lib, options) -> <estimate>`` (:data:`ESTIMATE`) - the dry run, called by
  :func:`seedgraph.work.estimate.estimate` with the same ``options`` as ``select``. It
  returns the task's own object; it should expose ``text_tokens``, ``agent_tokens_low``
  and ``agent_tokens_high`` (:class:`TokenEstimate`).

A task whose ``check`` declares no :class:`QuoteRef` gets no anchor preview (concept tasks
cite claim refs and never write quotes).

Also here: the fixed selection language (design §1.2), resolved once at batch creation:
``held [papers] [in graph <slug>] [without current note | with stale note |
needing metadata | lens <id> missing]``.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from .store import WorkError

if TYPE_CHECKING:
    from ..library.store import Library
    from .store import BatchItem, WorkPackage


# --- value types -------------------------------------------------------------------------

@dataclass(frozen=True)
class Eligibility:
    """A subject's state at batch creation: ``queued`` / ``preserved`` / ``held_back`` /
    ``failed``, with the binding a queued subject is packaged from."""

    state: str
    reason: str | None = None
    markdown_id: str | None = None
    markdown_hash: str | None = None
    expected_current_note_id: str | None = None
    attention_item_id: str | None = None


@dataclass(frozen=True)
class PackagePlan:
    """One package of a subject: its text and its range in the full markdown."""

    part: int
    part_count: int
    start_char: int | None
    end_char: int | None
    text: str
    est_tokens: int


@dataclass(frozen=True)
class Instructions:
    instructions_md: str
    labels_json: str


@dataclass(frozen=True)
class QuoteRef:
    field_key: str
    quote: str | None


@dataclass
class CheckResult:
    """The pure check of one submission: errors make it ``invalid``; ``quotes`` are the
    found claims' quotes for the anchor preview; ``claims_kept`` counts the claim rows
    the submission would produce."""

    errors: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    quotes: list[QuoteRef] = field(default_factory=list)
    claims_kept: int = 0


@dataclass(frozen=True)
class AcceptedPart:
    """A saved submission of one package, as the task's ``save`` sees it."""

    part: int
    part_count: int
    payload: dict[str, Any]
    worker_kind: str
    client: str | None
    model: str | None
    usage: dict[str, Any] | None


@dataclass(frozen=True)
class SaveResult:
    result_ref: str | None  # note id, lens extraction run id, paper id ...
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SaveContext:
    """What the protocol hands ``save`` besides the item and parts."""

    lib: Library
    batch_id: str
    markdown_text: str
    options: Mapping[str, Any]


#: Optional post-commit hook a task type may define (not part of :class:`TaskType`, so
#: task types without one need nothing): ``after_commit(lib, item, result)``. The routine
#: calls it after the submit transaction that saved the item ``done`` has committed, with
#: no transaction open (network work such as the metadata task's DOI verification belongs
#: here, never inside ``save``); an exception from it is logged, never raised.
AFTER_COMMIT = "after_commit"

#: Optional hook names for subjects that are not papers (see the module doc).
SELECT = "select"
SUBJECT_LABEL = "subject_label"
ESTIMATE = "estimate"
TASK_OPTIONS = "task_options"


class TokenEstimate(Protocol):
    """What an ``estimate`` hook's result should expose (the totals a dry run shows)."""

    @property
    def text_tokens(self) -> int: ...
    @property
    def agent_tokens_low(self) -> int: ...
    @property
    def agent_tokens_high(self) -> int: ...


def task_option_defaults(task: TaskType) -> dict[str, Any]:
    """The task's own batch options with their defaults (``{}`` for most tasks)."""
    return dict(getattr(task, TASK_OPTIONS, None) or {})


def split_options(task: TaskType, options: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(protocol options, task options)``: the task's declared keys (defaults filled in)
    split off from the rest, which go to ``BatchOptions``."""
    defaults = task_option_defaults(task)
    given = dict(options or {})
    own = {**defaults, **{k: v for k, v in given.items() if k in defaults}}
    return {k: v for k, v in given.items() if k not in defaults}, own


def hook_options(option_map: Mapping[str, Any], graph: str | None) -> dict[str, Any]:
    """The map ``select`` and ``estimate`` receive: the batch options plus ``graph``."""
    return {**option_map, "graph": graph}


def selects_subjects(task: TaskType) -> bool:
    """Whether the task chooses its own subjects (defines ``select``)."""
    return callable(getattr(task, SELECT, None))


class TaskType(Protocol):
    name: str
    version: str
    subject_kind: str
    allowed_workers: frozenset[str]

    def resolve(self, lib: Library, query: Selection) -> list[str]: ...
    def eligibility(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> Eligibility: ...
    def plan(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> list[PackagePlan]: ...
    def instructions(self, lib: Library, *, has_parts: bool) -> Instructions: ...
    def check(self, lib: Library, package: WorkPackage, payload: dict[str, Any]) -> CheckResult: ...
    def save(self, conn: sqlite3.Connection, item: BatchItem, parts: Sequence[AcceptedPart],
             ctx: SaveContext) -> SaveResult: ...


# --- registry ----------------------------------------------------------------------------

_BY_NAME: dict[str, Callable[[Library], TaskType]] = {}
_BY_PREFIX: dict[str, Callable[..., TaskType]] = {}


def register(name: str, factory: Callable[[Library], TaskType]) -> None:
    _BY_NAME[name] = factory


def register_prefix(prefix: str, factory: Callable[..., TaskType]) -> None:
    """``<prefix>:<arg>`` task names, e.g. ``lens:<lens_id>`` (stage 4: ``concepts:*``).
    ``factory(lib, arg, version=None)``: ``version`` pins an existing batch's version."""
    _BY_PREFIX[prefix] = factory


def _ensure_builtins() -> None:
    if "notes" in _BY_NAME:
        return
    from . import tasks_lens, tasks_metadata, tasks_notes

    register("notes", tasks_notes.NotesTask)
    register("metadata", tasks_metadata.MetadataTask)
    register_prefix("lens", tasks_lens.LensTask)


def get_task(lib: Library, name: str) -> TaskType:
    """The task type registered as ``name`` (``WorkError`` if unknown)."""
    _ensure_builtins()
    if name in _BY_NAME:
        return _BY_NAME[name](lib)
    prefix, sep, arg = name.partition(":")
    if sep and prefix in _BY_PREFIX and arg:
        return _BY_PREFIX[prefix](lib, arg)
    known = sorted(_BY_NAME) + [f"{p}:<id>" for p in sorted(_BY_PREFIX)]
    raise WorkError(f"unknown task type {name!r}; known: {', '.join(known)}")


def task_for_batch(lib: Library, task_type: str, task_version: str) -> TaskType:
    """The task type an existing batch was created with, at the batch's version: a
    prefix task (``lens:<id>``) is rebuilt at exactly ``task_version`` (even if that
    lens version was archived since); any other task whose version changed since the
    batch was created is refused, so a batch never mixes versions."""
    _ensure_builtins()
    prefix, sep, arg = task_type.partition(":")
    if sep and prefix in _BY_PREFIX and arg:
        tt = _BY_PREFIX[prefix](lib, arg, version=task_version)  # type: ignore[call-arg]
    else:
        tt = get_task(lib, task_type)
    if tt.version != task_version:
        raise WorkError(f"task {task_type!r} is now version {tt.version}, but this batch was created at "
                        f"{task_version}; create a new batch")
    return tt


# --- selection language ------------------------------------------------------------------

_QUERY_RE = re.compile(
    r"^held(?:\s+papers?)?"
    r"(?:\s+in\s+graph\s+(?P<graph>[a-z0-9._-]+))?"
    r"(?:\s+(?P<filter>without\s+(?:a\s+)?(?:current\s+)?notes?"
    r"|with\s+(?:a\s+)?stale\s+notes?"
    r"|needing\s+metadata"
    r"|lens\s+(?P<lens>\S+)\s+missing))?$"
)


@dataclass(frozen=True)
class Selection:
    """A parsed selection query."""

    text: str
    graph: str | None = None
    filter: str | None = None  # 'without_note' | 'stale_note' | 'needing_metadata' | 'lens_missing'
    lens_id: str | None = None


def parse_query(text: str) -> Selection:
    norm = " ".join(text.strip().lower().split())
    m = _QUERY_RE.match(norm)
    if m is None:
        raise WorkError(
            f"cannot read the selection {text!r}; use: held [in graph <slug>] [without current note | "
            "with stale note | needing metadata | lens <id> missing]")
    raw = m.group("filter") or ""
    kind = None
    if raw.startswith("without"):
        kind = "without_note"
    elif raw.startswith("with"):
        kind = "stale_note"
    elif raw.startswith("needing"):
        kind = "needing_metadata"
    elif raw.startswith("lens"):
        kind = "lens_missing"
    return Selection(text=text.strip(), graph=m.group("graph"), filter=kind, lens_id=m.group("lens"))


def held_papers(lib: Library, graph: str | None = None) -> list[str]:
    """Held library papers (in ``graph``'s members when given), by ``paper_id``."""
    held = [r[0] for r in lib.conn.execute(
        "SELECT paper_id FROM paper_status WHERE status = 'held' ORDER BY paper_id")]
    if graph is None:
        return held
    from ..library import graphs

    with graphs.open_graph(lib, graph) as g:
        members = {p.paper_id for p in graphs.membership(g, status="member")}
    return [p for p in held if p in members]


def resolve_selection(lib: Library, sel: Selection) -> list[str]:
    """The paper ids a selection names (filters are task-independent library facts)."""
    from . import tasks_lens, tasks_metadata, tasks_notes

    papers = held_papers(lib, sel.graph)
    if sel.filter == "without_note":
        return [p for p in papers if tasks_notes.current_note(lib.conn, p) is None]
    if sel.filter == "stale_note":
        return [p for p in papers if tasks_notes.note_freshness(lib.conn, p) == "stale"]
    if sel.filter == "needing_metadata":
        return [p for p in papers if tasks_metadata.needs_metadata(lib.conn, p)]
    if sel.filter == "lens_missing":
        lens = tasks_lens.load_lens(lib.conn, sel.lens_id or "")
        return [p for p in papers if not tasks_lens.lens_is_current(lib.conn, lens, p)]
    return papers
