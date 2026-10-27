"""The concept agent tasks on the one protocol (stage 4 design §3, §4, §6.3): one adapter,
registered as ``register_prefix("concepts", ConceptTask)``.

- ``concepts:merge`` - lexical groups ("Tidy labels"): pairs of look-alike labels with
  example claims -> merge / link / split / unrelated / unsure.
- ``concepts:bridge`` - sense checks, then sense-level neighbour pairs, citation-distant
  first, families in bridge order (assumptions, estimands, methods; results and data
  in a second pass) -> the same verdicts plus ``same_sense``. Every answer is stored, so
  no pair is judged twice.
- ``concepts:name_literatures`` - one package per graph grouping: a name and a one-line
  description per literature (agent inferred; a researcher name always wins).

All three use the C3 hooks: ``task_options``, ``select`` (the packages are the subjects:
``subject_kind = 'concept_package'``, eligibility always ``queued`` with no markdown),
``subject_label`` and ``estimate`` (the dry run, :mod:`seedgraph.concepts.estimate`; the
same function builds the batch, so the two agree). Concept tasks cite claim refs and never
write quotes: ``check`` declares no ``QuoteRef``, so no anchor preview runs.

**Frozen packages.** ``select`` builds every package once (refreshing the concepts first
when stale) and keeps them on the instance; ``plan`` returns a package's text and
``instructions`` writes every package's manifest (refs -> paper, span, claim fingerprint)
to a content-addressed file under ``<home>/work/_concept_manifests/``; the batch's small
``labels.json`` names it by sha. ``check`` and ``save`` read the manifest back, so they
judge exactly what the agent saw, and the agent's files stay small.

**Feedback.** A guardrail failure at check time is an ``invalid`` reply with one error per
item ("item 3: merge vetoed: ...; answer link, unrelated or unsure"); the protocol keeps
``needs_revision`` for unanchored quotes, which concept tasks never write.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..concepts import decide
from ..concepts import estimate as cest
from ..concepts.instructions import BY_TASK
from ..concepts.packages import KIND_WORDS, Package
from . import store, tasks
from .store import WorkError

if TYPE_CHECKING:
    from ..library.store import Library
    from .store import BatchItem, WorkPackage

TASK_VERSION = "1.0.0"
SUBJECT_KIND = "concept_package"
_SLUG_WORDS = {"pairs": "pair", "checks": "sense_check", "groups": "group"}


def _work_root(lib: Library) -> Path:
    return Path(lib.root).parent / "work"


def subject_label(subject_id: str) -> str:
    """``assumption-pairs-00051-00100`` -> "assumption pairs 51-100";
    ``literatures-<graph>-run3`` -> "literatures of <graph> (run 3)"."""
    if subject_id.startswith("literatures-"):
        graph, _, run = subject_id[len("literatures-"):].rpartition("-run")
        return f"literatures of {graph} (run {run})"
    parts = subject_id.split("-")
    if len(parts) >= 4 and parts[-3] in _SLUG_WORDS:
        family = "-".join(parts[:-3])
        return f"{family} {KIND_WORDS[_SLUG_WORDS[parts[-3]]]} {int(parts[-2])}-{int(parts[-1])}"
    return subject_id


MANIFEST_DIR = "_concept_manifests"


def write_manifests(lib: Library, manifests: Mapping[str, Any]) -> str:
    """Store the frozen package manifests in a content-addressed file under
    ``<home>/work/_concept_manifests/`` (written before the batch id exists); returns its sha."""
    data = json.dumps(manifests, ensure_ascii=False, sort_keys=True).encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    folder = _work_root(lib) / MANIFEST_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sha}.json"
    if not path.is_file():
        path.write_bytes(data)
    return sha


def load_manifest(lib: Library, batch_id: str, subject_id: str) -> dict[str, Any]:
    """The frozen manifest of one package: ``labels.json`` names the manifest file by sha."""
    try:
        labels = json.loads((_work_root(lib) / batch_id / "labels.json").read_bytes().decode("utf-8"))
        sha = labels["manifest_sha256"]
        data = json.loads((_work_root(lib) / MANIFEST_DIR / f"{sha}.json").read_bytes().decode("utf-8"))
        return data[subject_id]
    except (OSError, ValueError, KeyError) as exc:
        raise WorkError(f"the frozen concept package {subject_id!r} of batch {batch_id} is missing ({exc})") from exc


class ConceptTask:
    """``concepts:<arg>`` with ``arg`` in merge / bridge / name_literatures."""

    subject_kind = SUBJECT_KIND
    allowed_workers = frozenset({"interactive", "background"})

    def __init__(self, lib: Library, arg: str, version: str | None = None) -> None:
        if arg not in cest.TASKS:
            raise WorkError(f"unknown concept task {arg!r}; use concepts:merge, concepts:bridge or "
                            "concepts:name_literatures")
        self.lib = lib
        self.arg = arg
        self.name = f"concepts:{arg}"
        self.version = TASK_VERSION
        self.task_options = dict(cest.OPTIONS[arg])
        self._built: dict[str, Package] = {}

    # --- C3 hooks ---------------------------------------------------------------------------
    def _planned(self, lib: Library, options: Mapping[str, Any]) -> cest.Planned:
        try:
            return cest.plan_packages(lib.conn, self.arg, options, library_root=lib.root)
        except (cest.OptionError, ValueError) as exc:
            raise WorkError(str(exc)) from exc

    def select(self, lib: Library, options: Mapping[str, Any]) -> list[str]:
        from ..concepts import refresh as rf

        if self.arg != "name_literatures" or options.get("graph"):
            rf.refresh_if_stale(lib)
        planned = self._planned(lib, options)
        self._built = {p.subject_id: p for p in planned.result.packages}
        return list(self._built)

    def subject_label(self, lib: Library, subject_id: str) -> str:
        return subject_label(subject_id)

    def estimate(self, lib: Library, options: Mapping[str, Any]) -> cest.ConceptEstimate:
        return cest.summarize(self._planned(lib, options))

    # --- the stage 2 seam -------------------------------------------------------------------
    def resolve(self, lib: Library, query: tasks.Selection) -> list[str]:
        raise WorkError(f"task {self.name!r} chooses its own subjects")

    def eligibility(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> tasks.Eligibility:
        return tasks.Eligibility("queued")

    def plan(self, lib: Library, subject_id: str, options: Mapping[str, Any]) -> list[tasks.PackagePlan]:
        pkg = self._built.get(subject_id)
        if pkg is None:
            raise WorkError(f"concept package {subject_id!r} was not selected by this task")
        return [tasks.PackagePlan(1, 1, None, None, pkg.text, pkg.est_tokens)]

    def instructions(self, lib: Library, *, has_parts: bool) -> tasks.Instructions:
        if not self._built:
            raise WorkError("concept task instructions are written after select() and plan()")
        sha = write_manifests(lib, {sid: p.manifest() for sid, p in self._built.items()})
        labels = {"task": self.name, "version": self.version, "answer": "see instructions.md",
                  "manifest_sha256": sha,
                  "note": "internal: the frozen package manifests are kept by seedgraph; nothing to read here"}
        return tasks.Instructions(BY_TASK[self.arg], json.dumps(labels, ensure_ascii=False, sort_keys=True))

    def _context(self, conn: sqlite3.Connection, batch_id: str, ordinal: int) -> tuple[dict, str | None, str]:
        batch = store.require_batch(conn, batch_id)
        item = store.get_item(conn, batch_id, ordinal)
        assert item is not None
        return load_manifest(self.lib, batch_id, item.subject_id), batch.graph_slug, item.subject_id

    def check(self, lib: Library, package: WorkPackage, payload: dict[str, Any]) -> tasks.CheckResult:
        from ..concepts import refresh as rf

        manifest, graph, _ = self._context(lib.conn, package.batch_id, package.ordinal)
        if self.arg == "name_literatures":
            return tasks.CheckResult(errors=decide.validate(manifest, payload),
                                     claims_kept=len(manifest["items"]))
        rf.refresh_if_stale(lib)
        guard = decide.Guard.load(lib.conn, graph)
        errors = decide.validate(manifest, payload, guard)
        warns = [] if errors else decide.warnings(manifest, payload, guard)
        return tasks.CheckResult(errors=errors, warnings=warns, claims_kept=len(manifest["items"]))

    def save(self, conn: sqlite3.Connection, item: BatchItem, parts: Sequence[tasks.AcceptedPart],
             ctx: tasks.SaveContext) -> tasks.SaveResult:
        manifest, graph, _ = self._context(conn, item.batch_id, item.ordinal)
        part = parts[-1]
        packet = next((p.packet_id for p in store.list_packages(conn, item.batch_id, ordinal=item.ordinal)
                       if p.state == "saved"), None)
        if self.arg == "name_literatures":
            detail = save_literature_names(ctx.lib, manifest, part.payload, graph)
        else:
            detail = decide.save_package(conn, manifest, part.payload, batch_id=item.batch_id, packet_id=packet,
                                         model=part.model, graph=graph).as_dict()
        store.append_event(conn, item.batch_id, "concepts_saved", packet_id=packet,
                           detail={k: v for k, v in detail.items() if k != "feedback"})
        ref = f"{detail.get('applied', detail.get('named', 0))} applied, {detail.get('queued', 0)} queued"
        return tasks.SaveResult(result_ref=ref, detail=detail)


def save_literature_names(lib: Library, manifest: Mapping[str, Any], payload: Mapping[str, Any],
                          graph: str | None) -> dict[str, Any]:
    """Write the agent names, only while the graph's grouping is the one the package froze
    (literature ids change per regroup). ``set_agent_name`` never touches a researcher name."""
    from ..library import graphs, literatures

    answers = {a["item"]: a for a in payload.get("items", []) if isinstance(a, Mapping)}
    out: dict[str, Any] = {"named": 0, "dropped": 0, "feedback": []}
    if not graph:
        out["feedback"].append({"reason": "no_graph", "message": "the batch has no context graph"})
        return out
    with graphs.open_graph(lib, graph) as g:
        run = literatures.latest_run(g)
        for item in manifest["items"]:
            lit = item["literature"]
            ans = answers.get(item["item"])
            if ans is None:
                continue
            if run is None or run.run_id != lit["run_id"]:
                out["dropped"] += 1
                out["feedback"].append({"item": item["item"], "reason": "regrouped",
                                        "message": "the graph was regrouped since the package was made"})
                continue
            literatures.set_agent_name(g, lit["literature_id"], ans["name"].strip(),
                                       (ans.get("description") or "").strip() or None)
            out["named"] += 1
    return out
