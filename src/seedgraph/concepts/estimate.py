"""The concept tasks' dry run and cost model (stage 4 design §4.4-§4.5).

:func:`plan_packages` is the one function that turns a task argument and its options into
packages: candidates (:mod:`.candidates`), items with example claims and the budget cut
(:mod:`.packages`). The task's ``select`` / ``plan`` and :func:`estimate` both call it, so
the dry run's pairs, sense checks, packages and text tokens equal the created batch's.

**Writes nothing.** Everything is read through :func:`seedgraph.concepts.refresh.plan`
(in memory), so the estimate runs on a read-only connection and on a library whose
concept tables are stale or not yet created.

Cost model: ``text_tokens`` is the packages' text (chars / 4, the protocol's estimator).
Agent tokens are ``rho * (package text + instructions) + answer tokens per item``, with
``rho`` the stage 2 overhead range (:data:`.packages.RHO_LOW` / ``RHO_HIGH``); the low end
reads the instructions once per batch, the high end once per package.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..llm.tokens import estimate_tokens
from . import candidates as cand
from . import packages as pk
from .families import BY_NAME, CONCEPT_FAMILIES
from .instructions import BY_TASK

TASKS = ("merge", "bridge", "name_literatures")
CANDIDATE_SOURCE = "keyword"  # lexical groups / TF-IDF neighbours (no embeddings yet)

#: The task options (``task_options``) and their defaults, per task argument.
OPTIONS: dict[str, dict[str, Any]] = {
    "merge": {"families": None, "budget_tokens": None},
    "bridge": {"families": ",".join(cand.FIRST_PASS), "k": cand.K_DEFAULT, "budget_tokens": None,
               "sense_checks": "all"},
    "name_literatures": {"include_named": False},
}


class OptionError(ValueError):
    pass


def families_option(value: Any, default: Sequence[str]) -> tuple[str, ...]:
    if value is None or value == "":
        return tuple(default)
    items = value.split(",") if isinstance(value, str) else list(value)
    fams = tuple(dict.fromkeys(str(f).strip() for f in items if str(f).strip()))
    bad = [f for f in fams if f not in CONCEPT_FAMILIES]
    if bad or not fams:
        raise OptionError(f"unknown concept family {bad or value!r}; families: {', '.join(CONCEPT_FAMILIES)}")
    return fams


def _int_option(name: str, value: Any, *, minimum: int = 1) -> int | None:
    if value is None:
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise OptionError(f"{name} must be a whole number, not {value!r}") from None
    if n < minimum:
        raise OptionError(f"{name} must be at least {minimum}")
    return n


@dataclass(frozen=True)
class Options:
    task: str
    families: tuple[str, ...] = ()
    k: int = cand.K_DEFAULT
    budget_tokens: int | None = None
    sense_checks: str = "all"
    graph: str | None = None
    include_named: bool = False

    @classmethod
    def coerce(cls, task: str, options: Mapping[str, Any] | None) -> Options:
        if task not in TASKS:
            raise OptionError(f"unknown concept task {task!r}; use one of {', '.join(TASKS)}")
        o = {**OPTIONS[task], **dict(options or {})}
        graph = o.get("graph")
        if task == "name_literatures":
            return cls(task, graph=graph, include_named=bool(o.get("include_named")))
        default = CONCEPT_FAMILIES if task == "merge" else cand.FIRST_PASS
        fams = families_option(o.get("families"), default)
        mode = o.get("sense_checks", "all") or "all"
        if mode not in cand.SENSE_CHECK_MODES:
            raise OptionError(f"sense_checks must be one of {', '.join(cand.SENSE_CHECK_MODES)}")
        if task == "bridge":
            fams = tuple(sorted(fams, key=lambda f: (BY_NAME[f].bridge_order or 99, f)))
        return cls(task, fams, _int_option("k", o.get("k", cand.K_DEFAULT)) or cand.K_DEFAULT,
                   _int_option("budget_tokens", o.get("budget_tokens")), mode, graph)


def instructions_tokens(task: str) -> int:
    return estimate_tokens(BY_TASK[task])


@dataclass
class Planned:
    options: Options
    result: pk.BuildResult
    candidates: list[cand.Candidate]
    state: cand.ConceptState | None = None


def plan_packages(conn: sqlite3.Connection, task: str, options: Mapping[str, Any] | None = None, *,
                  library_root=None, state: cand.ConceptState | None = None) -> Planned:
    """Candidates -> items -> packages (with the budget cut) for ``concepts:<task>``."""
    opts = Options.coerce(task, options)
    if task == "name_literatures":
        res = pk.literature_build(conn, opts.graph, include_named=opts.include_named, library_root=library_root)
        return Planned(opts, res, [])
    state = state or cand.load(conn, graph=opts.graph, families=_load_families(opts), library_root=library_root)
    if task == "merge":
        cands = cand.merge_candidates(state, families=opts.families)
        kind = "group"
    else:
        cands = cand.bridge_candidates(state, families=opts.families, k=opts.k, sense_checks=opts.sense_checks)
        kind = "pair"
    res = pk.from_candidates(state, cands, f"concepts:{task}", task_kind=kind, budget_tokens=opts.budget_tokens,
                             instructions_tokens=instructions_tokens(task))
    return Planned(opts, res, cands, state)


def _load_families(opts: Options) -> tuple[str, ...]:
    fams = set(opts.families)
    for f in opts.families:
        fams |= set(cand.ADJACENT.get(f, ()))
    return tuple(sorted(fams))


@dataclass(frozen=True)
class ConceptEstimate:
    """The dry run (exposes :class:`seedgraph.work.tasks.TokenEstimate`)."""

    task: str
    families: tuple[str, ...]
    graph: str | None
    candidate_source: str
    pairs: int
    sense_checks: int
    groups: int
    literatures: int
    packages: int
    text_tokens: int
    agent_tokens_low: int
    agent_tokens_high: int
    budget_tokens: int | None
    items_total: int
    items_packaged: int
    cut_at: int | None
    per_family: dict[str, dict[str, int]] = field(default_factory=dict)
    distance_share: dict[str, float] = field(default_factory=dict)
    subjects: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "subjects"}
        d["families"] = list(self.families)
        if self.budget_tokens is not None:
            d["cut"] = (f"the first {self.items_packaged} of {self.items_total} items by priority fit"
                        if self.cut_at else f"all {self.items_total} items fit")
        return d


def summarize(planned: Planned) -> ConceptEstimate:
    res, opts = planned.result, planned.options
    instr = instructions_tokens(opts.task)
    pkgs = res.packages
    text = sum(p.est_tokens for p in pkgs)
    low, high = pk.total_low(pkgs, instr), pk.total_high(pkgs, instr)
    items = [it for p in pkgs for it in p.items]
    return ConceptEstimate(
        task=f"concepts:{opts.task}", families=opts.families, graph=opts.graph,
        candidate_source=CANDIDATE_SOURCE if opts.task != "name_literatures" else "literatures",
        pairs=sum(1 for it in items if it.kind == "pair"),
        sense_checks=sum(1 for it in items if it.kind == "sense_check"),
        groups=len({(it.family, it.group) for it in items if it.group is not None}),
        literatures=sum(1 for it in items if it.kind == "literature"), packages=len(pkgs), text_tokens=text,
        agent_tokens_low=low, agent_tokens_high=high, budget_tokens=opts.budget_tokens,
        items_total=res.items_total, items_packaged=res.items_packaged, cut_at=res.cut_at,
        per_family=res.stats.get("families", {}), distance_share=res.stats.get("distance_share", {}),
        subjects=tuple(p.subject_id for p in pkgs))


def estimate(conn: sqlite3.Connection, task: str, options: Mapping[str, Any] | None = None, *,
             library_root=None) -> ConceptEstimate:
    """The dry run for ``concepts:<task>`` (writes nothing)."""
    return summarize(plan_packages(conn, task, options, library_root=library_root))
