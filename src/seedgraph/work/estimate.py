"""The "Write notes" estimate shown before a batch is created (stage 3 design §2.2).

:func:`estimate` answers, for a paper selection and a whole-paper task (``notes`` or
``lens:<id>``): which papers would be queued and which skipped (preserved, held back,
failed) with the reason, each queued paper's **text tokens** and a **range of agent
tokens**, and the totals over the queued papers.

- Text tokens come from the packaging estimator: :func:`packages.plan_parts` (chars / 4
  after stripping references, :func:`seedgraph.llm.tokens.estimate_tokens`), summed over
  the parts, so the number matches the packages ``create_batch`` would write.
- Agent tokens are ``text_tokens`` times :data:`AGENT_FACTOR_LOW` / :data:`AGENT_FACTOR_HIGH`.
- **Read-only.** The text-quality gate in ``create_batch`` caches verdicts and raises
  attention items, so the estimate never calls it: it reads a cached verdict for the
  current ``checker_version`` if there is one (``text_checked``), and otherwise counts the
  paper as queued; ``create_batch`` may still hold it back.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..library import attention, textquality
from . import packages, tasks
from .store import WorkError
from .tasks_notes import WholePaperTask

if TYPE_CHECKING:
    from ..library.store import Library

#: Agent tokens per paper-text token: the low and high ends of the per-paper range (the
#: agent reads the paper text plus the instructions and writes the note). Calibrated on
#: the stage 2 live run, where whole-paper notes cost roughly 1.5-2.6x the package text.
AGENT_FACTOR_LOW = 1.5
AGENT_FACTOR_HIGH = 2.7


@dataclass(frozen=True)
class PaperEstimate:
    paper_id: str
    state: str                # queued | preserved | held_back | failed
    reason: str | None        # why a paper is skipped (None when queued)
    text_tokens: int          # 0 when skipped
    parts: int                # packages the paper would get (0 when skipped)
    agent_tokens_low: int
    agent_tokens_high: int
    text_checked: bool        # a cached text-quality verdict was found (else checked at batch creation)


@dataclass(frozen=True)
class Estimate:
    task: str
    papers: tuple[PaperEstimate, ...]   # selection order, duplicates removed

    @property
    def queued(self) -> tuple[PaperEstimate, ...]:
        return tuple(p for p in self.papers if p.state == "queued")

    @property
    def skipped(self) -> tuple[PaperEstimate, ...]:
        return tuple(p for p in self.papers if p.state != "queued")

    @property
    def text_tokens(self) -> int:
        return sum(p.text_tokens for p in self.queued)

    @property
    def agent_tokens_low(self) -> int:
        return sum(p.agent_tokens_low for p in self.queued)

    @property
    def agent_tokens_high(self) -> int:
        return sum(p.agent_tokens_high for p in self.queued)


def _skip(paper_id: str, state: str, reason: str, *, text_checked: bool = False) -> PaperEstimate:
    return PaperEstimate(paper_id, state, reason, 0, 0, 0, 0, text_checked)


def _cached_gate(lib: Library, paper: packages.PaperText) -> tuple[bool, str | None]:
    """``(text_checked, hold-back reason)`` from the cached verdict only (no writes)."""
    row = lib.conn.execute("SELECT status, checker_version FROM text_quality WHERE markdown_id = ?",
                           (paper.markdown_id,)).fetchone()
    if row is None or row[1] != textquality.CHECKER_VERSION:
        return False, None
    if row[0] == "pass":
        return True, None
    kind = packages.GATE_KINDS.get(row[0])
    if kind is None:
        return True, None
    item = attention.get_item(lib, attention.item_id_for(
        kind, {"paper_id": paper.paper_id, "markdown_id": paper.markdown_id}))
    if item is not None and item.status == "resolved" and item.resolution == packages.ACCEPT_TEXT:
        return True, None
    return True, f"{kind}: the paper text failed the text-quality check"


def _one(lib: Library, task: WholePaperTask, paper_id: str, replace: bool) -> PaperEstimate:
    conn = lib.conn
    if not conn.execute("SELECT 1 FROM papers WHERE paper_id = ?", (paper_id,)).fetchone():
        return _skip(paper_id, "failed", "not a library paper")
    paper = packages.load_paper_text(conn, paper_id)
    if paper is None:
        return _skip(paper_id, "failed", "no converted markdown (or its file is missing)")
    if not replace and task.is_preserved(conn, paper_id):
        return _skip(paper_id, "preserved", "a fresh current result exists (use replace to redo it)")
    held = packages.title_hold(conn, paper_id)
    if held is not None:
        return _skip(paper_id, "held_back", held.reason)
    checked, gate_reason = _cached_gate(lib, paper)
    if gate_reason is not None:
        return _skip(paper_id, "held_back", gate_reason, text_checked=checked)
    parts = packages.plan_parts(paper.text, paper.reference_ranges, paper.section_starts)
    tokens = sum(p.est_tokens for p in parts)
    return PaperEstimate(paper_id, "queued", None, tokens, len(parts), round(tokens * AGENT_FACTOR_LOW),
                         round(tokens * AGENT_FACTOR_HIGH), checked)


def estimate(lib: Library, task: str, paper_ids: Iterable[str], *, replace: bool = False) -> Estimate:
    """The estimate for running whole-paper ``task`` over ``paper_ids`` (see the module doc).
    ``replace`` mirrors the batch option: preserved papers are counted too."""
    tt = tasks.get_task(lib, task)
    if not isinstance(tt, WholePaperTask):
        raise WorkError(f"no estimate for task {task!r}: only whole-paper tasks (notes, lens:<id>)")
    ids = list(dict.fromkeys(paper_ids))
    return Estimate(task=tt.name, papers=tuple(_one(lib, tt, pid, replace) for pid in ids))
