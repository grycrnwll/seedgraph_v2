"""The Ollama worker loop (design §6): an in-process worker over the same routine
calls every other worker uses, never a side path.

    run_ollama_worker(lib, batch_id, *, model, max_packages, base_url, client,
                      worker_id, retries, timeout)              -> OllamaRunReport

Loop: ``claim_next`` as ``WorkerRef('ollama', ...)`` -> system prompt =
``instructions.md`` (read once per batch), user prompt = the package text (for
``metadata`` the first ~2 600 characters plus the id hits, as the task wrote it) ->
:meth:`OllamaClient.complete_json` with the task's JSON schema -> ``submit`` with the
raw model text (so an invalid submission records what the model said) -> repeat.

Stops on: ``finished``, ``paused``, ``no_package`` (other workers hold the rest),
``not_allowed`` (the task type does not take Ollama: notes, lenses), ``max_packages``.
Failures:
- a model or connection failure (:class:`OllamaError`) releases the package with
  ``give_back`` and stops with ``stopped='ollama_error'``; no retry, never a pause;
- invalid output (empty, no JSON, or ``submit`` says ``invalid`` / ``needs_revision``)
  is retried ``retries`` times (default once) on the same claim with the errors added to
  the prompt, then released with ``give_back`` and the loop stops with
  ``stopped='invalid_output'`` (claiming again would get the same package back);
- a refused claim (``ClaimRefused``: e.g. the batch was stopped now) stops the loop
  with ``stopped='claim_refused'`` and nothing to release;
- any other exception releases the package and is re-raised.
"""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..llm.ollama import DEFAULT_TIMEOUT_S, OllamaClient, OllamaError
from . import routine
from .claims import ClaimRefused, WorkerRef

if TYPE_CHECKING:
    from ..library.store import Library

CLIENT_NAME = "seedgraph-ollama"
JSON_ONLY = "\n\nAnswer with the JSON object only: no prose, no code fence."

#: Ollama ``format`` schemas per task type (the shapes the task's ``check`` accepts).
_NULLABLE_STR = {"type": ["string", "null"]}
SCHEMAS: dict[str, dict[str, Any]] = {
    "metadata": {
        "type": "object",
        "properties": {
            "title": _NULLABLE_STR,
            "authors": {"type": ["array", "null"], "items": {"type": "string"}},
            "year": {"type": ["integer", "null"]},
            "venue": _NULLABLE_STR,
            "doi": _NULLABLE_STR,
            "arxiv": _NULLABLE_STR,
        },
        "required": ["title", "authors", "year", "venue", "doi", "arxiv"],
    },
}


@dataclass
class OllamaRunReport:
    """``stopped``: ``finished`` | ``paused`` | ``no_package`` | ``not_allowed`` |
    ``max_packages`` | ``ollama_error`` | ``invalid_output`` | ``claim_refused``."""

    batch_id: str
    model: str
    worker_id: str
    stopped: str = "finished"
    claimed: int = 0
    saved: int = 0
    retries: int = 0
    released: list[str] = field(default_factory=list)
    packets: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    next_action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_worker_id() -> str:
    return f"ollama-{socket.gethostname()}-{os.getpid()}"


def _read(path: str) -> str:
    return Path(path).read_bytes().decode("utf-8")


def _retry_prompt(text: str, errors: list[str]) -> str:
    listed = "\n".join(f"- {e}" for e in errors) or "- the answer was not a JSON object"
    return (f"{text}\n\n---\nYour previous answer was rejected:\n{listed}\n"
            "Answer again with one JSON object with exactly the keys the instructions name.")


def _submit_errors(result: routine.SubmitResult) -> list[str]:
    errs = list(result.errors)
    errs += [f"{q['field_key']}: {q.get('hint') or q.get('reason')}" for q in result.failed_quotes]
    return errs


def run_ollama_worker(lib: Library, batch_id: str, *, model: str, max_packages: int | None = None,
                      base_url: str | None = None, client: OllamaClient | None = None,
                      worker_id: str | None = None, retries: int = 1,
                      timeout: float = DEFAULT_TIMEOUT_S) -> OllamaRunReport:
    """Work ``batch_id`` with Ollama ``model`` until it stops (see the module docstring)."""
    if not model:
        raise ValueError("an Ollama model is required")
    if max_packages is not None and max_packages < 1:
        raise ValueError("max_packages must be positive")
    ollama = client or OllamaClient(base_url, timeout=timeout)
    worker = WorkerRef("ollama", worker_id or default_worker_id(), client=CLIENT_NAME, model=model)
    report = OllamaRunReport(batch_id=batch_id, model=model, worker_id=worker.worker_id)
    instructions: str | None = None

    while True:
        if max_packages is not None and report.claimed >= max_packages:
            report.stopped = "max_packages"
            return report
        reply = routine.claim_next(lib, batch_id, worker)
        report.next_action = reply.next_action
        if reply.status != "claimed":
            report.stopped = reply.status
            return report
        msg = reply.message
        assert msg is not None
        report.claimed += 1
        packet_id, claim_id = msg["packet_id"], msg["claim_id"]
        try:
            if instructions is None:
                instructions = _read(msg["instructions_path"])
            text = _read(msg["text_path"])
            outcome = _work_package(lib, ollama, model, msg, instructions + JSON_ONLY, text, retries, report)
        except ClaimRefused as exc:
            report.stopped, report.error = "claim_refused", str(exc)
            return report
        except BaseException as exc:
            _give_back(lib, packet_id, claim_id, f"ollama worker error: {exc!r}", report)
            raise
        if outcome is not None:
            report.stopped, report.error = outcome
            report.next_action = None
            return report


def _work_package(lib: Library, ollama: OllamaClient, model: str, msg: dict[str, Any], system: str, text: str,
                  retries: int, report: OllamaRunReport) -> tuple[str, str] | None:
    """One package: ask, submit, retry on invalid output. ``None`` when saved, else the
    ``(stopped, error)`` that ends the loop (the package is already released)."""
    packet_id, claim_id = msg["packet_id"], msg["claim_id"]
    schema = SCHEMAS.get(msg["task_type"])
    prompt = text
    errors: list[str] = []
    for attempt in range(retries + 1):
        if attempt:
            report.retries += 1
            prompt = _retry_prompt(text, errors)
        try:
            answer = ollama.complete_json(model, system, prompt, schema)
        except OllamaError as exc:
            _give_back(lib, packet_id, claim_id, f"ollama {exc.code}: {exc.message}", report)
            report.packets.append({"packet_id": packet_id, "status": "released", "error": str(exc)})
            return "ollama_error", str(exc)
        result = routine.submit(lib, packet_id, claim_id, note=answer.text, model=answer.model or model,
                                usage=answer.usage or None)
        if result.status == "accepted":
            report.saved += 1
            report.packets.append({"packet_id": packet_id, "status": "accepted", "paper_id": msg["paper_id"],
                                   "paper_status": result.paper_status, "attempts": attempt + 1})
            return None
        errors = _submit_errors(result)
    error = f"invalid output after {retries + 1} attempt(s): " + "; ".join(errors)
    _give_back(lib, packet_id, claim_id, error, report)
    report.packets.append({"packet_id": packet_id, "status": "released", "error": error})
    return "invalid_output", error


def _give_back(lib: Library, packet_id: str, claim_id: str, message: str, report: OllamaRunReport) -> None:
    try:
        routine.release(lib, packet_id, claim_id, "give_back", message=message[:500])
    except ClaimRefused:
        return  # the claim already ended (stopped, re-claimed): nothing to hand back
    report.released.append(packet_id)
