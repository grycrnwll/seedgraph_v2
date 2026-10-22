"""One warm converter: a long-lived worker child process pinned to one GPU.

The child (``sg_converter.worker`` in the separate converter environment) loads the
Marker models once and then converts PDFs over a stdin/stdout pipe
(docs/CONVERTER_PROTOCOL.md). This module only speaks the protocol; it never imports
Marker or torch.

Lifecycle: started lazily on the first job; stopped after ``idle_timeout`` seconds
without work; killed (whole process tree) when a PDF exceeds its timeout or the child
dies mid-PDF, in which case that PDF raises :class:`ConverterError` and the next job
starts a fresh child.
"""

from __future__ import annotations

import collections
import itertools
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import SeedgraphError

PROTOCOL_VERSION = 1

#: Options the worker understands, with the defaults v2 used (paginated markdown).
DEFAULT_OPTIONS: dict = {"paginate_output": True, "force_ocr": False,
                         "output_format": "markdown"}

_EOF = object()


class ConverterError(SeedgraphError):
    """A PDF could not be converted by the warm converter (error result, crash,
    timeout, failed start or a converter-version mismatch)."""


class ConverterCrashed(ConverterError):
    """The converter process itself failed (died mid-PDF, timed out and was killed, or
    never became ready), as opposed to the engine reporting an error for this PDF.
    Retrying the same PDF with other options (e.g. OCR) is not expected to help."""


@dataclass
class ConvertOutput:
    """What one conversion returns to the app (which then saves it)."""

    markdown: str
    page_count: int
    metadata: dict
    warnings: list[str]
    timings: dict  # convert_seconds (in worker), wall_seconds (incl. pipe/queue)
    converter_version: str
    gpu: str | None = None  # CUDA_VISIBLE_DEVICES pin of the converter used
    cuda_device: str | None = None  # device name the worker reported
    ready: dict = field(default_factory=dict, repr=False)


class WarmConverter:
    """A single worker child. Thread-safe: one PDF at a time (internal lock)."""

    def __init__(
        self,
        command: list[str],
        *,
        gpu: str | None = None,
        extra_env: dict[str, str] | None = None,
        idle_timeout: float = 600.0,
        ready_timeout: float = 900.0,
        convert_timeout: float = 1800.0,
        expected_version: str | None = None,
    ) -> None:
        self.command = list(command)
        self.gpu = gpu
        self.extra_env = dict(extra_env or {})
        self.idle_timeout = idle_timeout
        self.ready_timeout = ready_timeout
        self.convert_timeout = convert_timeout
        self.expected_version = expected_version
        self.ready_info: dict | None = None
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._messages: queue.Queue | None = None
        self._stderr_tail: collections.deque[str] = collections.deque(maxlen=60)
        self._ids = itertools.count(1)
        self._last_used = 0.0
        self._idle_timer: threading.Timer | None = None

    # -- public -------------------------------------------------------------

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return proc.pid if proc is not None and proc.poll() is None else None

    @property
    def running(self) -> bool:
        return self.pid is not None

    def convert(self, pdf_path: Path | str, options: dict | None = None, *,
                timeout: float | None = None) -> ConvertOutput:
        opts = dict(DEFAULT_OPTIONS)
        opts.update(options or {})
        limit = self.convert_timeout if timeout is None else timeout
        with self._lock:
            self._cancel_idle_timer()
            try:
                self._ensure_started()
                t0 = time.perf_counter()
                req_id = f"r{next(self._ids)}"
                self._send({"type": "convert", "id": req_id,
                            "pdf_path": str(Path(pdf_path).resolve()), "options": opts})
                msg = self._await_result(req_id, limit, pdf_path)
                wall = time.perf_counter() - t0
            finally:
                self._last_used = time.monotonic()
                self._arm_idle_timer()
            if not msg.get("ok"):
                raise ConverterError(f"conversion failed for {Path(pdf_path).name}: "
                                     f"{msg.get('error')}")
            timings = dict(msg.get("timings") or {})
            timings["wall_seconds"] = round(wall, 4)
            ready = self.ready_info or {}
            return ConvertOutput(
                markdown=msg["markdown"],
                page_count=int(msg["page_count"]),
                metadata=dict(msg.get("metadata") or {}),
                warnings=list(msg.get("warnings") or []),
                timings=timings,
                converter_version=ready.get("converter", {}).get("version", ""),
                gpu=self.gpu,
                cuda_device=ready.get("cuda_device"),
                ready=dict(ready),
            )

    def start(self) -> dict:
        """Start the child (if needed) and return its ready message."""
        with self._lock:
            self._ensure_started()
            self._arm_idle_timer()
            return dict(self.ready_info or {})

    def stop(self, grace: float = 10.0) -> None:
        """Ask the child to exit; kill its process tree if it does not."""
        with self._lock:
            self._cancel_idle_timer()
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is None:
                try:
                    self._send({"type": "shutdown"})
                    proc.stdin.close()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    pass
            self._kill()

    def stop_now(self) -> None:
        """Stop without waiting for a conversion in progress: an idle child is asked
        to exit; a busy one is killed (its PDF raises). Used at process exit."""
        if self._lock.acquire(blocking=False):
            try:
                self.stop()
            finally:
                self._lock.release()
        else:
            self._cancel_idle_timer()
            self.kill()

    def kill(self) -> None:
        """Kill the child's whole process tree now (does not take the job lock, so it
        interrupts a conversion in progress; that conversion raises)."""
        proc = self._proc
        if proc is not None:
            _kill_tree(proc)

    # -- internals ----------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._kill()  # reap any dead predecessor
        env = dict(os.environ)
        env.pop("VIRTUAL_ENV", None)  # the app's venv must not leak into uv's
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        env.update(self.extra_env)
        kwargs: dict = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        else:
            kwargs["start_new_session"] = True
        self._stderr_tail.clear()
        try:
            proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, **kwargs,
            )
        except OSError as exc:
            raise ConverterCrashed(f"could not start converter {self.command[0]!r}: {exc}") from exc
        messages: queue.Queue = queue.Queue()
        threading.Thread(target=_read_stdout, args=(proc, messages),
                         name="sg-converter-stdout", daemon=True).start()
        threading.Thread(target=_drain_stderr, args=(proc, self._stderr_tail),
                         name="sg-converter-stderr", daemon=True).start()
        self._proc, self._messages = proc, messages
        self.ready_info = None
        try:
            msg = self._next_message(self.ready_timeout)
        except _Timeout:
            self._kill()
            raise ConverterCrashed(
                f"converter did not become ready within {self.ready_timeout:g}s"
                f"{self._tail()}") from None
        if msg is _EOF:
            code = self._kill()
            raise ConverterCrashed(f"converter exited (code {code}) before ready{self._tail()}")
        if msg.get("type") != "ready" or msg.get("protocol") != PROTOCOL_VERSION:
            self._kill()
            raise ConverterError(f"unexpected converter handshake: {msg!r}")
        version = (msg.get("converter") or {}).get("version")
        if self.expected_version and version != self.expected_version:
            self._kill()
            raise ConverterError(
                f"converter reports marker-pdf {version}, but converter/uv.lock pins "
                f"{self.expected_version}; re-sync the converter environment")
        self.ready_info = msg

    def _await_result(self, req_id: str, limit: float, pdf_path) -> dict:
        deadline = time.monotonic() + limit
        while True:
            try:
                msg = self._next_message(max(0.0, deadline - time.monotonic()))
            except _Timeout:
                self._kill()
                raise ConverterCrashed(
                    f"conversion of {Path(pdf_path).name} timed out after {limit:g}s; "
                    f"converter killed") from None
            if msg is _EOF:
                code = self._kill()
                raise ConverterCrashed(
                    f"converter exited (code {code}) mid-PDF while converting "
                    f"{Path(pdf_path).name}{self._tail()}")
            if msg.get("type") == "result" and msg.get("id") == req_id:
                return msg
            # anything else (stray pong / error for another id) is ignored

    def _next_message(self, timeout: float):
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty:
            raise _Timeout from None

    def _send(self, msg: dict) -> None:
        try:
            self._proc.stdin.write(json.dumps(msg).encode("utf-8") + b"\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            pass  # a dead child shows up as EOF on stdout

    def _kill(self) -> int | None:
        proc, self._proc = self._proc, None
        if proc is None:
            return None
        if proc.poll() is None:
            _kill_tree(proc)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            code = None
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        return code

    def _tail(self) -> str:
        lines = [ln for ln in list(self._stderr_tail)[-8:] if ln.strip()]
        return (": " + " | ".join(lines)) if lines else ""

    def _arm_idle_timer(self) -> None:
        if self._proc is None or self.idle_timeout is None or self.idle_timeout <= 0:
            return
        self._cancel_idle_timer()
        timer = threading.Timer(self.idle_timeout, self._idle_check)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _cancel_idle_timer(self) -> None:
        timer, self._idle_timer = self._idle_timer, None
        if timer is not None:
            timer.cancel()

    def _idle_check(self) -> None:
        if not self._lock.acquire(blocking=False):
            return  # busy: the job's finally re-arms the timer
        try:
            if self._proc is None:
                return
            if time.monotonic() - self._last_used >= self.idle_timeout * 0.95:
                self.stop()
        finally:
            self._lock.release()


class _Timeout(Exception):
    pass


def _read_stdout(proc: subprocess.Popen, messages: queue.Queue) -> None:
    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            try:
                messages.put(json.loads(line.decode("utf-8")))
            except ValueError:
                continue  # never happens with a well-behaved worker; ignore garbage
    except (OSError, ValueError):
        pass
    finally:
        messages.put(_EOF)


def _drain_stderr(proc: subprocess.Popen, tail: collections.deque) -> None:
    try:
        for raw in proc.stderr:
            tail.append(raw.decode("utf-8", "replace").rstrip())
    except (OSError, ValueError):
        pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the child and its descendants (``uv`` spawns the real python)."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                       capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass
    else:
        import signal

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
