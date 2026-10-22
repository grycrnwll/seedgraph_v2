"""The converter pool: one warm converter per GPU, shared by the whole app process.

``get_pool().convert(pdf_path, options)`` is the one call the app makes to turn a PDF
into markdown. A job goes to the first idle converter, so concurrent callers spread
over both GPUs. GPUs are counted with ``nvidia-smi -L`` (never by importing torch);
``SEEDGRAPH_MARKER_SLOTS`` overrides the count; with no GPU there is one unpinned slot.

Environment knobs (read when the shared pool is first built):

- ``SEEDGRAPH_MARKER_SLOTS``         number of converters (default: GPU count, else 1)
- ``SEEDGRAPH_MARKER_TIMEOUT``       per-PDF timeout in seconds (default 1800)
- ``SEEDGRAPH_MARKER_IDLE_SECONDS``  idle shutdown in seconds (default 600)
- ``SEEDGRAPH_MARKER_READY_TIMEOUT`` model-load timeout in seconds (default 900)
- ``SEEDGRAPH_CONVERTER_DIR``        the converter uv project (default: ``converter/``
  next to the app's ``src/``)
"""

from __future__ import annotations

import atexit
import os
import queue
import shutil
import subprocess
import threading
import tomllib
from pathlib import Path

from .warm import ConverterError, ConvertOutput, WarmConverter


class ConverterPool:
    """One :class:`WarmConverter` per entry of ``gpus`` (``None`` = unpinned)."""

    def __init__(
        self,
        gpus: list[str | None],
        *,
        command: list[str],
        extra_env: dict[str, str] | None = None,
        idle_timeout: float = 600.0,
        ready_timeout: float = 900.0,
        convert_timeout: float = 1800.0,
        expected_version: str | None = None,
    ) -> None:
        if not gpus:
            gpus = [None]
        self.expected_version = expected_version
        self.converters = [
            WarmConverter(command, gpu=g, extra_env=extra_env, idle_timeout=idle_timeout,
                          ready_timeout=ready_timeout, convert_timeout=convert_timeout,
                          expected_version=expected_version)
            for g in gpus
        ]
        self._free: queue.Queue[WarmConverter] = queue.Queue()
        for c in self.converters:
            self._free.put(c)

    @property
    def size(self) -> int:
        return len(self.converters)

    def convert(self, pdf_path: Path | str, options: dict | None = None, *,
                timeout: float | None = None) -> ConvertOutput:
        """Convert one PDF on the first idle converter (blocks while all are busy).

        ``options`` overrides :data:`~seedgraph.conversion.warm.DEFAULT_OPTIONS`
        (``paginate_output``, ``force_ocr``, ``output_format``). ``timeout`` overrides
        the per-PDF timeout. Raises :class:`ConverterError` on any failure; the pool
        stays usable (a crashed or timed-out converter restarts on its next job)."""
        conv = self._free.get()
        try:
            return conv.convert(pdf_path, options, timeout=timeout)
        finally:
            self._free.put(conv)

    @property
    def converter_version(self) -> str:
        """marker-pdf version this pool converts with. Taken from converter/uv.lock
        without starting anything (every worker's ready message is checked against it);
        if the lock cannot be read, a converter is started and asked."""
        if self.expected_version:
            return self.expected_version
        for c in self.converters:
            if c.ready_info:
                return c.ready_info["converter"]["version"]
        conv = self._free.get()
        try:
            return conv.start()["converter"]["version"]
        finally:
            self._free.put(conv)

    def running_count(self) -> int:
        return sum(1 for c in self.converters if c.running)

    def converter_pids(self) -> list[int | None]:
        return [c.pid for c in self.converters]

    def kill_converter(self, index: int) -> None:
        """Kill one converter's process tree immediately (operator / test hook)."""
        self.converters[index].kill()

    def shutdown(self) -> None:
        """Stop every converter. Never waits for a conversion in progress: a busy
        converter is killed and its PDF fails (so Ctrl+C on serve exits promptly)."""
        for c in self.converters:
            c.stop_now()


# --- construction from the environment ---------------------------------------


def converter_dir() -> Path:
    env = os.environ.get("SEEDGRAPH_CONVERTER_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "converter"


def default_command() -> list[str]:
    """``uv run --frozen --project <converter> python -m sg_converter.worker``.
    ``--frozen`` because plain ``uv run`` re-locks and re-syncs the environment."""
    uv = shutil.which("uv") or "uv"
    return [uv, "run", "--frozen", "--project", str(converter_dir()),
            "python", "-m", "sg_converter.worker"]


def locked_marker_version(project: Path | None = None) -> str | None:
    """The marker-pdf version pinned in the converter's ``uv.lock`` (or ``None``)."""
    lock = (project or converter_dir()) / "uv.lock"
    try:
        data = tomllib.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for pkg in data.get("package", []):
        if pkg.get("name") == "marker-pdf":
            return pkg.get("version")
    return None


def _nvidia_smi_gpu_count() -> int:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0
    try:
        out = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return 0
    if out.returncode != 0:
        return 0
    return sum(1 for ln in out.stdout.splitlines() if ln.startswith("GPU "))


def discover_gpus() -> list[str | None]:
    """One entry per converter: the GPU index it is pinned to, or ``None``.

    ``SEEDGRAPH_MARKER_SLOTS`` sets how many converters (clamped >= 1; garbage => 1);
    they are spread round-robin over the GPUs ``nvidia-smi -L`` lists. With no GPU
    visible there is a single unpinned converter."""
    n_gpus = _nvidia_smi_gpu_count()
    env = (os.environ.get("SEEDGRAPH_MARKER_SLOTS") or "").strip()
    if env:
        try:
            slots = max(1, int(env))
        except ValueError:
            slots = 1
    else:
        slots = max(1, n_gpus)
    if n_gpus == 0:
        return [None] * slots
    return [str(i % n_gpus) for i in range(slots)]


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def pool_from_environment() -> ConverterPool:
    return ConverterPool(
        discover_gpus(),
        command=default_command(),
        idle_timeout=_env_float("SEEDGRAPH_MARKER_IDLE_SECONDS", 600.0),
        ready_timeout=_env_float("SEEDGRAPH_MARKER_READY_TIMEOUT", 900.0),
        convert_timeout=_env_float("SEEDGRAPH_MARKER_TIMEOUT", 1800.0),
        expected_version=locked_marker_version(),
    )


_POOL: ConverterPool | None = None
_POOL_LOCK = threading.Lock()


def get_pool() -> ConverterPool:
    """The process-wide pool (built on first use; no child starts until a job)."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = pool_from_environment()
        return _POOL


def set_pool(pool: ConverterPool | None) -> ConverterPool | None:
    """Replace the process-wide pool (tests, or a caller with its own settings);
    returns the previous one, which the caller should shut down."""
    global _POOL
    with _POOL_LOCK:
        old, _POOL = _POOL, pool
        return old


@atexit.register
def _shutdown_at_exit() -> None:
    pool = _POOL
    if pool is not None:
        pool.shutdown()


__all__ = [
    "ConverterError",
    "ConverterPool",
    "converter_dir",
    "default_command",
    "discover_gpus",
    "get_pool",
    "locked_marker_version",
    "pool_from_environment",
    "set_pool",
]
