"""Isolate every test into a fresh, throwaway SEEDGRAPH_HOME and a clean env."""

from __future__ import annotations

import builtins
import importlib
import io
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fail-loud cross-phase contract import guard (Build F chunk 7).
#
# Runs at conftest IMPORT — before ANY collection, including targeted
# single-file runs — so a vanished contract symbol is an immediate hard
# failure naming the symbol, never a skip. The CI workflow re-runs the same
# list as an inline pre-pytest step (.github/workflows/ci.yml); a deliberate
# rename updates BOTH in the same PR. Keep the list short: only the pinned
# cross-phase contract seams other builds import against.
# ---------------------------------------------------------------------------

_CONTRACT_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("seedgraph.notes.spans", "ensure_span"),
    ("seedgraph.citation.bib_parser", "parse_references"),
    ("seedgraph.citation.resolve", "resolve_references"),
    ("seedgraph.sections.parser", "parse_sections"),
    ("seedgraph.work.routine", "claim_next"),
    ("seedgraph.llm.ollama", "check_reachable"),
    ("seedgraph.project.identity", "canonical_key"),
)


def _assert_contract_symbols() -> None:
    missing = []
    for module_name, symbol in _CONTRACT_SYMBOLS:
        module = importlib.import_module(module_name)
        if not hasattr(module, symbol):
            missing.append(f"{module_name}.{symbol}")
    if missing:
        raise RuntimeError(
            f"cross-phase contract symbol(s) missing: {missing!r} — fail-loud "
            f"import guard (Build F ch7). If a rename is intentional, update "
            f"_CONTRACT_SYMBOLS here AND the CI import-guard step in the same PR."
        )


_assert_contract_symbols()


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "seedgraph_home"
    home.mkdir()
    monkeypatch.setenv("SEEDGRAPH_HOME", str(home))
    # Make sure no real provider keys leak into routing/doctor behavior.
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "S2_API_KEY",
        "CORE_API_KEY",
        "OPENALEX_API_KEY",
        "SEEDGRAPH_CORE_KEY",
        "SEEDGRAPH_CONTACT_EMAIL",
        "SEEDGRAPH_LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return home


@pytest.fixture(autouse=True)
def _guard_paper_library(monkeypatch):
    """Belt-and-suspenders guard: fail any test that write-opens the regression corpus.

    HARD RULE (v1 AGENTS.md section 0, verbatim):

        NEVER WRITE TO ``D:\\Documents\\research\\paper_library``. IT IS A
        READ-ONLY REGRESSION CORPUS.

    Wraps ``builtins.open`` — and ``io.open``, which ``pathlib``'s read/write
    helpers call — to raise ``AssertionError`` on any write/append/create/``+``
    mode open of a path under the guarded root; read-only opens pass (the
    ``-m corpus`` gate reads the library). Raw-fd paths (``os.open``, low-level
    ``shutil``) are accepted residual risk, same as v1: the corpus gate's
    byte-size snapshot catches any write after the fact.

    The root comes from ``SEEDGRAPH_REGRESSION_CORPUS``, defaulting to the
    machine-local ``D:/Documents/research/paper_library``. It is resolved per
    open call, not at fixture setup: this autouse fixture installs the wrapper
    before the test body runs, so a test's own ``monkeypatch.setenv`` must
    still be able to retarget the guard (the self-test in
    ``test_corpus_guard.py`` depends on this).
    """
    real_open = builtins.open
    v2_home = str(Path.home() / ".seedgraph").lower()

    def guarded_open(file, mode="r", *args, **kwargs):
        root = os.environ.get(
            "SEEDGRAPH_REGRESSION_CORPUS", "D:/Documents/research/paper_library"
        )
        lib_str = str(Path(root)).lower()
        try:
            p = str(Path(file)).lower()
        except Exception:  # noqa: BLE001 - int fds, bytes paths: never guarded
            p = str(file).lower()
        writing = any(m in mode for m in ("w", "a", "x", "+"))
        if p.startswith(lib_str) and writing:
            raise AssertionError(
                "HARD RULE violation: attempted write-open of read-only corpus: "
                f"{file!r} (mode={mode!r})"
            )
        # v2's live home (~/.seedgraph) is read-only for v3 and its tests.
        if writing and (p == v2_home or p.startswith(v2_home + os.sep)):
            raise AssertionError(
                f"attempted write-open under v2's home {v2_home!r}: {file!r} (mode={mode!r})"
            )
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    # builtins.open and io.open are the same object at interpreter start;
    # pathlib calls the latter, so both names must point at the wrapper.
    monkeypatch.setattr(io, "open", guarded_open)
    yield


@pytest.fixture(autouse=True)
def _no_real_converter():
    """Tests never start the real (GPU) converter by accident: the shared pool is
    replaced by one whose worker exits at once, so an un-faked conversion fails fast
    with a clear message. Tests that need conversion inject a backend or a fake pool
    (tests/support/converter.py)."""
    import sys

    from seedgraph.conversion import ConverterPool, set_pool

    disabled = ConverterPool(
        [None],
        command=[sys.executable, "-c",
                 "import sys; sys.stderr.write('real converter disabled in tests'); "
                 "sys.exit(2)"],
        expected_version="1.10.2",
    )
    old = set_pool(disabled)
    yield
    current = set_pool(old)
    if current is not None:
        current.shutdown()
