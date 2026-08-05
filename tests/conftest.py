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
    ("seedgraph.spans.store", "ensure_span"),
    ("seedgraph.citation.edges", "write_edge"),
    ("seedgraph.citation.bib_parser", "parse_references"),
    ("seedgraph.sections.parser", "parse_sections"),
    ("seedgraph.llm.executor", "run_llm"),
    ("seedgraph.llm.secrets", "resolve_named_secret"),
    ("seedgraph.project.identity", "upsert_work"),
    ("seedgraph.project.identity", "merge_decision"),
    ("seedgraph.doctor", "collect_checks"),
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
        "SEEDGRAPH_LOG_LEVEL",
    ):
        monkeypatch.delenv(var, raising=False)
    # Same hermeticity for the OS keyring (ADR-0002 made `key_source: auto`
    # consult it): the suite must never read this machine's real credential
    # store. Tests that exercise keyring behavior re-patch ``_keyring_get``
    # themselves (their later patch wins within the test).
    from seedgraph.llm import secrets

    monkeypatch.setattr(secrets, "_keyring_get", lambda name: None)
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

    def guarded_open(file, mode="r", *args, **kwargs):
        root = os.environ.get(
            "SEEDGRAPH_REGRESSION_CORPUS", "D:/Documents/research/paper_library"
        )
        lib_str = str(Path(root)).lower()
        try:
            p = str(Path(file)).lower()
        except Exception:  # noqa: BLE001 - int fds, bytes paths: never guarded
            p = str(file).lower()
        if p.startswith(lib_str) and any(m in mode for m in ("w", "a", "x", "+")):
            raise AssertionError(
                "HARD RULE violation: attempted write-open of read-only corpus: "
                f"{file!r} (mode={mode!r})"
            )
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    # builtins.open and io.open are the same object at interpreter start;
    # pathlib calls the latter, so both names must point at the wrapper.
    monkeypatch.setattr(io, "open", guarded_open)
    yield
