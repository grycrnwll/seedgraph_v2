"""One process owns the converters (docs/V3_UI_DESIGN.md §6): while ``serve`` holds
``<home>/run/converter.lock``, ``seedgraph convert`` and ``library reconvert`` refuse with
the lock message. The holder here is a live helper process; nothing real is converted."""

from __future__ import annotations

import subprocess
import sys

import pytest
from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.library import reconvert
from seedgraph.library.store import open_library
from seedgraph.web.jobs import SERVE_RUNNING_MESSAGE, ConverterLockHeld, acquire_converter_lock, converter_lock_path

from .metadata_support import insert_paper
from .test_files import text_pdf


@pytest.fixture
def home(isolated_home):
    open_library(isolated_home).close()
    return isolated_home


@pytest.fixture
def serve_holds_lock(home):
    """Another live process (standing in for ``seedgraph serve``) holds the lock."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    path = converter_lock_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(proc.pid), encoding="ascii")
    yield proc.pid
    proc.kill()
    proc.wait()


def test_convert_refuses_while_serve_holds_the_lock(home, serve_holds_lock, tmp_path):
    pdf = text_pdf(tmp_path / "a.pdf", ["An invented cover"])
    result = CliRunner().invoke(app, ["convert", str(pdf), "--root", str(home)])
    assert result.exit_code != 0
    assert SERVE_RUNNING_MESSAGE in result.output


def test_library_reconvert_refuses_while_serve_holds_the_lock(home, serve_holds_lock):
    result = CliRunner().invoke(app, ["library", "reconvert", "lp_any", "--ocr", "--root", str(home)])
    assert result.exit_code != 0
    assert SERVE_RUNNING_MESSAGE in result.output


def test_reconvert_refuses_before_touching_the_paper(home, serve_holds_lock, tmp_path):
    with open_library(home) as lib:
        insert_paper(lib, "lp_locked", title="An Invented Paper")
        with pytest.raises(ConverterLockHeld):
            reconvert.reconvert_ocr(lib, "lp_locked")


def test_serve_itself_may_reconvert_while_it_holds_the_lock(home, tmp_path):
    # The serve process's own jobs are not refused by its own lock.
    with acquire_converter_lock(home), open_library(home) as lib:
        insert_paper(lib, "lp_mine", title="An Invented Paper")
        with pytest.raises(reconvert.LibraryError, match="no PDF attached"):
            reconvert.reconvert_ocr(lib, "lp_mine")


def test_a_stale_lock_does_not_refuse(home, tmp_path):
    path = converter_lock_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("999999999", encoding="ascii")
    result = CliRunner().invoke(app, ["convert", "--root", str(home)])
    assert SERVE_RUNNING_MESSAGE not in result.output
