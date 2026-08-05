"""Self-test for conftest's autouse ``_guard_paper_library`` write guard.

The guard is retargeted at a tmp dir via ``SEEDGRAPH_REGRESSION_CORPUS`` so
these tests never go anywhere near the real machine-local corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def guarded_root(tmp_path, monkeypatch):
    """A tmp corpus holding one pre-existing note, installed as the guarded root.

    The note is written BEFORE the env var points the guard at the tmp dir,
    so its creation is not itself blocked.
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "note.md").write_text("existing", encoding="utf-8")
    monkeypatch.setenv("SEEDGRAPH_REGRESSION_CORPUS", str(corpus))
    return corpus


@pytest.mark.parametrize("mode", ["w", "a", "x", "r+", "w+", "wb", "ab"])
def test_write_open_under_root_raises(guarded_root, mode):
    with pytest.raises(AssertionError, match="HARD RULE"):
        open(guarded_root / "note.md", mode)
    assert (guarded_root / "note.md").read_text(encoding="utf-8") == "existing"


def test_write_open_never_creates_new_file(guarded_root):
    with pytest.raises(AssertionError, match="HARD RULE"):
        open(guarded_root / "brand_new.md", "w")
    assert not (guarded_root / "brand_new.md").exists()


def test_pathlib_write_under_root_raises(guarded_root):
    # pathlib's helpers go through io.open, which the guard also wraps.
    with pytest.raises(AssertionError, match="HARD RULE"):
        (guarded_root / "note.md").write_text("clobber", encoding="utf-8")
    assert (guarded_root / "note.md").read_text(encoding="utf-8") == "existing"


def test_guard_is_case_insensitive(guarded_root):
    shouted = Path(str(guarded_root).upper()) / "note.md"
    with pytest.raises(AssertionError, match="HARD RULE"):
        open(shouted, "w")


def test_read_open_under_root_passes(guarded_root):
    with open(guarded_root / "note.md", encoding="utf-8") as fh:
        assert fh.read() == "existing"
    with open(guarded_root / "note.md", "rb") as fh:
        assert fh.read() == b"existing"


def test_write_outside_root_untouched(guarded_root, tmp_path):
    outside = tmp_path / "outside.txt"
    with open(outside, "w", encoding="utf-8") as fh:
        fh.write("ok")
    assert outside.read_text(encoding="utf-8") == "ok"
