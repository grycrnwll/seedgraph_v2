"""JobRunner (design §6): resumes the conversion queue through the converter pool,
per-thread library connections, generic jobs; and the cross-process converter lock.

The pool is an in-process fake with the stage 0 pool's surface (``size``,
``converter_version``, ``convert``): no subprocess, no GPU. PDFs are synthetic and all
text is invented.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from seedgraph.conversion import ConvertOutput, ConverterError
from seedgraph.library import files
from seedgraph.library.store import open_library_at
from seedgraph.web.jobs import (
    SERVE_RUNNING_MESSAGE,
    ConverterLockHeld,
    JobFailed,
    JobResult,
    JobRunner,
    ThreadLibraries,
    acquire_converter_lock,
    check_converter_free,
    converter_lock_holder,
    converter_lock_path,
)

from library.test_files import PAGED_MD, TITLE, add_paper, text_pdf


class FakePool:
    def __init__(self, size: int = 2, *, fail: bool = False, delay: float = 0.0) -> None:
        self.size = size
        self.converter_version = "fake-9.9"
        self.fail = fail
        self.delay = delay
        self.calls: list[str] = []
        self.threads: set[int] = set()
        self._lock = threading.Lock()

    def convert(self, pdf_path, options=None, *, timeout=None) -> ConvertOutput:
        with self._lock:
            self.calls.append(str(pdf_path))
            self.threads.add(threading.get_ident())
        time.sleep(self.delay)
        if self.fail:
            raise ConverterError("invented converter failure")
        # Distinct text per PDF, as real papers have (identical markdown from two PDFs
        # converted at once is a separate case; see the U4 report).
        markdown = PAGED_MD + f"\nInvented note for {os.path.basename(str(pdf_path))}.\n"
        return ConvertOutput(markdown=markdown, page_count=2, metadata={}, warnings=[],
                             timings={}, converter_version=self.converter_version)


@pytest.fixture
def lib_dir(tmp_path):
    return tmp_path / "home" / "library"


def queue_papers(lib_dir, tmp_path, n: int) -> list[str]:
    """``n`` papers with a PDF attached and waiting in ``conversion_queue``."""
    lib = open_library_at(lib_dir)
    try:
        out = []
        for i in range(n):
            pid = add_paper(lib, paper_id=f"lp_{i + 1:020d}", title=f"{TITLE} {i}")
            res = files.attach_pdf(lib, pid, text_pdf(tmp_path / f"p{i}.pdf", [f"{TITLE} {i}"],
                                                      marker=f"n{i}"))
            out.append(res.source_file_id)
        assert sorted(files.queued_files(lib)) == sorted(out)
        return out
    finally:
        lib.close()


def queued_now(lib_dir) -> list[str]:
    lib = open_library_at(lib_dir)
    try:
        return files.queued_files(lib)
    finally:
        lib.close()


@pytest.fixture
def runner_factory(lib_dir):
    made = []

    def _make(**kw):
        r = JobRunner(lib_dir, **kw)
        made.append(r)
        return r

    yield _make
    for r in made:
        r.stop()


# --- conversion queue ------------------------------------------------------------------


def test_start_resumes_the_conversion_queue_through_the_pool(lib_dir, tmp_path, runner_factory):
    sfs = queue_papers(lib_dir, tmp_path, 3)
    pool = FakePool(size=2)
    runner = runner_factory(pool=pool)

    job_ids = runner.start()

    assert len(job_ids) == 3
    assert runner.wait_idle(timeout=30)
    assert len(pool.calls) == 3
    assert queued_now(lib_dir) == []
    jobs = {j.job_id: j for j in runner.jobs()}
    assert {jobs[j].state for j in job_ids} == {"done"}, [jobs[j].message for j in job_ids]
    assert all(jobs[j].kind == "convert" and jobs[j].paper_ids for j in job_ids)
    lib = open_library_at(lib_dir)
    try:
        linked = {r[0] for r in lib.conn.execute(
            "SELECT source_file_id FROM paper_files WHERE markdown_id IS NOT NULL")}
    finally:
        lib.close()
    assert linked == set(sfs)


def test_both_converters_are_kept_busy(lib_dir, tmp_path, runner_factory):
    queue_papers(lib_dir, tmp_path, 4)
    pool = FakePool(size=2, delay=0.2)
    runner = runner_factory(pool=pool)
    runner.start()
    assert runner.wait_idle(timeout=30)
    assert len(pool.threads) == 2


def test_a_failed_conversion_stays_queued_and_the_runner_goes_idle(lib_dir, tmp_path,
                                                                   runner_factory):
    [sf] = queue_papers(lib_dir, tmp_path, 1)
    pool = FakePool(size=1, fail=True)
    runner = runner_factory(pool=pool)
    [job_id] = runner.start()
    assert runner.wait_idle(timeout=30)
    job = runner.job(job_id)
    assert job.state == "failed" and "invented converter failure" in job.message
    assert queued_now(lib_dir) == [sf]  # durable: the next start picks it up
    time.sleep(0.2)
    assert len(pool.calls) == 1  # never retried in a loop


def test_a_restart_resumes_what_was_left_queued(lib_dir, tmp_path, runner_factory):
    [sf] = queue_papers(lib_dir, tmp_path, 1)
    first = runner_factory(pool=FakePool(size=1, fail=True))
    first.start()
    assert first.wait_idle(timeout=30)
    first.stop()

    second = runner_factory(pool=FakePool(size=1))
    [job_id] = second.start()
    assert second.wait_idle(timeout=30)
    assert second.job(job_id).state == "done"
    assert queued_now(lib_dir) == []


def test_converting_a_file_twice_while_active_reuses_the_job(lib_dir, tmp_path, runner_factory):
    [sf] = queue_papers(lib_dir, tmp_path, 1)
    runner = runner_factory(pool=FakePool(size=1, delay=0.3))
    [job_id] = runner.start()
    assert runner.convert(sf) == job_id
    assert runner.wait_idle(timeout=30)


def test_start_is_idempotent_and_an_empty_queue_starts_idle(lib_dir, runner_factory):
    open_library_at(lib_dir).close()
    runner = runner_factory(pool=FakePool(size=1))
    assert runner.start() == []
    assert runner.start() == []
    assert runner.wait_idle(timeout=1)
    assert runner.summary().active == 0


# --- generic jobs and per-thread connections -------------------------------------------


def test_submit_runs_a_job_with_its_threads_own_library(lib_dir, runner_factory):
    open_library_at(lib_dir).close()
    seen = []

    def job(lib, n, *, tag):
        seen.append((threading.get_ident(), id(lib), lib.conn.execute("SELECT ?", (n,)).fetchone()[0]))
        return JobResult(f"{tag} {n}", paper_ids=("lp_1",))

    runner = runner_factory(pool=FakePool(size=1), workers=2)
    runner.start()
    ids = [runner.submit("probe", job, i, tag="did") for i in range(6)]
    assert runner.wait_idle(timeout=30)

    by_thread: dict[int, set[int]] = {}
    for tid, lib_id, _ in seen:
        by_thread.setdefault(tid, set()).add(lib_id)
    assert all(len(libs) == 1 for libs in by_thread.values())  # one Library per thread
    all_libs = [next(iter(v)) for v in by_thread.values()]
    assert len(set(all_libs)) == len(all_libs)  # never shared across threads
    assert threading.get_ident() not in by_thread
    assert [runner.job(i).message for i in ids] == [f"did {i}" for i in range(6)]
    assert runner.job(ids[0]).paper_ids == ("lp_1",)


def test_a_job_that_raises_is_marked_failed_and_the_worker_survives(lib_dir, runner_factory):
    open_library_at(lib_dir).close()
    runner = runner_factory(pool=FakePool(size=1))
    runner.start()

    def bad(lib):
        raise RuntimeError("invented failure")

    def soft(lib):
        raise JobFailed("not found", paper_ids=("lp_2",))

    a = runner.submit("x", bad)
    b = runner.submit("x", soft)
    c = runner.submit("x", lambda lib: "ok")
    assert runner.wait_idle(timeout=30)
    assert runner.job(a).state == "failed" and "invented failure" in runner.job(a).message
    assert (runner.job(b).state, runner.job(b).paper_ids) == ("failed", ("lp_2",))
    assert (runner.job(c).state, runner.job(c).message) == ("done", "ok")
    assert runner.jobs(active_only=True) == []
    assert runner.summary().failed == 2


def test_thread_libraries_give_each_thread_its_own_library(lib_dir):
    open_library_at(lib_dir).close()
    libs = ThreadLibraries(lib_dir)
    mine = libs.get()
    assert libs.get() is mine
    other = []
    t = threading.Thread(target=lambda: (other.append(libs.get()), libs.close_current()))
    t.start()
    t.join()
    assert other[0] is not mine
    libs.close_all()


# --- converter lock --------------------------------------------------------------------


def test_the_lock_is_held_until_released(tmp_path):
    home = tmp_path / "home"
    lock = acquire_converter_lock(home)
    assert converter_lock_path(home).read_text(encoding="ascii") == str(os.getpid())
    assert converter_lock_holder(home) == os.getpid()
    with pytest.raises(ConverterLockHeld) as err:
        check_converter_free(home)
    assert SERVE_RUNNING_MESSAGE in str(err.value)
    with pytest.raises(ConverterLockHeld):
        acquire_converter_lock(home)
    lock.release()
    assert not converter_lock_path(home).exists()
    check_converter_free(home)
    with acquire_converter_lock(home):
        assert converter_lock_holder(home) == os.getpid()
    assert converter_lock_holder(home) is None


@pytest.mark.parametrize("content", ["999999999", "not a pid", ""])
def test_a_stale_lock_is_ignored_and_replaced(tmp_path, content):
    home = tmp_path / "home"
    path = converter_lock_path(home)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="ascii")
    assert converter_lock_holder(home) is None
    check_converter_free(home)
    with acquire_converter_lock(home):
        assert converter_lock_holder(home) == os.getpid()


def test_release_leaves_a_lock_that_another_process_took_over(tmp_path):
    home = tmp_path / "home"
    lock = acquire_converter_lock(home)
    converter_lock_path(home).write_text("424242", encoding="ascii")
    lock.release()
    assert converter_lock_path(home).read_text(encoding="ascii") == "424242"
