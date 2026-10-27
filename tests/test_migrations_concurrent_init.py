"""Several threads initialising the same fresh home at once (queue threads share one
process) must all succeed and leave the schema at the latest version."""

from __future__ import annotations

import sqlite3
import threading

from seedgraph.cache.db import init_cache_db
from seedgraph.db.bootstrap import ensure_project_db
from seedgraph.db.connection import cache_db_path, project_db_path
from seedgraph.db.migrations import current_version, latest_version

THREADS = 8
ROUNDS = 10


def _race(target) -> list[BaseException]:
    barrier = threading.Barrier(THREADS)
    errors: list[BaseException] = []

    def run():
        barrier.wait()
        try:
            target()
        except BaseException as exc:  # noqa: BLE001 - collected and asserted on
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def _version(path) -> int:
    conn = sqlite3.connect(path)
    try:
        return current_version(conn)
    finally:
        conn.close()


def test_concurrent_init_cache_db_on_fresh_home(tmp_path):
    for i in range(ROUNDS):
        home = tmp_path / f"home{i}"
        errors = _race(lambda: init_cache_db(home))
        assert errors == []
        assert _version(cache_db_path(home)) == latest_version("cache")


def test_concurrent_ensure_project_db_on_fresh_home(tmp_path):
    for i in range(ROUNDS):
        home = tmp_path / f"home{i}"
        errors = _race(lambda: ensure_project_db("proj", home))
        assert errors == []
        assert _version(project_db_path("proj", home)) == latest_version("project")


