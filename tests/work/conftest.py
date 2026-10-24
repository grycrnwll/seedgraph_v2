"""Shared fixtures for the work protocol state core (invented subjects only)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from seedgraph.library import store as libstore
from seedgraph.work import store

T0 = datetime(2026, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
TTL = 600


def at(seconds: float) -> datetime:
    """T0 plus ``seconds``: an injected clock, so no test sleeps."""
    return T0 + timedelta(seconds=seconds)


@pytest.fixture
def lib(tmp_path):
    library = libstore.open_library_at(tmp_path / "library")
    yield library
    library.close()


def make_batch(lib, parts=(1,), *, ttl=TTL, task_type="notes", now=T0) -> str:
    """A batch with one queued paper per entry of ``parts`` (its part count), and its
    packages. Papers are soft refs, so none need to exist."""
    with lib.write() as conn:
        batch_id = store.insert_batch(
            conn, task_type=task_type, task_version="2.0.0",
            selection=[f"lp_{i:02d}" for i in range(1, len(parts) + 1)],
            options={"claim_ttl_s": ttl}, now=now,
        )
        for ordinal, n in enumerate(parts, start=1):
            store.insert_item(conn, batch_id, ordinal, f"lp_{ordinal:02d}",
                              markdown_id=f"md_{ordinal:02d}", markdown_hash=f"h{ordinal:02d}",
                              part_count=n, now=now)
            for part in range(1, n + 1):
                store.insert_package(conn, batch_id=batch_id, ordinal=ordinal, part=part, part_count=n,
                                     markdown_hash=f"h{ordinal:02d}", task_version="2.0.0",
                                     start_char=(part - 1) * 100, end_char=part * 100,
                                     est_tokens=25, now=now)
    return batch_id
