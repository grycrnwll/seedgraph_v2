"""Per-package concept manifests (stage 4 slice C9, part B): each package's frozen manifest is
its own content-addressed file, so ``check`` and ``save`` read one small file instead of a
library-wide manifest. Invented text only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from concepts._world import World, a, m  # tests/ is on sys.path (rootdir conftest)

from seedgraph.concepts import packages
from seedgraph.work import routine
from seedgraph.work.claims import WorkerRef
from seedgraph.work.store import WorkError
from seedgraph.work.tasks_concepts import MANIFEST_DIR, load_manifest

from .conftest import T0, at

W1 = WorkerRef("interactive", "w1", client="claude")
WORDS = ["cog", "lever", "flange", "pivot", "spindle", "ratchet", "detent", "bushing", "sprocket", "gimbal",
         "pawl", "cam", "spline", "collet", "arbor", "toggle"]


@pytest.fixture
def small_packages(monkeypatch):
    monkeypatch.setattr(packages, "PAIRS_PER_PACKAGE", 3)
    monkeypatch.setattr(packages, "CHECKS_PER_PACKAGE", 3)


def busy_world(w: World, n: int = 16) -> None:
    for i in range(n):
        x, y, z = WORDS[i % 16], WORDS[(i + 1) % 16], WORDS[(i + 3) % 16]
        w.note(f"lp_b{i:02d}", [
            a(f"{x} {y} condition variant", f"Invented: the {x} and the {y} move together in widget {i}."),
            a(f"{x} {y} restriction", f"Invented: the {x} restricts the {y} for gadget {i}."),
            m(f"{y} {z} estimator", f"Invented: estimate the {y} from the {z} for widget {i}."),
        ])


def _manifest_dir(lib) -> Path:
    return Path(lib.root).parent / "work" / MANIFEST_DIR


def _labels(lib, batch_id: str) -> dict:
    return json.loads((Path(lib.root).parent / "work" / batch_id / "labels.json").read_text(encoding="utf-8"))


def _index(lib, batch_id: str) -> dict[str, str]:
    sha = _labels(lib, batch_id)["manifest_index_sha256"]
    return json.loads((_manifest_dir(lib) / f"{sha}.json").read_text(encoding="utf-8"))


def _subjects(lib, batch_id: str) -> list[str]:
    return [r[0] for r in lib.conn.execute(
        "SELECT subject_id FROM batch_items WHERE batch_id = ? ORDER BY ordinal", (batch_id,))]


def _refs(manifest: dict) -> set[str]:
    return {r["claim_id"] for it in manifest["items"] for side in ("a", "b") if it[side]
            for r in it[side]["refs"].values()}


def test_each_package_has_its_own_content_addressed_manifest(lib, small_packages):
    busy_world(World(lib))
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption,method"}, now=T0)
    subjects = _subjects(lib, view.batch_id)
    assert len(subjects) >= 3
    index = _index(lib, view.batch_id)
    assert sorted(index) == sorted(subjects)
    assert len(set(index.values())) == len(subjects)  # one file per package
    sizes = {}
    for sid, sha in index.items():
        path = _manifest_dir(lib) / f"{sha}.json"
        own = json.loads(path.read_text(encoding="utf-8"))
        assert own == load_manifest(lib, view.batch_id, sid)
        assert own["subject_id"] == sid and set(own) == {"subject_id", "task", "family", "kind", "items"}
        sizes[sid] = (len(own["items"]), path.stat().st_size)
    # sizes scale with the package, and no file holds every package's refs
    small = min(sizes.values())
    big = max(sizes.values())
    assert big[0] > small[0] and big[1] > small[1]
    total_refs = set().union(*(_refs(load_manifest(lib, view.batch_id, s)) for s in subjects))
    assert all(len(_refs(load_manifest(lib, view.batch_id, s))) < len(total_refs) for s in subjects)
    assert len(json.dumps(_labels(lib, view.batch_id))) < 1000


def test_check_and_save_read_only_their_own_manifest(lib, small_packages):
    busy_world(World(lib))
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption,method"}, now=T0)
    msg = routine.claim_next(lib, view.batch_id, W1, now=at(1)).message
    index = _index(lib, view.batch_id)
    mine = load_manifest(lib, view.batch_id, msg["subject_id"])
    for sid, sha in index.items():  # every other package's manifest is gone
        if sid != msg["subject_id"]:
            (_manifest_dir(lib) / f"{sha}.json").unlink()
    answer = {"items": [{"item": it["item"], "verdict": "unsure", "confidence": "low"} for it in mine["items"]]}
    res = routine.submit(lib, msg["packet_id"], msg["claim_id"], note=answer, now=at(2))
    assert res.status == "accepted", res
    other = next(s for s in index if s != msg["subject_id"])
    with pytest.raises(WorkError, match="missing"):
        load_manifest(lib, view.batch_id, other)


def test_a_batch_with_the_old_single_manifest_is_refused_clearly(lib):
    busy_world(World(lib), n=4)
    view = routine.create_batch(lib, "concepts:bridge", options={"families": "assumption,method"}, now=T0)
    sid = _subjects(lib, view.batch_id)[0]
    path = Path(lib.root).parent / "work" / view.batch_id / "labels.json"
    labels = _labels(lib, view.batch_id)
    labels["manifest_sha256"] = labels.pop("manifest_index_sha256")
    path.write_text(json.dumps(labels), encoding="utf-8")
    with pytest.raises(WorkError, match="recreate"):
        load_manifest(lib, view.batch_id, sid)
