"""``web/static/js/fold.js`` against the shared fold vectors (design §3.2).

Opt-in: runs only when Node.js is on PATH and ``SEEDGRAPH_NODE_TESTS=1`` (the browser
suite runs the same vectors through Playwright). Invented text only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FOLD_JS = REPO / "src" / "seedgraph" / "web" / "static" / "js" / "fold.js"
VECTORS = REPO / "tests" / "fixtures" / "textfold_vectors.json"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or os.environ.get("SEEDGRAPH_NODE_TESTS") != "1",
    reason="opt-in: needs node on PATH and SEEDGRAPH_NODE_TESTS=1",
)

RUNNER = r"""
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const [foldPath, vecPath] = process.argv.slice(1);
const { fold, findPassage } = await import(pathToFileURL(foldPath).href);
const v = JSON.parse(readFileSync(vecPath, "utf8"));
const out = {
  fold: v.fold.map((x) => fold(x.raw).text),
  map: v.map.map((x) => { const f = fold(x.raw); return [f.text, f.starts.map((s, i) => [s, f.ends[i]])]; }),
  find: v.find.map((x) => findPassage(x.haystack, x.needle)),
};
process.stdout.write(JSON.stringify(out));
"""


def _run_js() -> dict:
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", RUNNER, str(FOLD_JS), str(VECTORS)],
        capture_output=True, check=True, timeout=60,
    )
    return json.loads(proc.stdout.decode("utf-8"))


def test_fold_js_matches_every_shared_vector():
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))
    got = _run_js()
    assert got["fold"] == [v["folded"] for v in vectors["fold"]]
    assert got["map"] == [[v["folded"], v["spans"]] for v in vectors["map"]]
    assert got["find"] == [v["hits"] for v in vectors["find"]]
