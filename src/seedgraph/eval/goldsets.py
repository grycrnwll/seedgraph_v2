"""Load + INLINE required-key validation of gold/probe/checklist JSONL (phase_9 §4/§5).

Over-engineering deliberately removed (phase_9 §4/§12): there is NO ``evals/schemas/``
directory, NO JSON-Schema files, and NO ``jsonschema`` dependency. Each file type
is validated against a small constant required-key set held here; a missing OR
extra key raises a clear ``ValueError`` naming the offending line. Stdlib ``json``
only.

Per-project gold/probe/checklist files live under ``projects/{slug}/eval/`` and are
PRIVATE-by-default (they may quote spans / encode corpus-specific relevance — doc
03 §6 / doc 12 §5). Only the hand-authored repo-level ``evals/fixtures`` (no
restricted full text) is shareable.
"""

from __future__ import annotations

import json
from pathlib import Path

# Exact required keys per file type (phase_9 §4). Row keys must EQUAL the set —
# any missing or extra key is a ValueError.
RETRIEVAL_GOLD_KEYS: frozenset[str] = frozenset(
    {"question", "query_type", "relevant_work_ids", "relevant_span_ids"}
)
LEAKAGE_PROBE_KEYS: frozenset[str] = frozenset({"question", "rationale"})

REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "retrieval_gold": RETRIEVAL_GOLD_KEYS,
    "leakage_probes": LEAKAGE_PROBE_KEYS,
}

# Encoded conversion-checklist item vocab (doc 09 §3) — one boolean per item per work.
CONVERSION_CHECKLIST_ITEMS: tuple[str, ...] = (
    "section_headers",
    "paragraph_text",
    "equations",
    "inline_math",
    "tables",
    "figures_captions",
    "assumptions_theorems_propositions",
    "estimands",
    "references",
    "page_markers",
)

# Econometrics-prioritized subset — flagged priority=true so report.py can weight
# / surface them (phase_9 §4). Optional and NOT gated.
CONVERSION_PRIORITY_ITEMS: frozenset[str] = frozenset(
    {"assumptions_theorems_propositions", "equations", "estimands", "tables", "references"}
)


def load_jsonl(path: Path, kind: str) -> list[dict]:
    """Load a gold/probe JSONL file and validate each row inline against
    ``REQUIRED_KEYS[kind]``.

    Parses one JSON object per (non-blank) line; asserts each row's key set
    EQUALS the required set for ``kind`` (missing OR extra key → ``ValueError``
    that names the file + 1-indexed line). Returns the parsed rows. No JSON-Schema,
    no third-party validator (phase_9 §4).
    """
    if kind not in REQUIRED_KEYS:
        raise ValueError(
            f"unknown gold/probe kind {kind!r}; expected one of {sorted(REQUIRED_KEYS)}"
        )
    required = set(REQUIRED_KEYS[kind])
    path = Path(path)
    rows: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {lineno}: invalid JSON ({exc})") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"{path} line {lineno}: row is not a JSON object")
        keys = set(obj)
        if keys != required:
            missing = sorted(required - keys)
            extra = sorted(keys - required)
            raise ValueError(
                f"{path} line {lineno}: key mismatch for {kind} "
                f"(missing={missing}, extra={extra})"
            )
        rows.append(obj)
    return rows


def load_retrieval_gold(path: Path) -> list[dict]:
    """Load ``retrieval_gold.jsonl`` rows
    (``{question, query_type, relevant_work_ids[], relevant_span_ids[]}``).
    Thin wrapper over :func:`load_jsonl` with ``kind='retrieval_gold'``.
    """
    return load_jsonl(path, "retrieval_gold")


def load_leakage_probes(path: Path) -> list[dict]:
    """Load ``leakage_probes.jsonl`` rows (``{question, rationale}``). Each probe is
    OUT-of-corpus and must elicit ``insufficient_evidence=true`` (doc 09 §9). Thin
    wrapper over :func:`load_jsonl` with ``kind='leakage_probes'``.
    """
    return load_jsonl(path, "leakage_probes")


def load_conversion_checklist(path: Path) -> list[dict]:
    """Load the OPTIONAL encoded ``conversion_checklist.jsonl`` (one row per work,
    booleans over :data:`CONVERSION_CHECKLIST_ITEMS`; rows flag the
    :data:`CONVERSION_PRIORITY_ITEMS` subset). Validates each declared item is a
    bool; unknown item keys → ``ValueError``. Optional and NOT gated (doc 09 §3).
    """
    path = Path(path)
    allowed = set(CONVERSION_CHECKLIST_ITEMS) | {"work_id", "priority"}
    rows: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"{path} line {lineno}: row is not a JSON object")
        for key, value in obj.items():
            if key not in allowed:
                raise ValueError(
                    f"{path} line {lineno}: unknown checklist item {key!r} "
                    f"(allowed: {sorted(allowed)})"
                )
            if key in CONVERSION_CHECKLIST_ITEMS and not isinstance(value, bool):
                raise ValueError(
                    f"{path} line {lineno}: checklist item {key!r} must be a bool"
                )
        # Surface the econometrics-prioritized subset so report.py can weight them.
        obj.setdefault(
            "priority",
            any(obj.get(item) for item in CONVERSION_PRIORITY_ITEMS),
        )
        rows.append(obj)
    return rows
