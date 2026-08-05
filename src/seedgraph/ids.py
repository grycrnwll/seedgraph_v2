"""ID scheme (decisions 7/29/44/65, ratified as D1).

Two id families:

* **Mutable identity rows** (works, projects, claims, concepts, runs, ...) get
  opaque uuid ids: ``new_id(prefix) -> f"{prefix}_{uuid4().hex}"``. Hashes are
  never their primary key.
* **Immutable, content-defined cache blobs** are the explicit exception: their id
  *is* the content hash, prefixed — ``source_file_id = "sf_" + sha256_hex(bytes)``,
  ``markdown_id = "md_" + sha256_hex(bytes)``. Use ``content_id`` for the
  string-parts variant; use ``sha256_hex`` directly on file/markdown bytes.

All content hashes are SHA-256, lowercase hex, stored bare in columns.
"""

from __future__ import annotations

import hashlib
import uuid

# Known id prefixes (registry — extended by later phases as new entities land).
PREFIXES: frozenset[str] = frozenset(
    {
        # content-addressed cache blobs (id == hash)
        "sf",  # source_file
        "md",  # markdown_document
        # deterministic derived project artifacts (id == f(content hash))
        "sec",  # document_section (deterministic: sha256(markdown_hash|ordinal))
        "span",  # evidence_span (manual: uuid; auto: deterministic)
        # mutable identity rows (uuid)
        "work",
        "proj",
        "claim",
        "concept",
        "run",
        # project-scope opaque rows
        "rq",  # review_queue item
        "ue",  # llm_usage_event
    }
)


def sha256_hex(data: bytes) -> str:
    """SHA-256 of ``data`` as a lowercase hex string."""
    return hashlib.sha256(data).hexdigest()


def new_id(prefix: str) -> str:
    """Mint an opaque, globally-unique id: ``f"{prefix}_{uuid4().hex}"``."""
    return f"{prefix}_{uuid.uuid4().hex}"


def content_id(prefix: str, *parts: str) -> str:
    """Deterministic content-addressed id over string ``parts``.

    ``content_id(p, a, b) == content_id(p, a, b)`` always. Parts are joined on
    ``"|"`` before hashing.
    """
    digest = sha256_hex("|".join(parts).encode("utf-8"))
    return f"{prefix}_{digest}"


# --- phase_3: deterministic derived ids (sections + auto paragraph spans) ----
# Keyed on the markdown *content hash* (decisions 2/24/42/52), so re-parsing /
# re-indexing identical markdown bytes yields identical ids — the idempotency
# substrate. Under D1 the cache ``markdown_id`` is itself ``"md_"+hash``, so
# keying on ``markdown_hash`` is equivalent to keying on ``markdown_id`` and is
# rebuild-safe by construction.


def section_id(markdown_hash: str, ordinal: int) -> str:
    """Deterministic ``document_sections.section_id`` — ``'sec_' + sha256(markdown_hash|ordinal)[:16]``.

    Stable across two parses of identical markdown and unchanged when the cache
    ``markdown_id`` row id differs but bytes are identical (rebuild-safety).
    """
    digest = sha256_hex(f"{markdown_hash}|{int(ordinal)}".encode("utf-8"))
    return f"sec_{digest[:16]}"


def auto_span_id(markdown_hash: str, start: int, end: int, span_kind: str) -> str:
    """Deterministic auto-span id — ``'span_' + sha256(markdown_hash|start|end|span_kind)[:24]``.

    Re-indexing the same markdown produces the SAME ids, so paragraph spans never
    duplicate (no duplicate rows, no duplicate ``span_fts`` rows). Manual spans use
    the opaque random :func:`new_id` instead.
    """
    digest = sha256_hex(
        f"{markdown_hash}|{int(start)}|{int(end)}|{span_kind}".encode("utf-8")
    )
    return f"span_{digest[:24]}"
