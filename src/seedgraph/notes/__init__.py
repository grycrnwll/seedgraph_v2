"""Notes boundary on the library (v3 stage 2, design §3, §4).

- :mod:`.format` - schema 2.0.0 envelopes, ``validate_note``, ``check_note``,
  ``normalize_parts``.
- :mod:`.vocab` - fixed ``*_type`` vocabularies and the variant table (``map_label``).
- :mod:`.spans` - the library span writer (``ensure_span``, ``read_markdown``).
- :mod:`.save` - ``save_note``: one new version, anchored, in the caller's transaction.
- :mod:`.instructions` / :mod:`.labels` - ``instructions.md`` text and ``labels.json``.
"""

from .format import (
    PROMPT_VERSION,
    SCHEMA_ID,
    SCHEMA_VERSION,
    ClaimDraft,
    Issue,
    NormalizedNote,
    NoteCheck,
    NoteV2,
    check_note,
    normalize_note,
    normalize_parts,
    validate_note,
)
from .instructions import render_instructions
from .labels import label_groups, labels_json
from .save import (
    UNCHECKED,
    NoteConflict,
    NoteProvenance,
    NoteSaveError,
    NoteStale,
    SavedNote,
    current_note_id,
    save_note,
)
from .spans import ensure_span, read_markdown
from .vocab import VOCABULARIES, map_label

__all__ = [
    "PROMPT_VERSION", "SCHEMA_ID", "SCHEMA_VERSION", "UNCHECKED", "VOCABULARIES", "ClaimDraft",
    "Issue", "NormalizedNote", "NoteCheck", "NoteConflict", "NoteProvenance", "NoteSaveError",
    "NoteStale", "NoteV2", "SavedNote", "check_note", "current_note_id", "ensure_span",
    "label_groups", "labels_json", "map_label", "normalize_note", "normalize_parts",
    "read_markdown", "render_instructions", "save_note", "validate_note",
]
