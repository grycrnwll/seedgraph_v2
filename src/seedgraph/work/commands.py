"""The copyable "work on this batch" command text (design §8), from one template so the
batch reply, the CLI (``seedgraph work command``) and the stage 3 UI never drift.

During stages 2-6 the skill and MCP server carry the ``-v3`` / ``_v3`` suffixes (§7.3,
§7.4); stage 7 drops them by changing the two names below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

WORK_SKILL_NAME = "seedgraph-v3-work"
MCP_SERVER_NAME = "seedgraph_v3"

CLAUDE_CODE_TEMPLATE = "/{skill} {batch_id}"
CODEX_TEMPLATE = (
    "Use the {skill} skill and the {server} MCP tools to work Seedgraph batch {batch_id} "
    "until it is finished or paused. Submit results with note_path; follow the feedback."
)


@dataclass(frozen=True)
class WorkCommands:
    claude_code: str
    codex: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def work_commands(batch_id: str) -> WorkCommands:
    """The Claude Code skill command and the Codex prompt for ``batch_id``."""
    fields = {"skill": WORK_SKILL_NAME, "server": MCP_SERVER_NAME, "batch_id": batch_id}
    return WorkCommands(CLAUDE_CODE_TEMPLATE.format(**fields), CODEX_TEMPLATE.format(**fields))
