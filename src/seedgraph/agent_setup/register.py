"""Register the v3 MCP server and skills with Claude Code and Codex (design §7.3, §7.4).

Show first, write only when asked:

    entry  = server_entry(worktree=..., home=...)           # the seedgraph_v3 launch entry
    change = plan_claude_config(path_to_claude_json, entry)  # or plan_codex_config(config_toml, entry)
    print(change.render())                                   # ONLY the seedgraph_v3 block
    apply_change(change, write=True)                         # nothing is written without write=True

    skills = plan_skills(skills_dir, layout="skills")        # seedgraph-v3, seedgraph-v3-work
    apply_skills(skills, write=True)

``plan_setup`` bundles these for ``seedgraph setup agents --client claude|codex|both``.

**Other entries are never touched or shown.** A config file can hold other servers'
keys and tokens, so the edit is a text splice of the one ``seedgraph_v3`` block: every
other byte of the file is kept, which the planner verifies by parsing the old and new
text (``json`` / ``tomllib``) and comparing everything outside ``seedgraph_v3``. Rendered
output holds only that block; values in an existing ``seedgraph_v3`` block that seedgraph
did not write (other env keys, headers) are shown as ``<redacted>``. Errors name a line
and column, never file text. The full new file text is kept off ``repr``.

Nothing here has a default path into the user's real configuration: callers pass every
path. :func:`default_targets` only *names* the usual locations for a given user home.
"""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from ..errors import SeedgraphError
from ..work.commands import MCP_SERVER_NAME, WORK_SKILL_NAME

SERVER_NAME = MCP_SERVER_NAME  # 'seedgraph_v3' during stages 2-6
READ_SKILL_NAME = "seedgraph-v3"
SKILL_SOURCES = {READ_SKILL_NAME: "seedgraph_v3_skill.md", WORK_SKILL_NAME: "seedgraph_v3_work_skill.md"}
CODEX_STARTUP_TIMEOUT_S = 60
#: design §13 Q12: Seedgraph never edits Claude Code's permission settings.
RECORD_ANSWER_HINT = (
    "Claude Code asks permission the first time an agent calls `record_answer` (it saves a record, so it is not "
    "read-only). Approve it once with \"always allow\"; Seedgraph does not edit Claude Code's permission settings.")
OUR_ENV_KEYS = ("SEEDGRAPH_HOME", "PYTHONUTF8", "PYTHONIOENCODING")
REDACTED = "<redacted>"

Client = Literal["claude", "codex"]
Action = Literal["add", "replace", "unchanged"]


class RegisterError(SeedgraphError):
    """A registration change was refused (unparseable file, a form we cannot edit
    safely, or the file changed since the plan)."""


# --- the entry -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerEntry:
    command: str
    args: tuple[str, ...]
    env: dict[str, str]
    startup_timeout_sec: int = CODEX_STARTUP_TIMEOUT_S
    name: str = SERVER_NAME


def server_entry(*, worktree: Path | str, home: Path | str, launcher: Literal["exe", "python"] = "exe",
                 python: Path | str | None = None) -> ServerEntry:
    """The ``seedgraph_v3`` entry: the worktree venv's absolute ``seedgraph-v3.exe`` (or its
    ``python -m seedgraph``) with ``mcp serve``, an explicit ``SEEDGRAPH_HOME`` and UTF-8 stdio."""
    scripts = Path(worktree) / ".venv" / "Scripts"
    env = {"SEEDGRAPH_HOME": str(Path(home)), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    if launcher == "exe":
        return ServerEntry(str(scripts / "seedgraph-v3.exe"), ("mcp", "serve"), env)
    if launcher == "python":
        return ServerEntry(str(Path(python) if python else scripts / "python.exe"),
                           ("-m", "seedgraph", "mcp", "serve"), env)
    raise RegisterError(f"launcher must be 'exe' or 'python', not {launcher!r}")


def claude_add_command(entry: ServerEntry) -> str:
    """The equivalent ``claude mcp add --scope user`` command line (for display)."""
    argv = ["claude", "mcp", "add", "--scope", "user", entry.name]
    for key, value in entry.env.items():
        argv += ["--env", f"{key}={value}"]
    argv += ["--", entry.command, *entry.args]
    return subprocess.list2cmdline(argv)


def claude_value(entry: ServerEntry) -> dict[str, Any]:
    return {"type": "stdio", "command": entry.command, "args": list(entry.args), "env": dict(entry.env)}


def _toml_str(value: str) -> str:
    if "'" not in value and not any(ord(c) < 0x20 or c == "\x7f" for c in value):
        return f"'{value}'"
    return json.dumps(value, ensure_ascii=False)


def codex_block(entry: ServerEntry, newline: str = "\n") -> str:
    """The ``[mcp_servers.seedgraph_v3]`` TOML block (with its ``.env`` subtable)."""
    lines = [
        f"[mcp_servers.{entry.name}]",
        f"command = {_toml_str(entry.command)}",
        "args = [" + ", ".join(json.dumps(a, ensure_ascii=False) for a in entry.args) + "]",
        f"startup_timeout_sec = {entry.startup_timeout_sec}",
        "",
        f"[mcp_servers.{entry.name}.env]",
        *(f"{k} = {json.dumps(v, ensure_ascii=False) if k != 'SEEDGRAPH_HOME' else _toml_str(v)}"
          for k, v in entry.env.items()),
    ]
    return newline.join(lines) + newline


# --- the change ----------------------------------------------------------------------------------

@dataclass
class ConfigChange:
    """A planned edit of one config file. ``render()`` is the only thing to show."""

    client: Client
    path: Path
    action: Action
    entry: ServerEntry
    new_block: str
    old_block: str | None = None  # redacted
    original: bytes | None = field(default=None, repr=False)  # None: the file does not exist
    new_bytes: bytes = field(default=b"", repr=False)

    def render(self) -> str:
        where = (f'Claude Code (user scope, "mcpServers"): {self.path}' if self.client == "claude"
                 else f"Codex: {self.path}")
        verb = {"add": f'add "{self.entry.name}"', "replace": f'replace "{self.entry.name}"',
                "unchanged": f'"{self.entry.name}" is already registered as shown; nothing to write'}[self.action]
        out = [where, f"Change: {verb}. Every other entry and setting in this file is kept byte for byte "
                      "and is not shown."]
        if self.original is None:
            out.append("The file does not exist yet; it will be created.")
        if self.client == "claude":
            out += ["Equivalent command:", f"  {claude_add_command(self.entry)}"]
        out.append("")
        if self.action == "replace" and self.old_block:
            out += [f"- {line}" for line in self.old_block.splitlines()]
        mark = "  " if self.action == "unchanged" else "+ "
        out += [f"{mark}{line}" for line in self.new_block.splitlines()]
        return "\n".join(out) + "\n"


def apply_change(change: ConfigChange, *, write: bool = False) -> bool:
    """Write the planned file only when ``write=True`` and something changes. Refuses if
    the file changed since the plan. Returns whether it wrote."""
    if not write or change.action == "unchanged":
        return False
    current = change.path.read_bytes() if change.path.exists() else None
    if current != change.original:
        raise RegisterError(f"{change.path} changed since the change was planned; plan it again")
    change.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = change.path.with_name(change.path.name + ".seedgraph-tmp")
    tmp.write_bytes(change.new_bytes)
    tmp.replace(change.path)
    return True


def _read(path: Path) -> tuple[bytes | None, str, bool]:
    """``(raw bytes or None, text without BOM, had_bom)``."""
    if not path.exists():
        return None, "", False
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegisterError(f"{path} is not UTF-8 text (byte {exc.start})") from None
    bom = text.startswith("﻿")
    return raw, text[1:] if bom else text, bom


def _newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


# --- Claude Code JSON ----------------------------------------------------------------------------

_WS = " \t\r\n"
_DECODER = json.JSONDecoder()


@dataclass(frozen=True)
class _Member:
    key: str
    key_start: int
    value_start: int
    value_end: int


def _skip(text: str, i: int) -> int:
    while i < len(text) and text[i] in _WS:
        i += 1
    return i


def _line_col(text: str, pos: int) -> str:
    line = text.count("\n", 0, pos) + 1
    return f"line {line}, column {pos - (text.rfind(chr(10), 0, pos) + 1) + 1}"


def _scan_object(text: str, start: int, path: Path) -> tuple[list[_Member], int]:
    """Members of the JSON object whose ``{`` is at ``start``, and the index of its ``}``."""
    if start >= len(text) or text[start] != "{":
        raise RegisterError(f"{path}: expected a JSON object at {_line_col(text, start)}")
    members: list[_Member] = []
    i = _skip(text, start + 1)
    if i < len(text) and text[i] == "}":
        return members, i
    while True:
        if i >= len(text) or text[i] != '"':
            raise RegisterError(f"{path}: expected a key at {_line_col(text, i)}")
        key_start = i
        key, i = json.decoder.scanstring(text, i + 1)
        i = _skip(text, i)
        if i >= len(text) or text[i] != ":":
            raise RegisterError(f"{path}: expected ':' at {_line_col(text, i)}")
        i = _skip(text, i + 1)
        try:
            _, end = _DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            raise RegisterError(f"{path}: invalid JSON value at {_line_col(text, i)}") from None
        members.append(_Member(key, key_start, i, end))
        i = _skip(text, end)
        if i < len(text) and text[i] == ",":
            i = _skip(text, i + 1)
            continue
        if i < len(text) and text[i] == "}":
            return members, i
        raise RegisterError(f"{path}: expected ',' or '}}' at {_line_col(text, i)}")


def _indent_of(text: str, pos: int) -> str | None:
    """The whitespace before ``pos`` on its line, or ``None`` when ``pos`` is not the first
    thing on its line (an inline object)."""
    line_start = text.rfind("\n", 0, pos) + 1
    prefix = text[line_start:pos]
    return prefix if prefix.strip() == "" else None


def _render_member(key: str, value: Any, indent: str, unit: str, nl: str) -> str:
    """``"key": value`` with nested lines indented by ``indent`` + ``unit``; compact when
    ``unit`` is empty (an inline file)."""
    if not unit:
        return f"{json.dumps(key)}: {json.dumps(value, ensure_ascii=False)}"
    body = json.dumps(value, indent=unit, ensure_ascii=False).replace("\n", nl + indent)
    return f"{json.dumps(key)}: {body}"


def _insert_member(text: str, members: list[_Member], close: int, open_brace: int, member_text: str,
                   nl: str, unit: str, *, parent: str = "") -> str:
    """Insert ``member_text`` as the object's last member, right after the current last
    value (so every existing byte stays where it was). ``parent`` is the indent of the line
    holding the object's ``{`` (used for an empty object)."""
    if members:
        last = members[-1]
        indent = _indent_of(text, last.key_start)
        sep = "," + (nl + indent if indent is not None else " ")
        return text[:last.value_end] + sep + member_text + text[last.value_end:]
    chunk = nl + parent + unit + member_text + nl + parent if unit else member_text
    return text[:open_brace + 1] + chunk + text[open_brace + 1:]


def _strip_ours(data: Any, name: str) -> Any:
    if isinstance(data, dict):
        data = dict(data)
        servers = data.get("mcpServers")
        if isinstance(servers, dict) and name in servers:
            servers = {k: v for k, v in servers.items() if k != name}
            data["mcpServers"] = servers
    return data


def _redact_claude(value: Any) -> Any:
    if not isinstance(value, dict):
        return REDACTED
    out: dict[str, Any] = {}
    for key, item in value.items():
        if key in ("type", "command", "args"):
            out[key] = item
        elif key == "env" and isinstance(item, dict):
            out[key] = {k: (v if k in OUR_ENV_KEYS else REDACTED) for k, v in item.items()}
        else:
            out[key] = REDACTED
    return out


def plan_claude_config(path: Path | str, entry: ServerEntry) -> ConfigChange:
    """Plan adding (or replacing) ``mcpServers.seedgraph_v3`` in a ``.claude.json``."""
    path = Path(path)
    raw, text, bom = _read(path)
    nl = _newline(text)
    value = claude_value(entry)
    shown = _render_member(entry.name, value, "", "  ", "\n")
    if not text.strip():
        new_text = "{" + nl + "  " + _render_member("mcpServers", {entry.name: value}, "  ", "  ", nl) + nl + "}" + nl
        return _claude_change(path, entry, "add", shown, None, raw, new_text, bom, text)
    try:
        old_data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RegisterError(f"{path} is not valid JSON (line {exc.lineno}, column {exc.colno})") from None
    if not isinstance(old_data, dict):
        raise RegisterError(f"{path}: the top level is not a JSON object")
    top_open = _skip(text, 0)
    top, top_close = _scan_object(text, top_open, path)
    first_indent = _indent_of(text, top[0].key_start) if top else "  "
    unit = first_indent if first_indent is not None else ""
    servers = [m for m in top if m.key == "mcpServers"]
    if len(servers) > 1:
        raise RegisterError(f'{path}: "mcpServers" appears more than once')
    if not servers:
        member = _render_member("mcpServers", {entry.name: value}, unit, unit, nl)
        new_text = _insert_member(text, top, top_close, top_open, member, nl, unit)
        return _claude_change(path, entry, "add", shown, None, raw, new_text, bom, text, old_data)
    srv = servers[0]
    if not isinstance(old_data.get("mcpServers"), dict):
        raise RegisterError(f'{path}: "mcpServers" is not an object')
    inner, inner_close = _scan_object(text, srv.value_start, path)
    ours = [m for m in inner if m.key == entry.name]
    if len(ours) > 1:
        raise RegisterError(f'{path}: "{entry.name}" appears more than once')
    if ours:
        m = ours[0]
        old_value = old_data["mcpServers"][entry.name]
        old_shown = _render_member(entry.name, _redact_claude(old_value), "", "  ", "\n")
        if old_value == value:
            return _claude_change(path, entry, "unchanged", shown, old_shown, raw, text, bom, text, old_data)
        indent = _indent_of(text, m.key_start)
        body = (json.dumps(value, indent=unit, ensure_ascii=False).replace("\n", nl + indent)
                if unit and indent is not None else json.dumps(value, ensure_ascii=False))
        new_text = text[:m.value_start] + body + text[m.value_end:]
        return _claude_change(path, entry, "replace", shown, old_shown, raw, new_text, bom, text, old_data)
    srv_indent = _indent_of(text, srv.key_start)
    if inner:
        indent = _indent_of(text, inner[0].key_start)
    else:
        indent = None if srv_indent is None or not unit else srv_indent + unit
    member = _render_member(entry.name, value, indent or "", unit if indent is not None else "", nl)
    new_text = _insert_member(text, inner, inner_close, srv.value_start, member, nl,
                              unit if indent is not None else "", parent=srv_indent or "")
    return _claude_change(path, entry, "add", shown, None, raw, new_text, bom, text, old_data)


def _claude_change(path: Path, entry: ServerEntry, action: Action, shown: str, old_shown: str | None,
                   raw: bytes | None, new_text: str, bom: bool, old_text: str,
                   old_data: Any = None) -> ConfigChange:
    try:
        new_data = json.loads(new_text)
    except json.JSONDecodeError as exc:  # pragma: no cover - a splice bug, never user text
        raise RegisterError(f"internal error: the planned {path.name} would not parse "
                            f"(line {exc.lineno}, column {exc.colno})") from None
    if new_data.get("mcpServers", {}).get(entry.name) != claude_value(entry):
        raise RegisterError(f"internal error: the planned {path.name} does not hold the new entry")
    before = _strip_ours(old_data if old_data is not None else {}, entry.name)
    after = _strip_ours(new_data, entry.name)
    if isinstance(after, dict) and after.get("mcpServers") == {} and "mcpServers" not in (before or {}):
        after = {k: v for k, v in after.items() if k != "mcpServers"}
    if after != (before or {}):
        raise RegisterError(f"internal error: the planned {path.name} changes more than {entry.name}")
    data = (("﻿" if bom else "") + new_text).encode("utf-8")
    return ConfigChange("claude", path, action, entry, shown, old_shown, raw, data)


# --- Codex TOML ----------------------------------------------------------------------------------

_HEADER = re.compile(r"^[ \t]*\[(?!\[)(?P<name>[^\]]*)\][ \t]*(?:#.*)?$")


def _table_path(name: str) -> list[str]:
    parts = []
    for raw in re.findall(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'|[^.]+', name):
        raw = raw.strip()
        if raw.startswith('"'):
            parts.append(json.loads(raw))
        elif raw.startswith("'"):
            parts.append(raw[1:-1])
        elif raw:
            parts.append(raw)
    return parts


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return f'"{REDACTED}"'


def _redact_toml_entry(existing: Mapping[str, Any], name: str) -> str:
    """The existing ``seedgraph_v3`` entry, rendered from its parsed value (so comments,
    including commented-out keys, are never shown) with every value seedgraph does not
    write replaced by ``<redacted>``."""
    allowed = ("command", "args", "startup_timeout_sec", "tool_timeout_sec", "enabled", "cwd")
    lines = [f"[mcp_servers.{name}]"]
    subtables: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in existing.items():
        if isinstance(value, dict):
            subtables.append((key, value))
        elif key in allowed:
            lines.append(f"{key} = {_toml_value(value)}")
        else:
            lines.append(f'{key} = "{REDACTED}"')
    for sub, table in subtables:
        lines += ["", f"[mcp_servers.{name}.{sub}]"]
        for key, value in table.items():
            keep = sub == "env" and key in OUR_ENV_KEYS and isinstance(value, str)
            lines.append(f"{key} = {_toml_value(value)}" if keep else f'{key} = "{REDACTED}"')
    return "\n".join(lines) + "\n"


def plan_codex_config(path: Path | str, entry: ServerEntry) -> ConfigChange:
    """Plan adding (or replacing) ``[mcp_servers.seedgraph_v3]`` in a Codex ``config.toml``."""
    path = Path(path)
    raw, text, bom = _read(path)
    try:
        old_data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        where = str(exc).rsplit("(", 1)[-1].rstrip(")") if "(" in str(exc) else "unknown position"
        raise RegisterError(f"{path} is not valid TOML ({where})") from None
    nl = _newline(text)
    block = codex_block(entry, nl)
    shown = codex_block(entry)
    lines = text.splitlines(keepends=True)
    headers = [(i, _table_path(m.group("name"))) for i, line in enumerate(lines)
               if (m := _HEADER.match(line.rstrip("\r\n")))]
    ours = [i for i, p in headers if p[:2] == ["mcp_servers", entry.name]]
    existing = old_data.get("mcp_servers", {}).get(entry.name) if isinstance(old_data.get("mcp_servers"), dict) \
        else None
    if existing is not None and not ours:
        raise RegisterError(f"{path}: {entry.name} is written as an inline table or dotted keys; "
                            "edit that entry by hand or remove it first")
    if ours:
        order = [i for i, _ in headers]
        first = ours[0]
        k = order.index(first)
        stop_idx = len(lines)
        for i, p in headers[k:]:
            if p[:2] != ["mcp_servers", entry.name]:
                stop_idx = i
                break
        if any(i >= stop_idx for i in ours):
            raise RegisterError(f"{path}: the {entry.name} tables are split across the file; "
                                "edit them by hand or put them together first")
        end = stop_idx
        # end at the block's last header or key line: blank and comment lines after it
        # belong to whatever follows and are kept untouched
        while end > first and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
            end -= 1
        old_shown = _redact_toml_entry(existing, entry.name) if isinstance(existing, dict) else None
        if existing == tomllib.loads(shown)["mcp_servers"][entry.name]:
            return _codex_change(path, entry, "unchanged", shown, old_shown, raw, text, bom, old_data)
        new_text = "".join(lines[:first]) + block + "".join(lines[end:])
        return _codex_change(path, entry, "replace", shown, old_shown, raw, new_text, bom, old_data)
    if not text:
        sep = ""
    elif text.endswith("\n"):
        sep = nl
    else:
        sep = nl + nl
    return _codex_change(path, entry, "add", shown, None, raw, text + sep + block, bom, old_data)


def _codex_change(path: Path, entry: ServerEntry, action: Action, shown: str, old_shown: str | None,
                  raw: bytes | None, new_text: str, bom: bool, old_data: dict) -> ConfigChange:
    try:
        new_data = tomllib.loads(new_text)
    except tomllib.TOMLDecodeError:  # pragma: no cover - a splice bug, never user text
        raise RegisterError(f"internal error: the planned {path.name} would not parse") from None
    want = tomllib.loads(shown)["mcp_servers"][entry.name]
    if new_data.get("mcp_servers", {}).get(entry.name) != want:
        raise RegisterError(f"internal error: the planned {path.name} does not hold the new entry")

    def strip(data: dict) -> dict:
        data = dict(data)
        servers = data.get("mcp_servers")
        if isinstance(servers, dict):
            servers = {k: v for k, v in servers.items() if k != entry.name}
            if servers or "mcp_servers" in old_data:
                data["mcp_servers"] = servers
            else:
                del data["mcp_servers"]
        return data

    if strip(new_data) != strip(old_data):
        raise RegisterError(f"internal error: the planned {path.name} changes more than {entry.name}")
    data = (("﻿" if bom else "") + new_text).encode("utf-8")
    return ConfigChange("codex", path, action, entry, shown, old_shown, raw, data)


# --- skills --------------------------------------------------------------------------------------

def skill_text(name: str) -> str:
    """The bundled text of skill ``seedgraph-v3`` or ``seedgraph-v3-work``."""
    if name not in SKILL_SOURCES:
        raise RegisterError(f"unknown skill {name!r}; known: {sorted(SKILL_SOURCES)}")
    return files(__package__).joinpath(SKILL_SOURCES[name]).read_text(encoding="utf-8")


@dataclass
class SkillFile:
    name: str
    path: Path
    action: Action
    text: str = field(repr=False)

    def render(self) -> str:
        verb = {"add": "add", "replace": "replace", "unchanged": "unchanged"}[self.action]
        return f"skill {self.name}: {verb} {self.path}"


def plan_skills(target_dir: Path | str, *, layout: Literal["skills", "prompts"] = "skills",
                names: Sequence[str] = tuple(SKILL_SOURCES)) -> list[SkillFile]:
    """Plan installing the v3 skills under ``target_dir``: ``<name>/SKILL.md`` for a skills
    directory (Claude Code ``~/.claude/skills``, Codex ``~/.codex/skills``), or ``<name>.md``
    for a prompts directory (Codex ``~/.codex/prompts``)."""
    target = Path(target_dir)
    out = []
    for name in names:
        text = skill_text(name)
        path = target / name / "SKILL.md" if layout == "skills" else target / f"{name}.md"
        if layout not in ("skills", "prompts"):
            raise RegisterError(f"layout must be 'skills' or 'prompts', not {layout!r}")
        if not path.exists():
            action: Action = "add"
        else:
            action = "unchanged" if path.read_bytes() == text.encode("utf-8") else "replace"
        out.append(SkillFile(name, path, action, text))
    return out


def apply_skills(planned: Sequence[SkillFile], *, write: bool = False) -> list[Path]:
    """Write the planned skill files only when ``write=True``; returns the paths written."""
    if not write:
        return []
    written = []
    for skill in planned:
        if skill.action == "unchanged":
            continue
        skill.path.parent.mkdir(parents=True, exist_ok=True)
        skill.path.write_bytes(skill.text.encode("utf-8"))
        written.append(skill.path)
    return written


# --- setup bundle --------------------------------------------------------------------------------

@dataclass(frozen=True)
class Targets:
    claude_config: Path
    claude_skills_dir: Path
    codex_config: Path
    codex_skills_dir: Path
    codex_prompts_dir: Path


def default_targets(user_home: Path | str) -> Targets:
    """The usual locations under ``user_home``. Names paths only; reads and writes nothing."""
    h = Path(user_home)
    return Targets(h / ".claude.json", h / ".claude" / "skills", h / ".codex" / "config.toml",
                   h / ".codex" / "skills", h / ".codex" / "prompts")


@dataclass
class SetupPlan:
    changes: list[ConfigChange]
    skills: list[SkillFile]

    def render(self) -> str:
        parts = [c.render() for c in self.changes]
        if self.skills:
            parts.append("\n".join(s.render() for s in self.skills) + "\n")
        if any(c.client == "claude" for c in self.changes):
            parts.append(RECORD_ANSWER_HINT + "\n")
        parts.append("Nothing has been written. Re-run with --write to apply exactly this.\n")
        return "\n".join(parts)

    def apply(self, *, write: bool = False) -> list[Path]:
        if not write:
            return []
        written = [c.path for c in self.changes if apply_change(c, write=True)]
        return written + apply_skills(self.skills, write=True)


def plan_setup(client: Literal["claude", "codex", "both"], entry: ServerEntry, *,
               claude_config: Path | str | None = None, claude_skills_dir: Path | str | None = None,
               codex_config: Path | str | None = None, codex_skills_dir: Path | str | None = None,
               codex_layout: Literal["skills", "prompts"] = "skills") -> SetupPlan:
    """Plan the registration and skills for one or both clients, from the paths given."""
    if client not in ("claude", "codex", "both"):
        raise RegisterError(f"client must be claude, codex or both, not {client!r}")
    changes: list[ConfigChange] = []
    skills: list[SkillFile] = []
    wanted: Mapping[str, tuple[Any, Any, str]] = {
        "claude": (claude_config, claude_skills_dir, "skills"),
        "codex": (codex_config, codex_skills_dir, codex_layout),
    }
    for name in ("claude", "codex"):
        if client not in (name, "both"):
            continue
        config, skills_dir, layout = wanted[name]
        if config is None or skills_dir is None:
            raise RegisterError(f"{name}: pass both the config path and the skills directory")
        changes.append((plan_claude_config if name == "claude" else plan_codex_config)(config, entry))
        skills += plan_skills(skills_dir, layout=layout)  # type: ignore[arg-type]
    return SetupPlan(changes, skills)
