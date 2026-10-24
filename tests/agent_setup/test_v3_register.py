"""v3 agent registration (design §7.4): render the exact `seedgraph_v3` change for Claude
Code (user-scope `mcpServers` in a `.claude.json`) and Codex (`[mcp_servers.seedgraph_v3]`
in a `config.toml`), show only that block, write only with `write=True`, and keep every
other byte of the file. Every path here is a temp file; secret-looking values are invented.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from seedgraph.agent_setup import register
from seedgraph.agent_setup.register import RegisterError

WORKTREE = Path(r"C:\work\seedgraph v3")
HOME = Path(r"C:\Users\someone\.seedgraph-v3")
SECRET = "sk-invented-SECRET-0123456789"
OTHER_SECRET = "ghp_inventedTOKENabcdef"
EXE = str(WORKTREE / ".venv" / "Scripts" / "seedgraph-v3.exe")


@pytest.fixture
def entry():
    return register.server_entry(worktree=WORKTREE, home=HOME)


CLAUDE_JSON = (
    '{\n'
    '  "numStartups": 12,\n'
    '  "oauthAccount": {"emailAddress": "someone@example.invalid", "token": "' + SECRET + '"},\n'
    '  "mcpServers": {\n'
    '    "seedgraph": {\n'
    '      "type": "stdio",\n'
    '      "command": "seedgraph",\n'
    '      "args": ["mcp", "serve"]\n'
    '    },\n'
    '    "other": {"command": "npx", "env": {"API_KEY": "' + OTHER_SECRET + '"}}\n'
    '  },\n'
    '  "projects": {"C:/x": {"allowedTools": []}}\n'
    '}\n'
)

CODEX_TOML = (
    'model = "invented-model"\n'
    '\n'
    '[mcp_servers.seedgraph]\n'
    'command = "seedgraph"\n'
    'args = ["mcp", "serve"]\n'
    '\n'
    '[mcp_servers.other]\n'
    'command = "npx"\n'
    '\n'
    '[mcp_servers.other.env]\n'
    'API_KEY = "' + OTHER_SECRET + '"\n'
    '\n'
    '[profiles.secret]\n'
    'token = "' + SECRET + '"\n'
)


def _write(path: Path, text: str, newline: str = "\n") -> bytes:
    data = text.replace("\n", newline).encode("utf-8")
    path.write_bytes(data)
    return data


def _cut(new: bytes, old: bytes) -> bytes:
    """``new`` with the one inserted chunk removed, assuming a pure insertion."""
    i = 0
    while i < len(old) and new[i] == old[i]:
        i += 1
    j = 0
    while j < len(old) - i and new[-1 - j] == old[-1 - j]:
        j += 1
    return new[:i] + new[len(new) - j:] if j else new[:i]


# --- the entry -----------------------------------------------------------------------------------

def test_server_entry_defaults(entry):
    assert entry.name == "seedgraph_v3"
    assert entry.command == EXE
    assert entry.args == ("mcp", "serve")
    assert entry.env == {"SEEDGRAPH_HOME": str(HOME), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    assert entry.startup_timeout_sec == 60


def test_server_entry_python_launcher():
    e = register.server_entry(worktree=WORKTREE, home=HOME, launcher="python")
    assert e.command == str(WORKTREE / ".venv" / "Scripts" / "python.exe")
    assert e.args == ("-m", "seedgraph", "mcp", "serve")


def test_claude_add_command(entry):
    cmd = register.claude_add_command(entry)
    assert cmd.startswith("claude mcp add --scope user seedgraph_v3 ")
    assert f"--env SEEDGRAPH_HOME={HOME} " in cmd
    assert "--env PYTHONUTF8=1 --env PYTHONIOENCODING=utf-8 -- " in cmd
    assert cmd.endswith(f'"{EXE}" mcp serve')


# --- Claude Code JSON ----------------------------------------------------------------------------

def test_claude_plan_writes_nothing_without_write(tmp_path, entry):
    path = tmp_path / ".claude.json"
    old = _write(path, CLAUDE_JSON)
    change = register.plan_claude_config(path, entry)
    assert change.action == "add"
    assert register.apply_change(change) is False
    assert register.apply_change(change, write=False) is False
    assert path.read_bytes() == old


def test_claude_add_preserves_every_other_byte(tmp_path, entry):
    path = tmp_path / ".claude.json"
    old = _write(path, CLAUDE_JSON)
    change = register.plan_claude_config(path, entry)
    assert register.apply_change(change, write=True) is True
    new = path.read_bytes()
    assert _cut(new, old) == old
    data = json.loads(new)
    assert data["mcpServers"]["seedgraph_v3"] == {
        "type": "stdio", "command": EXE, "args": ["mcp", "serve"],
        "env": {"SEEDGRAPH_HOME": str(HOME), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}}
    before = json.loads(old)
    del data["mcpServers"]["seedgraph_v3"]
    assert data == before
    # idempotent: a second plan is unchanged and writes nothing
    again = register.plan_claude_config(path, entry)
    assert again.action == "unchanged"
    assert register.apply_change(again, write=True) is False
    assert path.read_bytes() == new


def test_claude_replace_existing_entry(tmp_path, entry):
    path = tmp_path / ".claude.json"
    stale = CLAUDE_JSON.replace(
        '    "other":',
        '    "seedgraph_v3": {"command": "old.exe", "env": {"SEEDGRAPH_HOME": "C:/old", "MY_TOKEN": "'
        + SECRET + '"}},\n    "other":')
    _write(path, stale)
    change = register.plan_claude_config(path, entry)
    assert change.action == "replace"
    shown = change.render()
    assert "old.exe" in shown and SECRET not in shown and "MY_TOKEN" in shown
    register.apply_change(change, write=True)
    text = path.read_text(encoding="utf-8")
    assert text.count('"seedgraph_v3"') == 1
    assert json.loads(text)["mcpServers"]["seedgraph_v3"]["command"] == EXE
    assert text.startswith(stale[:stale.index('"seedgraph_v3"')])
    assert text.endswith(stale[stale.index(',\n    "other":'):])


def test_claude_render_shows_only_the_block(tmp_path, entry):
    path = tmp_path / ".claude.json"
    _write(path, CLAUDE_JSON)
    shown = register.plan_claude_config(path, entry).render()
    for secret in (SECRET, OTHER_SECRET, "someone@example.invalid", "numStartups", "allowedTools", '"other"'):
        assert secret not in shown
    assert '+ "seedgraph_v3": {' in shown
    assert str(path) in shown and "claude mcp add --scope user seedgraph_v3" in shown
    assert "SEEDGRAPH_HOME" in shown


def test_claude_change_repr_hides_file_text(tmp_path, entry):
    path = tmp_path / ".claude.json"
    _write(path, CLAUDE_JSON)
    change = register.plan_claude_config(path, entry)
    assert SECRET not in repr(change) and OTHER_SECRET not in repr(change)


@pytest.mark.parametrize("text", ["", "{}", "{\n}\n", '{"numStartups": 1}', '\ufeff{"a": 1}\n'])
def test_claude_shapes_without_servers(tmp_path, entry, text):
    path = tmp_path / ".claude.json"
    path.write_bytes(text.encode("utf-8"))
    register.apply_change(register.plan_claude_config(path, entry), write=True)
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    assert data["mcpServers"]["seedgraph_v3"]["command"] == EXE
    if text.startswith("\ufeff"):
        assert raw.startswith("\ufeff".encode("utf-8")) and data["a"] == 1


def test_claude_missing_file(tmp_path, entry):
    path = tmp_path / "nested" / ".claude.json"
    change = register.plan_claude_config(path, entry)
    assert change.action == "add" and not path.exists()
    register.apply_change(change, write=True)
    assert json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["seedgraph_v3"]["args"] == ["mcp", "serve"]


def test_claude_crlf_kept(tmp_path, entry):
    path = tmp_path / ".claude.json"
    old = _write(path, CLAUDE_JSON, newline="\r\n")
    register.apply_change(register.plan_claude_config(path, entry), write=True)
    new = path.read_bytes()
    assert _cut(new, old) == old
    assert b"\n" not in new.replace(b"\r\n", b"")


def test_claude_invalid_json_refused_without_echo(tmp_path, entry):
    path = tmp_path / ".claude.json"
    path.write_text('{"token": "' + SECRET + '", oops}', encoding="utf-8")
    with pytest.raises(RegisterError) as info:
        register.plan_claude_config(path, entry)
    assert SECRET not in str(info.value) and "line 1" in str(info.value)


def test_claude_file_changed_after_plan_is_refused(tmp_path, entry):
    path = tmp_path / ".claude.json"
    _write(path, CLAUDE_JSON)
    change = register.plan_claude_config(path, entry)
    path.write_text(CLAUDE_JSON.replace("12", "13"), encoding="utf-8")
    with pytest.raises(RegisterError):
        register.apply_change(change, write=True)


# --- Codex TOML ----------------------------------------------------------------------------------

def test_codex_block_text(entry):
    block = register.codex_block(entry)
    assert block.splitlines()[:4] == [
        "[mcp_servers.seedgraph_v3]",
        f"command = '{EXE}'",
        'args = ["mcp", "serve"]',
        "startup_timeout_sec = 60",
    ]
    data = tomllib.loads(block)["mcp_servers"]["seedgraph_v3"]
    assert data["env"] == {"SEEDGRAPH_HOME": str(HOME), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def test_codex_plan_writes_nothing_without_write(tmp_path, entry):
    path = tmp_path / "config.toml"
    old = _write(path, CODEX_TOML)
    change = register.plan_codex_config(path, entry)
    assert change.action == "add"
    assert register.apply_change(change) is False
    assert path.read_bytes() == old


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_codex_add_preserves_every_other_byte(tmp_path, entry, newline):
    path = tmp_path / "config.toml"
    old = _write(path, CODEX_TOML, newline=newline)
    register.apply_change(register.plan_codex_config(path, entry), write=True)
    new = path.read_bytes()
    assert new.startswith(old) and _cut(new, old) == old
    data = tomllib.loads(new.decode("utf-8"))
    assert data["mcp_servers"]["seedgraph_v3"]["startup_timeout_sec"] == 60
    del data["mcp_servers"]["seedgraph_v3"]
    assert data == tomllib.loads(old.decode("utf-8"))
    assert register.plan_codex_config(path, entry).action == "unchanged"


def test_codex_no_trailing_newline_and_empty(tmp_path, entry):
    path = tmp_path / "config.toml"
    path.write_text('model = "x"', encoding="utf-8")
    register.apply_change(register.plan_codex_config(path, entry), write=True)
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert data["model"] == "x" and data["mcp_servers"]["seedgraph_v3"]["command"] == EXE
    empty = tmp_path / "empty.toml"
    register.apply_change(register.plan_codex_config(empty, entry), write=True)
    assert empty.read_text(encoding="utf-8") == register.codex_block(entry)


def test_codex_replace_existing_block(tmp_path, entry):
    path = tmp_path / "config.toml"
    stale = CODEX_TOML.replace(
        "[mcp_servers.other]\n",
        "[mcp_servers.seedgraph_v3]\ncommand = 'old.exe'\n\n[mcp_servers.seedgraph_v3.env]\n"
        "SEEDGRAPH_HOME = 'C:/old'\nMY_TOKEN = \"" + SECRET + "\"\n\n[mcp_servers.other]\n")
    _write(path, stale)
    change = register.plan_codex_config(path, entry)
    assert change.action == "replace"
    shown = change.render()
    assert "old.exe" in shown and "MY_TOKEN" in shown and SECRET not in shown and OTHER_SECRET not in shown
    register.apply_change(change, write=True)
    text = path.read_text(encoding="utf-8")
    assert text.count("[mcp_servers.seedgraph_v3]") == 1 and "old.exe" not in text and "MY_TOKEN" not in text
    head = stale[:stale.index("[mcp_servers.seedgraph_v3]")]
    tail = stale[stale.index("\n[mcp_servers.other]"):]
    assert text.startswith(head) and text.endswith(tail)
    data = tomllib.loads(text)
    assert data["mcp_servers"]["other"]["env"]["API_KEY"] == OTHER_SECRET


def test_codex_render_shows_only_the_block(tmp_path, entry):
    path = tmp_path / "config.toml"
    _write(path, CODEX_TOML)
    shown = register.plan_codex_config(path, entry).render()
    for hidden in (SECRET, OTHER_SECRET, "invented-model", "[mcp_servers.other]", "profiles"):
        assert hidden not in shown
    assert "+ [mcp_servers.seedgraph_v3]" in shown and "+ startup_timeout_sec = 60" in shown


@pytest.mark.parametrize("text", [
    "[mcp_servers]\nseedgraph_v3 = { command = 'x' }\n",
    "mcp_servers.seedgraph_v3.command = 'x'\n",
    "[mcp_servers.seedgraph_v3]\ncommand = 'x'\n[other]\na = 1\n[mcp_servers.seedgraph_v3.env]\nA = '1'\n",
])
def test_codex_forms_we_cannot_edit_are_refused(tmp_path, entry, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RegisterError):
        register.plan_codex_config(path, entry)


def test_codex_invalid_toml_refused_without_echo(tmp_path, entry):
    path = tmp_path / "config.toml"
    path.write_text('token = "' + SECRET + '"\nbroken = = 1\n', encoding="utf-8")
    with pytest.raises(RegisterError) as info:
        register.plan_codex_config(path, entry)
    assert SECRET not in str(info.value)


def test_codex_quote_in_path_uses_basic_string(tmp_path):
    e = register.server_entry(worktree=Path(r"C:\it's here"), home=HOME)
    data = tomllib.loads(register.codex_block(e))
    assert data["mcp_servers"]["seedgraph_v3"]["command"] == e.command


def test_codex_replace_keeps_comments_and_never_shows_them(tmp_path, entry):
    """A comment above the next table belongs to that table: kept byte for byte, never shown.
    A commented-out key inside the stale block is never shown either."""
    path = tmp_path / "config.toml"
    stale = CODEX_TOML.replace(
        "[mcp_servers.other]\n",
        "[mcp_servers.seedgraph_v3]\ncommand = 'old.exe'\n# API_KEY = \"" + SECRET + "\"\n\n"
        "# token for the other server: " + OTHER_SECRET + "\n[mcp_servers.other]\n")
    _write(path, stale)
    change = register.plan_codex_config(path, entry)
    shown = change.render()
    assert SECRET not in shown and OTHER_SECRET not in shown and "old.exe" in shown
    register.apply_change(change, write=True)
    text = path.read_text(encoding="utf-8")
    tail = stale[stale.index("\n# token for the other server"):]
    assert text.endswith(tail)
    assert text.startswith(stale[:stale.index("[mcp_servers.seedgraph_v3]")])
