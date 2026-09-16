"""CLI/MCP contracts use the same real preparation/import services."""
import json

from typer.testing import CliRunner

from seedgraph.cli import app
from seedgraph.mcp.context import ServerContext
from seedgraph.mcp.tools_extraction import register_extraction_tools
from seedgraph.vocab import AccessClass
from test_phase_4 import _make_doc, good_note_dict


class Registry:
    def __init__(self):
        self.tools = {}
        self.annotations = {}

    def tool(self, **kwargs):
        def register(fn):
            self.tools[fn.__name__] = fn
            self.annotations[fn.__name__] = kwargs["annotations"]
            return fn
        return register


def test_cli_roundtrip_and_mcp_duplicate(tmp_path):
    handle, work_id, _ = _make_doc("surfaces")
    cli = CliRunner()
    def command(*args):
        result = cli.invoke(app, ["agent-extract", *args])
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)
    batch = command("prepare", handle.slug, "--work", work_id, "--client", "codex")["batch_id"]
    packet = command("next", handle.slug, batch)
    payload = {"binding": packet["binding"], "note": good_note_dict()}
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps(payload), encoding="utf-8")
    accepted = command("submit", handle.slug, batch, "--packet", packet["packet_id"],
                       "--result", str(result_file))
    assert accepted["status"] == "accepted"
    registry = Registry()
    register_extraction_tools(registry, ServerContext(root=handle.root))
    duplicate = registry.tools["extraction_submit"](handle.slug, batch, packet["packet_id"], result=payload)
    assert duplicate["status"] == "duplicate"
    assert command("status", handle.slug, batch)["counts"] == {"accepted": 1}
    assert registry.annotations["extraction_next"]["readOnlyHint"] is False
    assert registry.annotations["extraction_status"]["readOnlyHint"] is True


def test_mcp_redaction_withholds_private_packet_even_with_consent():
    handle, work_id, _ = _make_doc("redact_packets", access_class=AccessClass.user_supplied_private)
    registry = Registry()
    register_extraction_tools(registry, ServerContext(root=handle.root, redact_private=True))
    batch = registry.tools["extraction_prepare"](handle.slug, [work_id], confirm_external=True)["batch_id"]
    response = registry.tools["extraction_next"](handle.slug, batch)
    assert response["status"] == "policy_withheld"
    assert "user_prompt" not in response and "system_prompt" not in response
