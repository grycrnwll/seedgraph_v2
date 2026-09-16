"""Ambient-trigger install for a research working directory.

Two artifacts turn a plain project directory into one where an agent reflexively
consults the project's Seedgraph corpus:

* **Carrier A** — a per-project ``CLAUDE.md`` norm block (the reflex prose),
  fenced by idempotent markers so re-running only ever updates the fenced region.
* **Carrier B** — the global ``seedgraph`` skill deployed to ``~/.claude/skills``.

The repo is the source of truth: both artifacts are bundled *canonical* templates
(``claude_md_block.md`` and ``seedgraph_skill.md``), and this module deploys them.
Pure functions only — no Typer — so the logic is unit-testable.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

_BEGIN = "<!-- seedgraph:begin -->"
_END = "<!-- seedgraph:end -->"


def _template(name: str) -> str:
    """Read a bundled template from this package via importlib.resources."""
    return files(__package__).joinpath(name).read_text(encoding="utf-8")


def render_claude_block(slug: str) -> str:
    """Render the CLAUDE.md norm block for ``slug`` (the proven §7 prose)."""
    return _template("claude_md_block.md").replace("{slug}", slug)


def _fenced(block: str) -> str:
    """Wrap a rendered block in the idempotent marker fences."""
    return f"{_BEGIN}\n{block.rstrip()}\n{_END}\n"


def write_claude_block(target_dir: Path, slug: str) -> Path:
    """Write/update the fenced norm block into ``<target_dir>/CLAUDE.md``.

    * No file → create it with just the fenced block.
    * File with fences → replace only the fenced region, preserving the rest.
    * File without fences → append the fenced block (one blank line before).

    Idempotent: re-running never duplicates the block. Returns the CLAUDE.md path.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "CLAUDE.md"
    fenced = _fenced(render_claude_block(slug))

    if not path.exists():
        path.write_text(fenced, encoding="utf-8")
        return path

    existing = path.read_text(encoding="utf-8")
    start = existing.find(_BEGIN)
    end = existing.find(_END)
    if start != -1 and end != -1 and end > start:
        end += len(_END)
        updated = existing[:start] + fenced.rstrip("\n") + existing[end:]
    else:
        prefix = existing if existing.endswith("\n") else existing + "\n"
        updated = f"{prefix}\n{fenced}"
    path.write_text(updated, encoding="utf-8")
    return path


def install_skill(skills_dir: Path) -> tuple[Path, bool]:
    """Deploy the bundled skill to ``<skills_dir>/seedgraph/SKILL.md``.

    Idempotent: only writes when content differs. Returns ``(path, changed)``.
    """
    skills_dir = Path(skills_dir)
    path = skills_dir / "seedgraph" / "SKILL.md"
    content = _template("seedgraph_skill.md")
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path, True


def install_extraction_skill(skills_dir: Path, *, client: str = "claude") -> tuple[Path, bool]:
    """Install the separately selected extraction skill; never change ambient rules."""
    if client not in {"claude", "codex"}:
        raise ValueError("client must be claude or codex")
    path = Path(skills_dir) / "seedgraph-extract" / "SKILL.md"
    content = _template("seedgraph_extract_skill.md")
    if client == "claude":
        content = content.replace("name: seedgraph-extract\n",
                                  "name: seedgraph-extract\ndisable-model-invocation: true\n", 1)
    changed = False
    artifacts = {path: content}
    if client == "codex":
        artifacts[path.parent / "agents" / "openai.yaml"] = (
            "policy:\n  allow_implicit_invocation: false\n"
        )
    for destination, text in artifacts.items():
        if not destination.exists() or destination.read_text(encoding="utf-8") != text:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
            changed = True
    return path, changed
