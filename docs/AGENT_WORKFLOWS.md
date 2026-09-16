# Active-session agent workflows

Seedgraph exposes **find -> inspect -> contribute** through the CLI and MCP. These
operations use the existing corpus and cache; source acquisition and conversion
remain deliberate separate steps.

## Reliable answers

```powershell
seedgraph ask "Which assumptions identify the effect?" --project demo --no-llm --no-save --json
```

CLI JSON stdout is one JSON value. Both CLI and MCP attempt answer/trace saving by
default; `--no-save` or MCP `no_save=true` skips those artifacts. The envelope's
`persistence.status` is `saved`, `skipped`, or `not_saved`. Only a complete durable
answer/trace pair counts as saved. A save failure still returns usable evidence
and success; source/retrieval failures remain errors. Diagnostic output uses stderr.

Combined no-LLM/no-save requests open an existing, compatible corpus/cache without
initialization or migrations. The strict read-only opener requires a checkpointed,
quiescent database: close writers and checkpoint separately if it reports setup
required. Missing initialization or an older schema needs separate setup/upgrade.
No-save by itself does not disable paid calls or their usage logging.

## Bounded source inspection

```powershell
seedgraph work-read demo WORK_ID --json
seedgraph work-read demo WORK_ID --section SECTION_ID --subtree --json
seedgraph work-read demo WORK_ID --span SPAN_ID --json
seedgraph work-read demo WORK_ID --cursor CURSOR --max-chars 12000 --json
```

The MCP tool is `work_read`. The default text budget is 12,000 Unicode codepoints;
the maximum is 48,000. Offsets are half-open `[start_char,end_char)` into the pinned
Markdown. `selection_complete` describes the selected work/section, and a non-null
`continuation` is passed back as `cursor` with the same selection. A section reads
only its own text unless `subtree=true`; a span selects its full anchored source
version and returns its anchor offsets/section for locating the relevant passage.

Section inventories are separately paged with `section_offset`/`section_limit`
(default 50, maximum 100); headings and breadcrumbs are length-bounded. No section
rows need to be created to read a document. Missing full text returns metadata,
a reason and a next action, without downloading or converting anything.

Reconversion does not mix versions across pages: continuation retains the original
Markdown and flags newer text. If the old blob disappears, restart guidance is
returned. Replacing the entire source file can continue only when existing
section/span/note provenance still links the old version to this work; otherwise
the reader reports `pinned_lineage_unavailable` and asks for a fresh read.

Local CLI reading may display private content. Use `--external` when handing it to
a remote model; MCP enforces the external content gate by default. Explicit sharing
consent uses `--confirm-external` / `confirm_external=true`. `--redact-private`
withholds private body text **and headings**, even with external sharing consent.
The gate rechecks current policy and access on each read.

## Install the explicit extraction skill

The existing ambient `seedgraph` skill consults the corpus. The separately selected
`seedgraph-extract` skill prepares and imports notes only when requested.

```powershell
# Codex: default extraction skill destination is ~/.agents/skills
seedgraph project agent-setup demo --extraction --client codex
# Claude: default extraction skill destination is ~/.claude/skills
seedgraph project agent-setup demo --extraction --client claude
```

Use `--skills-dir PATH` for a chosen installation directory. Extraction-only setup
does not modify CLAUDE.md. Codex gets explicit-invocation metadata; Claude gets an
explicit slash-command skill. No client authentication, subscription purchase or
model call occurs during setup.

## Single-paper and resumed-batch smoke paths

In an active signed-in **Codex** session, invoke `$seedgraph-extract` and say:
"Extract work WORK_ID in project demo. Report the batch ID and final note status."
In an active signed-in **Claude** session, invoke `/seedgraph-extract` with the
same instruction. Both clients follow the same packet protocol:

```powershell
seedgraph agent-extract prepare demo --work WORK_ID --client codex
seedgraph agent-extract next demo BATCH_ID
# Complete the returned prompt/schema in the active client and save UTF-8 JSON:
# {"binding": <unchanged packet binding>, "note": <completed note object>}
seedgraph agent-extract submit demo BATCH_ID --packet PACKET_ID --result result.json
seedgraph agent-extract status demo BATCH_ID
```

Use `--client claude` for Claude provenance. Model identity is optional and
host-reported via submit `--model`; unknown usage/cost stays unknown. Preparation's
`--input-budget-tokens` defaults to 12,000 and includes prompt/schema overhead.
Private source text requires existing policy permission or explicit sharing consent
at prepare with `--confirm-external`. A local CLI does not make inference local.

For the batch smoke path, prepare an ordered selection with repeated `--work`.
Submit at least one packet, stop, then ask either client to resume that recorded
batch ID. `next` returns the same unfinished packet until submitted; it never
silently discards it. A reported pause resumes with `next --resume`; failed-paper
retry is explicit with `next --retry-failed`. Status is bounded by `--offset/--limit`.

MCP equivalents are `extraction_prepare`, `extraction_next`, `extraction_submit`,
and `extraction_status`. Prepare, next and submit are explicit mutations (next can
reconcile/checkpoint a committed result). A redacting MCP server withholds private
packets; it never substitutes redacted text into a signed-off source packet.

Notes activate automatically after schema, version, complete coverage and exact
quote checks. These checks establish structure/evidence links, not scientific
correctness. Existing current notes are preserved unless preparation uses explicit
`--replace`. Replaced notes remain inspectable; ordinary search/ask use the current
interpretation. Identical submissions are idempotent. Source changes retain the
submission for inspection but require fresh extraction. Incomplete long-paper
results stay out of ordinary search until every planned chunk completes.

Paper failures (`extraction_failed`, `invalid_response`, `source_unavailable`) can
be submitted with `--failure` and the batch continues. Actual host-reported session
failures (`authentication`, `quota`, `batch_access`) pause it. Abrupt interruption
resumes from durable checkpoints. There is no unattended client launcher, silent
API fallback or claim of unlimited subscription use.

These are runnable client smoke instructions. Automated tests exercise fixture
round trips; they do not prove a live signed-in client session was run.
