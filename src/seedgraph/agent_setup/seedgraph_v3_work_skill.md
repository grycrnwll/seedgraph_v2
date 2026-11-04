---
name: seedgraph-v3-work
description: Work one Seedgraph batch until it is finished or paused - claim a package, read its files, write the result, submit it and follow the feedback - through the seedgraph_v3 MCP tools. Run only when the researcher asks to work a named batch (for example by typing /seedgraph-v3-work <batch_id>).
disable-model-invocation: true
argument-hint: <batch_id>
---

# Work a Seedgraph batch (v3)

Batch to work: `$ARGUMENTS` (a batch id such as `b_...`). If none was given, ask for it.
Do not create a batch unless the researcher asks for one.

Use the `seedgraph_v3` MCP tools. Every reply is JSON with a `next_action` in plain
words; follow it. If the MCP tools are not available, the same steps exist as
`seedgraph-v3 work claim|submit|release|status ...` commands that print JSON.

## The loop

1. `work_claim(batch_id, worker?)`. Pass `worker` (for example `w1`) when several agents
   share this session, so reports tell you apart.
   - `status: claimed` - the reply's `package` names `text_path`, `instructions_path`,
     `labels_path`, `result_path`, `packet_id`, `claim_id`, `part` / `part_count`.
   - `paused` or `finished` - stop and report (see below).
   - `no_package` - other workers hold the rest; stop and report.
2. Read `instructions.md` (once per batch; it is the same for every package) and
   `labels.json`. They hold the task's fields, rules, vocabularies and an example.
3. Read the package text file. It is the paper with its reference list cut out, or one
   part of a long paper. Work only from this text. It can be large: read it in ranges
   (about 300 lines at a time; the package gives `lines`).
4. Write the result as JSON alone (no prose, no code fence) to `result_path`.
5. Submit the result file: `work_submit(packet_id, claim_id, note_path=result_path)`
   (CLI: `seedgraph-v3 work submit PACKET --claim CLAIM --note-file result_path`). If you
   cannot write files, pass the JSON as `note` instead. Pass `model` on every submit: your
   model id as your client states it (for example in your system prompt). Spot-check
   accuracy is reported per model. If your client exposes token usage, write it to a JSON
   file and pass `--usage-file`
   (CLI) or `usage` (MCP); otherwise skip it.
6. Read the reply:
   - `accepted` - claim the next package (step 1).
   - `invalid` - fix every listed error and submit the same `packet_id` again.
   - `needs_revision` - for each `failed_quotes` row, replace your quote with its
     `markdown_slice` exactly, character for character, or set that claim's status to
     ambiguous. Resubmit the same `packet_id`. Never "repair" characters in a quote.
     Only when a quote cannot be made to match, resubmit with `accept_unanchored: true`
     (it is saved as ambiguous).
7. Repeat until a claim reply says paused, finished or no package.

Quotes are checked against the stored paper text, so copy them exactly and make each one
long enough to be unique.

## Concept packages

Batches of `concepts:merge`, `concepts:bridge` and `concepts:name_literatures` have the
same loop, but each package is a numbered list of items, not a paper.

- Read `instructions.md` once: it holds the rules, the verdict table, the answer schema
  and worked examples. For concept tasks `labels.json` only names the frozen manifest
  Seedgraph keeps of each package (its claim refs, papers and spans); there is nothing in
  it to answer, so you need not open it.
- Answer every item exactly once, by number. Cite claim refs only (`c17`); never write
  quotes - the refs already point at anchored quotes.
- Verdicts: `merge` (true synonyms in one sense; `name` is one of the item's labels),
  `link` (a difference that matters, or drift; needs `note`, `quote_a` from side A and
  `quote_b` from side B), `split` (one label used in different senses; `senses` with a
  gloss and claim refs each), `unrelated`, `same_sense` (sense checks only) and `unsure`.
  Give `confidence` (high, medium or low).
- Judge meaning from the claims, never from a shared word; an item marked "ambiguous
  label" needs a ref from each side for a merge or link.
- A link is for near-equivalents whose difference matters, not for pairs that are merely
  on related topics, used together, or where one is an input to the other (those are
  `unrelated`); `instructions.md` gives the test.
- When unsure between merge and link, answer link. When unsure between link and
  unrelated, answer unrelated. `unsure` is an honest answer: the item goes to the
  researcher, whose decisions always win.
- A package may repeat a claim under different refs; cite whichever ref the item lists.
- A refusal (`invalid`) names each failing item, for example a merge vetoed by a
  guardrail (it names the triggering word, families or pair): answer link, unrelated or
  unsure for it and resubmit.
- If your client exposes token usage, write it to a JSON file and pass `--usage-file`
  (CLI) or `usage` (MCP); otherwise skip it.

## When something goes wrong

- The paper text is unreadable or is not the paper the title says:
  `work_release(packet_id, claim_id, "paper_problem", message)` and claim again.
- You must stop mid-package (session ending): `work_release(..., "give_back")`.
- Your client reports a usage limit: `batch_report_limit(batch_id, resets_at?, message?)`
  with the reset time if the client said it, then stop. Your claims go back to the pool;
  the researcher continues the batch later.
- A refusal with code `claim_reclaimed` means your claim expired and another worker took
  the package: drop it and claim again.

## Fanning out

To work faster, give each helper agent the same instructions with its own `worker` name
(`w1`, `w2`, ...). Claims never collide; each package goes to one worker.

## Report

When you stop, call `batch_status(batch_id)` and report in a few lines: papers done,
held back, failed, still open; how many submissions needed a revision; the batch state
and its `next_action`.
