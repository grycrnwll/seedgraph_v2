---
name: seedgraph-v3
description: Ground answers about research papers in the researcher's Seedgraph library through the seedgraph_v3 MCP tools - search paper text, notes and claims, read bounded slices of a paper, and see what a context graph holds. Use when a question about methods, assumptions, results or the literature in the researcher's library would change with evidence from the papers; skip it for general knowledge.
---

# Seedgraph library (v3)

The researcher keeps one **library** of papers. A **context graph** is a named selection
of library papers for one line of research (its members, a frontier of cited papers not
yet held, seeds and chosen lenses). Tools take a graph by its slug, `graph`.

Use the `seedgraph_v3` MCP tools. Every reply is JSON. An error reply carries `code`,
`message` and `next_action`; follow `next_action`.

## Find your footing

- `graph_list` - the context graphs and their sizes. If the researcher names a topic,
  pick the matching graph; ask if two could fit.
- `graph_show(graph)` - seeds, chosen lenses, and members that still have no note.
- `paper_list(graph?, status?, needs?, query?, offset, limit)` - papers by title text,
  graph membership (`status` member or frontier), or need (`needs` note or metadata).

## Ground a claim

1. `search(query, kind, graph?, paper_id?, section_kind?, limit)`. `kind` is `claims`
   (labelled statements from notes), `notes` (whole notes) or `spans` (the paper text).
   The query is matched as one phrase, so search for the words a paper would use:
   "parallel trends", not "is the parallel trends assumption testable".
2. `paper_show(paper_id)` - metadata, the section outline, and the current note's claims
   with the pages their quotes come from.
3. `paper_read(paper_id, start?, end? | section?)` - the paper's own text, at most
   48 000 characters per call; continue from `next_start`. Read the passage before you
   rely on a claim.

Cite what you used as paper title, year and page (from the claim's anchors or the span's
`page_start`). Say plainly when the library has nothing on a point; do not fill the gap
from memory as if it came from the papers.

## The researcher's version

Notes and claims are the researcher's current version of each paper. The researcher may
have reworded a claim, hidden one, added their own, or written `researcher_notes`. Every
claim carries `trust`, a `trust_label` and a `trust_note`:

- `paper states` - an agent claim backed by a quote from the paper;
- `agent inferred` - an agent claim with no quote (inferred, ambiguous or not found);
- `researcher` - a claim the researcher edited (`edited_by_researcher: true`, with the
  agent's wording under `original`) or added. An added claim with `claim_kind`
  `assertion` has no quote: its `trust_note` reads "researcher assertion - not the
  paper's words". An added `quote` claim's anchor is the paper's words; its claim text
  is the researcher's.

Three rules:

1. When the researcher's version and the agent's `original` (or your own reading)
   disagree, follow the researcher's version and say so.
2. Never present a researcher assertion as the paper's text. Quote the paper only from
   a claim's anchors or from `paper_read`.
3. Say "researcher" when you cite researcher content, for example "the researcher notes
   that..." or "(edited by the researcher)".

## Hidden notes

Some papers have `notes_hidden: true` because an open attention item (for example a
possibly wrong PDF) is waiting for the researcher. Their notes and claims are left out
of search and `paper_show` on purpose. Mention the item if it matters; do not try to
work around it. The paper text itself stays readable.

## Lenses

`lens_results(lens_id, graph?, status?)` lists what a lens found in each paper's current
text; with `graph` it also names the members with no current result.

## What not to do

- Do not create batches or claim work unless the researcher asks; noting papers is the
  `seedgraph-v3-work` skill's job.
- Do not paste long stretches of paper text into the conversation; quote the sentence
  you rely on.
