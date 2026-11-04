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
- For a question about ideas across a graph, start with `literatures(graph)` and
  `concept_hubs(graph?, family?, limit)` (also the resource
  `seedgraph://graphs/{slug}/hubs`): the graph's literatures, its most common concepts
  and the bridges between literatures. Search concepts before spans when the question is
  about an idea (see "Concepts" below).

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

## Record your answer

Seedgraph keeps a record of each answer you give from the library: the searches, what
you opened and what you cited. The researcher reads these records later.

- When you answer the researcher from the library, call `record_answer(question, answer,
  evidence, judgement?, session?)` once per answer, at the end. Do not record answers
  that did not use the library.
- Choose a session name at the start of each question thread: a topic slug plus the
  clock time, for example `parallel-trends-1432` (lowercase letters, digits and hyphens).
  Pass it as `session` on every seedgraph call for that question and its follow-ups,
  and on `record_answer`. Every reply echoes the `session` in use.
- When you start subagents to search for you, give each a helper name `<yours>/a`,
  `<yours>/b`, and tell it to pass that name as `session`. If you are a subagent and were
  given no name, choose your own.
- Cite only what you opened: read the passage with `paper_read` (or see its quote in
  `concept_show`) before you cite a span. A search hit alone counts as found, not read,
  and a claim's anchors in `paper_show` give pages, not text.
- `evidence` lists what you cite: `{"span_id": ...}`, `{"claim_id": ...}` or
  `{"paper_id": ..., "page": n}`, each with an optional `"why"`.
- Put what the papers say in `answer` and your own view in `judgement`.
- If the reply flags a citation ("cited without being read"), tell the researcher which
  citations were not read, or read them and record again. The reply's `page` is the
  record's page in the Seedgraph app.

Search replies carry a one-line `record_reminder`; it is the same rule.

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

## Concepts, senses and literatures

A **concept** groups claims from many papers that name the same idea. Concept identity is
**label + sense**: one label can carry several meanings ("identification" as a source of
exogenous variation, or as a unique solution of a system of equations). Handles read
`family/label` or `family/label~s2` for one sense of a split label.

- `concept_search(query, family?, graph?, k)` - sense-level hits: handle, name, gloss,
  paper and claim counts, and the best example claim with its quote. `search_mode` says
  whether the search ran by keyword or by meaning; with keyword search, try the words a
  paper would use.
- `concept_show(handle, graph?, claims_offset, claims_limit)` - the member senses, the
  supporting claims with quotes, merges, closely related links, literatures (with
  `graph`) and often-appears-with. A split label answers with a chooser: pick a sense.
- `literatures(graph)` - each literature's name, size and top concepts; bridges; and
  concepts marked "same words, sense not checked".

How to read them:

1. A **merge** joins true synonyms in one sense. A **closely related link** keeps two
   concepts apart: its note states the difference that matters, or the history when two
   terms drifted together. Do not describe linked concepts as the same thing; give the
   note.
2. An **ambiguous label** is used in several senses across the literature. Never treat an
   ambiguous label, or a concept marked "same words, sense not checked", as one concept
   across literatures or as evidence of a cross-literature connection. Compare the claims
   themselves, and say the senses have not been checked.
3. A **bridge** is a checked concept (or a closely related link) that spans literatures.
   Only bridges show a connection between literatures.
4. Researcher decisions win: a researcher's merge, link, split, gloss or literature name
   overrides an agent's. Glosses, link notes and literature names carry `gloss_by`,
   `note_by` or `name_source`. Say "agent inferred" when you use an agent's text, and
   "researcher" for the researcher's.
5. Counts and claims leave out hidden notes and claims with no quote; cite a concept's
   claims as you cite any claim (title, year, page), after reading the passage.

## Lenses

`lens_results(lens_id, graph?, status?)` lists what a lens found in each paper's current
text; with `graph` it also names the members with no current result.

## What not to do

- Do not create batches or claim work unless the researcher asks; noting papers is the
  `seedgraph-v3-work` skill's job.
- Do not paste long stretches of paper text into the conversation; quote the sentence
  you rely on.
