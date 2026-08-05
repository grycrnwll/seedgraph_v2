---
name: seedgraph
description: Consult this project's Seedgraph literature corpus — a provenance-preserving graph (concepts → citation lineage → verbatim evidence spans) queried BY MEANING (Claude's own semantic match, near-synonyms included; never lexical). Use PROACTIVELY when answering well turns on a fact about what this literature says or does: what the field does about X, positioning an idea or checking novelty, drafting or outlining a literature review, or comparing/tracing citation lineage across papers — anything you'd otherwise assert from training-prior memory. Skip turns about the user's own work (method, data, code, plan) that don't hinge on a corpus fact, and don't re-query a thread you already grounded. Grounds claims in the corpus instead of training-prior memory, surfaces cross-community bridges and god nodes, and gives plain-language node explanations. Also invoked explicitly via /seedgraph.
trigger: /seedgraph
---

**Fire proactively — but only when grounding changes the answer.** Invoke this skill
when answering well turns on a fact about what this literature says or does, the way a
coding agent reaches for `grep`: asking what the field does about X; positioning an
idea or checking novelty; drafting or outlining a literature review; comparing,
connecting, or tracing lineage across papers — anything you'd otherwise assert from
your training prior. **Your training prior is not authoritative about this literature
— consult the corpus.** Skip turns about the user's own work — their method, data,
code, or plan — that don't hinge on a corpus fact, and if you already pulled the
corpus on this thread, build on what you have rather than re-querying every follow-up.
When a borderline turn *does* hinge on a literature fact, invoke anyway — silent
non-use is the failure this corpus exists to prevent. The skill is also invoked
explicitly when the user types `/seedgraph`.

When you fire proactively (no explicit `query`/`explain` and no `[project]`), default
to the **query** workflow with the user's research turn as the query term, and
resolve the project slug per step 2 below (one project → use it; several → ask).

This skill is the **semantic query front door** to a seedgraph V2 project. It does
NOT build a graph — the seedgraph CLI's build phases do that (`corpus`, `cite`,
`extract`, `concepts build`). It lets Claude *read* a project's concept table and
citation graph and reason over them **at query time**. The crucial point that
distinguishes this skill from a normal search: **the semantic match is Claude's own
understanding of the text, not a lexical/string score.** There is NO bundled helper
in V2 — everything lives in SQLite under `~/.seedgraph/projects/{slug}/project.db`,
and the skill drives the `seedgraph` CLI directly. Never grep/regex/string-match the
query term to pre-filter a concept or paper list — the matches you want are often
near-synonyms that share no words with the term. Load the WHOLE list and select by
meaning.

---

## Usage

```
/seedgraph query "<term or phrase>"   [project]
/seedgraph explain "<node title or work_id>"   [project]
```

- `query` — find the concepts/papers whose **meaning** matches the term (near-synonyms
  included: "co-integration" ≈ "error-correction model", "deep learning" ≈ "neural
  network training"), pull the papers/claims/spans beneath them, run a query-driven
  retrieval + citation-neighborhood pass, and **surface any cross-community bridge**
  and any god node in the graph's structural summary.
- `explain` — give a plain-language, graph-grounded account of a single paper: what
  it is about and HOW it sits in this corpus's citation structure.
- `[project]` — a **project slug** (e.g. `test_proj`), NOT a directory. **This is
  changed from v1**, which took a project *path*. There is no "current directory"
  project in V2 — every project lives under the seedgraph home, keyed by slug. If
  omitted, resolve it per step 2 below.

---

## What you must do when invoked

**1. Resolve the CLI invocation once, at skill start.**

There is no bundled helper — you call the `seedgraph` CLI directly. Try, in order:

```
seedgraph version
python -m seedgraph version
```

(Not `--version` — that flag doesn't exist on this CLI; `version` is a subcommand
that prints the package + SQLite version and exits 0.) Whichever form exits 0, use
that as the command prefix for every command below (call it `SG` in your head — e.g.
`SG project list`). If BOTH fail, STOP: tell the user seedgraph V2 isn't
installed/importable in this environment, and don't proceed.

On the machine this skill was verified on, both forms work (`seedgraph` is on PATH
via an installed console-script entry point, and `python -m seedgraph` also works)
— but don't hardcode that assumption; re-probe each time, since the invoking
environment may differ.

**2. Resolve the project (slug, not a directory).**

- If `[project]` was given, use it as the slug directly.
- If omitted, run `SG project list` (one slug per line). If exactly one project
  exists, use it. If several, list them and ask the user which one. Never guess.

**3. Check a graph actually exists before querying — STOP if not.**

Run `SG concepts list <slug>` and `SG graph analyze <slug>`. If `concepts list`
comes back empty, there is no concept layer yet — tell the user to run
`SG concepts build <slug>` first. If `graph analyze` reports zero works/edges (or
errors because no citation run exists), there is no citation graph yet — tell the
user to run the corpus + cite phases first (`SG corpus run <slug>` then
`SG cite run <slug>`, or the individual `corpus resolve/walk/acquire` and
`cite project/build` steps). Do NOT invent a graph in either case — tell the user
what to run, then have them re-run `/seedgraph`. It's fine for query to proceed with
only ONE of the two artifacts present (e.g. concepts but no citation graph yet) as
long as you say plainly which half is missing.

Then proceed to `## For query` or `## For explain` below.

---

## V1 → V2 mechanism map (for reference — v1 users, read this)

V1 drove a bundled `sg_graph.py` script that read on-disk `runs/run-*/graph/graph.json`
files (plus an optional `graphify/graph.json` concept layer). V2 has none of that —
there is no bundled helper, no `graph.json`, no `runs/` directory to discover. Every
project's data lives in one SQLite file, and the seedgraph CLI is the only interface:

| V1 | V2 |
|---|---|
| `sg_graph.py table` (citation node table) | `SG project show <slug>` (corpus) + `SG graph analyze <slug>` (structure) |
| `sg_graph.py concepts` (concept layer) | `SG concepts list <slug>` |
| `sg_graph.py lightup ID1,ID2` | `SG concepts show <slug> <concept_id>` (one concept at a time) |
| `sg_graph.py neighborhood UIDs --depth N` + bridge detection | `SG ask "<query>" --project <slug> --no-llm --graph-depth N` (query-driven neighborhood) and `SG graph analyze <slug>` (whole-graph god nodes + bridges) |
| `notes/{uid}.md`, `marker_out/{uid}/paper.md` grounding text | verbatim **evidence spans** returned by `ask` / `search` / `concepts show` (source-linked; stronger provenance than v1 notes) |
| project = directory containing `runs/` | project = **slug** |

One v1 capability has **no clean V2 equivalent**: v1's `lightup` returned
`related_concepts` — concepts reached via a typed semantic edge
(`co_occurs_with`/`jaccard`, `conceptually_related_to`, `implements`,
`shares_data_with`) from a lit-up concept, giving an explicit "what to light up next"
list. `concepts build` still constructs `co_occurs_with` edges in the database, but
there is currently no CLI command that surfaces them — `concepts show` returns only
the concept + its papers/claims/spans, nothing about neighboring concepts. **Do not
paper over this**: if the user wants "concepts related to X," the only honest path
right now is a second manual pass over the WHOLE `concepts list` table by meaning
(and say so), not a graph traversal.

---

## For query

Goal: given `query "<term or question>"`, light up the concepts whose MEANING
matches, pull the papers/claims/spans beneath them, run a query-driven retrieval +
citation pass, and report the structural picture (god nodes, cross-community
bridges) — distinguishing *semantic* reach (shared concept) from *citation* reach
(an actual `cites` edge).

**Step (a) — Orient with the overview sheet FIRST, then scope by type — never a
lexical pre-filter.**

```
SG concepts overview <slug>
```

The overlay on a real corpus is large (a live project carries ~12,800 concepts, ~88%
of them mentioned in a single paper) — dumping the whole `concepts list` table
drowns orientation. `overview` is the compact, **recurrence-ranked** reference sheet:
a one-line scale summary; a **background frame** of the ubiquitous "assumed context"
concepts (those mentioned in the most papers — generics that light up a huge slice of
the corpus and discriminate little); then, per concept_type in research-salience
order, the top few concepts by distinct-paper recurrence *with those generics factored
out*, each `Label·pf<N>`, plus a `(+N more)` tail. `pf` = distinct papers.

Read the sheet to learn the corpus's *shape* — which types are populated, what recurs,
what the assumed background is — and to pick the concept *type(s)* whose kind matches
the query (e.g. `method`, `identification_assumption`, `data`). Then scope to that type
and read the WHOLE slice by meaning:

```
SG concepts list <slug> --type <T>
```

`--type` is a structural filter on metadata, NOT a lexical filter on the query text —
do NOT grep or string-match the query term against `canonical_label`: the concepts you
want are often near-synonyms that share no words with the term. Read that whole typed
slice. (Columns: `concept_id | canonical_label | type | paper_frequency | weight |
status | access_class`, emitted **sharp-first** by `weight`.) Only fall back to the
untyped `SG concepts list <slug>` when the query genuinely spans types AND the corpus
is small enough to read whole; on a large corpus, lean on `overview` + per-type slices
instead. Ground specifics with `concepts show` / `search` (steps (c)/(d)), never off
the synthesized labels.

**Step (b) — Claude lights up the matching concepts BY MEANING, down-ranking
low-weight ones.**

`overview` and the typed `concepts list` slice rank the same reality on two axes that
agree: the overview's **background frame** is exactly the ubiquitous, low-`weight`
concepts (high `paper_frequency`, seen in many papers), and its per-type featured
lists are the distinctive, high-`weight` ones — so nothing whipsaws when you move from
the sheet to the `weight`-sorted slice.

- **`weight` is discriminativeness (IDF), not importance.** HIGH weight = a
  distinctive concept mentioned in few papers (a strong signal). LOW weight =
  ubiquitous (e.g. generic assumptions or results mentioned everywhere) — it lights
  up a big chunk of the corpus and tells you little. Prefer high-weight, on-topic
  concepts; only reach for a low-weight one when it's genuinely central to the query.
- `paper_frequency` corroborates this directly — a high-frequency concept is broad,
  a low-frequency one is sharp.
- Collect the chosen `concept_id`s **exactly as printed in column 1** (the
  `concept::…` string) — that's the only valid key into the next step.

**Step (c) — Pull papers/claims/spans beneath each chosen concept.**

```
SG concepts show <slug> <concept_id>
```

**This takes the `concept_id` from column 1 of `concepts list`, NOT the
`canonical_label`.** Passing the label instead of the id does not error — it exits 0
and prints `no concept '<label>'`, so check the message, not just the exit code, if
something looks empty. Run this once per chosen concept (it's a single-concept
query, unlike v1's `lightup` which took a batch).

Output is four kinds of tab-separated rows, tagged by a leading kind column:
- `concept  <concept_id>  <canonical_label>  <type>  <access_class>` — the concept
  itself.
- `paper  <work_id>  <title>  <year>` — every paper anchored to this concept (year
  may be blank).
- `claim  <claim_id>  <work_id>  <claim_type>  <text>` — extracted claims that
  discuss this concept, each tied to its paper.
- `span  <span_id>  <work_id>  <quote>` — verbatim evidence spans backing those
  claims. **These spans are your grounding text** — quote from here, not from the
  concept's label/definition (the label is the extractor's paraphrase/synthesis
  across papers, not a verbatim phrase from any one of them).

**Step (d) — Query-driven retrieval + citation-neighborhood expansion.**

```
SG ask "<the user's query>" --project <slug> --no-llm --graph-depth 1 --limit 40
```

**Always pass `--no-llm`.** This skill reasons over the retrieval; it does not pay
seedgraph to synthesize prose. `--no-llm` degrades `ask` to its deterministic FTS5 +
citation-expansion floor: zero LLM cost, ranked and cited. Output:
`[retrieval-only] no prose generated; ranked evidence below.`, then a `citations:`
block — `[n] work_id (title, year) <verbatim span snippet>` — ranked by relevance,
already following `--graph-depth 1` of citation-neighborhood expansion around the
retrieved works; then a `recommendations:` block listing corpus works that are
`present in the corpus as metadata-only (no extracted full text) [metadata_only]` —
**this is the CLI's own thin-evidence flag; surface it as-is** rather than silently
dropping those works.

Start at `--graph-depth 1`. Only widen to `--graph-depth 2` to try to join two
citation clusters that depth 1 leaves disconnected — and if depth 2 pulls in most of
the graph (this corpus's citation graph is dense: 1,397 works / 1,740 edges / 24
communities in the verified project, so a depth-2 expansion from a well-cited seed
can engulf a large fraction of it), say so and fall back to the depth-1 view for the
bridge claim rather than reasoning over the blob.

**Step (e) — Whole-graph structure: god nodes + cross-community bridges.**

```
SG graph analyze <slug>
```

No-LLM, deterministic. Prints, in order: a `run <run_id>: N work(s), M citation
edge(s)` header; `communities: K` with a per-community work count; `god nodes: K`
with `work_id<TAB>title` rows (high-centrality hubs); `bridges: K edge(s), M
node(s)` with rows `source_id -> target_id [community X->Y; flagged|derived]`
(`flagged` = the build already marked this edge as a bridge; `derived` = inferred
from the two ends' differing community labels); and `read next (top 10 of N
unread):` with ranked unread works. Cross-reference the `work_id`s from steps
(c)/(d) against this output to see whether any of them ARE a god node, or sit on
either end of a bridge.

**Step (f) — Report as a grounded colleague, in two parts.** First **what the corpus
actually says** — grounded, evidence attached, honest negatives included ("no paper
states X"). Then, explicitly labeled as **your own judgment**, what you infer from it
— including latent structure the literature hasn't made explicit (e.g. two concepts
linked via a shared third) — with the evidence trail shown so the reader can check
it. You are licensed to venture a view; you are trusted because you show your work,
never because you launder a guess as grounding. Within part one, tell the user:

- **Which concepts lit up** and why, naming `weight`/`paper_frequency` so the reader
  knows distinctive vs. ubiquitous. Say if you down-ranked a ubiquitous one.
- **The papers/claims/spans beneath them**, citing which concept anchored each paper,
  quoting from spans (never from the concept label).
- **What `ask --no-llm` retrieved**, including any `metadata_only` flags on thin
  works — call those out as weak evidence, don't treat them as if they were fully
  read.
- **God nodes and cross-community bridges** from `graph analyze` that touch the
  papers above.
- **CRUCIALLY distinguish two kinds of reach:** *semantic reach* — two papers share a
  lit-up concept (both anchored to it in `concepts show`) — they're about the same
  thing, this is NOT a citation. *citation reach* — an actual `cites` edge shows up
  in `ask`'s neighborhood expansion or in `graph analyze`'s bridge list — one paper
  actually references the other. Never report one as the other.

A negative answer is valid: if no concept matches the query, or the lit-up concepts'
papers sit in one community with no bridge, say so plainly — do not manufacture a
match or a bridge.

---

## For explain

Goal: given `explain "<node>"`, give a plain-language, graph-grounded account of one
paper — what it's about, and how it sits in the corpus's citation structure.

**Step 1 — Locate the work.**

```
SG project show <slug>
```

Columns: `work_id | title | year | inclusion_status | is_seed | access_status`. Pick
the single row whose title (or `work_id`) best matches the user's string — your
judgment, best match even if not exact. **Show the user which work you resolved to**
(its full `work_id` and title) before explaining, so an ambiguous request is
transparent.

**Step 2 — Gather its concepts, claims, and spans.**

- Find which concepts anchor it: scan `SG concepts list <slug>` by meaning for
  concepts plausibly about this paper's topic, then confirm with
  `SG concepts show <slug> <concept_id>` and check whether this `work_id` appears in
  the `paper` rows.
- Pull its own evidence spans directly:
  `SG search "<a distinctive term from its title>" --project <slug> --work <work_id> --limit 20`
  — this scopes FTS5 span search to just this work (exact-term, no stemming, so pick
  a phrase likely to appear verbatim).

**Step 3 — Gather its structural role.**

```
SG graph analyze <slug>
```

Check: which `community` number contains it (from the community sizes / from
appearing as a bridge endpoint), whether it's listed under `god nodes:` (a
high-centrality hub), whether it appears as a `source` or `target` in the `bridges:`
list (and if so, which communities it connects and whether `flagged` or `derived`),
and whether it shows up in `read next:` (meaning it's unread / thin evidence in this
corpus — if so, flag that explicitly, don't treat it as fully characterized).

**Step 4 — Explain in plain language.** In a few sentences, say what the paper is
about (grounded in its spans/claims from step 2) and HOW it sits in the citation
structure (step 3) — e.g. "a god node that bridges the DiD-identification cluster
(community 6) to several adjacent clusters," or "a metadata-only work with no
extracted spans yet — thin evidence, flag this to the user." Ground every claim in a
real row from the CLI output. No invented edges, concepts, or communities.

---

## Honesty Rules

- **Answer only from the CLI's output** (`concepts list/show`, `search`, `ask
  --no-llm`, `graph analyze`, `project show/list`). If the graph lacks the
  information, say so — don't fill the gap from outside knowledge.
- **Ground claims in verbatim evidence spans**, not in a concept's
  `canonical_label`. A concept label/definition is the extractor's synthesis across
  papers, not a quote from any one of them — never present it as if a paper used
  that exact phrase.
- **Never fabricate a concept, a paper, an edge, a community, or a bridge.** Only
  report what a command actually printed. If `concepts show` says `no concept
  '<x>'`, that id/label doesn't exist — don't invent it.
- **Flag thin evidence.** A work marked `metadata_only` in `ask`'s
  `recommendations:` block (or one with no `paper`/`claim`/`span` rows anywhere) has
  no extracted full text — it's a weak signal (title/metadata only). Say so plainly
  rather than treating it as read.
- **`weight` is discriminativeness (IDF), not importance.** High weight = distinctive
  (few papers); low weight = ubiquitous (many papers). Down-rank low-weight concepts
  in a query — never present `weight` as a relevance or quality score.
- **Distinguish semantic reach from citation reach.** Two papers anchored to the same
  concept are about the same thing (semantic reach) — that is NOT a citation. An
  actual `cites` edge (seen in `ask`'s neighborhood or `graph analyze`'s bridge list)
  means one paper references the other (citation reach). Never conflate the two.
- **A negative answer is a valid answer.** If nothing matches the term, or no
  cross-community bridge exists, say that honestly rather than manufacturing a match.
- **The semantic match is Claude's judgment, not a computed score.** There is no
  embedding or similarity number behind concept selection — state plainly that you
  read the concept table and chose by meaning.
- **No CLI surface for concept-to-concept relations exists in V2** (see the v1→v2
  map above). Don't imply a "related concepts" traversal happened when it didn't —
  if asked for it, say the honest workaround is a second manual scan of the whole
  `concepts list` table.

---

## Troubleshooting

### `seedgraph version` and `python -m seedgraph version` both fail

Seedgraph V2 isn't installed/importable in this environment. Stop and tell the user;
don't fall back to guessing a path or inventing output.

### "no concept `<x>`" from `concepts show`

This means `<x>` isn't a valid `concept_id` — it exits 0, so check the message text,
not just the return code. Re-copy the id from column 1 of `concepts list` (it must be
the full `concept::…` string, not the `canonical_label`).

### `concepts list` / `graph analyze` come back empty

No concept layer / no citation graph has been built yet for this project. Tell the
user to run `SG concepts build <slug>` (concepts) and/or `SG corpus run <slug>` then
`SG cite run <slug>` (citation graph), then re-run `/seedgraph`. Do not invent
output for either.

### Ambiguous or missing `[project]`

V2 projects are slugs, not directories — there's no "current directory" fallback.
Run `SG project list`; if there's exactly one, use it; if several, list them and ask.

### PowerShell quoting

Wrap the query/term in double quotes: `SG ask "some question" --project slug
--no-llm`. Multi-word `--work`/`--section-kind` filter values also need quotes.
`seedgraph` itself is shell-agnostic — only the surrounding quoting is
PowerShell-flavored.

### A command's output is very large

`concepts list` on a big project can run into the thousands of rows, and `graph
analyze`'s `bridges:` list can run into the hundreds. Read it in full for the
semantic-match steps (concepts) — never truncate before matching by meaning — but
for `graph analyze`, it's fine to grep the output for specific `work_id`s you
already care about (from steps (c)/(d)) rather than eyeballing every bridge row;
that's filtering by a key you already resolved, not lexically pre-filtering the
query term.
