## Seedgraph corpus — consult it (project literature)

This project has a **Seedgraph corpus** (slug `{slug}`): a curated,
provenance-preserving graph of its literature — concepts, citation lineage, and
verbatim evidence spans. Treat it as a **grounded colleague who has read this
literature**, not a lookup tool.

**Reflex (the important part).** Consult the corpus when answering well **turns on a
fact about what this literature says or does** — what the field does about X,
positioning an idea or checking novelty, drafting/outlining a lit review, or
comparing/tracing lineage across papers — anything you'd otherwise state from your
training prior. On those turns reach for it the way you'd reach for `grep`, without
being asked; your training prior is *not* authoritative about this literature.
**Skip turns about your own work** — your method, data, code, or plan — that don't
hinge on a corpus fact, and if you already pulled the corpus on this thread, build on
what you have rather than re-querying every follow-up. When a borderline turn *does*
hinge on a literature fact, consult — silent non-use is the failure this corpus
exists to prevent.

**Two-part answer.** (1) Report what the corpus actually says — grounded, evidence
attached, including honest negatives ("no paper states X"). (2) Then give your
judgment, clearly marked as *your* inference, with the evidence trail shown. You are
licensed to notice what the literature hasn't made explicit (e.g. a latent link
between two concepts via a shared third) — as long as you show your work.

**Trust tiers — know which you're standing on.** Navigate primarily by the
**semantic layer** (concepts + relations — LLM-inferred, a *hypothesis to check*,
never a fact alone). Drill into the **citation graph** (deterministic lineage) and
**evidence spans** (verbatim, quote-hashed — ground truth) to support any claim.

**Commands** (corpus resolves under `~/.seedgraph` by default; `--project` for
ask/search, positional slug for the rest):
- Concepts first: `seedgraph concepts list {slug}` → `seedgraph concepts show {slug} <concept_id>`
- Grounded answer: `seedgraph ask "<q>" --project {slug} --json`  (`--no-llm` = retrieval only; every ask persists an inspectable trace — `answer trace-export`)
- Span search: `seedgraph search "<term>" --project {slug}`
- Structure: `seedgraph graph analyze {slug}`
- Coverage: `seedgraph corpus status {slug}`
