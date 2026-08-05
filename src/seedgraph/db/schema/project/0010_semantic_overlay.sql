-- project.db — Phase 7 (Semantic graph overlay). Authoritative schema (D6).
--
-- NEW tables: concepts, concept_aliases, claim_concepts, project_graph_edges
-- (finalized constrained polymorphic overlay), edge_spans (defined, NOT
-- populated this phase), concept_constraints. Applies on top of phase_0 +
-- every lower-numbered project migration (works / extracted_claims /
-- claim_spans / evidence_spans / review_queue must already exist).
--
-- Binding decisions honored here:
--   D2  — origin field is named `epistemic_type` everywhere (claims AND edges);
--         closed 6-tier CHECK. `assertion_status` is claim-only and is NOT a
--         column on any edge/concept table (graph edges are not author
--         assertions).
--   D6  — this numbered .sql is the single source of truth; no CREATE TABLE in
--         Python, no create_all() authoring/evolution. The ORM classes only
--         mirror this schema.
--   D10 — concept identity is deterministic ('concept::' || normalized_label);
--         anti-overmerge guard + borderline→review_queue is an application
--         concern; the schema makes user splits/merges sticky via
--         concept_constraints. Perfect merges are NOT required.
--
-- No cross-database FK (project.db-local only). PRAGMA foreign_keys=ON is set by
-- db/connection.py for project.db; every FK below targets a project.db table.
-- No `project_id` column anywhere (one project per project.db; the slug is the
-- scope). `access_class` carries NO `IN (...)` CHECK on purpose — it is stamped
-- fail-closed by the application (MAX over contributing extracted_claims) and a
-- runtime CHECK would have to enumerate the full AccessClass lattice and stay in
-- lockstep with vocab.py; the export gate (phase_0 field_allowed) is the guard.

-- Canonical concept. concept_id is DETERMINISTIC = 'concept::' || normalized_label.
-- normalized_label is the merge_key (UNIQUE). concept_type is an attribute, not
-- identity, drawn from the single ClaimType vocab in vocab.py (decision 48).
CREATE TABLE concepts (
    concept_id        TEXT PRIMARY KEY,                              -- 'concept::' || normalized_label
    normalized_label  TEXT NOT NULL UNIQUE,                          -- merge_key = normalize(canonical_label)
    canonical_label   TEXT NOT NULL,                                 -- human-facing surface form (field-typed label wins)
    concept_type      TEXT NOT NULL,                                 -- ClaimType vocab (decision 48)
    definition        TEXT,                                          -- borrowed sentence from a contributing claim; nullable; full-text-derived
    paper_frequency   INTEGER NOT NULL DEFAULT 0,                    -- distinct works mentioning it
    status            TEXT NOT NULL DEFAULT 'auto'
        CHECK (status IN ('auto','provisional','user_confirmed','user_split')),
    epistemic_type    TEXT NOT NULL DEFAULT 'deterministic'         -- D2 origin field (6-tier closed)
        CHECK (epistemic_type IN
          ('deterministic','metadata_resolved','llm_extracted','llm_inferred','user_supplied','user_validated')),
    access_class      TEXT NOT NULL DEFAULT 'user_supplied_private', -- MAX over contributing extracted_claims.access_class
    run_id            TEXT,                                          -- build run that last produced it
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- Alias = TRUE interchangeability only (merged_from pre-image of a fold).
CREATE TABLE concept_aliases (
    alias_id        INTEGER PRIMARY KEY,                             -- internal-only join row
    concept_id      TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    alias_label     TEXT NOT NULL,                                   -- the raw/surface label folded in
    fold_reason     TEXT NOT NULL
        CHECK (fold_reason IN ('exact_key','acronym','llm_proposed_reviewed')),
    epistemic_type  TEXT NOT NULL DEFAULT 'deterministic'
        CHECK (epistemic_type IN
          ('deterministic','metadata_resolved','llm_extracted','llm_inferred','user_supplied','user_validated')),
    run_id          TEXT,
    created_at      TEXT NOT NULL,
    UNIQUE(concept_id, alias_label)
);

-- Claim --mentions_concept--> Concept. High-volume deterministic junction.
CREATE TABLE claim_concepts (
    claim_concept_id INTEGER PRIMARY KEY,
    claim_id         TEXT NOT NULL REFERENCES extracted_claims(claim_id) ON DELETE CASCADE,
    concept_id       TEXT NOT NULL REFERENCES concepts(concept_id) ON DELETE CASCADE,
    work_id          TEXT NOT NULL REFERENCES works(work_id),
    epistemic_type   TEXT NOT NULL DEFAULT 'deterministic'          -- deterministic (label match) | llm_extracted
        CHECK (epistemic_type IN
          ('deterministic','metadata_resolved','llm_extracted','llm_inferred','user_supplied','user_validated')),
    confidence       REAL,
    run_id           TEXT,
    created_at       TEXT NOT NULL,
    UNIQUE(claim_id, concept_id)
);
CREATE INDEX ix_claim_concepts_concept ON claim_concepts(concept_id);
CREATE INDEX ix_claim_concepts_work    ON claim_concepts(work_id);

-- Constrained polymorphic interpretive overlay. Deterministic citation edges
-- stay in citation_edges; this table is interpretive edges ONLY. Polymorphic
-- endpoints cannot be FK'd in SQLite — integrity is enforced at the single
-- application insert path (edges.py::insert_edge) + a doctor dangling scan.
-- Changes vs the doc 03 §9 sketch: add run_id + access_class, drop the scalar
-- evidence_span_id (evidence, if ever needed, via edge_spans), add closed-vocab
-- CHECKs.
CREATE TABLE project_graph_edges (
    edge_id           TEXT PRIMARY KEY,                              -- 'pge_' || uuid4hex
    source_node_type  TEXT NOT NULL,
    source_node_id    TEXT NOT NULL,
    target_node_type  TEXT NOT NULL,
    target_node_id    TEXT NOT NULL,
    edge_type         TEXT NOT NULL,                                 -- discusses|related_to|broader_than|narrower_than|contrasts_with|co_occurs_with|...
    epistemic_type    TEXT NOT NULL,                                 -- D2 origin field (closed 6-tier)
    confidence        REAL,
    access_class      TEXT NOT NULL DEFAULT 'user_supplied_private',
    run_id            TEXT,
    created_at        TEXT NOT NULL,
    CHECK (source_node_type IN ('Work','Concept','Claim','EvidenceSpan')),
    CHECK (target_node_type IN ('Work','Concept','Claim','EvidenceSpan')),
    CHECK (epistemic_type IN
      ('deterministic','metadata_resolved','llm_extracted','llm_inferred','user_supplied','user_validated')),
    UNIQUE(source_node_type, source_node_id, target_node_type, target_node_id, edge_type, run_id)
);
CREATE INDEX ix_pge_src ON project_graph_edges(source_node_type, source_node_id);
CREATE INDEX ix_pge_tgt ON project_graph_edges(target_node_type, target_node_id);

-- Evidence junction for span-bound interpretive edges. Shape defined now
-- (decision 70); NOT populated in phase 7 (no concept-layer edge type carries
-- single-span evidence yet). First consumer is the deferred citation-context
-- enrichment.
CREATE TABLE edge_spans (
    edge_id  TEXT NOT NULL REFERENCES project_graph_edges(edge_id) ON DELETE CASCADE,
    span_id  TEXT NOT NULL REFERENCES evidence_spans(span_id),
    PRIMARY KEY (edge_id, span_id)
);

-- Sticky human decisions consulted before each canon run (closes the v1
-- flip-on-growth gap). Must/cannot-link constraints over normalized labels.
CREATE TABLE concept_constraints (
    constraint_id  INTEGER PRIMARY KEY,
    kind           TEXT NOT NULL CHECK (kind IN ('must_link','cannot_link')),
    label_a        TEXT NOT NULL,                                    -- normalized labels
    label_b        TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'user',                    -- user (review resolution)
    created_at     TEXT NOT NULL,
    UNIQUE(kind, label_a, label_b)
);
