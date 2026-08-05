-- Build B: analysis/ranking columns (decision 71 weight; §5.3 co-occurrence strength).
ALTER TABLE concepts ADD COLUMN weight REAL NOT NULL DEFAULT 0.0;
    -- anti-stopword IDF: log(N_total / paper_frequency) over the STAGED extraction
    -- set (distinct works in extracted_claims), NOT the whole citation graph.
    -- Recomputed on every concepts build (table is delete-and-rewrite).
ALTER TABLE project_graph_edges ADD COLUMN shared_count INTEGER;
    -- co_occurs_with strength provenance: |P(A) ∩ P(B)|; NULL on non-co-occurrence
    -- edge types. Jaccard goes in the existing (previously writer-less) confidence REAL.
