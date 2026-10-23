-- library.db 0007 — membership of automatically merged papers (v3 design §4.2, §7.3).
--
-- A rule merge (two or more v2 works joined into one paper by DOI, OpenAlex, arXiv,
-- identical PDF or a unique title+year match) yields ONE paper, so it cannot be a
-- duplicate_decisions pair. Each member work of a merged paper gets one row here,
-- naming the key through which it first joined the group and the work on the other
-- side of that join, so the researcher can split the merge later. Papers that stayed
-- separate keep using duplicate_decisions (researcher verdicts).
-- The v2 work -> paper mapping itself is also in v2_id_map.

CREATE TABLE paper_merge_members (
    paper_id            TEXT NOT NULL REFERENCES papers(paper_id),
    v2_project          TEXT NOT NULL,
    v2_work_id          TEXT NOT NULL,
    joined_by_key_type  TEXT NOT NULL
        CHECK (joined_by_key_type IN ('doi', 'openalex', 'arxiv', 'pdf', 'title_year')),
    joined_by_key_value TEXT NOT NULL,
    joined_with_project TEXT NOT NULL,
    joined_with_work_id TEXT NOT NULL,
    origin              TEXT NOT NULL DEFAULT 'migration',
    PRIMARY KEY (v2_project, v2_work_id)
);
CREATE INDEX ix_paper_merge_members_paper ON paper_merge_members (paper_id);
