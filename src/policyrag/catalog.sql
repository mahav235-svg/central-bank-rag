-- Corpus catalog and evaluation store (DuckDB).
-- Parsed documents and their chunks live here; the vector index only holds embeddings plus the
-- metadata needed for filtering, and can be rebuilt from this file at any time.

CREATE TABLE IF NOT EXISTS documents (
    doc_id        VARCHAR PRIMARY KEY,
    bank          VARCHAR NOT NULL,
    doc_type      VARCHAR NOT NULL,
    meeting_date  DATE    NOT NULL,
    published     DATE,
    title         VARCHAR NOT NULL,
    url           VARCHAR NOT NULL,
    n_sections    INTEGER NOT NULL,
    n_words       INTEGER NOT NULL,
    ingested_at   TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id      VARCHAR PRIMARY KEY,
    doc_id        VARCHAR NOT NULL,
    position      INTEGER NOT NULL,
    heading       VARCHAR,
    speaker       VARCHAR,
    n_words       INTEGER NOT NULL,
    text          VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id        VARCHAR PRIMARY KEY,
    kind          VARCHAR NOT NULL,      -- 'retrieval' or 'answer'
    label         VARCHAR NOT NULL,      -- e.g. 'hybrid+rerank'
    config        JSON,
    started_at    TIMESTAMP NOT NULL DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS eval_results (
    run_id        VARCHAR NOT NULL,
    question_id   VARCHAR NOT NULL,
    metric        VARCHAR NOT NULL,
    value         DOUBLE,
    detail        VARCHAR,
    PRIMARY KEY (run_id, question_id, metric)
);

CREATE OR REPLACE VIEW corpus_summary AS
SELECT bank,
       doc_type,
       count(*)                     AS documents,
       min(meeting_date)            AS first_meeting,
       max(meeting_date)            AS last_meeting,
       sum(n_words)                 AS words,
       (SELECT count(*) FROM chunks c JOIN documents d2 USING (doc_id)
         WHERE d2.bank = d.bank AND d2.doc_type = d.doc_type) AS chunks
FROM documents d
GROUP BY bank, doc_type
ORDER BY bank, doc_type;

CREATE OR REPLACE VIEW eval_summary AS
SELECT r.kind, r.label, e.metric, round(avg(e.value), 4) AS mean, count(*) AS n, max(r.started_at) AS run_at
FROM eval_results e JOIN eval_runs r USING (run_id)
GROUP BY r.kind, r.label, e.metric
ORDER BY r.kind, e.metric, mean DESC;
