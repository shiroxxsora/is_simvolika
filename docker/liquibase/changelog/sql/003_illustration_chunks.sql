CREATE TABLE IF NOT EXISTS rag_illustration_chunks (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    doc_name TEXT NOT NULL,
    source_doc TEXT,
    source_chapter TEXT,
    source_page TEXT,
    content_hash TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1024) NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS rag_illustration_chunks_doc_hash_chunk_uniq
    ON rag_illustration_chunks (doc_id, content_hash, chunk_index);

CREATE INDEX IF NOT EXISTS rag_illustration_chunks_embedding_idx
    ON rag_illustration_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
