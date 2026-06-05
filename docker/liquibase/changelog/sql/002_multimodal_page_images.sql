-- Per-page image embeddings for multimodal retrieval.
-- Размерность должна совпадать с IMAGE_EMBEDDING_DIM / EMBEDDING_DIM (часто 1024 при общем API с текстом).

CREATE TABLE IF NOT EXISTS rag_page_image_embeddings (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    source_page TEXT NOT NULL,
    embedding vector(1024) NOT NULL,
    UNIQUE (doc_id, source_page)
);

CREATE INDEX IF NOT EXISTS rag_page_image_embeddings_emb_idx
    ON rag_page_image_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
