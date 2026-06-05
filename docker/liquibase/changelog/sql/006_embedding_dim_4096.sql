-- Switch embedding dimension to 4096 (e.g. text-embedding-qwen3-embedding-8b).
-- Existing vectors of a different dimension are not castable, so we truncate and rebuild the index via /reindex.
--
-- IMPORTANT: pgvector ivfflat index supports up to 2000 dimensions, so for 4096 we intentionally do NOT recreate
-- ivfflat indexes. Retrieval will fall back to a sequential scan + distance sort, which is acceptable for small corpora.

DROP INDEX IF EXISTS rag_chunks_embedding_idx;
DROP INDEX IF EXISTS rag_illustration_chunks_embedding_idx;
DROP INDEX IF EXISTS rag_page_image_embeddings_emb_idx;

TRUNCATE TABLE rag_chunks;
TRUNCATE TABLE rag_illustration_chunks;
TRUNCATE TABLE rag_page_image_embeddings;

ALTER TABLE rag_chunks
    ALTER COLUMN embedding TYPE vector(4096);

ALTER TABLE rag_illustration_chunks
    ALTER COLUMN embedding TYPE vector(4096);

ALTER TABLE rag_page_image_embeddings
    ALTER COLUMN embedding TYPE vector(4096);

