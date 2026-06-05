from psycopg import connect
from psycopg import errors as pg_errors

from rag_service.config import Settings
from rag_service.retrieval_hit import RetrievalHit
from rag_service.source_page_norm import normalize_source_page


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


class RAGRepository:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def document_already_indexed(self, doc_id: str, content_hash: str) -> bool:
        with connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM rag_chunks WHERE doc_id = %s AND content_hash = %s LIMIT 1",
                    (doc_id, content_hash),
                )
                return cur.fetchone() is not None

    def delete_document_chunks(self, doc_id: str) -> None:
        with connect(self.settings.postgres_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_chunks WHERE doc_id = %s", (doc_id,))
                try:
                    cur.execute("DELETE FROM rag_illustration_chunks WHERE doc_id = %s", (doc_id,))
                except pg_errors.UndefinedTable:
                    pass
                try:
                    cur.execute("DELETE FROM rag_page_image_embeddings WHERE doc_id = %s", (doc_id,))
                except pg_errors.UndefinedTable:
                    pass

    def clear_all_chunks(self) -> None:
        with connect(self.settings.postgres_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("TRUNCATE TABLE rag_page_image_embeddings")
                except pg_errors.UndefinedTable:
                    pass
                try:
                    cur.execute("TRUNCATE TABLE rag_illustration_chunks")
                except pg_errors.UndefinedTable:
                    pass
                cur.execute("TRUNCATE TABLE rag_chunks")

    def insert_chunk(
        self,
        doc_id: str,
        doc_name: str,
        source_doc: str | None,
        source_chapter: str | None,
        source_page: str | None,
        content_hash: str,
        chunk_index: int,
        chunk_text: str,
        embedding: list[float],
    ) -> None:
        source_page = normalize_source_page(source_page)
        with connect(self.settings.postgres_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_chunks (
                        doc_id, doc_name, source_doc, source_chapter, source_page,
                        content_hash, chunk_index, chunk_text, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    """,
                    (
                        doc_id,
                        doc_name,
                        source_doc,
                        source_chapter,
                        source_page,
                        content_hash,
                        chunk_index,
                        chunk_text,
                        vector_literal(embedding),
                    ),
                )

    def insert_illustration_chunk(
        self,
        doc_id: str,
        doc_name: str,
        source_doc: str | None,
        source_chapter: str | None,
        source_page: str | None,
        content_hash: str,
        chunk_index: int,
        chunk_text: str,
        embedding: list[float],
    ) -> None:
        source_page = normalize_source_page(source_page)
        try:
            with connect(self.settings.postgres_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO rag_illustration_chunks (
                            doc_id, doc_name, source_doc, source_chapter, source_page,
                            content_hash, chunk_index, chunk_text, embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                        """,
                        (
                            doc_id,
                            doc_name,
                            source_doc,
                            source_chapter,
                            source_page,
                            content_hash,
                            chunk_index,
                            chunk_text,
                            vector_literal(embedding),
                        ),
                    )
        except pg_errors.UndefinedTable:
            return

    def search_illustration_chunks(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievalHit]:
        try:
            with connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT doc_name, chunk_text, doc_id, source_doc, source_chapter, source_page,
                               chunk_index,
                               (embedding <=> %s::vector) AS distance
                        FROM rag_illustration_chunks
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vector_literal(query_embedding), vector_literal(query_embedding), top_k),
                    )
                    rows = cur.fetchall()
        except pg_errors.UndefinedTable:
            return []
        results: list[RetrievalHit] = []
        for doc_name, chunk_text, doc_id, source_doc, source_chapter, source_page, chunk_index, distance in rows:
            source_ref = self._build_source_reference(
                source_doc, source_chapter, source_page, str(doc_id)
            )
            source_ref = f"{source_ref}; блок: описание иллюстрации"
            results.append(
                RetrievalHit(
                    doc_name=str(doc_name),
                    content=str(chunk_text),
                    source_link=source_ref,
                    distance=float(distance),
                    chunk_index=int(chunk_index) if chunk_index is not None else None,
                    fragment_kind="illustration",
                )
            )
        return results

    def fetch_illustration_chunks_for_doc_page(
        self, doc_id: str, source_page: str
    ) -> list[RetrievalHit]:
        sp = normalize_source_page(source_page) or source_page
        page_int: int | None = int(sp) if sp and str(sp).isdigit() else None
        try:
            with connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    if page_int is not None:
                        cur.execute(
                            """
                            SELECT doc_name, chunk_text, source_doc, source_chapter, source_page, chunk_index
                            FROM rag_illustration_chunks
                            WHERE doc_id = %s
                              AND (
                                source_page = %s
                                OR (
                                  source_page IS NOT NULL
                                  AND source_page ~ '^[0-9]+$'
                                  AND source_page::bigint = %s
                                )
                              )
                            ORDER BY chunk_index ASC
                            """,
                            (doc_id, sp, page_int),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT doc_name, chunk_text, source_doc, source_chapter, source_page, chunk_index
                            FROM rag_illustration_chunks
                            WHERE doc_id = %s AND source_page = %s
                            ORDER BY chunk_index ASC
                            """,
                            (doc_id, sp),
                        )
                    rows = cur.fetchall()
        except pg_errors.UndefinedTable:
            return []

        results: list[RetrievalHit] = []
        for doc_name, chunk_text, source_doc, source_chapter, source_page_val, chunk_index in rows:
            source_ref = self._build_source_reference(
                source_doc, source_chapter, source_page_val, doc_id
            )
            source_ref = f"{source_ref}; блок: описание иллюстрации"
            results.append(
                RetrievalHit(
                    doc_name=str(doc_name),
                    content=str(chunk_text),
                    source_link=source_ref,
                    distance=0.0,
                    chunk_index=int(chunk_index) if chunk_index is not None else None,
                    fragment_kind="illustration",
                )
            )
        return results

    def upsert_page_image_embedding(self, doc_id: str, source_page: str, embedding: list[float]) -> None:
        sp = normalize_source_page(source_page) or source_page
        try:
            with connect(self.settings.postgres_dsn, autocommit=True) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO rag_page_image_embeddings (doc_id, source_page, embedding)
                        VALUES (%s, %s, %s::vector)
                        ON CONFLICT (doc_id, source_page) DO UPDATE SET embedding = EXCLUDED.embedding
                        """,
                        (doc_id, sp, vector_literal(embedding)),
                    )
        except pg_errors.UndefinedTable:
            return

    def search_pages_by_image_embedding(
        self, query_embedding: list[float], top_k: int
    ) -> list[tuple[str, str, float]]:
        try:
            with connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT doc_id, source_page, (embedding <=> %s::vector) AS distance
                        FROM rag_page_image_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (vector_literal(query_embedding), vector_literal(query_embedding), top_k),
                    )
                    rows = cur.fetchall()
        except pg_errors.UndefinedTable:
            return []
        return [(str(doc_id), str(source_page), float(distance)) for doc_id, source_page, distance in rows]

    def count_page_image_embeddings_for_doc(self, doc_id: str) -> int:
        try:
            with connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM rag_page_image_embeddings WHERE doc_id = %s",
                        (doc_id,),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
        except pg_errors.UndefinedTable:
            return 0

    def fetch_chunks_for_doc_page(self, doc_id: str, source_page: str) -> list[RetrievalHit]:
        """Чанки страницы без дополнительного векторного поиска (distance задаётся снаружи)."""
        sp = normalize_source_page(source_page) or source_page
        page_int: int | None = int(sp) if sp and str(sp).isdigit() else None
        with connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                if page_int is not None:
                    cur.execute(
                        """
                        SELECT doc_name, chunk_text, source_doc, source_chapter, source_page, chunk_index
                        FROM rag_chunks
                        WHERE doc_id = %s
                          AND (
                            source_page = %s
                            OR (
                              source_page IS NOT NULL
                              AND source_page ~ '^[0-9]+$'
                              AND source_page::bigint = %s
                            )
                          )
                        ORDER BY chunk_index ASC
                        """,
                        (doc_id, sp, page_int),
                    )
                else:
                    cur.execute(
                        """
                        SELECT doc_name, chunk_text, source_doc, source_chapter, source_page, chunk_index
                        FROM rag_chunks
                        WHERE doc_id = %s AND source_page = %s
                        ORDER BY chunk_index ASC
                        """,
                        (doc_id, sp),
                    )
                rows = cur.fetchall()

        results: list[RetrievalHit] = []
        for doc_name, chunk_text, source_doc, source_chapter, source_page_val, chunk_index in rows:
            source_ref = self._build_source_reference(
                source_doc, source_chapter, source_page_val, doc_id
            )
            results.append(
                RetrievalHit(
                    doc_name=str(doc_name),
                    content=str(chunk_text),
                    source_link=source_ref,
                    distance=0.0,
                    chunk_index=int(chunk_index) if chunk_index is not None else None,
                    fragment_kind="multimodal_page",
                )
            )
        return results

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievalHit]:
        with connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT doc_name, chunk_text, doc_id, source_doc, source_chapter, source_page,
                           chunk_index,
                           (embedding <=> %s::vector) AS distance
                    FROM rag_chunks
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (vector_literal(query_embedding), vector_literal(query_embedding), top_k),
                )
                rows = cur.fetchall()

        results: list[RetrievalHit] = []
        for doc_name, chunk_text, doc_id, source_doc, source_chapter, source_page, chunk_index, distance in rows:
            source_ref = self._build_source_reference(source_doc, source_chapter, source_page, doc_id)
            results.append(
                RetrievalHit(
                    doc_name=str(doc_name),
                    content=str(chunk_text),
                    source_link=source_ref,
                    distance=float(distance),
                    chunk_index=int(chunk_index) if chunk_index is not None else None,
                    fragment_kind="text",
                )
            )
        return results

    def count_rag_chunks_total(self) -> int:
        with connect(self.settings.postgres_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM rag_chunks")
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def count_page_image_embeddings_total(self) -> int:
        try:
            with connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM rag_page_image_embeddings")
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
        except pg_errors.UndefinedTable:
            return 0

    def count_illustration_chunks_total(self) -> int:
        try:
            with connect(self.settings.postgres_dsn) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM rag_illustration_chunks")
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
        except pg_errors.UndefinedTable:
            return 0

    @staticmethod
    def _build_source_reference(
        source_doc: str | None, source_chapter: str | None, source_page: str | None, doc_id: str
    ) -> str:
        parts: list[str] = [f"документ: {source_doc or doc_id}"]
        if source_chapter:
            parts.append(f"глава: {source_chapter}")
        if source_page:
            parts.append(f"страница: {source_page}")
        return "; ".join(parts)

