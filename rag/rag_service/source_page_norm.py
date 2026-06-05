"""Нормализация номера страницы для rag_chunks и rag_page_image_embeddings."""


def normalize_source_page(value: str | None) -> str | None:
    """Единый вид номера страницы в БД (без ведущих нулей)."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if v.isdigit():
        return str(int(v))
    return v
