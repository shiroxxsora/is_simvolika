import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    docs_dir: Path
    toc_dir: Path
    postgres_dsn: str
    llm_api_url: str
    llm_model: str
    vision_model: str
    llm_api_key: str
    embedding_api_url: str
    embedding_model: str
    embedding_dim: int
    embedding_timeout_sec: int
    image_embedding_api_url: str
    image_embedding_model: str
    image_embedding_dim: int
    image_embedding_timeout_sec: int
    image_embedding_request_format: str
    image_embedding_media_marker: str
    image_embedding_skip_legacy_embeddings_path: bool
    multimodal_enabled: bool
    multimodal_image_weight: float
    rag_pdf_root: Path | None
    multimodal_max_pages_per_doc: int
    rag_top_k: int
    rag_image_top_k: int
    rag_image_query_expansion_stars_ru: str
    rag_max_distance: float
    rag_image_max_distance: float
    rerank_vector_weight: float
    rerank_lexical_weight: float
    rerank_min_lexical_overlap: float
    rerank_min_lexical_overlap_image: float
    rag_fallback_on_empty: bool
    rag_fallback_max_distance: float
    chunk_size: int
    chunk_overlap: int
    chunk_min_merge_chars: int
    rag_context_budget_chars: int
    rag_prompt_max_chunk_chars: int
    rag_image_no_match_llm_fallback: bool
    rag_min_text_hits: int
    rag_max_illustration_hits: int


def _default_rag_pdf_root(docs_dir: Path) -> Path | None:
    """Если RAG_PDF_ROOT не задан — пробуем соседний каталог docs (как у prepare --raw-dir)."""
    cand = docs_dir.parent / "docs"
    try:
        return cand.resolve() if cand.is_dir() else None
    except OSError:
        return None


def load_settings() -> Settings:
    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    docs_dir = Path(os.getenv("RAG_DOCS_DIR", "/app/docs_prepared"))
    embedding_url = os.getenv("EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings").strip()
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1536"))
    image_emb_url = os.getenv("IMAGE_EMBEDDING_API_URL", "").strip() or embedding_url
    explicit_image_api = bool(os.getenv("IMAGE_EMBEDDING_API_URL", "").strip())
    img_model_env = os.getenv("IMAGE_EMBEDDING_MODEL", "").strip()
    img_dim_env = os.getenv("IMAGE_EMBEDDING_DIM", "").strip()
    if explicit_image_api:
        image_embedding_model = img_model_env or "clip"
        image_embedding_dim = int(img_dim_env) if img_dim_env else 512
    else:
        # Один URL с EMBEDDING_API_URL: можно задать только IMAGE_EMBEDDING_MODEL (VL-эмбеддер) и IMAGE_EMBEDDING_DIM,
        # не меняя EMBEDDING_MODEL для текстовых чанков.
        image_embedding_model = img_model_env or embedding_model
        image_embedding_dim = int(img_dim_env) if img_dim_env else embedding_dim
    rag_pdf_env = os.getenv("RAG_PDF_ROOT", "").strip()
    rag_pdf_root = Path(rag_pdf_env).resolve() if rag_pdf_env else _default_rag_pdf_root(docs_dir)

    rag_top_k = int(os.getenv("RAG_TOP_K", "3"))
    rag_image_top_k_env = os.getenv("RAG_IMAGE_TOP_K", "").strip()
    # Для /analyze-image по умолчанию не меньше 5 чанков, чтобы в классификацию попадали разные рубрики (звёзды vs «рука»).
    rag_image_top_k = int(rag_image_top_k_env) if rag_image_top_k_env else max(rag_top_k, 5)
    stars_exp = os.getenv(
        "RAG_IMAGE_QUERY_EXPANSION_STARS_RU",
        "подключичная звезда, вор в законе, воровская звезда, восьмиконечная звезда, звёзды воров в законе, "
        "воровской авторитет, АУЕ",
    ).strip()
    # По умолчанию v1_mtmd: сначала OpenAI /v1/embeddings с multimodal-объектом (llama-server).
    # server_content: только POST …/embeddings (полный llama.cpp; в LM Studio путь часто не реализован — см. лог Unexpected endpoint).
    img_req_fmt = os.getenv("IMAGE_EMBEDDING_REQUEST_FORMAT", "v1_mtmd").strip().lower()
    if img_req_fmt not in {"v1_mtmd", "openai_data_url", "server_content"}:
        img_req_fmt = "v1_mtmd"
    img_media_marker = os.getenv("IMAGE_EMBEDDING_MEDIA_MARKER", "<__media__>").strip() or "<__media__>"
    skip_legacy = os.getenv("IMAGE_EMBEDDING_SKIP_LEGACY_EMBEDDINGS_PATH", "").lower() in {"1", "true", "yes", "on"}

    return Settings(
        docs_dir=docs_dir,
        toc_dir=Path(os.getenv("RAG_TOC_DIR", "/app/docs_toc")),
        postgres_dsn=os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@postgres:5432/rag"),
        llm_api_url=os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
        llm_model=llm_model,
        vision_model=os.getenv("VISION_MODEL", llm_model),
        llm_api_key=os.getenv("LLM_API_KEY", ""),
        embedding_api_url=embedding_url,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_timeout_sec=int(os.getenv("EMBEDDING_TIMEOUT_SEC", "1800")),
        image_embedding_api_url=image_emb_url,
        image_embedding_model=image_embedding_model,
        image_embedding_dim=image_embedding_dim,
        image_embedding_timeout_sec=int(os.getenv("IMAGE_EMBEDDING_TIMEOUT_SEC", "2400")),
        image_embedding_request_format=img_req_fmt,
        image_embedding_media_marker=img_media_marker,
        image_embedding_skip_legacy_embeddings_path=skip_legacy,
        multimodal_enabled=os.getenv("MULTIMODAL_ENABLED", "true").lower() in {"1", "true", "yes", "on"},
        multimodal_image_weight=float(os.getenv("MULTIMODAL_IMAGE_WEIGHT", "0.45")),
        rag_pdf_root=rag_pdf_root,
        multimodal_max_pages_per_doc=int(os.getenv("MULTIMODAL_MAX_PAGES_PER_DOC", "500")),
        rag_top_k=rag_top_k,
        rag_image_top_k=rag_image_top_k,
        rag_image_query_expansion_stars_ru=stars_exp,
        rag_max_distance=float(os.getenv("RAG_MAX_DISTANCE", "0.38")),
        rag_image_max_distance=float(os.getenv("RAG_IMAGE_MAX_DISTANCE", "0.55")),
        rerank_vector_weight=float(os.getenv("RERANK_VECTOR_WEIGHT", "0.75")),
        rerank_lexical_weight=float(os.getenv("RERANK_LEXICAL_WEIGHT", "0.25")),
        rerank_min_lexical_overlap=float(os.getenv("RERANK_MIN_LEXICAL_OVERLAP", "0.015")),
        rerank_min_lexical_overlap_image=float(
            os.getenv("RERANK_MIN_LEXICAL_OVERLAP_IMAGE", "0.0")
        ),
        rag_fallback_on_empty=os.getenv("RAG_FALLBACK_ON_EMPTY", "true").lower() in {"1", "true", "yes", "on"},
        rag_fallback_max_distance=float(os.getenv("RAG_FALLBACK_MAX_DISTANCE", "0.52")),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "2000")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "320")),
        chunk_min_merge_chars=int(os.getenv("RAG_CHUNK_MIN_MERGE_CHARS", "0")),
        rag_context_budget_chars=int(os.getenv("RAG_CONTEXT_BUDGET_CHARS", "12000")),
        rag_prompt_max_chunk_chars=int(os.getenv("RAG_PROMPT_MAX_CHUNK_CHARS", "2500")),
        rag_image_no_match_llm_fallback=os.getenv("RAG_IMAGE_NO_MATCH_LLM_FALLBACK", "true").lower()
        in {"1", "true", "yes", "on"},
        rag_min_text_hits=int(os.getenv("RAG_MIN_TEXT_HITS", "2")),
        rag_max_illustration_hits=int(os.getenv("RAG_MAX_ILLUSTRATION_HITS", "1")),
    )


def image_embedding_url_uses_text_embedding_fallback() -> bool:
    """True, если IMAGE_EMBEDDING_API_URL не задавали явно (взяли EMBEDDING_API_URL)."""
    return not os.getenv("IMAGE_EMBEDDING_API_URL", "").strip()

