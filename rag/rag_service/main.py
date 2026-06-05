import logging
import threading

from fastapi import FastAPI, HTTPException

from rag_service.config import load_settings
from rag_service.schemas import (
    ChatRequest,
    ChatResponse,
    ImageAnalyzeRequest,
    ImageAnalyzeResponse,
    ReindexResponse,
)
from rag_service.service import RAGService


settings = load_settings()
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__).info(
    "Embedding: EMBEDDING_MODEL=%s dim=%s | IMAGE_EMBEDDING_MODEL=%s dim=%s (поле model в JSON к /v1/embeddings)",
    settings.embedding_model,
    settings.embedding_dim,
    settings.image_embedding_model,
    settings.image_embedding_dim,
)
service = RAGService(settings)
app = FastAPI(title="PGVector RAG Service")
logger = logging.getLogger(__name__)

_indexing_lock = threading.Lock()
_indexing_run_lock = threading.Lock()
_indexing_started = False
_indexing_done = False
_indexing_error: str | None = None


def _rag_index_fields() -> tuple[bool, int, str | None]:
    """Статус индекса для API: готовность стартовой индексации и число строк в rag_chunks."""
    err = _indexing_error
    ready = _indexing_done and err is None
    try:
        n = service.repo.count_rag_chunks_total()
    except Exception as exc:  # noqa: BLE001
        return False, 0, f"{err or ''}; chunk_count_failed: {exc}".strip("; ")
    return ready, n, err


def _run_startup_indexing() -> None:
    global _indexing_done, _indexing_error
    try:
        with _indexing_run_lock:
            service.index_documents()
            _indexing_done = True
            logger.info("Startup indexing completed.")
    except HTTPException as exc:
        detail = exc.detail
        _indexing_error = detail if isinstance(detail, str) else str(detail)
        logger.error("Startup indexing failed: %s", _indexing_error)
    except Exception as exc:  # noqa: BLE001
        _indexing_error = str(exc)
        logger.exception("Startup indexing failed.")


@app.on_event("startup")
def startup() -> None:
    global _indexing_started
    with _indexing_lock:
        if _indexing_started:
            return
        _indexing_started = True
        threading.Thread(target=_run_startup_indexing, daemon=True).start()


@app.get("/health")
def health() -> dict[str, str]:
    status = "ok"
    if _indexing_error:
        status = "degraded"
    ready, chunk_n, _err = _rag_index_fields()
    if ready and chunk_n == 0:
        status = "degraded"
    try:
        page_img_n = service.repo.count_page_image_embeddings_total()
        ill_n = service.repo.count_illustration_chunks_total()
    except Exception:  # noqa: BLE001
        page_img_n = -1
        ill_n = -1
    return {
        "status": status,
        "indexing_started": str(_indexing_started).lower(),
        "indexing_done": str(_indexing_done).lower(),
        "index_ready": str(ready).lower(),
        "rag_chunk_count": str(chunk_n),
        "rag_page_image_embeddings_count": str(page_img_n),
        "rag_illustration_chunks_count": str(ill_n),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question не должен быть пустым")

    answer, context_docs = service.ask_text(question)
    idx_ready, idx_n, idx_err = _rag_index_fields()
    return ChatResponse(
        answer=answer,
        context_files=[h.doc_name for h in context_docs],
        matches=service.to_matches(context_docs),
        index_ready=idx_ready,
        index_chunk_count=idx_n,
        index_error=idx_err,
    )


@app.post("/analyze-image", response_model=ImageAnalyzeResponse)
def analyze_image(request: ImageAnalyzeRequest) -> ImageAnalyzeResponse:
    if not request.image_base64.strip():
        raise HTTPException(status_code=400, detail="image_base64 не должен быть пустым")
    try:
        image_base64 = service.normalize_image_base64(request.image_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="image_base64 имеет некорректный формат") from exc

    (
        image_description,
        context_docs,
        classification,
        no_sources_summary,
        report_docx_b64,
        report_name,
    ) = service.analyze_image(
        image_base64=image_base64,
        mime_type=request.mime_type,
        user_hint=request.user_hint,
        specialist_full_name=request.specialist_full_name,
        specialist_education=request.specialist_education,
        specialist_qualification=request.specialist_qualification,
        specialist_additional_training=request.specialist_additional_training,
        specialist_position=request.specialist_position,
        specialist_research_interests=request.specialist_research_interests,
        specialist_experience_years=request.specialist_experience_years,
        report_basis=request.report_basis,
    )
    idx_ready, idx_n, idx_err = _rag_index_fields()
    return ImageAnalyzeResponse(
        image_description=image_description,
        matches=service.to_matches(context_docs),
        classification=classification,
        no_sources_summary=no_sources_summary,
        report_docx_base64=report_docx_b64,
        report_file_name=report_name,
        index_ready=idx_ready,
        index_chunk_count=idx_n,
        index_error=idx_err,
    )


@app.post("/reindex", response_model=ReindexResponse)
def reindex() -> ReindexResponse:
    try:
        with _indexing_run_lock:
            service.index_documents(full_resync=True)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Reindex failed.")
        raise HTTPException(status_code=500, detail=f"Reindex failed: {exc!s}") from exc
    return ReindexResponse(
        status="ok",
        message="Full reindex completed (table resynced).",
    )

