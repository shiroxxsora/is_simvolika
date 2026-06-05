"""Индексация эмбеддингов страниц PDF для multimodal-поиска (изображение страницы → вектор)."""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

from rag_service.config import Settings
from rag_service.ingestion import IngestDocument
from rag_service.llm_client import LLMClient
from rag_service.repository import RAGRepository

logger = logging.getLogger(__name__)


def _numeric_pages_for_document(document: IngestDocument, max_pages: int) -> list[int]:
    pages: set[int] = set()
    for unit in document.units:
        if unit.page and unit.page.isdigit():
            pages.add(int(unit.page))
    if not pages and document.content:
        for m in re.finditer(r"\[PAGE\s+(\d+)\]", document.content, flags=re.IGNORECASE):
            pages.add(int(m.group(1)))
    ordered = sorted(pages)
    if len(ordered) > max_pages:
        logger.warning(
            "Документ %s: страниц для индекса %s, ограничение MULTIMODAL_MAX_PAGES_PER_DOC=%s",
            document.doc_id,
            len(ordered),
            max_pages,
        )
        ordered = ordered[:max_pages]
    return ordered


def resolve_pdf_path_for_document(
    document: IngestDocument,
    docs_dir: Path,
    pdf_root: Path | None,
) -> Path | None:
    """Находит PDF для подготовленного .txt или для прямого .pdf в каталоге документов."""
    direct = docs_dir / document.doc_id
    if direct.suffix.lower() == ".pdf" and direct.is_file():
        return direct

    meta = (document.meta.source_doc or "").strip()
    if not meta.lower().endswith(".pdf"):
        return None

    rel_parent = Path(document.doc_id).parent
    name = Path(meta).name
    search_bases: list[Path] = []
    if pdf_root is not None:
        search_bases.append(pdf_root)
    search_bases.append(docs_dir)
    sibling_docs = docs_dir.parent / "docs"
    if sibling_docs.is_dir():
        search_bases.append(sibling_docs)

    for base in search_bases:
        for candidate in (base / meta, base / rel_parent / name, base / name):
            try:
                if candidate.is_file():
                    return candidate
            except OSError:
                continue

    if pdf_root is not None:
        for found in pdf_root.rglob(name):
            if found.is_file():
                return found
    return None


def _resize_png_max_side(png_bytes: bytes, max_side: int = 768) -> bytes:
    with Image.open(BytesIO(png_bytes)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        w, h = im.size
        if max(w, h) <= max_side:
            out = BytesIO()
            im.save(out, format="PNG")
            return out.getvalue()
        scale = max_side / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)
        out = BytesIO()
        im.save(out, format="PNG")
        return out.getvalue()


def render_pdf_page_png(pdf_path: Path, page_1based: int, zoom: float = 2.0) -> bytes:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_1based - 1]
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def try_index_page_images_for_document(
    settings: Settings,
    llm: LLMClient,
    repo: RAGRepository,
    document: IngestDocument,
    docs_dir: Path,
) -> None:
    if not settings.multimodal_enabled:
        logger.info(
            "Multimodal: пропуск индекса страниц (MULTIMODAL_ENABLED=false), doc_id=%s",
            document.doc_id,
        )
        return
    if not settings.image_embedding_api_url.strip():
        logger.info(
            "Multimodal: пропуск индекса страниц (пустой IMAGE_EMBEDDING_API_URL), doc_id=%s",
            document.doc_id,
        )
        return

    pdf_path = resolve_pdf_path_for_document(document, docs_dir, settings.rag_pdf_root)
    if pdf_path is None:
        root_hint = f" RAG_PDF_ROOT={settings.rag_pdf_root}" if settings.rag_pdf_root else " RAG_PDF_ROOT не задан"
        logger.info(
            "Multimodal: PDF не найден для doc_id=%s (source_doc=%s);%s — страницы не индексируются",
            document.doc_id,
            document.meta.source_doc,
            root_hint,
        )
        return

    pages = _numeric_pages_for_document(document, settings.multimodal_max_pages_per_doc)
    if not pages:
        logger.info(
            "Multimodal: нет страниц с цифровым unit.page для doc_id=%s — rag_page_image_embeddings не заполняется "
            "(проверьте маркеры [PAGE N] в подготовленном TXT)",
            document.doc_id,
        )
        return

    logger.info(
        "Multimodal: старт индекса страниц doc_id=%s pdf=%s страниц_к_обработке=%s "
        "IMAGE_EMBEDDING_MODEL=%s IMAGE_EMBEDDING_DIM=%s",
        document.doc_id,
        pdf_path,
        len(pages),
        settings.image_embedding_model,
        settings.image_embedding_dim,
    )

    ok = 0
    first_error: str | None = None
    for page_num in pages:
        try:
            png = render_pdf_page_png(pdf_path, page_num)
            png = _resize_png_max_side(png)
            emb = llm.get_image_embedding(png, mime_type="image/png")
            repo.upsert_page_image_embedding(document.doc_id, str(page_num), emb)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            if first_error is None:
                first_error = str(exc)
            logger.warning(
                "Multimodal: не удалось проиндексировать страницу %s документа %s: %s",
                page_num,
                document.doc_id,
                exc,
            )

    logger.info(
        "Multimodal: завершено doc_id=%s upsert_страниц=%s/%s pdf=%s%s",
        document.doc_id,
        ok,
        len(pages),
        pdf_path,
        f" первая_ошибка={first_error!r}" if first_error else "",
    )
