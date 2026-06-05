import base64
import binascii
import datetime
import hashlib
import logging
import re

from rag_service.config import Settings, image_embedding_url_uses_text_embedding_fallback
from rag_service.ingestion import IngestDocument, chunk_document, read_documents, resolve_chapter_for_illustration_page
from rag_service.llm_client import LLMClient
from rag_service.page_image_index import try_index_page_images_for_document
from rag_service.report_docx import generate_conclusion_docx
from rag_service.report_models import (
    ConclusionReportData,
    MaterialsInfo,
    ReportMeta,
    ReportPhoto,
    SpecialistInfo,
)
from rag_service.repository import RAGRepository
from rag_service.retrieval_hit import RetrievalHit
from rag_service.schemas import RAGMatch
from rag_service.tattoo_image_prompts import USER_PHOTO_VL_SYSTEM_RULES_RU

logger = logging.getLogger(__name__)

_FRAGMENT_LABEL = {
    "text": "текст",
    "illustration": "иллюстрация",
    "multimodal_page": "страница по фото",
}


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = LLMClient(settings)
        self.repo = RAGRepository(settings)
        pdf_root = settings.rag_pdf_root
        logger.info(
            "RAG multimodal config: MULTIMODAL_ENABLED=%s IMAGE_EMBEDDING_API_URL=%s RAG_PDF_ROOT=%s "
            "RAG_IMAGE_TOP_K=%s",
            settings.multimodal_enabled,
            "(пусто)" if not settings.image_embedding_api_url.strip() else "задан",
            str(pdf_root) if pdf_root else "(не задан)",
            settings.rag_image_top_k,
        )
        if settings.multimodal_enabled and settings.image_embedding_api_url.strip():
            logger.info(
                "Multimodal: IMAGE_EMBEDDING_REQUEST_FORMAT=%s SKIP_LEGACY_EMBEDDINGS_PATH=%s "
                "(LM Studio: multimodal image embedding часто недоступен — см. README в .env.example).",
                settings.image_embedding_request_format,
                settings.image_embedding_skip_legacy_embeddings_path,
            )
            if image_embedding_url_uses_text_embedding_fallback():
                same_as_text = (
                    settings.image_embedding_model == settings.embedding_model
                    and settings.image_embedding_dim == settings.embedding_dim
                )
                if same_as_text:
                    logger.warning(
                        "IMAGE_EMBEDDING_API_URL не задан — для картинок используется тот же URL и та же модель, что "
                        "для текста (%s, dim=%s). Для rag_page_image_embeddings лучше отдельная VL-модель "
                        "(IMAGE_EMBEDDING_MODEL / IMAGE_EMBEDDING_DIM); размерность должна совпадать с vector в "
                        "Liquibase 002 и ответом API.",
                        settings.embedding_model,
                        settings.embedding_dim,
                    )
                else:
                    logger.info(
                        "Multimodal: тот же URL, что EMBEDDING_API_URL; для картинок — IMAGE_EMBEDDING_MODEL=%s "
                        "(dim=%s), для текста — %s (dim=%s).",
                        settings.image_embedding_model,
                        settings.image_embedding_dim,
                        settings.embedding_model,
                        settings.embedding_dim,
                    )
        elif settings.multimodal_enabled:
            logger.warning(
                "MULTIMODAL включён, но нет URL для эмбеддингов (ни IMAGE_EMBEDDING_API_URL, ни EMBEDDING_API_URL) — "
                "rag_page_image_embeddings не будет заполняться."
            )

    def index_documents(self, full_resync: bool = False) -> None:
        documents = read_documents(self.settings.docs_dir, self.settings.toc_dir)
        if full_resync:
            self.repo.clear_all_chunks()

        for document in documents:
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            if not full_resync and self.repo.document_already_indexed(document.doc_id, content_hash):
                self._catch_up_page_images_if_missing(document)
                continue

            if not full_resync:
                self.repo.delete_document_chunks(document.doc_id)
            chunks = chunk_document(
                document,
                self.settings.chunk_size,
                self.settings.chunk_overlap,
                min_merge_chars=self.settings.chunk_min_merge_chars,
            )
            for chunk_index, chunk in enumerate(chunks):
                embedding = self.llm.get_embedding(chunk.text)
                self.repo.insert_chunk(
                    doc_id=document.doc_id,
                    doc_name=document.doc_name,
                    source_doc=document.meta.source_doc,
                    source_chapter=chunk.chapter or document.meta.source_chapter,
                    source_page=chunk.page or document.meta.source_page,
                    content_hash=content_hash,
                    chunk_index=chunk_index,
                    chunk_text=chunk.text,
                    embedding=embedding,
                )
            for ill_index, (ill_page, ill_text) in enumerate(document.illustration_segments):
                ill_emb = self.llm.get_embedding(ill_text)
                ill_chapter = resolve_chapter_for_illustration_page(
                    document.doc_name,
                    str(ill_page),
                    document.meta.source_chapter,
                    self.settings.toc_dir,
                )
                self.repo.insert_illustration_chunk(
                    doc_id=document.doc_id,
                    doc_name=document.doc_name,
                    source_doc=document.meta.source_doc,
                    source_chapter=ill_chapter,
                    source_page=ill_page,
                    content_hash=content_hash,
                    chunk_index=ill_index,
                    chunk_text=ill_text,
                    embedding=ill_emb,
                )
            try:
                try_index_page_images_for_document(
                    self.settings, self.llm, self.repo, document, self.settings.docs_dir
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Индексация страниц для multimodal (%s): %s", document.doc_id, exc)

    def _catch_up_page_images_if_missing(self, document: IngestDocument) -> None:
        """Если чанки уже проиндексированы, но эмбеддингов страниц ещё нет — добиваем multimodal без смены content_hash."""
        if not self.settings.multimodal_enabled or not self.settings.image_embedding_api_url.strip():
            return
        if self.repo.count_page_image_embeddings_for_doc(document.doc_id) > 0:
            return
        try:
            try_index_page_images_for_document(
                self.settings, self.llm, self.repo, document, self.settings.docs_dir
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Multimodal catch-up для %s: %s", document.doc_id, exc)

    def _finalize_ranked(
        self,
        ranked: list[RetrievalHit],
        raw_matches: list[RetrievalHit],
        k: int,
        *,
        log_tag: str = "retrieve",
        multimodal_extra: str = "",
    ) -> list[RetrievalHit]:
        n_raw = len(raw_matches)
        n_ranked = len(ranked)
        if ranked:
            out = ranked[:k]
            logger.info(
                "RAG %s: raw=%s ranked_after_filter=%s returned=%s fallback=no%s",
                log_tag,
                n_raw,
                n_ranked,
                len(out),
                multimodal_extra,
            )
            return out
        if not raw_matches:
            logger.info(
                "RAG %s: raw=0 ranked=0 returned=0 fallback=n/a (no candidates)%s",
                log_tag,
                multimodal_extra,
            )
            return []
        if self.settings.rag_fallback_on_empty:
            fb_max = self.settings.rag_fallback_max_distance
            loose = [row for row in raw_matches if row.distance <= fb_max]
            if loose:
                out = sorted(loose, key=lambda row: row.distance)[:k]
                logger.warning(
                    "RAG %s: raw=%s ranked_after_filter=0 fallback=vector_threshold distance<=%s returned=%s%s",
                    log_tag,
                    n_raw,
                    fb_max,
                    len(out),
                    multimodal_extra,
                )
                return out
            out = sorted(raw_matches, key=lambda row: row.distance)[:k]
            logger.warning(
                "RAG %s: raw=%s ranked_after_filter=0 fallback=vector_no_threshold returned=%s%s",
                log_tag,
                n_raw,
                len(out),
                multimodal_extra,
            )
            return out
        logger.info(
            "RAG %s: raw=%s ranked=0 returned=0 fallback=disabled%s",
            log_tag,
            n_raw,
            multimodal_extra,
        )
        return []

    @staticmethod
    def _chunk_key_hit(hit: RetrievalHit) -> tuple[str, str, str]:
        return (hit.doc_name, hit.source_link, hit.content)

    def _merge_text_and_image_hits(
        self,
        text_hits: list[RetrievalHit],
        image_hits: list[RetrievalHit],
        w_img: float,
    ) -> list[RetrievalHit]:
        text_map = {self._chunk_key_hit(h): h.distance for h in text_hits}
        img_map = {self._chunk_key_hit(h): h.distance for h in image_hits}
        templates: dict[tuple[str, str, str], RetrievalHit] = {}
        for h in text_hits:
            templates[self._chunk_key_hit(h)] = h
        for h in image_hits:
            k = self._chunk_key_hit(h)
            if k not in templates:
                templates[k] = h
        merged: list[RetrievalHit] = []
        keys = set(text_map) | set(img_map)
        for key in keys:
            dt = text_map.get(key)
            di = img_map.get(key)
            name, source, content = key
            base = templates[key]
            if dt is not None and di is not None:
                final = w_img * di + (1.0 - w_img) * dt
            elif dt is not None:
                final = dt
            else:
                final = di if di is not None else 0.0
            merged.append(
                RetrievalHit(
                    doc_name=name,
                    content=content,
                    source_link=source,
                    distance=final,
                    chunk_index=base.chunk_index,
                    fragment_kind=base.fragment_kind,
                )
            )
        merged.sort(key=lambda h: h.distance)
        return merged

    @staticmethod
    def _merge_vector_search_results(
        primary: list[RetrievalHit],
        secondary: list[RetrievalHit],
        limit: int,
    ) -> list[RetrievalHit]:
        merged = primary + secondary
        merged.sort(key=lambda h: h.distance)
        return merged[:limit]

    @staticmethod
    def _merge_same_source_by_min_distance(
        primary: list[RetrievalHit],
        secondary: list[RetrievalHit],
    ) -> list[RetrievalHit]:
        """Для keywords-boost: два результата одного и того же чанка схлопываем по min distance."""
        key = lambda h: (h.doc_name, h.source_link, h.content)  # noqa: E731
        best: dict[tuple[str, str, str], RetrievalHit] = {}
        for h in primary + secondary:
            k = key(h)
            cur = best.get(k)
            if cur is None or h.distance < cur.distance:
                best[k] = h
        return sorted(best.values(), key=lambda h: h.distance)

    def _enforce_fragment_mix(self, ranked: list[RetrievalHit], k: int) -> list[RetrievalHit]:
        """Стабилизирует контекст: не даём иллюстрациям вытеснять основной текст.

        По умолчанию: минимум N текстовых чанков + максимум M иллюстрационных.
        Это помогает вытаскивать «кто набивает/значение/локализация», которые чаще в тексте, а не в описании картинок.
        """
        if not ranked or k <= 0:
            return []
        min_text = max(0, int(getattr(self.settings, "rag_min_text_hits", 0) or 0))
        max_ill = max(0, int(getattr(self.settings, "rag_max_illustration_hits", 0) or 0))
        min_text = min(min_text, k)

        text_hits = [h for h in ranked if (h.fragment_kind or "text") != "illustration"]
        ill_hits = [h for h in ranked if (h.fragment_kind or "") == "illustration"]

        selected: list[RetrievalHit] = []
        used: set[tuple[str, str, int, str]] = set()

        def _add(hit: RetrievalHit) -> None:
            ck = hit.chunk_index if hit.chunk_index is not None else -1
            key = (hit.doc_name, hit.source_link, ck, hit.fragment_kind)
            if key in used:
                return
            used.add(key)
            selected.append(hit)

        # 1) Сначала добираем минимум текста.
        for h in text_hits:
            if len(selected) >= min_text:
                break
            _add(h)

        # 2) Затем добавляем иллюстрации, но не больше лимита.
        ill_added = 0
        for h in ill_hits:
            if len(selected) >= k or ill_added >= max_ill:
                break
            _add(h)
            ill_added += 1

        # 3) Остаток — лучшими по рангу, независимо от типа.
        for h in ranked:
            if len(selected) >= k:
                break
            _add(h)

        return selected[:k]

    def retrieve_context(self, query: str, top_k: int | None = None) -> list[RetrievalHit]:
        k = top_k or self.settings.rag_top_k
        search_limit = min(120, max(50, k * 25))
        query_embedding = self.llm.get_embedding(query)
        raw_primary = self.repo.search(query_embedding, search_limit)
        ill_raw = self.repo.search_illustration_chunks(query_embedding, search_limit)
        raw_matches = self._merge_vector_search_results(raw_primary, ill_raw, search_limit)
        ranked = self._filter_and_rerank(query=query, raw_matches=raw_matches)
        ranked = self._enforce_fragment_mix(ranked, k)
        return self._finalize_ranked(ranked, raw_matches, k, log_tag="retrieve")

    def retrieve_context_multimodal(
        self,
        query: str,
        image_bytes: bytes | None,
        mime_type: str,
        top_k: int | None = None,
        keywords_query: str = "",
    ) -> list[RetrievalHit]:
        """Текстовый RAG + (опционально) поиск страниц по эмбеддингу изображения пользователя.

        Если задан keywords_query (короткие термины рубрик из VL [KEYWORDS]), выполняется
        второй векторный поиск по keywords-эмбеддингу; результаты мёрджатся по минимальной дистанции.
        Для повторного поиска для /analyze-image рекомендуется снижать порог lexical overlap
        через RERANK_MIN_LEXICAL_OVERLAP_IMAGE — VL-описания редко пересекаются лексически с корпусом.
        """
        k = top_k or self.settings.rag_top_k
        search_limit = min(120, max(50, k * 25))
        query_embedding = self.llm.get_embedding(query)
        raw_primary = self.repo.search(query_embedding, search_limit)
        ill_raw = self.repo.search_illustration_chunks(query_embedding, search_limit)

        kw = (keywords_query or "").strip()
        if kw and kw != query.strip():
            try:
                kw_embedding = self.llm.get_embedding(kw)
                kw_primary = self.repo.search(kw_embedding, search_limit)
                kw_ill = self.repo.search_illustration_chunks(kw_embedding, search_limit)
                logger.info(
                    "retrieve_mm: keywords_boost desc_primary=%s desc_ill=%s kw_primary=%s kw_ill=%s",
                    len(raw_primary),
                    len(ill_raw),
                    len(kw_primary),
                    len(kw_ill),
                )
                raw_primary = self._merge_same_source_by_min_distance(raw_primary, kw_primary)
                ill_raw = self._merge_same_source_by_min_distance(ill_raw, kw_ill)
            except Exception as exc:  # noqa: BLE001
                logger.warning("retrieve_mm: keywords embedding не удался (%s) — без boost", exc)

        raw_matches = self._merge_vector_search_results(raw_primary, ill_raw, search_limit)
        # Запрос для rerank — description + keywords (для lexical overlap), чтобы и термины рубрик считались.
        rerank_query = query if not kw else f"{query}\n{kw}"
        text_ranked = self._filter_and_rerank(
            query=rerank_query,
            raw_matches=raw_matches,
            min_overlap_override=self.settings.rerank_min_lexical_overlap_image,
        )
        text_ranked = self._enforce_fragment_mix(text_ranked, k)

        merged: list[RetrievalHit] = []
        mm_extra = ""
        use_image = (
            image_bytes
            and self.settings.multimodal_enabled
            and bool(self.settings.image_embedding_api_url.strip())
        )
        if use_image:
            try:
                img_emb = self.llm.get_image_embedding(image_bytes, mime_type)
                page_hits = self.repo.search_pages_by_image_embedding(
                    img_emb, top_k=min(80, search_limit)
                )
                img_max = self.settings.rag_image_max_distance
                page_hits = [h for h in page_hits if h[2] <= img_max]
                img_chunks: list[RetrievalHit] = []
                for doc_id, page, dist in page_hits:
                    for hit in self.repo.fetch_chunks_for_doc_page(doc_id, page):
                        img_chunks.append(
                            RetrievalHit(
                                doc_name=hit.doc_name,
                                content=hit.content,
                                source_link=hit.source_link,
                                distance=dist,
                                chunk_index=hit.chunk_index,
                                fragment_kind=hit.fragment_kind,
                            )
                        )
                    for hit in self.repo.fetch_illustration_chunks_for_doc_page(doc_id, page):
                        img_chunks.append(
                            RetrievalHit(
                                doc_name=hit.doc_name,
                                content=hit.content,
                                source_link=hit.source_link,
                                distance=dist,
                                chunk_index=hit.chunk_index,
                                fragment_kind=hit.fragment_kind,
                            )
                        )
                if page_hits and not img_chunks:
                    logger.warning(
                        "Multimodal: по image-вектору найдены страницы %s, но чанков rag_chunks для этих "
                        "doc_id+source_page нет (проверьте нормализацию страниц и переиндексацию).",
                        [(d, p, f"{di:.4f}") for d, p, di in page_hits[:5]],
                    )
                merged = self._merge_text_and_image_hits(
                    text_ranked, img_chunks, self.settings.multimodal_image_weight
                )
                mm_extra = f" multimodal_page_hits={len(page_hits)} multimodal_chunks={len(img_chunks)}"
                logger.info(
                    "RAG retrieve_mm: text_ranked=%s page_hits=%s img_chunks=%s merged_pre_final=%s",
                    len(text_ranked),
                    len(page_hits),
                    len(img_chunks),
                    len(merged),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Multimodal retrieval: используем только текст: %s", exc)

        if merged:
            return self._finalize_ranked(
                merged, raw_matches, k, log_tag="retrieve_mm", multimodal_extra=mm_extra
            )
        return self._finalize_ranked(
            text_ranked, raw_matches, k, log_tag="retrieve_mm", multimodal_extra=mm_extra
        )

    @staticmethod
    def to_matches(context_docs: list[RetrievalHit]) -> list[RAGMatch]:
        best: dict[tuple[str, str, int, str], RetrievalHit] = {}
        for item in context_docs:
            ck = item.chunk_index if item.chunk_index is not None else -1
            key = (item.doc_name, item.source_link, ck, item.fragment_kind)
            if key not in best or item.distance < best[key].distance:
                best[key] = item

        ordered = sorted(best.values(), key=lambda row: row.distance)
        return [
            RAGMatch(
                object_name=h.doc_name,
                description=h.content[:800],
                source_link=h.source_link,
                score=1.0 - h.distance,
                fragment_kind=h.fragment_kind,
                chunk_index=h.chunk_index,
            )
            for h in ordered
        ]

    def _format_hits_for_prompt(self, hits: list[RetrievalHit]) -> str:
        if not hits:
            return "Контекст не найден."
        ordered = sorted(hits, key=lambda h: h.distance)
        remaining = self.settings.rag_context_budget_chars
        per_max = self.settings.rag_prompt_max_chunk_chars
        blocks: list[str] = []
        for h in ordered:
            if remaining <= 0:
                break
            label = _FRAGMENT_LABEL.get(h.fragment_kind, h.fragment_kind)
            head = f"[{h.doc_name}] [{label}]"
            if h.chunk_index is not None:
                head += f" chunk={h.chunk_index}"
            max_this = min(len(h.content), per_max, remaining)
            if max_this <= 0:
                continue
            chunk = h.content[:max_this]
            blocks.append(f"{head}\n{chunk}")
            remaining -= max_this
        return "\n\n".join(blocks) if blocks else "Контекст не найден."

    def build_prompt(self, question: str, context_docs: list[RetrievalHit]) -> str:
        context_text = self._format_hits_for_prompt(context_docs)
        return (
            "Ты ассистент. Отвечай на русском языке.\n"
            "Обязательно используй RAG-контекст ниже как главный источник фактов.\n"
            "Если в контексте не хватает данных, честно скажи об этом.\n\n"
            f"RAG-контекст:\n{context_text}\n\n"
            f"Вопрос пользователя: {question}"
        )

    def ask_text(self, question: str) -> tuple[str, list[RetrievalHit]]:
        context_docs = self.retrieve_context(question)
        prompt = self.build_prompt(question, context_docs)
        answer = self.llm.chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return answer, context_docs

    def _build_image_classification_prompt(
        self,
        description_text: str,
        user_hint: str,
        context_docs: list[RetrievalHit],
    ) -> str:
        lines = [
            "Ты помощник по классификации татуировок и символики по справочным текстам.",
            "Опирайся ТОЛЬКО на фрагменты ниже. Не выдумывай факты, которых нет в текстах.",
            "Не давай юридической квалификации и не делай выводов о правонарушениях.",
            "Если в фрагментах описан тот же мотив (форма звезды, туз пик, типичная «воровская» иконография), что на фото,",
            "не считай данных «недостаточными» только из-за того, что в книге указана другая типичная зона тела",
            "(например в заголовке «подключичная звезда», а на фото — кисть или рука): классифицируй по смыслу и иконографии",
            "из фрагментов; при необходимости явно отдели «зона на фото» от «зона в справочнике».",
            "",
            "Описание с изображения:",
            description_text.strip() or "(нет)",
        ]
        if user_hint.strip():
            lines.extend(["", "Дополнительный запрос пользователя:", user_hint.strip()])
        lines.extend(["", "Фрагменты из базы знаний (единственный опорный контекст):"])
        ctx_block = self._format_hits_for_prompt(context_docs)
        lines.append(ctx_block)
        lines.extend(
            [
                "",
                "Задача: краткая финальная классификация для отчёта — рубрика/тип в терминологии источников,",
                "1–3 предложения обоснования с отсылкой к формулировкам из фрагментов.",
                "Если на фото в описании видны мотивы, типичные для уголовной иконографии (восьмиконечная или «воровская» звезда,",
                "туз/масть на пальцах, набор мелких знаков на кисти и т.п.), а во фрагментах есть хотя бы общие сведения об уголовных",
                "наколках, символике на руках или «наборе» знаков — формулируй классификацию в этом регистре",
                "(уголовная символика, типичные знаки среды), а не нейтральное «просто часть общего набора наколок без темы»,",
                "если тексты это позволяют. Нейтральную общую рубрику («татуировки на руке как часть набора наколок») используй только когда",
                "во фрагментах нет ни слова, ни косвенного контекста про уголовную символику, а есть лишь описание зоны тела или композиции.",
                "Если точного совпадения каждого мотива во фрагментах нет, но есть сведения о татуировках на руках, кистях, пальцах, запястьях —",
                "всё равно свяжи вывод с ближайшей рубрикой из текстов; оговори, что семантика отдельного знака в цитатах может не раскрываться.",
                "Не ограничивайся одной фразой «данных недостаточно», если есть хоть какие-то релевантные общие формулировки.",
                "Полный отказ («данных недостаточно для любой классификации») — только если во фрагментах нет ни общих сведений о подобной зоне тела,",
                "ни о наборе/композиции наколок, ни релевантной рубрики.",
            ]
        )
        return "\n".join(lines)

    def _classify_image_from_sources(
        self,
        description_text: str,
        user_hint: str,
        context_docs: list[RetrievalHit],
    ) -> str:
        if not context_docs:
            return ""
        prompt = self._build_image_classification_prompt(
            description_text, user_hint, context_docs
        )
        return self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            timeout=6000,
        ).strip()

    def analyze_image(
        self,
        image_base64: str,
        mime_type: str,
        user_hint: str,
        *,
        specialist_full_name: str = "",
        specialist_education: str = "",
        specialist_qualification: str = "",
        specialist_additional_training: str = "",
        specialist_position: str = "",
        specialist_research_interests: str = "",
        specialist_experience_years: str = "",
        report_basis: str = "",
    ) -> tuple[str, list[RetrievalHit], str, str, str, str]:
        user_text = USER_PHOTO_VL_SYSTEM_RULES_RU
        if user_hint.strip():
            user_text += f"\nДополнительный запрос пользователя: {user_hint.strip()}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                ],
            }
        ]
        image_description_raw = self.llm.chat_completion(
            messages=messages, model=self.settings.vision_model, temperature=0.15, timeout=9000
        )
        description_text, keywords_text = self._parse_vision_response(image_description_raw)

        # Пользователю отдаём очищенное от повторов описание, а не сырой ответ VL.
        if description_text.strip():
            image_description = description_text.strip()
        else:
            image_description = self._trim_repetitive_vision_description(
                image_description_raw.strip()
            )
        base_desc = (description_text or image_description or "").strip()
        # Основной (description-)запрос: описание + пользовательский hint (без keywords — они пойдут отдельным embedding).
        desc_only_q = self._merge_description_and_hint(base_desc, user_hint)
        retrieval_query = self._shorten_for_retrieval_query(desc_only_q)
        # Keywords-запрос: короткие термины рубрик (+ опц. звёздочное расширение) для второго векторного поиска.
        keywords_query = self._build_keywords_retrieval_query(
            keywords_text, base_desc, user_hint
        )
        if keywords_text.strip():
            logger.info(
                "analyze_image: KEYWORDS длина=%s keywords_query_len=%s (отдельный embedding запрос)",
                len(keywords_text.strip()),
                len(keywords_query.strip()),
            )

        try:
            raw_image = base64.b64decode(image_base64, validate=True)
        except (ValueError, binascii.Error):
            raw_image = b""

        context_docs = self.retrieve_context_multimodal(
            retrieval_query,
            raw_image if raw_image else None,
            mime_type,
            top_k=self.settings.rag_image_top_k,
            keywords_query=keywords_query,
        )
        classification = ""
        no_sources_summary = ""
        if context_docs:
            try:
                desc_for_class = description_text.strip() or image_description
                classification = self._classify_image_from_sources(
                    description_text=desc_for_class,
                    user_hint=user_hint,
                    context_docs=context_docs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Классификация по источникам не выполнена: %s", exc)
        elif self.settings.rag_image_no_match_llm_fallback and base_desc:
            try:
                no_sources_summary = self._unsourced_image_summary(
                    description_text.strip() or image_description,
                    user_hint,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Краткий ответ без источников не выполнен: %s", exc)

        report_b64 = ""
        report_name = ""
        try:
            raw_photo = raw_image if raw_image else b""
            docx_bytes, report_name = self._build_specialist_conclusion_docx(
                image_description=image_description,
                classification=classification,
                matches=context_docs,
                photo_bytes=raw_photo,
                photo_mime=mime_type,
                specialist_full_name=specialist_full_name,
                specialist_education=specialist_education,
                specialist_qualification=specialist_qualification,
                specialist_additional_training=specialist_additional_training,
                specialist_position=specialist_position,
                specialist_research_interests=specialist_research_interests,
                specialist_experience_years=specialist_experience_years,
                report_basis=report_basis,
            )
            if docx_bytes:
                report_b64 = base64.b64encode(docx_bytes).decode("ascii")
        except Exception as exc:  # noqa: BLE001
            logger.warning("DOCX report generation failed: %s", exc)
        return image_description, context_docs, classification, no_sources_summary, report_b64, report_name

    def _build_specialist_conclusion_docx(
        self,
        *,
        image_description: str,
        classification: str,
        matches: list[RetrievalHit],
        photo_bytes: bytes,
        photo_mime: str,
        specialist_full_name: str = "",
        specialist_education: str = "",
        specialist_qualification: str = "",
        specialist_additional_training: str = "",
        specialist_position: str = "",
        specialist_research_interests: str = "",
        specialist_experience_years: str = "",
        report_basis: str = "",
    ) -> tuple[bytes, str]:
        """Генерация DOCX «Заключение специалиста» в стиле примера 84_Kolevatova."""
        today = datetime.date.today()
        date_iso = today.strftime("%d.%m.%Y")
        number = "1"
        basis = report_basis.strip() or "запрос учреждения (данные письма/исходящий номер задаются в следующей итерации)"
        specialist = SpecialistInfo(
            full_name=specialist_full_name.strip() or "Некрасов Иван Сергеевич",
            education=specialist_education.strip() or "высшее (данные диплома уточняются)",
            qualification=specialist_qualification.strip() or "юрист, 030501 Юриспруденция (данные диплома уточняются)",
            additional_training=specialist_additional_training.strip()
            or (
                "«Криминологическая экспертиза материалов, связанных с занятием лицами высшего положения "
                "в преступной иерархии и противоправной деятельности запрещенного международного "
                "общественного движения «Арестантское уголовное единство» "
                "(чья деятельность запрещена на территории Российской Федерации)» "
                "(данные удостоверения уточняются)."
            ),
            position=specialist_position.strip()
            or (
                "Старший преподаватель кафедры оперативно-розыскной деятельности "
                "ФКОУ ВО Кузбасский институт ФСИН России, г. Новокузнецк."
            ),
            research_interests=specialist_research_interests.strip()
            or (
                "пенитенциарная преступность; криминальная субкультура; средства коммуникации осужденных; "
                "профилактика экстремизма и терроризма; информационное обеспечение раскрытия и расследования "
                "преступлений, совершаемых в исправительных учреждениях."
            ),
            experience_years=specialist_experience_years.strip() or "в уголовно-исполнительной системе – 15 лет.",
        )

        materials = MaterialsInfo(
            materials_text=(
                "Специалисту представлено сопроводительное письмо и приложенные к нему электронные материалы "
                "в графическом формате JPEG (.JPG) в виде фотографии."
            ),
            person_text="(сведения о лице указываются в следующей итерации по входным полям)",
        )

        question = (
            "Содержат ли представленные на исследование материалы атрибутику или символику, сходные до степени "
            "смешения с атрибутикой или символикой международного общественного движения "
            "«Арестантское уголовное единство» (А.У.Е.) (чья деятельность признана экстремистской и запрещена "
            "на территории Российской Федерации)?"
        )
        sources = self._build_report_sources(matches)
        methods_text = (
            "В качестве базового метода, объединяющего все остальные, применен системно-структурный, "
            "позволивший провести научный анализ материала исследования. Кроме того, использованы другие методы "
            "познания: исторический, моделирование, логический; обобщение и описание полученных данных, "
            "семантический и т. д."
        )

        research_paragraphs = self._build_report_research_paragraphs(
            image_description=image_description,
            classification=classification,
            context_docs=matches,
        )

        conclusion_text = self._build_report_conclusion_text(
            image_description=image_description,
            classification=classification,
            context_docs=matches,
        )
        note_text = (
            "Данная информация является мнением специалиста, при этом информируем, что сотрудники (работники) "
            "ФКОУ ВО Кузбасский институт ФСИН России не имеют возможности выступать в качестве экспертов и "
            "специалистов в уголовном судопроизводстве."
        )

        photos: list[ReportPhoto] = []
        if photo_bytes:
            photos.append(ReportPhoto(number=1, image_bytes=photo_bytes, mime_type=photo_mime))

        data = ConclusionReportData(
            meta=ReportMeta(number=number, date_iso=date_iso, basis=basis),
            specialist=specialist,
            materials=materials,
            question=question,
            sources=sources,
            methods_text=methods_text,
            research_paragraphs=research_paragraphs,
            conclusion_text=conclusion_text,
            note_text=note_text,
            photos=photos,
        )
        docx_bytes = generate_conclusion_docx(data)
        file_name = f"Заключение_специалиста_№{number}_от_{date_iso.replace('.', '-')}.docx"
        return docx_bytes, file_name

    @staticmethod
    def _fixed_report_sources() -> list[str]:
        """Источники, которые заказчик хочет видеть всегда (нормативная/методическая база)."""
        return [
            "Решение Верховного Суда Российской Федерации от 17 августа 2020 года о признании международного общественного движения «Арестантское уголовное единство» экстремистским и запрете его деятельности на территории Российской Федерации.",
            "Шиков А.А., Агарков А.В., Капустин К.В. Использование содействия специалистов в оперативно-розыскном мероприятии «исследование предметов и документов» по делам о преступной деятельности ячеек экстремистской организации «АУЕ» в местах лишения свободы: методические рекомендации. 26.09.2023.",
        ]

    def _build_report_sources(self, matches: list[RetrievalHit]) -> list[str]:
        """Комбинация фиксированных источников + источников по фактическим match'ам."""
        fixed = self._fixed_report_sources()
        from_hits = self._sources_list_from_matches(matches, max_items=10)
        # Дедуп по строке
        seen: set[str] = set()
        out: list[str] = []
        for s in fixed + from_hits:
            s2 = (s or "").strip()
            if not s2 or s2 in seen:
                continue
            seen.add(s2)
            out.append(s2)
        return out

    def _build_report_research_paragraphs(
        self,
        *,
        image_description: str,
        classification: str,
        context_docs: list[RetrievalHit],
    ) -> list[str]:
        """LLM-генерация абзацев «Исследование» в стиле примера (строго по источникам)."""
        if not context_docs:
            # Без источников не вызываем LLM: риск галлюцинаций и «официального» текста без опоры.
            base = image_description.strip() or "(описание отсутствует)"
            return [
                f"На представленной фотографии констатируется: {base} (См. фото № 1).",
                "Контекст по справочнику не найден; выводы о семантике и принадлежности символики не делаются без источников.",
            ]
        ctx = self._format_hits_for_prompt(context_docs)
        lines = [
            "Сгенерируй официальный, содержательный раздел «Исследование» для заключения специалиста.",
            "Требования: меньше общих фраз, больше конкретики по рисункам и признакам (что именно изображено, где, как выполнено).",
            "Структура (в этом порядке, отдельными абзацами):",
            "1) Констатация + локализация: зона тела (лево/право), размер/расположение, перечисли 2–6 ключевых элементов рисунка.",
            "2) Техника исполнения: контур/заливка, штриховка, симметрия/парность, цвет (ч/б), надписи (если есть) — 2–5 признаков.",
            "3) Семантика мотивов: по КАЖДОМУ ключевому элементу дай короткую интерпретацию ТОЛЬКО если это подтверждено контекстом.",
            "4) Кто обычно набивает/кому присваивается и значение расположения (кисть/плечо/грудь/пальцы и т.п.) — только если это есть в источниках; если нет — прямо напиши, что в предоставленных источниках это не описано.",
            "5) Связка с решением ВС: упомяни решение и описанную там символику только как справочный фрагмент, без категоричных выводов сверх контекста.",
            "Стиль: официальный, без «может ассоциироваться», без повторов «типичные знаки среды». Не делай списков и заголовков.",
            "Опирайся ТОЛЬКО на контекст ниже. Если чего-то нет в контексте — формулируй осторожно или пропусти.",
            "Не давай юридической квалификации. Не упоминай, что ты ИИ.",
            "",
            "Описание изображения:",
            image_description.strip() or "(нет)",
            "",
            "Классификация по источникам (если есть):",
            classification.strip() or "(нет)",
            "",
            "Контекст (цитаты из базы знаний):",
            ctx,
            "",
            "Верни 5–9 абзацев обычным текстом (каждый 1–3 предложения), без списков и без заголовков.",
        ]
        prompt = "\n".join(lines)
        text = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            timeout=6000,
        )
        paras = [p.strip() for p in re.split(r"\n\s*\n+", (text or "").strip()) if p.strip()]
        if not paras:
            paras = [f"На представленной фотографии констатируется: {image_description.strip()} (См. фото № 1)."]
        return paras

    @staticmethod
    def _prettify_citation_title(raw: str) -> str:
        """Показываем «название работы» (поле из «документ:») без служебного расширения и с читаемыми пробелами."""
        s = (raw or "").strip()
        for suf in (".pdf", ".txt", ".PDF", ".TXT"):
            if s.endswith(suf):
                s = s[: -len(suf)]
                break
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    @staticmethod
    def _citation_title_from_source_link(source_link: str, doc_name: str) -> str:
        m = re.search(r"документ:\s*([^;]+)", source_link, flags=re.IGNORECASE)
        if m:
            return RAGService._prettify_citation_title(m.group(1))
        if (doc_name or "").strip():
            base = doc_name.rsplit(".", 1)[0] if "." in doc_name else doc_name
            return RAGService._prettify_citation_title(base)
        return doc_name

    @staticmethod
    def _source_reference_tail(source_link: str) -> str:
        """Всё после «документ: …» (глава, страница, блок); если только документ — хвост пустой."""
        if not (source_link or "").strip():
            return ""
        s = source_link.strip()
        m = re.search(r"документ:\s*[^;]+", s, flags=re.IGNORECASE)
        if not m:
            return s
        rest = s[m.end() :].lstrip()
        if rest.startswith(";"):
            rest = rest[1:].lstrip()
        return rest

    @staticmethod
    def _format_match_source_line(h: RetrievalHit) -> str:
        link = (h.source_link or "").strip()
        title = RAGService._citation_title_from_source_link(link, h.doc_name)
        tail = RAGService._source_reference_tail(link)
        if tail:
            return f"{title} — {tail}"
        return title

    @staticmethod
    def _sources_list_from_matches(matches: list[RetrievalHit], *, max_items: int = 10) -> list[str]:
        """Список источников по match'ам: в начале — название работы из «документ:», а не имя .txt-файла чанка."""
        if not matches:
            return []
        seen: set[str] = set()
        out: list[str] = []
        for h in sorted(matches, key=lambda x: x.distance):
            line = RAGService._format_match_source_line(h)
            if not line or line in seen:
                continue
            seen.add(line)
            out.append(line)
            if len(out) >= max_items:
                break
        return out

    def _build_report_conclusion_text(
        self,
        *,
        image_description: str,
        classification: str,
        context_docs: list[RetrievalHit],
    ) -> str:
        """Формулирует «Вывод» в стиле примера: АУЕ (сходно до степени смешения) + что значит/кто набивает."""
        if not context_docs:
            return (
                "Представленный на исследование материал не позволяет сделать однозначный вывод по символике без "
                "подтверждающих источников из справочника."
            )
        ctx = self._format_hits_for_prompt(context_docs)
        lines = [
            "Составь раздел «Вывод» для заключения специалиста в официальном стиле, максимально близко к образцу.",
            "Нужно в одном выводе отразить ДВА пункта:",
            "1) Формулировку как в образце: «Представленный на исследование материал содержит атрибутику или символику, сходные до степени смешения … АУЕ …».",
            "2) Конкретно (3–6 предложений) указать семантику: что означает каждый ключевой мотив, кому/кем обычно наносится, значение расположения — только если это есть в источниках.",
            "Если в источниках нет сведений про «кто набивает» или «значение расположения» — отдельным предложением прямо напиши, что в предоставленных источниках это не описано.",
            "Не добавляй фактов вне контекста. Не упоминай, что ты ИИ.",
            "",
            "Описание изображения:",
            image_description.strip() or "(нет)",
            "",
            "Черновик классификации (может помогать, но не источник):",
            classification.strip() or "(нет)",
            "",
            "Контекст (цитаты из базы знаний):",
            ctx,
            "",
            "Ограничение: избегай общих слов («типично», «может», «часто») без привязки к видимым элементам на фото.",
        ]
        prompt = "\n".join(lines)
        out = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            timeout=6000,
        ).strip()
        return out or "Вывод: (не сформирован)"

    @staticmethod
    def _merge_retrieval_query_parts(description: str, keywords: str, user_hint: str) -> str:
        parts: list[str] = []
        if description.strip():
            parts.append(description.strip())
        if keywords.strip():
            parts.append(keywords.strip())
        if user_hint.strip():
            parts.append(user_hint.strip())
        return "\n".join(parts)

    @staticmethod
    def _merge_description_and_hint(description: str, user_hint: str) -> str:
        parts: list[str] = []
        if description.strip():
            parts.append(description.strip())
        if user_hint.strip():
            parts.append(user_hint.strip())
        return "\n".join(parts)

    def _build_keywords_retrieval_query(
        self, keywords: str, description: str, user_hint: str
    ) -> str:
        """Короткий запрос на термины рубрик для отдельного embedding (без длинного описания)."""
        base = keywords.strip()
        extra = (self.settings.rag_image_query_expansion_stars_ru or "").strip()
        if extra and self._text_hints_star_motif_or_ace_ru(description, keywords, user_hint):
            base = f"{base}, {extra}".strip(", ").strip() if base else extra
        hint = user_hint.strip()
        if hint and len(hint) <= 200:
            base = f"{base}\n{hint}".strip() if base else hint
        return base.strip()

    @staticmethod
    def _text_hints_star_motif_or_ace_ru(*parts: str) -> bool:
        """Эвристика: восьмиконечная/воровская звезда, туз пик — для доп. лексики в запросе к RAG."""
        raw = "\n".join(p for p in parts if p and p.strip()).lower()
        if not raw.strip():
            return False
        compact = re.sub(r"\s+", "", raw)
        if "туз" in raw and ("пик" in raw or "пиков" in raw):
            return True
        if "восьмиконечн" in compact or "8-конечн" in compact or "8конечн" in compact:
            return True
        if "восьми" in raw and "звезд" in raw:
            return True
        if "звезд" in raw and ("конечн" in raw or "луч" in raw or "лучей" in raw):
            return True
        if "розаветра" in compact or "розыветра" in compact:
            return True
        if "воровск" in raw and "звезд" in raw:
            return True
        return False

    def _unsourced_image_summary(self, description: str, user_hint: str) -> str:
        lines = [
            "Ты помощник по татуировкам. Дай краткий ответ пользователю (3–6 предложений) только на основе текста ниже.",
            "В первой строке напиши ровно: «Без подтверждения из справочника:»",
            "Не добавляй факты, которых нет в описании. Не давай юридической квалификации.",
            "",
            "Описание с фото:",
            description.strip() or "(пусто)",
        ]
        if user_hint.strip():
            lines.extend(["", "Подсказка пользователя:", user_hint.strip()])
        prompt = "\n".join(lines)
        return self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=1200,
        ).strip()

    @staticmethod
    def _shorten_for_retrieval_query(text: str, max_chars: int = 2200) -> str:
        """Укорачивает текст запроса к RAG, сохраняя начало (обычно суть сюжета тату)."""
        text = text.strip()
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars]
        if " " in cut:
            return cut.rsplit(" ", 1)[0].strip()
        return cut.strip()

    @staticmethod
    def normalize_image_base64(raw: str) -> str:
        cleaned = "".join(raw.split())
        base64.b64decode(cleaned, validate=True)
        return cleaned

    @staticmethod
    def _trim_repetitive_vision_description(text: str, *, max_chars: int = 5000) -> str:
        """Обрезка зацикливания VL (одни и те же предложения сотни раз) и верхняя граница длины."""
        t = text.strip()
        if not t:
            return t
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
        if len(sentences) < 6:
            out = t[:max_chars] if len(t) > max_chars else t
            return out
        # Период 2: A B A B A B… (как в логах «на груди череп / на спине череп»)
        for i in range(len(sentences) - 5):
            a0, b0 = sentences[i], sentences[i + 1]
            if (
                sentences[i + 2] == a0
                and sentences[i + 3] == b0
                and sentences[i + 4] == a0
                and sentences[i + 5] == b0
            ):
                trimmed = " ".join(sentences[: i + 2])
                note = "\n\n(Повторяющийся фрагмент ответа модели удалён.)"
                return (trimmed + note)[: max_chars + len(note)]
        # Период 1: одно и то же предложение подряд — не более двух подряд (AAA… → AA)
        out_s: list[str] = []
        for s in sentences:
            if len(out_s) >= 2 and out_s[-1] == s and out_s[-2] == s:
                continue
            out_s.append(s)
        joined = " ".join(out_s)
        if len(joined) > max_chars:
            joined = joined[:max_chars].rsplit(" ", 1)[0] + "…"
        return joined

    @staticmethod
    def _parse_vision_response(raw: str) -> tuple[str, str]:
        desc_m = re.search(
            r"\[DESCRIPTION\]\s*(.*?)(?=\[KEYWORDS\]|\[OCR_TEXT\]|\Z)",
            raw,
            flags=re.IGNORECASE | re.DOTALL,
        )
        key_m = re.search(r"\[KEYWORDS\]\s*(.*)", raw, flags=re.IGNORECASE | re.DOTALL)
        description = desc_m.group(1).strip() if desc_m else raw.strip()
        description = RAGService._trim_repetitive_vision_description(description)
        keywords = ""
        if key_m:
            rest = key_m.group(1)
            cut = re.search(r"\n\s*\[[^\]]+\]", rest)
            keywords = (rest[: cut.start()] if cut else rest).strip()
        keywords = re.sub(r"\s+", " ", keywords).strip()
        return description, keywords

    def _filter_and_rerank(
        self,
        query: str,
        raw_matches: list[RetrievalHit],
        *,
        min_overlap_override: float | None = None,
    ) -> list[RetrievalHit]:
        query_tokens = self._tokenize(query)
        rescored: list[tuple[RetrievalHit, float]] = []
        min_overlap = (
            min_overlap_override
            if min_overlap_override is not None
            else self.settings.rerank_min_lexical_overlap
        )

        for hit in raw_matches:
            if hit.distance > self.settings.rag_max_distance:
                continue

            lexical_overlap = self._lexical_overlap(query_tokens, hit.content)
            if lexical_overlap < min_overlap:
                continue

            vector_score = 1.0 - min(max(hit.distance, 0.0), 1.0)
            combined_score = (
                self.settings.rerank_vector_weight * vector_score
                + self.settings.rerank_lexical_weight * lexical_overlap
            )
            rescored.append((hit, combined_score))

        rescored.sort(key=lambda row: row[1], reverse=True)
        return [hit for hit, _ in rescored]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-zА-Яа-я0-9_]+", text.lower()) if len(token) > 2}

    @staticmethod
    def _lexical_overlap(query_tokens: set[str], content: str) -> float:
        if not query_tokens:
            return 0.0
        content_tokens = {token for token in re.findall(r"[A-Za-zА-Яа-я0-9_]+", content.lower()) if len(token) > 2}
        if not content_tokens:
            return 0.0
        intersection = query_tokens.intersection(content_tokens)
        if not intersection:
            return 0.0
        denom = min(len(query_tokens), 48)
        return len(intersection) / denom
