"""Очистка выхода VL для prepare_ocr_docs; склейка текстового ответа и блока [ИЛЛЮСТРАЦИИ]."""

from __future__ import annotations

import re

from rag_service.tattoo_image_prompts import (
    VL_PAGE_EMPTY_TEXT_MARKER,
    VL_PAGE_NO_ILLUSTRATIONS_MARKER,
)


def clean_page_text_after_vl(text: str) -> str:
    """Light cleanup after OCR/VL; keeps letters/digits/Cyrillic, drops obvious noise."""
    cleaned = text.replace("\x0c", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(
        r"[^\w\s.,:;!?()\[\]#*•\"'«»\-\u0400-\u04FF]+",
        " ",
        cleaned,
        flags=re.UNICODE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def finalize_llm_page_text(raw: str) -> str:
    """Cleanup for vision-LLM output (do not apply OCR-specific char maps)."""
    t = raw.strip()
    if t.upper() in {"ПУСТО", "ПУСТО.", "EMPTY", "NONE", "N/A"}:
        return ""
    return clean_page_text_after_vl(t)


def merge_vl_prepare_passes(text_pass_raw: str, illustrations_pass_raw: str) -> str:
    """Склеивает текст полосы и блок [ИЛЛЮСТРАЦИИ] из двух VL-запросов."""
    tp = text_pass_raw.strip()
    if tp.upper().replace(".", "") == VL_PAGE_EMPTY_TEXT_MARKER.upper():
        body = ""
    else:
        body = finalize_llm_page_text(text_pass_raw)

    ip = illustrations_pass_raw.strip()
    if ip.upper().replace(".", "") == VL_PAGE_NO_ILLUSTRATIONS_MARKER.upper():
        ill = ""
    else:
        ill = finalize_llm_page_text(illustrations_pass_raw)

    if body and ill:
        return f"{body}\n\n{ill}"
    return body or ill
