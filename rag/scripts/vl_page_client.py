"""OpenAI-compatible vision API for per-page PDF text extraction (LM Studio, etc.)."""

from __future__ import annotations

import base64
import json
import logging
import socket
import urllib.error
import urllib.request

from rag_service.llm_output_cleanup import merge_vl_prepare_passes
from rag_service.tattoo_image_prompts import VL_PAGE_ILLUSTRATIONS_PASS_RU, VL_PAGE_TEXT_PASS_RU

logger = logging.getLogger(__name__)


def strip_model_artifacts(text: str) -> str:
    return text.strip()


def _vl_chat_completion(
    *,
    image_png_bytes: bytes,
    prompt: str,
    api_url: str,
    model: str,
    api_key: str,
    timeout: int,
    max_tokens: int = 8192,
) -> str:
    b64 = base64.b64encode(image_png_bytes).decode("ascii")
    payload: dict = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url=api_url, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return strip_model_artifacts(content)
            return strip_model_artifacts(str(content))
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"VL API HTTP {exc.code}: {err}") from exc
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError("VL API timeout") from exc
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"VL API error: {exc}") from exc


def call_vl_extract_page(
    image_png_bytes: bytes,
    *,
    api_url: str,
    model: str,
    api_key: str,
    timeout: int,
    two_pass: bool = True,
) -> str:
    """POST chat/completions with one image; returns assistant text.

    two_pass=True (по умолчанию): два запроса (текст полосы, затем [ИЛЛЮСТРАЦИИ]), результат склеивается
    в merge_vl_prepare_passes — чтобы не упираться в max_tokens одним ответом.
    """
    if not two_pass:
        return _vl_chat_completion(
            image_png_bytes=image_png_bytes,
            prompt=VL_PAGE_TEXT_PASS_RU,
            api_url=api_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )

    text_raw = _vl_chat_completion(
        image_png_bytes=image_png_bytes,
        prompt=VL_PAGE_TEXT_PASS_RU,
        api_url=api_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
    )
    try:
        ill_raw = _vl_chat_completion(
            image_png_bytes=image_png_bytes,
            prompt=VL_PAGE_ILLUSTRATIONS_PASS_RU,
            api_url=api_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("VL: запрос блока иллюстраций не выполнен: %s", exc)
        ill_raw = ""

    return merge_vl_prepare_passes(text_raw, ill_raw)
