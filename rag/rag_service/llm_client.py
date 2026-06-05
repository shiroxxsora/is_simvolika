import base64
import json
import logging
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from fastapi import HTTPException

from rag_service.config import Settings

logger = logging.getLogger(__name__)

# region agent log
_DEBUG_LOG = Path(__file__).resolve().parents[2] / "debug-6d9dc9.log"


def _debug_ndjson(
    *,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict | None = None,
    run_id: str = "pre-fix",
) -> None:
    try:
        line = {
            "sessionId": "6d9dc9",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "timestamp": int(time.time() * 1000),
            "data": data or {},
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except OSError:
        pass


# endregion


def _exception_is_embedding_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    if isinstance(exc, urllib.error.URLError) and exc.reason is not None:
        r = exc.reason
        if isinstance(r, (TimeoutError, socket.timeout)):
            return True
        if "timed out" in str(r).lower():
            return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def _url_ipv4_for_host_docker_internal(url: str) -> str:
    """Соединение с хостом по IPv4 (AF_INET), иначе на части Docker/Linux бывает errno 101 ENETUNREACH для IPv6."""
    if not url:
        return url
    parsed = urlparse(url)
    if (parsed.hostname or "").lower() != "host.docker.internal":
        return url
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    try:
        infos = socket.getaddrinfo(
            "host.docker.internal",
            port,
            socket.AF_INET,
            socket.SOCK_STREAM,
        )
    except OSError:
        return url
    if not infos:
        return url
    ip = infos[0][4][0]
    if parsed.port is not None:
        netloc = f"{ip}:{parsed.port}"
    else:
        netloc = f"{ip}:{port}"
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _v1_to_legacy_embeddings_url(v1_url: str) -> str:
    u = v1_url.strip()
    if "/v1/embeddings" in u:
        return u.replace("/v1/embeddings", "/embeddings", 1)
    return u


def _parse_embedding_response_body(body: object) -> list[float] | None:
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, list) and data:
            emb = data[0].get("embedding") if isinstance(data[0], dict) else None
            if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
                return emb
        emb = body.get("embedding")
        if isinstance(emb, list) and emb:
            if isinstance(emb[0], (int, float)):
                return emb
            # pooling none: список векторов по токенам — берём последний (часто совпадает с «last» pooling)
            if isinstance(emb[0], list) and emb[0] and isinstance(emb[0][0], (int, float)):
                last = emb[-1]
                return last if isinstance(last, list) else None
    if isinstance(body, list) and body:
        first = body[0]
        if isinstance(first, dict):
            emb = first.get("embedding")
            if isinstance(emb, list) and emb:
                if isinstance(emb[0], (int, float)):
                    return emb
                if isinstance(emb[0], list) and emb[0] and isinstance(emb[0][0], (int, float)):
                    last = emb[-1]
                    return last if isinstance(last, list) else None
    return None


def _error_is_v1_input_type_rejection(error_text: str) -> bool:
    t = error_text.lower()
    return "input" in t and "string" in t and ("must be" in t or "array of strings" in t)


def _image_embedding_from_http_200_body(
    settings: Settings,
    raw: str,
    *,
    label: str,
    post_url: str,
) -> list[float]:
    """Разбор ответа 200 от embedding API; при невалидном теле — явная ошибка (LM Studio часто отдаёт заглушку на POST /embeddings)."""
    raw_preview = raw.strip()[:900]
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        # region agent log
        _debug_ndjson(
            hypothesis_id="H5",
            location="llm_client.py:_image_embedding_from_http_200_body",
            message="json_decode_failed",
            data={"label": label, "preview": raw_preview[:400]},
        )
        # endregion
        raise HTTPException(
            status_code=502,
            detail=(
                f"Ответ image embedding не является JSON (attempt={label}, url …{post_url[-56:]}). "
                "LM Studio часто не реализует POST /embeddings (в логе: Unexpected endpoint … Returning 200 anyway). "
                "Задайте IMAGE_EMBEDDING_SKIP_LEGACY_EMBEDDINGS_PATH=true и/или MULTIMODAL_ENABLED=false, либо "
                "отдельный llama-server с multimodal. Фрагмент ответа: "
                f"{raw_preview!r}"
            ),
        ) from e
    # region agent log
    usage = body.get("usage") if isinstance(body, dict) else None
    _debug_ndjson(
        hypothesis_id="H2",
        location="llm_client.py:_image_embedding_from_http_200_body",
        message="response_ok_json",
        data={"label": label, "post_url_suffix": post_url[-48:], "usage": usage},
    )
    # endregion
    embedding = _parse_embedding_response_body(body)
    if not isinstance(embedding, list) or not embedding:
        # region agent log
        _debug_ndjson(
            hypothesis_id="H5",
            location="llm_client.py:_image_embedding_from_http_200_body",
            message="no_embedding_vector",
            data={
                "label": label,
                "top_type": type(body).__name__,
                "preview": raw_preview[:400],
            },
        )
        # endregion
        raise HTTPException(
            status_code=502,
            detail=(
                f"В ответе нет вектора data[].embedding (attempt={label}). "
                "Типично для LM Studio: /v1/embeddings не принимает multimodal-объект в input, "
                "а POST /embeddings не подключён. Отключите multimodal (MULTIMODAL_ENABLED=false) или используйте "
                "сервер llama.cpp с поддержкой multimodal embeddings. Фрагмент ответа: "
                f"{raw_preview!r}"
            ),
        )
    if len(embedding) != settings.image_embedding_dim:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Ожидался размер image embedding {settings.image_embedding_dim}, "
                f"но получен {len(embedding)}. Проверьте IMAGE_EMBEDDING_DIM и модель."
            ),
        )
    return embedding


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._logged_image_embedding_model = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"
        return headers

    def get_embedding(self, input_text: str) -> list[float]:
        payload = {"model": self.settings.embedding_model, "input": input_text}
        data = json.dumps(payload).encode("utf-8")
        embed_url = _url_ipv4_for_host_docker_internal(self.settings.embedding_api_url)
        # region agent log
        _debug_ndjson(
            hypothesis_id="H2",
            location="llm_client.py:get_embedding",
            message="request",
            data={
                "model": self.settings.embedding_model,
                "embed_url_suffix": embed_url[-64:],
                "input_len": len(input_text),
            },
        )
        # endregion
        request = urllib.request.Request(
            url=embed_url,
            data=data,
            method="POST",
            headers=self._headers(),
        )

        timeout_sec = self.settings.embedding_timeout_sec
        try:
            with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                raw = response.read().decode("utf-8")
                body = json.loads(raw)
                # region agent log
                if isinstance(body, dict) and body.get("error") is not None:
                    _debug_ndjson(
                        hypothesis_id="H5",
                        location="llm_client.py:get_embedding",
                        message="http_200_body_has_error_key",
                        data={
                            "error_field": str(body.get("error"))[:500],
                            "model": self.settings.embedding_model,
                        },
                    )
                # endregion
                embedding = body["data"][0]["embedding"]
                if len(embedding) != self.settings.embedding_dim:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            f"Ожидался размер embedding {self.settings.embedding_dim}, "
                            f"но получен {len(embedding)}. Проверьте EMBEDDING_MODEL/EMBEDDING_DIM."
                        ),
                    )
                return embedding
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore")
            # region agent log
            _debug_ndjson(
                hypothesis_id="H1",
                location="llm_client.py:get_embedding",
                message="http_error",
                data={
                    "status": exc.code,
                    "error_excerpt": error_text[:900],
                    "model": self.settings.embedding_model,
                    "embed_url_suffix": embed_url[-64:],
                },
            )
            # endregion
            raise HTTPException(status_code=502, detail=f"Ошибка embedding API: {error_text}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Таймаут embedding API (лимит {timeout_sec}s). Увеличьте EMBEDDING_TIMEOUT_SEC в .env; "
                    "загрузите модель эмбеддингов в LM Studio до массовой индексации (первый запрос может быть долгим)."
                ),
            ) from exc
        except urllib.error.URLError as exc:
            if _exception_is_embedding_timeout(exc):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Таймаут embedding API (лимит {timeout_sec}s). Увеличьте EMBEDDING_TIMEOUT_SEC в .env; "
                        "загрузите модель эмбеддингов в LM Studio до массовой индексации."
                    ),
                ) from exc
            raise HTTPException(status_code=502, detail="Не удалось получить embedding.") from exc
        except OSError as exc:
            if _exception_is_embedding_timeout(exc):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Таймаут embedding API (лимит {timeout_sec}s). Увеличьте EMBEDDING_TIMEOUT_SEC в .env; "
                        "загрузите модель эмбеддингов в LM Studio до массовой индексации."
                    ),
                ) from exc
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Нет связи с embedding API ({self.settings.embedding_api_url}): {exc!s}. "
                    "Проверьте: LM Studio запущен, локальный сервер включён, загружена модель эмбеддингов, "
                    "порт совпадает с EMBEDDING_API_URL; в LM Studio включите доступ с сети (не только localhost). "
                    "В Docker: extra_hosts host.docker.internal:host-gateway в compose; при 101 попробуйте в .env "
                    "явный IP хоста Windows: http://<IPv4>:<порт>/v1/embeddings."
                ),
            ) from exc
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=502, detail="Не удалось получить embedding.") from exc

    def get_image_embedding(self, image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
        """Эмбеддинг изображения (тот же сервис должен индексировать страницы PDF и запрос пользователя).

        LM Studio: /v1/embeddings принимает только string | string[] в input; multimodal-объект отклоняется.
        POST /embeddings (нативный llama.cpp) в LM Studio часто не реализован (лог: Unexpected endpoint … 200 anyway).
        По умолчанию формат v1_mtmd: сначала /v1/embeddings с объектом input; затем fallback POST …/embeddings,
        если IMAGE_EMBEDDING_SKIP_LEGACY_EMBEDDINGS_PATH не включён (для LM Studio лучше skip=true — без бесполезного POST /embeddings).

        server_content: только нативный /embeddings (для полного llama-server, не для LM Studio).
        openai_data_url: data URL в input — на многих серверах base64 токенизируется как текст.

        В payload уходит settings.image_embedding_model. Некоторые локальные серверы (LM Studio) в логах
        показывают имя загруженной в UI модели, а не поле model — при расхождении загрузите VL-эмбеддер
        как активную модель эмбеддингов или проверьте, что сервер читает model из тела запроса.
        """
        if not self.settings.image_embedding_api_url:
            raise HTTPException(
                status_code=503,
                detail="IMAGE_EMBEDDING_API_URL не задан — multimodal-поиск по изображению недоступен.",
            )
        b64_raw = base64.b64encode(image_bytes).decode("ascii")
        model = self.settings.image_embedding_model
        fmt = self.settings.image_embedding_request_format
        marker = self.settings.image_embedding_media_marker
        base_v1 = _url_ipv4_for_host_docker_internal(self.settings.image_embedding_api_url)
        legacy_url = _url_ipv4_for_host_docker_internal(
            _v1_to_legacy_embeddings_url(self.settings.image_embedding_api_url)
        )
        if not self._logged_image_embedding_model:
            self._logged_image_embedding_model = True
            logger.info(
                "POST image embedding: format=%s model=%r dim=%s marker=%r skip_legacy_path=%s",
                fmt,
                model,
                self.settings.image_embedding_dim,
                marker,
                self.settings.image_embedding_skip_legacy_embeddings_path,
            )
        # region agent log
        _debug_ndjson(
            hypothesis_id="H1",
            location="llm_client.py:get_image_embedding",
            message="payload_shape",
            data={
                "format": fmt,
                "b64_len": len(b64_raw),
                "mime_type": mime_type,
                "legacy_url_suffix": legacy_url[-48:],
            },
        )
        # endregion

        content_payload = {
            "model": model,
            "content": [{"prompt_string": marker, "multimodal_data": [b64_raw]}],
        }
        mtmd_v1_payload = {
            "model": model,
            "input": {"prompt_string": marker, "multimodal_data": [b64_raw]},
        }
        data_url = f"data:{mime_type};base64,{b64_raw}"
        openai_payload: dict = {"model": model, "input": [data_url]}

        if fmt == "openai_data_url":
            attempts: list[tuple[str, dict, str]] = [(base_v1, openai_payload, "openai_data_url")]
        elif fmt == "server_content":
            attempts = [(legacy_url, content_payload, "server_content")]
        elif self.settings.image_embedding_skip_legacy_embeddings_path:
            attempts = [(base_v1, mtmd_v1_payload, "v1_mtmd")]
        else:
            attempts = [
                (base_v1, mtmd_v1_payload, "v1_mtmd"),
                (legacy_url, content_payload, "server_content_fallback"),
            ]

        img_timeout = self.settings.image_embedding_timeout_sec
        try:
            for post_url, payload, label in attempts:
                data = json.dumps(payload).encode("utf-8")
                request = urllib.request.Request(
                    url=post_url,
                    data=data,
                    method="POST",
                    headers=self._headers(),
                )
                try:
                    with urllib.request.urlopen(request, timeout=img_timeout) as response:
                        raw = response.read().decode("utf-8")
                        return _image_embedding_from_http_200_body(
                            self.settings,
                            raw,
                            label=label,
                            post_url=post_url,
                        )
                except urllib.error.HTTPError as exc:
                    error_text = exc.read().decode("utf-8", errors="ignore")
                    # region agent log
                    _debug_ndjson(
                        hypothesis_id="H3",
                        location="llm_client.py:get_image_embedding",
                        message="http_error",
                        data={
                            "attempt": label,
                            "status": exc.code,
                            "error_excerpt": error_text[:500],
                        },
                    )
                    # endregion
                    if (
                        label == "v1_mtmd"
                        and _error_is_v1_input_type_rejection(error_text)
                        and len(attempts) > 1
                    ):
                        # region agent log
                        _debug_ndjson(
                            hypothesis_id="H4",
                            location="llm_client.py:get_image_embedding",
                            message="v1_mtmd_rejected_retrying_server_content",
                            data={"error_excerpt": error_text[:300]},
                        )
                        # endregion
                        continue
                    raise HTTPException(status_code=502, detail=f"Ошибка image embedding API: {error_text}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    f"Таймаут image embedding API (лимит {img_timeout}s). Увеличьте IMAGE_EMBEDDING_TIMEOUT_SEC; "
                    "запрос с PNG/base64 тяжелее текстового embedding."
                ),
            ) from exc
        except urllib.error.URLError as exc:
            if _exception_is_embedding_timeout(exc):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Таймаут image embedding API (лимит {img_timeout}s). Увеличьте IMAGE_EMBEDDING_TIMEOUT_SEC."
                    ),
                ) from exc
            raise HTTPException(status_code=502, detail="Не удалось получить image embedding.") from exc
        except OSError as exc:
            if _exception_is_embedding_timeout(exc):
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Таймаут image embedding API (лимит {img_timeout}s). Увеличьте IMAGE_EMBEDDING_TIMEOUT_SEC."
                    ),
                ) from exc
            raise HTTPException(
                status_code=502,
                detail=f"Нет связи с image embedding API ({self.settings.image_embedding_api_url}): {exc!s}.",
            ) from exc
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise HTTPException(status_code=502, detail="Не удалось получить image embedding.") from exc

    def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.2,
        timeout: int = 600,
    ) -> str:
        payload = {
            "model": model or self.settings.llm_model,
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        chat_url = _url_ipv4_for_host_docker_internal(self.settings.llm_api_url)
        request = urllib.request.Request(
            url=chat_url,
            data=data,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore")
            raise HTTPException(status_code=502, detail=f"Ошибка LLM API: {error_text}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise HTTPException(
                status_code=504,
                detail=(
                    "Таймаут при запросе к LLM API. "
                    "Проверьте, что модель загружена в LM Studio и попробуйте снова."
                ),
            ) from exc
        except OSError as exc:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Нет связи с LLM API ({self.settings.llm_api_url}): {exc!s}. "
                    "Проверьте LM Studio и адрес LLM_API_URL (для Docker — host.docker.internal)."
                ),
            ) from exc
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=502, detail="Не удалось получить ответ от LLM.") from exc

