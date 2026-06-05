import asyncio
import base64
import json
import re
import socket
import urllib.error
import urllib.request

TEXT_REQUEST_TIMEOUT_SEC = 6000
IMAGE_REQUEST_TIMEOUT_SEC = 9000

RESPONSE_SECTION_SEPARATOR = "\n\n----------------------\n\n"


class RAGClient:
    def __init__(self, rag_url: str) -> None:
        self.rag_url = rag_url

    async def ask_text(self, question: str) -> str:
        payload = json.dumps({"question": question}).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.rag_url}/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _send() -> str:
            with urllib.request.urlopen(request, timeout=TEXT_REQUEST_TIMEOUT_SEC) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                answer = parsed.get("answer", "RAG не вернул ответ")
                matches = parsed.get("matches", [])
                base = self._format_user_response(
                    main_text=self._sanitize_text(answer),
                    matches=matches,
                    intro="Где это найдено в базе знаний:",
                )
                return base + self._index_status_note(parsed)

        try:
            return await asyncio.to_thread(_send)
        except (TimeoutError, socket.timeout):
            return (
                "RAG/LLM отвечает слишком долго (таймаут после длительного ожидания). "
                "Попробуйте еще раз."
            )
        except urllib.error.URLError:
            return "Не удалось обратиться к RAG сервису."
        except json.JSONDecodeError:
            return "RAG вернул некорректный ответ."

    async def ask_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        user_hint: str,
        *,
        specialist_profile=None,
    ) -> tuple[str, bytes | None, str]:
        extra = {}
        if specialist_profile is not None:
            extra = {
                "specialist_full_name": (getattr(specialist_profile, "full_name", None) or ""),
                "specialist_education": (getattr(specialist_profile, "specialist_education", None) or ""),
                "specialist_qualification": (getattr(specialist_profile, "specialist_qualification", None) or ""),
                "specialist_additional_training": (getattr(specialist_profile, "specialist_additional_training", None) or ""),
                "specialist_position": (getattr(specialist_profile, "specialist_position", None) or ""),
                "specialist_research_interests": (getattr(specialist_profile, "specialist_research_interests", None) or ""),
                "specialist_experience_years": (getattr(specialist_profile, "specialist_experience_years", None) or ""),
                "report_basis": (getattr(specialist_profile, "report_basis", None) or ""),
            }
        payload = json.dumps(
            {
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
                "mime_type": mime_type,
                "user_hint": user_hint,
                **extra,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.rag_url}/analyze-image",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        def _send() -> tuple[str, bytes | None, str]:
            with urllib.request.urlopen(request, timeout=IMAGE_REQUEST_TIMEOUT_SEC) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body)
                description = self._sanitize_text(
                    parsed.get("image_description", "Описание не получено.")
                )
                matches = parsed.get("matches", [])
                classification = self._sanitize_text(parsed.get("classification") or "")
                no_sources = self._sanitize_text(parsed.get("no_sources_summary") or "")
                report_b64 = (parsed.get("report_docx_base64") or "").strip()
                report_name = (parsed.get("report_file_name") or "").strip() or "заключение.docx"
                report_bytes = None
                if report_b64:
                    try:
                        report_bytes = base64.b64decode(report_b64)
                    except Exception:
                        report_bytes = None

                if not matches:
                    lines = [
                        "В справочнике не найдено близких фрагментов по этому фото.",
                        "",
                        "Что можно сделать: добавьте подпись с термином из книги или отправьте крупнее "
                        "область тату.",
                        "",
                        f"Описание изображения:\n{description}",
                    ]
                    if no_sources.strip():
                        lines.extend(
                            [
                                "",
                                "Краткий вывод (без подтверждения из справочника):",
                                no_sources.strip(),
                            ]
                        )
                    return "\n".join(lines) + self._index_status_note(parsed), report_bytes, report_name

                base = self._format_user_response(
                    main_text="Описание изображения:\n" + description,
                    matches=matches,
                    intro="Источники и цитаты:",
                )
                note = self._index_status_note(parsed)
                if classification.strip():
                    text_out = (
                        base
                        + note
                        + RESPONSE_SECTION_SEPARATOR
                        + "Классификация по источникам:\n"
                        + classification.strip()
                    )
                    return text_out, report_bytes, report_name
                return base + note, report_bytes, report_name

        try:
            return await asyncio.to_thread(_send)
        except (TimeoutError, socket.timeout):
            return (
                "Обработка изображения заняла слишком много времени "
                "(таймаут после длительного ожидания). Попробуйте еще раз."
            ), None, ""
        except urllib.error.URLError:
            return "Не удалось обратиться к RAG сервису.", None, ""
        except json.JSONDecodeError:
            return "RAG вернул некорректный ответ.", None, ""

    @staticmethod
    def _index_status_note(parsed: dict) -> str:
        ready = parsed.get("index_ready", True)
        n = parsed.get("index_chunk_count", 0)
        err = parsed.get("index_error")
        if err:
            return f"\n\n[Индекс: ошибка — {err}]"
        if not ready:
            return "\n\n[Индекс: ещё не готов после старта сервиса]"
        if n == 0:
            return "\n\n[Индекс: в базе 0 чанков — ответы без опоры на книги]"
        return ""

    @staticmethod
    def _fragment_kind_ru(kind: str) -> str:
        return {
            "text": "текст",
            "illustration": "иллюстрация",
            "multimodal_page": "страница по фото",
        }.get(kind, kind)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        # Hide chain-of-thought tags returned by some local models.
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _format_user_response(self, main_text: str, matches: list[dict], intro: str) -> str:
        if not matches:
            return main_text

        lines = [main_text, "", intro]
        for idx, item in enumerate(matches, start=1):
            file_name = item.get("object_name", "Неизвестный файл")
            source_link = item.get("source_link", "источник не указан")
            score = item.get("score")
            quote = self._sanitize_text(item.get("description", ""))[:500]
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "n/a"
            fk = item.get("fragment_kind") or "text"
            ci = item.get("chunk_index")
            kind_line = f"   Тип: {self._fragment_kind_ru(fk)}"
            if ci is not None:
                kind_line += f", chunk_index={ci}"
            lines.append(
                f"{idx}) Файл: {file_name}\n"
                f"{kind_line}\n"
                f"   Где найдено: {source_link}\n"
                f"   Релевантность: {score_text}\n"
                f"   Цитата: {quote}"
            )
        return "\n".join(lines)

