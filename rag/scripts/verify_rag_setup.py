#!/usr/bin/env python3
"""Быстрая проверка по чеклисту: счётчики таблиц и опционально GET /health RAG."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    dsn = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/rag")
    rag_url = (os.getenv("RAG_URL") or os.getenv("RAG_HEALTH_URL") or "").strip().rstrip("/")

    try:
        from psycopg import connect
    except ImportError:
        print("Установите psycopg: pip install psycopg[binary]", file=sys.stderr)
        raise SystemExit(1) from None

    print("=== PostgreSQL (POSTGRES_DSN) ===")
    counts: dict[str, int] = {}
    try:
        with connect(dsn) as conn:
            with conn.cursor() as cur:
                for table in ("rag_chunks", "rag_illustration_chunks", "rag_page_image_embeddings"):
                    try:
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        n = cur.fetchone()[0]
                        counts[table] = n
                        print(f"  {table}: {n}")
                    except Exception as exc:  # noqa: BLE001
                        print(f"  {table}: (ошибка: {exc})")
    except Exception as exc:  # noqa: BLE001
        print(f"  Ошибка подключения: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if rag_url:
        url = f"{rag_url}/health"
        print(f"\n=== GET {url} ===")
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            for k in sorted(data.keys()):
                print(f"  {k}: {data[k]}")
        except urllib.error.URLError as exc:
            print(f"  Недоступно: {exc}", file=sys.stderr)
    else:
        print("\n(RAG_URL не задан — пропуск HTTP /health; задайте RAG_URL=http://localhost:8000)")

    n_chunks = counts.get("rag_chunks", 0)
    n_page = counts.get("rag_page_image_embeddings", 0)
    if n_chunks > 0 and n_page == 0:
        print(
            "\nЗамечание multimodal: rag_chunks > 0, но rag_page_image_embeddings = 0 — "
            "поиск страниц по эмбеддингу фото в /analyze-image не сработает; проверьте "
            "MULTIMODAL_ENABLED, IMAGE_EMBEDDING_*, RAG_PDF_ROOT и переиндексацию (reindex)."
        )

    print("\nГотово. Ожидание: rag_chunks > 0 при наличии документов; rag_page_image_embeddings > 0 при настроенном multimodal.")


if __name__ == "__main__":
    main()
