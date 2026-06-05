#!/usr/bin/env python3
"""Статистика по rag_chunks: длины chunk_text и число чанков на source_page (POSTGRES_DSN из окружения)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk length / per-page distribution for rag_chunks.")
    parser.add_argument(
        "--problem-doc-limit",
        type=int,
        default=15,
        help="How many doc_id to list with many short or long chunks.",
    )
    args = parser.parse_args()

    dsn = os.getenv("POSTGRES_DSN", "postgresql://postgres:postgres@localhost:5432/rag")
    try:
        from psycopg import connect
    except ImportError:
        print("Install psycopg: pip install psycopg[binary]", file=sys.stderr)
        raise SystemExit(1) from None

    with connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  COUNT(*) AS n,
                  AVG(LENGTH(chunk_text))::float AS avg_len,
                  MIN(LENGTH(chunk_text)) AS min_len,
                  MAX(LENGTH(chunk_text)) AS max_len,
                  percentile_cont(0.5) WITHIN GROUP (ORDER BY LENGTH(chunk_text)) AS p50,
                  percentile_cont(0.9) WITHIN GROUP (ORDER BY LENGTH(chunk_text)) AS p90
                FROM rag_chunks
                """
            )
            row = cur.fetchone()
            print("=== length(chunk_text) ===")
            print(f"  chunks: {row[0]}  avg: {row[1]:.1f}  min/max: {row[2]}/{row[3]}  p50: {row[4]:.0f}  p90: {row[5]:.0f}")

            cur.execute(
                """
                SELECT COUNT(*) FROM rag_chunks WHERE LENGTH(chunk_text) < 80
                """
            )
            short_n = cur.fetchone()[0]
            cur.execute(
                """
                SELECT COUNT(*) FROM rag_chunks WHERE LENGTH(chunk_text) > 1800
                """
            )
            long_n = cur.fetchone()[0]
            print(f"  < 80 chars: {short_n}   > 1800 chars: {long_n}")

            print("\n=== chunks per source_page (top 20 by count) ===")
            cur.execute(
                """
                SELECT COALESCE(source_page::text, '(null)'), COUNT(*) AS c
                FROM rag_chunks
                GROUP BY 1
                ORDER BY c DESC
                LIMIT 20
                """
            )
            for sp, c in cur.fetchall():
                print(f"  {sp}: {c}")

            print("\n=== doc_id with many short chunks (<80) ===")
            cur.execute(
                """
                SELECT doc_id, COUNT(*) AS c
                FROM rag_chunks
                WHERE LENGTH(chunk_text) < 80
                GROUP BY doc_id
                ORDER BY c DESC
                LIMIT %s
                """,
                (args.problem_doc_limit,),
            )
            for doc_id, c in cur.fetchall():
                print(f"  {doc_id}: {c}")

            print("\n=== doc_id with many long chunks (>1800) ===")
            cur.execute(
                """
                SELECT doc_id, COUNT(*) AS c
                FROM rag_chunks
                WHERE LENGTH(chunk_text) > 1800
                GROUP BY doc_id
                ORDER BY c DESC
                LIMIT %s
                """,
                (args.problem_doc_limit,),
            )
            for doc_id, c in cur.fetchall():
                print(f"  {doc_id}: {c}")


if __name__ == "__main__":
    main()
