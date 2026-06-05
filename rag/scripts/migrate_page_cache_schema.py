#!/usr/bin/env python3
"""
Дописывает в существующие page_*.json поля schema_version=2, body_text, illustrations_block
без повторного VL: читает поле text и делит через split_vl_page_to_body_and_illustrations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def migrate_dir(cache_dir: Path, dry_run: bool) -> tuple[int, int]:
    from rag_service.ingestion import split_vl_page_to_body_and_illustrations

    updated = 0
    skipped = 0
    for path in sorted(cache_dir.glob("page_*.json")):
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            skipped += 1
            continue
        if int(data.get("schema_version", 0)) >= 2 and "body_text" in data:
            skipped += 1
            continue
        text = str(data.get("text", ""))
        body, ill = split_vl_page_to_body_and_illustrations(text)
        data["schema_version"] = 2
        data["body_text"] = body
        data["illustrations_block"] = ill
        if not dry_run:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        updated += 1
    return updated, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate page_*.json cache to schema v2 (body vs illustrations).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Каталог prepare (ищет .prepare_page_cache/**/page_*.json).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report counts, do not write.")
    args = parser.parse_args()

    root = args.out_dir / ".prepare_page_cache"
    if not root.is_dir():
        print(f"No cache at {root}", file=sys.stderr)
        raise SystemExit(1)

    total_up = total_sk = 0
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        u, s = migrate_dir(sub, args.dry_run)
        if u or s:
            print(f"{sub.name}: updated={u} skipped/already={s}")
        total_up += u
        total_sk += s
    print(f"Total: updated={total_up} skipped={total_sk}")


if __name__ == "__main__":
    main()
