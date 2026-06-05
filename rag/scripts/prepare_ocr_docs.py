import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for _p in (ROOT_DIR, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

logger = logging.getLogger("prepare_ocr_docs")

PAGE_CACHE_DIRNAME = ".prepare_page_cache"


def _pdf_rel_posix(rel: Path) -> str:
    return rel.as_posix()


def _page_cache_root(out_dir: Path) -> Path:
    return out_dir / PAGE_CACHE_DIRNAME


def _pdf_cache_dir(out_dir: Path, rel: Path) -> Path:
    key = hashlib.sha256(_pdf_rel_posix(rel).encode("utf-8")).hexdigest()
    return _page_cache_root(out_dir) / key


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "meta.json"


def _page_json_path(cache_dir: Path, page_num_1based: int) -> Path:
    return cache_dir / f"page_{page_num_1based:05d}.json"


def _cache_meta_matches(
    cache_dir: Path,
    pdf_path: Path,
    rel: Path,
    pdf_text_first: bool,
    page_engine: str,
    vl_model: str,
    vl_two_pass: bool,
) -> bool:
    mp = _meta_path(cache_dir)
    if not mp.is_file():
        return False
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    st = pdf_path.stat()
    if abs(float(meta.get("pdf_mtime", 0)) - st.st_mtime) > 1e-3:
        return False
    if int(meta.get("pdf_size", -1)) != st.st_size:
        return False
    if meta.get("rel") != _pdf_rel_posix(rel):
        return False
    if bool(meta.get("pdf_text_first")) != pdf_text_first:
        return False
    if str(meta.get("page_engine", "")) != page_engine:
        return False
    if str(meta.get("vl_model", "")) != vl_model:
        return False
    if bool(meta.get("vl_two_pass", True)) != vl_two_pass:
        return False
    return True


def _write_cache_meta(
    cache_dir: Path,
    rel: Path,
    pdf_path: Path,
    pdf_text_first: bool,
    page_engine: str,
    vl_model: str,
    vl_two_pass: bool,
) -> None:
    st = pdf_path.stat()
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "rel": _pdf_rel_posix(rel),
        "pdf_mtime": st.st_mtime,
        "pdf_size": st.st_size,
        "pdf_text_first": pdf_text_first,
        "page_engine": page_engine,
        "vl_model": vl_model,
        "vl_two_pass": vl_two_pass,
    }
    _meta_path(cache_dir).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_source_rel_marker(cache_dir: Path, rel: Path) -> None:
    """Human-readable label: which PDF this cache folder belongs to (folder name is a hash)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "source_pdf_rel.txt").write_text(_pdf_rel_posix(rel) + "\n", encoding="utf-8")


def _count_page_cache_files(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    return sum(1 for p in cache_dir.glob("page_*.json") if p.is_file())


def _write_page_cache_file(cache_dir: Path, page_num_1based: int, text: str, source: str) -> None:
    from rag_service.ingestion import split_vl_page_to_body_and_illustrations

    cache_dir.mkdir(parents=True, exist_ok=True)
    body, ill = split_vl_page_to_body_and_illustrations(text)
    payload = {
        "schema_version": 2,
        "source": source,
        "text": text,
        "body_text": body,
        "illustrations_block": ill,
    }
    _page_json_path(cache_dir, page_num_1based).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def _read_page_cache_file(cache_dir: Path, page_num_1based: int) -> tuple[str, str] | None:
    p = _page_json_path(cache_dir, page_num_1based)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return str(data.get("text", "")), str(data.get("source", "cache"))
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def _finalize_page_text(raw: str) -> str:
    """Apply same latin/cyrillic fixes as RAG ingestion, then light cleanup."""
    from rag_service.ingestion import _normalize_ocr_output
    from rag_service.llm_output_cleanup import clean_page_text_after_vl

    return clean_page_text_after_vl(_normalize_ocr_output(raw))


def _finalize_llm_page_text(raw: str) -> str:
    """Cleanup for vision-LLM output (do not apply OCR-specific char maps)."""
    from rag_service.llm_output_cleanup import finalize_llm_page_text

    return finalize_llm_page_text(raw)


def _page_to_png_bytes(page, zoom: float) -> bytes:
    import fitz

    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _is_quality_text(text: str, min_chars: int, min_alpha_ratio: float) -> bool:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) < min_chars:
        return False

    alpha_count = len(re.findall(r"[A-Za-zА-Яа-яЁё]", cleaned))
    if alpha_count == 0:
        return False
    alpha_ratio = alpha_count / max(1, len(cleaned))
    if alpha_ratio < min_alpha_ratio:
        return False

    # Reject pages dominated by isolated symbols/noisy OCR fragments.
    long_words = re.findall(r"[A-Za-zА-Яа-яЁё]{4,}", cleaned)
    return len(long_words) >= 12


def _ocr_single_page(
    pdf_path: Path,
    page_index: int,
    lang: str,
    pdf_text_first: bool,
    page_engine: str,
    vl_api_url: str | None,
    vl_model: str,
    vl_api_key: str,
    vl_timeout: int,
    vl_zoom: float,
    fallback_tesseract: bool,
    vl_two_pass: bool,
    cache_dir: Path | None,
    read_page_cache: bool,
) -> tuple[int, str, str]:
    """Returns (page_num_1based, text, source: pdf_text|vl|ocr).

    With page_engine 'vl', every page is sent to the vision LLM (no shortcut on PDF text layer).
    """
    import fitz
    from rag_service.ingestion import _ocr_pdf_page, _text_quality_score
    from vl_page_client import call_vl_extract_page

    page_num = page_index + 1
    if cache_dir and read_page_cache:
        cached = _read_page_cache_file(cache_dir, page_num)
        if cached is not None:
            text, src = cached
            return page_num, text, src

    os.environ["TESSERACT_LANG"] = lang
    os.environ["PDF_OCR_LANG"] = lang
    pdf_doc = fitz.open(str(pdf_path))
    try:
        page = pdf_doc.load_page(page_index)
        extracted_raw = (page.get_text() or "").strip()
        ext_final = _finalize_page_text(extracted_raw)
        ext_score = _text_quality_score(ext_final)

        def _tesseract_fallback() -> tuple[str, str]:
            ocr_raw = _ocr_pdf_page(pdf_doc, page_index)
            return _finalize_page_text(ocr_raw), "ocr"

        if not pdf_text_first:
            if page_engine == "tesseract":
                text, src = _tesseract_fallback()
                return page_index + 1, text, src
            # vl / auto: still run VL below without skipping on good pdf

        # Good embedded text — skip Tesseract OCR (and skip VL only when not using VL engine).
        if (
            page_engine != "vl"
            and pdf_text_first
            and len(ext_final) >= 80
            and ext_score >= 2.15
        ):
            return page_index + 1, ext_final, "pdf_text"

        if page_engine == "tesseract":
            ocr_raw = _ocr_pdf_page(pdf_doc, page_index)
            ocr_final = _finalize_page_text(ocr_raw)
            ocr_score = _text_quality_score(ocr_final)
            if pdf_text_first and ext_score >= ocr_score and len(ext_final) >= 30:
                return page_index + 1, ext_final, "pdf_text"
            return page_index + 1, ocr_final, "ocr"

        # VL (or auto): vision LLM
        if not vl_api_url:
            logger.warning("page_engine=%s but vl_api_url is empty; using tesseract", page_engine)
            text, src = _tesseract_fallback()
            return page_index + 1, text, src

        png = _page_to_png_bytes(page, vl_zoom)
        try:
            raw_vl = call_vl_extract_page(
                png,
                api_url=vl_api_url,
                model=vl_model,
                api_key=vl_api_key,
                timeout=vl_timeout,
                two_pass=vl_two_pass,
            )
        except Exception as exc:
            logger.warning("VL failed page %s: %s", page_index + 1, exc)
            if fallback_tesseract:
                text, src = _tesseract_fallback()
                return page_index + 1, text, src
            return page_index + 1, "", "vl_error"

        vl_final = _finalize_llm_page_text(raw_vl)
        if vl_final.strip():
            return page_index + 1, vl_final, "vl"
        if fallback_tesseract:
            text, src = _tesseract_fallback()
            return page_index + 1, text, src
        return page_index + 1, "", "vl"
    finally:
        pdf_doc.close()


def _ocr_pdf_pages_parallel(
    pdf_path: Path,
    workers: int,
    lang: str,
    pdf_index: int,
    pdf_total: int,
    pdf_text_first: bool,
    page_engine: str,
    vl_api_url: str | None,
    vl_model: str,
    vl_api_key: str,
    vl_timeout: int,
    vl_zoom: float,
    fallback_tesseract: bool,
    vl_two_pass: bool,
    cache_dir: Path | None,
    read_page_cache: bool,
    write_page_cache: bool,
) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    total_pages = len(PdfReader(str(pdf_path)).pages)
    if total_pages == 0:
        return []

    cached_before = _count_page_cache_files(cache_dir) if cache_dir else 0
    logger.info(
        "[%s/%s] %s pages=%s workers=%s engine=%s pdf_text_first=%s vl=%s | "
        "json_в_кеше=%s/%s read_cache=%s",
        pdf_index,
        pdf_total,
        pdf_path.name,
        total_pages,
        max(1, workers),
        page_engine,
        pdf_text_first,
        vl_api_url or "(none)",
        cached_before,
        total_pages,
        read_page_cache,
    )

    results: dict[int, str] = {}
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {
            executor.submit(
                _ocr_single_page,
                pdf_path,
                page_index,
                lang,
                pdf_text_first,
                page_engine,
                vl_api_url,
                vl_model,
                vl_api_key,
                vl_timeout,
                vl_zoom,
                fallback_tesseract,
                vl_two_pass,
                cache_dir,
                read_page_cache,
            ): page_index
            for page_index in range(total_pages)
        }
        for future in concurrent.futures.as_completed(future_map):
            page_num, text, source = future.result()
            results[page_num] = text
            if cache_dir and write_page_cache:
                _write_page_cache_file(cache_dir, page_num, text, source)
            completed += 1
            char_count = len(text.strip())
            preview = " ".join(text.split())[:120]
            if char_count == 0:
                logger.info(
                    "[%s/%s] %s page %s/%s finished (%s/%s) [%s] -> empty",
                    pdf_index,
                    pdf_total,
                    pdf_path.name,
                    page_num,
                    total_pages,
                    completed,
                    total_pages,
                    source,
                )
            else:
                logger.info(
                    "[%s/%s] %s page %s/%s finished (%s/%s) [%s] | %s chars | %s",
                    pdf_index,
                    pdf_total,
                    pdf_path.name,
                    page_num,
                    total_pages,
                    completed,
                    total_pages,
                    source,
                    char_count,
                    preview,
                )

    return [(page_num, results.get(page_num, "")) for page_num in range(1, total_pages + 1)]


def prepare_docs(
    raw_dir: Path,
    toc_dir: Path,
    out_dir: Path,
    min_chars: int,
    min_alpha_ratio: float,
    workers: int,
    tesseract_lang: str,
    pdf_text_first: bool,
    page_engine: str,
    vl_api_url: str | None,
    vl_model: str,
    vl_api_key: str,
    vl_timeout: int,
    vl_zoom: float,
    fallback_tesseract: bool,
    vl_two_pass: bool = True,
    *,
    force: bool = False,
    no_page_cache: bool = False,
    skip_up_to_date_output: bool = True,
    legacy_single_stream: bool = True,
) -> tuple[int, int]:
    from rag_service.ingestion import _load_toc_map_for_doc

    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    generated = 0

    pdf_files = sorted(raw_dir.rglob("*.pdf"))
    logger.info("Found %s PDF files in %s", len(pdf_files), raw_dir)
    logger.info("Tesseract lang: %s", tesseract_lang)
    logger.info("page_engine=%s vl_model=%s vl_two_pass=%s", page_engine, vl_model, vl_two_pass)
    logger.info(
        "cache: page_cache=%s force=%s skip_up_to_date_output=%s legacy_single_stream=%s",
        not no_page_cache,
        force,
        skip_up_to_date_output,
        legacy_single_stream,
    )

    for index, pdf_path in enumerate(pdf_files, start=1):
        total += 1
        rel = pdf_path.relative_to(raw_dir)
        out_path = (out_dir / rel).with_suffix(".txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if skip_up_to_date_output and not force and out_path.is_file():
            if out_path.stat().st_mtime >= pdf_path.stat().st_mtime:
                logger.info(
                    "[%s/%s] Skip (output up to date vs PDF): %s",
                    index,
                    len(pdf_files),
                    rel,
                )
                continue

        cache_dir: Path | None = None if no_page_cache else _pdf_cache_dir(out_dir, rel)
        read_page_cache = False
        write_page_cache = cache_dir is not None
        if cache_dir is not None:
            if force:
                shutil.rmtree(cache_dir, ignore_errors=True)
            else:
                meta_ok = _cache_meta_matches(
                    cache_dir, pdf_path, rel, pdf_text_first, page_engine, vl_model, vl_two_pass
                )
                meta_exists = _meta_path(cache_dir).is_file()
                has_partial = cache_dir.is_dir() and any(cache_dir.glob("page_*.json"))
                if meta_ok:
                    pass
                elif not meta_exists and has_partial:
                    logger.warning(
                        "[%s/%s] Page cache: есть page_*.json без meta.json для %s — "
                        "проставляю meta (если меняли PDF или пайплайн, запустите с --force)",
                        index,
                        len(pdf_files),
                        rel,
                    )
                    _write_cache_meta(
                        cache_dir, rel, pdf_path, pdf_text_first, page_engine, vl_model, vl_two_pass
                    )
                    _write_source_rel_marker(cache_dir, rel)
                elif not meta_ok:
                    shutil.rmtree(cache_dir, ignore_errors=True)

            # meta.json до воркеров: при обрыве перезапуск продолжит с закешированных страниц.
            _write_cache_meta(
                cache_dir, rel, pdf_path, pdf_text_first, page_engine, vl_model, vl_two_pass
            )
            _write_source_rel_marker(cache_dir, rel)
            read_page_cache = not force and _cache_meta_matches(
                cache_dir, pdf_path, rel, pdf_text_first, page_engine, vl_model, vl_two_pass
            )
            n_cached = _count_page_cache_files(cache_dir)
            logger.info(
                "[%s/%s] Кеш страниц: файл=%s | папка=%s | уже сохранено json-страниц: %s",
                index,
                len(pdf_files),
                rel,
                cache_dir.name,
                n_cached,
            )

        logger.info("[%s/%s] Processing %s", index, len(pdf_files), rel)
        pages = _ocr_pdf_pages_parallel(
            pdf_path=pdf_path,
            workers=workers,
            lang=tesseract_lang,
            pdf_index=index,
            pdf_total=len(pdf_files),
            pdf_text_first=pdf_text_first,
            page_engine=page_engine,
            vl_api_url=vl_api_url,
            vl_model=vl_model,
            vl_api_key=vl_api_key,
            vl_timeout=vl_timeout,
            vl_zoom=vl_zoom,
            fallback_tesseract=fallback_tesseract,
            vl_two_pass=vl_two_pass,
            cache_dir=cache_dir,
            read_page_cache=read_page_cache,
            write_page_cache=write_page_cache,
        )

        good_pages = [
            (page_num, text)
            for page_num, text in pages
            if _is_quality_text(text, min_chars=min_chars, min_alpha_ratio=min_alpha_ratio)
        ]
        if not good_pages:
            logger.warning("[%s/%s] Skipped %s (no extracted pages)", index, len(pdf_files), rel)
            continue

        toc_map = _load_toc_map_for_doc(pdf_path.name, toc_dir)
        from rag_service.ingestion import split_vl_page_to_body_and_illustrations

        blocks: list[str] = [f"source_doc: {pdf_path.name}"]
        first_page = good_pages[0][0]
        first_chapter = toc_map.get(first_page) if toc_map else None
        if first_chapter:
            blocks.append(f"source_chapter: {first_chapter}")
        blocks.append(f"source_page: {first_page}")
        blocks.append("")

        for page_num, text in good_pages:
            blocks.append(f"[PAGE {page_num}]")
            chapter = toc_map.get(page_num) if toc_map else None
            if chapter:
                blocks.append(chapter)
            page_text = text.strip()
            if legacy_single_stream:
                blocks.append(page_text)
            else:
                body, ill = split_vl_page_to_body_and_illustrations(page_text)
                blocks.append("### RAG_PAGE_BODY")
                blocks.append(body)
                if ill.strip():
                    blocks.append("### RAG_PAGE_ILLUSTRATIONS")
                    blocks.append(ill.strip())
            blocks.append("")

        mapped_text = "\n".join(blocks).strip()
        if not mapped_text.strip():
            logger.warning("[%s/%s] Skipped %s (empty mapped output)", index, len(pdf_files), rel)
            continue

        out_path.write_text(mapped_text, encoding="utf-8")
        generated += 1
        logger.info(
            "[%s/%s] Saved %s (kept pages: %s/%s)",
            index,
            len(pdf_files),
            out_path.relative_to(out_dir),
            len(good_pages),
            len(pages),
        )

    return total, generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare mapped TXT from PDFs: vision LLM per page (default) or Tesseract-only + TOC; VL adds [ИЛЛЮСТРАЦИИ] with tattoo-centric captions.",
    )
    parser.add_argument("--raw-dir", required=True, help="Directory with raw PDF files")
    parser.add_argument("--toc-dir", required=True, help="Directory with TOC files ([TOC] format)")
    parser.add_argument("--out-dir", required=True, help="Output directory for generated TXT files")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument("--min-chars", type=int, default=180, help="Minimum characters to keep page text.")
    parser.add_argument(
        "--min-alpha-ratio",
        type=float,
        default=0.55,
        help="Minimum letter ratio (0..1) to keep page text.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel page workers. For vision-LLM on one GPU use 1 (default); Tesseract-only can try 2–4.",
    )
    parser.add_argument(
        "--tesseract-lang",
        default=None,
        help="Tesseract language pack(s), e.g. rus+eng. Default: TESSERACT_LANG env or rus+eng.",
    )
    parser.add_argument(
        "--pdf-text-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "With --page-engine tesseract: prefer embedded PDF text when it looks good, else Tesseract. "
            "Ignored for --page-engine vl: every page always goes through vision LLM."
        ),
    )
    parser.add_argument(
        "--page-engine",
        choices=["vl", "tesseract"],
        default=os.getenv("PREPARE_PAGE_ENGINE", "vl"),
        help="vl=vision LLM per page, tesseract=Tesseract only.",
    )
    parser.add_argument(
        "--vl-api-url",
        default=os.getenv("PREPARE_VL_API_URL") or os.getenv("LLM_API_URL") or "",
        help="OpenAI-compatible chat URL (e.g. http://127.0.0.1:1234/v1/chat/completions).",
    )
    parser.add_argument(
        "--vl-model",
        default=os.getenv("PREPARE_VL_MODEL") or os.getenv("VISION_MODEL", "qwen3-vl-8b-instruct"),
        help="Vision model id in LM Studio / OpenAI-compatible server.",
    )
    parser.add_argument(
        "--vl-api-key",
        default=os.getenv("LLM_API_KEY", ""),
        help="Optional Bearer token for the VL API.",
    )
    parser.add_argument(
        "--vl-timeout",
        type=int,
        default=int(os.getenv("PREPARE_VL_TIMEOUT", "6000")),
        help="Per-page HTTP timeout for VL (seconds).",
    )
    parser.add_argument(
        "--vl-zoom",
        type=float,
        default=float(os.getenv("PREPARE_VL_ZOOM", "3")),
        help="PDF page render zoom before sending image to VL (higher = sharper, slower). Default 3 for dense book text.",
    )
    parser.add_argument(
        "--vl-two-pass",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("PREPARE_VL_TWO_PASS", "true").lower() in {"1", "true", "yes", "on"},
        help=(
            "Два запроса VL на страницу: сначала только текст, затем блок [ИЛЛЮСТРАЦИИ] (склейка). "
            "Снимает усечение текста из-за одного лимита max_tokens. "
            "--no-vl-two-pass — один проход (только текст, без иллюстраций)."
        ),
    )
    parser.add_argument(
        "--fallback-tesseract",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If VL fails or returns empty, run Tesseract on that page.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess every PDF: do not skip up-to-date TXT; clear per-PDF page cache and re-run VL/OCR.",
    )
    parser.add_argument(
        "--no-page-cache",
        action="store_true",
        help="Disable read/write of per-page cache (out_dir/.prepare_page_cache/...).",
    )
    parser.add_argument(
        "--no-skip-up-to-date-output",
        action="store_true",
        help="Rebuild mapped TXT even when it is newer than the source PDF (still uses page cache unless --no-page-cache).",
    )
    parser.add_argument(
        "--legacy-single-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Один поток текста страницы (тело + [ИЛЛЮСТРАЦИИ] вместе), как раньше. "
            "--no-legacy-single-stream — секции ### RAG_PAGE_BODY / ### RAG_PAGE_ILLUSTRATIONS для RAG."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    if args.page_engine == "vl" and not (args.vl_api_url or "").strip():
        parser.error(
            "For --page-engine vl set --vl-api-url or PREPARE_VL_API_URL / LLM_API_URL "
            "(e.g. http://host.docker.internal:1234/v1/chat/completions in Docker)"
        )

    tesseract_lang = args.tesseract_lang or os.getenv("TESSERACT_LANG", "rus+eng")
    os.environ["TESSERACT_LANG"] = tesseract_lang
    os.environ["PDF_OCR_LANG"] = tesseract_lang

    raw_dir = Path(args.raw_dir)
    toc_dir = Path(args.toc_dir)
    out_dir = Path(args.out_dir)
    logger.info("Starting prepare_docs job")
    logger.info("raw_dir=%s", raw_dir)
    logger.info("toc_dir=%s", toc_dir)
    logger.info("out_dir=%s", out_dir)
    logger.info("workers=%s", args.workers)
    logger.info("tesseract_lang=%s", tesseract_lang)
    logger.info("pdf_text_first=%s", args.pdf_text_first)
    logger.info("page_engine=%s", args.page_engine)
    logger.info("vl_api_url=%s", args.vl_api_url or "(empty)")
    logger.info("vl_model=%s", args.vl_model)
    logger.info("vl_two_pass=%s", args.vl_two_pass)

    total, generated = prepare_docs(
        raw_dir=raw_dir,
        toc_dir=toc_dir,
        out_dir=out_dir,
        min_chars=args.min_chars,
        min_alpha_ratio=args.min_alpha_ratio,
        workers=args.workers,
        tesseract_lang=tesseract_lang,
        pdf_text_first=args.pdf_text_first,
        page_engine=args.page_engine,
        vl_api_url=(args.vl_api_url or "").strip() or None,
        vl_model=args.vl_model,
        vl_api_key=args.vl_api_key or "",
        vl_timeout=args.vl_timeout,
        vl_zoom=args.vl_zoom,
        fallback_tesseract=args.fallback_tesseract,
        vl_two_pass=args.vl_two_pass,
        force=args.force,
        no_page_cache=args.no_page_cache,
        skip_up_to_date_output=not args.no_skip_up_to_date_output,
        legacy_single_stream=args.legacy_single_stream,
    )
    logger.info("Processed PDFs: %s", total)
    logger.info("Generated TXT: %s", generated)
    logger.info("Quality filter: min_chars=%s min_alpha_ratio=%.2f", args.min_chars, args.min_alpha_ratio)
    logger.info("Output dir: %s", out_dir)


if __name__ == "__main__":
    main()
