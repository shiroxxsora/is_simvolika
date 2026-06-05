"""
Пересобрать templates/conclusion_jinja_smoke_test.docx из текущего
templates/conclusion_base.docx (стили/шрифты в базе — переносятся в дымовой рендер).

Не трогает conclusion_base.docx.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docx import Document  # noqa: E402

from rag_service.report_docx_from_template import (  # noqa: E402
    _insert_photo_after_marker,
    _remove_all_inline_pictures_from_document,
)
from rag_service.report_models import ReportPhoto  # noqa: E402

try:
    from docxtpl import DocxTemplate  # noqa: WPS433
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install docxtpl: pip install -r requirements.txt (rag/)") from e

BASE = ROOT / "templates" / "conclusion_base.docx"
OUT = ROOT / "templates" / "conclusion_jinja_smoke_test.docx"

CTX = {
    "report_title_line": "T",
    "basis": "b",
    "full_name": "f",
    "education": "e",
    "qualification": "q",
    "additional_training": "a",
    "position": "p",
    "research_interests": "r",
    "experience_years": "x",
    "materials_text": "m",
    "question": "Q",
    "source_lines": ["1. s"],
    "methods_text": "M",
    "research_paragraphs": ["R1", "R2"],
    "conclusion_text": "C",
    "note_text": "N",
}


def main() -> None:
    if not BASE.is_file():
        raise SystemExit(f"Not found: {BASE}")

    tpl = DocxTemplate(str(BASE))
    tpl.render(CTX)
    buf = io.BytesIO()
    tpl.save(buf)
    buf.seek(0)
    doc = Document(buf)
    _remove_all_inline_pictures_from_document(doc)
    imgp = ROOT.parent / "exemple" / "photo1.jpg"
    if imgp.is_file():
        _insert_photo_after_marker(
            doc, "(Фото № 1)", ReportPhoto(1, imgp.read_bytes(), "image/jpeg")
        )
    else:
        print("Skip photo: exemple/photo1.jpg not found")
    doc.save(str(OUT))
    print("Written:", OUT)


if __name__ == "__main__":
    main()
