"""Генерация заключения через docxtpl (Jinja2 в conclusion_base.docx) + вставка фото."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from docx import Document
from docxtpl import DocxTemplate

from rag_service.report_models import ConclusionReportData
from rag_service.report_docx_from_template import (  # noqa: WPS300
    _insert_photo_after_marker,
    _remove_all_inline_pictures_from_document,
)

logger = logging.getLogger(__name__)


def _initials_name(full_name: str) -> str:
    parts = [p for p in (full_name or "").split() if p.strip()]
    if not parts:
        return ""
    surname = parts[0]
    first = parts[1] if len(parts) > 1 else ""
    patronymic = parts[2] if len(parts) > 2 else ""
    ini = []
    if first:
        ini.append(f"{first[0].upper()}.")
    if patronymic:
        ini.append(f"{patronymic[0].upper()}.")
    if ini:
        return f"{' '.join(ini)} {surname}"
    return surname


def render_conclusion_docx(data: ConclusionReportData) -> bytes:
    base = Path(__file__).resolve().parents[1] / "templates" / "conclusion_base.docx"
    if not base.is_file():
        raise FileNotFoundError(f"Base DOCX template not found: {base}")

    source_lines = [f"{i}. {s}" for i, s in enumerate(data.sources, start=1)]
    context = {
        "report_title_line": f"ЗАКЛЮЧЕНИЕ СПЕЦИАЛИСТА № {data.meta.number} от {data.meta.date_iso}",
        "basis": data.meta.basis,
        "full_name": data.specialist.full_name,
        "initials_name": _initials_name(data.specialist.full_name),
        "education": data.specialist.education,
        "qualification": data.specialist.qualification,
        "additional_training": data.specialist.additional_training,
        "position": data.specialist.position,
        "research_interests": data.specialist.research_interests,
        "experience_years": data.specialist.experience_years,
        "materials_text": data.materials.materials_text,
        "question": data.question,
        "source_lines": source_lines,
        "methods_text": data.methods_text,
        "research_paragraphs": list(data.research_paragraphs),
        "conclusion_text": data.conclusion_text,
        "note_text": data.note_text,
    }

    tpl = DocxTemplate(str(base))
    try:
        tpl.render(context)
    except Exception:  # noqa: BLE001
        logger.exception("docxtpl render failed (template/variables?)")
        raise

    out_buf = io.BytesIO()
    tpl.save(out_buf)
    out_buf.seek(0)
    doc = Document(out_buf)
    n_stripped = _remove_all_inline_pictures_from_document(doc)
    if n_stripped:
        logger.debug("Stripped %s images after jinja (template safety)", n_stripped)

    for ph in data.photos:
        _insert_photo_after_marker(doc, f"(Фото № {ph.number})", ph)

    final = io.BytesIO()
    doc.save(final)
    return final.getvalue()
