from __future__ import annotations

from rag_service.report_docx_tpl import render_conclusion_docx
from rag_service.report_models import ConclusionReportData


def generate_conclusion_docx(data: ConclusionReportData) -> bytes:
    """Генерация DOCX: шаблон conclusion_base.docx с Jinja2, подстановка в docxtpl (сохраняет стили)."""
    return render_conclusion_docx(data)

