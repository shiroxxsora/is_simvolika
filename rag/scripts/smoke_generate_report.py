from __future__ import annotations

import base64
from pathlib import Path

from rag_service.report_docx import generate_conclusion_docx
from rag_service.report_models import (
    ConclusionReportData,
    MaterialsInfo,
    ReportMeta,
    ReportPhoto,
    SpecialistInfo,
)


def main() -> None:
    # Minimal smoke: build docx with an example photo (optional)
    img_path = Path(__file__).resolve().parents[2] / "exemple" / "photo1.jpg"
    photos = []
    if img_path.is_file():
        photos = [ReportPhoto(number=1, image_bytes=img_path.read_bytes(), mime_type="image/jpeg")]

    data = ConclusionReportData(
        meta=ReportMeta(number="84", date_iso="06.04.2026", basis="запрос ... (smoke)"),
        specialist=SpecialistInfo(
            full_name="Некрасов Иван Сергеевич",
            education="высшее ...",
            qualification="юрист ...",
            additional_training="(smoke) ...",
            position="Старший преподаватель ...",
            research_interests="...",
            experience_years="15 лет",
        ),
        materials=MaterialsInfo(materials_text="(smoke)", person_text="(smoke)"),
        question="(smoke) вопрос ...",
        sources=["src1", "src2"],
        methods_text="(smoke) методы ...",
        research_paragraphs=["(smoke) исследование абзац 1", "(smoke) абзац 2"],
        conclusion_text="(smoke) вывод ...",
        note_text="(smoke) примечание ...",
        photos=photos,
    )

    docx = generate_conclusion_docx(data)
    out = Path(__file__).resolve().parents[2] / "debug-report.docx"
    out.write_bytes(docx)
    print("Wrote", out)
    print("base64_len", len(base64.b64encode(docx)))


if __name__ == "__main__":
    main()

