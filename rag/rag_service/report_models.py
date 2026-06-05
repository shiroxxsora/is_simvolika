from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReportPhoto:
    number: int
    image_bytes: bytes
    mime_type: str = "image/jpeg"


@dataclass(frozen=True)
class SpecialistInfo:
    full_name: str
    education: str
    qualification: str
    additional_training: str
    position: str
    research_interests: str
    experience_years: str


@dataclass(frozen=True)
class ReportMeta:
    number: str
    date_iso: str  # dd.mm.yyyy (как в примере)
    basis: str


@dataclass(frozen=True)
class MaterialsInfo:
    materials_text: str
    person_text: str


@dataclass(frozen=True)
class ConclusionReportData:
    meta: ReportMeta
    specialist: SpecialistInfo
    materials: MaterialsInfo
    question: str
    sources: list[str]
    methods_text: str
    research_paragraphs: list[str]
    conclusion_text: str
    note_text: str
    photos: list[ReportPhoto]

