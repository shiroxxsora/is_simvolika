from typing import List

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class RAGMatch(BaseModel):
    object_name: str
    description: str
    source_link: str
    score: float
    fragment_kind: str = "text"
    chunk_index: int | None = None


class ChatResponse(BaseModel):
    answer: str
    context_files: List[str]
    matches: List[RAGMatch]
    index_ready: bool = True
    index_chunk_count: int = 0
    index_error: str | None = None


class ImageAnalyzeRequest(BaseModel):
    image_base64: str
    mime_type: str = "image/jpeg"
    user_hint: str = ""
    specialist_full_name: str = ""
    specialist_education: str = ""
    specialist_qualification: str = ""
    specialist_additional_training: str = ""
    specialist_position: str = ""
    specialist_research_interests: str = ""
    specialist_experience_years: str = ""
    report_basis: str = ""


class ImageAnalyzeResponse(BaseModel):
    image_description: str
    matches: List[RAGMatch]
    classification: str = ""
    no_sources_summary: str = ""
    report_docx_base64: str = ""
    report_file_name: str = ""
    index_ready: bool = True
    index_chunk_count: int = 0
    index_error: str | None = None


class ReindexResponse(BaseModel):
    status: str
    message: str

