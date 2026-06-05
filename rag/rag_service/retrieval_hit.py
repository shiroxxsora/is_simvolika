"""Единая модель строки результата retrieval для логов, промптов и API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalHit:
    doc_name: str
    content: str
    source_link: str
    distance: float
    chunk_index: int | None = None
    fragment_kind: str = "text"
