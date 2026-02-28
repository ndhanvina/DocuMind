from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of a document with source metadata."""

    chunk_id: str
    text: str
    source: str  # file path or URL
    title: str = ""
    page: int | None = None
    start_char: int = 0
    end_char: int = 0

    def citation_label(self) -> str:
        label = self.title or self.source
        if self.page is not None:
            label += f", p.{self.page}"
        return label


class ScoredChunk(BaseModel):
    """A chunk with a retrieval / reranking score."""

    chunk: Chunk
    score: float
    origin: str = ""  # "bm25", "vector", "rrf", "reranker"


class Citation(BaseModel):
    """A single citation reference inside a generated answer."""

    ref_id: int
    source: str
    title: str
    quote: str = ""


class RAGRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGResponse(BaseModel):
    answer: str
    citations: list[Citation]
    chunks_used: list[ScoredChunk]
    query: str
