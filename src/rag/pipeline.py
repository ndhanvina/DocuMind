from __future__ import annotations

from pathlib import Path

import structlog

from .bm25_index import BM25Index
from .config import settings
from .generator import generate
from .hybrid_retriever import HybridRetriever
from .ingest import ingest_directory
from .models import Chunk, RAGResponse
from .reranker import rerank
from .vector_store import VectorStore

log = structlog.get_logger()


class RAGPipeline:
    """End-to-end pipeline: ingest → retrieve → rerank → generate."""

    def __init__(self) -> None:
        self._bm25 = BM25Index()
        self._vector = VectorStore()
        self._retriever = HybridRetriever(self._bm25, self._vector)
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def chunk_count(self) -> int:
        return self._vector.count

    def ingest(
        self,
        docs_dir: Path | None = None,
        extensions: set[str] | None = None,
    ) -> int:
        """Ingest documents from disk and build both indexes."""
        directory = docs_dir or settings.docs_dir
        chunks = ingest_directory(directory, extensions=extensions)
        return self.index_chunks(chunks)

    def index_chunks(self, chunks: list[Chunk]) -> int:
        """Build indexes from pre-loaded chunks."""
        if not chunks:
            log.warning("no_chunks_to_index")
            return 0

        self._bm25.build(chunks)
        self._bm25.save()
        self._vector.add_chunks(chunks)
        self._ready = True

        log.info("pipeline_indexed", total_chunks=len(chunks))
        return len(chunks)

    def load_indexes(self) -> None:
        """Load pre-built indexes from disk."""
        self._bm25.load()
        self._ready = True
        log.info("pipeline_loaded", vector_count=self._vector.count)

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Run the full RAG pipeline on a question."""
        if not self._ready:
            raise RuntimeError("Pipeline not ready. Call ingest() or load_indexes() first.")

        # Step 1: Hybrid retrieval (BM25 + vector → RRF fusion)
        candidates = self._retriever.retrieve(question)

        # Step 2: Cross-encoder reranking
        reranked = rerank(question, candidates, top_k=top_k)

        # Step 3: Generate answer with citation enforcement
        answer, citations = generate(question, reranked)

        return RAGResponse(
            answer=answer,
            citations=citations,
            chunks_used=reranked,
            query=question,
        )
