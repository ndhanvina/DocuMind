from __future__ import annotations

import structlog

from .bm25_index import BM25Index
from .config import settings
from .models import ScoredChunk
from .vector_store import VectorStore

log = structlog.get_logger()


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    k: int | None = None,
) -> list[ScoredChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score for document d = sum over all lists of 1 / (k + rank_in_list)
    """
    rrf_k = k or settings.rrf_k
    chunk_scores: dict[str, float] = {}
    chunk_map: dict[str, ScoredChunk] = {}

    for results in result_lists:
        for rank, sc in enumerate(results):
            cid = sc.chunk.chunk_id
            chunk_scores[cid] = chunk_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
            # Keep the highest-scored version
            if cid not in chunk_map or sc.score > chunk_map[cid].score:
                chunk_map[cid] = sc

    ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        ScoredChunk(
            chunk=chunk_map[cid].chunk,
            score=score,
            origin="rrf",
        )
        for cid, score in ranked
    ]


class HybridRetriever:
    """Combines BM25 sparse retrieval with dense vector search via RRF."""

    def __init__(self, bm25: BM25Index, vector: VectorStore) -> None:
        self._bm25 = bm25
        self._vector = vector

    def retrieve(
        self,
        query: str,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
        final_top_k: int | None = None,
    ) -> list[ScoredChunk]:
        bm25_results = self._bm25.search(query, top_k=bm25_top_k)
        vector_results = self._vector.search(query, top_k=vector_top_k)

        log.debug(
            "hybrid_retrieval",
            bm25_hits=len(bm25_results),
            vector_hits=len(vector_results),
        )

        fused = reciprocal_rank_fusion([bm25_results, vector_results])

        k = final_top_k or (settings.bm25_top_k + settings.vector_top_k)
        return fused[:k]
