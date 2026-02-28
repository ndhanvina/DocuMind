from __future__ import annotations

import structlog
from sentence_transformers import CrossEncoder

from .config import settings
from .models import ScoredChunk

log = structlog.get_logger()

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(settings.reranker_model)
    return _model


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_k: int | None = None,
) -> list[ScoredChunk]:
    """Re-score candidates with a cross-encoder and return top-k."""
    if not candidates:
        return []

    k = top_k or settings.rerank_top_k
    model = _get_model()

    pairs = [[query, sc.chunk.text] for sc in candidates]
    scores = model.predict(pairs)

    reranked: list[ScoredChunk] = []
    for sc, score in zip(candidates, scores):
        reranked.append(
            ScoredChunk(
                chunk=sc.chunk,
                score=float(score),
                origin="reranker",
            )
        )

    reranked.sort(key=lambda x: x.score, reverse=True)

    log.debug("reranked", input_count=len(candidates), output_count=min(k, len(reranked)))
    return reranked[:k]
