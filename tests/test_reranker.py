from src.rag.models import Chunk, ScoredChunk


def _make_scored(texts: list[str]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(chunk_id=f"c{i}", text=t, source="test.txt"),
            score=float(len(texts) - i),
            origin="rrf",
        )
        for i, t in enumerate(texts)
    ]


def test_rerank_reorders():
    """Cross-encoder should reorder candidates by relevance to query."""
    from src.rag.reranker import rerank

    candidates = _make_scored([
        "The capital of France is Paris",
        "Python is a programming language",
        "Cooking pasta requires boiling water",
    ])

    reranked = rerank("What is the capital of France?", candidates, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].origin == "reranker"
    # The France-related chunk should score highest
    assert "France" in reranked[0].chunk.text or "Paris" in reranked[0].chunk.text


def test_rerank_empty():
    from src.rag.reranker import rerank

    assert rerank("query", [], top_k=5) == []


def test_rerank_respects_top_k():
    from src.rag.reranker import rerank

    candidates = _make_scored(["a", "b", "c", "d", "e"])
    reranked = rerank("query", candidates, top_k=2)
    assert len(reranked) == 2
