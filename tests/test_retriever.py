import pytest

from src.rag.bm25_index import BM25Index
from src.rag.hybrid_retriever import reciprocal_rank_fusion
from src.rag.models import Chunk, ScoredChunk


def _make_chunks(texts: list[str]) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"c{i}", text=t, source="test.txt", title="Test")
        for i, t in enumerate(texts)
    ]


class TestBM25:
    def test_build_and_search(self):
        chunks = _make_chunks([
            "Python is a programming language",
            "Java is also a programming language",
            "Cooking recipes for pasta",
        ])
        idx = BM25Index()
        idx.build(chunks)

        results = idx.search("python programming", top_k=2)
        assert len(results) >= 1
        assert results[0].chunk.chunk_id == "c0"
        assert results[0].origin == "bm25"

    def test_search_no_match(self):
        chunks = _make_chunks(["alpha beta gamma"])
        idx = BM25Index()
        idx.build(chunks)

        results = idx.search("xylophone", top_k=5)
        assert len(results) == 0

    def test_empty_index_raises(self):
        idx = BM25Index()
        with pytest.raises(RuntimeError):
            idx.search("anything")

    def test_save_and_load(self, tmp_path):
        chunks = _make_chunks(["hello world", "foo bar baz"])
        idx = BM25Index()
        idx.build(chunks)

        path = tmp_path / "bm25.json"
        idx.save(path)

        idx2 = BM25Index()
        idx2.load(path)
        results = idx2.search("hello", top_k=1)
        assert len(results) == 1
        assert results[0].chunk.text == "hello world"


class TestRRF:
    def test_fusion_merges_lists(self):
        c1 = Chunk(chunk_id="a", text="doc a", source="s")
        c2 = Chunk(chunk_id="b", text="doc b", source="s")
        c3 = Chunk(chunk_id="c", text="doc c", source="s")

        list1 = [
            ScoredChunk(chunk=c1, score=1.0, origin="bm25"),
            ScoredChunk(chunk=c2, score=0.8, origin="bm25"),
        ]
        list2 = [
            ScoredChunk(chunk=c2, score=0.9, origin="vector"),
            ScoredChunk(chunk=c3, score=0.7, origin="vector"),
        ]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [sc.chunk.chunk_id for sc in fused]

        # c2 appears in both lists, should rank high
        assert "b" in ids
        assert len(fused) == 3

    def test_fusion_empty_lists(self):
        fused = reciprocal_rank_fusion([[], []])
        assert fused == []

    def test_fusion_single_list(self):
        c = Chunk(chunk_id="x", text="only", source="s")
        results = [ScoredChunk(chunk=c, score=1.0, origin="bm25")]
        fused = reciprocal_rank_fusion([results], k=60)
        assert len(fused) == 1
