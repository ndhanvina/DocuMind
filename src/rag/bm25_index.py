from __future__ import annotations

import json
import re
from pathlib import Path

import structlog
from rank_bm25 import BM25L

from .config import settings
from .models import Chunk, ScoredChunk

log = structlog.get_logger()

_TOKENIZE_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKENIZE_RE.findall(text.lower())


class BM25Index:
    """Sparse BM25 retrieval index over chunks."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25L | None = None
        self._corpus: list[list[str]] = []

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25L(self._corpus)
        log.info("bm25_built", num_docs=len(chunks))

    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built. Call build() first.")

        k = top_k or settings.bm25_top_k
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [
            ScoredChunk(chunk=self._chunks[idx], score=float(score), origin="bm25")
            for idx, score in ranked
            if score > 0
        ]

    def save(self, path: Path | None = None) -> None:
        save_path = path or settings.bm25_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.model_dump() for c in self._chunks],
        }
        save_path.write_text(json.dumps(data), encoding="utf-8")
        log.info("bm25_saved", path=str(save_path))

    def load(self, path: Path | None = None) -> None:
        load_path = path or settings.bm25_path
        data = json.loads(load_path.read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in data["chunks"]]
        self.build(chunks)
        log.info("bm25_loaded", path=str(load_path), num_docs=len(chunks))
