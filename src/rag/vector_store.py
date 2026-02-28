from __future__ import annotations

import structlog
import chromadb

from .config import settings
from .embeddings import embed_query, embed_texts
from .models import Chunk, ScoredChunk

log = structlog.get_logger()

_COLLECTION_NAME = "documents"


class VectorStore:
    """Dense vector retrieval using ChromaDB."""

    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or str(settings.chroma_dir)
        self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunk_map: dict[str, Chunk] = {}

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 128) -> None:
        if not chunks:
            return

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c.chunk_id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {"source": c.source, "title": c.title, "page": c.page or 0}
                for c in batch
            ]
            embeddings = embed_texts(texts).tolist()

            self._collection.upsert(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            for c in batch:
                self._chunk_map[c.chunk_id] = c

        log.info("vectors_upserted", count=len(chunks))

    def search(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or settings.vector_top_k
        query_emb = embed_query(query).tolist()

        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        scored: list[ScoredChunk] = []
        if not results["ids"] or not results["ids"][0]:
            return scored

        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],  # type: ignore[index]
            results["metadatas"][0],  # type: ignore[index]
            results["distances"][0],  # type: ignore[index]
        ):
            # ChromaDB cosine distance → similarity
            similarity = 1.0 - float(dist)
            chunk = self._chunk_map.get(cid) or Chunk(
                chunk_id=cid,
                text=doc,
                source=meta.get("source", ""),
                title=meta.get("title", ""),
                page=meta.get("page") or None,
            )
            scored.append(ScoredChunk(chunk=chunk, score=similarity, origin="vector"))

        return scored

    @property
    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunk_map.clear()
