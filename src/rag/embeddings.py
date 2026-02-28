from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts into dense vectors."""
    model = _get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Encode a single query string."""
    return embed_texts([query])[0]
