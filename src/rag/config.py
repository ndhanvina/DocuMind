from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env into os.environ so non-prefixed vars (e.g. GEMINI_API_KEY) are available
load_dotenv()


class Settings(BaseSettings):
    model_config = {
        "env_prefix": "RAG_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Models
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gemini-2.5-flash"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    bm25_top_k: int = 25
    vector_top_k: int = 25
    rerank_top_k: int = 5
    rrf_k: int = 60

    # Paths
    docs_dir: Path = Path("./docs")
    data_dir: Path = Path("./data")
    chroma_dir: Path = Path("./data/chroma")
    bm25_path: Path = Path("./data/bm25_index.json")

    # Evaluation thresholds
    eval_golden_path: Path = Path("./eval/golden.jsonl")
    eval_faithfulness_threshold: float = 0.7
    eval_relevance_threshold: float = 0.7
    eval_citation_threshold: float = 0.9

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def gemini_api_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "")


settings = Settings()
