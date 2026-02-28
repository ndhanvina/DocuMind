#!/usr/bin/env python3
"""Ingest documents from the docs directory into the RAG pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.pipeline import RAGPipeline
from src.rag.config import settings


def main() -> None:
    docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else settings.docs_dir

    if not docs_dir.exists():
        print(f"Error: {docs_dir} does not exist")
        sys.exit(1)

    pipe = RAGPipeline()
    count = pipe.ingest(docs_dir=docs_dir)
    print(f"Ingested {count} chunks from {docs_dir}")


if __name__ == "__main__":
    main()
