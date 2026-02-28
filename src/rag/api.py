from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from .config import settings
from .models import RAGRequest, RAGResponse
from .pipeline import RAGPipeline

log = structlog.get_logger()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup: try to load existing indexes, or ingest if docs exist
    if settings.bm25_path.exists():
        log.info("loading_existing_indexes")
        pipeline.load_indexes()
    elif settings.docs_dir.exists() and any(settings.docs_dir.iterdir()):
        log.info("ingesting_docs_on_startup", dir=str(settings.docs_dir))
        count = pipeline.ingest()
        log.info("startup_ingest_complete", chunks=count)
    else:
        log.warning("no_docs_found", dir=str(settings.docs_dir))
    yield


app = FastAPI(
    title="DocuMind",
    description="DocuMind — Production RAG with hybrid retrieval, cross-encoder reranking, and citations",
    version="0.1.0",
    lifespan=lifespan,
)


def _validate_docs_path(directory: Path) -> Path:
    """Ensure the path is within the allowed docs directory."""
    allowed_root = settings.docs_dir.resolve()
    resolved = directory.resolve()
    if not (resolved == allowed_root or str(resolved).startswith(str(allowed_root) + "/")):
        # On Windows also check backslash
        if not str(resolved).startswith(str(allowed_root) + "\\"):
            raise HTTPException(403, "Access denied: path outside allowed docs directory")
    return resolved


@app.get("/health")
def health() -> dict[str, str]:
    status = "ready" if pipeline.is_ready else "not_ready"
    return {"status": status, "chunks": str(pipeline.chunk_count)}


class IngestRequest(BaseModel):
    docs_dir: str = ""


class IngestResponse(BaseModel):
    chunks_indexed: int


@app.post("/ingest", response_model=IngestResponse)
def ingest_docs(req: IngestRequest) -> IngestResponse:
    """Ingest / re-ingest documents from a directory."""
    directory = Path(req.docs_dir) if req.docs_dir else settings.docs_dir
    directory = _validate_docs_path(directory)
    if not directory.exists():
        raise HTTPException(404, f"Directory not found: {directory}")
    count = pipeline.ingest(docs_dir=directory)
    return IngestResponse(chunks_indexed=count)


@app.post("/query", response_model=RAGResponse)
def query_docs(req: RAGRequest) -> RAGResponse:
    """Ask a question against the indexed documents."""
    if not pipeline.is_ready:
        raise HTTPException(503, "Pipeline not ready. Ingest documents first.")
    return pipeline.query(req.query, top_k=req.top_k)


def _sanitize_filename(filename: str) -> str:
    """Strip path separators to prevent directory traversal."""
    return Path(filename).name


@app.post("/upload")
async def upload_file(file: UploadFile) -> IngestResponse:
    """Upload a single document file for ingestion."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    safe_name = _sanitize_filename(file.filename)
    if not safe_name:
        raise HTTPException(400, "Invalid filename")

    # Read with size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_UPLOAD_BYTES // (1024*1024)} MB")

    upload_dir = settings.docs_dir / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / safe_name

    dest.write_bytes(content)
    log.info("file_uploaded", path=str(dest), size=len(content))

    count = pipeline.ingest()
    return IngestResponse(chunks_indexed=count)


def create_app() -> FastAPI:
    """Factory for testing."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.rag.api:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
