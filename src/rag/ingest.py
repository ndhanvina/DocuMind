from __future__ import annotations

import structlog
from pathlib import Path

from .chunker import chunk_markdown, chunk_text
from .models import Chunk

log = structlog.get_logger()


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples."""
    import pymupdf

    pages: list[tuple[int, str]] = []
    with pymupdf.open(str(path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append((i + 1, text))
    return pages


def ingest_file(path: Path) -> list[Chunk]:
    """Load a single file and return chunks."""
    suffix = path.suffix.lower()
    source = path.as_posix()

    if suffix == ".pdf":
        pages = _read_pdf(path)
        chunks: list[Chunk] = []
        for page_num, text in pages:
            chunks.extend(chunk_text(text, source=source, title=path.stem, page=page_num))
        log.info("ingested_pdf", path=source, pages=len(pages), chunks=len(chunks))
        return chunks

    text = _read_text_file(path)
    if not text.strip():
        log.warning("empty_file", path=source)
        return []

    if suffix in (".md", ".markdown"):
        chunks = chunk_markdown(text, source=source)
    else:
        chunks = chunk_text(text, source=source, title=path.stem)

    log.info("ingested_file", path=source, chunks=len(chunks))
    return chunks


def ingest_directory(
    directory: Path,
    glob_pattern: str = "**/*",
    extensions: set[str] | None = None,
) -> list[Chunk]:
    """Recursively ingest all supported files from a directory."""
    if extensions is None:
        extensions = {".md", ".markdown", ".txt", ".pdf", ".rst"}

    all_chunks: list[Chunk] = []
    files = sorted(directory.glob(glob_pattern))

    for path in files:
        if not path.is_file():
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            all_chunks.extend(ingest_file(path))
        except Exception:
            log.exception("ingest_error", path=str(path))

    log.info("ingest_complete", directory=str(directory), total_chunks=len(all_chunks))
    return all_chunks
