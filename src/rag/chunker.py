from __future__ import annotations

import hashlib
import re

from .config import settings
from .models import Chunk

# Split boundaries ordered from strongest to weakest
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(text: str, sep: str) -> list[str]:
    if sep == "":
        return list(text)
    return text.split(sep)


def _recursive_split(
    text: str,
    max_size: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text trying the strongest separator first."""
    if len(text) <= max_size:
        return [text]

    sep = separators[0]
    remaining_seps = separators[1:] if len(separators) > 1 else [""]
    parts = _split_text(text, sep)

    chunks: list[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If a single part exceeds max_size, split it further
            if len(part) > max_size:
                chunks.extend(_recursive_split(part, max_size, remaining_seps))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


import uuid


def _make_chunk_id(source: str, index: int) -> str:
    return hashlib.sha256(f"{source}:{index}:{uuid.uuid4().hex}".encode()).hexdigest()[:16]


def chunk_text(
    text: str,
    source: str,
    title: str = "",
    page: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks with metadata."""
    size = chunk_size if chunk_size is not None else settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    raw_chunks = _recursive_split(text, size, _SEPARATORS)

    # Apply overlap by prepending tail of previous chunk
    chunks: list[Chunk] = []
    offset = 0
    for i, raw in enumerate(raw_chunks):
        raw = raw.strip()
        if not raw:
            offset += len(raw_chunks[i]) if i < len(raw_chunks) else 0
            continue

        # Overlap: prepend last `overlap` chars of previous chunk
        if i > 0 and overlap > 0 and chunks:
            prev_text = raw_chunks[i - 1].strip()
            overlap_text = prev_text[-overlap:] if len(prev_text) > overlap else prev_text
            raw = overlap_text + " " + raw

        start = max(0, offset - overlap) if i > 0 else 0
        chunk = Chunk(
            chunk_id=_make_chunk_id(source, i),
            text=raw,
            source=source,
            title=title,
            page=page,
            start_char=start,
            end_char=start + len(raw),
        )
        chunks.append(chunk)
        offset += len(raw_chunks[i])

    return chunks


def chunk_markdown(text: str, source: str) -> list[Chunk]:
    """Chunk markdown by headings, then by size within each section."""
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    sections: list[tuple[str, str]] = []

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return chunk_text(text, source=source, title=source)

    # Text before first heading
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(2).strip(), text[m.start() : end]))

    all_chunks: list[Chunk] = []
    for title, section_text in sections:
        all_chunks.extend(chunk_text(section_text, source=source, title=title))

    return all_chunks
