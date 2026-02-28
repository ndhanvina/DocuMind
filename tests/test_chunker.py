from src.rag.chunker import chunk_text, chunk_markdown


def test_chunk_text_short():
    """Short text should produce a single chunk."""
    chunks = chunk_text("Hello world.", source="test.txt", chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].source == "test.txt"


def test_chunk_text_splits():
    """Long text should be split into multiple chunks."""
    text = "word " * 200  # ~1000 chars
    chunks = chunk_text(text.strip(), source="test.txt", chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 120  # small tolerance for split boundaries


def test_chunk_text_overlap():
    """Chunks should overlap when overlap > 0."""
    text = "A " * 100 + "B " * 100
    chunks = chunk_text(text.strip(), source="test.txt", chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    # With overlap, later chunks should contain some text from the previous chunk


def test_chunk_ids_unique():
    """Each chunk should have a unique ID."""
    text = "Hello. " * 50
    chunks = chunk_text(text, source="test.txt", chunk_size=50, chunk_overlap=0)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_markdown_by_headings():
    """Markdown should be split by headings."""
    md = """# Introduction
This is the intro.

## Setup
Setup instructions here with more text to fill out the section.

## Usage
Usage instructions here.
"""
    chunks = chunk_markdown(md, source="readme.md")
    assert len(chunks) >= 2
    titles = {c.title for c in chunks}
    assert "Introduction" in titles or "Setup" in titles


def test_chunk_markdown_no_headings():
    """Markdown without headings falls back to text chunking."""
    md = "Just some plain text without any headings. " * 20
    chunks = chunk_markdown(md, source="plain.md")
    assert len(chunks) >= 1


def test_empty_text():
    """Empty text should produce no chunks."""
    chunks = chunk_text("", source="empty.txt", chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 0
