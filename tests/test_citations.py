from src.rag.citations import (
    build_citation_map,
    citation_coverage,
    extract_citation_ids,
    format_citation_block,
    validate_citations,
)
from src.rag.models import Chunk, Citation, ScoredChunk


def test_extract_citation_ids():
    text = "This is true [1] and also [3]. See [1] again."
    ids = extract_citation_ids(text)
    assert ids == [1, 3, 1]


def test_extract_no_citations():
    assert extract_citation_ids("No citations here.") == []


def test_build_citation_map():
    chunks = [
        ScoredChunk(
            chunk=Chunk(chunk_id="a", text="text a", source="doc.md", title="Doc A"),
            score=1.0,
        ),
        ScoredChunk(
            chunk=Chunk(chunk_id="b", text="text b", source="doc2.md", title="Doc B"),
            score=0.9,
        ),
    ]
    cmap = build_citation_map(chunks)
    assert 1 in cmap
    assert 2 in cmap
    assert cmap[1].source == "doc.md"
    assert cmap[2].title == "Doc B"


def test_validate_citations_removes_invalid():
    cmap = {
        1: Citation(ref_id=1, source="a.md", title="A"),
        2: Citation(ref_id=2, source="b.md", title="B"),
    }
    answer = "Claim [1] and wrong [5] and [2]."
    cleaned, citations = validate_citations(answer, cmap)
    assert "[5]" not in cleaned
    assert len(citations) == 2
    assert citations[0].ref_id == 1


def test_validate_citations_deduplicates():
    cmap = {1: Citation(ref_id=1, source="a.md", title="A")}
    answer = "See [1] and again [1]."
    _, citations = validate_citations(answer, cmap)
    assert len(citations) == 1


def test_format_citation_block():
    citations = [
        Citation(ref_id=1, source="doc.md", title="Document A"),
        Citation(ref_id=2, source="doc2.md", title="Document B"),
    ]
    block = format_citation_block(citations)
    assert "[1] Document A" in block
    assert "[2] Document B" in block


def test_format_empty_citations():
    assert format_citation_block([]) == ""


def test_citation_coverage_full():
    answer = "Point [1] and [2] and [3]."
    assert citation_coverage(answer, 3) == 1.0


def test_citation_coverage_partial():
    answer = "Only [1]."
    coverage = citation_coverage(answer, 3)
    assert 0.3 < coverage < 0.4  # 1/3


def test_citation_coverage_zero_chunks():
    assert citation_coverage("anything", 0) == 1.0
