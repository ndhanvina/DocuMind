from __future__ import annotations

import re

import structlog

from .models import Citation, ScoredChunk

log = structlog.get_logger()

_CITE_SINGLE = re.compile(r"\[(\d+)\]")
_CITE_GROUP = re.compile(r"\[([\d,\s]+)\]")


def extract_citation_ids(text: str) -> list[int]:
    """Pull all citation IDs from generated text.

    Handles both [N] and [1, 2, 3] formats.
    """
    ids: list[int] = []
    for m in _CITE_GROUP.finditer(text):
        for num in re.findall(r"\d+", m.group(1)):
            ids.append(int(num))
    return ids


def build_citation_map(chunks: list[ScoredChunk]) -> dict[int, Citation]:
    """Build a 1-indexed citation map from the reranked chunks."""
    cmap: dict[int, Citation] = {}
    for i, sc in enumerate(chunks, start=1):
        cmap[i] = Citation(
            ref_id=i,
            source=sc.chunk.source,
            title=sc.chunk.title,
            quote=sc.chunk.text[:200],
        )
    return cmap


def validate_citations(
    answer: str,
    citation_map: dict[int, Citation],
) -> tuple[str, list[Citation]]:
    """Validate that all citation IDs in the answer exist in the map.

    Removes invalid citations and returns the cleaned answer + used citations.
    """
    used_ids = extract_citation_ids(answer)
    valid_ids: list[int] = []
    invalid_ids: list[int] = []

    for cid in used_ids:
        if cid in citation_map:
            valid_ids.append(cid)
        else:
            invalid_ids.append(cid)

    if invalid_ids:
        log.warning("invalid_citations_removed", ids=invalid_ids)
        for cid in set(invalid_ids):
            answer = answer.replace(f"[{cid}]", "")

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique_ids: list[int] = []
    for cid in valid_ids:
        if cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)

    citations = [citation_map[cid] for cid in unique_ids]
    return answer.strip(), citations


def format_citation_block(citations: list[Citation]) -> str:
    """Render citations as a reference footer."""
    if not citations:
        return ""
    lines = ["\n\n---\n**Sources:**"]
    for c in citations:
        lines.append(f"[{c.ref_id}] {c.title or c.source}")
    return "\n".join(lines)


def citation_coverage(answer: str, num_chunks: int) -> float:
    """Fraction of provided chunks that were actually cited."""
    if num_chunks == 0:
        return 1.0
    cited = set(extract_citation_ids(answer))
    expected = set(range(1, num_chunks + 1))
    return len(cited & expected) / len(expected)
