from __future__ import annotations

import structlog
from google import genai
from google.genai import types

from .citations import build_citation_map, validate_citations
from .config import settings
from .models import Citation, ScoredChunk

log = structlog.get_logger()

SYSTEM_PROMPT = """\
You are a precise, helpful assistant that answers questions using ONLY the \
provided reference documents. Follow these rules strictly:

1. Base your answer ONLY on the provided references. Do not use prior knowledge.
2. Cite every claim using [N] notation, where N is the reference number.
3. If the references do not contain enough information, say so explicitly.
4. Every paragraph must include at least one citation.
5. Be concise and direct. Do not repeat the question.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _build_context_block(chunks: list[ScoredChunk]) -> str:
    lines: list[str] = []
    for i, sc in enumerate(chunks, start=1):
        source = sc.chunk.title or sc.chunk.source
        lines.append(f"[{i}] (Source: {source})\n{sc.chunk.text}\n")
    return "\n".join(lines)


def generate(
    query: str,
    chunks: list[ScoredChunk],
    model: str | None = None,
    temperature: float = 0.1,
) -> tuple[str, list[Citation]]:
    """Generate an answer with enforced citations."""
    if not chunks:
        return "I don't have enough information to answer this question.", []

    llm_model = model or settings.llm_model
    context = _build_context_block(chunks)
    citation_map = build_citation_map(chunks)

    client = _get_client()

    response = client.models.generate_content(
        model=llm_model,
        contents=f"References:\n{context}\n\nQuestion: {query}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        ),
    )

    raw_answer = response.text or ""

    answer, citations = validate_citations(raw_answer, citation_map)

    # If the model produced no citations at all, flag it
    if not citations and chunks:
        log.warning("no_citations_in_answer", query=query)
        answer += (
            "\n\n*Note: The model failed to produce inline citations. "
            "The answer is based on the following sources:*"
        )
        citations = list(citation_map.values())

    log.info("generated", model=llm_model, citations=len(citations))
    return answer, citations
