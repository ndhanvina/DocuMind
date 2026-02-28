from __future__ import annotations

import re

from google import genai
from google.genai import types
from pydantic import BaseModel

from src.rag.citations import extract_citation_ids
from src.rag.models import RAGResponse

from .dataset import GoldenExample


class EvalScores(BaseModel):
    """Per-example evaluation scores."""

    question: str
    faithfulness: float  # Is the answer grounded in the retrieved chunks?
    relevance: float  # Does the answer address the question?
    citation_accuracy: float  # Are citations valid and present?
    source_recall: float  # Did we retrieve the expected sources?


def _llm_judge(
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
) -> float:
    """Use an LLM to score on a 0-1 scale."""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an evaluation judge. Score the following on a scale of 0.0 to 1.0. "
                "Return ONLY a decimal number, nothing else."
            ),
            temperature=0.0,
        ),
    )
    raw = (response.text or "0").strip()
    # Extract first float from response and clamp to [0, 1]
    match = re.search(r"(\d+\.?\d*)", raw)
    score = float(match.group(1)) if match else 0.0
    return max(0.0, min(1.0, score))


def score_faithfulness(response: RAGResponse, api_key: str) -> float:
    """Score whether the answer is grounded in the provided chunks."""
    chunks_text = "\n---\n".join(sc.chunk.text for sc in response.chunks_used)
    prompt = (
        f"Given these reference passages:\n{chunks_text}\n\n"
        f"And this answer:\n{response.answer}\n\n"
        "Score how well the answer is grounded in the references (0.0 = not grounded, "
        "1.0 = fully grounded). Penalize claims not in the references."
    )
    return _llm_judge(prompt, api_key)


def score_relevance(response: RAGResponse, api_key: str) -> float:
    """Score whether the answer actually addresses the question."""
    prompt = (
        f"Question: {response.query}\n\n"
        f"Answer: {response.answer}\n\n"
        "Score how well the answer addresses the question (0.0 = completely irrelevant, "
        "1.0 = perfectly relevant and complete)."
    )
    return _llm_judge(prompt, api_key)


def score_citation_accuracy(response: RAGResponse) -> float:
    """Score citation quality: are [N] refs present and valid?"""
    if not response.chunks_used:
        return 1.0

    num_chunks = len(response.chunks_used)
    cited_ids = set(extract_citation_ids(response.answer))
    valid_ids = {i for i in cited_ids if 1 <= i <= num_chunks}

    if not cited_ids:
        return 0.0

    # Precision: are all cited refs pointing to real chunks?
    precision = len(valid_ids) / len(cited_ids) if cited_ids else 0.0

    # Sufficiency: did the model cite a reasonable number of sources?
    # Expect at least min(3, num_chunks) — not every chunk needs citing
    expected_min = min(3, num_chunks)
    sufficiency = min(1.0, len(valid_ids) / expected_min) if expected_min > 0 else 1.0

    return 0.8 * precision + 0.2 * sufficiency


def score_source_recall(response: RAGResponse, expected_sources: list[str]) -> float:
    """What fraction of expected sources appear in retrieved chunks?"""
    if not expected_sources:
        return 1.0

    # Normalize all paths to forward slashes for cross-platform matching
    retrieved_sources = {sc.chunk.source.replace("\\", "/") for sc in response.chunks_used}
    hits = sum(
        1
        for src in expected_sources
        if any(src.replace("\\", "/") in rs for rs in retrieved_sources)
    )
    return hits / len(expected_sources)


def evaluate_example(
    response: RAGResponse,
    golden: GoldenExample,
    api_key: str,
) -> EvalScores:
    """Run all metrics on a single example."""
    return EvalScores(
        question=golden.question,
        faithfulness=score_faithfulness(response, api_key),
        relevance=score_relevance(response, api_key),
        citation_accuracy=score_citation_accuracy(response),
        source_recall=score_source_recall(response, golden.expected_sources),
    )
