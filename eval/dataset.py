from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel


class GoldenExample(BaseModel):
    """A single evaluation example with expected answer and source."""

    question: str
    expected_answer: str
    expected_sources: list[str] = []
    tags: list[str] = []


def load_golden_set(path: Path) -> list[GoldenExample]:
    """Load evaluation examples from a JSONL file."""
    examples: list[GoldenExample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(GoldenExample(**json.loads(line)))
    return examples


def save_golden_set(examples: list[GoldenExample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
