from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from src.rag.config import settings
from src.rag.pipeline import RAGPipeline

from .dataset import GoldenExample, load_golden_set
from .metrics import EvalScores, evaluate_example

log = structlog.get_logger()


@dataclass
class EvalReport:
    scores: list[EvalScores] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        return sum(s.faithfulness for s in self.scores) / len(self.scores) if self.scores else 0

    @property
    def avg_relevance(self) -> float:
        return sum(s.relevance for s in self.scores) / len(self.scores) if self.scores else 0

    @property
    def avg_citation_accuracy(self) -> float:
        return (
            sum(s.citation_accuracy for s in self.scores) / len(self.scores)
            if self.scores
            else 0
        )

    @property
    def avg_source_recall(self) -> float:
        return (
            sum(s.source_recall for s in self.scores) / len(self.scores) if self.scores else 0
        )

    def passed(self) -> bool:
        return (
            self.avg_faithfulness >= settings.eval_faithfulness_threshold
            and self.avg_relevance >= settings.eval_relevance_threshold
            and self.avg_citation_accuracy >= settings.eval_citation_threshold
        )

    def summary(self) -> dict[str, object]:
        return {
            "num_examples": len(self.scores),
            "avg_faithfulness": round(self.avg_faithfulness, 3),
            "avg_relevance": round(self.avg_relevance, 3),
            "avg_citation_accuracy": round(self.avg_citation_accuracy, 3),
            "avg_source_recall": round(self.avg_source_recall, 3),
            "thresholds": {
                "faithfulness": settings.eval_faithfulness_threshold,
                "relevance": settings.eval_relevance_threshold,
                "citation": settings.eval_citation_threshold,
            },
            "passed": self.passed(),
        }


def run_evaluation(
    pipeline: RAGPipeline,
    golden_path: Path | None = None,
    api_key: str | None = None,
) -> EvalReport:
    """Run the full eval pipeline against a golden dataset."""
    gpath = golden_path or settings.eval_golden_path
    key = api_key or settings.gemini_api_key

    golden = load_golden_set(gpath)
    if not golden:
        log.warning("empty_golden_set", path=str(gpath))
        return EvalReport()

    report = EvalReport()

    for example in golden:
        log.info("evaluating", question=example.question)
        try:
            response = pipeline.query(example.question)
            scores = evaluate_example(response, example, api_key=key)
            report.scores.append(scores)

            log.info(
                "eval_result",
                question=example.question[:60],
                faithfulness=scores.faithfulness,
                relevance=scores.relevance,
                citation=scores.citation_accuracy,
            )
        except Exception:
            log.exception("eval_example_failed", question=example.question[:60])

    return report


def main() -> None:
    """CLI entrypoint for evaluation."""
    pipe = RAGPipeline()

    if settings.bm25_path.exists():
        pipe.load_indexes()
    else:
        pipe.ingest()

    report = run_evaluation(pipe)
    summary = report.summary()

    print(json.dumps(summary, indent=2))

    if not report.passed():
        print("\nEVALUATION FAILED — below threshold", file=sys.stderr)
        sys.exit(1)

    print("\nEVALUATION PASSED")


if __name__ == "__main__":
    main()
