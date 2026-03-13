"""
MedQA Dataset Downloader

Downloads the MedQA (USMLE-style) dataset from Hugging Face for
AXIOM clinical reasoning validation.

MedQA contains ~12K multiple-choice medical questions (4 options each).
We extract factual triples from both the questions and answer explanations
to build a complementary knowledge base alongside PubMedQA.

Usage:
    python -m scripts.download_medqa [--max-questions 3000]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom_hdc.config import data as data_cfg
from axiom_hdc.utils import setup_logging

logger = logging.getLogger("axiom.scripts.download_medqa")


def download_medqa(max_questions: int = 3000) -> Path:
    """
    Download MedQA and save questions + answer options for triple extraction.

    Args:
        max_questions: Maximum number of questions to save.

    Returns:
        Path to the saved JSON file.
    """
    setup_logging()

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' library is required. "
            "Install with: pip install datasets"
        )
        sys.exit(1)

    data_cfg.ensure_dirs()
    output_dir = data_cfg.dataset_dir / "medqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading MedQA from Hugging Face …")
    logger.info("=" * 60)

    # MedQA USMLE-style 4-option questions (parquet format, no auth needed)
    ds = load_dataset(
        "GBaker/MedQA-USMLE-4-options",
        split="train",
        trust_remote_code=False,
    )

    logger.info("Downloaded %d entries from MedQA", len(ds))

    questions: list[dict] = []
    for i, row in enumerate(ds):
        if i >= max_questions:
            break

        question_text = row.get("question", "")
        options = row.get("options", {})
        answer = row.get("answer", "")
        answer_idx = row.get("answer_idx", "")
        metamap = row.get("metamap_phrases", [])

        if not question_text.strip():
            continue

        # Combine options into a list for triple extraction
        choices = [f"{k}: {v}" for k, v in options.items()] if isinstance(options, dict) else []

        questions.append({
            "id": f"medqa_{i}",
            "question": question_text,
            "choices": choices,
            "answer": [answer] if isinstance(answer, str) else answer,
            "context": " ".join(metamap) if isinstance(metamap, list) else "",
        })

    output_path = output_dir / "medqa_questions.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d questions → %s", len(questions), output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MedQA dataset")
    parser.add_argument(
        "--max-questions",
        type=int,
        default=3000,
        help="Maximum questions to download (default: 3000)",
    )
    args = parser.parse_args()
    download_medqa(max_questions=args.max_questions)
