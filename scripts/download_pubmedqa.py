"""
PubMedQA Dataset Downloader

Downloads the PubMedQA dataset from Hugging Face and extracts
medical knowledge triples via NER for AXIOM real-world validation.

PubMedQA contains ~211K QA pairs from PubMed abstracts — fully open access.
We extract (subject, relation, object) triples from the context passages
using the same scispaCy NER pipeline used throughout AXIOM.

Usage:
    python -m scripts.download_pubmedqa [--max-passages 2000]
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

logger = logging.getLogger("axiom.scripts.download_pubmedqa")


def download_pubmedqa(max_passages: int = 2000) -> Path:
    """
    Download PubMedQA and save raw passages for triple extraction.

    Args:
        max_passages: Maximum number of passages to save (controls NER time).

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
    output_dir = data_cfg.dataset_dir / "pubmedqa"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Downloading PubMedQA from Hugging Face …")
    logger.info("=" * 60)

    # PubMedQA artificial subset — 211K pairs, no auth required
    ds = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_artificial",
        split="train",
        trust_remote_code=False,
    )

    logger.info("Downloaded %d entries from PubMedQA (pqa_artificial)", len(ds))

    # Extract passages and QA pairs
    passages: list[dict] = []
    for i, row in enumerate(ds):
        if i >= max_passages:
            break

        # Each row has: pubid, question, context (dict of labels + contexts),
        # long_answer, final_decision
        context_texts = row.get("context", {})
        contexts = context_texts.get("contexts", [])
        labels = context_texts.get("labels", [])

        # Combine all context sentences into one passage
        full_context = " ".join(contexts) if isinstance(contexts, list) else str(contexts)

        if not full_context.strip():
            continue

        passages.append({
            "pubid": str(row.get("pubid", f"pqa_{i}")),
            "question": row.get("question", ""),
            "context": full_context,
            "answer": row.get("long_answer", ""),
            "decision": row.get("final_decision", ""),
        })

    output_path = output_dir / "pubmedqa_passages.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"passages": passages}, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d passages → %s", len(passages), output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download PubMedQA dataset")
    parser.add_argument(
        "--max-passages",
        type=int,
        default=2000,
        help="Maximum passages to download (default: 2000)",
    )
    args = parser.parse_args()
    download_pubmedqa(max_passages=args.max_passages)
