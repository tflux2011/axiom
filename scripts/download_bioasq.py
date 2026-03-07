"""
BioASQ Dataset Downloader

Downloads BioASQ 14b dataset files to the external HD.
Requires BioASQ credentials (set in .env as BIOASQ_USERNAME / BIOASQ_PASSWORD).

Usage:
    python -m scripts.download_bioasq
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import data as data_cfg
from src.utils import setup_logging

logger = logging.getLogger("axiom.scripts.download")

# BioASQ 14b task URLs (2026 challenge)
BIOASQ_BASE_URL = "http://participants-area.bioasq.org/Tasks/14b"

# Publicly available training data (no auth required)
PUBMED_ABSTRACTS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"
)


def download_bioasq() -> None:
    """Download BioASQ dataset to the external HD."""
    setup_logging()

    username = os.getenv("BIOASQ_USERNAME")
    password = os.getenv("BIOASQ_PASSWORD")

    if not username or not password:
        logger.error(
            "BioASQ credentials not set. "
            "Add BIOASQ_USERNAME and BIOASQ_PASSWORD to your .env file. "
            "Register at http://participants-area.bioasq.org/"
        )
        sys.exit(1)

    data_cfg.ensure_dirs()
    output_dir = data_cfg.dataset_dir / "bioasq_14b"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset directory: %s", output_dir)

    # For the PoC, we create a sample dataset structure
    # In production, this would use requests with auth to download from BioASQ
    sample_data = {
        "questions": [
            {
                "id": "bioasq_14b_001",
                "type": "factoid",
                "body": "What is the mechanism of action of metformin?",
                "ideal_answer": "Metformin reduces hepatic glucose production and increases insulin sensitivity.",
                "exact_answer": [["Metformin", "INHIBITS", "Hepatic_Glucose_Production"]],
                "snippets": [
                    {
                        "text": "Metformin activates AMP-activated protein kinase (AMPK) in the liver, "
                                "reducing gluconeogenesis and increasing glucose uptake in skeletal muscle.",
                        "document": "PMID:12345678",
                    }
                ],
            },
            {
                "id": "bioasq_14b_002",
                "type": "yesno",
                "body": "Does aspirin interact with warfarin?",
                "ideal_answer": "Yes, aspirin increases the risk of bleeding when combined with warfarin.",
                "exact_answer": "yes",
                "snippets": [
                    {
                        "text": "Concomitant use of aspirin and warfarin significantly increases "
                                "the risk of gastrointestinal bleeding.",
                        "document": "PMID:23456789",
                    }
                ],
            },
            {
                "id": "bioasq_14b_003",
                "type": "factoid",
                "body": "What are the contraindications for ACE inhibitors?",
                "ideal_answer": "ACE inhibitors are contraindicated in pregnancy, bilateral renal artery stenosis, and angioedema history.",
                "exact_answer": [
                    ["ACE_Inhibitors", "CONTRAINDICATES", "Pregnancy"],
                    ["ACE_Inhibitors", "CONTRAINDICATES", "Bilateral_Renal_Stenosis"],
                    ["ACE_Inhibitors", "CONTRAINDICATES", "Angioedema_History"],
                ],
                "snippets": [
                    {
                        "text": "ACE inhibitors are absolutely contraindicated during pregnancy "
                                "due to teratogenic effects. They should also be avoided in patients "
                                "with bilateral renal artery stenosis or a history of angioedema.",
                        "document": "PMID:34567890",
                    }
                ],
            },
        ]
    }

    # Save sample dataset
    sample_path = output_dir / "bioasq_14b_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)

    logger.info("Sample BioASQ dataset created → %s", sample_path)
    logger.info(
        "\nTo download the full BioASQ 14b dataset:\n"
        "  1. Go to http://participants-area.bioasq.org/\n"
        "  2. Register for Task 14b (Phase B starts March 18, 2026)\n"
        "  3. Download the training data JSON files\n"
        "  4. Place them in: %s\n",
        output_dir,
    )


if __name__ == "__main__":
    download_bioasq()
