"""
AXIOM End-to-End Distillation Pipeline

Takes raw BioASQ JSON data and produces a distilled Axiom Map + Item Memory
ready for deployment on the 64 GB SD card.

Pipeline:
    1. Load BioASQ JSON → extract snippets
    2. Run NER encoder → produce MedicalFact triples
    3. Distill triples → Axiom Map (Superposition Vector)
    4. Save map + item memory to external HD

Usage:
    python -m scripts.distill_corpus [--dataset path/to/bioasq.json] [--use-structured]
"""

from __future__ import annotations
from axiom_hdc.utils import setup_logging, timer
from axiom_hdc.encoder import AxiomEncoder
from axiom_hdc.distiller import AxiomDistiller, MedicalFact
from axiom_hdc.config import data as data_cfg

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger("axiom.scripts.distill")


def _load_bioasq_snippets(dataset_path: Path) -> list[str]:
    """Extract text snippets from a BioASQ JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for q in data.get("questions", []):
        for snippet in q.get("snippets", []):
            text = snippet.get("text", "").strip()
            if text:
                texts.append(text)

    logger.info("Loaded %d snippets from %s", len(texts), dataset_path)
    return texts


def _load_bioasq_structured(dataset_path: Path) -> list[tuple[str, str, str]]:
    """Extract pre-structured triples from BioASQ exact_answer fields."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    triples = []
    for q in data.get("questions", []):
        answers = q.get("exact_answer", [])
        if isinstance(answers, list):
            for ans in answers:
                if isinstance(ans, list) and len(ans) == 3:
                    triples.append(tuple(ans))

    logger.info("Loaded %d structured triples from %s",
                len(triples), dataset_path)
    return triples


def run_distillation(
    dataset_path: Path | None = None,
    use_structured: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Run the full distillation pipeline."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("AXIOM Distillation Pipeline")
    logger.info("=" * 60)

    # Resolve paths
    if dataset_path is None:
        dataset_path = data_cfg.dataset_dir / "bioasq_14b" / "bioasq_14b_sample.json"

    if output_dir is None:
        output_dir = data_cfg.axiom_maps_dir / "bioasq_14b"

    if not dataset_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "Run 'python -m scripts.download_bioasq' first.",
            dataset_path,
        )
        sys.exit(1)

    distiller = AxiomDistiller()

    if use_structured:
        # Path A: Pre-structured triples (faster, no NER needed)
        logger.info("Mode: Structured triples (no NER)")
        triples = _load_bioasq_structured(dataset_path)
        encoder = AxiomEncoder()
        facts = list(encoder.extract_from_structured(triples))
    else:
        # Path B: Raw text → NER → facts
        logger.info("Mode: Raw text with NER extraction")
        texts = _load_bioasq_snippets(dataset_path)
        encoder = AxiomEncoder()

        facts = []
        with timer("NER extraction"):
            for fact in encoder.extract_batch(texts):
                facts.append(fact)

        logger.info("Extracted %d facts via NER", len(facts))

    if not facts:
        logger.warning("No facts extracted. Exiting.")
        sys.exit(1)

    # Distil into Axiom Map
    with timer("HD distillation"):
        distiller.distill(facts)

    # Save
    distiller.save(output_dir)

    logger.info("-" * 60)
    logger.info("Distillation complete.")
    logger.info("  Facts encoded:  %d", distiller.fact_count)
    logger.info("  Unique entities: %d", distiller.item_memory.size)
    logger.info("  Map size:       %s bytes", f"{distiller.map_size_bytes:,}")
    logger.info("  Output:         %s", output_dir)
    logger.info("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXIOM Distillation Pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to BioASQ JSON dataset file",
    )
    parser.add_argument(
        "--use-structured",
        action="store_true",
        help="Use pre-structured triples instead of NER extraction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the Axiom Map",
    )
    args = parser.parse_args()

    run_distillation(
        dataset_path=args.dataset,
        use_structured=args.use_structured,
        output_dir=args.output,
    )
