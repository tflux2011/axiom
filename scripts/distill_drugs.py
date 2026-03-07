"""
AXIOM Drug Interaction Distillation Pipeline

Loads the curated drug interaction dataset and distils it into an Axiom Map
for the Offline Drug Interaction Checker.

Usage:
    python -m scripts.distill_drugs [--dataset path] [--output path]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from axiom_hdc.drug_checker import DrugInteractionChecker
from axiom_hdc.utils import setup_logging

logger = logging.getLogger("axiom.scripts.distill_drugs")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         AXIOM Drug Interaction Knowledge Distiller          ║
║     Hyperdimensional Encoding for Offline Safety Checks     ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_distillation(
    dataset_path: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Distil drug interactions into an Axiom Map."""
    setup_logging()

    print(BANNER)
    logger.info("=" * 60)
    logger.info("AXIOM Drug Interaction Distillation Pipeline")
    logger.info("=" * 60)

    # Resolve paths
    project_root = Path(__file__).resolve().parent.parent
    if dataset_path is None:
        dataset_path = project_root / "data" / "drug_interactions.json"

    if output_dir is None:
        output_dir = project_root / "data" / "distilled"

    if not dataset_path.exists():
        logger.error(
            "Dataset not found: %s\n"
            "Ensure data/drug_interactions.json exists.",
            dataset_path,
        )
        sys.exit(1)

    # Create checker and run pipeline
    checker = DrugInteractionChecker()

    # Step 1: Load dataset
    logger.info("Step 1: Loading drug interaction dataset...")
    count = checker.load_dataset(dataset_path)
    logger.info("  → Loaded %d interaction triples", count)

    # Step 2: Distil into Axiom Map
    logger.info("Step 2: Distilling into Axiom Map via HDC...")
    checker.distill()

    # Step 3: Save
    logger.info("Step 3: Saving distilled state...")
    checker.save(output_dir)

    # Summary
    stats = checker.get_stats()
    logger.info("-" * 60)
    logger.info("Distillation complete!")
    logger.info("  Interactions:  %d", stats["total_interactions"])
    logger.info("  Unique drugs:  %d", stats["unique_drugs"])
    logger.info("  Aliases:       %d", stats["aliases_count"])
    logger.info("  HDC dim:       %d", stats["axiom_map_dimensions"])
    logger.info("  Facts:         %d", stats["facts_distilled"])
    logger.info("  Entities:      %d", stats["entities_in_memory"])
    logger.info("  Map size:      %s bytes", f"{stats['map_size_bytes']:,}")
    logger.info("  Output:        %s", output_dir)
    logger.info("-" * 60)

    # Quick verification
    logger.info("Verifying with sample query...")
    result = checker.check("warfarin", "aspirin")
    logger.info(
        "  warfarin + aspirin → %s (severity=%s, confidence=%.4f)",
        "FOUND" if result.found else "NOT FOUND",
        result.severity,
        result.confidence,
    )

    result2 = checker.check("metformin", "renal failure")
    logger.info(
        "  metformin + renal failure → %s (severity=%s, confidence=%.4f)",
        "FOUND" if result2.found else "NOT FOUND",
        result2.severity,
        result2.confidence,
    )

    print("\n  ✅ Drug interaction Axiom Map is ready for offline use!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AXIOM Drug Interaction Distillation"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to drug_interactions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for distilled Axiom Map",
    )
    args = parser.parse_args()

    run_distillation(
        dataset_path=args.dataset,
        output_dir=args.output,
    )
