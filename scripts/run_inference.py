"""
AXIOM Interactive Inference Demo

Demonstrates the full pipeline:
    1. Load distilled Axiom Map from disk
    2. Load Llama-3.2-3B base model (quantised)
    3. Prime the model via KV-cache injection
    4. Run interactive medical QA with Safety Governor

Usage:
    python -m scripts.run_inference [--map-dir path] [--cpu]
"""

from __future__ import annotations
from src.utils import setup_logging, timer
from src.priming import load_base_model, prime_model
from src.governor import SafetyGovernor
from src.distiller import AxiomDistiller
from src.config import hdc, model as model_cfg, data as data_cfg

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger("axiom.scripts.inference")

WELCOME_MSG = """
╔══════════════════════════════════════════════════════════════╗
║                    AXIOM Medical QA                         ║
║       Hyperdimensional Zero-Retrieval Inference Engine       ║
║                                                              ║
║  Type a medical question and press Enter.                    ║
║  Type 'quit' or 'exit' to stop.                              ║
║  Type 'status' to see system info.                           ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_inference(
    map_dir: Path | None = None,
    use_cpu: bool = False,
    demo_mode: bool = False,
) -> None:
    """Run interactive inference with the AXIOM system."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("AXIOM Inference Engine — Starting")
    logger.info("=" * 60)

    # Resolve Axiom Map path
    if map_dir is None:
        map_dir = data_cfg.axiom_maps_dir / "bioasq_14b"

    if not (map_dir / "axiom_map.pt").exists():
        logger.error(
            "Axiom Map not found at %s\n"
            "Run 'python -m scripts.distill_corpus' first.",
            map_dir,
        )
        sys.exit(1)

    # Load distilled knowledge
    distiller = AxiomDistiller()
    with timer("Loading Axiom Map"):
        distiller.load(map_dir)

    logger.info("Axiom Map loaded: %d entities", distiller.item_memory.size)

    # Build Safety Governor
    governor = SafetyGovernor(
        axiom_map=distiller.axiom_map,
        item_memory=distiller.item_memory._store,
        cfg=hdc,
    )

    if demo_mode:
        # Demo mode: run without full model (HD queries only)
        _run_demo_mode(distiller, governor)
        return

    # Load base model
    cfg = model_cfg
    if use_cpu:
        from dataclasses import replace
        cfg = replace(cfg, device="cpu", quantisation="none")

    with timer("Loading base model"):
        model, tokenizer = load_base_model(cfg, cache_dir=data_cfg.model_cache)

    # Prime model with Axiom Map
    with timer("Priming model"):
        injector = prime_model(model, distiller.axiom_map, hdc, cfg)

    # Get logits processor for generation
    processor = governor.get_logits_processor(tokenizer)

    print(WELCOME_MSG)

    try:
        while True:
            query = input("\n🔬 AXIOM > ").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("\nGoodbye.")
                break
            if query.lower() == "status":
                _print_status(distiller, governor)
                continue

            # Generate response
            with timer("Inference"):
                inputs = tokenizer(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                    f"You are a medical knowledge assistant powered by the AXIOM system. "
                    f"Answer questions based on verified medical facts. "
                    f"If uncertain, say so clearly.<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n"
                    f"{query}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n",
                    return_tensors="pt",
                ).to(model.device)

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    logits_processor=[processor],
                )

                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

            # Safety report
            report = processor.get_safety_report()

            print(f"\n📋 {response}")
            print(
                f"\n🛡️  Safety: {report['tokens_suppressed']} tokens suppressed "
                f"({report['suppression_rate']:.1%} suppression rate)"
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye.")
    finally:
        injector.detach()
        logger.info("AXIOM hooks detached. Session ended.")


def _run_demo_mode(distiller: AxiomDistiller, governor: SafetyGovernor) -> None:
    """
    Lightweight demo using HD queries only (no full model required).
    Shows the Distiller and Governor working with cosine similarity.
    """
    print(WELCOME_MSG)
    print("  ⚡ DEMO MODE (HD queries only — no LLM loaded)\n")

    while True:
        query = input("🔬 AXIOM [demo] > ").strip()

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodbye.")
            break
        if query.lower() == "status":
            _print_status(distiller, governor)
            continue

        # Simple entity extraction from query
        words = query.split()
        # Heuristic: capitalised words are likely entities
        entities = [w for w in words if w[0].isupper()] if words else []

        if len(entities) < 1:
            print("  Please include at least one medical entity (capitalised).")
            continue

        entity = entities[0]
        relation = "TREATS"  # Default

        # Check relation keywords
        for word in words:
            w = word.upper()
            if w in ("TREATS", "CAUSES", "PREVENTS", "INHIBITS", "REGULATES",
                     "CONTRAINDICATES", "INTERACTS", "ACTIVATES"):
                relation = w
                break

        query_hv = distiller.query(entity, relation)
        similarity = distiller.similarity(query_hv)

        is_supported = similarity >= hdc.safety_threshold

        if is_supported:
            print(f"  ✅ VERIFIED (similarity: {similarity:.4f})")
            print(
                f"     The knowledge base supports: {entity} → {relation} → [match found]")
        else:
            print(f"  ⚠️  UNVERIFIED (similarity: {similarity:.4f})")
            print(f"     Cannot confirm: {entity} → {relation}")
            print(f"     {governor.FALLBACK_MSG}")


def _print_status(distiller: AxiomDistiller, governor: SafetyGovernor) -> None:
    """Print system status."""
    print("\n  AXIOM System Status:")
    print(f"    Axiom Map dimensions: {distiller.cfg.dimensions:,}")
    print(f"    Facts distilled:      {distiller.fact_count:,}")
    print(f"    Unique entities:      {distiller.item_memory.size:,}")
    print(f"    Map size:             {distiller.map_size_bytes:,} bytes")
    print(f"    Safety threshold:     {governor.cfg.safety_threshold}")
    print(f"    Device:               {distiller.device}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXIOM Inference Demo")
    parser.add_argument("--map-dir", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode (no LLM)")
    args = parser.parse_args()

    run_inference(
        map_dir=args.map_dir,
        use_cpu=args.cpu,
        demo_mode=args.demo,
    )
