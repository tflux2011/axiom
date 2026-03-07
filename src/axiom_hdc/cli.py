"""
AXIOM CLI — Command-line interface for the AXIOM framework.

Commands:
    axiom distill   Distil facts into an AxiomMap
    axiom query     Query an existing AxiomMap
    axiom inspect   Show AxiomMap statistics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _cmd_distill(args: argparse.Namespace) -> None:
    """Distil a JSONL facts file into an AxiomMap."""
    from axiom_hdc.config import HDCConfig
    from axiom_hdc.distiller import AxiomDistiller, MedicalFact
    from axiom_hdc.axiom_map import AxiomMap

    facts_path = Path(args.facts)
    if not facts_path.exists():
        print(f"Error: facts file not found: {facts_path}", file=sys.stderr)
        sys.exit(1)

    cfg = HDCConfig(dimensions=args.dim)
    distiller = AxiomDistiller(cfg=cfg, seed=args.seed)

    # Load facts from JSONL: each line is {"subject": ..., "relation": ..., "object": ...}
    facts: list[MedicalFact] = []
    with open(facts_path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                facts.append(MedicalFact(
                    subject=obj["subject"],
                    relation=obj["relation"],
                    obj=obj["object"],
                ))
            except (json.JSONDecodeError, KeyError) as exc:
                print(
                    f"Warning: skipping line {line_num}: {exc}",
                    file=sys.stderr,
                )

    if not facts:
        print("Error: no valid facts found.", file=sys.stderr)
        sys.exit(1)

    print(f"Distilling {len(facts)} facts (D={args.dim})...")
    start = time.perf_counter()
    distiller.distill(facts)
    elapsed = time.perf_counter() - start

    axiom_map = AxiomMap(
        vector=distiller.axiom_map,
        item_memory=distiller.item_memory._store,
        metadata={
            "fact_count": distiller.fact_count,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": str(facts_path),
            "seed": args.seed,
        },
    )
    axiom_map.save(args.output)
    print(f"Done in {elapsed:.2f}s. Saved to {args.output}")
    print(axiom_map.info())


def _cmd_query(args: argparse.Namespace) -> None:
    """Query an AxiomMap for a (subject, relation) pair."""
    import torch
    from torchhd import functional as F

    from axiom_hdc.axiom_map import AxiomMap
    from axiom_hdc.distiller import _cyclic_shift

    axiom_map = AxiomMap.load(args.map, device="cpu")

    subject = args.subject.strip().upper()
    relation = args.relation.strip().upper()

    if subject not in axiom_map.item_memory:
        print(f"Warning: '{args.subject}' not found in item memory.")
    if relation not in axiom_map.item_memory:
        print(f"Warning: '{args.relation}' not found in item memory.")

    # Get or generate hypervectors
    v_sub = axiom_map.item_memory.get(
        subject, F.random(1, axiom_map.dim))
    v_rel = axiom_map.item_memory.get(
        relation, F.random(1, axiom_map.dim))

    # Construct query probe
    query_probe = F.bind(v_sub, _cyclic_shift(v_rel, 1))

    # Unbind from map
    recovered = F.bind(axiom_map.vector, query_probe)

    # Find nearest entity in item memory
    best_match = None
    best_sim = -1.0
    for name, hv in axiom_map.item_memory.items():
        sim = torch.nn.functional.cosine_similarity(
            recovered.float(), hv.float()
        ).item()
        if sim > best_sim:
            best_sim = sim
            best_match = name

    print(f"Query: {args.subject} --[{args.relation}]--> ?")
    print(f"Answer: {best_match}  (cosine: {best_sim:.4f})")


def _cmd_inspect(args: argparse.Namespace) -> None:
    """Print AxiomMap statistics."""
    from axiom_hdc.axiom_map import AxiomMap

    axiom_map = AxiomMap.load(args.map, device="cpu")
    print(axiom_map.info())

    if args.entities:
        print(f"\nEntities ({axiom_map.entity_count}):")
        for name in sorted(axiom_map.item_memory.keys()):
            print(f"  - {name}")


def main() -> None:
    """Entry point for the axiom CLI."""
    parser = argparse.ArgumentParser(
        prog="axiom",
        description="AXIOM: Zero-Retrieval Knowledge Injection via HDC",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- distill ---
    p_distill = subparsers.add_parser(
        "distill", help="Distil facts into an AxiomMap"
    )
    p_distill.add_argument(
        "--facts", required=True,
        help="Path to JSONL file with facts "
             '(each line: {"subject": ..., "relation": ..., "object": ...})',
    )
    p_distill.add_argument(
        "--dim", type=int, default=10_000,
        help="Hypervector dimensionality (default: 10000)",
    )
    p_distill.add_argument(
        "--output", "-o", default="output.axiom",
        help="Output file path (default: output.axiom)",
    )
    p_distill.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p_distill.set_defaults(func=_cmd_distill)

    # --- query ---
    p_query = subparsers.add_parser(
        "query", help="Query an AxiomMap"
    )
    p_query.add_argument(
        "--map", required=True, help="Path to .axiom file",
    )
    p_query.add_argument(
        "--subject", "-s", required=True, help="Subject entity",
    )
    p_query.add_argument(
        "--relation", "-r", required=True, help="Relation type",
    )
    p_query.set_defaults(func=_cmd_query)

    # --- inspect ---
    p_inspect = subparsers.add_parser(
        "inspect", help="Show AxiomMap statistics"
    )
    p_inspect.add_argument(
        "--map", required=True, help="Path to .axiom file",
    )
    p_inspect.add_argument(
        "--entities", action="store_true",
        help="List all entities in the item memory",
    )
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
