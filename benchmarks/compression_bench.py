"""
AXIOM Benchmark — Phase A: Compression Ratio

Compares storage efficiency between:
    Baseline: FAISS IVF-PQ Vector Index
    AXIOM:    Hyperdimensional Superposition Map

Metrics:
    - Total bytes on disk
    - Bits per medical fact
    - Compression ratio (Baseline / AXIOM)

Usage:
    python -m benchmarks.compression_bench --num-facts 100000
"""

from __future__ import annotations
from src.utils import setup_logging, save_json
from src.distiller import AxiomDistiller, MedicalFact
from src.config import hdc, data as data_cfg

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger("axiom.bench.compression")


# ---------------------------------------------------------------------------
# Synthetic fact generator (for reproducible benchmarking)
# ---------------------------------------------------------------------------

def _generate_synthetic_facts(n: int, seed: int = 42) -> list[MedicalFact]:
    """Generate n synthetic medical facts for benchmarking."""
    rng = np.random.default_rng(seed)

    drugs = [f"Drug_{i:04d}" for i in range(500)]
    conditions = [f"Condition_{i:04d}" for i in range(300)]
    relations = ["TREATS", "CAUSES", "PREVENTS", "INHIBITS", "REGULATES",
                 "CONTRAINDICATES", "INTERACTS_WITH", "ACTIVATES"]

    facts = []
    for _ in range(n):
        subj = drugs[rng.integers(len(drugs))]
        rel = relations[rng.integers(len(relations))]
        obj = conditions[rng.integers(len(conditions))]
        facts.append(MedicalFact(subject=subj, relation=rel, obj=obj))

    return facts


# ---------------------------------------------------------------------------
# FAISS baseline
# ---------------------------------------------------------------------------

def _measure_faiss_storage(facts: list[MedicalFact], dim: int = 768) -> dict:
    """
    Build a FAISS index from fact embeddings and measure storage.

    Uses simulated embeddings (random vectors) since we're measuring
    storage overhead, not retrieval quality.
    """
    import faiss

    n = len(facts)
    logger.info("Building FAISS index for %d facts (dim=%d)...", n, dim)

    # Simulate dense embeddings (float32)
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((n, dim)).astype(np.float32)

    # Normalise for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build IVF-PQ index (standard RAG configuration)
    nlist = min(256, n // 10)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, 32, 8)

    t0 = time.perf_counter()
    index.train(embeddings)
    index.add(embeddings)
    build_time = time.perf_counter() - t0

    # Measure disk size
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
        faiss.write_index(index, tmp.name)
        faiss_bytes = os.path.getsize(tmp.name)
        os.unlink(tmp.name)

    # Also measure raw embedding storage
    raw_bytes = embeddings.nbytes

    # Metadata storage (fact text as JSON)
    meta_json = json.dumps(
        [{"s": f.subject, "r": f.relation, "o": f.obj} for f in facts]
    )
    meta_bytes = len(meta_json.encode("utf-8"))

    total_bytes = faiss_bytes + meta_bytes

    return {
        "method": "FAISS_IVF_PQ",
        "num_facts": n,
        "embedding_dim": dim,
        "index_bytes": faiss_bytes,
        "raw_embedding_bytes": raw_bytes,
        "metadata_bytes": meta_bytes,
        "total_bytes": total_bytes,
        "bits_per_fact": (total_bytes * 8) / n,
        "build_time_s": build_time,
    }


# ---------------------------------------------------------------------------
# AXIOM measurement
# ---------------------------------------------------------------------------

def _measure_axiom_storage(facts: list[MedicalFact]) -> dict:
    """
    Distill facts into an Axiom Map and measure storage.
    """
    n = len(facts)
    logger.info("Distilling %d facts into Axiom Map (D=%d)...",
                n, hdc.dimensions)

    distiller = AxiomDistiller()

    t0 = time.perf_counter()
    distiller.distill(facts)
    build_time = time.perf_counter() - t0

    # Axiom Map size
    map_bytes = distiller.map_size_bytes

    # Item Memory size (the base vectors for all unique entities)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        distiller.item_memory.save(Path(tmp.name))
        im_bytes = os.path.getsize(tmp.name)
        os.unlink(tmp.name)

    total_bytes = map_bytes + im_bytes

    return {
        "method": "AXIOM_HDC",
        "num_facts": n,
        "hdc_dimensions": hdc.dimensions,
        "axiom_map_bytes": map_bytes,
        "item_memory_bytes": im_bytes,
        "total_bytes": total_bytes,
        "bits_per_fact": (total_bytes * 8) / n,
        "build_time_s": build_time,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_compression_benchmark(num_facts: int = 100_000) -> dict:
    """Run the full compression benchmark and return results."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("AXIOM Compression Benchmark — %d facts", num_facts)
    logger.info("=" * 60)

    facts = _generate_synthetic_facts(num_facts)

    axiom_results = _measure_axiom_storage(facts)
    faiss_results = _measure_faiss_storage(facts)

    # Compute compression ratio
    ratio = faiss_results["total_bytes"] / max(axiom_results["total_bytes"], 1)

    results = {
        "benchmark": "compression_ratio",
        "num_facts": num_facts,
        "axiom": axiom_results,
        "faiss_baseline": faiss_results,
        "compression_ratio": ratio,
        "axiom_advantage": f"{ratio:.1f}x smaller",
    }

    # Print summary
    logger.info("-" * 60)
    logger.info("RESULTS:")
    logger.info(
        "  FAISS total:  %s bytes (%.2f MB)",
        f"{faiss_results['total_bytes']:,}",
        faiss_results["total_bytes"] / 1e6,
    )
    logger.info(
        "  AXIOM total:  %s bytes (%.2f MB)",
        f"{axiom_results['total_bytes']:,}",
        axiom_results["total_bytes"] / 1e6,
    )
    logger.info("  Compression ratio: %.1fx", ratio)
    logger.info("-" * 60)

    # Save results
    data_cfg.ensure_dirs()
    out_path = data_cfg.results_dir / "compression_benchmark.json"
    save_json(results, out_path)
    logger.info("Results saved → %s", out_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXIOM Compression Benchmark")
    parser.add_argument("--num-facts", type=int, default=100_000)
    args = parser.parse_args()
    run_compression_benchmark(args.num_facts)
