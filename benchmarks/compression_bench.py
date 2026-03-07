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

Note: FAISS and PyTorch OpenMP libraries can conflict in the same process,
so we run FAISS measurement in a subprocess to isolate memory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger("axiom.bench.compression")


# ---------------------------------------------------------------------------
# Synthetic fact generator (for reproducible benchmarking)
# ---------------------------------------------------------------------------

def _generate_synthetic_facts(n: int, seed: int = 42) -> list[dict]:
    """Generate n synthetic medical facts for benchmarking.

    Returns dicts (not MedicalFact objects) so this works
    without importing torch/torchhd.
    """
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
        facts.append({"subject": subj, "relation": rel, "obj": obj})

    return facts


# ---------------------------------------------------------------------------
# FAISS baseline
# ---------------------------------------------------------------------------

def _measure_faiss_storage(facts: list[dict], dim: int = 768) -> dict:
    """
    Build a FAISS index and measure storage — runs in a subprocess
    to avoid OpenMP clashes with PyTorch.
    """
    n = len(facts)
    logger.info(
        "Building FAISS index for %d facts (dim=%d) [subprocess]...", n, dim)

    # Self-contained script that only imports faiss + numpy
    script = f"""
import json, os, sys, tempfile, time
import numpy as np
import faiss

n = {n}
dim = {dim}
rng = np.random.default_rng(0)
embeddings = rng.standard_normal((n, dim)).astype(np.float32)
faiss.normalize_L2(embeddings)

nlist = min(256, max(1, n // 40))
m_sub = 48  # 768 / 48 = 16
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_sub, 8)

t0 = time.perf_counter()
train_size = min(n, max(nlist * 40, 10_000))
index.train(embeddings[:train_size])
index.add(embeddings)
build_time = time.perf_counter() - t0

# Measure disk size
with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
    faiss.write_index(index, tmp.name)
    faiss_bytes = os.path.getsize(tmp.name)
    os.unlink(tmp.name)

raw_bytes = embeddings.nbytes

# Metadata storage
meta = json.dumps([{{"s": f"Drug_{{i%500:04d}}", "r": "TREATS", "o": f"Condition_{{i%300:04d}}"}} for i in range(n)])
meta_bytes = len(meta.encode("utf-8"))
total_bytes = faiss_bytes + meta_bytes

result = {{
    "method": "FAISS_IVF_PQ",
    "num_facts": n,
    "embedding_dim": dim,
    "index_bytes": faiss_bytes,
    "raw_embedding_bytes": raw_bytes,
    "metadata_bytes": meta_bytes,
    "total_bytes": total_bytes,
    "bits_per_fact": (total_bytes * 8) / n,
    "build_time_s": build_time,
}}
print(json.dumps(result))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        timeout=300,
    )

    if result.returncode != 0:
        logger.error("FAISS subprocess failed: %s", result.stderr)
        raise RuntimeError(f"FAISS subprocess failed (rc={result.returncode})")

    return json.loads(result.stdout.strip())


# ---------------------------------------------------------------------------
# AXIOM measurement
# ---------------------------------------------------------------------------

def _measure_axiom_storage(facts: list[dict]) -> dict:
    """
    Distill facts into an Axiom Map and measure storage.
    Imports torch/torchhd lazily to avoid loading them before FAISS subprocess.
    """
    # Lazy imports — keep torch out of module scope
    from src.distiller import AxiomDistiller, MedicalFact
    from src.config import hdc

    n = len(facts)
    logger.info("Distilling %d facts into Axiom Map (D=%d)...",
                n, hdc.dimensions)

    # Convert dicts to MedicalFact objects
    medical_facts = [
        MedicalFact(subject=f["subject"], relation=f["relation"], obj=f["obj"])
        for f in facts
    ]

    distiller = AxiomDistiller()

    t0 = time.perf_counter()
    distiller.distill(medical_facts)
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
    from src.utils import setup_logging, save_json
    from src.config import data as data_cfg

    setup_logging()

    logger.info("=" * 60)
    logger.info("AXIOM Compression Benchmark — %d facts", num_facts)
    logger.info("=" * 60)

    facts = _generate_synthetic_facts(num_facts)

    # Run FAISS first (in subprocess, before torch is loaded)
    faiss_results = _measure_faiss_storage(facts)

    # Then run AXIOM (loads torch/torchhd)
    axiom_results = _measure_axiom_storage(facts)

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
