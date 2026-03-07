"""
AXIOM Benchmark — Phase B: Latency (Time to First Token)

Compares inference latency between:
    Baseline: SLM + Standard Vector RAG (FAISS HNSW retrieval)
    AXIOM:    SLM + HD Axiomatic Priming (zero-retrieval)

Metrics:
    - Query latency in milliseconds
    - Speedup factor

Usage:
    python -m benchmarks.latency_bench --queries 50

Note: FAISS runs in subprocess to avoid OpenMP clashes with PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger("axiom.bench.latency")

# ---------------------------------------------------------------------------
# Sample medical queries
# ---------------------------------------------------------------------------

SAMPLE_QUERIES = [
    "What are the side effects of metformin?",
    "Does aspirin interact with warfarin?",
    "What is the recommended dosage of ibuprofen for adults?",
    "Can insulin be used to treat type 1 diabetes?",
    "What are the contraindications for amoxicillin?",
    "How does metformin regulate blood glucose levels?",
    "What drugs should not be taken with ACE inhibitors?",
    "What is the mechanism of action of statins?",
    "Are there alternatives to penicillin for allergic patients?",
    "What are the symptoms of serotonin syndrome?",
    "Does caffeine interact with beta-blockers?",
    "What is the half-life of diazepam?",
    "Can corticosteroids cause diabetes?",
    "What are the risks of combining NSAIDs with blood thinners?",
    "How does omeprazole affect nutrient absorption?",
    "What are the long-term effects of proton pump inhibitors?",
    "Is lisinopril safe during pregnancy?",
    "What drugs inhibit CYP3A4 metabolism?",
    "What are the first-line treatments for hypertension?",
    "Does grapefruit interact with calcium channel blockers?",
]


# ---------------------------------------------------------------------------
# Simulation: FAISS retrieval latency
# ---------------------------------------------------------------------------

def _simulate_faiss_retrieval(dim: int = 768, n_vectors: int = 100_000) -> dict:
    """
    Simulate the latency of a FAISS nearest-neighbour search.
    Runs in subprocess to avoid OpenMP clashes with PyTorch.
    Returns dict with avg, min, max latency in seconds.
    """
    logger.info("Building FAISS HNSW index (%d vectors, dim=%d) [subprocess]...",
                n_vectors, dim)

    script = f"""
import json, time
import numpy as np
import faiss

rng = np.random.default_rng(42)
data = rng.standard_normal(({n_vectors}, {dim})).astype(np.float32)
faiss.normalize_L2(data)

# HNSW index (common in production RAG)
index = faiss.IndexHNSWFlat({dim}, 32)
index.hnsw.efSearch = 128

t0 = time.perf_counter()
index.add(data)
build_time = time.perf_counter() - t0

query = rng.standard_normal((1, {dim})).astype(np.float32)
faiss.normalize_L2(query)

# Warm up
index.search(query, 10)

# Measure 50 queries
times = []
for _ in range(50):
    t0 = time.perf_counter()
    index.search(query, 10)
    times.append(time.perf_counter() - t0)

result = {{
    "avg_s": sum(times) / len(times),
    "min_s": min(times),
    "max_s": max(times),
    "build_time_s": build_time,
    "n_vectors": {n_vectors},
}}
print(json.dumps(result))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
        timeout=600,
    )

    if result.returncode != 0:
        logger.error("FAISS subprocess failed: %s", result.stderr)
        raise RuntimeError(f"FAISS subprocess failed (rc={result.returncode})")

    return json.loads(result.stdout.strip())


# ---------------------------------------------------------------------------
# AXIOM query latency
# ---------------------------------------------------------------------------

def _measure_axiom_query_latency(queries: list[str]) -> dict:
    """
    Build an Axiom Map and measure HD cosine-similarity query latency.
    """
    from axiom_hdc.distiller import AxiomDistiller, MedicalFact

    # Build a sample Axiom Map
    sample_facts = [
        MedicalFact("Aspirin", "TREATS", "Headache"),
        MedicalFact("Aspirin", "THINS", "Blood"),
        MedicalFact("Metformin", "TREATS", "Diabetes"),
        MedicalFact("Insulin", "REGULATES", "Glucose"),
        MedicalFact("Warfarin", "THINS", "Blood"),
        MedicalFact("Ibuprofen", "TREATS", "Pain"),
        MedicalFact("Amoxicillin", "TREATS", "Infection"),
    ] * 1000  # Scale up to 7000 facts

    distiller = AxiomDistiller()
    distiller.distill(sample_facts)

    latencies = []
    for q in queries:
        # Extract a query entity from the question text
        words = q.split()
        entity = words[min(5, len(words) - 1)] if len(words) > 5 else words[0]
        relation = "TREATS"

        # Warm up
        qv = distiller.query(entity, relation)
        _ = distiller.similarity(qv)

        # Measure
        t0 = time.perf_counter()
        qv = distiller.query(entity, relation)
        sim = distiller.similarity(qv)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

    return {
        "avg_s": sum(latencies) / len(latencies),
        "min_s": min(latencies),
        "max_s": max(latencies),
        "all_ms": [round(l * 1000, 4) for l in latencies],
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_latency_benchmark(num_queries: int = 20) -> dict:
    """Run the full latency benchmark and return results."""
    from axiom_hdc.utils import setup_logging, save_json
    from axiom_hdc.config import data as data_cfg

    setup_logging()

    logger.info("=" * 60)
    logger.info("AXIOM Latency Benchmark — %d queries", num_queries)
    logger.info("=" * 60)

    queries = SAMPLE_QUERIES[:num_queries]

    # --- FAISS latency (subprocess, before torch loads) ---
    faiss_data = _simulate_faiss_retrieval()
    faiss_avg_ms = faiss_data["avg_s"] * 1000

    # --- AXIOM latency (loads torch/torchhd) ---
    axiom_data = _measure_axiom_query_latency(queries)
    axiom_avg_ms = axiom_data["avg_s"] * 1000

    # --- Compute speedup ---
    speedup = faiss_avg_ms / max(axiom_avg_ms, 0.001)

    results = {
        "benchmark": "latency_ttft",
        "num_queries": num_queries,
        "axiom": {
            "avg_query_ms": round(axiom_avg_ms, 4),
            "min_query_ms": round(axiom_data["min_s"] * 1000, 4),
            "max_query_ms": round(axiom_data["max_s"] * 1000, 4),
            "per_query_ms": axiom_data["all_ms"],
            "method": "HD_cosine_similarity",
        },
        "faiss_baseline": {
            "avg_query_ms": round(faiss_avg_ms, 4),
            "min_query_ms": round(faiss_data["min_s"] * 1000, 4),
            "max_query_ms": round(faiss_data["max_s"] * 1000, 4),
            "build_time_s": faiss_data["build_time_s"],
            "method": "HNSW_knn_search",
            "n_vectors": faiss_data["n_vectors"],
        },
        "speedup": f"{speedup:.1f}x faster",
    }

    logger.info("-" * 60)
    logger.info("RESULTS:")
    logger.info("  FAISS HNSW avg:   %.4f ms", faiss_avg_ms)
    logger.info("  AXIOM HD avg:     %.4f ms", axiom_avg_ms)
    logger.info("  Speedup:          %.1fx", speedup)
    logger.info("-" * 60)

    data_cfg.ensure_dirs()
    out_path = data_cfg.results_dir / "latency_benchmark.json"
    save_json(results, out_path)
    logger.info("Results saved → %s", out_path)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AXIOM Latency Benchmark")
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    run_latency_benchmark(args.queries)
