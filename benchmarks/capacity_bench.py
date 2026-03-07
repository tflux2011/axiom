#!/usr/bin/env python3
"""
AXIOM — HDC Capacity Benchmark

Measures retrieval accuracy as the number of superposed facts grows.
This directly addresses the reviewer concern: "Can a single 10,000-D
bipolar vector really store millions of distinguishable facts?"

Protocol
--------
For each scale N in {100, 500, 1K, 5K, 10K, 50K, 100K, 500K, 1M, 5M}:
  1. Generate N synthetic (Subject, Relation, Object) triples.
  2. Distil them into a single Axiom Map S via superposition.
  3. After sign-cleanup, attempt to retrieve each of 200 randomly sampled
     facts by constructing a query probe and unbinding the expected
     answer from S.
  4. Declare retrieval *correct* when the cosine similarity between the
     recovered HV and the ground-truth object HV exceeds τ = 0.35, AND
     the correct object is within the top-1 nearest neighbour in item
     memory.
  5. Record Recall@1, mean cosine similarity, and std.

Outputs
-------
- Console table.
- benchmarks/results/capacity_results.json
- benchmarks/results/capacity_curve.png   (matplotlib plot)

Security: No network calls; all synthetic data.
"""

from __future__ import annotations
from axiom_hdc.distiller import _cyclic_shift
import torchhd.functional as F

import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import numpy as np

# --- project imports --------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_entity_name(prefix: str, idx: int) -> str:
    """Deterministic, collision-free entity name."""
    return f"{prefix}_{idx:07d}"


def _generate_facts(
    n: int,
    n_subjects: int | None = None,
    n_relations: int = 12,
    n_objects: int | None = None,
    seed: int = 42,
) -> list[tuple[str, str, str]]:
    """
    Generate *n* unique (S, R, O) triples with controlled vocabulary sizes.

    Default vocabulary: n_subjects = n_objects ≈ ceil(sqrt(n)) * 3 to ensure
    enough unique combinations without excessive sharing.
    """
    rng = random.Random(seed)

    if n_subjects is None:
        n_subjects = max(100, int(math.ceil(math.sqrt(n) * 3)))
    if n_objects is None:
        n_objects = max(100, int(math.ceil(math.sqrt(n) * 3)))

    subjects = [_generate_entity_name("SUBJ", i) for i in range(n_subjects)]
    relations = [_generate_entity_name("REL", i) for i in range(n_relations)]
    objects_ = [_generate_entity_name("OBJ", i) for i in range(n_objects)]

    seen: set[tuple[str, str, str]] = set()
    facts: list[tuple[str, str, str]] = []
    attempts = 0
    max_attempts = n * 20

    while len(facts) < n and attempts < max_attempts:
        s = rng.choice(subjects)
        r = rng.choice(relations)
        o = rng.choice(objects_)
        triple = (s, r, o)
        if triple not in seen:
            seen.add(triple)
            facts.append(triple)
        attempts += 1

    if len(facts) < n:
        print(
            f"  [warn] only generated {len(facts)} unique triples for target {n}")

    return facts


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_capacity_experiment(
    dim: int = 10_000,
    scales: list[int] | None = None,
    n_probes: int = 200,
    threshold: float = 0.35,
    cleanup_iters: int = 3,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict]:
    """
    Run the full capacity scaling experiment.

    Returns a list of result dicts, one per scale.
    """
    if scales is None:
        scales = [100, 500, 1_000, 5_000, 10_000, 50_000,
                  100_000, 500_000, 1_000_000, 5_000_000]

    torch.manual_seed(seed)
    dev = torch.device(device)
    results: list[dict] = []

    for n_facts in scales:
        print(f"\n{'='*60}")
        print(f"  N = {n_facts:>10,} facts   |  D = {dim}")
        print(f"{'='*60}")

        t0 = time.perf_counter()

        # --- 1. Generate facts ---
        facts = _generate_facts(n_facts, seed=seed)

        # --- 2. Item Memory (lazy creation of entity HVs) ---
        item_memory: dict[str, torch.Tensor] = {}

        def get_or_create(name: str) -> torch.Tensor:
            if name not in item_memory:
                item_memory[name] = F.random(1, dim, device=dev)
            return item_memory[name]

        # --- 3. Encode & bundle ---
        axiom_map = torch.zeros(1, dim, device=dev)
        for s, r, o in facts:
            v_s = get_or_create(s)
            v_r = get_or_create(r)
            v_o = get_or_create(o)
            fact_hv = F.bind(
                F.bind(v_s, _cyclic_shift(v_r, 1)),
                _cyclic_shift(v_o, 2),
            )
            axiom_map = axiom_map + fact_hv

        # --- 4. Cleanup ---
        for _ in range(cleanup_iters):
            axiom_map = torch.sign(axiom_map)
            axiom_map[axiom_map == 0] = 1.0

        t_distill = time.perf_counter() - t0

        # --- 5. Probe a random subset ---
        rng = random.Random(seed + 1)
        probe_indices = rng.sample(
            range(len(facts)), min(n_probes, len(facts)))

        # Pre-stack all object HVs for nearest-neighbour search
        all_obj_names = [n for n in item_memory if n.startswith("OBJ_")]
        if not all_obj_names:
            # fallback: just use all items
            all_obj_names = list(item_memory.keys())
        obj_matrix = torch.cat(
            [item_memory[n] for n in all_obj_names], dim=0
        )  # (V, D)

        correct_threshold = 0
        correct_nn = 0
        cosine_sims: list[float] = []

        t1 = time.perf_counter()
        for idx in probe_indices:
            s, r, o = facts[idx]
            v_s = item_memory[s]
            v_r = item_memory[r]
            v_o_gt = item_memory[o]

            # Query probe
            query_probe = F.bind(v_s, _cyclic_shift(v_r, 1))

            # Unbind expected answer from Axiom Map
            v_answer = F.bind(axiom_map, query_probe)

            # Cosine similarity with ground-truth object
            cos_gt = torch.nn.functional.cosine_similarity(
                v_answer.float(), _cyclic_shift(v_o_gt, 2).float()
            ).item()
            cosine_sims.append(cos_gt)

            if cos_gt >= threshold:
                correct_threshold += 1

            # Nearest neighbour search in item memory (object vocab)
            # We need to compare against Π_2(v_o) for all objects
            obj_shifted = _cyclic_shift(obj_matrix, 2)
            cos_all = torch.nn.functional.cosine_similarity(
                v_answer.float().expand_as(obj_shifted),
                obj_shifted.float(),
                dim=1,
            )
            nn_idx = cos_all.argmax().item()
            nn_name = all_obj_names[nn_idx]
            if nn_name == o:
                correct_nn += 1

        t_probe = time.perf_counter() - t1

        n_probed = len(probe_indices)
        recall_threshold = correct_threshold / max(n_probed, 1)
        recall_nn = correct_nn / max(n_probed, 1)
        mean_cos = float(np.mean(cosine_sims))
        std_cos = float(np.std(cosine_sims))

        result = {
            "n_facts": n_facts,
            "dim": dim,
            "n_probes": n_probed,
            "recall_threshold": round(recall_threshold, 4),
            "recall_nn": round(recall_nn, 4),
            "mean_cosine": round(mean_cos, 4),
            "std_cosine": round(std_cos, 4),
            "threshold": threshold,
            "entities_in_memory": len(item_memory),
            "distill_time_s": round(t_distill, 2),
            "probe_time_s": round(t_probe, 2),
        }
        results.append(result)

        print(f"  Recall@1 (threshold): {recall_threshold:.1%}")
        print(f"  Recall@1 (NN):        {recall_nn:.1%}")
        print(f"  Mean cosine:          {mean_cos:.4f} ± {std_cos:.4f}")
        print(f"  Distill time:         {t_distill:.1f}s")
        print(f"  Probe time:           {t_probe:.1f}s")
        print(f"  Entities in memory:   {len(item_memory):,}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_capacity_curve(results: list[dict], out_path: Path) -> None:
    """Generate the capacity degradation curve (Figure for paper)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n_facts"] for r in results]
    recall_nn = [r["recall_nn"] * 100 for r in results]
    mean_cos = [r["mean_cosine"] for r in results]
    std_cos = [r["std_cosine"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Recall@1 vs. N ---
    ax1.semilogx(ns, recall_nn, "o-", color="#1E5AB4", linewidth=2,
                 markersize=8, label="Recall@1 (NN)")
    ax1.axhline(y=90, color="#D23C32", linestyle="--", alpha=0.6,
                label="90% baseline")
    ax1.set_xlabel("Number of Facts (N)", fontsize=12)
    ax1.set_ylabel("Recall@1 (%)", fontsize=12)
    ax1.set_title(
        "HDC Capacity: Retrieval Accuracy vs. Fact Count", fontsize=13)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Right: Mean cosine ± std vs. N ---
    ax2.semilogx(ns, mean_cos, "s-", color="#0096A0", linewidth=2,
                 markersize=7)
    ax2.fill_between(
        ns,
        [m - s for m, s in zip(mean_cos, std_cos)],
        [m + s for m, s in zip(mean_cos, std_cos)],
        alpha=0.2, color="#0096A0",
    )
    ax2.axhline(y=0.35, color="#D23C32", linestyle="--", alpha=0.6,
                label="τ = 0.35")
    ax2.set_xlabel("Number of Facts (N)", fontsize=12)
    ax2.set_ylabel("Cosine Similarity", fontsize=12)
    ax2.set_title(
        "Signal-to-Noise Ratio vs. Superposition Density", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")
    print(f"  PDF  saved → {out_path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dimension_scaling(
    n_facts: int = 500,
    dims: list[int] | None = None,
    n_probes: int = 200,
    threshold: float = 0.35,
    seed: int = 42,
) -> list[dict]:
    """
    Test how retrieval accuracy scales with dimension D at a fixed fact count.
    This demonstrates that capacity grows with D.
    """
    if dims is None:
        dims = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]

    results: list[dict] = []
    for d in dims:
        print(f"\n  Dimension scaling: D={d:,}, N={n_facts:,}")
        r = run_capacity_experiment(
            dim=d, scales=[n_facts], n_probes=n_probes,
            threshold=threshold, seed=seed,
        )
        r[0]["dim"] = d
        results.append(r[0])
    return results


def run_sharding_experiment(
    total_facts: int = 5_000,
    shard_sizes: list[int] | None = None,
    dim: int = 10_000,
    n_probes: int = 200,
    threshold: float = 0.35,
    seed: int = 42,
    device: str = "cpu",
) -> list[dict]:
    """
    Simulate Hierarchical Sharding: distribute N facts across multiple
    sub-maps and report per-shard recall.

    This demonstrates that sharding preserves retrieval fidelity at scale.
    """
    if shard_sizes is None:
        shard_sizes = [50, 100, 200, 500, 1_000, 2_500, 5_000]

    torch.manual_seed(seed)
    dev = torch.device(device)
    rng = random.Random(seed)

    # Generate the full fact corpus
    facts = _generate_facts(total_facts, seed=seed)
    results: list[dict] = []

    for shard_size in shard_sizes:
        if shard_size > total_facts:
            continue

        n_shards = math.ceil(total_facts / shard_size)

        # Distribute facts into shards
        shard_correct = 0
        shard_probed = 0

        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, total_facts)
            shard_facts = facts[start:end]
            if not shard_facts:
                continue

            # Build shard-local item memory and Axiom Map
            item_memory: dict[str, torch.Tensor] = {}

            def get_or_create(name: str) -> torch.Tensor:
                if name not in item_memory:
                    item_memory[name] = F.random(1, dim, device=dev)
                return item_memory[name]

            axiom_map = torch.zeros(1, dim, device=dev)
            for s, r, o in shard_facts:
                v_s = get_or_create(s)
                v_r = get_or_create(r)
                v_o = get_or_create(o)
                fact_hv = F.bind(
                    F.bind(v_s, _cyclic_shift(v_r, 1)),
                    _cyclic_shift(v_o, 2),
                )
                axiom_map = axiom_map + fact_hv

            for _ in range(3):
                axiom_map = torch.sign(axiom_map)
                axiom_map[axiom_map == 0] = 1.0

            # Probe random subset within the shard
            n_probe = min(n_probes // n_shards + 1, len(shard_facts))
            probe_indices = rng.sample(range(len(shard_facts)), n_probe)

            # Build NN lookup for objects in this shard
            obj_names = [n for n in item_memory if n.startswith("OBJ_")]
            if not obj_names:
                obj_names = list(item_memory.keys())
            obj_matrix = torch.cat([item_memory[n] for n in obj_names], dim=0)

            for idx in probe_indices:
                s, r, o = shard_facts[idx]
                v_s = item_memory[s]
                v_r = item_memory[r]
                query_probe = F.bind(v_s, _cyclic_shift(v_r, 1))
                v_answer = F.bind(axiom_map, query_probe)

                obj_shifted = _cyclic_shift(obj_matrix, 2)
                cos_all = torch.nn.functional.cosine_similarity(
                    v_answer.float().expand_as(obj_shifted),
                    obj_shifted.float(), dim=1,
                )
                nn_idx = cos_all.argmax().item()
                if obj_names[nn_idx] == o:
                    shard_correct += 1
                shard_probed += 1

        recall = shard_correct / max(shard_probed, 1)
        results.append({
            "total_facts": total_facts,
            "shard_size": shard_size,
            "n_shards": n_shards,
            "n_probed": shard_probed,
            "recall_nn": round(recall, 4),
        })
        print(f"  Shard size: {shard_size:>5,}  |  Shards: {n_shards:>3}  |  "
              f"Recall@1: {recall:.1%}")

    return results


def main() -> None:
    print("=" * 60)
    print("  AXIOM — HDC Capacity Benchmark")
    print("=" * 60)

    # Use smaller scales for faster runs; pass --full for full suite
    if "--full" in sys.argv:
        scales = [100, 500, 1_000, 5_000, 10_000, 50_000,
                  100_000, 500_000, 1_000_000, 5_000_000]
    else:
        # Quick run (completes in < 2 minutes)
        scales = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

    results = run_capacity_experiment(scales=scales)

    # Save results
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "capacity_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  JSON saved → {json_path}")

    # Plot
    plot_capacity_curve(results, out_dir / "capacity_curve.png")

    # Summary table
    print(f"\n{'='*72}")
    print(f"  {'N':>10s}  {'Recall@1':>10s}  {'NN-Recall':>10s}  "
          f"{'cos μ±σ':>14s}  {'Time(s)':>8s}")
    print(f"{'='*72}")
    for r in results:
        print(
            f"  {r['n_facts']:>10,}  {r['recall_threshold']:>10.1%}  "
            f"{r['recall_nn']:>10.1%}  "
            f"{r['mean_cosine']:>6.4f}±{r['std_cosine']:<6.4f}  "
            f"{r['distill_time_s']:>8.1f}"
        )
    print(f"{'='*72}")

    # --- Dimension scaling ---
    print(f"\n{'='*60}")
    print("  Dimension Scaling (N = 500 fixed)")
    print(f"{'='*60}")
    dim_results = run_dimension_scaling(n_facts=500)
    dim_path = out_dir / "dimension_scaling.json"
    with open(dim_path, "w") as fh:
        json.dump(dim_results, fh, indent=2)
    print(f"\n  {'D':>10s}  {'NN-Recall':>10s}  {'cos μ±σ':>14s}")
    print(f"  {'-'*40}")
    for r in dim_results:
        print(f"  {r['dim']:>10,}  {r['recall_nn']:>10.1%}  "
              f"{r['mean_cosine']:>6.4f}±{r['std_cosine']:<6.4f}")

    # --- Sharding experiment ---
    print(f"\n{'='*60}")
    print("  Hierarchical Sharding (N = 5,000 total)")
    print(f"{'='*60}")
    shard_results = run_sharding_experiment(total_facts=5_000)
    shard_path = out_dir / "sharding_results.json"
    with open(shard_path, "w") as fh:
        json.dump(shard_results, fh, indent=2)
    print(f"\n  Sharding results saved → {shard_path}")


if __name__ == "__main__":
    main()
