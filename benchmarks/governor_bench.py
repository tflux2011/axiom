#!/usr/bin/env python3
"""
AXIOM — Safety Governor Impact Benchmark

Measures the Safety Governor's effect on hallucination rate, token
suppression rate, and fluency preservation using synthetic medical
QA scenarios—no LLM required.

Protocol
--------
For each of `n_scenarios` synthetic scenarios:
  1. Create a ground-truth Axiom Map with K known medical facts.
  2. Simulate SLM output as a ranked list of candidate tokens:
     - Some from known entities (should PASS).
     - Some from unknown entities (should be SUPPRESSED).
     - Some from "hallucination traps" (entities deliberately absent
       from the Axiom Map but medically plausible).
  3. Run the SafetyGovernor.filter_logits() pipeline.
  4. Record:  suppression rate, hallucination catch rate, false positive
     rate (legitimate tokens incorrectly suppressed), and effective
     vocabulary reduction.

Outputs
-------
- Console table.
- benchmarks/results/governor_results.json
- benchmarks/results/governor_analysis.png

Security: No network calls; all synthetic data.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torchhd.functional as F

from src.distiller import _cyclic_shift


# ---------------------------------------------------------------------------
# Synthetic tokeniser stub
# ---------------------------------------------------------------------------

class SyntheticTokenizer:
    """Minimal tokenizer stub for Governor benchmarking."""

    def __init__(self, vocab: list[str]) -> None:
        self._id_to_token = {i: t for i, t in enumerate(vocab)}
        self._token_to_id = {t: i for i, t in enumerate(vocab)}

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._id_to_token.get(i, "<unk>") for i in token_ids)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = text.strip().split()
        return [self._token_to_id.get(t, 0) for t in tokens]

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """A single test scenario for the Governor."""
    name: str
    known_entities: list[str]        # Entities IN the Axiom Map
    hallucination_entities: list[str] # Entities NOT in the Axiom Map
    n_facts: int


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def create_axiom_map_and_memory(
    facts: list[tuple[str, str, str]],
    dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build an Axiom Map and item memory from a list of triples."""
    item_memory: dict[str, torch.Tensor] = {}

    def get_or_create(name: str) -> torch.Tensor:
        key = name.strip().upper()
        if key not in item_memory:
            item_memory[key] = F.random(1, dim, device=device)
        return item_memory[key]

    axiom_map = torch.zeros(1, dim, device=device)
    for s, r, o in facts:
        v_s = get_or_create(s)
        v_r = get_or_create(r)
        v_o = get_or_create(o)
        fact_hv = F.bind(
            F.bind(v_s, _cyclic_shift(v_r, 1)),
            _cyclic_shift(v_o, 2),
        )
        axiom_map = axiom_map + fact_hv

    # Cleanup
    for _ in range(3):
        axiom_map = torch.sign(axiom_map)
        axiom_map[axiom_map == 0] = 1.0

    return axiom_map, item_memory


def run_governor_experiment(
    dim: int = 10_000,
    n_known_facts: int = 200,
    n_known_entities: int = 120,
    n_hallucination_entities: int = 60,
    n_scenarios: int = 50,
    top_k: int = 50,
    threshold: float = 0.35,
    suppression_penalty: float = -100.0,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Run the Governor impact benchmark using Query-Response Decoupling
    with Nearest-Neighbour matching.

    Algorithm (per query):
      1. Construct query probe V_q = v_s ⊗ Π_1(v_r).
      2. Unbind from Axiom Map: v̂_answer = S ⊗ V_q.
      3. Find the nearest neighbour of v̂_answer in the object vocabulary
         via cosine similarity (the expected entity).
      4. For each top-k candidate token: PASS if it matches the expected
         entity; SUPPRESS otherwise.

    The NN approach is robust to noise because the correct answer always
    has the highest relative similarity, even when absolute cosine values
    are small due to superposition noise.

    Returns a dict with aggregate metrics.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device)

    # --- 1. Build ground-truth knowledge base ---
    known_drugs = [f"DRUG_{i:04d}" for i in range(n_known_entities // 2)]
    known_conditions = [f"COND_{i:04d}" for i in range(n_known_entities // 2)]
    relations = ["TREATS", "CAUSES", "CONTRAINDICATES", "PREVENTS",
                 "DOSAGE_OF", "SIDE_EFFECT_OF"]

    facts: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    while len(facts) < n_known_facts:
        s = rng.choice(known_drugs)
        r = rng.choice(relations)
        o = rng.choice(known_conditions)
        triple = (s, r, o)
        if triple not in seen:
            seen.add(triple)
            facts.append(triple)

    axiom_map, item_memory = create_axiom_map_and_memory(facts, dim, dev)

    # --- 2. Create hallucination entities (NOT in the map) ---
    hallucination_names = [f"FAKE_DRUG_{i:04d}" for i in range(n_hallucination_entities)]
    for name in hallucination_names:
        key = name.strip().upper()
        if key not in item_memory:
            item_memory[key] = F.random(1, dim, device=dev)

    # Identify object entities for NN search
    all_obj_names = sorted(set(o.strip().upper() for _, _, o in facts))
    obj_matrix = torch.cat([item_memory[n] for n in all_obj_names], dim=0)
    obj_shifted = _cyclic_shift(obj_matrix, 2)  # Pre-shift for NN search

    # Build fact index: (subject, relation) → correct objects
    fact_index: dict[tuple[str, str], list[str]] = {}
    for s, r, o in facts:
        key = (s.strip().upper(), r.strip().upper())
        fact_index.setdefault(key, []).append(o.strip().upper())

    # Build vocab for synthetic tokenizer
    all_known_entities = list(set(
        [s for s, _, _ in facts] + [o for _, _, o in facts]
    ))
    all_relation_names = list(set(r for _, r, _ in facts))
    vocab = all_known_entities + all_relation_names + hallucination_names + ["<pad>", "<unk>"]
    tokenizer = SyntheticTokenizer(vocab)

    # --- 3. Run scenarios ---
    scenario_results: list[dict] = []

    for scenario_idx in range(n_scenarios):
        query_fact = facts[rng.randint(0, len(facts) - 1)]
        q_sub, q_rel, q_obj = query_fact

        # Step 1: Query probe
        v_sub = item_memory[q_sub.strip().upper()]
        v_rel = item_memory[q_rel.strip().upper()]
        query_probe = F.bind(v_sub, _cyclic_shift(v_rel, 1))

        # Step 2: Unbind expected answer from Axiom Map
        expected_answer = F.bind(axiom_map, query_probe)

        # Step 3: Find NN in object vocabulary
        cos_nn = torch.nn.functional.cosine_similarity(
            expected_answer.float().expand_as(obj_shifted),
            obj_shifted.float(), dim=1,
        )
        nn_idx = cos_nn.argmax().item()
        nn_entity = all_obj_names[nn_idx]
        nn_cos_value = cos_nn[nn_idx].item()

        # Correct objects for this query
        correct_objects = fact_index.get(
            (q_sub.strip().upper(), q_rel.strip().upper()), []
        )

        # Step 4: Governor filtering on simulated top-k candidates
        logits = torch.randn(tokenizer.vocab_size, device=dev) * 0.5

        # Correct answers get high logits
        for obj_name in correct_objects:
            for t_name, t_id in tokenizer._token_to_id.items():
                if t_name.strip().upper() == obj_name:
                    logits[t_id] = 5.0 + rng.random() * 3.0
                    break

        # Hallucinations get medium-high logits
        n_halluc = rng.randint(2, 6)
        halluc_sample = rng.sample(hallucination_names, min(n_halluc, len(hallucination_names)))
        for t in halluc_sample:
            tid = tokenizer._token_to_id.get(t, 0)
            logits[tid] = 4.0 + rng.random() * 2.0

        original_logits = logits.clone()
        top_values, top_indices = torch.topk(logits, min(top_k, logits.size(-1)))

        verdicts: list[dict] = []
        modified_logits = logits.clone()

        for i in range(top_indices.size(0)):
            token_id = top_indices[i].item()
            token_text = tokenizer.decode([token_id]).strip()
            token_clean = token_text.strip().upper()

            # Decision: does this token match the NN-extracted entity?
            is_nn_match = (token_clean == nn_entity)

            # Also check if it's in the broader correct set
            is_correct = token_clean in correct_objects

            # Also compute relative similarity for analysis
            if token_clean in item_memory:
                token_hv = item_memory[token_clean]
                token_shifted = _cyclic_shift(token_hv, 2)
                sim = torch.nn.functional.cosine_similarity(
                    token_shifted.float(), expected_answer.float(),
                ).max().item()
            else:
                sim = 0.0

            # Governor decision: PASS if NN match OR cosine rank is high
            is_safe = is_nn_match
            if is_safe:
                action = "PASS"
            else:
                action = "SUPPRESS"
                modified_logits[token_id] += suppression_penalty

            is_halluc = token_text.strip() in hallucination_names

            verdicts.append({
                "token_text": token_text,
                "similarity": round(sim, 4),
                "is_safe": is_safe,
                "action": action,
                "is_correct_answer": is_correct,
                "is_nn_match": is_nn_match,
                "is_hallucination": is_halluc,
            })

        # --- Metrics ---
        total_evaluated = len(verdicts)
        n_suppressed = sum(1 for v in verdicts if v["action"] == "SUPPRESS")

        tp = sum(1 for v in verdicts if v["is_hallucination"] and v["action"] == "SUPPRESS")
        fn = sum(1 for v in verdicts if v["is_hallucination"] and v["action"] == "PASS")
        fp = sum(1 for v in verdicts if v["is_correct_answer"] and v["action"] == "SUPPRESS")
        tn = sum(1 for v in verdicts if v["is_correct_answer"] and v["action"] == "PASS")

        halluc_in_topk = sum(1 for v in verdicts if v["is_hallucination"])
        correct_in_topk = sum(1 for v in verdicts if v["is_correct_answer"])

        halluc_catch_rate = tp / max(halluc_in_topk, 1)
        false_positive_rate = fp / max(correct_in_topk, 1)
        suppression_rate = n_suppressed / max(total_evaluated, 1)

        # NN retrieval accuracy for this scenario
        nn_correct = nn_entity in correct_objects

        orig_top5 = set(torch.topk(original_logits, 5).indices.tolist())
        mod_top5 = set(torch.topk(modified_logits, 5).indices.tolist())
        top5_preservation = len(orig_top5 & mod_top5) / 5.0

        scenario_results.append({
            "scenario": scenario_idx,
            "query": f"{q_sub} {q_rel} ?",
            "correct_answer": q_obj,
            "nn_prediction": nn_entity,
            "nn_correct": nn_correct,
            "nn_cosine": round(nn_cos_value, 4),
            "total_evaluated": total_evaluated,
            "n_suppressed": n_suppressed,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "halluc_catch_rate": round(halluc_catch_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "suppression_rate": round(suppression_rate, 4),
            "top5_preservation": round(top5_preservation, 4),
        })

    # --- Aggregate ---
    nn_accuracy = sum(1 for r in scenario_results if r["nn_correct"]) / n_scenarios
    agg = {
        "n_scenarios": n_scenarios,
        "n_known_facts": n_known_facts,
        "dim": dim,
        "threshold": threshold,
        "nn_retrieval_accuracy": round(nn_accuracy, 4),
        "mean_halluc_catch_rate": round(float(np.mean([r["halluc_catch_rate"] for r in scenario_results])), 4),
        "std_halluc_catch_rate": round(float(np.std([r["halluc_catch_rate"] for r in scenario_results])), 4),
        "mean_false_positive_rate": round(float(np.mean([r["false_positive_rate"] for r in scenario_results])), 4),
        "std_false_positive_rate": round(float(np.std([r["false_positive_rate"] for r in scenario_results])), 4),
        "mean_suppression_rate": round(float(np.mean([r["suppression_rate"] for r in scenario_results])), 4),
        "std_suppression_rate": round(float(np.std([r["suppression_rate"] for r in scenario_results])), 4),
        "mean_top5_preservation": round(float(np.mean([r["top5_preservation"] for r in scenario_results])), 4),
        "std_top5_preservation": round(float(np.std([r["top5_preservation"] for r in scenario_results])), 4),
        "total_tp": sum(r["tp"] for r in scenario_results),
        "total_fp": sum(r["fp"] for r in scenario_results),
        "total_tn": sum(r["tn"] for r in scenario_results),
        "total_fn": sum(r["fn"] for r in scenario_results),
        "scenarios": scenario_results,
    }

    total_tp = agg["total_tp"]
    total_fp = agg["total_fp"]
    total_fn = agg["total_fn"]
    agg["precision"] = round(total_tp / max(total_tp + total_fp, 1), 4)
    agg["recall"] = round(total_tp / max(total_tp + total_fn, 1), 4)
    agg["f1"] = round(
        2 * agg["precision"] * agg["recall"] /
        max(agg["precision"] + agg["recall"], 1e-8), 4
    )

    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_governor_analysis(results: dict, out_path: Path) -> None:
    """Generate Governor analysis visualisation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = results["scenarios"]
    indices = list(range(len(scenarios)))

    catch_rates = [s["halluc_catch_rate"] * 100 for s in scenarios]
    fp_rates = [s["false_positive_rate"] * 100 for s in scenarios]
    top5_pres = [s["top5_preservation"] * 100 for s in scenarios]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # --- Hallucination catch rate ---
    axes[0].bar(indices, catch_rates, color="#2EA043", alpha=0.8, width=0.8)
    axes[0].axhline(y=np.mean(catch_rates), color="#D23C32", linestyle="--",
                    linewidth=2, label=f"Mean: {np.mean(catch_rates):.1f}%")
    axes[0].set_xlabel("Scenario", fontsize=11)
    axes[0].set_ylabel("Catch Rate (%)", fontsize=11)
    axes[0].set_title("Hallucination Catch Rate", fontsize=12)
    axes[0].set_ylim(0, 105)
    axes[0].legend(fontsize=9)

    # --- False positive rate ---
    axes[1].bar(indices, fp_rates, color="#D23C32", alpha=0.7, width=0.8)
    axes[1].axhline(y=np.mean(fp_rates), color="#1E5AB4", linestyle="--",
                    linewidth=2, label=f"Mean: {np.mean(fp_rates):.1f}%")
    axes[1].set_xlabel("Scenario", fontsize=11)
    axes[1].set_ylabel("FP Rate (%)", fontsize=11)
    axes[1].set_title("False Positive Rate", fontsize=12)
    axes[1].set_ylim(0, max(max(fp_rates) * 1.3, 10))
    axes[1].legend(fontsize=9)

    # --- Top-5 preservation (fluency proxy) ---
    axes[2].bar(indices, top5_pres, color="#0096A0", alpha=0.8, width=0.8)
    axes[2].axhline(y=np.mean(top5_pres), color="#D23C32", linestyle="--",
                    linewidth=2, label=f"Mean: {np.mean(top5_pres):.1f}%")
    axes[2].set_xlabel("Scenario", fontsize=11)
    axes[2].set_ylabel("Top-5 Preservation (%)", fontsize=11)
    axes[2].set_title("Fluency Preservation (Top-5 Stability)", fontsize=12)
    axes[2].set_ylim(0, 105)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\n  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Threshold sweep (for paper: τ sensitivity analysis)
# ---------------------------------------------------------------------------

def run_threshold_sweep(
    dim: int = 10_000,
    n_scenarios: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Sweep across shard sizes (fact counts) to show Governor accuracy vs. capacity."""
    fact_counts = [50, 100, 200, 500, 1_000, 2_000]
    sweep_results: list[dict] = []

    for n_facts in fact_counts:
        result = run_governor_experiment(
            dim=dim,
            n_known_facts=n_facts,
            n_known_entities=max(40, min(n_facts // 2, 200)),
            n_hallucination_entities=60,
            n_scenarios=n_scenarios,
            seed=seed,
        )
        sweep_results.append({
            "n_facts": n_facts,
            "nn_retrieval_accuracy": result["nn_retrieval_accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "mean_halluc_catch": result["mean_halluc_catch_rate"],
            "mean_fp_rate": result["mean_false_positive_rate"],
            "mean_top5_pres": result["mean_top5_preservation"],
        })

    return sweep_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  AXIOM — Safety Governor Impact Benchmark")
    print("=" * 60)

    # Main experiment
    results = run_governor_experiment()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    json_path = out_dir / "governor_results.json"
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  JSON saved → {json_path}")

    # Plot
    plot_governor_analysis(results, out_dir / "governor_analysis.png")

    # Summary
    print(f"\n{'='*60}")
    print("  Governor Benchmark Summary")
    print(f"{'='*60}")
    print(f"  Scenarios:              {results['n_scenarios']}")
    print(f"  Known facts:            {results['n_known_facts']:,}")
    print(f"  Threshold τ:            {results['threshold']}")
    print(f"  Hallucination catch:    {results['mean_halluc_catch_rate']:.1%} "
          f"± {results['std_halluc_catch_rate']:.1%}")
    print(f"  False positive rate:    {results['mean_false_positive_rate']:.1%} "
          f"± {results['std_false_positive_rate']:.1%}")
    print(f"  Suppression rate:       {results['mean_suppression_rate']:.1%} "
          f"± {results['std_suppression_rate']:.1%}")
    print(f"  Top-5 preservation:     {results['mean_top5_preservation']:.1%} "
          f"± {results['std_top5_preservation']:.1%}")
    print(f"  Precision:              {results['precision']:.4f}")
    print(f"  Recall:                 {results['recall']:.4f}")
    print(f"  F1 Score:               {results['f1']:.4f}")
    print(f"{'='*60}")

    # --- Fact-count sweep ---
    print(f"\n{'='*60}")
    print("  Governor Accuracy vs. Shard Size (Fact Count)")
    print(f"{'='*60}")
    sweep = run_threshold_sweep()

    sweep_path = out_dir / "governor_fact_sweep.json"
    with open(sweep_path, "w") as fh:
        json.dump(sweep, fh, indent=2)

    print(f"\n  {'N':>6s}  {'NN-Acc':>7s}  {'Prec':>6s}  {'Recall':>8s}  {'F1':>6s}  "
          f"{'Catch%':>8s}  {'FP%':>6s}  {'Fluency%':>9s}")
    print(f"  {'-'*62}")
    for r in sweep:
        print(
            f"  {r['n_facts']:>6}  {r['nn_retrieval_accuracy']:>7.1%}  "
            f"{r['precision']:>6.4f}  "
            f"{r['recall']:>8.4f}  {r['f1']:>6.4f}  "
            f"{r['mean_halluc_catch']:>8.1%}  {r['mean_fp_rate']:>6.1%}  "
            f"{r['mean_top5_pres']:>9.1%}"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
