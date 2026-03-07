"""
AXIOM Benchmark — Phase D: Ablation Study

Systematically removes each AXIOM component to measure its contribution:
    1. Full AXIOM:  HDC distillation + Safety Governor (threshold detection)
    2. -Governor:   HDC distillation only (no threshold-based filtering)
    3. -HDC Superposition: Each fact stored as separate vector (no binding)
    4. -Permutation Roles: Facts use naive binding without role encoding
    5. Dim Sensitivity: Vary D from 1000 to 20000

Metrics:
    - Retrieval Accuracy (NN)
    - Cosine separation (positive vs. negative)
    - Optimal F1

Usage:
    python -m benchmarks.ablation_bench
"""

from __future__ import annotations
from benchmarks.accuracy_bench import _build_knowledge_base, _build_test_queries
from axiom_hdc.utils import setup_logging, save_json
from axiom_hdc.config import hdc, data as data_cfg
from axiom_hdc.distiller import AxiomDistiller, MedicalFact, _cyclic_shift

import logging
import sys
import time
from pathlib import Path

import torch
import torchhd.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger("axiom.bench.ablation")


# ---------------------------------------------------------------------------
# Evaluation helper (shared across ablation variants)
# ---------------------------------------------------------------------------

def _evaluate_retrieval(
    axiom_map: torch.Tensor,
    item_memory,
    fact_tuples: list[tuple[str, str, str]],
    test_queries: list[dict],
    use_roles: bool = True,
) -> dict:
    """
    Evaluate retrieval accuracy on the given map + item memory.

    Args:
        axiom_map: (1, D) superposition map
        item_memory: object with .get(name) -> (1, D) HV
        fact_tuples: ground-truth facts
        test_queries: list of query dicts
        use_roles: whether role permutation was used in encoding
    """
    all_objects = sorted({f[2] for f in fact_tuples})

    # Build shifted object matrix
    if use_roles:
        obj_hvs = torch.stack([
            _cyclic_shift(item_memory.get(o), 2).squeeze(0)
            for o in all_objects
        ])
    else:
        obj_hvs = torch.stack([
            item_memory.get(o).squeeze(0)
            for o in all_objects
        ])

    pos_cosines = []
    neg_cosines = []
    retrieval_correct = 0
    retrieval_total = 0

    for q in test_queries:
        v_sub = item_memory.get(q["subject"])
        v_rel = item_memory.get(q["relation"])

        if use_roles:
            query_probe = F.bind(v_sub, _cyclic_shift(v_rel, 1))
        else:
            query_probe = F.bind(v_sub, v_rel)

        v_answer = F.bind(axiom_map, query_probe).squeeze(0)

        cos_all = torch.nn.functional.cosine_similarity(
            v_answer.float().unsqueeze(0).expand_as(obj_hvs),
            obj_hvs.float(),
            dim=1,
        )
        best_cos = cos_all.max().item()
        best_idx = cos_all.argmax().item()
        best_obj = all_objects[best_idx]

        if q["expected"] == "supported":
            pos_cosines.append(best_cos)
            retrieval_total += 1
            if best_obj == q.get("gold_object"):
                retrieval_correct += 1
        else:
            neg_cosines.append(best_cos)

    retrieval_acc = retrieval_correct / max(retrieval_total, 1)
    avg_pos = sum(pos_cosines) / max(len(pos_cosines), 1)
    avg_neg = sum(neg_cosines) / max(len(neg_cosines), 1)

    # Optimal F1
    all_cosines = [(c, True) for c in pos_cosines] + [(c, False)
                                                      for c in neg_cosines]
    all_cosines.sort(key=lambda x: x[0])
    best_f1 = 0.0
    best_tau = 0.0
    thresholds = sorted(set(c for c, _ in all_cosines))
    for tau in thresholds:
        s_tp = sum(1 for c in pos_cosines if c >= tau)
        s_fp = sum(1 for c in neg_cosines if c >= tau)
        s_fn = sum(1 for c in pos_cosines if c < tau)
        s_prec = s_tp / max(s_tp + s_fp, 1)
        s_rec = s_tp / max(s_tp + s_fn, 1)
        s_f1 = (2 * s_prec * s_rec) / max(s_prec + s_rec, 1e-9)
        if s_f1 > best_f1:
            best_f1 = s_f1
            best_tau = tau

    return {
        "retrieval_accuracy": round(retrieval_acc, 4),
        "avg_positive_cosine": round(avg_pos, 6),
        "avg_negative_cosine": round(avg_neg, 6),
        "cosine_separation": round(avg_pos - avg_neg, 6),
        "optimal_tau": round(best_tau, 6),
        "optimal_f1": round(best_f1, 4),
    }


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------

def _run_full_axiom(fact_tuples, test_queries) -> dict:
    """Full AXIOM system with HDC + roles."""
    logger.info("  [1] Full AXIOM")
    medical_facts = [MedicalFact(s, r, o) for s, r, o in fact_tuples]
    distiller = AxiomDistiller()
    distiller.distill(medical_facts)
    return _evaluate_retrieval(
        distiller.axiom_map, distiller.item_memory,
        fact_tuples, test_queries, use_roles=True,
    )


def _run_no_binarization(fact_tuples, test_queries) -> dict:
    """HDC superposition without binarization (continuous map)."""
    logger.info("  [2] No binarization (continuous map)")
    medical_facts = [MedicalFact(s, r, o) for s, r, o in fact_tuples]
    distiller = AxiomDistiller()

    # Distill but skip cleanup (binarization)
    for fact in medical_facts:
        fact_hv = distiller.encode_fact(fact)
        distiller.axiom_map = distiller.axiom_map + fact_hv

    # Do NOT call _cleanup — keep continuous superposition
    return _evaluate_retrieval(
        distiller.axiom_map, distiller.item_memory,
        fact_tuples, test_queries, use_roles=True,
    )


def _run_no_roles(fact_tuples, test_queries) -> dict:
    """No role-based permutation encoding — naive binding."""
    logger.info("  [3] No role permutation")
    distiller = AxiomDistiller()

    for s, r, o in fact_tuples:
        v_s = distiller.item_memory.get(s)
        v_r = distiller.item_memory.get(r)
        v_o = distiller.item_memory.get(o)
        # Bind without cyclic shift (no role encoding)
        fact_hv = F.bind(F.bind(v_s, v_r), v_o)
        distiller.axiom_map = distiller.axiom_map + fact_hv

    distiller.axiom_map = distiller._cleanup(distiller.axiom_map)

    return _evaluate_retrieval(
        distiller.axiom_map, distiller.item_memory,
        fact_tuples, test_queries, use_roles=False,
    )


def _run_lower_dim(fact_tuples, test_queries, dim: int) -> dict:
    """Test with reduced dimensionality."""
    logger.info("  [4] D=%d", dim)

    # Override dimensionality temporarily (frozen dataclass workaround)
    original_dim = hdc.dimensions
    object.__setattr__(hdc, 'dimensions', dim)

    medical_facts = [MedicalFact(s, r, o) for s, r, o in fact_tuples]
    distiller = AxiomDistiller()
    distiller.distill(medical_facts)

    result = _evaluate_retrieval(
        distiller.axiom_map, distiller.item_memory,
        fact_tuples, test_queries, use_roles=True,
    )

    # Restore original dimensionality
    object.__setattr__(hdc, 'dimensions', original_dim)
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_ablation_benchmark() -> dict:
    """Run the full ablation study."""
    setup_logging()

    logger.info("=" * 60)
    logger.info("AXIOM Ablation Study")
    logger.info("=" * 60)

    fact_tuples = _build_knowledge_base()
    test_queries = _build_test_queries(fact_tuples)

    logger.info("Knowledge base: %d facts, %d queries",
                len(fact_tuples), len(test_queries))

    results = {"benchmark": "ablation_study", "variants": {}}

    # (1) Full AXIOM
    results["variants"]["full_axiom"] = _run_full_axiom(
        fact_tuples, test_queries)
    logger.info("    → Retrieval: %.1f%%, F1: %.4f",
                results["variants"]["full_axiom"]["retrieval_accuracy"] * 100,
                results["variants"]["full_axiom"]["optimal_f1"])

    # (2) No binarization
    results["variants"]["no_binarization"] = _run_no_binarization(
        fact_tuples, test_queries)
    logger.info("    → Retrieval: %.1f%%, F1: %.4f",
                results["variants"]["no_binarization"]["retrieval_accuracy"] * 100,
                results["variants"]["no_binarization"]["optimal_f1"])

    # (3) No role permutation
    results["variants"]["no_roles"] = _run_no_roles(fact_tuples, test_queries)
    logger.info("    → Retrieval: %.1f%%, F1: %.4f",
                results["variants"]["no_roles"]["retrieval_accuracy"] * 100,
                results["variants"]["no_roles"]["optimal_f1"])

    # (4) Dimensionality ablation
    dim_results = {}
    for dim in [1000, 2000, 5000, 10000, 20000]:
        logger.info("  [dim] D=%d", dim)
        dim_results[str(dim)] = _run_lower_dim(fact_tuples, test_queries, dim)
        logger.info("    → Retrieval: %.1f%%, F1: %.4f",
                    dim_results[str(dim)]["retrieval_accuracy"] * 100,
                    dim_results[str(dim)]["optimal_f1"])
    results["variants"]["dim_sweep"] = dim_results

    # Summary table
    logger.info("-" * 60)
    logger.info("ABLATION SUMMARY:")
    logger.info("  %-25s  Ret.Acc   F1     Sep", "Variant")
    logger.info("  %-25s  ------  ------  ------", "-------")
    for name in ["full_axiom", "no_binarization", "no_roles"]:
        v = results["variants"][name]
        logger.info("  %-25s  %.1f%%   %.4f  %.4f",
                    name, v["retrieval_accuracy"] * 100,
                    v["optimal_f1"], v["cosine_separation"])
    for dim_str, v in dim_results.items():
        logger.info("  %-25s  %.1f%%   %.4f  %.4f",
                    f"D={dim_str}", v["retrieval_accuracy"] * 100,
                    v["optimal_f1"], v["cosine_separation"])
    logger.info("-" * 60)

    data_cfg.ensure_dirs()
    out_path = data_cfg.results_dir / "ablation_benchmark.json"
    save_json(results, out_path)
    logger.info("Results saved → %s", out_path)

    return results


if __name__ == "__main__":
    run_ablation_benchmark()
