#!/usr/bin/env python3
"""
AXIOM — Theoretical Capacity Validation

Tests three predictions from §3.4 of the paper against empirical
measurements using Bipolar MAP (Multiply-Add-Permute) hypervectors.

Key formulas for bipolar {-1, +1} MAP vectors:

  1. Detection SNR:    SNR_d = E[⟨rec, true⟩] / Std(⟨rec, true⟩) = √(d/(N-1))
  2. Cosine Decay:     E[cos(v̂, v*)] = 1/√N  (continuous superposition)
  3. Capacity Bound:   N_max ≤ d / (2·(Φ⁻¹(1 - α/|M|))²) + 1

We test in TWO modes:
  - Continuous: no sign() binarization — validates the closed-form exactly
  - Binarized:  with sign() cleanup     — the actual AXIOM implementation

Outputs:
  - Console table with THEORY vs EMPIRICAL columns
  - benchmarks/results/theory_validation.json
  - benchmarks/results/theory_vs_empirical.png

Security: No network calls; all synthetic data; deterministic seeds.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Optional: matplotlib for plots
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Theoretical predictions (Bipolar MAP model)
# ---------------------------------------------------------------------------

def theoretical_detection_snr(d: int, n: int) -> float:
    """
    Detection SNR = E[⟨recovered, true⟩] / Std(⟨recovered, true⟩).

    For bipolar MAP:
      E[inner product]  = d       (signal contribution)
      Var(inner product) = d(N-1) (noise contribution)
      SNR_d = d / √(d(N-1)) = √(d/(N-1))
    """
    if n <= 1:
        return float("inf")
    return math.sqrt(d / (n - 1))


def theoretical_cosine_continuous(d: int, n: int) -> float:
    """
    E[cos] for continuous (non-binarized) bipolar MAP superposition.

    recovered = gt + Σ noise_i,  ||gt||=√d,  E[||noise||²]=d(N-1)
    cos ≈ d / (√d · √(dN)) = 1/√N

    Exact: 1/√(1 + (N-1))  = 1/√N.
    """
    if n <= 1:
        return 1.0
    return 1.0 / math.sqrt(n)


def theoretical_capacity_bound(
    d: int, vocab_size: int, alpha: float = 0.05
) -> float:
    """N_max ≤ d / (2·c²) + 1 where c = Φ⁻¹(1 − α/|M|)."""
    from scipy.stats import norm

    c = norm.ppf(1.0 - alpha / vocab_size)
    return d / (2.0 * c * c) + 1.0


def theoretical_recall_approx(
    d: int, n: int, vocab_size: int
) -> float:
    """
    Approximate NN-Recall@1 probability using Gaussian approximation.

    For each of |M| distractors, P(beats signal) ≈ Φ(−SNR/√2).
    Union bound: P_err ≈ |M| · Φ(−SNR/√2).
    """
    from scipy.stats import norm

    if n <= 1:
        return 1.0
    snr = theoretical_detection_snr(d, n)
    p_single_distractor = norm.cdf(-snr / math.sqrt(2.0))
    p_err = vocab_size * p_single_distractor
    return max(0.0, min(1.0, 1.0 - p_err))


# ---------------------------------------------------------------------------
# Empirical measurement
# ---------------------------------------------------------------------------

def run_empirical_trial(
    d: int,
    n_facts: int,
    n_probes: int = 200,
    seed: int = 42,
    binarize: bool = True,
    device: str = "cpu",
) -> dict:
    """
    Bundle n_facts into a single superposition vector of dimension d.

    Args:
        binarize: If True, apply sign() cleanup (actual AXIOM).
                  If False, keep continuous superposition (theory test).

    Returns dict with keys: mean_cos, std_cos, recall_at_1,
    detection_snr, vocab_size, n_facts, d.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)

    # --- generate unique facts ---
    rng = random.Random(seed)
    n_ents = max(100, int(math.ceil(math.sqrt(n_facts) * 3)))
    subjects = [f"S{i:06d}" for i in range(n_ents)]
    relations = [f"R{i:04d}" for i in range(12)]
    objects = [f"O{i:06d}" for i in range(n_ents)]

    seen: set[tuple[str, str, str]] = set()
    facts: list[tuple[str, str, str]] = []
    while len(facts) < n_facts:
        t = (rng.choice(subjects), rng.choice(relations), rng.choice(objects))
        if t not in seen:
            seen.add(t)
            facts.append(t)

    # --- item memory ---
    im: dict[str, torch.Tensor] = {}

    def get(name: str) -> torch.Tensor:
        if name not in im:
            im[name] = F.random(1, d, device=dev)
        return im[name]

    # --- encode & bundle ---
    S = torch.zeros(1, d, device=dev)
    for s, r, o in facts:
        v_s, v_r, v_o = get(s), get(r), get(o)
        R = F.bind(F.bind(v_s, _cyclic_shift(v_r, 1)), _cyclic_shift(v_o, 2))
        S = S + R

    if binarize:
        S = torch.sign(S)
        S[S == 0] = 1.0

    # --- probe ---
    probe_idx = rng.sample(range(len(facts)), min(n_probes, len(facts)))

    # build object vocabulary matrix for NN search
    obj_names = sorted([k for k in im if k.startswith("O")])
    vocab_size = len(obj_names)

    cosines: list[float] = []
    inner_products: list[float] = []
    nn_correct = 0

    for idx in probe_idx:
        s, r, o = facts[idx]
        v_s, v_r = im[s], im[r]

        probe = F.bind(v_s, _cyclic_shift(v_r, 1))
        v_recovered = F.bind(S, probe)

        gt_shifted = _cyclic_shift(im[o], 2)

        # inner product with ground truth (for detection SNR)
        ip = torch.dot(
            v_recovered.float().squeeze(), gt_shifted.float().squeeze()
        ).item()
        inner_products.append(ip)

        # cosine with ground truth
        cos_val = torch.nn.functional.cosine_similarity(
            v_recovered.float(), gt_shifted.float()
        ).item()
        cosines.append(cos_val)

        # NN lookup (against shifted object vocab)
        shifted_obj_matrix = torch.cat(
            [_cyclic_shift(im[n], 2) for n in obj_names], dim=0
        )
        sims = torch.nn.functional.cosine_similarity(
            v_recovered.float(), shifted_obj_matrix.float()
        )
        best_idx = sims.argmax().item()
        if obj_names[best_idx] == o:
            nn_correct += 1

    mean_cos = float(np.mean(cosines))
    std_cos = float(np.std(cosines))
    recall = nn_correct / len(probe_idx)

    # Detection SNR = mean(inner_product) / std(inner_product)
    ip_arr = np.array(inner_products)
    det_snr = float(np.mean(ip_arr) / max(np.std(ip_arr), 1e-12))

    return {
        "d": d,
        "n_facts": n_facts,
        "vocab_size": vocab_size,
        "binarized": binarize,
        "mean_cos": round(mean_cos, 6),
        "std_cos": round(std_cos, 6),
        "recall_at_1": round(recall, 4),
        "detection_snr": round(det_snr, 4),
        "mean_inner_product": round(float(np.mean(ip_arr)), 2),
    }


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------

def run_validation(
    dim: int = 10_000,
    fact_counts: list[int] | None = None,
    n_probes: int = 200,
    seed: int = 42,
) -> dict:
    """
    Run theory vs. empirical comparison for continuous and binarized modes.
    """
    if fact_counts is None:
        fact_counts = [10, 50, 100, 200, 500, 1_000, 2_000]

    continuous_results: list[dict] = []
    binarized_results: list[dict] = []

    for mode_label, binarize, results_list in [
        ("CONTINUOUS (no sign)", False, continuous_results),
        ("BINARIZED  (sign())", True, binarized_results),
    ]:
        print(f"\n{'='*95}")
        print(f"  {mode_label}   D = {dim:,}")
        print(f"{'='*95}")
        header = (
            f"{'N':>7} │ {'SNR_thy':>8} {'SNR_emp':>8} {'Δ%':>6} │ "
            f"{'cos_thy':>8} {'cos_emp':>8} {'Δ%':>6} │ "
            f"{'R@1':>6} │ {'Match':>5}"
        )
        print(header)
        print("─" * 95)

        for n in fact_counts:
            print(f"  N={n:,}...", end="", flush=True)
            t0 = time.perf_counter()

            emp = run_empirical_trial(
                dim, n, n_probes=n_probes, seed=seed, binarize=binarize
            )
            t_elapsed = time.perf_counter() - t0

            # Theoretical predictions (for continuous model)
            snr_thy = theoretical_detection_snr(dim, n)
            cos_thy = theoretical_cosine_continuous(dim, n)

            snr_emp = emp["detection_snr"]
            cos_emp = emp["mean_cos"]
            recall_emp = emp["recall_at_1"]

            # relative errors
            snr_err = abs(snr_thy - snr_emp) / max(snr_thy, 1e-12) * 100
            cos_err = abs(cos_thy - cos_emp) / max(abs(cos_thy), 1e-12) * 100

            match = "✓" if snr_err < 25 and cos_err < 25 else "~"

            row = {
                "n_facts": n,
                "d": dim,
                "binarized": binarize,
                "vocab_size": emp["vocab_size"],
                "snr_theory": round(snr_thy, 4),
                "cos_theory": round(cos_thy, 6),
                "snr_empirical": snr_emp,
                "cos_empirical": cos_emp,
                "recall_empirical": recall_emp,
                "snr_rel_error_pct": round(snr_err, 1),
                "cos_rel_error_pct": round(cos_err, 1),
                "match": match == "✓",
                "time_s": round(t_elapsed, 2),
            }
            results_list.append(row)

            print(
                f"\r{n:>7,} │ {snr_thy:>8.2f} {snr_emp:>8.2f} {snr_err:>5.1f}% │ "
                f"{cos_thy:>8.4f} {cos_emp:>8.4f} {cos_err:>5.1f}% │ "
                f"{recall_emp:>5.1%} │ {match:>5}  ({t_elapsed:.1f}s)"
            )

    return {
        "continuous": continuous_results,
        "binarized": binarized_results,
    }


def run_capacity_bound_test(dim: int = 10_000) -> dict:
    """
    Test capacity bound: theoretical N_max vs empirical boundary.

    Sweep N to find where Recall@1 drops below 90% (binarized mode).
    """
    test_ns = [50, 100, 150, 200, 250, 300, 400, 500]
    print(f"\n{'='*70}")
    print(f"  Capacity Bound Test (binarized): Recall@1 < 90% boundary")
    print(f"{'='*70}")

    empirical_boundary = None
    for n in test_ns:
        emp = run_empirical_trial(dim, n, n_probes=200, seed=42, binarize=True)
        recall = emp["recall_at_1"]
        print(
            f"  N={n:>4}  Recall@1={recall:.1%}  (vocab={emp['vocab_size']})")
        if recall < 0.90 and empirical_boundary is None:
            empirical_boundary = n

    avg_vocab = 300
    n_max_thy = theoretical_capacity_bound(dim, avg_vocab, alpha=0.05)

    print(
        f"\n  Theoretical N_max (95% conf, |M|={avg_vocab}): {n_max_thy:.0f}")
    if empirical_boundary:
        print(f"  Empirical boundary (Recall@1 < 90%): {empirical_boundary}")
        pct_diff = abs(n_max_thy - empirical_boundary) / n_max_thy * 100
        print(f"  Difference: {pct_diff:.1f}%")
    else:
        print(f"  Empirical boundary: Recall still ≥ 90% at N={test_ns[-1]}")

    return {
        "theoretical_n_max": round(n_max_thy, 1),
        "empirical_boundary": empirical_boundary,
        "vocab_size_used": avg_vocab,
        "dimension": dim,
    }


def plot_theory_vs_empirical(
    continuous: list[dict],
    binarized: list[dict],
    outdir: Path,
) -> None:
    """Generate theory vs. empirical comparison figure (3 panels)."""
    if not HAS_MPL:
        print("  [skip] matplotlib not available, skipping plot")
        return

    ns_c = [r["n_facts"] for r in continuous]
    ns_b = [r["n_facts"] for r in binarized]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- Panel 1: Detection SNR ---
    ax = axes[0]
    ax.plot(ns_c, [r["snr_theory"] for r in continuous], "o--",
            color="#1E5AB4", label="Theory: √(d/(N−1))", linewidth=2, zorder=3)
    ax.plot(ns_c, [r["snr_empirical"] for r in continuous], "s",
            color="#2E7D32", label="Continuous", markersize=7, zorder=4)
    ax.plot(ns_b, [r["snr_empirical"] for r in binarized], "^",
            color="#E64A19", label="Binarized", markersize=7, zorder=4)
    ax.set_xlabel("N (bundled facts)")
    ax.set_ylabel("Detection SNR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Detection SNR = E[⟨rec,true⟩]/σ")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Mean Cosine ---
    ax = axes[1]
    ax.plot(ns_c, [r["cos_theory"] for r in continuous], "o--",
            color="#1E5AB4", label="Theory: 1/√N", linewidth=2, zorder=3)
    ax.plot(ns_c, [r["cos_empirical"] for r in continuous], "s",
            color="#2E7D32", label="Continuous", markersize=7, zorder=4)
    ax.plot(ns_b, [r["cos_empirical"] for r in binarized], "^",
            color="#E64A19", label="Binarized", markersize=7, zorder=4)
    ax.set_xlabel("N (bundled facts)")
    ax.set_ylabel("E[cos(v̂, v*)]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Cosine Decay Law")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Recall@1 ---
    ax = axes[2]
    ax.plot(ns_c, [r["recall_empirical"] * 100 for r in continuous], "s-",
            color="#2E7D32", label="Continuous", linewidth=2, markersize=6)
    ax.plot(ns_b, [r["recall_empirical"] * 100 for r in binarized], "^-",
            color="#E64A19", label="Binarized", linewidth=2, markersize=6)
    ax.axhline(y=90, color="#999", linestyle=":", linewidth=1, label="90%")
    ax.set_xlabel("N (bundled facts)")
    ax.set_ylabel("NN-Recall@1 (%)")
    ax.set_xscale("log")
    ax.set_title("Retrieval Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"AXIOM Theory vs. Empirical Validation (d = {continuous[0]['d']:,})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        outpath = outdir / f"theory_vs_empirical.{ext}"
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {outpath}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    dim = 10_000
    n_probes = 200

    # 1. Core validation: continuous & binarized
    all_results = run_validation(dim=dim, n_probes=n_probes)

    # 2. Capacity bound test (binarized)
    bound_result = run_capacity_bound_test(dim=dim)

    # 3. Save results
    outdir = Path(__file__).resolve().parent / "results"
    outdir.mkdir(exist_ok=True)

    payload = {
        "experiment": "theory_validation",
        "dimension": dim,
        "n_probes": n_probes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "continuous_results": all_results["continuous"],
        "binarized_results": all_results["binarized"],
        "capacity_bound": bound_result,
    }

    outpath = outdir / "theory_validation.json"
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to {outpath}")

    # 4. Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")

    for label, res_list in [
        ("Continuous (theory test)", all_results["continuous"]),
        ("Binarized  (AXIOM impl)", all_results["binarized"]),
    ]:
        matched = sum(1 for r in res_list if r["match"])
        total = len(res_list)
        m_snr = np.mean([r["snr_rel_error_pct"] for r in res_list])
        m_cos = np.mean([r["cos_rel_error_pct"] for r in res_list])
        print(
            f"  {label}: {matched}/{total} match  "
            f"(SNR err {m_snr:.1f}%, cos err {m_cos:.1f}%)"
        )

    print(
        f"  Theoretical N_max: {bound_result['theoretical_n_max']:.0f} facts")
    print(
        f"  Empirical boundary: {bound_result['empirical_boundary'] or '>500'}")

    # 5. Plot
    plot_theory_vs_empirical(
        all_results["continuous"], all_results["binarized"], outdir
    )

    print(f"\n  Done.\n")


if __name__ == "__main__":
    main()
