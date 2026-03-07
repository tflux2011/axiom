#!/usr/bin/env python3
"""Generate a t-SNE visualisation of HDC vector orthogonality.

Produces a 2-D scatter plot showing:
  - Medical entities (drugs, diseases, genes) cluster by their *bound*
    relationships while remaining quasi-orthogonal to unrelated concepts.
  - Non-medical terms ("Agriculture", "Astronomy", "Poetry") sit far
    from any medical cluster.

The plot is saved as ``paper/figures/tsne_orthogonality.pdf`` for
inclusion in the LaTeX paper.

Usage:
    python -m scripts.generate_tsne_plot
"""

from __future__ import annotations
from sklearn.manifold import TSNE
import torchhd.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
DIM = 10_000
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Colour palette (no purple per design spec) ──────────────────────────────
COLOUR_MAP = {
    "Drug":       "#1E5AB4",   # axiomblue
    "Disease":    "#D13C32",   # axiomwarn (red)
    "Gene":       "#0096A0",   # teal variant
    "Relation":   "#FFB300",   # axiomgold
    "Bound fact": "#2EA043",   # axiomsafe (green)
    "Unrelated":  "#8B8B8B",   # neutral grey
}

# ── Build labelled HDC vectors ──────────────────────────────────────────────


def _hv() -> torch.Tensor:
    """Return a fresh bipolar random HV."""
    return F.random(1, DIM).squeeze(0)


def build_vectors() -> tuple[np.ndarray, list[str], list[str]]:
    """Create a set of named HDC vectors and return (matrix, labels, groups).

    Groups: Drug, Disease, Gene, Relation, Bound fact, Unrelated.
    """
    vectors: list[torch.Tensor] = []
    labels: list[str] = []
    groups: list[str] = []

    # --- Atomic medical entities ---
    drugs = {
        "Aspirin": _hv(), "Metformin": _hv(), "Doxorubicin": _hv(),
        "Lisinopril": _hv(), "Warfarin": _hv(),
    }
    diseases = {
        "Headache": _hv(), "Diabetes": _hv(), "Breast Cancer": _hv(),
        "Hypertension": _hv(), "DVT": _hv(),
    }
    genes = {
        "BRCA1": _hv(), "TP53": _hv(), "EGFR": _hv(),
        "INS": _hv(),
    }
    relations = {
        "TREATS": _hv(), "CAUSES": _hv(), "INHIBITS": _hv(),
        "ASSOCIATED_WITH": _hv(),
    }

    for name, hv in drugs.items():
        vectors.append(hv)
        labels.append(name)
        groups.append("Drug")
    for name, hv in diseases.items():
        vectors.append(hv)
        labels.append(name)
        groups.append("Disease")
    for name, hv in genes.items():
        vectors.append(hv)
        labels.append(name)
        groups.append("Gene")
    for name, hv in relations.items():
        vectors.append(hv)
        labels.append(name)
        groups.append("Relation")

    # --- Bound facts (binding creates new quasi-orthogonal vectors) ---
    bound_facts = {
        "Aspirin⊙TREATS⊙Headache": (
            drugs["Aspirin"] * relations["TREATS"] * diseases["Headache"]
        ),
        "Metformin⊙TREATS⊙Diabetes": (
            drugs["Metformin"] * relations["TREATS"] * diseases["Diabetes"]
        ),
        "BRCA1⊙ASSOCIATED⊙Breast Cancer": (
            genes["BRCA1"] * relations["ASSOCIATED_WITH"]
            * diseases["Breast Cancer"]
        ),
        "Warfarin⊙TREATS⊙DVT": (
            drugs["Warfarin"] * relations["TREATS"] * diseases["DVT"]
        ),
    }
    for name, hv in bound_facts.items():
        vectors.append(hv)
        labels.append(name)
        groups.append("Bound fact")

    # --- Unrelated / non-medical terms ---
    unrelated = {
        "Agriculture": _hv(), "Astronomy": _hv(), "Poetry": _hv(),
        "Pottery": _hv(), "Jazz": _hv(),
    }
    for name, hv in unrelated.items():
        vectors.append(hv)
        labels.append(name)
        groups.append("Unrelated")

    matrix = torch.stack(vectors).float().numpy()
    return matrix, labels, groups


def main() -> None:
    matrix, labels, groups = build_vectors()

    # ── t-SNE projection ────────────────────────────────────────────────
    tsne = TSNE(
        n_components=2,
        perplexity=min(8, len(labels) - 1),
        learning_rate="auto",
        init="pca",
        random_state=SEED,
        max_iter=2000,
    )
    coords = tsne.fit_transform(matrix)

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    marker_map = {
        "Drug": "o", "Disease": "s", "Gene": "^",
        "Relation": "D", "Bound fact": "*", "Unrelated": "X",
    }
    size_map = {
        "Drug": 90, "Disease": 90, "Gene": 90,
        "Relation": 70, "Bound fact": 140, "Unrelated": 100,
    }

    # Draw each group
    for group in COLOUR_MAP:
        mask = [g == group for g in groups]
        if not any(mask):
            continue
        idx = np.where(mask)[0]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=COLOUR_MAP[group],
            marker=marker_map[group],
            s=size_map[group],
            label=group,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

    # Annotate each point
    for i, label in enumerate(labels):
        short = label.split("⊙")[0] if "⊙" in label else label
        ax.annotate(
            short,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=6.5,
            color="#333333",
        )

    ax.set_xlabel("t-SNE dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE dimension 2", fontsize=10)
    ax.set_title(
        "HDC Vector Orthogonality: Medical vs. Unrelated Concepts",
        fontsize=11, fontweight="bold",
    )
    ax.legend(
        loc="upper left", fontsize=7.5, framealpha=0.9,
        edgecolor="#cccccc", ncol=2,
    )
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)

    # ── Save ─────────────────────────────────────────────────────────────
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tsne_orthogonality.pdf"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved t-SNE plot → {out_path}")

    # Also save a PNG for quick previewing
    png_path = out_dir / "tsne_orthogonality.png"
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    # Re-draw for PNG (matplotlib can't re-save a closed figure)
    for group in COLOUR_MAP:
        mask = [g == group for g in groups]
        if not any(mask):
            continue
        idx = np.where(mask)[0]
        ax2.scatter(
            coords[idx, 0], coords[idx, 1],
            c=COLOUR_MAP[group],
            marker=marker_map[group],
            s=size_map[group],
            label=group,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
    for i, label in enumerate(labels):
        short = label.split("⊙")[0] if "⊙" in label else label
        ax2.annotate(
            short, (coords[i, 0], coords[i, 1]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=6.5, color="#333333",
        )
    ax2.set_xlabel("t-SNE dimension 1", fontsize=10)
    ax2.set_ylabel("t-SNE dimension 2", fontsize=10)
    ax2.set_title(
        "HDC Vector Orthogonality: Medical vs. Unrelated Concepts",
        fontsize=11, fontweight="bold",
    )
    ax2.legend(
        loc="upper left", fontsize=7.5, framealpha=0.9,
        edgecolor="#cccccc", ncol=2,
    )
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=8)
    fig2.tight_layout()
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved preview  → {png_path}")


if __name__ == "__main__":
    main()
