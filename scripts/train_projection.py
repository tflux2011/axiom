#!/usr/bin/env python3
"""
AXIOM — Projection Training Script

Trains the W1/W2 low-rank projection matrices that map the
Axiom Map (HDC space) into the SLM's KV-cache (transformer hidden space).

The projection is the ONLY learned component; the SLM backbone stays frozen.

Training procedure (from paper Section 3.2):
  1. Generate QA pairs from the knowledge graph.
  2. Project Axiom Map through W1(S) -> z, W2(z) -> h -> virtual tokens.
  3. Inject virtual tokens at layer L via forward hook.
  4. Compute next-token prediction loss on reference answers.
  5. Backprop into W1 and W2 only (SLM frozen).

Usage:
    python scripts/train_projection.py --model gpt2 --epochs 10

Security:
    - No hardcoded credentials.
    - HF_TOKEN read from environment only.
    - All data is synthetic / from our curated KG.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from axiom_hdc.config import HDCConfig, ModelConfig
from axiom_hdc.distiller import AxiomDistiller, MedicalFact
from axiom_hdc.priming import AxiomProjector, load_base_model
from axiom_hdc.utils import setup_logging, save_json

logger = logging.getLogger("axiom.train")


# ---------------------------------------------------------------------------
# QA pair generation from knowledge graph
# ---------------------------------------------------------------------------

def generate_qa_pairs(
    facts: list[tuple[str, str, str]],
    n_pairs: int = 5000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate medical QA pairs from knowledge graph triples.

    Templates vary by relation type to create diverse training signal.
    """
    rng = random.Random(seed)
    templates = {
        "TREATS": [
            ("What does {s} treat?", "{s} treats {o}."),
            ("What is {s} used for?", "{s} is used for {o}."),
            ("Which condition is treated by {s}?", "{o} is treated by {s}."),
        ],
        "CAUSES": [
            ("What side effects does {s} cause?", "{s} can cause {o}."),
            ("What does {s} cause?", "{s} causes {o}."),
        ],
        "CONTRAINDICATES": [
            ("What are the contraindications for {s}?",
             "{s} is contraindicated in {o}."),
            ("When should {s} not be used?",
             "{s} should not be used with {o}."),
        ],
        "INHIBITS": [
            ("What does {s} inhibit?", "{s} inhibits {o}."),
            ("What is the mechanism of {s}?",
             "{s} works by inhibiting {o}."),
        ],
        "REGULATES": [
            ("What does {s} regulate?", "{s} regulates {o}."),
            ("How does {s} work?",
             "{s} works by regulating {o}."),
        ],
        "INTERACTS_WITH": [
            ("Does {s} interact with {o}?",
             "Yes, {s} interacts with {o}."),
            ("What are the drug interactions of {s}?",
             "{s} has a known interaction with {o}."),
        ],
    }

    # Default template for unknown relations
    default_templates = [
        ("What is the relationship between {s} and {o}?",
         "{s} {r} {o}."),
    ]

    pairs = []
    for _ in range(n_pairs):
        s, r, o = rng.choice(facts)
        s_clean = s.replace("_", " ")
        o_clean = o.replace("_", " ")
        r_clean = r.replace("_", " ").lower()

        rel_templates = templates.get(r, default_templates)
        q_template, a_template = rng.choice(rel_templates)

        question = q_template.format(s=s_clean, o=o_clean, r=r_clean)
        answer = a_template.format(s=s_clean, o=o_clean, r=r_clean)

        pairs.append({
            "question": question,
            "answer": answer,
            "subject": s,
            "relation": r,
            "object": o,
        })

    return pairs


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_projection(
    model_id: str = "gpt2",
    n_epochs: int = 10,
    n_qa_pairs: int = 2000,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-2,
    batch_size: int = 4,
    max_seq_len: int = 128,
    bottleneck_dim: int = 512,
    num_virtual_tokens: int = 64,
    injection_layer: int | None = None,
    hdc_dim: int = 10_000,
    device_str: str = "cpu",
    seed: int = 42,
    output_dir: str = "data/projection",
) -> dict:
    """
    Train the W1/W2 projection matrices.

    Args:
        model_id: HuggingFace model identifier (default: gpt2).
        n_epochs: Training epochs.
        n_qa_pairs: Number of QA pairs for training.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size.
        max_seq_len: Maximum sequence length for tokenisation.
        bottleneck_dim: Bottleneck dimension r.
        num_virtual_tokens: Number of virtual tokens k.
        injection_layer: Layer L for injection (default: midpoint).
        hdc_dim: HDC dimension d.
        device_str: Device string (cpu/mps/cuda).
        seed: Random seed for reproducibility.
        output_dir: Where to save trained projection weights.

    Returns:
        Training results dict with loss history and metrics.
    """
    setup_logging()
    torch.manual_seed(seed)
    device = torch.device(device_str)

    logger.info("=" * 60)
    logger.info("AXIOM Projection Training")
    logger.info("  Model: %s", model_id)
    logger.info("  Epochs: %d, LR: %.1e, Batch: %d", n_epochs, learning_rate, batch_size)
    logger.info("  Virtual tokens k=%d, Bottleneck r=%d", num_virtual_tokens, bottleneck_dim)
    logger.info("  HDC dim d=%d", hdc_dim)
    logger.info("=" * 60)

    # ---- 1. Build Axiom Map from synthetic KG ----
    from benchmarks.accuracy_bench import _build_knowledge_base
    fact_tuples = _build_knowledge_base()
    medical_facts = [MedicalFact(s, r, o) for s, r, o in fact_tuples]

    hdc_cfg = HDCConfig(dimensions=hdc_dim)
    distiller = AxiomDistiller(cfg=hdc_cfg, device=device, seed=seed)
    distiller.distill(medical_facts)
    axiom_map = distiller.axiom_map.to(device)

    logger.info("Axiom Map: %d facts distilled, shape=%s",
                distiller.fact_count, axiom_map.shape)

    # ---- 2. Generate QA training pairs ----
    qa_pairs = generate_qa_pairs(fact_tuples, n_pairs=n_qa_pairs, seed=seed)
    logger.info("Generated %d QA pairs", len(qa_pairs))

    # ---- 3. Load frozen model ----
    model_cfg = ModelConfig(
        model_id=model_id,
        quantisation="none",  # No quantisation for training
        device=device_str,
    )
    model, tokenizer = load_base_model(model_cfg)
    model = model.to(device)

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    model_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers

    # Determine injection layer (midpoint by default)
    if injection_layer is None:
        injection_layer = n_layers // 2

    logger.info("Model: %s — d_model=%d, n_layers=%d, injection_layer=%d",
                model_id, model_dim, n_layers, injection_layer)

    # ---- 4. Create trainable projector ----
    projector = AxiomProjector(
        hdc_dim=hdc_dim,
        model_dim=model_dim,
        num_virtual_tokens=num_virtual_tokens,
        bottleneck_dim=bottleneck_dim,
    ).to(device)

    total_proj_params = sum(p.numel() for p in projector.parameters())
    logger.info("Projector: %.1fM trainable params", total_proj_params / 1e6)

    # ---- 5. Set up optimiser ----
    optimiser = AdamW(
        projector.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # ---- 6. Training loop ----
    # Uses embedding-level injection: virtual tokens are prepended to
    # the input embeddings and passed via inputs_embeds, avoiding shape
    # mismatches from mid-layer hook injection during training.
    loss_history = []
    best_loss = float("inf")
    t_start = time.perf_counter()

    for epoch in range(n_epochs):
        epoch_losses = []
        random.shuffle(qa_pairs)

        for batch_start in range(0, len(qa_pairs), batch_size):
            batch = qa_pairs[batch_start:batch_start + batch_size]

            # Tokenise the QA pair as: "Q: ... A: ..."
            texts = [
                f"Q: {p['question']} A: {p['answer']}"
                for p in batch
            ]
            encoding = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            # Project Axiom Map -> virtual tokens
            virtual_tokens = projector(axiom_map)  # (1, k, d_model)
            vt_expanded = virtual_tokens.expand(
                input_ids.size(0), -1, -1
            )  # (B, k, d_model)

            # Embedding-level injection: prepend virtual tokens to input
            # embeddings and pass inputs_embeds instead of input_ids.
            # This avoids shape mismatches from mid-layer hook injection.
            with torch.no_grad():
                input_embeds = model.transformer.wte(input_ids)  # (B, S, d_model)

            # Prepend virtual tokens to embeddings
            augmented_embeds = torch.cat(
                [vt_expanded.to(input_embeds.dtype), input_embeds], dim=1
            )  # (B, k+S, d_model)

            # Extend attention mask for virtual tokens
            vt_mask = torch.ones(
                input_ids.size(0), num_virtual_tokens,
                device=device, dtype=attention_mask.dtype,
            )
            extended_mask = torch.cat([vt_mask, attention_mask], dim=1)

            # Labels: shift input_ids right; -100 for padding
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            # Prepend -100 for virtual token positions (no loss there)
            vt_labels = torch.full(
                (input_ids.size(0), num_virtual_tokens),
                -100, device=device, dtype=labels.dtype,
            )
            extended_labels = torch.cat([vt_labels, labels], dim=1)

            outputs = model(
                inputs_embeds=augmented_embeds,
                attention_mask=extended_mask,
                labels=extended_labels,
            )
            loss = outputs.loss

            # Backward (only projector params have grad)
            optimiser.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)

            optimiser.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        logger.info(
            "Epoch %d/%d — avg loss: %.4f  (best: %.4f)",
            epoch + 1, n_epochs, avg_loss, best_loss,
        )

    training_time = time.perf_counter() - t_start

    # ---- 8. Save projection weights ----
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    proj_state = {
        "projector_state_dict": projector.state_dict(),
        "config": {
            "hdc_dim": hdc_dim,
            "model_dim": model_dim,
            "num_virtual_tokens": num_virtual_tokens,
            "bottleneck_dim": bottleneck_dim,
            "injection_layer": injection_layer,
            "model_id": model_id,
        },
    }
    torch.save(proj_state, out_path / "projector.pt")
    logger.info("Projector saved -> %s", out_path / "projector.pt")

    # ---- 9. Compile results ----
    results = {
        "model_id": model_id,
        "model_dim": model_dim,
        "n_layers": n_layers,
        "injection_layer": injection_layer,
        "hdc_dim": hdc_dim,
        "bottleneck_dim": bottleneck_dim,
        "num_virtual_tokens": num_virtual_tokens,
        "total_projection_params": total_proj_params,
        "n_epochs": n_epochs,
        "n_qa_pairs": n_qa_pairs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "training_time_s": round(training_time, 1),
        "loss_history": [round(l, 4) for l in loss_history],
        "final_loss": round(loss_history[-1], 4),
        "best_loss": round(best_loss, 4),
    }

    save_json(results, out_path / "training_results.json")
    logger.info("Training complete in %.1f s", training_time)
    logger.info("Final loss: %.4f, Best loss: %.4f", loss_history[-1], best_loss)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train AXIOM projection")
    parser.add_argument("--model", default="gpt2",
                        help="HuggingFace model ID (default: gpt2)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--qa-pairs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bottleneck", type=int, default=512)
    parser.add_argument("--virtual-tokens", type=int, default=64)
    parser.add_argument("--injection-layer", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="data/projection")
    args = parser.parse_args()

    train_projection(
        model_id=args.model,
        n_epochs=args.epochs,
        n_qa_pairs=args.qa_pairs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        bottleneck_dim=args.bottleneck,
        num_virtual_tokens=args.virtual_tokens,
        injection_layer=args.injection_layer,
        device_str=args.device,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
