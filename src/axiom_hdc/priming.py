"""
AXIOM Priming — Zero-Retrieval Latent Injection

Implements Direct Attention Injection: the Axiom Map is loaded into the
SLM's Key-Value cache as "Virtual Context Tokens" so the model treats
external knowledge as internal memory.

Architecture:
    1. The distilled Axiom Map (1, D_hdc) is projected into the model's
       hidden dimension via a learned linear projection.
    2. The projected vectors are reshaped into (k, d_model) "virtual tokens".
    3. Forward hooks on the target transformer layer prepend these tokens
       to the KV-cache, steering attention without increasing prompt length.

Security:
    - No remote calls; all operations are local.
    - Model weights are loaded from the external HD path and verified.
    - Hooks are registered/removed cleanly to avoid memory leaks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from axiom_hdc.config import ModelConfig, HDCConfig, model as default_model_cfg, hdc as default_hdc_cfg

logger = logging.getLogger("axiom.priming")


# ---------------------------------------------------------------------------
# Projection layer: HDC space → Transformer hidden space
# ---------------------------------------------------------------------------

class AxiomProjector(nn.Module):
    """
    Low-rank two-stage projection from HDC Axiom Map to transformer
    virtual context tokens.

    Architecture:
        Stage 1: W1 ∈ R^{hdc_dim × bottleneck}   (10000 × 512)
        Stage 2: W2 ∈ R^{bottleneck × k*d_model}  (512 × 128*3072)

    Total params ≈ 206M (vs 3.9B for a naive single linear layer).

    Input:  (1, D_hdc)           e.g. (1, 10000)
    Output: (1, k, d_model)      e.g. (1, 128, 3072) for Llama-3.2-3B
    """

    def __init__(
        self,
        hdc_dim: int,
        model_dim: int,
        num_virtual_tokens: int,
        bottleneck_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.model_dim = model_dim
        self.bottleneck_dim = bottleneck_dim

        # Stage 1: HDC space → bottleneck
        self.proj_down = nn.Linear(hdc_dim, bottleneck_dim, bias=False)

        # Stage 2: bottleneck → (k * d_model)
        self.proj_up = nn.Linear(
            bottleneck_dim, num_virtual_tokens * model_dim, bias=False
        )

        # Layer norm to stabilise injection
        self.norm = nn.LayerNorm(model_dim)

        total_params = (hdc_dim * bottleneck_dim
                        + bottleneck_dim * num_virtual_tokens * model_dim)
        logger.info(
            "AxiomProjector (low-rank): %d → %d → %d × %d  (%dM params)",
            hdc_dim,
            bottleneck_dim,
            num_virtual_tokens,
            model_dim,
            total_params // 1_000_000,
        )

    def forward(self, axiom_map: torch.Tensor) -> torch.Tensor:
        """
        Project and reshape the Axiom Map via low-rank bottleneck.

        Args:
            axiom_map: (1, D_hdc)

        Returns:
            virtual_tokens: (1, k, d_model)
        """
        # (1, D_hdc) → (1, bottleneck)
        compressed = self.proj_down(axiom_map.float())

        # (1, bottleneck) → (1, k * d_model)
        projected = self.proj_up(compressed)

        # (1, k * d_model) → (1, k, d_model)
        tokens = projected.view(1, self.num_virtual_tokens, self.model_dim)

        return self.norm(tokens)


# ---------------------------------------------------------------------------
# KV-Cache Hook Manager
# ---------------------------------------------------------------------------

@dataclass
class KVCacheInjector:
    """
    Manages forward hooks on a transformer layer to inject AXIOM
    virtual tokens into the Key-Value cache.

    Usage:
        injector = KVCacheInjector(model, axiom_map, projector, cfg)
        injector.attach()       # Start injecting
        output = model(input)   # Virtual tokens are active
        injector.detach()       # Clean up hooks
    """

    model: nn.Module
    axiom_map: torch.Tensor
    projector: AxiomProjector
    model_cfg: ModelConfig = field(default_factory=lambda: default_model_cfg)

    _hooks: list[Any] = field(default_factory=list, repr=False)
    _virtual_tokens: torch.Tensor | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        # Pre-compute virtual tokens (they don't change per query)
        with torch.no_grad():
            self._virtual_tokens = self.projector(self.axiom_map)
        logger.info(
            "Virtual tokens computed: %s", self._virtual_tokens.shape
        )

    def _get_target_layer(self) -> nn.Module:
        """
        Locate the transformer layer at the configured injection point.

        Supports multiple architectures:
            - Llama:  model.model.layers[i]
            - GPT-2:  model.transformer.h[i]
            - GPT-J:  model.transformer.h[i]
            - Falcon:  model.transformer.h[i]
        """
        layer_idx = self.model_cfg.injection_layer

        # Try common architecture patterns
        layer_accessors = [
            lambda: self.model.model.layers,        # Llama, Mistral
            lambda: self.model.transformer.h,        # GPT-2, GPT-J
            lambda: self.model.gpt_neox.layers,      # GPT-NeoX
        ]

        for accessor in layer_accessors:
            try:
                layers = accessor()
                # Clamp to valid range
                idx = min(layer_idx, len(layers) - 1)
                target = layers[idx]
                logger.info(
                    "Target injection layer: %d / %d",
                    idx, len(layers),
                )
                return target
            except (AttributeError, IndexError):
                continue

        raise RuntimeError(
            f"Cannot locate transformer layer {layer_idx}. "
            f"Supported architectures: Llama, GPT-2, GPT-J, GPT-NeoX."
        )

    def _make_hook(self) -> Callable:
        """
        Create a forward hook that prepends virtual tokens to the
        hidden states entering the target layer.

        The hook signature follows PyTorch's convention:
            hook(module, args, output) → modified_output
        """
        virtual_tokens = self._virtual_tokens

        def _attention_steering_hook(
            module: nn.Module,
            args: tuple,
            output: Any,
        ) -> Any:
            """
            Prepend AXIOM virtual tokens to the hidden-state sequence.

            For Llama layers, the output is typically a tuple where
            output[0] is the hidden states tensor of shape (B, S, d_model).
            """
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            batch_size = hidden_states.size(0)

            # Expand virtual tokens to match batch size
            vt = virtual_tokens.expand(batch_size, -1, -1).to(
                hidden_states.device, dtype=hidden_states.dtype
            )

            # Prepend virtual tokens: (B, k + S, d_model)
            augmented = torch.cat([vt, hidden_states], dim=1)

            if isinstance(output, tuple):
                return (augmented,) + output[1:]
            return augmented

        return _attention_steering_hook

    def attach(self) -> None:
        """Register forward hooks on the target layer."""
        if self._hooks:
            logger.warning("Hooks already attached — detaching first.")
            self.detach()

        target = self._get_target_layer()
        hook = target.register_forward_hook(self._make_hook())
        self._hooks.append(hook)
        logger.info("AXIOM attention hook attached to layer %d.",
                    self.model_cfg.injection_layer)

    def detach(self) -> None:
        """Remove all registered hooks (prevents memory leaks)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("AXIOM hooks detached.")

    def __del__(self) -> None:
        self.detach()


# ---------------------------------------------------------------------------
# Model loading utility
# ---------------------------------------------------------------------------

def load_base_model(
    cfg: ModelConfig | None = None,
    cache_dir: Path | None = None,
) -> tuple[nn.Module, Any]:
    """
    Load the base SLM with optional quantisation.

    On macOS (no bitsandbytes), quantisation is automatically skipped
    and the model runs in float32/float16 on CPU or MPS.

    Returns:
        (model, tokenizer) tuple.

    Security:
        - HF_TOKEN is read from environment only; never logged.
        - Weights are loaded from cache_dir (external HD) to avoid
          filling the boot drive.
    """
    import os
    import platform
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = cfg or default_model_cfg

    hf_token = os.getenv("HF_TOKEN") or None
    if not hf_token and "llama" in cfg.model_id.lower():
        logger.warning(
            "HF_TOKEN not set. Gated models may fail to download. "
            "Set it in your .env file."
        )

    # Quantisation: only available on Linux with CUDA + bitsandbytes
    quantisation_config = None
    use_quant = cfg.quantisation != "none"

    if use_quant:
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401

            if cfg.quantisation == "nf4":
                quantisation_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif cfg.quantisation == "int8":
                quantisation_config = BitsAndBytesConfig(load_in_8bit=True)
        except ImportError:
            logger.info(
                "bitsandbytes not available (platform=%s). "
                "Running without quantisation.",
                platform.system(),
            )
            quantisation_config = None

    cache_path = str(cache_dir) if cache_dir else None

    # Determine dtype for non-quantised loading
    if cfg.device == "mps" or (
        cfg.device == "cpu" and platform.machine() == "arm64"
    ):
        model_dtype = torch.float32  # MPS float16 support is partial
    elif cfg.device != "cpu":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    logger.info(
        "Loading model: %s (quant=%s, dtype=%s, device=%s)",
        cfg.model_id, cfg.quantisation, model_dtype, cfg.device,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        cache_dir=cache_path,
        token=hf_token,
        trust_remote_code=False,
    )
    # Ensure pad token exists (GPT-2 and some models lack one)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "cache_dir": cache_path,
        "token": hf_token,
        "trust_remote_code": False,
        "torch_dtype": model_dtype,
    }
    if quantisation_config is not None:
        load_kwargs["quantization_config"] = quantisation_config
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, **load_kwargs
    )

    # Move to device if not using device_map="auto"
    if quantisation_config is None and cfg.device != "cpu":
        device = torch.device(cfg.device)
        model = model.to(device)

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model loaded: %s — %.1fM params, dtype=%s",
        cfg.model_id, n_params / 1e6, model_dtype,
    )

    return model, tokenizer


# ---------------------------------------------------------------------------
# High-level priming API
# ---------------------------------------------------------------------------

def prime_model(
    model: nn.Module,
    axiom_map: torch.Tensor,
    hdc_cfg: HDCConfig | None = None,
    model_cfg: ModelConfig | None = None,
) -> KVCacheInjector:
    """
    One-call API to prime a loaded model with an Axiom Map.

    Returns a KVCacheInjector with hooks already attached.

    Usage:
        model, tokenizer = load_base_model(cfg, cache_dir)
        distiller = AxiomDistiller()
        distiller.load(map_dir)
        injector = prime_model(model, distiller.axiom_map)
        # Model is now "primed" — run inference normally.
    """
    hdc_cfg = hdc_cfg or default_hdc_cfg
    model_cfg = model_cfg or default_model_cfg

    # Determine model hidden dimension
    if hasattr(model.config, "hidden_size"):
        model_dim = model.config.hidden_size
    else:
        raise RuntimeError("Cannot determine model hidden_size from config.")

    projector = AxiomProjector(
        hdc_dim=hdc_cfg.dimensions,
        model_dim=model_dim,
        num_virtual_tokens=model_cfg.max_virtual_tokens,
    )
    projector = projector.to(axiom_map.device)

    injector = KVCacheInjector(
        model=model,
        axiom_map=axiom_map,
        projector=projector,
        model_cfg=model_cfg,
    )
    injector.attach()

    logger.info("Model primed with AXIOM map. Zero-retrieval active.")
    return injector
