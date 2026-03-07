"""
AXIOM Configuration Module

Centralised, validated configuration for all AXIOM components.
All secrets are loaded from environment variables — never hard-coded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STORAGE = os.getenv(
    "AXIOM_STORAGE_PATH", "/Volumes/WD Drive/axiom_data"
)


@dataclass(frozen=True)
class HDCConfig:
    """Hyperdimensional Computing parameters."""

    dimensions: int = 10_000
    """Dimensionality of bipolar hypervectors."""

    vector_type: Literal["bipolar", "binary"] = "bipolar"
    """bipolar {-1,+1} or binary {0,1}."""

    cleanup_iterations: int = 3
    """Number of iterative cleanup steps after bundling."""

    safety_threshold: float = 0.35
    """Cosine-similarity threshold for the Safety Governor.
    Tokens below this are suppressed as potential hallucinations."""


@dataclass(frozen=True)
class ModelConfig:
    """Base SLM configuration."""

    model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
    """Hugging Face model identifier."""

    quantisation: Literal["nf4", "int8", "none"] = "nf4"
    """Quantisation scheme for low-memory deployment."""

    injection_layer: int = 16
    """Transformer layer index where AXIOM map is injected."""

    max_virtual_tokens: int = 128
    """Number of virtual context tokens derived from the HD map."""

    projection_bottleneck: int = 512
    """Bottleneck dimension for the low-rank two-stage projection.
    Total projection params ≈ hdc_dim * bottleneck + bottleneck * k * d_model."""

    device: str = os.getenv("AXIOM_DEVICE", "cpu")
    """Target device: cpu | cuda | mps."""


@dataclass(frozen=True)
class DataConfig:
    """Dataset and storage paths."""

    storage_root: Path = field(default_factory=lambda: Path(_DEFAULT_STORAGE))
    """Root directory on external storage for large artefacts."""

    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)

    @property
    def model_cache(self) -> Path:
        """Where Hugging Face model weights are cached."""
        return self.storage_root / "models"

    @property
    def dataset_dir(self) -> Path:
        """Where BioASQ and other datasets are stored."""
        return self.storage_root / "datasets"

    @property
    def axiom_maps_dir(self) -> Path:
        """Where distilled HD maps are persisted."""
        return self.storage_root / "axiom_maps"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [
            self.model_cache,
            self.dataset_dir,
            self.axiom_maps_dir,
            self.results_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class NERConfig:
    """Named Entity Recognition settings."""

    spacy_model: str = "en_core_sci_md"
    """scispaCy biomedical NER model."""

    entity_types: tuple[str, ...] = (
        "CHEMICAL",
        "DISEASE",
        "GENE_OR_GENE_PRODUCT",
        "ORGANISM",
        "CELL_TYPE",
        "CELL_LINE",
        "DNA",
        "RNA",
        "PROTEIN",
    )
    """Entity categories extracted by the NER tagger."""

    relation_labels: tuple[str, ...] = (
        "TREATS",
        "CAUSES",
        "PREVENTS",
        "INDICATES",
        "CONTRAINDICATES",
        "INTERACTS_WITH",
        "REGULATES",
        "INHIBITS",
        "ACTIVATES",
        "METABOLISES",
    )
    """Relation types used in axiomatic binding."""


# ---------------------------------------------------------------------------
# Singleton-style defaults (import and use directly)
# ---------------------------------------------------------------------------
hdc = HDCConfig()
model = ModelConfig()
data = DataConfig()
ner = NERConfig()
