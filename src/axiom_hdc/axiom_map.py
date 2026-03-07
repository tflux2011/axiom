"""
AXIOM Map — Portable Hyperdimensional Knowledge Container

The AxiomMap wraps the raw superposition vector and item memory into
a single, self-describing object that can be saved, loaded, inspected,
and passed to the Primer and Governor.

File format (.axiom):
    A standard PyTorch save file containing a dict with keys:
        - "vector":      Axiom Map tensor (1, D)
        - "item_memory": dict[str, Tensor] entity → hypervector
        - "metadata":    dict with dim, fact_count, created_at, etc.

Security:
    - Loaded with weights_only=False because we control the save format.
    - File integrity is checked via dimension validation on load.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("axiom.map")


@dataclass
class AxiomMap:
    """
    A portable hyperdimensional knowledge container.

    Wraps the Axiom Map vector and its associated Item Memory into
    a single object with save/load, inspection, and capacity estimation.

    Usage:
        # Create from a distiller
        distiller = Distiller(dim=10_000)
        axiom_map = distiller.encode(facts)

        # Save and load
        axiom_map.save("medical.axiom")
        axiom_map = AxiomMap.load("medical.axiom")

        # Inspect
        print(axiom_map.info())
    """

    vector: torch.Tensor
    """The superposition vector (1, D), bipolar {-1, +1}."""

    item_memory: dict[str, torch.Tensor] = field(default_factory=dict)
    """Entity name → hypervector mapping."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary metadata (dim, fact_count, created_at, etc.)."""

    # ---- Properties --------------------------------------------------------

    @property
    def dim(self) -> int:
        """Dimensionality of the hypervectors."""
        return self.vector.shape[-1]

    @property
    def fact_count(self) -> int:
        """Number of facts encoded (from metadata)."""
        return self.metadata.get("fact_count", 0)

    @property
    def entity_count(self) -> int:
        """Number of entities in the Item Memory."""
        return len(self.item_memory)

    @property
    def size_bytes(self) -> int:
        """Total size in bytes (vector + item memory, float32)."""
        vector_bytes = self.vector.nelement() * self.vector.element_size()
        im_bytes = sum(
            v.nelement() * v.element_size() for v in self.item_memory.values()
        )
        return vector_bytes + im_bytes

    @property
    def capacity_remaining(self) -> int:
        """Estimated remaining facts before SNR degrades below threshold.

        Based on the theoretical bound: N_max ≈ d / 49 for SNR ≥ 7.
        """
        n_max = self.dim // 49
        return max(0, n_max - self.fact_count)

    # ---- Persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save the AxiomMap to a .axiom file.

        Args:
            path: File path (extension .axiom is added if missing).
        """
        path = Path(path)
        if path.suffix != ".axiom":
            path = path.with_suffix(".axiom")
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "vector": self.vector,
            "item_memory": self.item_memory,
            "metadata": {
                **self.metadata,
                "dim": self.dim,
                "entity_count": self.entity_count,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        torch.save(payload, path)
        logger.info(
            "AxiomMap saved → %s (%d facts, %d entities, %s)",
            path, self.fact_count, self.entity_count,
            _format_bytes(self.size_bytes),
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "AxiomMap":
        """
        Load an AxiomMap from a .axiom file.

        Args:
            path:   Path to the .axiom file.
            device: Target device for tensors.

        Returns:
            A restored AxiomMap instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"AxiomMap file not found: {path}")

        payload = torch.load(path, map_location=device, weights_only=False)

        vector = payload["vector"]
        item_memory = payload.get("item_memory", {})
        metadata = payload.get("metadata", {})

        # Validate
        if vector.dim() < 2 or vector.shape[0] != 1:
            raise ValueError(
                f"Invalid vector shape: {vector.shape}. Expected (1, D)."
            )

        instance = cls(
            vector=vector,
            item_memory=item_memory,
            metadata=metadata,
        )
        logger.info(
            "AxiomMap loaded ← %s (%d facts, %d entities, D=%d)",
            path, instance.fact_count, instance.entity_count, instance.dim,
        )
        return instance

    # ---- Inspection --------------------------------------------------------

    def info(self) -> str:
        """Return a human-readable summary of the AxiomMap."""
        snr = math.sqrt(self.dim / max(self.fact_count, 1))
        return (
            f"AxiomMap\n"
            f"  Dimension:      {self.dim:,}\n"
            f"  Facts encoded:  {self.fact_count:,}\n"
            f"  Entities:       {self.entity_count:,}\n"
            f"  Size:           {_format_bytes(self.size_bytes)}\n"
            f"  SNR:            {snr:.1f}\n"
            f"  Capacity left:  ~{self.capacity_remaining:,} facts\n"
            f"  Created:        {self.metadata.get('created_at', 'unknown')}"
        )

    def __repr__(self) -> str:
        return (
            f"AxiomMap(dim={self.dim}, facts={self.fact_count}, "
            f"entities={self.entity_count}, "
            f"size={_format_bytes(self.size_bytes)})"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_bytes(n: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if n >= 10 else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
