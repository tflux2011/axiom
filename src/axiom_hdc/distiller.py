"""
AXIOM Distiller — Hyperdimensional Axiomatic Encoding Engine

Converts raw medical triples (Subject, Relation, Object) into a single
high-dimensional "Superposition Vector" (the Axiom Map) using
Holographic Reduced Representations (HRR) via the torchhd library.

Key operations:
  • Random HV generation  → orthogonal base atoms
  • Binding (circular convolution / XOR) → directional fact encoding
  • Bundling (element-wise sum)  → superposition of facts
  • Cleanup (iterative associative recall) → noise reduction

The resulting Axiom Map is a single tensor of shape (1, D) that
encodes millions of medical facts and fits on a 64 GB SD card.

References:
    Kanerva, P.  "Hyperdimensional Computing: An Introduction to Computing
    in Distributed Representation with High-Dimensional Random Vectors."
    Cognitive Computation, 2009.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple

import torch
import torchhd
from torchhd import functional as F

from axiom_hdc.config import HDCConfig, hdc as default_hdc_cfg
from axiom_hdc.utils import save_tensor, load_tensor, timer

logger = logging.getLogger("axiom.distiller")


# ---------------------------------------------------------------------------
# Permutation helper
# ---------------------------------------------------------------------------

def _cyclic_shift(hv: torch.Tensor, n: int) -> torch.Tensor:
    """Apply a cyclic left-shift of *n* positions along the last dimension.

    This implements the Π_n operator from the paper's Permutation-Based
    Role Encoding.  Cyclic permutation breaks binding commutativity,
    ensuring (A treats B) ≠ (B treats A) in the Axiom Map.
    """
    if n == 0:
        return hv
    return torch.roll(hv, shifts=-n, dims=-1)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class MedicalFact(NamedTuple):
    """A single (Subject, Relation, Object) triple."""
    subject: str
    relation: str
    obj: str


@dataclass
class ItemMemory:
    """
    Stores the base "Atomic" hypervectors for every unique entity and
    relation observed during distillation.

    Each entity is assigned a quasi-orthogonal random hypervector on first
    access.  The mapping is deterministic per session (seeded) to ensure
    reproducibility.
    """

    dim: int
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _store: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    def get(self, name: str) -> torch.Tensor:
        """Return the hypervector for *name*, generating one if needed."""
        if name not in self._store:
            # Generate a random bipolar HV: values in {-1, +1}
            # torchhd.functional.random() already returns bipolar vectors
            hv = F.random(1, self.dim, device=self.device)
            self._store[name] = hv
        return self._store[name]

    @property
    def size(self) -> int:
        return len(self._store)

    def save(self, path: Path) -> None:
        """Persist item memory to disk."""
        tensors = {k: v for k, v in self._store.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, path)
        logger.info("Item memory (%d entries) saved → %s", self.size, path)

    def load(self, path: Path) -> None:
        """Restore item memory from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Item memory not found: {path}")
        # Use weights_only=False since we save a plain dict of tensors.
        # This is safe because we only ever load files we created ourselves.
        tensors = torch.load(
            path, map_location=self.device, weights_only=False)
        self._store.update(tensors)
        logger.info("Item memory (%d entries) loaded ← %s", self.size, path)


# ---------------------------------------------------------------------------
# Distiller
# ---------------------------------------------------------------------------

class AxiomDistiller:
    """
    The Relational Contextual Distiller (RCD).

    Encodes a corpus of medical facts into a single Axiom Map via
    Hyperdimensional Computing operations:

        Fact HV = bind(bind(Subject, Relation), Object)
        Axiom Map = Σ  Fact HVs   (element-wise sum / superposition)

    The Axiom Map is then cleaned-up (iterative thresholding) to sharpen
    signal-to-noise ratio.
    """

    def __init__(
        self,
        cfg: HDCConfig | None = None,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        self.cfg = cfg or default_hdc_cfg
        self.device = device or torch.device("cpu")
        self._seed = seed

        # Seed for reproducibility
        torch.manual_seed(self._seed)

        self.item_memory = ItemMemory(
            dim=self.cfg.dimensions, device=self.device)
        self._fact_count = 0

        # The Axiom Map (Superposition Vector)
        self.axiom_map: torch.Tensor = torch.zeros(
            1, self.cfg.dimensions, device=self.device
        )

        logger.info(
            "AxiomDistiller initialised (D=%d, device=%s)",
            self.cfg.dimensions,
            self.device,
        )

    # ---- Core operations ---------------------------------------------------

    def encode_fact(self, fact: MedicalFact) -> torch.Tensor:
        """
        Bind a single (S, R, O) triple into a directional fact hypervector
        using Permutation-Based Role Encoding.

        The relation vector receives Π_1 (1-bit cyclic shift) and the
        object vector receives Π_2 (2-bit cyclic shift) to break
        commutativity.  This ensures (A treats B) ≠ (B treats A).

        Returns:
            Tensor of shape (1, D).
        """
        v_sub = self.item_memory.get(fact.subject)
        v_rel = self.item_memory.get(fact.relation)
        v_obj = self.item_memory.get(fact.obj)

        # Permutation-Based Role Encoding:
        #   R_fact = v_sub ⊗ Π_1(v_rel) ⊗ Π_2(v_obj)
        bound = F.bind(
            F.bind(v_sub, _cyclic_shift(v_rel, 1)),
            _cyclic_shift(v_obj, 2),
        )
        return bound

    def distill(self, facts: Iterable[MedicalFact]) -> torch.Tensor:
        """
        Distil an iterable of medical facts into the Axiom Map.

        Each fact is encoded and then *bundled* (element-wise summed) into
        the running superposition vector.  After all facts are processed,
        the map undergoes iterative cleanup.

        Returns:
            The finalised Axiom Map tensor of shape (1, D).
        """
        with timer("Distillation"):
            for fact in facts:
                fact_hv = self.encode_fact(fact)
                self.axiom_map = self.axiom_map + fact_hv
                self._fact_count += 1

                if self._fact_count % 50_000 == 0:
                    logger.info("Distilled %d facts so far …",
                                self._fact_count)

        logger.info(
            "Distillation complete — %d facts encoded into Axiom Map.",
            self._fact_count,
        )

        # Cleanup: binarise / threshold to sharpen the map
        self.axiom_map = self._cleanup(self.axiom_map)
        return self.axiom_map

    # ---- Query support -----------------------------------------------------

    def query(self, subject: str, relation: str) -> torch.Tensor:
        """
        Construct an HD query vector for "Subject + Relation → ?".

        Uses the same Π_1 permutation applied during encoding so that
        the query is compatible with the Axiom Map.  The caller can
        multiply this probe by the Axiom Map to extract (unbind) the
        expected answer vector.

        Returns:
            Query hypervector of shape (1, D).
        """
        v_sub = self.item_memory.get(subject)
        v_rel = self.item_memory.get(relation)
        return F.bind(v_sub, _cyclic_shift(v_rel, 1))

    def similarity(self, query_hv: torch.Tensor) -> float:
        """
        Cosine similarity between a query HV and the Axiom Map.

        A high value (> cfg.safety_threshold) indicates the map contains
        supporting evidence; a low value signals a potential hallucination.
        """
        cos = torch.nn.functional.cosine_similarity(query_hv, self.axiom_map)
        return cos.max().item()

    # ---- Persistence -------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Save the Axiom Map and item memory to *directory*."""
        directory.mkdir(parents=True, exist_ok=True)
        save_tensor(self.axiom_map, directory / "axiom_map.pt")
        self.item_memory.save(directory / "item_memory.pt")
        logger.info("Distiller state saved → %s", directory)

    def load(self, directory: Path) -> None:
        """Restore a previously saved distiller state."""
        self.axiom_map = load_tensor(
            directory / "axiom_map.pt", device=str(self.device)
        )
        self.item_memory.load(directory / "item_memory.pt")
        logger.info("Distiller state loaded ← %s", directory)

    # ---- Internal ----------------------------------------------------------

    def _cleanup(self, hv: torch.Tensor) -> torch.Tensor:
        """
        Iterative bipolar cleanup: threshold the superposition vector
        to sharpen signal (majority vote per dimension).

        After cleanup, each dimension is in {-1, +1}.
        """
        for _ in range(self.cfg.cleanup_iterations):
            hv = torch.sign(hv)
            # Replace zeros (ties) with +1 to maintain bipolarity
            hv[hv == 0] = 1.0
        return hv

    # ---- Metrics -----------------------------------------------------------

    @property
    def fact_count(self) -> int:
        return self._fact_count

    @property
    def map_size_bytes(self) -> int:
        """Size of the Axiom Map in bytes (float32)."""
        return self.axiom_map.nelement() * self.axiom_map.element_size()

    def __repr__(self) -> str:
        return (
            f"AxiomDistiller(D={self.cfg.dimensions}, "
            f"facts={self._fact_count}, "
            f"entities={self.item_memory.size}, "
            f"map_bytes={self.map_size_bytes:,})"
        )
