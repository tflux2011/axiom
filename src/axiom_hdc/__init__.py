"""
AXIOM — Zero-Retrieval Knowledge Injection via Hyperdimensional Computing.

Public API
----------
Distiller : ``AxiomDistiller``, ``MedicalFact``, ``ItemMemory``
Map       : ``AxiomMap``
Governor  : ``SafetyGovernor``, ``GovernorVerdict``
Priming   : ``AxiomProjector``, ``prime_model``
Encoder   : ``AxiomEncoder``
Config    : ``HDCConfig``, ``ModelConfig``
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Core ---
from axiom_hdc.distiller import AxiomDistiller, MedicalFact, ItemMemory
from axiom_hdc.axiom_map import AxiomMap

# --- Safety ---
from axiom_hdc.governor import SafetyGovernor, GovernorVerdict

# --- Config ---
from axiom_hdc.config import HDCConfig, ModelConfig

__all__ = [
    # Core
    "AxiomDistiller",
    "MedicalFact",
    "ItemMemory",
    "AxiomMap",
    # Safety
    "SafetyGovernor",
    "GovernorVerdict",
    # Config
    "HDCConfig",
    "ModelConfig",
    # Version
    "__version__",
]

# Lazy imports for optional heavy modules (LLM / NER dependencies)
def __getattr__(name: str):
    if name == "AxiomProjector":
        from axiom_hdc.priming import AxiomProjector
        return AxiomProjector
    if name == "prime_model":
        from axiom_hdc.priming import prime_model
        return prime_model
    if name == "AxiomEncoder":
        from axiom_hdc.encoder import AxiomEncoder
        return AxiomEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
