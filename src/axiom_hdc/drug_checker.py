"""
AXIOM Drug Interaction Checker — Offline HDC-Powered Safety Module

Loads curated drug-drug and drug-condition interaction triples, distils
them into an Axiom Map, and provides instant, offline interaction checking.

Architecture:
    1. Drug interaction triples are loaded from a curated JSON dataset
    2. Triples are bound into the Axiom Map via the AxiomDistiller
    3. Queries use HDC cosine similarity to detect known interactions
    4. The Safety Governor validates results and flags unsafe combinations

Security:
    - All inputs are sanitised (no eval, no shell)
    - Works fully offline — no API calls
    - Drug aliases are resolved locally
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from axiom_hdc.config import HDCConfig, hdc as default_hdc_cfg
from axiom_hdc.distiller import AxiomDistiller, MedicalFact
from axiom_hdc.governor import SafetyGovernor
from axiom_hdc.utils import timer

logger = logging.getLogger("axiom.drug_checker")

# ---- Input sanitisation ----------------------------------------------------

_UNSAFE_CHARS = re.compile(r"[^\w\s\-/(),.]", re.UNICODE)


def _sanitise_drug_name(name: str) -> str:
    """
    Clean and normalise a drug name for safe HDC lookup.

    Removes potentially dangerous characters while preserving
    legitimate pharmaceutical naming (hyphens, parentheses, slashes).
    """
    if not isinstance(name, str):
        return ""
    cleaned = _UNSAFE_CHARS.sub("", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.upper()


# ---- Data structures -------------------------------------------------------

@dataclass(frozen=True)
class InteractionResult:
    """Result of a drug interaction query."""

    drug_a: str
    drug_b: str
    found: bool
    severity: str  # "major", "moderate", "minor", "none"
    mechanism: str
    clinical_note: str
    confidence: float  # HDC cosine similarity score
    interaction_type: str  # "drug_drug" or "drug_condition"
    relation: str  # e.g. "INTERACTS_WITH", "CONTRAINDICATES"

    @property
    def is_unsafe(self) -> bool:
        """True if the interaction is major severity."""
        return self.severity == "major"

    @property
    def is_contraindicated(self) -> bool:
        """True if the relation is a contraindication."""
        return self.relation == "CONTRAINDICATES"


@dataclass
class DrugInteractionEntry:
    """A single interaction entry from the dataset."""

    subject: str
    relation: str
    obj: str
    severity: str
    mechanism: str
    clinical_note: str
    interaction_type: str  # "drug_drug" or "drug_condition"


# ---- Drug Interaction Checker -----------------------------------------------

class DrugInteractionChecker:
    """
    Offline Drug Interaction Checker powered by AXIOM HDC.

    Distils drug interaction knowledge into a hyperdimensional Axiom Map
    and provides sub-millisecond interaction queries via cosine similarity.
    Works entirely offline — ideal for rural clinics without internet.

    Usage:
        checker = DrugInteractionChecker()
        checker.load_dataset("data/drug_interactions.json")
        checker.distill()

        result = checker.check("warfarin", "aspirin")
        print(result.severity)       # "major"
        print(result.clinical_note)  # "Concomitant use significantly ..."
    """

    def __init__(
        self,
        cfg: HDCConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.cfg = cfg or default_hdc_cfg
        self.device = device or torch.device("cpu")

        # AXIOM core components
        self._distiller = AxiomDistiller(cfg=self.cfg, device=self.device)

        # Interaction lookup table (for returning clinical details)
        self._interactions: dict[tuple[str, str], DrugInteractionEntry] = {}

        # Drug alias mapping (brand → generic)
        self._aliases: dict[str, str] = {}

        # Track whether distillation has been done
        self._is_distilled = False

        logger.info(
            "DrugInteractionChecker initialised (D=%d, device=%s)",
            self.cfg.dimensions,
            self.device,
        )

    # ---- Dataset loading ---------------------------------------------------

    def load_dataset(self, path: str | Path) -> int:
        """
        Load drug interaction triples from a JSON dataset.

        Args:
            path: Path to the drug_interactions.json file.

        Returns:
            Number of interaction triples loaded.

        Raises:
            FileNotFoundError: If the dataset file doesn't exist.
            ValueError: If the dataset format is invalid.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Drug interaction dataset not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            raise ValueError("Dataset must be a JSON object")

        count = 0

        # Load drug-drug interactions
        for entry in data.get("drug_drug_interactions", []):
            self._add_interaction(entry, "drug_drug")
            count += 1

        # Load drug-condition interactions
        for entry in data.get("drug_condition_interactions", []):
            self._add_interaction(entry, "drug_condition")
            count += 1

        # Load drug aliases (brand → generic mapping)
        aliases = data.get("drug_aliases", {})
        for brand, generic in aliases.items():
            cleaned_brand = _sanitise_drug_name(brand)
            cleaned_generic = _sanitise_drug_name(generic)
            if cleaned_brand and cleaned_generic:
                self._aliases[cleaned_brand] = cleaned_generic

        logger.info(
            "Loaded %d interactions and %d aliases from %s",
            count,
            len(self._aliases),
            path,
        )
        return count

    def _add_interaction(
        self, entry: dict[str, Any], interaction_type: str
    ) -> None:
        """Add a single interaction entry to the lookup table."""
        subject = _sanitise_drug_name(entry.get("subject", ""))
        relation = _sanitise_drug_name(entry.get("relation", ""))
        obj = _sanitise_drug_name(entry.get("object", ""))

        if not (subject and relation and obj):
            logger.warning("Skipping malformed entry: %s", entry)
            return

        ie = DrugInteractionEntry(
            subject=subject,
            relation=relation,
            obj=obj,
            severity=entry.get("severity", "unknown"),
            mechanism=entry.get("mechanism", ""),
            clinical_note=entry.get("clinical_note", ""),
            interaction_type=interaction_type,
        )

        # Store bidirectionally for drug-drug interactions
        self._interactions[(subject, obj)] = ie
        if interaction_type == "drug_drug":
            # Reverse entry so lookup works in either order
            reverse_ie = DrugInteractionEntry(
                subject=obj,
                relation=relation,
                obj=subject,
                severity=ie.severity,
                mechanism=ie.mechanism,
                clinical_note=ie.clinical_note,
                interaction_type=interaction_type,
            )
            self._interactions[(obj, subject)] = reverse_ie

    # ---- Distillation ------------------------------------------------------

    def distill(self) -> None:
        """
        Encode all loaded interactions into the Axiom Map via HDC.

        Must be called after load_dataset() and before check().
        """
        if not self._interactions:
            raise ValueError(
                "No interactions loaded. Call load_dataset() first."
            )

        facts = []
        seen = set()

        for key, entry in self._interactions.items():
            fact_key = (entry.subject, entry.relation, entry.obj)
            if fact_key in seen:
                continue
            seen.add(fact_key)

            facts.append(
                MedicalFact(
                    subject=entry.subject,
                    relation=entry.relation,
                    obj=entry.obj,
                )
            )

        with timer("Drug interaction distillation"):
            self._distiller.distill(facts)

        self._is_distilled = True
        logger.info(
            "Distilled %d interaction facts into Axiom Map (%d entities)",
            self._distiller.fact_count,
            self._distiller.item_memory.size,
        )

    # ---- Alias resolution --------------------------------------------------

    def resolve_alias(self, drug_name: str) -> str:
        """
        Resolve a brand name to its generic equivalent.

        Returns the generic name if an alias exists, otherwise the
        original (sanitised) name.
        """
        cleaned = _sanitise_drug_name(drug_name)
        return self._aliases.get(cleaned, cleaned)

    # ---- Interaction checking ----------------------------------------------

    def check(self, drug_a: str, drug_b: str) -> InteractionResult:
        """
        Check for interactions between two drugs (or a drug + condition).

        Performs both a direct lookup in the interaction table and an
        HDC cosine similarity query against the Axiom Map.

        Args:
            drug_a: First drug (generic or brand name).
            drug_b: Second drug, condition, or brand name.

        Returns:
            InteractionResult with severity, mechanism, and confidence.
        """
        if not self._is_distilled:
            raise RuntimeError(
                "Axiom Map not ready. Call distill() first."
            )

        # Sanitise and resolve aliases
        name_a = self.resolve_alias(drug_a)
        name_b = self.resolve_alias(drug_b)

        # Direct lookup
        entry = self._interactions.get((name_a, name_b))

        # Compute HDC similarity for confidence scoring
        confidence = self._compute_hdc_similarity(name_a, name_b)

        if entry is not None:
            return InteractionResult(
                drug_a=name_a,
                drug_b=name_b,
                found=True,
                severity=entry.severity,
                mechanism=entry.mechanism,
                clinical_note=entry.clinical_note,
                confidence=confidence,
                interaction_type=entry.interaction_type,
                relation=entry.relation,
            )

        # No direct match — check if HDC similarity suggests a relationship
        # This catches semantic neighbours and partial matches
        if confidence >= self.cfg.safety_threshold:
            return InteractionResult(
                drug_a=name_a,
                drug_b=name_b,
                found=True,
                severity="moderate",
                mechanism="HDC similarity suggests a potential interaction",
                clinical_note=(
                    "The AXIOM knowledge base detected similarity to known "
                    "interaction patterns. Consult a pharmacist or drug "
                    "interaction reference for confirmation."
                ),
                confidence=confidence,
                interaction_type="drug_drug",
                relation="POTENTIAL_INTERACTION",
            )

        return InteractionResult(
            drug_a=name_a,
            drug_b=name_b,
            found=False,
            severity="none",
            mechanism="No known interaction found in the knowledge base",
            clinical_note=(
                "No interaction detected between these medications. "
                "This does not guarantee safety — always verify with "
                "current clinical references."
            ),
            confidence=confidence,
            interaction_type="none",
            relation="NONE",
        )

    def check_multiple(
        self, drug_list: list[str]
    ) -> list[InteractionResult]:
        """
        Check all pairwise interactions for a list of medications.

        Useful for polypharmacy screening.

        Args:
            drug_list: List of drug names to cross-check.

        Returns:
            List of InteractionResults for all pairs with findings.
        """
        results = []
        checked = set()

        for i, drug_a in enumerate(drug_list):
            for j, drug_b in enumerate(drug_list):
                if i >= j:
                    continue
                pair = tuple(sorted([drug_a.upper(), drug_b.upper()]))
                if pair in checked:
                    continue
                checked.add(pair)

                result = self.check(drug_a, drug_b)
                results.append(result)

        # Sort by severity: major first, then moderate, then minor
        severity_order = {"major": 0, "moderate": 1, "minor": 2, "none": 3}
        results.sort(key=lambda r: severity_order.get(r.severity, 4))

        return results

    # ---- HDC operations ----------------------------------------------------

    def _compute_hdc_similarity(
        self, entity_a: str, entity_b: str
    ) -> float:
        """
        Compute HDC cosine similarity for an entity pair against the Axiom Map.

        Constructs a query probe from the two entities using the
        INTERACTS_WITH relation vector, then compares against the
        superposition vector.
        """
        # Build query: entity_a INTERACTS_WITH entity_b
        query_hv = self._distiller.query(entity_a, "INTERACTS_WITH")
        base_sim = self._distiller.similarity(query_hv)

        # Also try CONTRAINDICATES relation
        query_hv_contra = self._distiller.query(entity_a, "CONTRAINDICATES")
        contra_sim = self._distiller.similarity(query_hv_contra)

        # Return the stronger signal
        return max(base_sim, contra_sim)

    # ---- Governor integration ----------------------------------------------

    def get_governor(self) -> SafetyGovernor:
        """
        Create a SafetyGovernor backed by this checker's Axiom Map.

        The Governor can be used for token-level validation if pairing
        this checker with an SLM for natural language responses.
        """
        if not self._is_distilled:
            raise RuntimeError("Distill first before creating a Governor.")

        return SafetyGovernor(
            axiom_map=self._distiller.axiom_map,
            item_memory=self._distiller.item_memory._store,
            cfg=self.cfg,
        )

    # ---- Persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save the distilled state to disk for instant future loading."""
        directory = Path(directory)
        self._distiller.save(directory)

        # Save interactions table as JSON for the API layer
        interactions_data = {}
        for (k1, k2), entry in self._interactions.items():
            interactions_data[f"{k1}||{k2}"] = {
                "subject": entry.subject,
                "relation": entry.relation,
                "object": entry.obj,
                "severity": entry.severity,
                "mechanism": entry.mechanism,
                "clinical_note": entry.clinical_note,
                "interaction_type": entry.interaction_type,
            }

        meta_path = directory / "interactions_meta.json"
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "interactions": interactions_data,
                    "aliases": self._aliases,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("DrugInteractionChecker saved → %s", directory)

    def load(self, directory: str | Path) -> None:
        """Load a previously distilled state from disk."""
        directory = Path(directory)
        self._distiller.load(directory)

        meta_path = directory / "interactions_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

            for key_str, entry_data in meta.get("interactions", {}).items():
                parts = key_str.split("||")
                if len(parts) == 2:
                    self._interactions[(parts[0], parts[1])] = (
                        DrugInteractionEntry(
                            subject=entry_data["subject"],
                            relation=entry_data["relation"],
                            obj=entry_data["object"],
                            severity=entry_data["severity"],
                            mechanism=entry_data["mechanism"],
                            clinical_note=entry_data["clinical_note"],
                            interaction_type=entry_data["interaction_type"],
                        )
                    )

            self._aliases = meta.get("aliases", {})

        self._is_distilled = True
        logger.info("DrugInteractionChecker loaded ← %s", directory)

    # ---- Info --------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the loaded knowledge base."""
        return {
            "total_interactions": len(self._interactions),
            "unique_drugs": len(
                {k[0] for k in self._interactions}
                | {k[1] for k in self._interactions}
            ),
            "aliases_count": len(self._aliases),
            "axiom_map_dimensions": self.cfg.dimensions,
            "facts_distilled": self._distiller.fact_count,
            "entities_in_memory": self._distiller.item_memory.size,
            "map_size_bytes": self._distiller.map_size_bytes,
            "is_distilled": self._is_distilled,
        }

    def list_known_drugs(self) -> list[str]:
        """Return a sorted list of all drug names in the knowledge base."""
        drugs = set()
        for key in self._interactions:
            drugs.add(key[0])
            drugs.add(key[1])
        # Also include alias keys (brand names)
        for alias in self._aliases:
            drugs.add(alias)
        return sorted(drugs)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"DrugInteractionChecker("
            f"interactions={stats['total_interactions']}, "
            f"drugs={stats['unique_drugs']}, "
            f"distilled={self._is_distilled})"
        )
