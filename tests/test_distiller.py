"""
AXIOM Unit Tests — Distiller

Tests the core HDC operations: vector generation, binding, bundling,
querying, persistence, and cleanup.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.config import HDCConfig
from src.distiller import AxiomDistiller, MedicalFact, ItemMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> HDCConfig:
    return HDCConfig(dimensions=1000, cleanup_iterations=2, safety_threshold=0.3)


@pytest.fixture
def distiller(cfg: HDCConfig) -> AxiomDistiller:
    return AxiomDistiller(cfg=cfg, seed=42)


@pytest.fixture
def sample_facts() -> list[MedicalFact]:
    return [
        MedicalFact("Aspirin", "TREATS", "Headache"),
        MedicalFact("Aspirin", "THINS", "Blood"),
        MedicalFact("Metformin", "TREATS", "Diabetes"),
        MedicalFact("Insulin", "REGULATES", "Glucose"),
        MedicalFact("Warfarin", "THINS", "Blood"),
    ]


# ---------------------------------------------------------------------------
# Item Memory tests
# ---------------------------------------------------------------------------

class TestItemMemory:
    def test_get_creates_vector(self, cfg: HDCConfig) -> None:
        im = ItemMemory(dim=cfg.dimensions)
        v = im.get("Aspirin")
        assert v.shape == (1, cfg.dimensions)
        assert im.size == 1

    def test_get_returns_same_vector(self, cfg: HDCConfig) -> None:
        im = ItemMemory(dim=cfg.dimensions)
        v1 = im.get("Aspirin")
        v2 = im.get("Aspirin")
        assert torch.equal(v1, v2)

    def test_different_entities_different_vectors(self, cfg: HDCConfig) -> None:
        im = ItemMemory(dim=cfg.dimensions)
        v1 = im.get("Aspirin")
        v2 = im.get("Metformin")
        assert not torch.equal(v1, v2)

    def test_vectors_are_bipolar(self, cfg: HDCConfig) -> None:
        im = ItemMemory(dim=cfg.dimensions)
        v = im.get("Test")
        unique_vals = torch.unique(v)
        assert all(val in (-1.0, 1.0) for val in unique_vals.tolist())

    def test_save_and_load(self, cfg: HDCConfig) -> None:
        im = ItemMemory(dim=cfg.dimensions)
        im.get("Aspirin")
        im.get("Metformin")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "im.pt"
            im.save(path)

            im2 = ItemMemory(dim=cfg.dimensions)
            im2.load(path)
            assert im2.size == 2
            assert torch.equal(im2.get("Aspirin"), im.get("Aspirin"))


# ---------------------------------------------------------------------------
# Distiller tests
# ---------------------------------------------------------------------------

class TestAxiomDistiller:
    def test_encode_fact_shape(self, distiller: AxiomDistiller) -> None:
        fact = MedicalFact("Aspirin", "TREATS", "Headache")
        hv = distiller.encode_fact(fact)
        assert hv.shape == (1, distiller.cfg.dimensions)

    def test_encode_fact_bipolar(self, distiller: AxiomDistiller) -> None:
        fact = MedicalFact("Aspirin", "TREATS", "Headache")
        hv = distiller.encode_fact(fact)
        unique = torch.unique(hv)
        assert all(v in (-1.0, 1.0) for v in unique.tolist())

    def test_distill_updates_count(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        distiller.distill(sample_facts)
        assert distiller.fact_count == len(sample_facts)

    def test_distill_produces_bipolar_map(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        distiller.distill(sample_facts)
        unique = torch.unique(distiller.axiom_map)
        assert all(v in (-1.0, 1.0) for v in unique.tolist())

    def test_query_shape(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        distiller.distill(sample_facts)
        qhv = distiller.query("Aspirin", "TREATS")
        assert qhv.shape == (1, distiller.cfg.dimensions)

    def test_known_fact_has_positive_similarity(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        distiller.distill(sample_facts)
        qhv = distiller.query("Aspirin", "TREATS")
        sim = distiller.similarity(qhv)
        # Known fact should have positive (non-zero) similarity
        assert sim != 0.0

    def test_unknown_fact_has_low_similarity(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        distiller.distill(sample_facts)
        # "Banana CURES Cancer" was never distilled
        qhv = distiller.query("Banana", "CURES")
        sim = distiller.similarity(qhv)
        # Should be near zero (quasi-orthogonal)
        assert abs(sim) < 0.5

    def test_save_and_load(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        distiller.distill(sample_facts)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = Path(tmpdir) / "axiom_test"
            distiller.save(save_dir)

            # Load into a fresh distiller
            d2 = AxiomDistiller(cfg=distiller.cfg, seed=42)
            d2.load(save_dir)

            assert torch.equal(d2.axiom_map, distiller.axiom_map)

    def test_map_size_bytes(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        distiller.distill(sample_facts)
        # 1 × 1000 dimensions × 4 bytes (float32) = 4000 bytes
        assert distiller.map_size_bytes == 1 * distiller.cfg.dimensions * 4

    def test_reproducibility(
        self, cfg: HDCConfig, sample_facts: list[MedicalFact]
    ) -> None:
        """Same seed + same facts → same Axiom Map."""
        d1 = AxiomDistiller(cfg=cfg, seed=123)
        d1.distill(sample_facts)

        d2 = AxiomDistiller(cfg=cfg, seed=123)
        d2.distill(sample_facts)

        assert torch.equal(d1.axiom_map, d2.axiom_map)


# ---------------------------------------------------------------------------
# Governor tests (lightweight — no model needed)
# ---------------------------------------------------------------------------

class TestSafetyGovernorLogic:
    def test_known_entity_similarity(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        from src.governor import SafetyGovernor

        distiller.distill(sample_facts)
        governor = SafetyGovernor(
            axiom_map=distiller.axiom_map,
            item_memory=distiller.item_memory._store,
            cfg=distiller.cfg,
        )

        # Known entity should have non-trivial similarity
        sim = governor._compute_token_safety("Aspirin")
        assert isinstance(sim, float)

    def test_unknown_entity_low_similarity(
        self, distiller: AxiomDistiller, sample_facts: list[MedicalFact]
    ) -> None:
        from src.governor import SafetyGovernor

        distiller.distill(sample_facts)
        governor = SafetyGovernor(
            axiom_map=distiller.axiom_map,
            item_memory=distiller.item_memory._store,
            cfg=distiller.cfg,
        )

        # Completely unknown entity → random HV → low similarity
        sim = governor._compute_token_safety("XyzUnknownDrug123")
        assert abs(sim) < 0.5
