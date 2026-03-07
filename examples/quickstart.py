"""
AXIOM Quickstart — Distil → Save → Query → Govern

Run:
    python examples/quickstart.py
"""

from axiom_hdc import AxiomDistiller, AxiomMap, SafetyGovernor, HDCConfig, MedicalFact


def main() -> None:
    # 1. Configure
    cfg = HDCConfig(dimensions=10_000)
    distiller = AxiomDistiller(cfg=cfg, seed=42)

    # 2. Define some medical facts
    facts = [
        MedicalFact("METFORMIN", "TREATS", "TYPE_2_DIABETES"),
        MedicalFact("METFORMIN", "SIDE_EFFECT", "LACTIC_ACIDOSIS"),
        MedicalFact("AMOXICILLIN", "TREATS", "BACTERIAL_INFECTION"),
        MedicalFact("LISINOPRIL", "TREATS", "HYPERTENSION"),
        MedicalFact("IBUPROFEN", "CONTRAINDICATED", "RENAL_FAILURE"),
    ]

    # 3. Distil into a single hyperdimensional vector
    distiller.distill(facts)
    print(f"Distilled {distiller.fact_count} facts into a "
          f"{cfg.dimensions}-D bipolar vector")

    # 4. Save as a portable .axiom file
    axiom_map = AxiomMap(
        vector=distiller.axiom_map,
        item_memory=distiller.item_memory._store,
        metadata={
            "fact_count": distiller.fact_count,
            "source": "quickstart example",
        },
    )
    axiom_map.save("quickstart.axiom")
    print(f"\nSaved to quickstart.axiom ({axiom_map.size_bytes:,} bytes)")
    print(axiom_map.info())

    # 5. Load it back
    loaded = AxiomMap.load("quickstart.axiom")

    # 6. Query: What does Metformin treat?
    probe = distiller.query("METFORMIN", "TREATS")
    # Unbind probe from the Axiom Map to recover the answer vector
    import torch
    from torchhd import functional as F

    recovered = F.bind(distiller.axiom_map, probe)

    # Find the nearest entity in the item memory
    best_match, best_sim = None, -1.0
    for name, hv in distiller.item_memory._store.items():
        sim = torch.nn.functional.cosine_similarity(
            recovered.float(), hv.float()
        ).item()
        if sim > best_sim:
            best_sim = sim
            best_match = name

    print(f"\nQuery: METFORMIN --[TREATS]--> ?")
    print(f"Answer: {best_match}  (cosine: {best_sim:.4f})")

    # 7. Safety Governor — verify a claim
    governor = SafetyGovernor(
        axiom_map=distiller.axiom_map,
        item_memory=distiller.item_memory._store,
        cfg=cfg,
    )

    # Verify a known fact
    expected = governor.extract_expected_answer("METFORMIN", "TREATS")
    sim_good = governor.validate_token_against_probe("TYPE_2_DIABETES", expected)
    print(f"\nGovernor: 'Metformin treats Type 2 Diabetes'")
    print(f"  similarity={sim_good:.4f}, safe={sim_good >= cfg.safety_threshold}")

    # Verify a hallucinated claim
    sim_bad = governor.validate_token_against_probe("HEADACHE", expected)
    print(f"\nGovernor: 'Metformin treats Headache'")
    print(f"  similarity={sim_bad:.4f}, safe={sim_bad >= cfg.safety_threshold}")

    # Cleanup
    import os
    os.remove("quickstart.axiom")
    print("\nDone! Cleaned up quickstart.axiom")


if __name__ == "__main__":
    main()
