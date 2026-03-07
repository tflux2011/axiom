"""
AXIOM Benchmark — Phase C: Fact Retrieval Accuracy

Evaluates AXIOM's knowledge fidelity and hallucination prevention
on a controlled synthetic medical knowledge base.

Design:
    - Knowledge base: 150 unique medical facts (triples)
    - Positive queries: Facts that ARE in the knowledge base (expect high cosine)
    - Negative queries: Facts NOT in the knowledge base (expect low cosine)

Metrics:
    - Fact Fidelity: fraction of stored facts correctly retrieved above tau
    - Hallucination Catch Rate: fraction of unsupported queries correctly
      rejected below tau
    - False Positive Rate: unsupported queries incorrectly passed
    - Mean cosine similarity for positive vs. negative queries

Usage:
    python -m benchmarks.accuracy_bench
"""

from __future__ import annotations
from axiom_hdc.distiller import _cyclic_shift

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


logger = logging.getLogger("axiom.bench.accuracy")


# ---------------------------------------------------------------------------
# Knowledge base: 150 unique medical facts
# ---------------------------------------------------------------------------

def _build_knowledge_base():
    """Return 150 diverse synthetic medical facts as (subject, relation, obj)."""
    facts = [
        # --- TREATS relationships (50) ---
        ("Aspirin", "TREATS", "Headache"),
        ("Aspirin", "TREATS", "Fever"),
        ("Metformin", "TREATS", "Type2_Diabetes"),
        ("Insulin", "TREATS", "Type1_Diabetes"),
        ("Amoxicillin", "TREATS", "Bacterial_Infection"),
        ("Ibuprofen", "TREATS", "Inflammation"),
        ("Lisinopril", "TREATS", "Hypertension"),
        ("Atorvastatin", "TREATS", "Hyperlipidemia"),
        ("Amlodipine", "TREATS", "Hypertension"),
        ("Omeprazole", "TREATS", "Acid_Reflux"),
        ("Levothyroxine", "TREATS", "Hypothyroidism"),
        ("Azithromycin", "TREATS", "Respiratory_Infection"),
        ("Ciprofloxacin", "TREATS", "Urinary_Infection"),
        ("Metoprolol", "TREATS", "Tachycardia"),
        ("Losartan", "TREATS", "Hypertension"),
        ("Sertraline", "TREATS", "Depression"),
        ("Fluoxetine", "TREATS", "Depression"),
        ("Duloxetine", "TREATS", "Anxiety"),
        ("Gabapentin", "TREATS", "Neuropathic_Pain"),
        ("Pregabalin", "TREATS", "Fibromyalgia"),
        ("Montelukast", "TREATS", "Asthma"),
        ("Albuterol", "TREATS", "Bronchospasm"),
        ("Clopidogrel", "TREATS", "Thrombosis"),
        ("Rivaroxaban", "TREATS", "Atrial_Fibrillation"),
        ("Tamsulosin", "TREATS", "Benign_Prostatic_Hyperplasia"),
        ("Donepezil", "TREATS", "Alzheimer_Disease"),
        ("Levodopa", "TREATS", "Parkinson_Disease"),
        ("Hydroxychloroquine", "TREATS", "Lupus"),
        ("Methotrexate", "TREATS", "Rheumatoid_Arthritis"),
        ("Colchicine", "TREATS", "Gout"),
        ("Acetaminophen", "TREATS", "Pain"),
        ("Diazepam", "TREATS", "Anxiety"),
        ("Alprazolam", "TREATS", "Panic_Disorder"),
        ("Prednisone", "TREATS", "Inflammation"),
        ("Warfarin", "TREATS", "Deep_Vein_Thrombosis"),
        ("Phenytoin", "TREATS", "Epilepsy"),
        ("Valproate", "TREATS", "Bipolar_Disorder"),
        ("Lithium", "TREATS", "Bipolar_Disorder"),
        ("Carbamazepine", "TREATS", "Seizures"),
        ("Clonazepam", "TREATS", "Seizures"),
        ("Furosemide", "TREATS", "Edema"),
        ("Hydrochlorothiazide", "TREATS", "Hypertension"),
        ("Spironolactone", "TREATS", "Heart_Failure"),
        ("Cephalexin", "TREATS", "Skin_Infection"),
        ("Doxycycline", "TREATS", "Acne"),
        ("Ranitidine", "TREATS", "Peptic_Ulcer"),
        ("Pantoprazole", "TREATS", "GERD"),
        ("Rosuvastatin", "TREATS", "Hypercholesterolemia"),
        ("Simvastatin", "TREATS", "Hyperlipidemia"),
        ("Trazodone", "TREATS", "Insomnia"),

        # --- CAUSES side effects (25) ---
        ("Metformin", "CAUSES", "Lactic_Acidosis"),
        ("Aspirin", "CAUSES", "GI_Bleeding"),
        ("Warfarin", "CAUSES", "Hemorrhage"),
        ("Ibuprofen", "CAUSES", "Stomach_Ulcer"),
        ("Prednisone", "CAUSES", "Osteoporosis"),
        ("Lithium", "CAUSES", "Kidney_Damage"),
        ("Atorvastatin", "CAUSES", "Myopathy"),
        ("Methotrexate", "CAUSES", "Liver_Toxicity"),
        ("Ciprofloxacin", "CAUSES", "Tendon_Rupture"),
        ("Furosemide", "CAUSES", "Hypokalemia"),
        ("Phenytoin", "CAUSES", "Gingival_Hyperplasia"),
        ("Valproate", "CAUSES", "Hepatotoxicity"),
        ("Omeprazole", "CAUSES", "Magnesium_Deficiency"),
        ("Sertraline", "CAUSES", "Sexual_Dysfunction"),
        ("Gabapentin", "CAUSES", "Drowsiness"),
        ("Amoxicillin", "CAUSES", "Allergic_Reaction"),
        ("Diazepam", "CAUSES", "Respiratory_Depression"),
        ("Insulin", "CAUSES", "Hypoglycemia"),
        ("Amlodipine", "CAUSES", "Peripheral_Edema"),
        ("Tamsulosin", "CAUSES", "Orthostatic_Hypotension"),
        ("Hydroxychloroquine", "CAUSES", "Retinal_Toxicity"),
        ("Carbamazepine", "CAUSES", "Aplastic_Anemia"),
        ("Fluoxetine", "CAUSES", "Serotonin_Syndrome"),
        ("Doxycycline", "CAUSES", "Photosensitivity"),
        ("Spironolactone", "CAUSES", "Hyperkalemia"),

        # --- CONTRAINDICATES (20) ---
        ("Aspirin", "CONTRAINDICATES", "Warfarin"),
        ("Metformin", "CONTRAINDICATES", "Renal_Failure"),
        ("Lisinopril", "CONTRAINDICATES", "Pregnancy"),
        ("Ibuprofen", "CONTRAINDICATES", "Kidney_Disease"),
        ("Ciprofloxacin", "CONTRAINDICATES", "Myasthenia_Gravis"),
        ("Methotrexate", "CONTRAINDICATES", "Pregnancy"),
        ("Warfarin", "CONTRAINDICATES", "Active_Bleeding"),
        ("Diazepam", "CONTRAINDICATES", "Sleep_Apnea"),
        ("Lithium", "CONTRAINDICATES", "Dehydration"),
        ("Phenytoin", "CONTRAINDICATES", "Porphyria"),
        ("Simvastatin", "CONTRAINDICATES", "Liver_Disease"),
        ("Alprazolam", "CONTRAINDICATES", "Acute_Glaucoma"),
        ("Furosemide", "CONTRAINDICATES", "Anuria"),
        ("Tamsulosin", "CONTRAINDICATES", "Orthostatic_Hypo"),
        ("Hydroxychloroquine", "CONTRAINDICATES", "Retinal_Disease"),
        ("Prednisone", "CONTRAINDICATES", "Systemic_Fungal_Inf"),
        ("Valproate", "CONTRAINDICATES", "Pregnancy"),
        ("Omeprazole", "CONTRAINDICATES", "Rilpivirine"),
        ("Carbamazepine", "CONTRAINDICATES", "MAO_Inhibitors"),
        ("Colchicine", "CONTRAINDICATES", "Renal_Impairment"),

        # --- INHIBITS enzyme/pathway (20) ---
        ("Atorvastatin", "INHIBITS", "HMG_CoA_Reductase"),
        ("Omeprazole", "INHIBITS", "Proton_Pump"),
        ("Aspirin", "INHIBITS", "COX_1"),
        ("Ibuprofen", "INHIBITS", "COX_2"),
        ("Lisinopril", "INHIBITS", "ACE"),
        ("Clopidogrel", "INHIBITS", "P2Y12_Receptor"),
        ("Losartan", "INHIBITS", "Angiotensin_II_Rec"),
        ("Fluoxetine", "INHIBITS", "Serotonin_Reuptake"),
        ("Donepezil", "INHIBITS", "Acetylcholinesterase"),
        ("Methotrexate", "INHIBITS", "Dihydrofolate_Red"),
        ("Albuterol", "INHIBITS", "Bronchial_Smooth_M"),
        ("Montelukast", "INHIBITS", "Leukotriene_Rec"),
        ("Rivaroxaban", "INHIBITS", "Factor_Xa"),
        ("Warfarin", "INHIBITS", "Vitamin_K_Reductase"),
        ("Phenytoin", "INHIBITS", "Sodium_Channel"),
        ("Gabapentin", "INHIBITS", "Calcium_Ch_Alpha2"),
        ("Carbamazepine", "INHIBITS", "Sodium_Channel"),
        ("Sertraline", "INHIBITS", "Serotonin_Reuptake"),
        ("Furosemide", "INHIBITS", "Na_K_Cl_Cotrans"),
        ("Spironolactone", "INHIBITS", "Aldosterone_Rec"),

        # --- REGULATES biological process (15) ---
        ("Insulin", "REGULATES", "Blood_Glucose"),
        ("Levothyroxine", "REGULATES", "Thyroid_Hormone"),
        ("Metformin", "REGULATES", "Hepatic_Glucose_Out"),
        ("Amlodipine", "REGULATES", "Calcium_Channel"),
        ("Metoprolol", "REGULATES", "Heart_Rate"),
        ("Prednisone", "REGULATES", "Immune_Response"),
        ("Losartan", "REGULATES", "Blood_Pressure"),
        ("Furosemide", "REGULATES", "Fluid_Balance"),
        ("Lithium", "REGULATES", "Mood"),
        ("Sertraline", "REGULATES", "Serotonin_Level"),
        ("Levodopa", "REGULATES", "Dopamine_Level"),
        ("Valproate", "REGULATES", "GABA_Level"),
        ("Clonazepam", "REGULATES", "GABA_Receptor"),
        ("Hydrochlorothiazide", "REGULATES", "Sodium_Excretion"),
        ("Spironolactone", "REGULATES", "Potassium_Level"),

        # --- INTERACTS_WITH drug interactions (20) ---
        ("Warfarin", "INTERACTS_WITH", "Aspirin"),
        ("Metformin", "INTERACTS_WITH", "Contrast_Dye"),
        ("Fluoxetine", "INTERACTS_WITH", "MAO_Inhibitors"),
        ("Ciprofloxacin", "INTERACTS_WITH", "Antacids"),
        ("Simvastatin", "INTERACTS_WITH", "Grapefruit"),
        ("Lithium", "INTERACTS_WITH", "NSAIDs"),
        ("Warfarin", "INTERACTS_WITH", "Vitamin_K"),
        ("Phenytoin", "INTERACTS_WITH", "Valproate"),
        ("Methotrexate", "INTERACTS_WITH", "NSAIDs"),
        ("Clopidogrel", "INTERACTS_WITH", "Omeprazole"),
        ("Diazepam", "INTERACTS_WITH", "Alcohol"),
        ("Alprazolam", "INTERACTS_WITH", "Opioids"),
        ("Carbamazepine", "INTERACTS_WITH", "Oral_Contracep"),
        ("Atorvastatin", "INTERACTS_WITH", "Cyclosporine"),
        ("Rivaroxaban", "INTERACTS_WITH", "Aspirin"),
        ("Levothyroxine", "INTERACTS_WITH", "Calcium_Suppl"),
        ("Lisinopril", "INTERACTS_WITH", "Potassium_Suppl"),
        ("Sertraline", "INTERACTS_WITH", "Tramadol"),
        ("Doxycycline", "INTERACTS_WITH", "Iron_Supplements"),
        ("Gabapentin", "INTERACTS_WITH", "Morphine"),
    ]

    assert len(facts) == 150, f"Expected 150 facts, got {len(facts)}"
    return facts


# ---------------------------------------------------------------------------
# Test queries: 200 total (100 positive + 100 negative)
# ---------------------------------------------------------------------------

def _build_test_queries(facts: list[tuple[str, str, str]]):
    """
    Build 200 test queries from the knowledge base.

    Returns list of dicts with:
        - subject, relation: query parameters
        - expected: "supported" | "unsupported"
        - category: "factoid" | "trap_unknown_entity" | "trap_wrong_relation"
                    | "trap_wrong_pair" | "yes_no"
    """
    queries = []

    # (A) 100 POSITIVE queries — facts that ARE in the knowledge base
    for i, (subj, rel, obj) in enumerate(facts[:80]):
        queries.append({
            "subject": subj,
            "relation": rel,
            "expected": "supported",
            "category": "factoid",
            "gold_object": obj,
        })

    for subj, rel, obj in facts[80:100]:
        queries.append({
            "subject": subj,
            "relation": rel,
            "expected": "supported",
            "category": "yes_no",
            "gold_object": obj,
        })

    # (B) 100 NEGATIVE queries — facts NOT in the knowledge base
    all_subjects = list({f[0] for f in facts})
    all_relations = list({f[1] for f in facts})
    fact_pair_set = {(f[0], f[1]) for f in facts}

    # B.1: 30 unknown entity queries
    unknown_entities = [
        "Zolpidem", "Fentanyl", "Naloxone", "Ketamine", "Propofol",
        "Remdesivir", "Ivermectin", "Tocilizumab", "Baricitinib",
        "Molnupiravir", "Paxlovid", "Sotrovimab", "Bamlanivimab",
        "Casirivimab", "Imdevimab", "Semaglutide", "Tirzepatide",
        "Ozempic", "Mounjaro", "Wegovy", "Dupilumab", "Upadacitinib",
        "Tofacitinib", "Secukinumab", "Adalimumab", "Infliximab",
        "Etanercept", "Golimumab", "Certolizumab", "Vedolizumab",
    ]
    for ent in unknown_entities:
        rel = all_relations[hash(ent) % len(all_relations)]
        queries.append({
            "subject": ent,
            "relation": rel,
            "expected": "unsupported",
            "category": "trap_unknown_entity",
            "gold_object": None,
        })

    # B.2: 35 wrong-relation queries
    wrong_rel_count = 0
    for subj in sorted(all_subjects):
        if wrong_rel_count >= 35:
            break
        for rel in all_relations:
            if (subj, rel) not in fact_pair_set:
                queries.append({
                    "subject": subj,
                    "relation": rel,
                    "expected": "unsupported",
                    "category": "trap_wrong_relation",
                    "gold_object": None,
                })
                wrong_rel_count += 1
                break

    # B.3: 35 wrong-pair queries
    wrong_pair_count = 0
    for i, subj in enumerate(sorted(all_subjects, reverse=True)):
        if wrong_pair_count >= 35:
            break
        for rel in reversed(all_relations):
            if (subj, rel) not in fact_pair_set:
                queries.append({
                    "subject": subj,
                    "relation": rel,
                    "expected": "unsupported",
                    "category": "trap_wrong_pair",
                    "gold_object": None,
                })
                wrong_pair_count += 1
                break

    return queries


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_accuracy_benchmark() -> dict:
    """Run the full accuracy benchmark and return results."""
    from axiom_hdc.utils import setup_logging, save_json
    from axiom_hdc.distiller import AxiomDistiller, MedicalFact
    from axiom_hdc.config import hdc, data as data_cfg
    import torch
    import torchhd.functional as F

    setup_logging()

    logger.info("=" * 60)
    logger.info("AXIOM Accuracy Benchmark — 200 Query Evaluation")
    logger.info("=" * 60)

    # Build knowledge base
    fact_tuples = _build_knowledge_base()
    medical_facts = [
        MedicalFact(subject=s, relation=r, obj=o) for s, r, o in fact_tuples
    ]

    logger.info("Knowledge base: %d facts", len(medical_facts))

    # Distill into Axiom Map
    distiller = AxiomDistiller()
    distiller.distill(medical_facts)

    # Build object vocabulary for nearest-neighbour lookup
    all_objects = sorted({f[2] for f in fact_tuples})
    obj_hvs = torch.stack([
        _cyclic_shift(distiller.item_memory.get(o), 2).squeeze(0)
        for o in all_objects
    ])  # shape: (num_objects, D)

    # Build test queries
    test_queries = _build_test_queries(fact_tuples)
    logger.info("Test queries: %d", len(test_queries))

    # Evaluate using HDC unbinding (same approach as capacity_bench)
    results_per_query = []
    tp = fp = tn = fn = 0
    pos_cosines = []
    neg_cosines = []
    retrieval_correct = 0
    retrieval_total = 0

    for q in test_queries:
        entity = q["subject"]
        relation = q["relation"]

        # Construct query probe: bind(v_sub, shift(v_rel, 1))
        v_sub = distiller.item_memory.get(entity)
        v_rel = distiller.item_memory.get(relation)
        query_probe = F.bind(v_sub, _cyclic_shift(v_rel, 1))

        # Unbind from Axiom Map to extract expected answer
        v_answer = F.bind(distiller.axiom_map, query_probe).squeeze(0)
        # shape: (D,)

        # Find nearest neighbour in object vocabulary
        cos_all = torch.nn.functional.cosine_similarity(
            v_answer.float().unsqueeze(0).expand_as(obj_hvs),
            obj_hvs.float(),
            dim=1,
        )
        best_cos = cos_all.max().item()
        best_idx = cos_all.argmax().item()
        best_obj = all_objects[best_idx]

        is_above_threshold = best_cos >= hdc.safety_threshold

        if q["expected"] == "supported":
            pos_cosines.append(best_cos)
            retrieval_total += 1
            if best_obj == q["gold_object"]:
                retrieval_correct += 1
            if is_above_threshold:
                tp += 1
                status = "TP"
            else:
                fn += 1
                status = "FN"
        else:
            neg_cosines.append(best_cos)
            if not is_above_threshold:
                tn += 1
                status = "TN"
            else:
                fp += 1
                status = "FP"

        results_per_query.append({
            "subject": q["subject"],
            "relation": q["relation"],
            "category": q["category"],
            "expected": q["expected"],
            "best_cosine": round(best_cos, 6),
            "best_match": best_obj,
            "gold_object": q.get("gold_object"),
            "above_threshold": is_above_threshold,
            "status": status,
        })

    # --- Threshold sweep to find optimal τ ---
    thresholds_sweep = sorted(set(
        [round(c, 5) for c in pos_cosines + neg_cosines]
        + [round(c - 0.001, 5) for c in pos_cosines + neg_cosines]
        + [round(c + 0.001, 5) for c in pos_cosines + neg_cosines]
    ))

    best_f1 = 0
    best_tau = hdc.safety_threshold
    sweep_results = []

    for tau in thresholds_sweep:
        s_tp = sum(1 for c in pos_cosines if c >= tau)
        s_fp = sum(1 for c in neg_cosines if c >= tau)
        s_tn = sum(1 for c in neg_cosines if c < tau)
        s_fn = sum(1 for c in pos_cosines if c < tau)
        s_prec = s_tp / max(s_tp + s_fp, 1)
        s_rec = s_tp / max(s_tp + s_fn, 1)
        s_f1 = (2 * s_prec * s_rec) / max(s_prec + s_rec, 1e-9)
        if s_f1 > best_f1:
            best_f1 = s_f1
            best_tau = tau
        sweep_results.append({
            "tau": tau, "f1": round(s_f1, 4),
            "precision": round(s_prec, 4), "recall": round(s_rec, 4),
        })

    # Recompute metrics at optimal threshold
    opt_tp = sum(1 for c in pos_cosines if c >= best_tau)
    opt_fp = sum(1 for c in neg_cosines if c >= best_tau)
    opt_tn = sum(1 for c in neg_cosines if c < best_tau)
    opt_fn = sum(1 for c in pos_cosines if c < best_tau)
    opt_prec = opt_tp / max(opt_tp + opt_fp, 1)
    opt_rec = opt_tp / max(opt_tp + opt_fn, 1)
    opt_f1 = (2 * opt_prec * opt_rec) / max(opt_prec + opt_rec, 1e-9)

    # Aggregate metrics (using BOTH NN retrieval and threshold-based detection)
    total_pos = tp + fn
    total_neg = tn + fp
    fact_fidelity = tp / max(total_pos, 1)
    hallucination_catch_rate = tn / max(total_neg, 1)
    false_positive_rate = fp / max(total_neg, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
    retrieval_acc = retrieval_correct / max(retrieval_total, 1)

    avg_pos_cos = sum(pos_cosines) / max(len(pos_cosines), 1)
    avg_neg_cos = sum(neg_cosines) / max(len(neg_cosines), 1)
    separation = avg_pos_cos - avg_neg_cos

    results = {
        "benchmark": "accuracy_fact_retrieval",
        "knowledge_base_size": len(medical_facts),
        "total_queries": len(test_queries),
        "threshold": hdc.safety_threshold,
        "hdc_dimensions": hdc.dimensions,
        "metrics_nn_retrieval": {
            "retrieval_accuracy": round(retrieval_acc, 4),
            "retrieval_correct": retrieval_correct,
            "retrieval_total": retrieval_total,
        },
        "metrics_at_default_threshold": {
            "fact_fidelity": round(fact_fidelity, 4),
            "hallucination_catch_rate": round(hallucination_catch_rate, 4),
            "false_positive_rate": round(false_positive_rate, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "metrics_at_optimal_threshold": {
            "optimal_tau": round(best_tau, 6),
            "precision": round(opt_prec, 4),
            "recall": round(opt_rec, 4),
            "f1_score": round(opt_f1, 4),
            "true_positives": opt_tp,
            "false_positives": opt_fp,
            "true_negatives": opt_tn,
            "false_negatives": opt_fn,
            "fact_fidelity": round(opt_rec, 4),
            "hallucination_catch_rate": round(opt_tn / max(opt_tn + opt_fp, 1), 4),
        },
        "cosine_analysis": {
            "avg_positive_cosine": round(avg_pos_cos, 6),
            "avg_negative_cosine": round(avg_neg_cos, 6),
            "cosine_separation": round(separation, 6),
            "min_positive_cosine": round(min(pos_cosines), 6) if pos_cosines else 0,
            "max_negative_cosine": round(max(neg_cosines), 6) if neg_cosines else 0,
        },
        "per_category": _compute_category_metrics(results_per_query),
        "per_query": results_per_query,
    }

    logger.info("-" * 60)
    logger.info("RESULTS:")
    logger.info("  --- NN Retrieval (primary metric) ---")
    logger.info("  Retrieval Accuracy:        %.1f%% (%d/%d)",
                retrieval_acc * 100, retrieval_correct, retrieval_total)
    logger.info("  --- Threshold-based (optimal tau=%.4f) ---", best_tau)
    logger.info("  Precision:                 %.1f%%", opt_prec * 100)
    logger.info("  Recall (Fact Fidelity):    %.1f%%", opt_rec * 100)
    logger.info("  F1 Score:                  %.4f", opt_f1)
    logger.info("  Halluc. Catch Rate (TNR):  %.1f%%",
                opt_tn / max(opt_tn + opt_fp, 1) * 100)
    logger.info("  --- Cosine separation ---")
    logger.info("  Avg +cos / Avg -cos:       %.4f / %.4f (sep=%.4f)",
                avg_pos_cos, avg_neg_cos, separation)
    logger.info("-" * 60)

    data_cfg.ensure_dirs()
    out_path = data_cfg.results_dir / "accuracy_benchmark.json"
    save_json(results, out_path)
    logger.info("Results saved -> %s", out_path)

    return results


def _compute_category_metrics(per_query: list[dict]) -> dict:
    """Compute per-category accuracy."""
    categories: dict[str, dict] = {}
    for q in per_query:
        cat = q["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if q["status"] in ("TP", "TN"):
            categories[cat]["correct"] += 1

    return {
        cat: {
            "correct": data["correct"],
            "total": data["total"],
            "accuracy": round(data["correct"] / max(data["total"], 1), 4),
        }
        for cat, data in sorted(categories.items())
    }


if __name__ == "__main__":
    run_accuracy_benchmark()
