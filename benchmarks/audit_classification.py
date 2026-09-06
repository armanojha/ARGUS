"""Phase 21-B: Query Classification Audit.

Runs all eval queries through the canonical classifier and compares
against expected eval-plan classes. Produces accuracy, confusion matrix,
and misclassification analysis.

Usage:
    python benchmarks/audit_classification.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval.policy import QuestionPattern
from app.retrieval.router import RetrievalPolicyRouter


def load_eval_queries() -> list[dict]:
    eval_path = Path("benchmarks/eval_data/eval_plan_v1.json")
    with eval_path.open() as f:
        plan = json.load(f)
    return plan.get("queries", [])


def audit_classification():
    print("=" * 70)
    print("Phase 21-B: Query Classification Audit")
    print("=" * 70)

    router = RetrievalPolicyRouter()
    queries = load_eval_queries()

    results = []
    correct = 0
    total = 0
    misclassifications = []

    # Confusion matrix: expected -> predicted (include all canonical patterns)
    all_eval_classes = sorted(set(q["class"] for q in queries))
    all_pred_classes = sorted(set(p.value for p in QuestionPattern))
    all_classes = sorted(set(all_eval_classes) | set(all_pred_classes))
    confusion = {e: {p: 0 for p in all_classes} for e in all_classes}

    for q in queries:
        expected_class = q["class"]
        query_text = q["query"]

        # Classify using canonical router
        predicted_pattern = router.classify_question(query_text)
        predicted_class = predicted_pattern.value

        # Check if the predicted pattern matches the expected class
        # Some eval classes map to different canonical patterns
        expected_pattern = QuestionPattern.from_eval_class(expected_class)
        is_correct = (predicted_pattern == expected_pattern)

        if is_correct:
            correct += 1
        else:
            misclassifications.append({
                "id": q["id"],
                "query": query_text,
                "expected_class": expected_class,
                "predicted_class": predicted_class,
                "expected_pattern": expected_pattern.value,
                "predicted_pattern": predicted_pattern.value,
                "query_text": query_text,
            })

        confusion[expected_class][predicted_class] += 1
        total += 1

        results.append({
            "id": q["id"],
            "query": query_text,
            "expected_class": expected_class,
            "predicted_class": predicted_class,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0

    # Print results
    print(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.1%}")

    # Per-class breakdown
    print(f"\n{'Class':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 55)
    for cls in all_classes:
        cls_results = [r for r in results if r["expected_class"] == cls]
        cls_correct = sum(1 for r in cls_results if r["correct"])
        cls_total = len(cls_results)
        cls_acc = cls_correct / cls_total if cls_total > 0 else 0
        print(f"{cls:<25} {cls_correct:>8} {cls_total:>6} {cls_acc:>10.1%}")

    # Confusion matrix
    print(f"\nConfusion Matrix (rows=expected, cols=predicted):")
    # Header
    header = f"{'':>25}" + "".join(f"{c[:12]:>14}" for c in all_classes)
    print(header)
    for expected in all_classes:
        row = f"{expected:>25}"
        for predicted in all_classes:
            count = confusion[expected][predicted]
            row += f"{count:>14}"
        print(row)

    # Misclassifications
    print(f"\nMisclassifications ({len(misclassifications)}):")
    print("-" * 70)
    for m in misclassifications:
        print(f"\n  [{m['id']}] Expected: {m['expected_class']} -> Predicted: {m['predicted_class']}")
        print(f"    Query: {m['query_text'][:80]}")
        # Explain why
        if m["predicted_class"] in ("conceptual", "normal_qa"):
            print(f"    Cause: Fell through to CONCEPTUAL fallback (no specific keyword match)")
        elif m["expected_class"] == "conflict" and m["predicted_class"] != "conflict":
            print(f"    Cause: Conflict keywords not detected in query")
        elif m["expected_class"] == "multi_hop" and m["predicted_class"] != "multi_hop":
            print(f"    Cause: Multi-hop pattern words not detected")
        elif m["expected_class"] == "complex_research" and m["predicted_class"] != "complex_research":
            print(f"    Cause: Complex research words not detected")
        elif m["expected_class"] == "numerical" and m["predicted_class"] != "numerical":
            print(f"    Cause: Numerical query lacks specific numerical keywords")
        elif m["expected_class"] == "adversarial" and m["predicted_class"] != "adversarial":
            print(f"    Cause: Adversarial indicators not matched")
        elif m["expected_class"] == "absent_info" and m["predicted_class"] != "absent_info":
            print(f"    Cause: Absent info pattern not detected")
        elif m["expected_class"] == "simple_lookup" and m["predicted_class"] != "simple_lookup":
            print(f"    Cause: Simple lookup not recognized (no exact-term indicators)")

    # Damage assessment
    print(f"\nDamage Assessment:")
    print("-" * 70)
    # Categorize misclassifications by severity
    high_damage = []
    medium_damage = []
    low_damage = []

    for m in misclassifications:
        exp = m["expected_class"]
        pred = m["predicted_class"]

        # High damage: planner should activate but doesn't, or vice versa
        planner_classes = {"conflict", "complex_research", "multi_hop"}
        if exp in planner_classes and pred not in planner_classes:
            high_damage.append(m)
        elif exp not in planner_classes and pred in planner_classes:
            medium_damage.append(m)
        # Medium damage: different retrieval strategy
        elif exp in ("numerical", "technical_explanation") and pred == "conceptual":
            medium_damage.append(m)
        else:
            low_damage.append(m)

    print(f"  High damage (planner mismatch): {len(high_damage)}")
    for m in high_damage:
        print(f"    [{m['id']}] {m['expected_class']} -> {m['predicted_class']}: {m['query_text'][:60]}")
    print(f"  Medium damage (strategy mismatch): {len(medium_damage)}")
    for m in medium_damage:
        print(f"    [{m['id']}] {m['expected_class']} -> {m['predicted_class']}: {m['query_text'][:60]}")
    print(f"  Low damage (harmless distinction): {len(low_damage)}")
    for m in low_damage:
        print(f"    [{m['id']}] {m['expected_class']} -> {m['predicted_class']}: {m['query_text'][:60]}")

    # Save report
    report = {
        "audit": "phase21_classification_audit",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class": {},
        "confusion_matrix": confusion,
        "misclassifications": misclassifications,
        "damage_assessment": {
            "high": len(high_damage),
            "medium": len(medium_damage),
            "low": len(low_damage),
            "high_details": high_damage,
            "medium_details": medium_damage,
        },
    }
    for cls in all_classes:
        cls_results = [r for r in results if r["expected_class"] == cls]
        cls_correct = sum(1 for r in cls_results if r["correct"])
        cls_total = len(cls_results)
        report["per_class"][cls] = {
            "accuracy": cls_correct / cls_total if cls_total > 0 else 0,
            "correct": cls_correct,
            "total": cls_total,
        }

    report_dir = Path("data/benchmark_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "phase21_classification_audit.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to data/benchmark_reports/phase21_classification_audit.json")

    return report


if __name__ == "__main__":
    audit_classification()
