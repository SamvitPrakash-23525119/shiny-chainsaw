from src.LogisticRegression.pipeline.threat_detection_pipeline import ThreatDetectionPipeline
from src.IsolationForest.preprocessing.vectorizer import FEATURE_NAMES

def main():
    print("=" * 60)
    print("LOGISTIC REGRESSION - INSIDER THREAT DETECTION")
    print("=" * 60)

    # --- Train ---
    print("\nTraining model...")
    pipeline = ThreatDetectionPipeline()
    pipeline.load_model(force_retrain=True)

    # --- Evaluation metrics ---
    metrics = pipeline.get_metrics()
    print("\n[ Evaluation Metrics ]")
    print(f"  Accuracy:  {metrics.evaluation.accuracy:.4f}")
    print(f"  Precision: {metrics.evaluation.precision:.4f}")
    print(f"  Recall:    {metrics.evaluation.recall:.4f}")
    print(f"  F1 Score:  {metrics.evaluation.f1_score:.4f}")
    print(f"  TP: {metrics.evaluation.tp}  TN: {metrics.evaluation.tn}  "
          f"FP: {metrics.evaluation.fp}  FN: {metrics.evaluation.fn}")

    # --- Pattern report: learned feature weights ---
    print("\n[ Pattern Report - Learned Risk Indicators (ranked by weight) ]")
    pattern_report = pipeline.model.get_pattern_report(FEATURE_NAMES)
    for rank, entry in enumerate(pattern_report, start=1):
        direction_tag = "[+] RISK" if entry["direction"] == "risk" else "[-] protective"
        print(f"  {rank:>2}. {entry['feature']:<30}  weight: {entry['weight']:>8.4f}  {direction_tag}")

    # --- Per-prediction explanations for flagged users ---
    print("\n[ Flagged Users - Pattern Match Explanations ]")
    predictions = pipeline.test_csv()
    flagged = [p for p in predictions if p.is_malicious]
    flagged_sorted = sorted(flagged, key=lambda p: p.anomaly_score, reverse=True)

    if not flagged:
        print("  No users flagged on test set.")
    else:
        print(f"  {len(flagged)} user(s) flagged. Showing top 5 by suspicion score:\n")
        for i, pred in enumerate(flagged_sorted[:5], start=1):
            print(f"  User {i}  |  Suspicion score: {pred.anomaly_score:.4f}")
            explanation = pipeline.model.explain_prediction(
                pred.user_entity.vector, FEATURE_NAMES, top_n=5
            )
            for factor in explanation:
                tag = "[+]" if factor["contribution"] > 0 else "[-]"
                print(f"    {tag} {factor['feature']:<30}  "
                      f"value: {factor['value']:>8.4f}  "
                      f"contribution: {factor['contribution']:>8.4f}")
            print()

    print("=" * 60)


if __name__ == "__main__":
    main()