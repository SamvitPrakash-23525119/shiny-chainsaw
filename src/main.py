from src.LogisticRegression.pipeline.threat_detection_pipeline import ThreatDetectionPipeline

def main():
    print("Training Logistic Regression model...")
    pipeline = ThreatDetectionPipeline()
    pipeline.load_model(force_retrain=True)

    metrics = pipeline.get_metrics()
    print(f"Accuracy:  {metrics.evaluation.accuracy:.4f}")
    print(f"Precision: {metrics.evaluation.precision:.4f}")
    print(f"Recall:    {metrics.evaluation.recall:.4f}")
    print(f"F1 Score:  {metrics.evaluation.f1_score:.4f}")
    print(f"TP: {metrics.evaluation.tp}  TN: {metrics.evaluation.tn}  FP: {metrics.evaluation.fp}  FN: {metrics.evaluation.fn}")


if __name__ == "__main__":
    main()