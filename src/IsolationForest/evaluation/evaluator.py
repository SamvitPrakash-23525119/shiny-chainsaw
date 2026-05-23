import numpy as np
from src.utils.metrics import accuracy, precision, recall, f1
from src.models.evaluation_result import EvaluationResult

class Evaluator:
    """Evaluator for anomaly detection models.

    Computes binary predictions from anomaly scores produced by an
    `IsolationForest`, then calculates evaluation metrics (accuracy,
    precision, recall, F1) and returns an `EvaluationResult`.
    """

    def evaluate(self, forest: IsolationForest, X, actual_X, threshold) -> EvaluationResult:
        """Evaluate a trained forest on a dataset and return metrics.

        The method computes anomaly scores for each observation in `X`,
        thresholds the scores at the given percentile, produces binary
        predictions and computes standard classification metrics against
        the provided ground-truth labels `actual_X`.

        Args:
            forest (IsolationForest): Trained isolation forest used to
                compute anomaly scores.
            X (Iterable): Iterable or array of observations to score. Each
                element should be suitable input for
                `forest.anomaly_score`.
            actual_X (array-like): Ground-truth binary labels (1 for
                anomalous/malicious, 0 for normal) aligned with `X`.
            threshold (float): Percentile (0-100) used to select the score
                cutoff. For example, `threshold=95` uses the 95th
                percentile of scores as the decision boundary.

        Returns:
            EvaluationResult: Container with computed accuracy, precision,
                recall, F1 score and confusion matrix counts (tp, tn, fp,
                fn).
        """
        scores = np.array([forest.anomaly_score(x) for x in X])
        t = np.percentile(scores, threshold)

        predictions_binary = self._predict(scores, t)

        tp = np.sum((predictions_binary == 1) & (actual_X == 1))
        fp = np.sum((predictions_binary == 1) & (actual_X == 0))
        tn = np.sum((predictions_binary == 0) & (actual_X == 0))
        fn = np.sum((predictions_binary == 0) & (actual_X == 1))

        accuracy_score = accuracy(tp, tn, fp, fn)
        precision_score = precision(tp, fp)
        recall_score = recall(tp, fn)
        f1_score = f1(precision_score, recall_score)

        return EvaluationResult(
            accuracy=accuracy_score,
            precision=precision_score,
            recall=recall_score,
            f1_score=f1_score,
            tp=tp,
            tn=tn,
            fp=fp,
            fn= fn
        )

    def _predict(self, scores: np.ndarray, threshold):
        """Convert continuous anomaly scores into binary predictions.

        Args:
            scores (np.ndarray): Array of anomaly scores for observations.
            threshold (float): Score threshold; scores >= threshold are
                predicted as anomalies (1), otherwise normal (0).

        Returns:
            np.ndarray: Binary prediction array of the same shape as
                `scores` with values {0, 1}.
        """
        return np.where(scores < threshold , 0, 1)



        