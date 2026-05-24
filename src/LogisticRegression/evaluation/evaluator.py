import numpy as np
from src.IsolationForest.utils.metrics import accuracy, precision, recall, f1
from src.LogisticRegression.models.evaluation_result import EvaluationResult

class Evaluator:
    """Evaluator for logistic regression models.

    Obtains binary predictions directly from a trained
    ``LogisticRegressionClassifier`` and computes standard classification
    metrics, returning an ``EvaluationResult``.
    """

    def evaluate(self, model, X: np.ndarray, actual_X: np.ndarray) -> EvaluationResult:
        """Evaluate a trained model on a dataset and return metrics.

        Args:
            model (LogisticRegressionClassifier): Trained logistic regression
                model used to generate predictions.
            X (np.ndarray): 2D feature matrix of shape ``(n_samples,
                n_features)`` to evaluate.
            actual_X (np.ndarray): Ground-truth binary labels of shape
                ``(n_samples,)`` where ``1`` indicates malicious and ``0``
                indicates normal.

        Returns:
            EvaluationResult: Container with computed accuracy, precision,
                recall, F1 score and confusion matrix counts (tp, tn, fp,
                fn).
        """
        predictions = model.predict(X)

        tp = int(np.sum((predictions == 1) & (actual_X == 1)))
        fp = int(np.sum((predictions == 1) & (actual_X == 0)))
        tn = int(np.sum((predictions == 0) & (actual_X == 0)))
        fn = int(np.sum((predictions == 0) & (actual_X == 1)))

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
            fn=fn
        )
