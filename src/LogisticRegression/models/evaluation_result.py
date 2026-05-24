from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """Container for evaluation metrics and confusion matrix counts.

    Attributes:
        f1_score (float): F1 score combining precision and recall.
        precision (float): Precision score.
        recall (float): Recall score.
        accuracy (float): Accuracy score.
        tp (int): True positives count.
        tn (int): True negatives count.
        fp (int): False positives count.
        fn (int): False negatives count.
    """
    f1_score: float
    precision: float
    recall: float
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int
