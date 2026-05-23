from dataclasses import dataclass
from .evaluation_result import EvaluationResult

@dataclass
class TrainingResult:
    """Container for model training configuration and evaluation output.

    Attributes:
        n_trees (int): Number of trees used during training.
        sampling_size (int): Number of samples used per tree.
        threshold (float): Threshold value used for evaluation.
        evaluation (EvaluationResult): Evaluation metrics produced after
            training.
    """
    n_trees: int
    sampling_size: int
    threshold: float
    evaluation: EvaluationResult