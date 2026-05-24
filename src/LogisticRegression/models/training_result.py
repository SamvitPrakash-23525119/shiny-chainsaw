from dataclasses import dataclass
from .evaluation_result import EvaluationResult

@dataclass
class TrainingResult:
    """Container for logistic regression training configuration and evaluation output.

    Attributes:
        max_iter (int): Maximum number of solver iterations used during training.
        class_weight (str | None): Class weighting strategy applied to handle
            class imbalance. ``'balanced'`` weights classes inversely
            proportional to their frequencies; ``None`` applies no weighting.
        random_state (int): Random seed used for reproducibility.
        evaluation (EvaluationResult): Evaluation metrics produced after
            training.
    """
    max_iter: int
    class_weight: str | None
    random_state: int
    evaluation: EvaluationResult
