import json
import os
import numpy as np
from datetime import datetime
from src.LogisticRegression.models.training_result import TrainingResult

class TrainingLogger:
    """Persist logistic regression training iteration results to a JSONL log file."""

    def __init__(self, log_path):
        """Initialise the logger and ensure the log directory exists.

        Args:
            log_path (str): Path to the JSONL log file.
        """
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, iteration: int, result: TrainingResult):
        """Append a training result for one iteration to the log file.

        Args:
            iteration (int): Training iteration number.
            result (TrainingResult): Training result to serialise and log.
        """
        evaluation_metrics = {
            'precision': result.evaluation.precision,
            'recall': result.evaluation.recall,
            'f1_score': result.evaluation.f1_score,
            'accuracy': result.evaluation.accuracy,
            'tp': result.evaluation.tp,
            'tn': result.evaluation.tn,
            'fp': result.evaluation.fp,
            'fn': result.evaluation.fn,
        }

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'max_iter': result.max_iter,
            'class_weight': result.class_weight,
            'random_state': result.random_state,
            'evaluation_metrics': evaluation_metrics,
        }

        log_entry = self._sanitize(log_entry)

        with open(self.log_path, 'a') as log_file:
            log_file.write(json.dumps(log_entry) + '\n')

    def _sanitize(self, obj):
        """Convert numpy scalars inside nested structures to plain Python types.

        Args:
            obj: Object, list or dictionary to sanitise.

        Returns:
            object: Sanitised object with numpy scalar values converted to
                native Python types.
        """
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        return obj
