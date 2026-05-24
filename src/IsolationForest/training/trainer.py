import random
import numpy as np
from .training_logger import TrainingLogger
from src.IsolationForest.evaluation.evaluator import Evaluator
from src.IsolationForest.detection.isolation_forest import IsolationForest
from src.IsolationForest.utils.pickler import Pickler
from src.IsolationForest.models.training_result import TrainingResult
from src.IsolationForest.config.file_paths import BEST_MODEL_PATH, TRAINING_LOG_PATH

class Trainer:
    """Train and select the best isolation forest model.

    The trainer repeatedly samples hyperparameters, fits an
    `IsolationForest`, evaluates it, logs the result and keeps the best
    model according to recall.
    """
    N_TREES_RANGE = (10, 40)
    SAMPLING_SIZE_RANGE = (100, 768)
    THRESHOLD_RANGE = (90, 99)

    def __init__(self, iterations):
        """Initialize the trainer.

        Args:
            iterations (int): Number of training iterations to run while
                searching for the best model.
        """
        self.iterations = iterations
        self.evaluator = Evaluator()
        self.logger = TrainingLogger(log_path=TRAINING_LOG_PATH)
        self.pickler = Pickler(file_path=BEST_MODEL_PATH)
        self.model = None

    def train(self, X, actual_X, n_trees_range=N_TREES_RANGE, sampling_size_range=SAMPLING_SIZE_RANGE, threshold_range=THRESHOLD_RANGE):
        """Train multiple candidate models and return the best result.

        The method filters normal observations for training, samples
        hyperparameters on each iteration, evaluates each model and keeps
        the result with the highest recall.

        Args:
            X (np.ndarray): Feature matrix used for training and evaluation.
            actual_X (np.ndarray): Binary labels aligned with `X`.
            n_trees_range (tuple[int, int]): Inclusive range for the number
                of trees to sample.
            sampling_size_range (tuple[int, int]): Inclusive range for the
                sampling size to sample.
            threshold_range (tuple[int, int]): Inclusive range for the
                evaluation threshold percentile to sample.

        Returns:
            TrainingResult: Best training result observed across all
                iterations.
        """
        best_result = None
        best_model = None
        X_prime = X[actual_X == 0]
        length = len(X_prime)

        for i in range(self.iterations):
            n_trees = random.randint(n_trees_range[0], n_trees_range[1])
            sampling_size = random.randint(sampling_size_range[0], min(sampling_size_range[1], length))
            threshold = random.randint(threshold_range[0], threshold_range[1])

            model = IsolationForest(n_trees=n_trees, sampling_size=sampling_size)
            model.fit(X_prime)

            result = self.evaluator.evaluate(model, X, actual_X, threshold)
            training_result = TrainingResult(
                    n_trees=n_trees,
                    sampling_size=sampling_size,
                    threshold=threshold,
                    evaluation=result
                )

            self.logger.log(i, training_result)

            if best_result is None or result.recall >= best_result.evaluation.recall:
                best_result = training_result
                best_model = model
                    
        
        self.pickler.save(best_model, best_result)
        self.model = best_model

        return best_result

    def get_model(self):
        """Return the most recently selected best model.

        Returns:
            IsolationForest | None: The trained model if available,
                otherwise `None`.
        """
        if self.model is None:
            return None
        return self.model