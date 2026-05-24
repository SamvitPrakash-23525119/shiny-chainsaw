import random
import numpy as np
from .training_logger import TrainingLogger
from src.LogisticRegression.evaluation.evaluator import Evaluator
from src.LogisticRegression.detection.logistic_regression import LogisticRegressionClassifier
from src.LogisticRegression.models.training_result import TrainingResult
from src.IsolationForest.utils.pickler import Pickler
from src.LogisticRegression.config.file_paths import LR_MODEL_PATH, LR_TRAINING_LOG_PATH

class Trainer:
    """Train and select the best logistic regression model.

    The trainer repeatedly samples hyperparameters, fits a
    ``LogisticRegressionClassifier``, evaluates it on held-out data, logs
    the result and keeps the best model according to recall.
    """

    MAX_ITER_RANGE = (500, 5000)
    CLASS_WEIGHT_OPTIONS = ['balanced', None]
    RANDOM_STATE_RANGE = (0, 9999)

    def __init__(self, iterations: int):
        """Initialise the trainer.

        Args:
            iterations (int): Number of training iterations to run while
                searching for the best model.
        """
        self.iterations = iterations
        self.evaluator = Evaluator()
        self.logger = TrainingLogger(log_path=LR_TRAINING_LOG_PATH)
        self.pickler = Pickler(file_path=LR_MODEL_PATH)
        self.model = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_iter_range: tuple = MAX_ITER_RANGE,
        class_weight_options: list = CLASS_WEIGHT_OPTIONS,
        random_state_range: tuple = RANDOM_STATE_RANGE,
    ) -> TrainingResult:
        """Train multiple candidate models and return the best result.

        Samples hyperparameters on each iteration, fits a classifier on the
        training split, evaluates on the test split and keeps the result with
        the highest recall.

        Args:
            X_train (np.ndarray): Feature matrix for training, shape
                ``(n_train, n_features)``.
            y_train (np.ndarray): Binary labels for training, shape
                ``(n_train,)``.
            X_test (np.ndarray): Feature matrix for evaluation, shape
                ``(n_test, n_features)``.
            y_test (np.ndarray): Binary labels for evaluation, shape
                ``(n_test,)``.
            max_iter_range (tuple[int, int]): Inclusive range for the maximum
                solver iteration count to sample.
            class_weight_options (list): Options for the class weighting
                strategy to sample from.
            random_state_range (tuple[int, int]): Inclusive range for the
                random seed to sample.

        Returns:
            TrainingResult: Best training result observed across all
                iterations.
        """
        best_result = None
        best_model = None

        for i in range(self.iterations):
            max_iter = random.randint(max_iter_range[0], max_iter_range[1])
            class_weight = random.choice(class_weight_options)
            random_state = random.randint(random_state_range[0], random_state_range[1])

            model = LogisticRegressionClassifier(
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
            )
            model.fit(X_train, y_train)

            result = self.evaluator.evaluate(model, X_test, y_test)
            training_result = TrainingResult(
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                evaluation=result,
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
            LogisticRegressionClassifier | None: The trained model if
                available, otherwise ``None``.
        """
        if self.model is None:
            return None
        return self.model
