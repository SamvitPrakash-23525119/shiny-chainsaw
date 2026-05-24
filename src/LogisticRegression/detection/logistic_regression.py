import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier:
    """Logistic Regression classifier for insider threat detection.

    Wraps a logistic regression model and exposes a fit/predict interface
    consistent with the rest of the threat detection pipeline.
    """

    def __init__(self, max_iter: int, class_weight: str | None, random_state: int):
        """Initialise the classifier.

        Args:
            max_iter (int): Maximum number of iterations for the solver to
                converge.
            class_weight (str | None): Weighting strategy for class labels.
                Pass ``'balanced'`` to weight classes inversely proportional
                to their frequencies, or ``None`` for no weighting.
            random_state (int): Seed for the random number generator to
                ensure reproducibility.

        Attributes:
            model: Underlying model instance; ``None`` until ``fit`` is called.
        """
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the logistic regression model on labelled data.

        Args:
            X (np.ndarray): 2D feature matrix of shape ``(n_samples,
                n_features)``.
            y (np.ndarray): 1D binary label array of shape ``(n_samples,)``
                where ``1`` indicates a malicious observation and ``0``
                indicates a normal one.

        Returns:
            None
        """
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class predictions for observations in ``X``.

        Args:
            X (np.ndarray): 2D feature matrix of shape ``(n_samples,
                n_features)``.

        Returns:
            np.ndarray: 1D integer array of shape ``(n_samples,)`` containing
                predicted class labels (``0`` for normal, ``1`` for
                malicious).
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates for observations in ``X``.

        Args:
            X (np.ndarray): 2D feature matrix of shape ``(n_samples,
                n_features)``.

        Returns:
            np.ndarray: 2D array of shape ``(n_samples, 2)`` where column
                ``0`` holds the probability of the normal class and column
                ``1`` holds the probability of the malicious class.
        """
        return self.model.predict_proba(X)
