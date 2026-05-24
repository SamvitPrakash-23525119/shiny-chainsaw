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

    def get_pattern_report(self, feature_names: list) -> list:
        """Return learned feature weights ranked by importance.

        Each weight represents how strongly that feature pushes toward the
        malicious class. A high positive weight means the feature is a strong
        pattern indicator of suspicious behaviour. Sorted by absolute weight
        so both strongly positive and strongly negative drivers are visible.

        Args:
            feature_names (list[str]): Names of features in the same order
                they were passed to ``fit()``.

        Returns:
            list[dict]: Ranked list of dicts with keys ``feature``,
                ``weight``, and ``direction`` (``'risk'`` for positive
                weights, ``'protective'`` for negative).
        """
        weights = self.model.coef_[0]
        paired = sorted(
            zip(feature_names, weights),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return [
            {
                "feature": name,
                "weight": round(float(w), 4),
                "direction": "risk" if w >= 0 else "protective",
            }
            for name, w in paired
        ]

    def explain_prediction(self, x: np.ndarray, feature_names: list, top_n: int = 5) -> list:
        """Return the top feature contributions for a single observation.

        Contribution is computed as ``weight * feature_value`` — the portion
        of the log-odds score that each feature is responsible for. Features
        with the largest absolute contribution drove the prediction most.

        Args:
            x (np.ndarray): 1D feature vector for a single observation.
            feature_names (list[str]): Names matching the order of ``x``.
            top_n (int): Number of top contributors to return. Defaults to 5.

        Returns:
            list[dict]: Top ``top_n`` features sorted by absolute
                contribution, each with keys ``feature``, ``value``,
                ``weight``, and ``contribution``.
        """
        weights = self.model.coef_[0]
        contributions = weights * x
        ranked = sorted(
            zip(feature_names, x, weights, contributions),
            key=lambda t: abs(t[3]),
            reverse=True
        )
        return [
            {
                "feature": name,
                "value": round(float(val), 4),
                "weight": round(float(w), 4),
                "contribution": round(float(contrib), 4),
            }
            for name, val, w, contrib in ranked[:top_n]
        ]
