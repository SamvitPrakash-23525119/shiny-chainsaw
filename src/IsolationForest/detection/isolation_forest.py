import numpy as np
from .isolation_tree import IsolationTree
from src.IsolationForest.utils.c_factor import c_factor

class IsolationForest:
    """Isolation Forest ensemble for anomaly scoring.

    The `IsolationForest` builds a collection of `IsolationTree` instances
    using random subsamples of the data and produces an anomaly score for
    observations based on average path lengths through the trees.
    """
    def __init__(self, n_trees: int, sampling_size: int):
        """Initialize the isolation forest.

        Args:
            n_trees (int): Number of trees in the forest.
            sampling_size (int): Number of samples to draw for each tree.

        Attributes:
            forest (List[IsolationTree]): List that will hold trained trees.
        """
        self.n_trees = n_trees
        self.sampling_size = sampling_size
        self.forest = []

    def fit(self, X: np.ndarray):
        """Train the forest on the provided data matrix.

        Builds `n_trees` isolation trees by sampling `sampling_size`
        observations from `X` without replacement and fitting an
        `IsolationTree` to each subsample.

        Args:
            X (np.ndarray): 2D array of shape `(n_samples, n_features)`
                containing the training data.

        Returns:
            None
        """
        height_limit = int(np.ceil(np.log2(self.sampling_size)))

        for i in range(self.n_trees):
            X_prime = self._sample(X)
            tree = IsolationTree(max_height=height_limit)
            tree.fit(X_prime)
            self.forest.append(tree)

    def _sample(self, X: np.ndarray) -> np.ndarray:
        """Draw a random subsample of rows from `X`.

        Args:
            X (np.ndarray): 2D array of shape `(n_samples, n_features)` to
                sample from.

        Returns:
            np.ndarray: Subsampled matrix of shape `(sampling_size,
                n_features)`.
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.sampling_size, replace=False)
        return X[indices]

    def anomaly_score(self, X: np.ndarray) -> float:
        """Compute an anomaly score for a single observation.

        The method computes the path length of `X` in each tree, averages
        those lengths and converts the average to a score using the
        `c_factor` normalisation. Higher returned scores indicate more
        anomalous observations (shorter average path lengths).

        Args:
            X (np.ndarray): 1D array representing a single observation, or a
                2D array shaped `(1, n_features)`.

        Returns:
            float: Anomaly score in the range (0, 1], where values closer to
                1 represent stronger anomalies.
        """
        path_lengths = []

        for i in range(self.n_trees):
            path_length = self.forest[i].path_length(X)
            path_lengths.append(path_length)

        avg_path_length = np.mean(path_lengths)
        score = 2 ** (-(avg_path_length / c_factor(self.sampling_size)))

        return score