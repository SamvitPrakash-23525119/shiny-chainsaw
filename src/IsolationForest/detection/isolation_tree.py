import random
import numpy as np
from .isolation_tree_node import IsolationTreeNode
from src.utils.c_factor import c_factor


class IsolationTree:
    """Single isolation tree used by the Isolation Forest ensemble.

    The `IsolationTree` recursively partitions data by randomly selecting a
    feature and split value until a maximum height is reached or leaf
    conditions are met. The tree can compute the path length for an
    observation which is used in anomaly scoring.
    """

    def __init__(self, max_height: int):
        """Initialize the isolation tree.

        Args:
            max_height (int): Maximum depth (height) of the tree. Tree
                growth stops when `current_height >= max_height`.
        """
        self.max_height = max_height
        self.root = None

    def fit(self, X: np.ndarray):
        """Build the isolation tree from the provided data.

        Args:
            X (np.ndarray): 2D array of shape `(n_samples, n_features)`
                containing the data to build the tree from.

        Returns:
            None
        """
        self.root = self._build_tree(X, current_height=0)

    def _build_tree(self, X: np.ndarray, current_height: int) -> IsolationTreeNode:
        """Recursively build the isolation tree.

        The recursion terminates when the `current_height` reaches
        `self.max_height` or when there is one or zero samples at the node,
        in which case a leaf node is returned with the node `size`.

        Args:
            X (np.ndarray): Subset of data reaching the current node.
            current_height (int): Current depth of recursion.

        Returns:
            IsolationTreeNode: Constructed tree node (internal or leaf).
        """

        n_samples, n_features = X.shape

        if current_height >= self.max_height or n_samples <= 1:
            return IsolationTreeNode(
                size=n_samples,
                is_leaf=True
            )

        feature_index = random.randint(0, n_features - 1)

        feature_values = X[:, feature_index]

        min_val = np.min(feature_values)
        max_val = np.max(feature_values)

        split_value = random.uniform(min_val, max_val)
        
        feature_values = X[:, feature_index]

        left_mask = feature_values < split_value
        right_mask = feature_values >= split_value

        left = X[left_mask]
        right = X[right_mask]


        left_tree = self._build_tree(left, current_height + 1)
        right_tree = self._build_tree(right, current_height + 1)

        return IsolationTreeNode(
            feature_index=feature_index,
            split_value=split_value,
            left=left_tree,
            right=right_tree,
            size=n_samples,
            is_leaf=False
        )

    def path_length(self, X: np.ndarray) -> float:
        """Compute the path length for an observation through the tree.

        Args:
            X (np.ndarray): 1D array representing a single observation.

        Returns:
            float: Path length (possibly fractional due to `c_factor`
                correction for leaf sizes).
        """
        return self._traverse(self.root, X, 0)

    def _traverse(self, node: IsolationTreeNode, X: np.ndarray, current_length: int) -> float:
        """Traverse the tree following splits to compute path length.

        The traversal descends left or right depending on whether the
        observation's feature value is below the node's split. When a leaf
        is reached, the `c_factor` correction is added to the current
        traversal length to approximate the expected path length for
        instances in that leaf.

        Args:
            node (IsolationTreeNode): Current node in the traversal.
            X (np.ndarray): Observation being traversed.
            current_length (int): Length (depth) accumulated so far.

        Returns:
            float: Final path length including leaf correction.
        """

        if node.is_leaf:
            return current_length + c_factor(node.size)

        if X[node.feature_index] < node.split_value:
            return self._traverse(node.left, X, current_length + 1)
        else:
            return self._traverse(node.right, X, current_length + 1)