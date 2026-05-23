class IsolationTreeNode:
    """Node for an isolation tree.

    Represents an internal or leaf node of an `IsolationTree`. Internal
    nodes store the index of the feature to split on and the split value;
    leaf nodes carry a `size` indicating how many samples reached the node.
    """
    def __init__(self, feature_index=None, split_value=None,
                 left=None, right=None, size=0, is_leaf=False):
        """Create a new IsolationTreeNode.

        Args:
            feature_index (int | None): Index of the feature used for the
                split at this node. `None` for leaf nodes.
            split_value (float | None): Threshold value to split the feature
                on. `None` for leaf nodes.
            left (IsolationTreeNode | None): Left child node (values <
                `split_value`).
            right (IsolationTreeNode | None): Right child node (values >=
                `split_value`).
            size (int): Number of samples present at this node. For leaf
                nodes this is the leaf size; for internal nodes it may be
                unused or represent the subtree size.
            is_leaf (bool): Whether this node is a leaf.
        """

        self.feature_index = feature_index
        self.split_value = split_value
        self.left = left
        self.right = right
        self.size = size
        self.is_leaf = is_leaf