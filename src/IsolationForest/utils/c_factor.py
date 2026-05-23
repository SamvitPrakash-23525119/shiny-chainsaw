import numpy as np

def c_factor(n: int) -> float:
    """Compute the normalization factor used by isolation trees.

    The returned value is the expected path length of unsuccessful search in
    a binary search tree and is used to normalize anomaly scores.

    Args:
        n (int): Number of samples in the leaf or subsample.

    Returns:
        float: Normalization factor for the given sample size. Returns `0`
            when `n <= 1`.
    """
    if n <= 1:
        return 0
    return 2 * (np.log(n - 1) + 0.5772156649) - (2 * (n - 1) / n)