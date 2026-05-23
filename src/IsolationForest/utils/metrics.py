def accuracy(tp, tn, fp, fn):
    """Compute classification accuracy.

    Args:
        tp: True positives count.
        tn: True negatives count.
        fp: False positives count.
        fn: False negatives count.

    Returns:
        float: Accuracy score.
    """
    if(tp + tn + fp + fn == 0):
        return 0.0

    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    """Compute classification precision.

    Args:
        tp: True positives count.
        fp: False positives count.

    Returns:
        float: Precision score.
    """
    if(tp + fp == 0):
        return 0.0
    
    return tp / (tp + fp)


def recall(tp, fn):
    """Compute classification recall.

    Args:
        tp: True positives count.
        fn: False negatives count.

    Returns:
        float: Recall score.
    """
    if(tp + fn == 0):
        return 0.0

    return tp / (tp + fn)


def f1(p, r):
    """Compute the F1 score from precision and recall.

    Args:
        p: Precision value.
        r: Recall value.

    Returns:
        float: F1 score.
    """
    if(p + r == 0):
        return 0.0

    return 2 * (p * r) / (p + r)