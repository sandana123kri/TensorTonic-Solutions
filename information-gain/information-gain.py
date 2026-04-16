import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    y = np.asarray(y)
    mask = np.asarray(split_mask, dtype=bool)

    # Split
    y_left = y[split_mask]
    y_right = y[~split_mask]

    N = len(y)
    nL = len(y_left)
    nR = len(y_right)

    # Edge case: empty split
    if nL == 0 or nR == 0:
        return 0.0

    # Entropies
    H_parent = _entropy(y)
    H_left = _entropy(y_left)
    H_right = _entropy(y_right)

    # Weighted entropy
    weighted = (nL / N) * H_left + (nR / N) * H_right

    return H_parent - weighted
