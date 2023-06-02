# Dynamic time warping

import numpy as np

def dtw(s, t):
    """ Calculate the distance between two sequences using DTW

    Parameter:
        s, t: two time series

    Return:
        The distance between `s` and `t`
    """
    m, n = len(s), len(t)
    # d[i, j] = distance between s[:i] and t[:j]
    d = np.ones((m + 1, n + 1)) * np.inf
    d[0, 0] = 0

    for i, x in enumerate(s):
        for j, y in enumerate(t):
            d[i + 1, j + 1] = abs(x - y) +\
                              min(d[i, j], d[i, j + 1], d[i + 1, j])

    return d[-1, -1]
