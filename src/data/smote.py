# SMOTE implementation and oversmpling of minority classes

import numpy as np
from random import random
from dtw import dtw

def smote(ds, k, l):
    """ Synthesize new examples using SMOTE.

        Parameters:
            ds: a dataset
            k: number of synthetic examples to be created for each member of `ds`
            l: the metric to be used in nearest neighbour search

        Return:
            A synthetic dataset
    """
    n = ds.shape[0]
    # Container for synthetic examples
    syn = np.empty((n * k, ds.shape[1])) 
    # Pairwise distance
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            dist[i, j] = l(ds[i], ds[j])
    # A metric is symmetric
    dist += dist.T
    # Find the nearest neighbour by argsorting the distance matrix
    ind = dist.argsort()

    for i, d in enumerate(ds):
        # The closest element to `d` is guaranteed to be `d` itself,
        # since dtw(d, d) = 0
        nn = ind[i, 1:k+1]
        for j, d_ in enumerate(nn):
            # Create a synthetic example by interpolating and add it to
            # the synthtic dataset
            syn[i * k + j] = d + (ds[d_] - d) * random()

    return syn

train_x, train_y = np.load('data/train_x.npy'), np.load('data/train_y.npy')

# Split the data according to their labels
trains = []
for l in range(5):
    trains.append(train_x[train_y == l])

# Synthesize data for class S and F since they are the minority classes
# To reach the 3000 threshold, we set `k` = 1 for class S and 5 for class F
syn_s = smote(trains[1], 1, dtw)
syn_f = smote(trains[3], 5, dtw)

np.save('data/syn_s', syn_s)
np.save('data/syn_f', syn_f)
