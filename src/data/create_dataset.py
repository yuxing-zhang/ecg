# Read datasets from CSV file using Pandas and save them as numpy arrays.

import numpy as np
import pandas as pd

train = pd.read_csv('data/mitbih_train.csv', header=None)
test = pd.read_csv('data/mitbih_test.csv', header=None)

train = train.to_numpy()
test = test.to_numpy()

# Split the data into ecg signals and labels and convert label to integers
train_x, train_y = train[:, :160], train[:, -1].astype(int)
test_x, test_y = test[:, :160], test[:, -1].astype(int)

# Creating validation set with 10% training data from each class
train_xs, train_ys, valid_xs, valid_ys = [], [], [], []
for l in range(5):
    temp = train_x[train_y == l]
    v = len(temp) // 10
    t = len(temp) - v
    train_xs.append(temp[:t])
    train_ys.append(np.array([l] * t))
    valid_xs.append(temp[t:])
    valid_ys.append(np.array([l] * v))

# Combine examples from each class into one dataset
train_x = np.vstack(train_xs)
train_y = np.hstack(train_ys)
valid_x = np.vstack(valid_xs)
valid_y = np.hstack(valid_ys)

# Serialize the datasets
for f, d in zip(['train_x', 'train_y', 'valid_x', 'valid_y',
                'test_x', 'test_y'], [train_x, train_y,
                valid_x, valid_y, test_x, test_y]):
    np.save(f'data/{f}', d)
