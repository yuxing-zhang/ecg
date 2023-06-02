# Exploratory data analysis

import numpy as np
import matplotlib.pyplot as plt

train_x, train_y = np.load('data/train_x.npy'), np.load('data/train_y.npy')
valid_x, valid_y = np.load('data/valid_x.npy'), np.load('data/valid_y.npy')
test_x, test_y = np.load('data/test_x.npy'), np.load('data/test_y.npy')

# Split the data according to their labels
trains, valids, tests = [], [], []
for l in range(5):
    trains.append(train_x[train_y == l])
    valids.append(valid_x[valid_y == l])
    tests.append(test_x[test_y == l])

# Get the number of examples in each class, in training and test sets
print('Training set:\nN\tS\tV\tF\tQ\tTotal\n' + '\t'.join(str(d.shape[0]) for
      d in trains) + f'\t{train_x.shape[0]}')
print('Validation set:\nN\tS\tV\tF\tQ\tTotal\n' + '\t'.join(str(d.shape[0]) for
      d in valids) + f'\t{valid_x.shape[0]}')
print('Test set:\nN\tS\tV\tF\tQ\tTotal\n' + '\t'.join(str(d.shape[0]) for
      d in tests) + f'\t{test_x.shape[0]}')

# Plot an example from each class
fig, axs = plt.subplots(1, 5, sharey=True)
for i, d, n in zip(range(5), trains, 'NSVFQ'):
    axs[i].plot(d[0])
    axs[i].set_title(n)
    axs[i].set_xlabel('Time (ms)')

axs[0].set_ylabel('Amplitude')
fig.suptitle('Different Types of Heartbeat')
plt.show()
