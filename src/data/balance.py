# Create a balanced training dataset with SMOTE synthesized data
# Each class will have 3000 examples

import numpy as np

train_x = np.load('data/train_x.npy')
train_y = np.load('data/train_y.npy')

# Loading synthetic data in classes S and F
syn_s = np.load('data/syn_s.npy')
syn_f = np.load('data/syn_f.npy')

trains = [train_x[train_y == i] for i in range(5)]

# Append the synthetic examples to their corresponding classes
trains[1] = np.vstack([trains[1], syn_s])
trains[3] = np.vstack([trains[3], syn_f])

# Sample 3000 examples from each class, without replacement
rng = np.random.default_rng()
trains = [rng.choice(t, size=3000, replace=False) for t in
          trains]

train_x_balanced = np.vstack(trains)
train_y_balanced = np.hstack([np.ones(3000, dtype=int) * i for i in range(5)])

print(train_x_balanced.shape, train_y_balanced.shape)
np.save('data/train_x_balanced', train_x_balanced)
np.save('data/train_y_balanced', train_y_balanced)

"""
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 5)
for i in range(5):
    axs[i].plot(trains[i][0])
plt.show()
"""
