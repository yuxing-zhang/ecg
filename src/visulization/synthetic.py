# Visualize original and synthetic examples in class S

import numpy as np
import matplotlib.pyplot as plt

train_x = np.load('data/train_x.npy')
train_y = np.load('data/train_y.npy')

train_s = train_x[train_y==1]

syn_s = np.load('data/syn_s.npy')

fig, axs = plt.subplots(3, 2, sharey=True, sharex=True)
for i in range(4, 7):
    axs[i-4, 0].plot(train_s[i])
    axs[i-4, 0].set_ylabel('Amplitude')
    axs[i-4, 1].plot(syn_s[i])

for i in range(2):
    axs[2, i].set_xlabel('Time (ms)')

axs[0, 0].set_title('Original')
axs[0, 1].set_title('Synthetic')

plt.show()
