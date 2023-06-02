import numpy as np
import matplotlib.pyplot as plt

cm = np.load('cm.npy')
ncm = np.load('ncm.npy')
cls = 'NSVFQ'

fig, ax = plt.subplots()
im = ax.imshow(ncm)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(cls)), labels=cls)
ax.set_yticks(np.arange(len(cls)), labels=cls)

# Loop over data dimensions and create text annotations.
for i in range(len(cls)):
    for j in range(len(cls)):
        text = ax.text(j, i, cm[i, j],
                       ha="center", va="center", color="w")

fig.tight_layout()
plt.show()
