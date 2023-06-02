import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Rebuild saved model
model = nn.Sequential(nn.Conv1d(1, 4, 4, 2, 1),
                      # (4, 80)
                      nn.BatchNorm1d(4),
                      nn.ReLU(),
                      nn.Conv1d(4, 8, 4, 2, 1),
                      # (8, 40)
                      nn.BatchNorm1d(8),
                      nn.ReLU(),
                      nn.Dropout(),
                      nn.Conv1d(8, 16, 4, 2, 1),
                      # (16, 20)
                      nn.BatchNorm1d(16),
                      nn.ReLU(),
                      nn.Dropout(),
                      nn.Conv1d(16, 32, 4, 2, 1),
                      # (32, 10)
                      nn.BatchNorm1d(32),
                      nn.ReLU(),
                      nn.Dropout(),
                      nn.Conv1d(32, 64, 4, 2, 1),
                      # (64, 5)
                      nn.BatchNorm1d(64),
                      nn.ReLU(),
                      nn.Conv1d(64, 5, 5)
                      # (5, 1)
                      )
model.load_state_dict(torch.load('model.pt'))

# Create testing set
test_x = torch.tensor(np.load('data/test_x.npy'), dtype=torch.float32,
                      device=device).unsqueeze(1)
test_y = torch.tensor(np.load('data/test_y.npy'))

# Make prediction and obtain the confusion matrix
y_ = model(test_x).reshape(-1, 5).cpu()
y_ = y_.argmax(dim=1)
cm = confusion_matrix(test_y, y_)

np.save('cm', cm)

