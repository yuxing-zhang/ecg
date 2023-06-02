import numpy as np
import torch
from torch import nn, optim 
from torch.utils.data import TensorDataset, DataLoader
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--lr', type=float, help='Learning rate')

epoch = 300
lr = parser.parse_args().lr
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                      # Inital shape (1, 160)
model = nn.Sequential(nn.Conv1d(1, 4, 4, 2, 1),
                      # (4, 80)
                      nn.BatchNorm1d(4),
                      nn.ReLU(),
                      nn.Conv1d(4, 8, 4, 2, 1),
                      # (8, 40)
                      nn.BatchNorm1d(8),
                      nn.ReLU(),
#                      nn.Dropout(),
                      nn.Conv1d(8, 16, 4, 2, 1),
                      # (16, 20)
                      nn.BatchNorm1d(16),
                      nn.ReLU(),
#                      nn.Dropout(),
                      nn.Conv1d(16, 32, 4, 2, 1),
                      # (32, 10)
                      nn.BatchNorm1d(32),
                      nn.ReLU(),
#                      nn.Dropout(),
                      nn.Conv1d(32, 64, 4, 2, 1),
                      # (64, 5)
                      nn.BatchNorm1d(64),
                      nn.ReLU(),
                      nn.Conv1d(64, 5, 5)
                      # (5, 1)
                      )
model.to(device)

# Transforming training data into Pytorch Dataset
train_x = torch.tensor(np.load('data/train_x_balanced.npy'),
                       dtype=torch.float32,
                       device=device).unsqueeze(1)
train_y = torch.tensor(np.load('data/train_y_balanced.npy'), device=device)
train_ds = TensorDataset(train_x, train_y)

# Creating a dataloader
loader = DataLoader(train_ds, batch_size=64, shuffle=True)

# Creating validation dataset
valid_x = torch.tensor(np.load('data/valid_x.npy'), dtype=torch.float32,
                       device=device).unsqueeze(1)
valid_y = torch.tensor(np.load('data/valid_y.npy'), device=device)

# Using gradient descent with momentum
op = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Using cross entropy loss
loss = nn.CrossEntropyLoss()

# Implementing early stop
class ES():
    """ Early stop.

    Stop training when the validation loss no longer decreases.

    Parameters:
        tol: number of occurence allowed for loss to not decrease.
        delta: loss is considerred to be not decreasing if the difference
               surpasses 1delta`
    """
    def __init__(self, tol=10, delta=.1):
        self.tol = tol
        self.delta = delta
        self.count = 0
        self.min_loss = float('inf')
    def check(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.count = 0
            return False
        elif loss > self.min_loss + self.delta:
            self.count += 1
            if self.count > self.tol:
                return True
        # Do nothing when (loss - min_loss) < delta
        else:
            return False

# Start the Training process
model.train()
es = ES()

# Containers for the training and validation losses
tls, vls = [], []

for e in range(epoch):
    for i, (x, y) in enumerate(loader):
        y_ = model(x).view(-1, 5)
        l = loss(y_, y)
        op.zero_grad()
        l.backward()
        op.step()

        # Validation loss
        print(f'epoch: {e} / {epoch} batch: {i} / {len(loader)} loss: {l}')
     
    # Training loss over the entire dataset
    tl = loss(model(train_x).view(-1, 5), train_y)
    tls.append(tl.item())

    # Early stoping
    vl = loss(model(valid_x).view(-1, 5), valid_y)
    vls.append(vl.item())
#    if es.check(vl):
#        print('Early stops now!')
#        break

# Serialize trained model
torch.save(model.state_dict(), f'model_{lr}.pt')

# Save the losses
np.save(f'tl_{lr}', np.array(tls))
np.save(f'vl_{lr}', np.array(vls))
