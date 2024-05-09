import torch.nn as nn


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 10)

    def forward(self, X):
        return self.fc1(X)
