from typing import List

import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, shapes: List[int], activation: nn.Module = nn.ReLU) -> None:
        super().__init__()

        self.act = activation()
        self._num_layers = len(shapes) - 1

        for i in range(self._num_layers):
            setattr(self, f'fc_{i}', nn.Linear(shapes[i], shapes[i + 1]))

    def forward(self, x) -> torch.Tensor:
        for i in range(self._num_layers):
            x = getattr(self, f'fc_{i}')(x)
        return x
