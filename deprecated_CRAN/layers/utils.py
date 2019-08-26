import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, output_size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(num_layers - 2)])
        self.linears.insert(0, nn.Linear(input_size, hidden_size))
        self.linears.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for l in self.linears:
            x = F.relu(l(x))
        return x
