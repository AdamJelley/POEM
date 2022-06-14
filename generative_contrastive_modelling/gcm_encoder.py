import torch as T
import torch.nn as nn
import torch.nn.functional as F
import math


class GCMEncoder(nn.Module):
    def __init__(self, input_shape, z_dim):
        super().__init__()
        self.input_shape = input_shape
        self.z_dim = z_dim
        self.input_layer = nn.Linear(math.prod(self.input_shape[1:]), self.z_dim)
        self.input_norm = nn.BatchNorm1d(self.z_dim)
        self.hidden_layer = nn.Linear(self.z_dim, self.z_dim)
        self.hidden_norm = nn.BatchNorm1d(self.z_dim)
        self.mean_head = nn.Linear(self.z_dim, self.z_dim)
        self.precision_head = nn.Linear(self.z_dim, self.z_dim)

    def forward(self, input):
        out = T.flatten(input, start_dim=1)
        out = F.elu(self.input_norm(self.input_layer(out)))
        out = F.elu(self.hidden_norm(self.hidden_layer(out)))
        mean_out = self.mean_head(out)
        precision_out = T.exp(self.precision_head(out))
        return mean_out, precision_out
