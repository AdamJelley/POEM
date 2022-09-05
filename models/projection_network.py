import torch as T
import torch.nn as nn
import torch.nn.functional as F


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.projection_network = nn.Sequential(
            self.linear_block(self.input_dim, self.hid_dim),
            self.linear_block(self.hid_dim, self.hid_dim),
            self.linear_block(self.hid_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.output_dim),
        )

    def linear_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.InstanceNorm1d(out_features),
            nn.ELU(),
        )

    def forward(self, env_embedding):
        out = self.projection_network.forward(env_embedding)
        return out
