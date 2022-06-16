import torch as T
import torch.nn as nn
import torch.nn.functional as F

from generative_contrastive_modelling.encoder import Encoder


class GCMEncoder(Encoder):
    def __init__(self, input_shape, hid_dim, z_dim):
        super().__init__(x_dim=input_shape, hid_dim=hid_dim, z_dim=z_dim)
        self.encoder_embedding_size = max(1, (input_shape[-1] // (2**4)) ** 2) * z_dim
        self.mean_head = nn.Sequential(
            nn.Linear(self.encoder_embedding_size, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ELU(),
            nn.Linear(z_dim, z_dim),
        )
        self.precision_head = nn.Sequential(
            nn.Linear(self.encoder_embedding_size, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ELU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, input):
        out = Encoder.forward(self, input)
        mean_out = self.mean_head(out)
        precision_out = T.exp(self.precision_head(out))
        return mean_out, precision_out
