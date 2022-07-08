import torch as T
import torch.nn as nn
import torch.nn.functional as F


class RecurrentEncoder(nn.Module):
    def __init__(self, z_obs_dim, z_env_dim):
        """Load recurrent encoder network

        Args:
            x_dim (tuple): dimensions of input
            hid_dim (int): dimension of hidden layers in conv blocks
            z_dim (int): dimension of embedding

        Returns:
            Recurrent encoder Network
        """

        super().__init__()
        self.z_obs_dim = z_obs_dim
        self.z_env_dim = z_env_dim
        self.recurrent_layer = nn.GRU(
            input_size=self.z_obs_dim, hidden_size=self.z_env_dim, num_layers=1
        )

    def forward(self, env_support_embeddings):
        out, hidden_state = self.recurrent_layer(env_support_embeddings)
        return out, hidden_state
