import torch as T
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, x_dim, hid_dim, z_dim):
        """Load encoder network

        Args:
            x_dim (tuple): dimensions of input
            hid_dim (int): dimension of hidden layers in conv blocks
            z_dim (int): dimension of embedding

        Returns:
            Encoder Network
        """

        super().__init__()
        self.x_dim = x_dim
        self.hid_dim = hid_dim
        self.z_dim = z_dim

        self.encoder_backbone = nn.Sequential(
            self.conv_block(x_dim[0], hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, z_dim),
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ELU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        z = self.encoder_backbone.forward(x)
        z = T.flatten(z, start_dim=1)
        return z
