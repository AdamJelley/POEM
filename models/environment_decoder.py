import torch as T
import torch.nn as nn
from math import prod


class EnvironmentDecoder(nn.Module):
    def __init__(self, embedding_dim, hid_dim, environment_shape):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hid_dim = hid_dim
        self.environment_shape = environment_shape
        self.output_dim = prod(environment_shape)

        self.projection_head = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim * 4),
        )

        self.decoder_backbone = nn.Sequential(
            self.transpose_conv_block(self.embedding_dim, self.hid_dim),
            self.transpose_conv_block(self.hid_dim, self.hid_dim),
            self.transpose_conv_block(self.hid_dim, self.hid_dim),
            self.transpose_conv_block(self.hid_dim, self.hid_dim),
        )

        self.final_projection = nn.Sequential(
            nn.ConvTranspose2d(
                self.hid_dim, self.environment_shape[0], 2, padding=1, stride=2
            ),
            nn.Flatten(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.Sigmoid(),
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ELU(),
            nn.Linear(self.embedding_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ELU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.Sigmoid(),
        )

    def transpose_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, padding=1, stride=2),
            nn.InstanceNorm2d(out_channels),
            nn.ELU(),
        )

    def forward(self, means, precisions=None, sample=False):
        if sample:
            stds = T.sqrt(T.reciprocal(precisions))
            out = T.normal(means, stds)
        else:
            out = means
        # out = self.projection_head(out)
        # out = out.reshape(-1, self.embedding_dim, 2, 2)
        # out = self.decoder_backbone.forward(out)
        # out = self.final_projection(out).reshape(-1, *self.environment_shape)
        out = self.fc_decoder(out)
        out = out.reshape(-1, *self.environment_shape)
        return out
