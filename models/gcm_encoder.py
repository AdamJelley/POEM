import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder


class GCMEncoder(Encoder):
    def __init__(
        self, input_shape, hid_dim, z_dim, use_location, use_direction, use_coordinates
    ):
        super().__init__(x_dim=input_shape, hid_dim=hid_dim, z_dim=z_dim)
        self.use_location = use_location
        self.use_direction = use_direction
        self.use_coordinates = use_coordinates
        self.encoder_embedding_size = max(1, (input_shape[-1] // (2**5)) ** 2) * z_dim
        if self.use_location:
            self.encoder_embedding_size += 2
        if self.use_direction:
            self.encoder_embedding_size += 4
        if self.use_coordinates:
            self.encoder_embedding_size += 4
        self.mean_head = nn.Sequential(
            nn.Linear(self.encoder_embedding_size, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ELU(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ELU(),
            nn.Linear(z_dim, z_dim),
        )
        self.precision_head = nn.Sequential(
            nn.Linear(self.encoder_embedding_size, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ELU(),
            nn.Linear(z_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ELU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, observations, locations, directions, coordinates):
        out = Encoder.forward(self, observations)
        if self.use_location:
            out = T.cat([out, locations], dim=1)
        if self.use_direction:
            out = T.cat([out, directions], dim=1)
        if self.use_coordinates:
            out = T.cat([out, coordinates], dim=1)
        mean_out = self.mean_head(out)
        precision_out = T.exp(self.precision_head(out))
        return mean_out, precision_out

    def save_checkpoint(self, checkpoint_dir, model_name="gcm_encoder_chkpt.pt"):
        print("Saving state encoder network checkpoint...")
        checkpoint_file = os.path.join(checkpoint_dir, model_name)
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_dir, model_name="gcm_encoder_chkpt.pt"):
        print("Loading state encoder network checkpoint...")
        checkpoint_file = os.path.join(checkpoint_dir, model_name)
        self.load_state_dict(T.load(checkpoint_file))
