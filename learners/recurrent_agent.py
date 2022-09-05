import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from models.proto_encoder import ProtoEncoder
from models.recurrent_encoder import RecurrentEncoder
from models.projection_network import ProjectionNetwork


class RecurrentAgent(nn.Module):
    def __init__(
        self,
        input_shape,
        hid_dim,
        z_dim,
        use_location,
        use_direction,
        use_coordinates,
        project_embedding,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.use_location = use_location
        self.use_direction = use_direction
        self.use_coordinates = use_coordinates
        self.project_embedding = project_embedding
        self.encoder = ProtoEncoder(
            input_shape, hid_dim, z_dim, use_location, use_direction, use_coordinates
        )
        self.recurrent_encoder = RecurrentEncoder(z_dim, z_dim)
        if self.project_embedding:
            self.projection_network = ProjectionNetwork(z_dim, z_dim, z_dim)

    def euclidian_distances(self, prototypes, embeddings):
        distances = T.sum(
            (prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
        )
        return distances

    def compute_environment_representations(self, support_trajectories):
        support_embeddings, _ = self.encoder.forward(
            support_trajectories["observations"],
            support_trajectories["locations"] if self.use_location else None,
            support_trajectories["directions"] if self.use_direction else None,
            support_trajectories["coordinates"] if self.use_coordinates else None,
        )

        support_embeddings = support_embeddings.unsqueeze(0)
        support_targets = support_trajectories["targets"].unsqueeze(0)

        num_environments = len(support_targets.unique())
        env_proto_embeddings = T.zeros(1, num_environments, self.z_dim)

        for target in support_targets.unique():
            trajectory_embedding = support_embeddings[support_targets == target]
            _, env_embedding = self.recurrent_encoder(trajectory_embedding)
            env_proto_embeddings[0, target, :] = env_embedding.unsqueeze(0)
        env_proto_precisions = T.ones_like(env_proto_embeddings)

        return env_proto_embeddings, env_proto_precisions

    def compute_loss(self, support_trajectories, query_views):

        support_embeddings, _ = self.encoder.forward(
            support_trajectories["observations"],
            support_trajectories["locations"] if self.use_location else None,
            support_trajectories["directions"] if self.use_direction else None,
            support_trajectories["coordinates"] if self.use_coordinates else None,
        )

        query_embeddings, _ = self.encoder.forward(
            query_views["observations"],
            query_views["locations"] if self.use_location else None,
            query_views["directions"] if self.use_location else None,
            query_views["coordinates"] if self.use_coordinates else None,
        )

        support_embeddings = support_embeddings.unsqueeze(0)
        query_embeddings = query_embeddings.unsqueeze(0)

        support_targets = support_trajectories["targets"].unsqueeze(0)
        query_targets = query_views["targets"].unsqueeze(0)

        num_environments = len(support_targets.unique())
        env_proto_embeddings = T.zeros(1, num_environments, self.z_dim)

        for target in support_targets.unique():
            trajectory_embedding = support_embeddings[support_targets == target]
            _, env_embedding = self.recurrent_encoder(trajectory_embedding)
            env_proto_embeddings[0, target, :] = env_embedding.unsqueeze(0)

        if self.project_embedding:
            env_projections = self.projection_network.forward(env_proto_embeddings)
        else:
            env_projections = env_proto_embeddings

        euclidian_distances = self.euclidian_distances(
            env_projections, query_embeddings
        )

        _, predictions = euclidian_distances.min(1)

        loss = F.cross_entropy(-euclidian_distances, query_targets)

        accuracy = T.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy

        return output
