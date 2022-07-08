import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from generative_contrastive_modelling.recurrent_encoder import RecurrentEncoder
from generative_contrastive_modelling.proto_encoder import ProtoEncoder


class RecurrentAgent(nn.Module):
    def __init__(
        self,
        input_shape,
        hid_dim,
        z_dim,
        use_location,
        use_direction,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.use_location = use_location
        self.use_direction = use_direction
        self.encoder = ProtoEncoder(
            input_shape, hid_dim, z_dim, use_location, use_direction
        )
        self.recurrent_encoder = RecurrentEncoder(z_dim, z_dim)

    def euclidian_distances(self, prototypes, embeddings):
        distances = T.sum(
            (prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
        )
        return distances

    def compute_loss(self, support_trajectories, query_views):

        num_support_obs = support_trajectories["targets"].shape[0]
        num_query_obs = query_views["targets"].shape[0]

        observations = T.cat(
            [support_trajectories["observations"], query_views["observations"]], dim=0
        ).detach()
        if self.use_location:
            locations = T.cat(
                [support_trajectories["locations"], query_views["locations"]], dim=0
            ).detach()
        else:
            locations = None
        if self.use_direction:
            directions = T.cat(
                [support_trajectories["directions"], query_views["directions"]], dim=0
            )
        else:
            directions = None
        observation_embeddings, _ = self.encoder.forward(
            observations, locations, directions
        )

        support_embeddings = observation_embeddings[:num_support_obs].unsqueeze(0)
        query_embeddings = observation_embeddings[num_support_obs:].unsqueeze(0)

        support_targets = support_trajectories["targets"].unsqueeze(0)
        query_targets = query_views["targets"].unsqueeze(0)

        num_environments = len(support_targets.unique())
        env_proto_embeddings = T.zeros(1, num_environments, self.z_dim)

        for target in support_targets.unique():
            trajectory_embedding = support_embeddings[support_targets == target]
            _, env_embedding = self.recurrent_encoder(trajectory_embedding)
            env_proto_embeddings[0, target, :] = env_embedding.unsqueeze(0)
        env_proto_embeddings = self.get_prototypes(support_embeddings, support_targets)

        euclidian_distances = self.euclidian_distances(
            env_proto_embeddings, query_embeddings
        )

        _, predictions = euclidian_distances.min(1)

        loss = F.cross_entropy(-euclidian_distances, query_targets)

        accuracy = T.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy

        return output
