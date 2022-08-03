import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_contrastive_modelling.proto_encoder import ProtoEncoder
from generative_contrastive_modelling.projection_network import ProjectionNetwork
from process_trajectories import orientate_observations


class CompleteObservationLearner(nn.Module):
    def __init__(
        self,
        input_shape,
        hid_dim,
        z_dim,
        orient_queries=True,
        use_location=False,
        use_direction=False,
        use_coordinates=False,
    ):
        super().__init__()
        self.orient_queries = orient_queries
        self.use_location = use_location
        self.use_direction = use_direction
        self.use_coordinates = use_coordinates
        self.environment_encoder = ProtoEncoder(
            input_shape, hid_dim, z_dim, False, False, False
        )
        self.query_encoder = ProtoEncoder(
            input_shape, hid_dim, z_dim, use_location, use_direction, use_coordinates
        )
        self.projection_network = ProjectionNetwork(z_dim, z_dim, z_dim)

    def euclidian_distances(self, prototypes, embeddings):
        distances = torch.sum(
            (prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
        )
        return distances

    def compute_loss(self, support_trajectories, query_views):

        num_environments = support_trajectories["targets"].unique().shape[0]
        num_queries = query_views["targets"].shape[0]

        if self.orient_queries:
            query_views = orientate_observations(query_views)

        environment_embeddings, _ = self.environment_encoder.forward(
            support_trajectories["environments"], None, None, None
        )
        query_embeddings, _ = self.query_encoder.forward(
            query_views["observations"],
            query_views["locations"] if self.use_locations else None,
            query_views["directions"] if self.use_directions else None,
            query_views["coordinates"] if self.use_coordinates else None,
        )

        environment_embeddings = (
            environment_embeddings.unsqueeze(0)
            .repeat(num_queries, 1, 1)
            .reshape(
                1, num_queries * num_environments, environment_embeddings.shape[-1]
            )
        )

        query_embeddings = query_embeddings.unsqueeze(0)

        environment_targets = support_trajectories["environment_targets"].unsqueeze(0)
        query_targets = query_views["targets"].unsqueeze(0)

        environment_projections = self.projection_network.forward(
            environment_embeddings
        )

        euclidian_distances = self.euclidian_distances(
            environment_projections, query_embeddings
        )

        euclidian_distances = euclidian_distances.reshape(
            1, num_queries, num_environments, num_queries
        ).diagonal(dim1=1, dim2=3)

        _, predictions = euclidian_distances.min(1)

        loss = F.cross_entropy(-euclidian_distances, query_targets)
        accuracy = torch.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy

        return output

        # environment_embeddings = environment_embeddings.unsqueeze(2).repeat(
        #     1, 1, num_queries, 1
        # )
        # query_embeddings = query_embeddings.unsqueeze(1).repeat(
        #     1, num_environments, 1, 1
        # )

        # environment_query_pairs = T.cat(
        #     (environment_embeddings, query_embeddings), dim=-1
        # )

        # environment_query_pairs = environment_query_pairs.reshape(-1, 256)

        # # environment_projections = self.projection_network.forward(
        # #     environment_embeddings
        # # )

        # # euclidian_distances = self.euclidian_distances(
        # #     environment_projections, query_embeddings
        # # )

        # similarities = self.projection_network.forward(environment_query_pairs)
        # similarities = similarities.reshape(1, num_environments, num_queries)

        # _, predictions = similarities.min(1)

        # loss = F.cross_entropy(similarities, query_targets)
        # accuracy = T.eq(predictions, query_targets).float().mean()
