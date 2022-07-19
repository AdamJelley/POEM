import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_contrastive_modelling.proto_encoder import ProtoEncoder
from generative_contrastive_modelling.projection_network import ProjectionNetwork


class PrototypicalNetwork(nn.Module):
    def __init__(
        self,
        input_shape,
        hid_dim,
        z_dim,
        use_location=False,
        use_direction=False,
        project_embedding=False,
    ):
        super().__init__()
        self.use_location = use_location
        self.use_direction = use_direction
        self.project_embedding = project_embedding
        self.encoder = ProtoEncoder(
            input_shape, hid_dim, z_dim, use_location, use_direction
        )
        if self.project_embedding:
            self.projection_network = ProjectionNetwork(z_dim, z_dim, z_dim)

    def get_num_samples(self, targets, num_classes, dtype=None):
        batch_size = targets.size(0)
        with torch.no_grad():
            # log.info(f"Batch size is {batch_size}")
            ones = torch.ones_like(targets, dtype=dtype)
            # log.info(f"Ones tensor is {ones.shape}")
            num_samples = ones.new_zeros((batch_size, num_classes))
            # log.info(f"Num samples tensor is {num_samples.shape}")
            num_samples.scatter_add_(1, targets, ones)
        return num_samples

    def get_prototypes(self, embeddings, targets):
        """Compute the prototypes (the mean vector of the embedded training/support
        points belonging to its class) for each classes in the task.

        Parameters
        ----------
        embeddings : `torch.FloatTensor` instance
            A tensor containing the embeddings of the support points. This tensor
            has shape `(batch_size, num_examples, embedding_size)`.

        targets : `torch.LongTensor` instance
            A tensor containing the targets of the support points. This tensor has
            shape `(batch_size, num_examples)`.

        Returns
        -------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        """
        batch_size, num_examples, embedding_size = embeddings.shape
        num_classes = len(torch.unique(targets))

        num_samples = self.get_num_samples(targets, num_classes, dtype=embeddings.dtype)
        num_samples.unsqueeze_(-1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))

        prototypes = embeddings.new_zeros((batch_size, num_classes, embedding_size))
        indices = targets.unsqueeze(-1).expand_as(embeddings)
        prototypes.scatter_add_(1, indices, embeddings).div_(num_samples)

        return prototypes

    def euclidian_distances(self, prototypes, embeddings):
        distances = torch.sum(
            (prototypes.unsqueeze(2) - embeddings.unsqueeze(1)) ** 2, dim=-1
        )
        return distances

    def compute_environment_representations(self, support_trajectories):
        support_embeddings, _ = self.encoder.forward(
            support_trajectories["observations"],
            support_trajectories["locations"] if self.use_location else None,
            support_trajectories["directions"] if self.use_direction else None,
        )

        support_embeddings = support_embeddings.unsqueeze(0)
        support_targets = support_trajectories["targets"].unsqueeze(0)

        env_proto_embeddings = self.get_prototypes(support_embeddings, support_targets)
        env_proto_precisions = torch.ones_like(env_proto_embeddings)

        return env_proto_embeddings, env_proto_precisions

    def compute_loss(self, support_trajectories, query_views):

        support_embeddings, _ = self.encoder.forward(
            support_trajectories["observations"],
            support_trajectories["locations"] if self.use_location else None,
            support_trajectories["directions"] if self.use_direction else None,
        )

        query_embeddings, _ = self.encoder.forward(
            query_views["observations"],
            query_views["locations"] if self.use_location else None,
            query_views["directions"] if self.use_location else None,
        )

        support_embeddings = support_embeddings.unsqueeze(0)
        query_embeddings = query_embeddings.unsqueeze(0)

        support_targets = support_trajectories["targets"].unsqueeze(0)
        query_targets = query_views["targets"].unsqueeze(0)

        env_proto_embeddings = self.get_prototypes(support_embeddings, support_targets)

        if self.project_embedding:
            env_projections = self.projection_network.forward(env_proto_embeddings)
        else:
            env_projections = env_proto_embeddings

        euclidian_distances = self.euclidian_distances(
            env_projections, query_embeddings
        )

        _, predictions = euclidian_distances.min(1)

        loss = F.cross_entropy(-euclidian_distances, query_targets)
        accuracy = torch.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy

        return output
