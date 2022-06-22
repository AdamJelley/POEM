import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_contrastive_modelling.proto_encoder import ProtoEncoder


class PrototypicalNetwork(nn.Module):
    def __init__(
        self, input_shape, hid_dim, z_dim, use_location=False, use_direction=False
    ):
        super().__init__()
        self.use_location = use_location
        self.use_direction = use_direction
        self.encoder = ProtoEncoder(
            input_shape, hid_dim, z_dim, use_location, use_direction
        )

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

    def compute_loss(self, support_trajectories, query_views):

        num_support_obs = support_trajectories["targets"].shape[0]
        num_query_obs = query_views["targets"].shape[0]

        observations = torch.cat(
            [support_trajectories["observations"], query_views["observations"]], dim=0
        ).detach()
        if self.use_location:
            locations = torch.cat(
                [support_trajectories["locations"], query_views["locations"]], dim=0
            ).detach()
        else:
            locations = None
        if self.use_direction:
            directions = torch.cat(
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

        env_proto_embeddings = self.get_prototypes(support_embeddings, support_targets)

        euclidian_distances = self.euclidian_distances(
            env_proto_embeddings, query_embeddings
        )

        _, predictions = euclidian_distances.min(1)

        loss = F.cross_entropy(-euclidian_distances, query_targets)
        accuracy = torch.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy

        return output
