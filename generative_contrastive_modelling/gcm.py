import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative_contrastive_modelling.gcm_encoder import GCMEncoder


class GenerativeContrastiveModelling(nn.Module):
    def __init__(
        self, input_shape, hid_dim, z_dim, use_location, use_direction, use_coordinates
    ):
        super().__init__()
        self.use_location = use_location
        self.use_direction = use_direction
        self.use_coordinates = use_coordinates
        self.encoder = GCMEncoder(
            input_shape, hid_dim, z_dim, use_location, use_direction, use_coordinates
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

    def inner_gaussian_product(self, means, precisions, targets):
        """Compute the product of n Gaussians for each trajectory (where n can vary by trajectory) from their means and precisions.
        Parameters
        ----------
        means : `torch.FloatTensor` instance
            A tensor containing the means of the Gaussian embeddings. This tensor has shape
            `(batch_size, num_examples, embedding_size)`.
        precisions : `torch.FloatTensor` instance
            A tensor containing the precisions of the Gaussian embeddings. This tensor has shape
            `(batch_size, num_examples, embedding_size)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the query points. This tensor has
            shape `(meta_batch_size, num_examples)`.
        Returns
        -------
        product_mean : `torch.FloatTensor` instance
            A tensor containing the mean of the resulting product Gaussian. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        product_precision : `torch.FloatTensor` instance
            A tensor containing the precision of resulting product Gaussians. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        log_product_normalisation: `torch.FloatTensor` instance
            A tensor containing the log of the normalisation of resulting product Gaussians. This tensor has shape
            `(batch_size, num_classes)`.
        """
        assert means.shape == precisions.shape
        batch_size, num_examples, embedding_size = means.shape
        num_classes = len(torch.unique(targets))

        num_samples = self.get_num_samples(targets, num_classes, dtype=means.dtype)
        num_samples.unsqueeze_(-1)
        num_samples = torch.max(
            num_samples, torch.ones_like(num_samples)
        )  # Backup for testing only, always >= 1-shot in practice

        indices = targets.unsqueeze(-1).expand_as(means)

        # NOTE: If this approach doesn't work well, try first normalising precisions by number of samples with:
        product_precision = precisions.new_zeros(
            (batch_size, num_classes, embedding_size)
        )
        product_precision.scatter_add_(1, indices, precisions)

        product_mean = means.new_zeros((batch_size, num_classes, embedding_size))
        product_mean = torch.reciprocal(product_precision) * product_mean.scatter_add_(
            1, indices, precisions * means
        )

        product_normalisation_exponent = means.new_zeros(
            (batch_size, num_classes, embedding_size)
        )
        product_normalisation_exponent = 0.5 * (
            product_precision * torch.square(product_mean)
            - product_normalisation_exponent.scatter_add_(
                1, indices, precisions * torch.square(means)
            )
        )

        log_product_normalisation = means.new_zeros(
            (batch_size, num_classes, embedding_size)
        )
        log_product_normalisation = (
            (0.5 * (1 - num_samples))
            * torch.log(torch.ones_like(num_samples) * (2 * math.pi))
            + 0.5
            * (
                log_product_normalisation.scatter_add_(
                    1, indices, torch.log(precisions)
                )
                - torch.log(product_precision)
            )
            + product_normalisation_exponent
        )

        log_product_normalisation = log_product_normalisation.sum(dim=-1)

        return (
            product_mean,
            product_precision,
            log_product_normalisation,
        )

    def outer_gaussian_product(self, x_mean, x_precision, y_mean, y_precision):
        """
        Computes all Gaussian product pairs between Gaussian x and y.
        Args:
            x_mean : `torch.FloatTensor` instance
                A tensor containing the means of the query Gaussians. This tensor has shape
                `(batch_size, num_query_examples, embedding_size)`.
            x_precision : `torch.FloatTensor` instance
                A tensor containing the precisions of the query Gaussians. This tensor has shape
                `(batch_size, num_query_examples, embedding_size)`.
            y_mean : `torch.FloatTensor` instance
                A tensor containing the means of the proto Gaussians. This tensor has shape
                `(batch_size, num_classes, embedding_size)`.
            y_precision : `torch.FloatTensor` instance
                A tensor containing the precisions of the proto Gaussians. This tensor has shape
                `(batch_size, num_classes, embedding_size)`.
        Returns:
        product_mean : `torch.FloatTensor` instance
            A tensor containing the mean of the resulting product Gaussian. This tensor has shape
            `(batch_size, num_classes, num_query_examples, embedding_size)`.
        product_precision : `torch.FloatTensor` instance
            A tensor containing the precision of resulting product Gaussians. This tensor has shape
            `(batch_size, num_classes, num_query_examples, embedding_size)`.
        log_product_normalisation: `torch.FloatTensor` instance
            A tensor containing the log of the normalisation of resulting product Gaussians. This tensor has shape
            `(batch_size, num_classes, num_query_examples)`.
        """

        assert x_mean.shape == x_precision.shape
        assert y_mean.shape == y_precision.shape
        (batch_size, num_query_examples, embedding_size) = x_mean.shape
        num_classes = y_mean.size(1)
        assert x_mean.size(0) == y_mean.size(0)
        assert x_mean.size(2) == y_mean.size(2)

        x_mean = x_mean.unsqueeze(1).expand(
            batch_size, num_classes, num_query_examples, embedding_size
        )
        x_precision = x_precision.unsqueeze(1).expand(
            batch_size, num_classes, num_query_examples, embedding_size
        )
        y_mean = y_mean.unsqueeze(2).expand(
            batch_size, num_classes, num_query_examples, embedding_size
        )
        y_precision = y_precision.unsqueeze(2).expand(
            batch_size, num_classes, num_query_examples, embedding_size
        )

        product_precision = x_precision + y_precision
        product_mean = torch.reciprocal(product_precision) * (
            x_precision * x_mean + y_precision * y_mean
        )
        product_normalisation_exponent = 0.5 * (
            product_precision * torch.square(product_mean)
            - x_precision * torch.square(x_mean)
            - y_precision * torch.square(y_mean)
        )
        log_product_normalisation = (
            -0.5
            * torch.log(torch.ones_like(product_normalisation_exponent) * (2 * math.pi))
            + 0.5
            * (
                torch.log(x_precision)
                + torch.log(y_precision)
                - torch.log(product_precision)
            )
            + product_normalisation_exponent
        ).sum(dim=-1)
        return product_mean, product_precision, log_product_normalisation

    def compute_environment_representations(self, support_trajectories):

        support_means, support_precisions = self.encoder.forward(
            support_trajectories["observations"],
            support_trajectories["locations"] if self.use_location else None,
            support_trajectories["directions"] if self.use_direction else None,
            support_trajectories["coordinates"] if self.use_coordinates else None,
        )

        support_means = support_means.unsqueeze(0)
        support_precisions = support_precisions.unsqueeze(0)
        support_targets = support_trajectories["targets"].unsqueeze(0)

        (
            env_means,
            env_precisions,
            log_env_normalisation,
        ) = self.inner_gaussian_product(
            support_means, support_precisions, support_targets
        )

        return env_means, env_precisions

    def compute_loss(self, support_trajectories, query_views):

        support_means, support_precisions = self.encoder.forward(
            support_trajectories["observations"],
            support_trajectories["locations"] if self.use_location else None,
            support_trajectories["directions"] if self.use_direction else None,
            support_trajectories["coordinates"] if self.use_coordinates else None,
        )

        query_means, query_precisions = self.encoder.forward(
            query_views["observations"],
            query_views["locations"] if self.use_location else None,
            query_views["directions"] if self.use_direction else None,
            query_views["coordinates"] if self.use_coordinates else None,
        )

        support_means = support_means.unsqueeze(0)
        support_precisions = support_precisions.unsqueeze(0)
        support_targets = support_trajectories["targets"].unsqueeze(0)
        query_means = query_means.unsqueeze(0)
        query_precisions = query_precisions.unsqueeze(0)
        query_targets = query_views["targets"].unsqueeze(0)

        (
            env_proto_means,
            env_proto_precisions,
            log_env_proto_normalisation,
        ) = self.inner_gaussian_product(
            support_means, support_precisions, support_targets
        )

        (
            env_obs_product_mean,
            env_obs_product_precision,
            log_env_obs_normalisation,
        ) = self.outer_gaussian_product(
            query_means, query_precisions, env_proto_means, env_proto_precisions
        )

        _, predictions = log_env_obs_normalisation.max(1)

        loss = F.cross_entropy(log_env_obs_normalisation, query_targets)
        accuracy = torch.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy
        output["support_precision_mean"] = support_precisions.mean()
        output["support_precision_var"] = support_precisions.var()
        output["query_precision_mean"] = query_precisions.mean()
        output["query_precision_var"] = query_precisions.var()


        return output

    def compute_loss2(self, support_trajectories, query_views):

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
            ).detach()
        else:
            directions = None
        observation_means, observation_precisions = self.encoder.forward(
            observations, locations, directions
        )

        support_means = observation_means[:num_support_obs].unsqueeze(0)
        query_means = observation_means[num_support_obs:].unsqueeze(0)
        support_precisions = observation_precisions[:num_support_obs].unsqueeze(0)
        query_precisions = observation_precisions[num_support_obs:].unsqueeze(0)

        support_targets = support_trajectories["targets"].unsqueeze(0)
        query_targets = query_views["targets"].unsqueeze(0)

        (
            env_proto_means,
            env_proto_precisions,
            log_env_proto_normalisation,
        ) = self.inner_gaussian_product(
            support_means, support_precisions, support_targets
        )

        (
            env_obs_product_mean,
            env_obs_product_precision,
            log_env_obs_normalisation,
        ) = self.outer_gaussian_product(
            query_means, query_precisions, env_proto_means, env_proto_precisions
        )

        _, predictions = log_env_obs_normalisation.max(1)

        loss = F.cross_entropy(log_env_obs_normalisation, query_targets)
        accuracy = torch.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy

        return output
