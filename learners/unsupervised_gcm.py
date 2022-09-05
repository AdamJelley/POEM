import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gcm_encoder import GCMEncoder


class UnsupervisedGenerativeContrastiveModelling(nn.Module):
    def __init__(
        self,
        input_shape,
        hid_dim,
        z_dim,
        prior_precision,
        use_location,
        use_direction,
        use_coordinates,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.prior_precision = prior_precision
        self.use_location = use_location
        self.use_direction = use_direction
        self.use_coordinates = use_coordinates
        self.encoder = GCMEncoder(
            input_shape, hid_dim, z_dim, use_location, use_direction, use_coordinates
        )

    def get_num_samples(self, targets, dtype=None):
        batch_size = targets.size(0)
        num_classes = len(torch.unique(targets))
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

        num_samples = self.get_num_samples(targets, dtype=means.dtype)
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

    def calculate_Gaussian_prior_product(self, prior_shape, prior_powers):
        """Calculate product of prior_power Gaussian priors with mean 0 and precision prior_precision.

        Args:
            prior_shape (tuple): Shape of Gaussian prior.
            prior_powers (tensor): Powers to raise denominator unit Gaussian (usually V or V-1 for V = num_samples per class)

        Returns:
            prior_product_mean,
            prior_product_precision,
            log_prior_product_normalisation,
        """

        prior_product_mean = 0
        prior_product_precision = (
            torch.maximum(prior_powers, torch.ones_like(prior_powers))
            * self.prior_precision
        )
        log_prior_product_normalisation = 0.5 * (1 - prior_powers) * math.log(
            2 * math.pi
        ) + 0.5 * (
            prior_powers * math.log(self.prior_precision)
            - torch.log(prior_product_precision)
        )

        prior_product_mean = torch.zeros(prior_shape).to(prior_powers.device)
        prior_product_precision = prior_product_precision.unsqueeze(-1).expand(
            prior_shape
        )
        log_prior_product_normalisation = (
            log_prior_product_normalisation.unsqueeze(-1)
            .expand(prior_shape)
            .sum(dim=-1)
        )
        return (
            prior_product_mean,
            prior_product_precision,
            log_prior_product_normalisation,
        )

    def normalise_by_Gaussian_prior(
        self, means, precisions, prior_product_mean, prior_product_precision
    ):
        """Divide Gaussian with means and precisions by Gaussian prior product

        Args:
            means (tensor): Numerator Gaussian means with shape (batch_size, num_classes) or (batch_size, num_classes, num_queries)
            precisions (tensor): Numerator Gaussian precisions with shape (batch_size, num_classes) or (batch_size, num_classes, num_queries)
            prior_product_mean (tensor): Denominator Gaussian means with shape (batch_size, num_classes) or (batch_size, num_classes, num_queries)
            prior_product_precision (tensor): Denominator Gaussian means with shape (batch_size, num_classes) or (batch_size, num_classes, num_queries)
        """

        quotient_precisions = precisions - prior_product_precision
        quotient_means = torch.reciprocal(quotient_precisions) * (
            precisions * means - prior_product_precision * prior_product_mean
        )
        quotient_exponent = 0.5 * (
            quotient_precisions * torch.square(quotient_means)
            - precisions * torch.square(means)
            + prior_product_precision * torch.square(prior_product_mean)
        )
        log_quotient_normalisation = (
            0.5 * (precisions - quotient_precisions) + quotient_exponent
        ).sum(dim=-1)
        return quotient_means, quotient_precisions, log_quotient_normalisation

    def calculate_posterior_q(self, support_means, support_precisions, support_targets):

        num_samples = self.get_num_samples(support_targets)

        (
            support_product_means,
            support_product_precisions,
            log_support_product_normalisation,
        ) = self.inner_gaussian_product(
            support_means, support_precisions, support_targets
        )
        (
            prior_product_mean,
            prior_product_precision,
            log_prior_product_normalisation,
        ) = self.calculate_Gaussian_prior_product(
            prior_shape=support_product_means.shape,
            prior_powers=num_samples - 1,
        )
        (
            posterior_means,
            posterior_precisions,
            log_quotient_normalisation,
        ) = self.normalise_by_Gaussian_prior(
            support_product_means,
            support_product_precisions,
            prior_product_mean,
            prior_product_precision,
        )

        log_Z = (
            log_quotient_normalisation
            + log_support_product_normalisation
            - log_prior_product_normalisation
        )
        return posterior_means, posterior_precisions, log_Z

    def calculate_Znv(
        self,
        posterior_means,
        posterior_precisions,
        log_Z,
        query_means,
        query_precisions,
    ):
        (
            additional_prior_posterior_means,
            additional_prior_posterior_precisions,
            log_additional_prior_normalisation,
        ) = self.normalise_by_Gaussian_prior(
            posterior_means,
            posterior_precisions,
            prior_product_mean=torch.zeros_like(posterior_means),
            prior_product_precision=self.prior_precision
            * torch.ones_like(posterior_precisions),
        )
        (
            outer_product_means,
            outer_product_precisions,
            log_outer_product_normalisation,
        ) = self.outer_gaussian_product(
            query_means,
            query_precisions,
            additional_prior_posterior_means,
            additional_prior_posterior_precisions,
        )

        log_Znv = (
            log_Z.unsqueeze(-1)
            + log_outer_product_normalisation
            + log_additional_prior_normalisation.unsqueeze(-1)
        )

        return log_Znv

    def calculate_posterior_q_no_prior(
        self, support_means, support_precisions, support_targets
    ):

        num_samples = self.get_num_samples(support_targets)

        (
            posterior_means,
            posterior_precisions,
            log_support_product_normalisation,
        ) = self.inner_gaussian_product(
            support_means, support_precisions, support_targets
        )

        log_Z = log_support_product_normalisation
        return posterior_means, posterior_precisions, log_Z

    def calculate_Znv_no_prior(
        self,
        posterior_means,
        posterior_precisions,
        log_Z,
        query_means,
        query_precisions,
    ):
        (
            outer_product_means,
            outer_product_precisions,
            log_outer_product_normalisation,
        ) = self.outer_gaussian_product(
            query_means,
            query_precisions,
            posterior_means,
            posterior_precisions,
        )

        log_Znv = log_Z.unsqueeze(-1) + log_outer_product_normalisation

        return log_Znv

    def compute_environment_representations(self, support_trajectories):

        support_means, support_precisions = self.encoder.forward(
            support_trajectories["observations"],
            support_trajectories["locations"] if self.use_location else None,
            support_trajectories["directions"] if self.use_direction else None,
            support_trajectories["coordinates"] if self.use_coordinates else None,
        )

        support_means = support_means.unsqueeze(0)
        support_precisions = support_precisions.unsqueeze(0) + self.prior_precision
        support_targets = support_trajectories["targets"].unsqueeze(0)

        (env_means, env_precisions, log_Z,) = self.calculate_posterior_q(
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
        support_precisions = support_precisions.unsqueeze(0) + self.prior_precision
        support_targets = support_trajectories["targets"].unsqueeze(0)
        num_samples = self.get_num_samples(support_targets)

        query_means = query_means.unsqueeze(0)
        query_precisions = query_precisions.unsqueeze(0) + self.prior_precision
        query_targets = query_views["targets"].unsqueeze(0)

        posterior_means, posterior_precisions, log_Z = self.calculate_posterior_q(
            support_means, support_precisions, support_targets
        )

        log_Znv = self.calculate_Znv(
            posterior_means,
            posterior_precisions,
            log_Z,
            query_means,
            query_precisions,
        )

        log_likelihood = (
            ((num_samples + 1) * log_Z)
            - num_samples * (torch.logsumexp(log_Znv, dim=-1))
            + (num_samples * math.log(log_Znv.shape[-1]))
        ).sum()

        # log_likelihood = (
        #     log_Z + num_samples*torch.log(torch.exp(log_Z)/torch.exp(log_Znv.mean(dim=-1)))
        # ).sum()
        print(support_precisions.mean())

        _, predictions = (log_Znv - log_Z.unsqueeze(-1)).max(1)

        loss = -log_likelihood
        # loss = F.cross_entropy(log_env_obs_normalisation, query_targets)
        accuracy = torch.eq(predictions, query_targets).float().mean()

        output = {}
        output["predictions"] = predictions
        output["loss"] = loss
        output["accuracy"] = accuracy
        output["mean_log_Z"] = log_Z.mean()
        output["mean_log_Znv"] = log_Znv.mean()
        output["loss_numerator"] = ((num_samples + 1) * log_Z).sum()
        output["loss_denominator"] = (torch.logsumexp(log_Znv, dim=-1)).sum()
        output["support_means"] = support_means.mean()
        output["support_precisions"] = support_precisions.mean()

        return output

    def compute_tau_scaling(self, precision):
        support_means = torch.zeros((1, 1, self.z_dim))
        support_precisions = precision * torch.ones((1, 1, self.z_dim))
        support_targets = torch.zeros((1, 1), dtype=torch.int64)
        num_samples = self.get_num_samples(support_targets)

        query_means = torch.zeros((1, 100, self.z_dim))
        query_precisions = precision * torch.ones((1, 100, self.z_dim))

        posterior_means, posterior_precisions, log_Z = self.calculate_posterior_q(
            support_means, support_precisions, support_targets
        )

        log_Znv = self.calculate_Znv(
            posterior_means,
            posterior_precisions,
            log_Z,
            query_means,
            query_precisions,
        )

        log_likelihood = (
            ((num_samples + 1) * log_Z)
            - num_samples * (torch.logsumexp(log_Znv, dim=-1))
            + (num_samples * math.log(log_Znv.shape[-1]))
        ).sum()

        return log_likelihood
