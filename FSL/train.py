import numpy as np
import torch as T
import torchvision
import wandb

from FSL.utils import crop_input, mask_input, rescale_input
from torchvision.utils import save_image
import time


def train(
    train,
    max_epochs,
    epoch_size,
    dataloader,
    device,
    learner,
    optimizer,
    n_way,
    n_support,
    n_query,
    group_classes,
    apply_cropping,
    apply_masking,
    num_crops,
    patch_size,
    invert,
    no_noise,
    output_shape,
    use_coordinates,
):
    for epoch in range(max_epochs):
        for episode in range(epoch_size):

            if train == "train":
                inputs, targets = next(dataloader)["train"]
            elif train == "test":
                inputs, targets = next(dataloader)["test"]

            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.shape[0]
            inputs = inputs.reshape(
                batch_size * n_way, n_support + n_query, *inputs.shape[2:]
            )
            targets = targets.reshape(batch_size * n_way, n_support + n_query).repeat(
                1, num_crops
            )

            if group_classes > 1:
                targets = (
                    T.tensor(
                        range(len(targets[:, 0]) // group_classes),
                        dtype=T.int64,
                        device=device,
                    )
                    .repeat_interleave(group_classes)
                    .unsqueeze(1)
                    .expand_as(targets)
                )

            if apply_cropping:
                cropped_inputs_list = []
                coordinates_list = []
                for i in range(num_crops):
                    cropped_inputs, coordinates = crop_input(inputs, patch_size)
                    image_source = (
                        T.tensor(list(range(coordinates.shape[1])))
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .repeat(coordinates.shape[0], 1, 1)
                    )
                    coordinates = T.cat([coordinates, image_source], dim=2)
                    cropped_inputs_list.append(cropped_inputs)
                    coordinates_list.append(coordinates)
                augmented_inputs = T.cat(cropped_inputs_list, dim=1)
                coordinates = T.cat(coordinates_list, dim=1)

            elif apply_masking:
                masked_inputs_list = []
                coordinates_list = []
                for i in range(num_crops):
                    masked_inputs, coordinates = mask_input(
                        inputs, patch_size, invert, no_noise
                    )
                    image_source = (
                        T.tensor(list(range(coordinates.shape[1])))
                        .unsqueeze(0)
                        .unsqueeze(2)
                        .repeat(coordinates.shape[0], 1, 1)
                    )
                    coordinates = T.cat([coordinates, image_source], dim=2)
                    masked_inputs_list.append(masked_inputs)
                    coordinates_list.append(coordinates)
                augmented_inputs = T.cat(masked_inputs_list, dim=1)
                coordinates = T.cat(coordinates_list, dim=1)

            rescaled_inputs = rescale_input(augmented_inputs, output_shape)

            support_images = rescaled_inputs[
                :, : n_support * num_crops, :, :, :
            ].reshape(-1, *output_shape)

            query_images = rescaled_inputs[
                :, n_support * num_crops :: num_crops, :, :, :
            ].reshape(-1, *output_shape)

            if use_coordinates:
                support_coordinates = (
                    coordinates[:, : n_support * num_crops, :].reshape(-1, 5).to(device)
                )
                support_coordinates[:, 2] = (
                    support_coordinates[:, 2] - support_coordinates[:, 0]
                )
                support_coordinates[:, 3] = (
                    support_coordinates[:, 3] - support_coordinates[:, 1]
                )
                query_coordinates = (
                    coordinates[:, n_support * num_crops :: num_crops, :]
                    .reshape(-1, 5)
                    .to(device)
                )
                query_coordinates[:, 2] = (
                    query_coordinates[:, 2] - query_coordinates[:, 0]
                )
                query_coordinates[:, 3] = (
                    query_coordinates[:, 3] - query_coordinates[:, 1]
                )
                query_coordinates[:, 4] = -1

            support_targets = targets[:, : n_support * num_crops].reshape(-1)
            query_targets = targets[:, n_support * num_crops :: num_crops].reshape(-1)

            support_trajectories = {
                "observations": support_images,
                "targets": support_targets,
                "coordinates": support_coordinates if use_coordinates else None,
            }
            query_views = {
                "observations": query_images,
                "targets": query_targets,
                "coordinates": query_coordinates if use_coordinates else None,
            }

            if train == "train":
                optimizer.zero_grad()

                outputs = learner.compute_loss(
                    support_trajectories=support_trajectories, query_views=query_views
                )

                outputs["loss"].backward()
                optimizer.step()

                if learner.__class__.__name__ == "PartialObservationExpertsModelling":
                    wandb.log(
                        {
                            "Training/Loss": outputs["loss"],
                            "Training/Accuracy": outputs["accuracy"],
                            "Training/Support Precision": outputs[
                                "support_precision_mean"
                            ],
                            "Training/Query Precision": outputs["query_precision_mean"],
                            "Training/Support Precision Var": outputs[
                                "support_precision_var"
                            ],
                            "Training/Query Precision Var": outputs[
                                "query_precision_var"
                            ],
                        }
                    )
                else:
                    wandb.log(
                        {
                            "Training/Loss": outputs["loss"],
                            "Training/Accuracy": outputs["accuracy"],
                        }
                    )

                if epoch == 0 and episode == 0:
                    wandb.log(
                        {
                            "Training/Support Images": wandb.Image(
                                torchvision.utils.make_grid(
                                    support_images[:15], nrow=5
                                ),
                                caption=f"Samples of support images in first task from images: {support_targets[:15]}",
                            ),
                            "Training/Query Images": wandb.Image(
                                torchvision.utils.make_grid(query_images[:15], nrow=10),
                                caption=f"Samples of query images in first task from images: {query_targets[:15]}",
                            ),
                        }
                    )
            elif train == "test":
                with T.no_grad():
                    outputs = learner.compute_loss(
                        support_trajectories=support_trajectories,
                        query_views=query_views,
                    )

                    wandb.log(
                        {
                            "Testing/Loss": outputs["loss"],
                            "Testing/Accuracy": outputs["accuracy"],
                        }
                    )

            print(
                f"Iteration: {episode}, \t"
                f"Loss: {outputs['loss']:.2f}, \t"
                f"Accuracy: {outputs['accuracy']:.2f}, \t"
                f"Predictions (5): {(outputs['predictions'][0,:5]).cpu().numpy()}, \t"
                f"Targets (5): {(query_views['targets'][:5]).cpu().numpy()}, \t"
                # f"Duration: {iteration_time:.1f}s"
            )

    return outputs
