import numpy as np
import torchvision
import wandb

from FSL.utils import crop_input, mask_input


def train(
    max_epochs,
    epoch_size,
    dataloader,
    device,
    learner,
    optimizer,
    n_way,
    n_support,
    n_query,
    apply_cropping,
    apply_masking,
    patch_size,
    invert,
    no_noise,
    output_shape,
    use_coordinates,
):
    for epoch in range(max_epochs):
        for episode in range(epoch_size):

            train_inputs, train_targets = next(dataloader)["train"]
            train_inputs = train_inputs.to(device)
            train_targets = train_targets.to(device)

            batch_size = train_inputs.shape[0]
            train_inputs = train_inputs.reshape(
                batch_size * n_way, n_support + n_query, *train_inputs.shape[2:]
            )
            train_targets = train_targets.reshape(
                batch_size * n_way, n_support + n_query
            )

            if apply_cropping:
                train_inputs, coordinates = crop_input(train_inputs, patch_size)
            elif apply_masking:
                train_inputs, coordinates = mask_input(
                    train_inputs, patch_size, invert, no_noise
                )

            support_images = train_inputs[:, :n_support, :, :, :].reshape(
                -1, *output_shape
            )

            query_images = train_inputs[:, n_support:, :, :, :].reshape(
                -1, *output_shape
            )

            if use_coordinates:
                support_coordinates = coordinates[:, :n_support, :].reshape(-1, 4).to(device)
                support_coordinates[:,2] = support_coordinates[:,2]-support_coordinates[:,0]
                support_coordinates[:,3] = support_coordinates[:,3]-support_coordinates[:,1]
                query_coordinates = coordinates[:, n_support:, :].reshape(-1, 4).to(device)
                query_coordinates[:,2] = query_coordinates[:,2]-query_coordinates[:,0]
                query_coordinates[:,3] = query_coordinates[:,3]-query_coordinates[:,1]

            support_targets = train_targets[:, :n_support].reshape(-1)
            query_targets = train_targets[:, n_support:].reshape(-1)

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

            optimizer.zero_grad()

            outputs = learner.compute_loss(
                support_trajectories=support_trajectories, query_views=query_views
            )

            outputs["loss"].backward()
            optimizer.step()

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
                            torchvision.utils.make_grid(support_images[:5], nrow=5),
                            caption=f"Samples of support images in first task from images: {support_coordinates[:5]}",
                        ),
                        "Training/Query Images": wandb.Image(
                            torchvision.utils.make_grid(query_images[:50], nrow=10),
                            caption=f"Samples of query images in first task from images: {query_targets}",
                        ),
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
