from requests import patch
import torch as T
import torch.optim as optim
import numpy as np
import argparse
import wandb
from dataclasses import astuple

from SSL.process_data import dataset_loader, observation_generators
from generative_contrastive_modelling.gcm import GenerativeContrastiveModelling
from generative_contrastive_modelling.protonet import PrototypicalNetwork


def parse_ssl_train_args():
    parser = argparse.ArgumentParser(
        description="SSL from partial observations training arguments"
    )
    parser.add_argument(
        "--learner",
        required=True,
        help="Learner to use. Currently support 'GCM' and 'proto'.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of complete epochs to train for",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=256,
        help="Number of images to classify from in batch",
    )
    parser.add_argument(
        "--n_support",
        type=int,
        default=5,
        help="Number of cropped observations of each image to learn from",
    )
    parser.add_argument(
        "--n_query",
        type=int,
        default=5,
        help="Number of query observations of each image",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=256,
        help="Embedding dimension for each observation/image",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for learner"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_ssl_train_args()
    wandb.init(project="gen-con-ssl")
    wandb.config.update(args)
    config = wandb.config

    dataloader = dataset_loader.CIFAR10Loader()
    trainset, testset, classes = dataloader.load_data(data_path="./SSL/data")

    # observation_generator = observation_generators.CropTrajectoryGenerator(
    #     batch_size=config.num_classes,
    #     image_shape=dataloader.image_shape,
    #     trajectory_length=config.n_support + config.n_query,
    #     random=False,
    #     patch_size=2,
    # )

    observation_generator = observation_generators.AugmentationTrajectoryGenerator(
        batch_size=config.num_classes,
        trajectory_length=config.n_support + config.n_query,
        image_shape=dataloader.image_shape,
    )

    trainiterator = dataloader.load_batches(
        dataset=trainset, batch_size=config.num_classes, shuffle=True
    )

    if config.learner == "GCM":
        learner = GenerativeContrastiveModelling(
            input_shape=astuple(dataloader.image_shape),
            hid_dim=config.embedding_dim,
            z_dim=config.embedding_dim,
            use_location=False,
            use_direction=False,
        )
    elif config.learner == "proto":
        learner = PrototypicalNetwork(
            input_shape=astuple(dataloader.image_shape),
            hid_dim=config.embedding_dim,
            z_dim=config.embedding_dim,
            use_location=False,
            use_direction=False,
            project_embedding=False,
        )

    optimizer = optim.Adam(learner.parameters(), lr=config.lr)

    num_episodes = (
        len(trainiterator)
        if len(trainset) % config.num_classes == 0
        else len(trainiterator) - 1
    )

    for epoch in range(config.num_epochs):
        for episode in range(num_episodes):

            images, labels = next(trainiterator)
            (
                cropped_images
                #                crop_coordinates,
            ) = observation_generator.generate_trajectories(images)

            support_images = cropped_images[:, : config.n_support, :, :, :].reshape(
                -1, *dataloader.image_shape
            )
            query_images = cropped_images[:, : config.n_query, :, :, :].reshape(
                -1, *dataloader.image_shape
            )
            support_targets = T.flatten(
                T.tensor(list(range(config.num_classes)), dtype=T.int64)
                .unsqueeze(1)
                .repeat(1, config.n_support)
            )
            query_targets = T.flatten(
                T.tensor(list(range(config.num_classes)), dtype=T.int64)
                .unsqueeze(1)
                .repeat(1, config.n_query)
            )

            # dataloader.display_image(images[0, :, :, :])
            # print(support_targets[:20])
            # dataloader.display_batch(support_images[:20, :, :, :])

            support_trajectories = {
                "observations": support_images,
                "targets": support_targets,
            }
            query_views = {"observations": query_images, "targets": query_targets}

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
                        "Training/Support Images": wandb.Image(support_images),
                        "Training/Query Images": wandb.Image(query_images),
                    }
                )

            print(
                f"Iteration: {episode}, \t"
                f"Loss: {outputs['loss']:.2f}, \t"
                f"Accuracy: {outputs['accuracy']:.2f}, \t"
                f"Predictions (5): {np.array(outputs['predictions'][0,:5])}, \t"
                f"Targets (5): {np.array(query_views['targets'][:5])}, \t"
                # f"Duration: {iteration_time:.1f}s"
            )

    wandb.finish()
