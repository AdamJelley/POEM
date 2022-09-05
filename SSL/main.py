import os
import torch as T
import torch.optim as optim
import torchvision
from torchvision import transforms as Transforms
import numpy as np
import argparse
import GPUtil
import wandb
from dataclasses import astuple

from SSL.process_data import dataset_loader, observation_generators
from learners.gcm import GenerativeContrastiveModelling
from learners.unsupervised_gcm import UnsupervisedGenerativeContrastiveModelling
from learners.protonet import PrototypicalNetwork

T.autograd.set_detect_anomaly(True)


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
        "--dataset",
        type=str,
        default="COCO",
        help="Dataset to use. Currently CIFAR10 or COCO.",
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
        default=10,
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
        "--environment_queries",
        action="store_true",
        default=False,
        help="Use full images as queries.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=-1,
        help="If cropping not random, then size of patch to crop.",
    )
    parser.add_argument(
        "--use_coordinates",
        action="store_true",
        default=False,
        help="Provide coordinates to learn representation.",
    )
    parser.add_argument(
        "--output_size",
        type=int,
        default=32,
        help="Output image size after cropping and resizing.",
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
    args.output_shape = (3, args.output_size, args.output_size)
    return args


if __name__ == "__main__":
    args = parse_ssl_train_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    deviceID = GPUtil.getFirstAvailable(
        order="load", maxLoad=0.1, maxMemory=0.1, attempts=1, interval=900, verbose=True
    )[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(deviceID)
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    wandb.init(project="gen-con-ssl")
    wandb.config.update(args)
    config = wandb.config

    if config.dataset == "CIFAR10":
        dataloader = dataset_loader.CIFAR10Loader()
        trainset, testset, classes = dataloader.load_data(data_path="./SSL/data")
    elif config.dataset == "COCO":
        dataloader = dataset_loader.COCOLoader()
        trainset = dataloader.load_data(data_path="/disk/scratch_fast/datasets/coco/")

    if config.environment_queries:
        trajectory_length = config.n_support
    else:
        trajectory_length = config.n_support + config.n_query

    observation_generator = observation_generators.CropTrajectoryGenerator(
        batch_size=config.num_classes,
        trajectory_length=trajectory_length,
        image_shape=dataloader.image_shape,
        output_shape=config.output_shape,
        patch_size=config.patch_size,
    )

    # observation_generator = observation_generators.AugmentationTrajectoryGenerator(
    #     batch_size=config.num_classes,
    #     trajectory_length=config.n_support + config.n_query,
    #     image_shape=dataloader.image_shape,
    #     output_shape=config.output_shape,
    # )

    if config.learner == "GCM":
        learner = GenerativeContrastiveModelling(
            input_shape=config.output_shape,
            hid_dim=config.embedding_dim,
            z_dim=config.embedding_dim,
            use_location=False,
            use_direction=False,
            use_coordinates=config.use_coordinates,
        ).to(device)
    elif config.learner == "unsupervised_GCM":
        learner = UnsupervisedGenerativeContrastiveModelling(
            input_shape=config.output_shape,
            hid_dim=config.embedding_dim,
            z_dim=config.embedding_dim,
            prior_precision=0.1,
            use_location=False,
            use_direction=False,
            use_coordinates=config.use_coordinates,
        ).to(device)
    elif config.learner == "proto":
        learner = PrototypicalNetwork(
            input_shape=config.output_shape,
            hid_dim=config.embedding_dim,
            z_dim=config.embedding_dim,
            use_location=False,
            use_direction=False,
            use_coordinates=config.use_coordinates,
            project_embedding=False,
        ).to(device)

    optimizer = optim.Adam(learner.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        trainiterator = dataloader.load_batches(
            dataset=trainset, batch_size=config.num_classes, shuffle=True
        )
        num_episodes = (
            len(trainiterator)
            if len(trainset) % config.num_classes == 0
            else len(trainiterator) - 1
        )
        for episode in range(num_episodes):

            images, labels = next(trainiterator)
            images = images.to(device)

            (
                cropped_images,
                crop_coordinates,
            ) = observation_generator.generate_trajectories(images)

            if config.environment_queries:
                support_images = cropped_images.reshape(-1, *config.output_shape).to(
                    device
                )
                query_images = Transforms.functional.resize(
                    images, size=(config.output_shape[1], config.output_shape[2])
                ).to(device)

                support_coordinates = crop_coordinates.reshape(-1, 4).to(device)
                query_coordinates = (
                    T.tensor([0, 0, config.output_shape[1], config.output_shape[2]])
                    .unsqueeze(0)
                    .repeat(query_images.shape[0], 1)
                    .to(device)
                )

                support_targets = T.flatten(
                    T.tensor(list(range(config.num_classes)), dtype=T.int64)
                    .unsqueeze(1)
                    .repeat(1, config.n_support)
                ).to(device)
                query_targets = T.flatten(
                    T.tensor(list(range(config.num_classes)), dtype=T.int64).unsqueeze(
                        1
                    )
                ).to(device)
            else:
                support_images = (
                    cropped_images[:, : config.n_support, :, :, :]
                    .reshape(-1, *config.output_shape)
                    .to(device)
                )
                query_images = (
                    cropped_images[:, config.n_support :, :, :, :]
                    .reshape(-1, *config.output_shape)
                    .to(device)
                )

                support_coordinates = (
                    crop_coordinates[:, : config.n_support, :].reshape(-1, 4).to(device)
                )
                query_coordinates = (
                    crop_coordinates[:, config.n_support :, :].reshape(-1, 4).to(device)
                )

                support_targets = T.flatten(
                    T.tensor(list(range(config.num_classes)), dtype=T.int64)
                    .unsqueeze(1)
                    .repeat(1, config.n_support)
                ).to(device)
                query_targets = T.flatten(
                    T.tensor(list(range(config.num_classes)), dtype=T.int64)
                    .unsqueeze(1)
                    .repeat(1, config.n_query)
                ).to(device)

            # dataloader.display_image(images[0, :, :, :])
            # print(support_targets[:20])
            # dataloader.display_batch(support_images[:20, :, :, :])

            support_trajectories = {
                "observations": support_images,
                "targets": support_targets,
                "coordinates": support_coordinates,
            }
            query_views = {
                "observations": query_images,
                "targets": query_targets,
                "coordinates": query_coordinates,
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
                            torchvision.utils.make_grid(support_images[:50], nrow=10),
                            caption=f"Samples of support images in first task from images: {support_targets}",
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
                f"Predictions (5): {outputs['predictions'][0,:5].cpu().numpy()}, \t"
                f"Targets (5): {query_views['targets'][:5].cpu().numpy()}, \t"
                # f"Duration: {iteration_time:.1f}s"
            )

    wandb.finish()
