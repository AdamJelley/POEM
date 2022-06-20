import numpy as np
import torch as T
import torch.nn.functional as F
import torchvision
import wandb


def data_to_tensors(dataset):

    observations = T.Tensor(
        np.array(
            [
                dataset[episode][step]["obs"]["partial_pixels"]
                for episode in range(len(dataset))
                for step in range(len(dataset[episode]))
            ]
        )
    )
    targets = T.tensor(
        np.array(
            [
                episode
                for episode in range(len(dataset))
                for step in range(len(dataset[episode]))
            ]
        )
    )
    locations = T.Tensor(
        np.array(
            [
                dataset[episode][step]["location"]
                for episode in range(len(dataset))
                for step in range(len(dataset[episode]))
            ]
        )
    )
    directions = F.one_hot(
        T.Tensor(
            np.array(
                [
                    dataset[episode][step]["direction"]
                    for episode in range(len(dataset))
                    for step in range(len(dataset[episode]))
                ]
            )
        ).to(T.int64),
        4,
    )

    trajectories = {
        "observations": observations,
        "targets": targets,
        "locations": locations,
        "directions": directions,
    }

    return trajectories


def sample_views(trajectories, num_queries):

    indices = T.randperm(trajectories["targets"].shape[0])[:num_queries]
    observations = trajectories["observations"][indices]
    targets = trajectories["targets"][indices]
    locations = trajectories["locations"][indices]
    directions = trajectories["directions"][indices]

    views = {
        "observations": observations,
        "targets": targets,
        "locations": locations,
        "directions": directions,
    }
    return views


def generate_visualisations(train_dataset, query_views):
    support_environments = wandb.Image(
        torchvision.utils.make_grid(
            [
                T.Tensor(train_dataset[episode][0]["obs"]["pixels"])
                for episode in range(len(train_dataset))
            ],
            nrow=5,
        ),
        caption=f"Environments from first task",
    )
    support_trajectory_env_view = wandb.Video(
        np.stack(
            [
                train_dataset[0][step]["obs"]["pixels"]
                for step in range(len(train_dataset[0]))
            ]
        ).astype("u1"),
        caption=f"Environment view (only for demonstration)",
        fps=5,
    )
    support_trajectory_agent_view = wandb.Video(
        np.stack(
            [
                np.repeat(
                    np.repeat(
                        train_dataset[0][step]["obs"]["partial_pixels"],
                        4,
                        axis=1,
                    ),
                    4,
                    axis=2,
                )
                for step in range(len(train_dataset[0]))
            ]
        ).astype("u1"),
        caption=f"Agent view (used for training)",
        fps=5,
    )
    query_images = wandb.Image(
        query_views["observations"][:5],
        caption=f"Samples of query images in first task from environments: {query_views['targets'][:5]}",
    )
    return (
        support_environments,
        query_images,
        support_trajectory_env_view,
        support_trajectory_agent_view,
    )
