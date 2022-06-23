import numpy as np
import torch as T
import torch.nn.functional as F
import torchvision
import copy
import wandb


def data_to_tensors(dataset):

    observations = T.Tensor(
        np.array(
            [
                dataset[episode][step]["obs"]["partial_pixels"]
                for episode in dataset
                for step in dataset[episode]
            ]
        )
    )
    targets = T.tensor(
        np.array([episode for episode in dataset for step in dataset[episode]])
    )
    locations = T.Tensor(
        np.array(
            [
                dataset[episode][step]["location"]
                for episode in dataset
                for step in dataset[episode]
            ]
        )
    )
    directions = F.one_hot(
        T.Tensor(
            np.array(
                [
                    dataset[episode][step]["direction"]
                    for episode in dataset
                    for step in dataset[episode]
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


def remove_seen_queries(query_dataset, train_dataset):
    # print(f"Initial queries: {sum([1 for episode in query_dataset for step in query_dataset[episode]])}")
    # count=0
    query_dataset_filtered = copy.deepcopy(query_dataset)
    for episode in range(len(train_dataset)):
        for query_step in range(len(query_dataset[episode])):
            for train_step in range(len(train_dataset[episode])):
                if (
                    query_dataset[episode][query_step]["location"]
                    == train_dataset[episode][train_step]["location"]
                ) and (
                    query_dataset[episode][query_step]["direction"]
                    == train_dataset[episode][train_step]["direction"]
                ):
                    del query_dataset_filtered[episode][query_step]
                    # print(episode, query_step, train_step)
                    # count += 1
                    break
    # print(f"Remaining queries: {sum([1 for j in query_dataset_filtered for i in query_dataset_filtered[j]])}")
    # print(f"Queries removed: {count}")
    return query_dataset_filtered


def sample_views(trajectories, num_queries):

    indices = T.randperm(trajectories["targets"].shape[0])[:num_queries]
    observations = trajectories["observations"][indices]
    targets = trajectories["targets"][indices]
    locations = trajectories["locations"][indices]
    directions = trajectories["directions"][indices]

    remaining_indices = T.tensor(
        [i for i in range(trajectories["targets"].shape[0]) if i not in indices]
    )
    remaining_observations = trajectories["observations"][remaining_indices]
    remaining_targets = trajectories["targets"][remaining_indices]
    remaining_locations = trajectories["locations"][remaining_indices]
    remaining_directions = trajectories["directions"][remaining_indices]

    views = {
        "observations": observations,
        "targets": targets,
        "locations": locations,
        "directions": directions,
    }

    remaining_trajectories = {
        "observations": remaining_observations,
        "targets": remaining_targets,
        "locations": remaining_locations,
        "directions": remaining_directions,
    }

    return views, remaining_trajectories


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
