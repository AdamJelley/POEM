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
    environments = F.interpolate(
        T.Tensor(
            np.array([dataset[episode][0]["obs"]["pixels"] for episode in dataset])
        ),
        size=dataset[0][0]["obs"]["partial_pixels"].shape[1:],
    )
    environment_targets = T.tensor(np.array([episode for episode in dataset]))

    trajectories = {
        "observations": observations,
        "targets": targets,
        "locations": locations,
        "directions": directions,
        "environments": environments,
        "environment_targets": environment_targets,
    }

    return trajectories


def orientate_observations(trajectories):
    for i in range(len(trajectories["directions"])):
        if trajectories["directions"][i][0] == 1:
            trajectories["observations"][i] = T.rot90(
                trajectories["observations"][i], -1, [1, 2]
            )
        elif trajectories["directions"][i][1] == 1:
            trajectories["observations"][i] = T.rot90(
                trajectories["observations"][i], 2, [1, 2]
            )
        elif trajectories["directions"][i][2] == 1:
            trajectories["observations"][i] = T.rot90(
                trajectories["observations"][i], 1, [1, 2]
            )
        elif trajectories["directions"][i][3] == 1:
            pass
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
        "environments": trajectories["environments"],
        "environment_targets": trajectories["environment_targets"],
    }

    return views, remaining_trajectories


def get_environment_queries(trajectories, num_queries):

    indices = T.randperm(trajectories["environment_targets"].shape[0])[:num_queries]
    environments = trajectories["environments"][indices]
    environment_targets = trajectories["environment_targets"][indices]
    locations = T.ones_like(trajectories["locations"][indices]) * -1
    directions = T.ones_like(trajectories["directions"][indices]) * -1

    views = {
        "observations": environments,
        "targets": environment_targets,
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
