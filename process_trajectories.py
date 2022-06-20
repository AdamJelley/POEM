import numpy as np
import torch as T
import torch.nn.functional as F


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
