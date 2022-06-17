import numpy as np
import torch as T


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

    trajectories = {
        "observations": observations,
        "targets": targets,
        "locations": locations,
    }

    return trajectories
