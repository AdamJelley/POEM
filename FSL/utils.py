import numpy as np
import torch
import torchvision


def crop_input(input, patch_size=-1):
    if patch_size == -1:
        coordinates = generate_random_coordinates(data_size=input.shape)
    else:
        coordinates = generate_patch_coordinates(
            data_size=input.shape, patch_size=patch_size
        )
    cropped_input = apply_crop(input, coordinates)
    coordinates = (
        torch.tensor(coordinates)
        .permute(2, 3, 0, 1)
        .reshape(input.shape[0], input.shape[1], -1)
    )
    return cropped_input, coordinates


def mask_input(input, patch_size=-1, invert=False, no_noise=False):
    if patch_size == -1:
        coordinates = generate_random_coordinates(data_size=input.shape)
    else:
        coordinates = generate_patch_coordinates(
            data_size=input.shape, patch_size=patch_size
        )
    masked_input = apply_mask(input, coordinates, invert, no_noise)
    coordinates = (
        torch.tensor(coordinates)
        .permute(2, 3, 0, 1)
        .reshape(input.shape[0], input.shape[1], -1)
    )
    return masked_input, coordinates

def rescale_input(input, output_shape):
    rescaled_input = input.reshape(-1,*input.shape[2:])
    rescaled_input = torchvision.transforms.functional.resize(rescaled_input, size = (output_shape[-2], output_shape[-1]))
    rescaled_input = rescaled_input.reshape(*input.shape[:2], *output_shape)
    return rescaled_input

def apply_crop(data, coordinates):
    ((x1, y1), (x2, y2)) = coordinates
    cropped_data = torch.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if x1[i, j] != x2[i, j] and y1[i, j] != y2[i, j]:
                cropped_data[i, j, :, :, :] = torchvision.transforms.functional.resize(
                    data[i, j, :, x1[i, j] : x2[i, j], y1[i, j] : y2[i, j]],
                    size=(data.shape[-2], data.shape[-1]),
                )
            else:
                cropped_data[i, j, :, :, :] = data[i, j, :, :, :]
    return cropped_data


def apply_mask(data, coordinates, invert, no_noise):
    ((x1, y1), (x2, y2)) = coordinates
    mask = torch.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            mask[i, j, :, x1[i, j] : x2[i, j], y1[i, j] : y2[i, j]] = 1
    if invert:
        mask = 1 - mask

    masked_data = data * (1 - mask)

    if not no_noise:
        masked_data += mask * torch.rand_like(data)
    return masked_data


def generate_patch_coordinates(data_size, patch_size):
    rng = np.random.default_rng()
    x1 = (np.floor(rng.random(data_size[0:2]) * (data_size[-1] - patch_size))).astype(
        np.int32
    )
    x2 = (x1 + patch_size).astype(np.int32)
    y1 = (np.floor(rng.random(data_size[0:2]) * (data_size[-1] - patch_size))).astype(
        np.int32
    )
    y2 = (y1 + patch_size).astype(np.int32)
    coordinates = ((x1, y1), (x2, y2))
    return coordinates


def generate_random_coordinates(data_size):

    rng = np.random.default_rng()

    x1 = rng.random(data_size[0:2]) * data_size[-1]
    x2 = rng.random(data_size[0:2]) * data_size[-1]
    y1 = rng.random(data_size[0:2]) * data_size[-1]
    y2 = rng.random(data_size[0:2]) * data_size[-1]

    xmin = np.rint(np.minimum(x1, x2)).astype(np.int32)
    xmax = np.rint(np.maximum(x1, x2)).astype(np.int32)
    ymin = np.rint(np.minimum(y1, y2)).astype(np.int32)
    ymax = np.rint(np.maximum(y1, y2)).astype(np.int32)

    coordinates = ((xmin, ymin), (xmax, ymax))
    return coordinates
