import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as Transforms
from dataclasses import astuple
import random


class CropTrajectoryGenerator:
    def __init__(
        self, batch_size, trajectory_length, image_shape, random=True, patch_size=10
    ):
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.image_shape = image_shape
        self.action_length = 4  # Corresponds to 4 coordinates of 2D crop
        self.random = random
        self.patch_size = patch_size

    def generate_random_crop_coordinates(self):

        self.rng = np.random.default_rng()

        x1 = (
            self.rng.random((self.batch_size, self.trajectory_length))
            * (self.image_shape.width - 2)
            + 1
        )
        x2 = (
            self.rng.random((self.batch_size, self.trajectory_length))
            * (self.image_shape.width - 2)
            + 1
        )
        y1 = (
            self.rng.random((self.batch_size, self.trajectory_length))
            * (self.image_shape.height - 2)
            + 1
        )
        y2 = (
            self.rng.random((self.batch_size, self.trajectory_length))
            * (self.image_shape.height - 2)
            + 1
        )

        xmin = (np.round(np.minimum(x1, x2)) - 1).astype(int)
        xmax = (np.round(np.maximum(x1, x2)) + 1).astype(int)
        ymin = (np.round(np.minimum(y1, y2)) - 1).astype(int)
        ymax = (np.round(np.maximum(y1, y2)) + 1).astype(int)

        coordinates = ((xmin, ymin), (xmax, ymax))
        return coordinates

    def generate_patch_crop_coordinates(self):

        self.rng = np.random.default_rng()

        xmin = np.floor(
            self.rng.random((self.batch_size, self.trajectory_length))
            * (self.image_shape.width - 1 - self.patch_size)
        ).astype(int)
        xmax = xmin + self.patch_size
        ymin = np.floor(
            self.rng.random((self.batch_size, self.trajectory_length))
            * (self.image_shape.height - 1 - self.patch_size)
        ).astype(int)
        ymax = ymin + self.patch_size

        coordinates = ((xmin, ymin), (xmax, ymax))
        return coordinates

    def generate_trajectories(self, images):
        """
        Function takes in batch of images, applies random rectangular crops, resizes and returns list of trajectories.

        Inputs:
            images (Tensor): Tensor of images of shape (batch_size, 3, height, width)
            trajectory_length (int): Number of random rectangular crops to apply to generate trajectory

        Returns:
            trajectories_list (list): List of trajectories of format {'states':state_list, 'actions':action_list}
        """

        repeated_images = images.unsqueeze(1).repeat(1, self.trajectory_length, 1, 1, 1)
        if self.random:
            ((xmin, ymin), (xmax, ymax)) = self.generate_random_crop_coordinates()
        else:
            ((xmin, ymin), (xmax, ymax)) = self.generate_patch_crop_coordinates()

        cropped_images = T.zeros_like(repeated_images)
        crop_coordinates = T.zeros(
            (self.batch_size, self.trajectory_length, self.action_length)
        )
        for i in range(self.batch_size):
            for j in range(self.trajectory_length):
                cropped_images[i, j, :, :, :] = Transforms.functional.resize(
                    repeated_images[
                        i, j, :, ymin[i, j] : ymax[i, j], xmin[i, j] : xmax[i, j]
                    ],
                    size=(self.image_shape.height, self.image_shape.width),
                )
                crop_coordinates[i, j, :] = T.Tensor(
                    [xmin[i, j], ymin[i, j], xmax[i, j], ymax[i, j]]
                )
        return cropped_images, crop_coordinates


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class AugmentationTrajectoryGenerator:
    def __init__(
        self,
        batch_size,
        trajectory_length,
        image_shape,
    ):
        self.batch_size = batch_size
        self.trajectory_length = trajectory_length
        self.image_shape = image_shape

        self.augment = nn.Sequential(
            RandomApply(Transforms.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.3),
            Transforms.RandomGrayscale(p=0.2),
            Transforms.RandomHorizontalFlip(),
            RandomApply(Transforms.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
            Transforms.RandomResizedCrop(
                (self.image_shape.height, self.image_shape.width)
            ),
            Transforms.Normalize(
                mean=T.tensor([0.485, 0.456, 0.406]),
                std=T.tensor([0.229, 0.224, 0.225]),
            ),
        )

    def generate_trajectories(self, images):
        augmented_images = (
            T.zeros_like(images).unsqueeze(1).repeat(1, self.trajectory_length, 1, 1, 1)
        )
        for i, image in enumerate(images):
            for view in range(self.trajectory_length):
                augmented_image = self.augment(image)
                augmented_images[i, view, :, :, :] = augmented_image
        return augmented_images
