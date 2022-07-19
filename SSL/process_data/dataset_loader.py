import torch
import torchvision
import torchvision.transforms as transforms

import os
import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass


@dataclass
class ImageShape:
    """Dataclass to hold image shape"""

    channels: int
    width: int
    height: int

    def __iter__(self):
        # Make this dataclass iterable so that we can convert shape to list
        yield from self.__dict__.values()


class CIFAR10Loader:
    def __init__(self):
        self.image_shape = (3, 32, 32)
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )  # TODO These normalisation values are approximate for simplicity and could be improved:
        # e.g. https://github.com/kuangliu/pytorch-cifar/issues/19

    def load_data(self, data_path="./data", download=False):
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=download, transform=self.normalize
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=download, transform=self.normalize
        )

        classes = trainset.classes

        return trainset, testset, classes

    def load_batches(self, dataset, batch_size, shuffle, num_workers=0):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        return iter(dataloader)

    def display_image(self, image):
        image = (
            image / 2.0 + 0.5
        )  # Unnormalise using approximate normalisation values above
        npimage = image.numpy()
        plt.imshow(np.transpose(npimage, (1, 2, 0)))
        plt.show()

    def display_batch(self, images):
        self.display_image(torchvision.utils.make_grid(images))


class COCOLoader:
    def __init__(self):
        self.image_shape = (3, 224, 224)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                transforms.Resize((224, 224)),
            ]
        )

    def load_data(self, data_path="/disk/scratch_fast/datasets/coco/", download=False):
        image_path = os.path.join(data_path, "train2017/")
        ann_file = os.path.join(data_path, "annotations/captions_train2017.json")
        trainset = torchvision.datasets.CocoCaptions(
            root=image_path,
            annFile=ann_file,
            transform=self.transform,
            target_transform=lambda x: x[0],
        )

        return trainset

    def load_batches(self, dataset, batch_size, shuffle, num_workers=0):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        return iter(dataloader)

    def display_image(self, image):
        image = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        )(
            image
        )  # Unnormalise
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.show()

    def display_batch(self, images):
        self.display_image(torchvision.utils.make_grid(images))
