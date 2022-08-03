from ctypes import ArgumentError
import torch as T
import torch.optim as optim
import argparse
import wandb

from torchmeta.datasets.helpers import omniglot, miniimagenet
from torchmeta.utils.data import BatchMetaDataLoader

from generative_contrastive_modelling.gcm import GenerativeContrastiveModelling
from generative_contrastive_modelling.protonet import PrototypicalNetwork
from FSL.train import train


# warnings.filterwarnings(action="ignore", module="torchvision")


def parse_fsl_args():
    parser = argparse.ArgumentParser(description="Run few-shot experiments.")
    parser.add_argument("--learner", required=True, type=str, help="Learner to use.")
    parser.add_argument(
        "--dataset", type=str, default="miniimagenet", help="Few-shot dataset to use."
    )
    parser.add_argument(
        "--masking", action="store_true", help="Apply masking to images."
    )
    parser.add_argument(
        "--cropping", action="store_true", help="Apply cropping to images."
    )
    parser.add_argument(
        "--patch_size", type=int, default=-1, help="Fix the width of crop or mask."
    )
    parser.add_argument(
        "--invert", action="store_true", help="Invert masking on images."
    )
    parser.add_argument(
        "--no_noise", action="store_true", help="Remove noise from masking."
    )
    parser.add_argument(
        "--use_coordinates",
        action="store_true",
        help="Use coordinates for learning representation.",
    )
    parser.add_argument(
        "--n_way", type=int, default=20, help="Number of classes to classify."
    )
    parser.add_argument(
        "--n_support", type=int, default=5, help="Number of support examples."
    )
    parser.add_argument(
        "--n_query", type=int, default=5, help="Number of query examples."
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="Representation size."
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for learner"
    )
    parser.add_argument("--num_epochs", type=int, default=10, help="Max epochs.")
    parser.add_argument("--epoch_size", type=int, default=2000, help="Epoch size.")
    parser.add_argument(
        "--test_episodes", type=int, default=2000, help="Number of evaluation episodes."
    )

    args = parser.parse_args()
    if args.use_coordinates and not (args.cropping or args.masking):
        raise ArgumentError("Cannot use coordinates without cropping or masking.")

    return args


if __name__ == "__main__":
    args = parse_fsl_args()
    wandb.init(project="gen-con-fsl")
    wandb.config.update(args)
    config = wandb.config

    if config.dataset == "omniglot":
        config.output_shape = (1, 28, 28)
        dataset = omniglot(
            "FSL/data/omniglot_data",
            ways=config.n_way,
            shots=config.n_support + config.n_query,
            test_shots=config.n_support + config.n_query,
            meta_train=True,
            download=True,
        )
    elif config.dataset == "miniimagenet":
        config.output_shape = (3, 84, 84)
        dataset = miniimagenet(
            "FSL/data/miniimagenet_data",
            ways=config.n_way,
            shots=config.n_support + config.n_query,
            test_shots=config.n_support + config.n_query,
            meta_train=True,
            download=False,
        )
    else:
        raise ValueError(f"Dataset {config.dataset} not recognised.")

    dataloader = iter(BatchMetaDataLoader(dataset, batch_size=1, num_workers=4))

    if config.learner == "GCM":
        learner = GenerativeContrastiveModelling(
            input_shape=config.output_shape,
            hid_dim=config.embedding_dim,
            z_dim=config.embedding_dim,
            use_location=False,
            use_direction=False,
            use_coordinates=config.use_coordinates,
        )
    elif config.learner == "proto":
        learner = PrototypicalNetwork(
            input_shape=config.output_shape,
            hid_dim=config.embedding_dim,
            z_dim=config.embedding_dim,
            use_location=False,
            use_direction=False,
            use_coordinates=config.use_coordinates,
            project_embedding=False,
        )

    optimizer = optim.Adam(learner.parameters(), lr=config.lr)

    outputs = train(
        max_epochs=config.num_epochs,
        epoch_size=config.epoch_size,
        dataloader=dataloader,
        learner=learner,
        optimizer=optimizer,
        n_way=config.n_way,
        n_support=config.n_support,
        n_query=config.n_query,
        apply_cropping=config.cropping,
        apply_masking=config.masking,
        patch_size=config.patch_size,
        invert=config.invert,
        no_noise=config.no_noise,
        output_shape=config.output_shape,
        use_coordinates=config.use_coordinates,
    )

    wandb.finish()
