import argparse
import numpy as np
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import wandb

import minigrid_rl_starter.utils as utils
from minigrid_rl_starter.utils import device
from generate_trajectories import generate_data
from process_trajectories import data_to_tensors
from generative_contrastive_modelling.gcm import GenerativeContrastiveModelling
from generative_contrastive_modelling.protonet import PrototypicalNetwork
from generative_contrastive_modelling.environment_decoder import EnvironmentDecoder


def parse_train_args():
    parser = argparse.ArgumentParser(
        description="Environment representation decoding training arguments"
    )
    parser.add_argument(
        "--env", required=True, help="name of the environment to be run (REQUIRED)"
    )
    parser.add_argument(
        "--agent",
        required=True,
        help="name of the trained model to generate trajectories to learn environment representation from (REQUIRED)",
    )
    parser.add_argument(
        "--learner",
        required=True,
        help="Representation learning method: GCM or proto currently supported (REQUIRED)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument(
        "--num_environments",
        type=int,
        default=5,
        help="number of environments in batch",
    )
    parser.add_argument(
        "--use_location",
        action="store_true",
        default=False,
        help="Allow leanrer to use agent location info",
    )
    parser.add_argument(
        "--use_direction",
        action="store_true",
        default=False,
        help="Allow learner to use agent direction info",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--shift",
        type=int,
        default=0,
        help="number of times the environment is reset at the beginning (default: 0)",
    )
    parser.add_argument(
        "--argmax",
        action="store_true",
        default=False,
        help="select the action with highest probability (default: False)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0,
        help="pause duration between two consequent actions of the agent (default: 0.1)",
    )
    parser.add_argument(
        "--memory", action="store_true", default=False, help="add a LSTM to the model"
    )
    parser.add_argument(
        "--text", action="store_true", default=False, help="add a GRU to the model"
    )
    parser.add_argument(
        "--render_agent",
        action="store_true",
        default=False,
        help="render trained agent data generation",
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer size")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--use_grid",
        action="store_true",
        default=False,
        help="Use grid input rather than pixels",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=-1,
        help="Frquency to log sample images to wandb (-1 corresponds to no logging)",
    )

    args = parser.parse_args()
    args.test_seed = args.seed + 1337
    if args.use_grid:
        args.input_shape = (3, 11, 11)
    else:
        # args.input_shape = (3, 352, 352)
        args.input_shape = (3, 32, 32)
    return args


if __name__ == "__main__":
    args = parse_train_args()
    wandb.init(project="gen-con-rl")
    wandb.config.update(args)
    config = wandb.config

    # Set seed for all randomness sources
    utils.seed(config.seed)

    # Set device

    print(f"Device: {device}\n")

    # Load environments

    env = utils.make_env(config.env, config.seed)
    for _ in range(config.shift):
        env.reset()

    # Load agents

    agent_model_dir = utils.get_model_dir(
        config.agent, storage_dir="minigrid_rl_starter"
    )

    agent = utils.Agent(
        env.observation_space,
        env.action_space,
        agent_model_dir,
        argmax=config.argmax,
        use_memory=config.memory,
        use_text=config.text,
    )
    print("Agent loaded\n")

    # Load learner and optimizer
    if config.learner == "GCM":
        learner = GenerativeContrastiveModelling(
            config.input_shape,
            config.hidden_dim,
            config.embedding_dim,
            config.use_location,
            config.use_direction,
        )

    elif config.learner == "proto":
        learner = PrototypicalNetwork(
            config.input_shape,
            config.hidden_dim,
            config.embedding_dim,
            config.use_location,
            config.use_direction,
        )

    decoder = EnvironmentDecoder(
        config.embedding_dim, config.hidden_dim, config.input_shape
    )
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.lr)

    for episode in range(config.num_episodes):
        train_dataset = generate_data(
            env=env,
            agent=agent,
            episodes=config.num_environments,
            render=config.render_agent,
        )
        train_trajectories = data_to_tensors(train_dataset)
        environments = (
            F.interpolate(
                T.Tensor(
                    np.array(
                        [
                            train_dataset[episode][0]["obs"]["pixels"]
                            for episode in train_dataset
                        ]
                    )
                ),
                size=config.input_shape[-1],
            )
            / 255.0
        )
        observations = train_trajectories["observations"]
        locations = train_trajectories["locations"] if config.use_location else None
        directions = train_trajectories["directions"] if config.use_direction else None

        means, precisions = learner.encoder.forward(observations, locations, directions)
        means = means.unsqueeze(0).detach()
        precisions = precisions.unsqueeze(0).detach()
        (
            env_proto_means,
            env_proto_precisions,
            log_env_proto_normalisation,
        ) = learner.inner_gaussian_product(
            means, precisions, train_trajectories["targets"].unsqueeze(0)
        )

        env_reconstructions = decoder.forward(
            env_proto_means.squeeze(), env_proto_precisions.squeeze()
        )
        reconstruction_loss = F.mse_loss(env_reconstructions, environments)

        decoder_optimizer.zero_grad()
        reconstruction_loss.backward()
        decoder_optimizer.step()

        print(f"Episode: {episode}, \tLoss: {reconstruction_loss}")
        wandb.log({"Training/Loss": reconstruction_loss})

        if config.log_frequency != -1 and episode % config.log_frequency == 0:
            wandb.log(
                {
                    "Visualisation/Environments": wandb.Image(
                        torchvision.utils.make_grid(environments, nrow=5),
                        caption="Environments (downsampled)",
                    ),
                    "Visualisation/Reconstructions": wandb.Image(
                        torchvision.utils.make_grid(env_reconstructions, nrow=5),
                        caption="Reconstructions",
                    ),
                }
            )
