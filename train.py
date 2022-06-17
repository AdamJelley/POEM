import sys
import numpy as np
import torch as T
import torch.optim as optim
import argparse
import wandb

sys.path.append("/Users/ajelley/Projects/gen-con-rl/minigrid-rl-starter/")
import utils
from utils import device
from generate_trajectories import generate_data
from process_trajectories import data_to_tensors, sample_views
from generative_contrastive_modelling.gcm import GenerativeContrastiveModelling
from generative_contrastive_modelling.protonet import PrototypicalNetwork


def parse_train_args():
    parser = argparse.ArgumentParser(
        description="Environment identification training arguments"
    )
    parser.add_argument(
        "--env", required=True, help="name of the environment to be run (REQUIRED)"
    )
    parser.add_argument(
        "--trained_agent",
        required=True,
        help="name of the trained model to generate support trajectories (REQUIRED)",
    )
    parser.add_argument(
        "--exploratory_agent",
        required=True,
        help="name of the exploratory model to generate query observations (REQUIRED)",
    )
    parser.add_argument(
        "--learner",
        required=True,
        help="Representation learning method: GCM or proto currently supported (REQUIRED)",
    )
    parser.add_argument(
        "--num_environments",
        type=int,
        default=10,
        help="number of environments to explore and classify",
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=100,
        help="Number of query observations to try to match",
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
        help="Allow leanrer to use agent direction info",
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
        "--render_trained",
        action="store_true",
        default=False,
        help="render trained agent data generation",
    )
    parser.add_argument(
        "--render_exploratory",
        action="store_true",
        default=False,
        help="render exploratory agent data generation",
    )
    parser.add_argument(
        "--num_tasks", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding size")
    parser.add_argument(
        "--use_grid",
        action="store_true",
        default=False,
        help="Use grid input rather than pixels",
    )

    args = parser.parse_args()
    if args.use_grid:
        args.input_shape = (3, 11, 11)
    else:
        # args.input_shape = (3, 352, 352)
        args.input_shape = (3, 56, 56)

    return args


if __name__ == "__main__":
    wandb.init(project="gen-con-rl")
    args = parse_train_args()
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

    env_copy = utils.make_env(config.env, config.seed)
    for _ in range(config.shift):
        env_copy.reset()
    print("Environment loaded\n")

    # Load agents

    trained_model_dir = utils.get_model_dir(
        config.trained_agent, storage_dir="minigrid-rl-starter"
    )
    exploratory_model_dir = utils.get_model_dir(
        config.exploratory_agent, storage_dir="minigrid-rl-starter"
    )

    trained_agent = utils.Agent(
        env.observation_space,
        env.action_space,
        trained_model_dir,
        argmax=config.argmax,
        use_memory=config.memory,
        use_text=config.text,
    )
    print("Trained agent loaded\n")

    exploratory_agent = utils.Agent(
        env.observation_space,
        env.action_space,
        exploratory_model_dir,
        argmax=config.argmax,
        use_memory=config.memory,
        use_text=config.text,
    )
    print("Exploratory agent loaded\n")

    # Load learner and optimizer
    if config.learner == "GCM":
        learner = GenerativeContrastiveModelling(
            config.input_shape,
            config.embedding_dim,
            config.embedding_dim,
            config.use_location,
            config.use_direction,
        )
        optimizer = optim.Adam(learner.gcm_encoder.parameters(), lr=0.001)
    elif config.learner == "proto":
        learner = PrototypicalNetwork(
            config.input_shape, config.embedding_dim, config.embedding_dim
        )
        optimizer = optim.Adam(learner.proto_encoder.parameters(), lr=0.001)

    # Start training
    for task in range(config.num_tasks):

        # Run the trained agent to generate training data
        train_dataset = generate_data(
            env,
            trained_agent,
            config.num_environments,
            render=config.render_trained,
        )

        support_trajectories = data_to_tensors(train_dataset)

        # Randomly sample query observations for now
        # indices = T.randperm(support_trajectories.shape[0])[: config.num_queries]
        # query_observations = support_trajectories[indices]
        # query_targets = support_targets[indices]

        # Run the exploratory agent to generate query data
        query_dataset = generate_data(
            env_copy,
            exploratory_agent,
            config.num_environments,
            render=config.render_exploratory,
        )

        query_trajectories = data_to_tensors(query_dataset)

        query_views = sample_views(query_trajectories, config.num_queries)

        # Reset optimizer
        optimizer.zero_grad()

        # Pass trajectories and queries to learner to train representations
        outputs = learner.compute_loss(
            support_trajectories=support_trajectories, query_views=query_views
        )

        outputs["loss"].backward()
        optimizer.step()

        print(
            f"Iteration: {task}, \tLoss: {outputs['loss']:.2f}, \tAccuracy: {outputs['accuracy']:.2f}"
        )
        wandb.log(
            {
                "Loss": outputs["loss"],
                "Accuracy": outputs["accuracy"],
            }
        )

    wandb.finish()
