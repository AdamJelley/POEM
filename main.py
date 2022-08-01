import sys
import numpy as np
import torch as T
import torch.optim as optim
import torchvision
import argparse
import wandb

from minigrid_rl_starter import utils
from minigrid_rl_starter.utils import device
from generate_trajectories import generate_data
from process_trajectories import data_to_tensors, sample_views, generate_visualisations
from generative_contrastive_modelling.gcm import GenerativeContrastiveModelling
from generative_contrastive_modelling.unsupervised_gcm import (
    UnsupervisedGenerativeContrastiveModelling,
)
from generative_contrastive_modelling.protonet import PrototypicalNetwork
from generative_contrastive_modelling.recurrent_agent import RecurrentAgent
from generative_contrastive_modelling.complete_observation_learner import (
    CompleteObservationLearner,
)
from train import train

T.autograd.set_detect_anomaly(True)


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
        type=str,
        default=None,
        help="name of the exploratory model to generate query observations (if None, queries are sampled and removed from support trajectories)",
    )
    parser.add_argument(
        "--learner",
        required=True,
        help="Representation learning method: GCM, unsupervised_GCM, proto, recurrent, complete_observations currently supported (REQUIRED)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of training episodes"
    )
    parser.add_argument(
        "--num_train_tasks", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument(
        "--num_test_tasks", type=int, default=100, help="Number of testing episodes"
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
        help="Allow learner to use agent direction info",
    )
    parser.add_argument(
        "--project_embedding",
        action="store_true",
        default=False,
        help="Project environment embedding",
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
        "--log_samples",
        action="store_true",
        default=False,
        help="Log sample images to wandb",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume previous run. Note: Must pass checkpoint path to resume from.",
    )
    parser.add_argument(
        "--run_path",
        type=str,
        default=None,
        help="Run path to resume, if resume flag is passed.",
    )

    args = parser.parse_args()
    args.test_seed = args.seed + 1337
    if args.use_grid:
        args.input_shape = (3, 11, 11)
    else:
        # args.input_shape = (3, 352, 352)
        args.input_shape = (3, 56, 56)
    if args.resume:
        args.run_id = args.run_path.split("/")[2]
        assert args.run_id is not None, "Must pass --run_path to resume run."

    return args


if __name__ == "__main__":
    args = parse_train_args()
    if args.resume:
        wandb.init(id=args.run_id, resume="must")
        args.resume = False  # To keep wandb config consistent
        args.run_path = None
    else:
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

    env_copy = utils.make_env(config.env, config.seed)
    for _ in range(config.shift):
        env_copy.reset()
    print("Environment loaded\n")

    # Load agents

    trained_model_dir = utils.get_model_dir(
        config.trained_agent, storage_dir="minigrid_rl_starter"
    )

    trained_agent = utils.Agent(
        obs_space=env.observation_space,
        action_space=env.action_space,
        model_dir=trained_model_dir,
        argmax=config.argmax,
        use_memory=config.memory,
        use_text=config.text,
    )
    print("Trained agent loaded\n")

    if config.exploratory_agent is not None:
        exploratory_model_dir = utils.get_model_dir(
            config.exploratory_agent, storage_dir="minigrid_rl_starter"
        )

        exploratory_agent = utils.Agent(
            obs_space=env.observation_space,
            action_space=env.action_space,
            model_dir=exploratory_model_dir,
            argmax=False,
            use_memory=config.memory,
            use_text=config.text,
        )
        print("Exploratory agent loaded\n")
    else:
        exploratory_agent = None

    # Load learner and optimizer
    if config.learner == "GCM":
        learner = GenerativeContrastiveModelling(
            input_shape=config.input_shape,
            hid_dim=config.hidden_dim,
            z_dim=config.embedding_dim,
            use_location=config.use_location,
            use_direction=config.use_direction,
        )

    elif config.learner == "unsupervised_GCM":
        learner = UnsupervisedGenerativeContrastiveModelling(
            input_shape=config.input_shape,
            hid_dim=config.hidden_dim,
            z_dim=config.embedding_dim,
            use_location=config.use_location,
            use_direction=config.use_direction,
        )

    elif config.learner == "proto":
        learner = PrototypicalNetwork(
            input_shape=config.input_shape,
            hid_dim=config.hidden_dim,
            z_dim=config.embedding_dim,
            use_location=config.use_location,
            use_direction=config.use_direction,
            project_embedding=config.project_embedding,
        )

    elif config.learner == "recurrent":
        learner = RecurrentAgent(
            input_shape=config.input_shape,
            hid_dim=config.hidden_dim,
            z_dim=config.embedding_dim,
            use_location=config.use_location,
            use_direction=config.use_direction,
            project_embedding=config.project_embedding,
        )
    elif config.learner == "complete_observations":
        learner = CompleteObservationLearner(
            input_shape=config.input_shape,
            hid_dim=config.hidden_dim,
            z_dim=config.embedding_dim,
            orient_queries=True,
            use_location=config.use_location,
            use_direction=config.use_direction,
        )

    optimizer = optim.Adam(learner.parameters(), lr=config.lr)
    # wandb.watch(learner, log="all", log_freq=1, log_graph=True)
    if config.resume:
        checkpoint = T.load(wandb.restore("checkpoint.pt").name)
        epoch = checkpoint["epoch"]
        learner.load_state_dict(checkpoint["learner_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        loss = checkpoint["loss"]
        accuracy = checkpoint["accuracy"]

    # Start training
    print("Starting training...")
    train(
        mode="train",
        num_epochs=config.num_epochs,
        num_tasks=config.num_train_tasks,
        num_environments=config.num_environments,
        num_queries=config.num_queries,
        env=env,
        env_copy=env_copy,
        env_seed=config.seed,
        trained_agent=trained_agent,
        exploratory_agent=exploratory_agent,
        learner=learner,
        optimizer=optimizer,
        render_trained=config.render_trained,
        render_exploratory=config.render_exploratory,
        log_samples=config.log_samples,
    )
    print("Training complete!")

    # Load trained model for testing
    # learner.encoder.load_checkpoint(checkpoint_dir=wandb.run.dir)

    # Test model
    print("Starting testing...")
    train(
        mode="test",
        num_epochs=1,
        num_tasks=config.num_test_tasks,
        num_environments=config.num_environments,
        num_queries=config.num_queries,
        env=env,
        env_copy=env_copy,
        env_seed=config.test_seed,
        trained_agent=trained_agent,
        exploratory_agent=exploratory_agent,
        learner=learner,
        optimizer=optimizer,
        render_trained=config.render_trained,
        render_exploratory=config.render_exploratory,
        log_samples=False,
    )
    print("Testing complete!\n")

    wandb.finish()
