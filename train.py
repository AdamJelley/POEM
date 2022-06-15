import sys
import numpy as np
import torch as T
import torch.optim as optim

sys.path.append("/Users/ajelley/Projects/gen-con-rl/minigrid-rl-starter/")
import utils
from utils import device
from generate_trajectories import parse_generation_args, generate_data
from generative_contrastive_modelling.gcm import GenerativeContrastiveModelling

num_tasks = 100
representation_dim = 64
input_shape = (3, 7, 7)

if __name__ == "__main__":
    args = parse_generation_args()

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device

    print(f"Device: {device}\n")

    # Load environment

    env = utils.make_env(args.env, args.seed)
    for _ in range(args.shift):
        env.reset()
    print("Environment loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args.model, storage_dir="minigrid-rl-starter")

    agent = utils.Agent(
        env.observation_space,
        env.action_space,
        model_dir,
        argmax=args.argmax,
        use_memory=args.memory,
        use_text=args.text,
    )
    print("Agent loaded\n")

    # Load GCM and optimizer
    GCM = GenerativeContrastiveModelling(input_shape, representation_dim)
    optimizer = optim.Adam(GCM.gcm_encoder.parameters(), lr=0.001)

    # Start training
    for task in range(num_tasks):

        # Run the agent to generate data
        dataset = generate_data(
            env,
            agent,
            args.episodes,
            render=args.render,
            gif=args.gif,
            save_data=args.save_data,
            dataset_name=args.dataset_name,
        )

        support_trajectories = T.Tensor(
            np.array(
                [
                    dataset[episode][step]["obs"]["partial_image"]
                    for episode in range(len(dataset))
                    for step in range(len(dataset[episode]))
                ]
            ).transpose(0, 3, 1, 2)
        )
        support_targets = T.tensor(
            np.array(
                [
                    episode
                    for episode in range(len(dataset))
                    for step in range(len(dataset[episode]))
                ]
            )
        )

        # Randomly sample query observations for now
        indices = T.randperm(support_trajectories.shape[0])[:5]
        query_observations = support_trajectories[indices]
        query_targets = support_targets[indices]

        # Reset optimizer
        optimizer.zero_grad()

        # Pass trajectories and queries to GCM to train representations
        GCM_outputs = GCM.compute_loss(
            support_trajectories=support_trajectories,
            support_targets=support_targets,
            query_observations=query_observations,
            query_targets=query_targets,
        )

        GCM_outputs["loss"].backward()
        optimizer.step()

        print(
            f"Iteration: {task}, \tPredictions: {GCM_outputs['predictions']}, \tTargets: {query_targets}, \tLoss: {GCM_outputs['loss']}, \tAccuracy: {GCM_outputs['accuracy']:.2f}"
        )
