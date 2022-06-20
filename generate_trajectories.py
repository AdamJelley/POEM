import argparse
from errno import ENODEV
import os
import numpy
import pickle
import sys
import matplotlib.pyplot as plt

import minigrid_rl_starter.utils as utils
from minigrid_rl_starter.utils import device


# Parse arguments
def parse_generation_args():
    parser = argparse.ArgumentParser(description="Data generation arguments")
    parser.add_argument(
        "--env", required=True, help="name of the environment to be run (REQUIRED)"
    )
    parser.add_argument(
        "--model", required=True, help="name of the trained model (REQUIRED)"
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
        "--gif",
        type=str,
        default=None,
        help="store output as gif with the given filename",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="number of episodes to visualize"
    )
    parser.add_argument(
        "--memory", action="store_true", default=False, help="add a LSTM to the model"
    )
    parser.add_argument(
        "--text", action="store_true", default=False, help="add a GRU to the model"
    )
    parser.add_argument(
        "--render", action="store_true", default=False, help="render data generation"
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        default=False,
        help="save generated data to file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name of dataset to store (default: env-model-epsiodes)",
    )

    args = parser.parse_args()
    if args.dataset_name is None:
        args.dataset_name = f"{args.env}-{args.model}-{args.episodes}"
    return args


def generate_data(
    env,
    agent,
    episodes,
    render=False,
    gif=False,
    save_data=False,
    dataset_name="test_data",
):

    dataset = {}

    if gif:
        from array2gif import write_gif

        frames = []

    for episode in range(episodes):
        obs = env.reset()
        step = 0
        trajectory = {}

        while True:
            if render:
                env.render("human")

            if gif:
                frames.append(numpy.moveaxis(env.render("rgb_array"), 2, 0))

            obs["pixels"] = numpy.moveaxis(env.render("rgb_array"), 2, 0)
            obs["partial_pixels"] = numpy.moveaxis(
                env.get_obs_render(obs["partial_image"], tile_size=8), 2, 0
            )
            location = env.agent_pos
            action = agent.get_action(obs)
            new_obs, reward, done, _ = env.step(action)

            new_obs["pixels"] = numpy.moveaxis(env.render("rgb_array"), 2, 0)
            new_obs["partial_pixels"] = numpy.moveaxis(
                env.get_obs_render(new_obs["partial_image"], tile_size=8), 2, 0
            )
            new_location = env.agent_pos

            transition = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "new_obs": new_obs,
                "done": done,
                "location": location,
                "new_location": new_location,
                "direction": obs["direction"],
                "new_direction": new_obs["direction"],
            }

            trajectory[step] = transition

            obs = new_obs
            agent.analyze_feedback(reward, done)
            step += 1

            if done or render and env.window.closed:
                break

        dataset[episode] = trajectory
        if render and env.window.closed:
            break

    if gif:
        print("Saving gif... ", end="")
        write_gif(numpy.array(frames), args.gif + ".gif", fps=1 / args.pause)
        print("Done.")

    if save_data:
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        with open(f"datasets/{dataset_name}.pickle", "wb") as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


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

    model_dir = utils.get_model_dir(args.model, storage_dir="minigrid_rl_starter")

    agent = utils.Agent(
        env.observation_space,
        env.action_space,
        model_dir,
        argmax=args.argmax,
        use_memory=args.memory,
        use_text=args.text,
    )
    print("Agent loaded\n")

    # Run the agent
    dataset = generate_data(
        env,
        agent,
        args.episodes,
        render=args.render,
        gif=args.gif,
        save_data=args.save_data,
        dataset_name=args.dataset_name,
    )

    print(dataset)
