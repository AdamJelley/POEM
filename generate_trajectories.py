import argparse
from errno import ENODEV
import os
import numpy
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append("/Users/ajelley/Projects/gen-con-rl/minigrid-rl-starter/")
import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
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
    default=0.1,
    help="pause duration between two consequent actions of the agent (default: 0.1)",
)
parser.add_argument(
    "--gif", type=str, default=None, help="store output as gif with the given filename"
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
    "--dataset_name",
    type=str,
    default="Test",
    help="Name of dataset to store (default: env-model-epsiodes)",
)

args = parser.parse_args()
args.dataset_name = f"{args.env}-{args.model}-{args.episodes}"

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

# Run the agent

if args.gif:
    from array2gif import write_gif

    frames = []

# Create a window to view the environment
env.render("human")

dataset = {}

for episode in range(args.episodes):
    obs = env.reset()
    step = 0
    trajectory = {}

    while True:
        env.render("human")

        if args.gif:
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

        # new_obs["image"][env.agent_pos[0], env.agent_pos[1], 2] = 0
        # if numpy.allclose(new_obs["image"], obs["image"]):
        #     print("Ivariant")
        # else:
        #     print("rotated")
        # print(obs["image"], env.agent_pos)
        # print(env.agent_pos, env.agent_dir)
        # if episode == 0:
        #     numpy.save("test.npz", obs["image"])
        # plt.imshow(obs)
        # plt.show()  # show the figure, non-blocking
        # _ = input("Press [enter] to continue.")  # wait for input from the user
        # plt.close()  # close the figure to show the next one.
        obs = new_obs
        agent.analyze_feedback(reward, done)
        step += 1

        if done or env.window.closed:
            break

    dataset[episode] = trajectory
    if env.window.closed:
        break

if not os.path.isdir("datasets"):
    os.mkdir("datasets")
with open(f"datasets/{args.dataset_name}.pickle", "wb") as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(dataset)

if args.gif:
    print("Saving gif... ", end="")
    write_gif(numpy.array(frames), args.gif + ".gif", fps=1 / args.pause)
    print("Done.")
