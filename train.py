import os
import numpy as np
import torch as T
import time
import wandb
import matplotlib.pyplot as plt

from generate_trajectories import generate_data
from process_trajectories import (
    data_to_tensors,
    orientate_observations,
    remove_seen_queries,
    sample_views,
    get_environment_queries,
    generate_visualisations,
)


def train(
    mode,
    num_epochs,
    num_tasks,
    num_environments,
    num_queries,
    env,
    env_copy,
    env_seed,
    trained_agent,
    exploratory_agent,
    learner,
    optimizer,
    environment_queries,
    render_trained=False,
    render_exploratory=False,
    log_samples=False,
):

    for epoch in range(num_epochs):
        # For identical epochs, reset seeds and environments
        T.manual_seed(0)
        env.seed(env_seed)
        env_copy.seed(env_seed)
        env.reset()
        env_copy.reset()
        print("---------")
        print(f"|Epoch {epoch+1}|")
        print("---------")
        for task in range(num_tasks):
            t0 = time.time()
            # Run the trained agent to generate training data
            train_dataset = generate_data(
                env,
                trained_agent,
                num_environments,
                render=render_trained,
            )

            support_trajectories = data_to_tensors(train_dataset)

            if exploratory_agent is not None:
                # Run the exploratory agent to generate query data
                query_dataset = generate_data(
                    env_copy,
                    exploratory_agent,
                    num_environments,
                    render=render_exploratory,
                )

                query_dataset_filtered = remove_seen_queries(
                    query_dataset, train_dataset
                )

                query_trajectories_filtered = data_to_tensors(query_dataset_filtered)

                query_views, _ = sample_views(query_trajectories_filtered, num_queries)

            elif environment_queries:
                query_views = get_environment_queries(support_trajectories, num_queries)

            else:
                query_views, remaining_support_trajectories = sample_views(
                    support_trajectories, num_queries
                )
                support_trajectories = remaining_support_trajectories

            assert (
                len(T.unique(support_trajectories["targets"])) == num_environments
            ), "No support data for an environment! Try reducing the number of queries or increasing the number of environments."

            if mode == "train":
                # Reset optimizer
                optimizer.zero_grad()

                # Pass trajectories and queries to learner to train representations
                outputs = learner.compute_loss(
                    support_trajectories=support_trajectories, query_views=query_views
                )

                outputs["loss"].backward()
                optimizer.step()

                wandb.log(
                    {
                        "Training/Loss": outputs["loss"],
                        "Training/Accuracy": outputs["accuracy"],
                    }
                )

            elif mode == "test":
                if task == 0:
                    learner.eval()
                    wandb.define_metric("Testing/Loss", summary="mean")
                    wandb.define_metric("Testing/Accuracy", summary="mean")

                with T.no_grad():
                    outputs = learner.compute_loss(
                        support_trajectories=support_trajectories,
                        query_views=query_views,
                    )

                wandb.log(
                    {
                        "Testing/Loss": outputs["loss"],
                        "Testing/Accuracy": outputs["accuracy"],
                    }
                )

            if log_samples and epoch == 0 and task == 0:
                (
                    support_environments,
                    query_images,
                    support_trajectory_env_view,
                    support_trajectory_agent_view,
                ) = generate_visualisations(train_dataset, query_views)

                wandb.log(
                    {
                        "Images/Environments": support_environments,
                        "Images/Query image samples": query_images,
                        "Trajectories/Trained agent navigating environment - env view": support_trajectory_env_view,
                        "Trajectories/Trained agent navigating environment - agent view": support_trajectory_agent_view,
                    }
                )

            iteration_time = time.time() - t0
            print(
                f"Iteration: {task}, \t"
                f"Loss: {outputs['loss']:.2f}, \t"
                f"Accuracy: {outputs['accuracy']:.2f}, \t"
                f"Predictions (5): {np.array(outputs['predictions'][0,:5])}, \t"
                f"Targets (5): {np.array(query_views['targets'][:5])}, \t"
                f"Duration: {iteration_time:.1f}s"
            )

        checkpoint_path = os.path.join(wandb.run.dir, "checkpoint.pt")
        print(f"Saving checkpoint to {checkpoint_path}...")
        T.save(
            {
                "epoch": epoch + 1,
                "learner_state_dict": learner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": outputs["loss"],
                "accuracy": outputs["accuracy"],
            },
            checkpoint_path,
        )
        wandb.save(checkpoint_path, base_path=checkpoint_path)
