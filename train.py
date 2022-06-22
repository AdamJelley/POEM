import torch as T
import time
import wandb

from generate_trajectories import generate_data
from process_trajectories import data_to_tensors, sample_views, generate_visualisations


def train(
    mode,
    num_tasks,
    num_environments,
    num_queries,
    env,
    env_copy,
    trained_agent,
    exploratory_agent,
    learner,
    optimizer,
    render_trained=False,
    render_exploratory=False,
    log_samples=False,
):

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

        # Run the exploratory agent to generate query data
        query_dataset = generate_data(
            env_copy,
            exploratory_agent,
            num_environments,
            render=render_exploratory,
        )

        query_trajectories = data_to_tensors(query_dataset)

        query_views = sample_views(query_trajectories, num_queries)

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
                    support_trajectories=support_trajectories, query_views=query_views
                )

            wandb.log(
                {
                    "Testing/Loss": outputs["loss"],
                    "Testing/Accuracy": outputs["accuracy"],
                }
            )

        if log_samples and task == 0:
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
            f"Iteration: {task}, \tLoss: {outputs['loss']:.2f}, \tAccuracy: {outputs['accuracy']:.2f}, \tDuration: {iteration_time:.1f}s"
        )
