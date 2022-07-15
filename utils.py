import os
import torch as T
import wandb


def load_checkpoint(run_path):
    run_id = run_path.split("/")[-1]
    model_save_path = f"./wandb/saved_checkpoints/{run_id}/checkpoint.pt"
    if os.path.exists(model_save_path):
        checkpoint = T.load(model_save_path)
        print("Model loaded!")
    else:
        print("Model not found. Downloading model via wandb api...")
        api = wandb.Api()
        run = api.run(run_path)
        run.file("checkpoint.pt").download(root=f"./wandb/saved_checkpoints/{run_id}")
        checkpoint = T.load(model_save_path)
        print("Model downloaded and loaded!")
    return checkpoint
