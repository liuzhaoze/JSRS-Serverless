import os
import random

import numpy as np
import torch
import ujson
import yaml


def load_data():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data")

    with open(os.path.join(data_path, "on-demand.json"), "r") as f:
        on_demand = ujson.load(f)
    with open(os.path.join(data_path, "spot.json"), "r") as f:
        spot = ujson.load(f)

    return on_demand, spot


def load_hyperparameters():
    cwd = os.getcwd()
    config_path = os.path.join(cwd, "config")

    with open(os.path.join(config_path, "hyperparameters.yml"), "r") as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    return hyperparameters


def set_seed(reproducibility: bool, seed: dict) -> dict:
    if reproducibility:
        random.setstate(eval(seed["random"]))
        np.random.seed(seed["numpy"])
        torch.manual_seed(seed["torch"])
        torch.cuda.manual_seed(seed["torch_cuda"])
        torch.cuda.manual_seed_all(seed["torch_cuda_all"])

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    return {
        "random": random.getstate(),
        "numpy": np.random.get_state()[1][0],
        "torch": torch.initial_seed(),
        "torch_cuda": torch.cuda.initial_seed(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


if __name__ == "__main__":
    print(load_hyperparameters())
