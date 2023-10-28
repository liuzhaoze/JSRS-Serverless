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


def set_seed(reproducibility: bool, seed):
    if type(seed) == int:
        seed = [seed] * 5

    # 必须指定 random 模块的种子，因为无法获得 random 模块当前使用的种子
    random.seed(seed[0])

    if reproducibility:
        np.random.seed(seed[1])
        torch.manual_seed(seed[2])
        torch.cuda.manual_seed(seed[3])
        torch.cuda.manual_seed_all(seed[4])

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print("Using seed:")
    print(f"\trandom.seed: {seed[0]}")
    print(f"\tnp.random.seed: {np.random.get_state()[1][0]}")
    print(f"\ttorch.manual_seed: {torch.initial_seed()}")
    print(f"\ttorch.cuda.manual_seed: {torch.cuda.initial_seed()}")


if __name__ == "__main__":
    print(load_hyperparameters())
