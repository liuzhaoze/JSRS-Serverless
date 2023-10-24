import os

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


if __name__ == "__main__":
    print(load_hyperparameters())
