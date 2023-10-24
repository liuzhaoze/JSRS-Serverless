import os

import ujson


def load_data():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data")

    with open(os.path.join(data_path, "on-demand.json"), "r") as f:
        on_demand = ujson.load(f)
    with open(os.path.join(data_path, "spot.json"), "r") as f:
        spot = ujson.load(f)

    return on_demand, spot
