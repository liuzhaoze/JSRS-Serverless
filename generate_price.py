import os

import numpy as np
import ujson

from environment import InstanceType, Zone
from utils import load_data

length = 1000

on_demand, _ = load_data()
instances_of_each_region = on_demand["t2"]

for it in InstanceType:
    mu = instances_of_each_region[it.name]["price"] * 0.9
    sigma = mu * 0.1
    spot_price = [round(x, 4) for x in np.random.normal(mu, sigma, length)]
    instances_of_each_region[it.name]["price"] = spot_price

spot = {"t2": {}}

for z in Zone:
    if z == Zone.no_record:
        continue
    spot["t2"][z.name] = instances_of_each_region

with open(os.path.join(os.getcwd(), "data", "spot.json"), "w", encoding="utf8") as f:
    ujson.dump(spot, f, indent=4)
