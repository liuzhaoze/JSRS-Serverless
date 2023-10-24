import os

import matplotlib.pyplot as plt
import ujson

from serverless import InstanceType, Zone

cwd = os.getcwd()
data_path = os.path.join(cwd, "data")

with open(os.path.join(data_path, "on-demand.json"), "r") as f:
    on_demand = ujson.load(f)
with open(os.path.join(data_path, "spot.json"), "r") as f:
    spot = ujson.load(f)


plt.figure(1, figsize=(22, 12))
for i, (zone, instance_type) in enumerate(
    [(z, it) for z in Zone for it in InstanceType]
):
    plt.subplot(len(Zone), len(InstanceType), i + 1)
    plt.title(f"{zone.name} {instance_type.name}")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.plot(spot["t2"][zone.name][instance_type.name]["price"])
    plt.axhline(
        y=on_demand["t2"][instance_type.name]["price"], color="r", linestyle="--"
    )

plt.tight_layout()
plt.show()
# NOTE: 同一实例类型在不同区域的价格相同
# NOTE: spot 实例价格远低于 on-demand 实例价格
