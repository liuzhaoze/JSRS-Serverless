import matplotlib.pyplot as plt

from environment import InstanceType, Zone
from utils import load_data

if __name__ == "__main__":
    on_demand, spot = load_data()
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
