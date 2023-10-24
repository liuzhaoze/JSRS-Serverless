from enum import Enum

from colorama import Fore

from utils import load_data, load_hyperparameters


class InstanceType(Enum):
    nano = 0
    micro = 1
    small = 2
    medium = 3
    large = 4
    xlarge = 5
    xxlarge = 6


class Zone(Enum):
    us_east_1 = 0
    us_east_2 = 1
    us_west_1 = 2
    us_west_2 = 3


class BillingType(Enum):
    on_demand = 0
    spot = 1


class Instance:
    hyperparameters = load_hyperparameters()
    on_demand_data, spot_data = load_data()
    load_speed = hyperparameters["load_speed"]
    suspend_speed = hyperparameters["suspend_speed"]
    rental_time = hyperparameters["rental_time"]

    def __init__(
        self, instance_type: InstanceType, zone: Zone, billing_type: BillingType
    ) -> None:
        match billing_type:
            case BillingType.on_demand:
                config = Instance.on_demand_data["t2"][instance_type.name]
            case BillingType.spot:
                config = Instance.spot_data["t2"][zone.name][instance_type.name]
            case _:
                raise ValueError("Invalid billing type")

        # 固有属性
        self.vCPU = config["vCPU"]
        self.memory = config["memory"]
        self.zone = zone
        self.billing_type = billing_type
        self.price = config["price"]

        # 状态属性
        self.remaining_lease_term = 0.0
        self.idle_time = 0.0

    def reset(self) -> None:
        self.remaining_lease_term = 0.0
        self.idle_time = 0.0

    def rent(self, renting_moment: float) -> float:
        self.remaining_lease_term += Instance.rental_time

        match self.billing_type:
            case BillingType.on_demand:
                current_price = self.price
            case BillingType.spot:
                current_price = self.price[int(renting_moment)]
            case _:
                raise ValueError("Invalid billing type")

        if current_price < 0:
            raise ValueError("There is no price record for this type of instance")

        return Instance.rental_time * current_price

    def __repr__(self) -> str:
        info = f"Instance {self.vCPU}CPU {self.memory}MB [{self.billing_type.name}] "
        info += f"Idle time: {self.idle_time:.4f} | Remain "
        info += (
            f"{Fore.RED}{self.remaining_lease_term:.4f}{Fore.RESET}"
            if self.remaining_lease_term == 0
            else f"{Fore.GREEN}{self.remaining_lease_term:.4f}{Fore.RESET}"
        )
        info += " hour(s)"
        return info
