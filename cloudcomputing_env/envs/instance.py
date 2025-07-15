from cloudcomputing_env.envs.config import InstanceConfig


class Instance:
    def __init__(self, config: InstanceConfig, region: int):
        self.cpu = config.cpu
        self.memory = config.memory
        self.price = config.price
        self.region = region
        self.__hour = 3600.0

        self.expired_time = 0.0
        self.idle_time = 0.0

    def rent(self, current_time: float, hour: int = 1) -> float:
        assert current_time >= self.expired_time
        self.expired_time = current_time + hour * self.__hour
        return self.price * hour
