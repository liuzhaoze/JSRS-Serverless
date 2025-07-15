import numpy as np

from cloudcomputing_env.envs.config import ClusterConfig
from cloudcomputing_env.envs.instance import Instance


class Cluster:
    def __init__(self, config: ClusterConfig):
        self.__data_transfer_speed = np.array(config.data_transfer_speed)
        self.__data_transfer_cost = np.array(config.data_transfer_cost)
        self.__instances: list[Instance] = []

        for region_cfg in config.regions:
            for instance_cfg in region_cfg.instances:
                for _ in range(instance_cfg.count):
                    self.__instances.append(Instance(instance_cfg, region_cfg.region_code))

    def __getitem__(self, instance_id: int) -> Instance:
        return self.__instances[instance_id]

    def data_transfer_speed(self, src: int, dst: int) -> float:
        return self.__data_transfer_speed[src, dst]

    def data_transfer_cost(self, src: int, dst: int) -> float:
        return self.__data_transfer_cost[src, dst]

    @property
    def instance_cpu(self) -> list[int]:
        return [instance.cpu for instance in self.__instances]

    @property
    def instance_idle_time(self) -> list[float]:
        return [instance.idle_time for instance in self.__instances]

    @property
    def instance_expired_time(self) -> list[float]:
        return [instance.expired_time for instance in self.__instances]

    @property
    def instance_region(self) -> list[int]:
        return [instance.region for instance in self.__instances]
