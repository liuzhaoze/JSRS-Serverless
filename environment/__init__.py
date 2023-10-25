import os

import torch
import yaml

from utils import load_hyperparameters

from .instance import *
from .job import *


class Environment:
    def __init__(self, device: torch.device) -> None:
        self.device = device

        # 设置超参数
        hyperparameters = load_hyperparameters()
        self.n_jobs = hyperparameters["job_number"]
        self.submit_speed = hyperparameters["submit_speed"]
        self.job_length_distribution = hyperparameters["job_length_distribution"]
        self.job_type_ratio = hyperparameters["job_type_ratio"]

        self.instances = []  # 可供选择的实例
        self.jobs = []
        self.submit_queue = []  # 使用 job_id 作为队列元素

    def __load_instances_config(self) -> None:
        if len(self.instances) != 0:
            raise RuntimeError(
                "Instances have been loaded. You should make self.instances empty before loading instances."
            )
        cwd = os.getcwd()
        config_path = os.path.join(cwd, "config", "instances.yml")
        with open(config_path, "r") as f:
            instances_config = yaml.load(f, Loader=yaml.FullLoader)
        for instance_config in instances_config:
            self.instances.append(
                Instance(
                    InstanceType[instance_config["instance_type"]],
                    Zone[instance_config["zone"]],
                    BillingType[instance_config["billing_type"]],
                )
            )

    def reset(self) -> None:
        self.instances.clear()
        self.jobs.clear()
        self.submit_queue.clear()
        self.__load_instances_config()

    def instances_info(self, index: int = None) -> str:
        if index is None:
            info = ""
            for i, instance in enumerate(self.instances):
                info += f"{i}: {instance}\n"
            return info
        else:
            return str(self.instances[index])
