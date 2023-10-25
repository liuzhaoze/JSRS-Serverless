import os
from queue import PriorityQueue

import numpy as np
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
        self.moldable_job_ratio = hyperparameters["moldable_job_ratio"]

        # 环境状态
        self.instances = []  # 可供选择的实例
        self.jobs = []  # 储存所有任务的状态
        self.submit_queue = PriorityQueue()  # (submit_time, job_id) submit_time 越小优先级越高

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

    def __max_cpu(self) -> int:
        return max([instance.vCPU for instance in self.instances])

    def __max_memory(self) -> float:
        return max([instance.memory for instance in self.instances])

    def __generate_workload(self) -> None:
        if len(self.jobs) != 0:
            raise RuntimeError(
                "Jobs have been generated. You should make self.jobs empty before generating jobs."
            )
        # 生成任务所需核心数
        required_cpus = np.random.randint(1, self.__max_cpu() + 1, self.n_jobs)

        # 生成任务所需内存
        required_memories = np.random.uniform(100.0, self.__max_memory(), self.n_jobs)

        # 生成任务长度
        match self.job_length_distribution["name"]:
            case "uniform":
                job_lengths = np.random.uniform(
                    self.job_length_distribution["parameters"][0],
                    self.job_length_distribution["parameters"][1],
                    self.n_jobs,
                )
            case "normal":
                job_lengths = np.random.normal(
                    self.job_length_distribution["parameters"][0],
                    self.job_length_distribution["parameters"][1],
                    self.n_jobs,
                )
            case _:
                raise ValueError(
                    f"Unknown job length distribution: {self.job_length_distribution['name']}"
                )

        # 生成任务提交间隔
        submit_intervals = np.random.exponential(1 / self.submit_speed, self.n_jobs)

        # 生成任务类型
        job_types = np.random.choice(
            [JobType.moldable, JobType.rigid],
            self.n_jobs,
            p=[self.moldable_job_ratio, 1.0 - self.moldable_job_ratio],
        )

        # 生成任务
        submit_time = 0.0
        for i in range(self.n_jobs):
            submit_time += submit_intervals[i]
            self.jobs.append(
                Job(
                    i,
                    required_cpus[i],
                    required_memories[i],
                    job_types[i],
                    submit_time,
                    job_lengths[i],
                )
            )

    def __submit_job(self, j: Job) -> None:
        self.submit_queue.put((j.submit_time, j.job_id))

    def __init_queue(self) -> None:
        if len(self.jobs) == 0:
            raise RuntimeError(
                "Jobs have not been generated. You should generate jobs by using __generate_workload before initializing queue."
            )
        for j in self.jobs:
            self.__submit_job(j)

    def reset(self) -> None:
        # 重置环境状态
        self.instances.clear()
        self.jobs.clear()
        with self.submit_queue.mutex:
            self.submit_queue.queue.clear()

        # 生成新状态
        self.__load_instances_config()
        self.__generate_workload()
        self.__init_queue()

    def instances_info(self, index: int = None) -> str:
        if index is None:
            info = ""
            for i, instance in enumerate(self.instances):
                info += f"{i}: {instance}\n"
            return info
        else:
            return str(self.instances[index])

    def jobs_info(self, index: int = None) -> str:
        if index is None:
            info = ""
            for job in self.jobs:
                info += str(job) + "\n"
            return info
        else:
            return str(self.jobs[index])

    def queue_info(self) -> str:
        return str(self.submit_queue.queue)
