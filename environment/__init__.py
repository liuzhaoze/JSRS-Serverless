import os
from collections import namedtuple
from queue import PriorityQueue

import numpy as np
import torch
import yaml

from utils import load_hyperparameters

from .instance import *
from .job import *
from .speedup_model import SpeedupModel


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

        # 评价指标
        self.total_cost = 0.0
        self.success_count = self.fail_count = 0

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

    def __pop_job_id(self) -> int:
        return self.submit_queue.get()[1]

    def __glimpse_job_id(self) -> int:
        return self.submit_queue.queue[0][1]

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

        # 重置评价指标
        self.total_cost = 0.0
        self.success_count = self.fail_count = 0

        # 生成新状态
        self.__load_instances_config()
        self.__generate_workload()
        self.__init_queue()

    def done(self) -> bool:
        return self.submit_queue.empty()

    def instances_info(self, index: int = None) -> str:
        if index is None:
            info = ""
            for i, instance in enumerate(self.instances):
                info += f"{i}: {instance}\n"
            return info
        else:
            return str(self.instances[index]) + "\n"

    def jobs_info(self, index: int = None) -> str:
        if index is None:
            info = ""
            for job in self.jobs:
                info += str(job) + "\n"
            return info
        else:
            return str(self.jobs[index]) + "\n"

    def queue_info(self) -> str:
        return str(self.submit_queue.queue) + "\n"

    def state_dim(self) -> int:
        return 4 + len(self.instances)

    def get_state(self) -> torch.Tensor:
        """
        状态向量的组成：
        1. 当前任务的所需CPU数
        2. 当前任务的所需内存
        3. 当前任务的类型
        4. 当前任务的上一次执行所在区域
        5- 当前任务分配到每个实例上需要等待的时间
        """
        if self.done():
            # episode 结束时的状态为全零向量
            return torch.zeros(self.state_dim(), device=self.device).float()

        current_job_id = self.__glimpse_job_id()
        current_job = self.jobs[current_job_id]
        state = [
            current_job.required_cpu,
            current_job.required_memory,
            current_job.job_type.value,
            current_job.last_zone.value,
        ]
        for instance in self.instances:
            state.append(max(instance.idle_time - current_job.submit_time, 0))

        return torch.tensor(state, device=self.device).float()

    def action_dim(self) -> int:
        return len(self.instances)

    def take_action(self, action: int) -> torch.Tensor:
        current_job_id = self.__pop_job_id()
        current_job = self.jobs[current_job_id]
        target_instance = self.instances[action]

        # 计算 reward
        if (current_job.required_memory > target_instance.memory) or (
            current_job.job_type == JobType.rigid
            and current_job.required_cpu > target_instance.vCPU
        ):
            updated_instance, updated_job = target_instance, current_job
            reward = -5.0
            self.fail_count += 1  # 更新评价指标
        else:
            updated_instance, updated_job, result = Environment.assign(
                target_instance, current_job
            )
            updated_job.add_history(
                result.submit_time,
                result.start_time,
                result.end_time,
                action,
                updated_job.finished(),
            )
            reward = (
                (1 + np.exp(1.5 - result.cost))
                * (result.end_time - result.start_time)
                / (result.end_time - result.submit_time)
            )
            self.success_count += 1  # 更新评价指标

        # 更新实例和任务状态
        self.instances[action] = updated_instance
        self.jobs[current_job_id] = updated_job
        # 更新任务队列
        if not updated_job.finished():
            self.__submit_job(updated_job)

        # 更新评价指标
        self.total_cost += result.cost

        return torch.tensor([reward], device=self.device).float()

    @staticmethod
    def load_time(instance: Instance, job: Job) -> float:
        if job.last_zone == Zone.no_record or job.last_zone == instance.zone:
            return job.required_memory / Instance.load_speed / 3600.0
        else:
            return 2 * job.required_memory / Instance.load_speed / 3600.0

    @staticmethod
    def suspend_time(instance: Instance, job: Job) -> float:
        return job.required_memory / Instance.suspend_speed / 3600.0

    @staticmethod
    def exec_time(instance: Instance, job: Job) -> float:
        match job.job_type:
            case JobType.moldable:
                return (
                    job.length
                    * SpeedupModel.SU(job.required_cpu)
                    / SpeedupModel.SU(instance.vCPU)
                )
            case JobType.rigid:
                if job.required_cpu > instance.vCPU:
                    raise RuntimeError("CPU is not enough.")
                return job.length

    @staticmethod
    def exec_time_invert(t_exec: float, instance: Instance, job: Job) -> float:
        match job.job_type:
            case JobType.moldable:
                return (
                    t_exec
                    * SpeedupModel.SU(instance.vCPU)
                    / SpeedupModel.SU(job.required_cpu)
                )
            case JobType.rigid:
                return t_exec

    AssignResult = namedtuple(
        "AssignResult", ["cost", "submit_time", "start_time", "end_time", "wasted_time"]
    )

    @staticmethod
    def assign(instance: Instance, job: Job) -> (Instance, Job, AssignResult):
        if job.required_memory > instance.memory:
            raise RuntimeError("Memory is not enough.")
        if job.job_type == JobType.rigid and job.required_cpu > instance.vCPU:
            raise RuntimeError("CPU is not enough.")

        t_submit = job.submit_time  # 作为返回值记录

        # 在实例空闲后提交任务，产生时间浪费
        t_wasted = 0.0
        if job.submit_time >= instance.idle_time + instance.remaining_lease_term:
            t_wasted = instance.remaining_lease_term
            # 更新实例状态
            instance.remaining_lease_term = 0.0
            instance.idle_time = job.submit_time
        elif job.submit_time > instance.idle_time:
            t_wasted = job.submit_time - instance.idle_time
            # 更新实例状态
            instance.remaining_lease_term -= t_wasted
            instance.idle_time = job.submit_time

        assert job.submit_time <= instance.idle_time

        t_load = Environment.load_time(instance, job)
        t_suspend = Environment.suspend_time(instance, job)
        t_exec = Environment.exec_time(instance, job)

        # 租用实例
        cost = 0.0
        while instance.remaining_lease_term - t_load - t_suspend <= 0:
            cost += instance.rent(instance.idle_time)

        # 执行任务
        if t_exec <= instance.remaining_lease_term - t_load:
            # 任务全部执行
            t_begin = instance.idle_time + t_load
            t_end = t_begin + t_exec
            # 更新实例状态
            instance.idle_time = t_end
            instance.remaining_lease_term -= t_load + t_exec
            # 更新任务状态
            job.length = 0.0
            job.last_zone = instance.zone
        else:
            # 任务部分执行
            t_begin = instance.idle_time + t_load
            t_end = instance.idle_time + instance.remaining_lease_term - t_suspend
            t_exec_actual = t_end - t_begin
            # 更新实例状态
            instance.idle_time += instance.remaining_lease_term
            instance.remaining_lease_term = 0.0
            # 更新任务状态
            job.length -= Environment.exec_time_invert(t_exec_actual, instance, job)
            job.last_zone = instance.zone
            job.submit_time = instance.idle_time

        if not 0 < t_submit < t_begin < t_end:
            raise RuntimeError(
                f"Time error.\n{t_submit} {t_begin} {t_end}\n{job}\n{instance}"
            )

        return (
            instance,
            job,
            Environment.AssignResult(cost, t_submit, t_begin, t_end, t_wasted),
        )

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_success_rate(self) -> float:
        return float(self.success_count) / float(self.success_count + self.fail_count)

    def get_jobs_response_time(self) -> list[float]:
        return [job.total_response_time() for job in self.jobs]
