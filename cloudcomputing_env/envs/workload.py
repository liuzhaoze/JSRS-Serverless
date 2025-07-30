import queue
from typing import NamedTuple

import numpy as np

from cloudcomputing_env.envs.config import WorkloadConfig
from cloudcomputing_env.envs.job import Job


class SubmittedJob(NamedTuple):
    submit_time: float
    job_id: int


class Workload:
    def __init__(self, config: WorkloadConfig):
        self.__submit_queue = queue.PriorityQueue()

        arrival_intervals = np.random.exponential(1.0 / config.arrival_rate, config.number)
        arrival_times = np.cumsum(arrival_intervals)
        lengths = np.random.normal(config.average_length, 0.1 * config.average_length, config.number)
        parallelisms = np.random.choice([1, 2, 4, 8], config.number, p=[0.6, 0.25, 0.1, 0.05])
        regions = np.random.randint(0, config.region_number, config.number)
        data_sizes = np.random.normal(config.average_data_size, 0.1 * config.average_data_size, config.number)
        self.__jobs = [
            Job(i, arrival_times[i], lengths[i], parallelisms[i], regions[i], int(data_sizes[i]))
            for i in range(config.number)
        ]

        for job in self.__jobs:
            self.__submit_queue.put(SubmittedJob(job.arrival_time, job.job_id))

    def __getitem__(self, job_id: int) -> Job:
        return self.__jobs[job_id]

    def __len__(self) -> int:
        return len(self.__jobs)

    def next(self) -> SubmittedJob | None:
        if self.__submit_queue.empty():
            return None
        return self.__submit_queue.get()

    def resubmit(self, current_time: float, job_id: int):
        self.__jobs[job_id].arrival_time = current_time
        self.__submit_queue.put(SubmittedJob(current_time, job_id))
