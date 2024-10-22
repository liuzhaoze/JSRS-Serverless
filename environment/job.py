from enum import Enum

import pandas as pd
import tabulate

from .instance import Zone


class JobType(Enum):
    moldable = 0
    rigid = 1


class Job:
    def __init__(
        self,
        job_id: int,
        required_cpu: int,
        required_memory: float,
        job_type: JobType,
        submit_time: float,
        length: float,
    ) -> None:
        self.job_id = job_id

        # 固有属性
        self.required_cpu = required_cpu
        self.required_memory = required_memory  # 执行任务所需的内存
        self.job_type = job_type

        # 状态属性
        self.length = length
        self.submit_time = submit_time
        self.last_zone = Zone.no_record

        self.exec_history = pd.DataFrame(
            columns=["submit_time", "start_time", "end_time", "instance_id", "finished"]
        )

    def finished(self) -> bool:
        return self.length == 0

    def add_history(
        self,
        submit_time: float,
        start_time: float,
        end_time: float,
        instance_id: int,
        finished: bool,
    ) -> None:
        self.exec_history.loc[len(self.exec_history)] = [
            submit_time,
            start_time,
            end_time,
            instance_id,
            finished,
        ]

    def get_history(self) -> str:
        return str(
            tabulate.tabulate(self.exec_history, headers="keys", tablefmt="fancy_grid")
        )

    def total_response_time(self) -> float:
        if not self.finished():
            raise Exception("Job is not finished yet.")
        return (self.exec_history["end_time"] - self.exec_history["submit_time"]).sum()

    def total_execution_time(self) -> float:
        if not self.finished():
            raise Exception("Job is not finished yet.")
        return (self.exec_history["end_time"] - self.exec_history["start_time"]).sum()

    def __repr__(self) -> str:
        info = f"Job({self.job_id}) {self.required_cpu}CPU {self.required_memory:.2f}MB [{self.job_type.name}] "
        info += f"Submit time: {self.submit_time:.4f} | Length: {self.length:.4f} hour(s) | Last zone: {self.last_zone.name}"
        return info
