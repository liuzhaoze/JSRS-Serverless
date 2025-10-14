import os
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, DirectoryPath, FilePath, field_validator, model_validator

from cloudcomputing_env.envs.config import ClusterConfig, WorkloadConfig


class RunArgument(BaseModel):
    device: str
    seed: int
    log_dir: DirectoryPath
    render_mode: Optional[str]
    workload_config_path: FilePath
    workload_config: Optional[WorkloadConfig] = None
    cluster_config_path: FilePath
    cluster_config: Optional[ClusterConfig] = None
    hidden_sizes: list[int]
    lr: float
    gamma: float
    td_step: int
    target_update_freq: int
    buffer_size: int
    prioritized_replay: bool
    alpha: float
    beta: float
    beta_final: float
    batch_size: int
    eps_begin: float
    eps_end: float
    eps_test: float
    anneal_start: int
    anneal_end: int
    reward_lambda: float
    num_train_env: int
    num_test_env: int
    epoch: int
    step_per_epoch: int
    step_per_collect: int
    update_per_step: float
    episode_per_test: int
    evaluation: bool
    model_path: Optional[FilePath]
    num_eval_env: int
    eval_episode: int
    baseline: Optional[Literal["random", "roundrobin", "earliest"]]

    @field_validator("log_dir", mode="before")
    def create_and_check_log_dir(cls, v):
        if not os.path.exists(v):
            os.makedirs(v)
        if len(os.listdir(v)) != 0:
            raise ValueError(f"{v} should be an empty directory.")
        return v

    @field_validator("workload_config_path", mode="before")
    def check_workload_config_path(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"{v} does not exist.")
        return v

    @field_validator("cluster_config_path", mode="before")
    def check_cluster_config_path(cls, v):
        if not os.path.isfile(v):
            raise ValueError(f"{v} does not exist.")
        return v

    @model_validator(mode="before")
    def load_config(cls, values):
        workload_config_path = values.get("workload_config_path")
        cluster_config_path = values.get("cluster_config_path")
        with open(workload_config_path, "r") as f:
            values["workload_config"] = WorkloadConfig(**yaml.safe_load(f))
        with open(cluster_config_path, "r") as f:
            values["cluster_config"] = ClusterConfig(**yaml.safe_load(f))
        return values

    @model_validator(mode="after")
    def validate_configs(self):
        if len(self.cluster_config.regions) != self.workload_config.region_number:
            raise ValueError(
                f"Number of regions in cluster config ({len(self.cluster_config.regions)}) does not match workload config ({self.workload_config.region_number})."
            )
        return self

    @model_validator(mode="after")
    def check_model_path_when_evaluation(self):
        if self.evaluation:
            match self.baseline:
                case "random" | "roundrobin" | "earliest":
                    pass  # model is not needed
                case None:
                    if self.model_path is None:
                        raise ValueError("model_path must be provided when evaluation.")
                case _:
                    raise ValueError(f"Unknown baseline: {self.baseline}.")
        return self
