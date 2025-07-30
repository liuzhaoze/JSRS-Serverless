from typing import Optional

from pydantic import BaseModel, model_validator


class WorkloadConfig(BaseModel):
    number: int
    arrival_rate: float
    average_length: float
    average_data_size: int
    region_number: int


class InstanceConfig(BaseModel):
    instance_type: str
    count: int
    cpu: int
    memory: int
    price: float


class RegionConfig(BaseModel):
    region_code: int
    region_name: str
    instances: list[InstanceConfig]


class ClusterConfig(BaseModel):
    regions: list[RegionConfig]
    data_transfer_speed: list[list[float]]
    data_transfer_cost: list[list[float]]

    @model_validator(mode="after")
    def validate_shape(self):
        n_regions = len(self.regions)
        x_speed, y_speed = len(self.data_transfer_speed), len(self.data_transfer_speed[0])
        x_cost, y_cost = len(self.data_transfer_cost), len(self.data_transfer_cost[0])
        if len({n_regions, x_speed, y_speed, x_cost, y_cost}) != 1:
            raise ValueError(
                f"Data transfer speed and cost matrices must match the number of regions.\nGot {n_regions} regions, speed: ({x_speed}, {y_speed}), cost: ({x_cost}, {y_cost})"
            )
        return self
