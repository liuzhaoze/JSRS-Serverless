import io
import os
import secrets
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

from cloudcomputing_env.envs.cluster import Cluster
from cloudcomputing_env.envs.speedup_model import S
from cloudcomputing_env.envs.workload import Workload

if TYPE_CHECKING:
    from argument.run import RunArgument


class CloudComputingEnv(gym.Env):
    metadata = {"render_modes": ["console", "file"]}

    def __init__(self, args: "RunArgument", render_mode=[]):
        self.__workload_config = args.workload_config
        self.__cluster_config = args.cluster_config
        self.__sigma = 0.0
        self.__lambda = args.reward_lambda

        self.N_REGION, self.N_INSTANCE = 0, 0
        for region in self.__cluster_config.regions:
            self.N_REGION += 1
            for instance in region.instances:
                self.N_INSTANCE += instance.count

        self.observation_space = gym.spaces.Dict(
            {
                "parallelism": gym.spaces.Box(0, 1.0, shape=(self.N_INSTANCE,), dtype=float),
                "wait_time": gym.spaces.Box(0, float("inf"), shape=(self.N_INSTANCE,), dtype=float),
                "expired_time": gym.spaces.Box(0, float("inf"), shape=(self.N_INSTANCE,), dtype=float),
                "job_region": gym.spaces.MultiBinary(self.N_REGION),
                "instance_region": gym.spaces.MultiBinary([self.N_REGION, self.N_INSTANCE]),
            }
        )

        self.action_space = gym.spaces.Discrete(self.N_INSTANCE)

        assert set(render_mode) <= set(self.metadata["render_modes"])
        self.render_mode = render_mode

        self.file = None
        self.log_dir = args.log_dir

    def _get_obs(self):
        return {
            "parallelism": np.array(self.__cluster.instance_cpu, dtype=float)
            / self.__workload[self.__current_job.job_id].parallelism,
            "wait_time": (
                np.array(self.__cluster.instance_idle_time, dtype=float) - self.__current_job.submit_time
            ).clip(min=0.0),
            "expired_time": (
                np.array(self.__cluster.instance_expired_time, dtype=float) - self.__current_job.submit_time
            )
            / self.__workload[self.__current_job.job_id].length,
            "job_region": np.eye(self.N_REGION, dtype=bool)[self.__workload[self.__current_job.job_id].region],
            "instance_region": np.eye(self.N_REGION, dtype=bool)[self.__cluster.instance_region],
        }

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.num_steps = 0

        self.__cluster = Cluster(self.__cluster_config)
        self.__workload = Workload(self.__workload_config)
        self.__current_job = self.__workload.next()

        if "file" in self.render_mode:
            if self.file is not None:
                # close the previous file belonging to the previous episode
                self.file.write("[\n" + self.file_buffer.getvalue() + "]\n")
                self.file.close()
            self.file = open(os.path.join(self.log_dir, f"env_log_{secrets.token_hex(3)}.json"), "w")
            self.file_buffer = io.StringIO()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        job_id = self.__current_job.job_id
        instance_id = action
        arrival_time = self.__workload[job_id].arrival_time

        if arrival_time >= self.__cluster[instance_id].expired_time:
            rental_cost = self.__cluster[instance_id].rent(arrival_time)
        else:
            rental_cost = 0.0

        start_time = max(arrival_time, self.__cluster[instance_id].idle_time)

        data_transfer_time = (
            self.__workload[job_id].data_size
            / 1_000_000  # Convert bytes to MB
            / self.__cluster.data_transfer_speed(self.__workload[job_id].region, self.__cluster[instance_id].region)
        )
        data_transfer_cost = (
            self.__workload[job_id].data_size
            / 1_000_000_000  # Convert bytes to GB
            * self.__cluster.data_transfer_cost(self.__workload[job_id].region, self.__cluster[instance_id].region)
        )

        parallelism = self.__workload[job_id].parallelism
        length = self.__workload[job_id].length
        execution_time = length * parallelism / S(parallelism, self.__sigma, self.__cluster[instance_id].cpu)

        finish_time = start_time + data_transfer_time + execution_time

        if finish_time <= self.__cluster[instance_id].expired_time:
            success = True
            self.__cluster[instance_id].idle_time = finish_time
        else:
            success = False
            self.__cluster[instance_id].idle_time = self.__cluster[instance_id].expired_time
            self.__workload.resubmit(self.__cluster[instance_id].expired_time, job_id)

        self.render_console_content = f"\rStep={self.num_steps}: Assign job({job_id}) to instance({instance_id})"
        self.render_file_content = (
            """{"arrival_time": %.6f, "job_id": %d, "success": %b, "start_time": %.6f, "finish_time": %.6f, "data_transfer_time": %.6f, "execution_time": %.6f, "data_transfer_cost": %.6f, "rental_cost": %.6f},\n"""
            % (
                arrival_time,
                job_id,
                success,
                start_time,
                finish_time,
                data_transfer_time,
                execution_time,
                data_transfer_cost,
                rental_cost,
            )
        )
        if job_id == len(self.__workload) - 1 and success:
            self.render_console_content += "\n"
            self.render_file_content = self.render_file_content[:-2] + "\n"

        self.__current_job = self.__workload.next()
        self.num_steps += 1

        terminated = self.__current_job is None
        reward = (
            (
                execution_time / (finish_time - arrival_time)
                + length / execution_time
                + np.exp(self.__lambda - (rental_cost + data_transfer_cost))
            )
            if success
            else -1.0
        )
        observation = self._get_obs()
        info = self._get_info()

        if len(self.render_mode) != 0:
            self.render()

        return observation, reward, terminated, False, info

    def render(self):
        if len(self.render_mode) == 0:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if "console" in self.render_mode:
            print(self.render_console_content, end="", flush=True)

        if "file" in self.render_mode:
            self.file_buffer.write(self.render_file_content)

    def close(self):
        if self.file is not None:
            self.file.write("[\n" + self.file_buffer.getvalue() + "]\n")
            self.file.close()
            self.file = None
