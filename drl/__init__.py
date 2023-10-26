import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    # TODO: 改用噪声网络
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features=state_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=action_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


Experience = namedtuple("Experience", ("state", "action", "reward", "next_state"))


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience: Experience) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience

        self.push_count += 1

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size: int, threshold: int) -> bool:
        """
        batch_size: Check if there are enough experiences in memory to provide samples.
        threshold: Control the timing of DQN updates using experience replay.
        """
        return len(self.memory) >= batch_size and len(self.memory) >= threshold


class EpsilonGreedyStrategy:
    def __init__(self, start: float, end: float, decay: float) -> None:
        self.start = start
        self.end = end
        self.decay = decay

    def get_epsilon(self, current_step: int) -> float:
        return self.end + (self.start - self.end) * math.exp(
            -1.0 * current_step * self.decay
        )


def extract_tensors(experiences: list[Experience]) -> tuple[torch.Tensor]:
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    states = torch.stack(batch.state)
    actions = torch.cat(batch.action)
    rewards = torch.cat(batch.reward)
    next_states = torch.stack(batch.next_state)

    return (states, actions, rewards, next_states)


class QValues:
    # TODO: 完成带 mask 版本的
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(
        policy_net, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states: torch.Tensor) -> torch.Tensor:
        # 首先找到下一个状态是终止状态的样本索引
        final_state_locations = (
            next_states.flatten(start_dim=1).max(dim=1).values.eq(0).type(torch.bool)
        )  # 向量的最大分量为 0 的向量是全零向量，代表终止状态
        # 然后找到下一个状态不是终止状态的样本索引
        non_final_state_locations = final_state_locations == False
        # 下一个状态不是终止状态的样本
        non_final_states = next_states[non_final_state_locations]

        next_q_values = torch.zeros(next_states.shape[0], device=QValues.device)
        next_q_values[non_final_state_locations] = (
            target_net(non_final_states).max(dim=1).values.detach()
        )
        # 终止状态样本的 next_q_values 为 0
        # 因为计算 TD target: target_q_value = reward + (next_q_value * gamma) 时
        # reward 已经是 return 中需要加的最后一个奖励了，所以后面的 next_q_value 应该是 0

        return next_q_values
