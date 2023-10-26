import math
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
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
    def __init__(self, capacity) -> None:
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
