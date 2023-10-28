import random

import torch


class AgentBase:
    def __init__(self, action_dim: int, device: torch.Tensor) -> None:
        self.action_dim = action_dim
        self.device = device

    def select_action(self, mask: torch.Tensor, *args) -> torch.Tensor:
        raise NotImplementedError


class DRLAgent(AgentBase):
    def __init__(self, strategy, action_dim: int, device: torch.Tensor) -> None:
        super().__init__(action_dim, device)

        self.strategy = strategy
        self.current_step = 0

    def select_action(
        self, mask: torch.Tensor, state: torch.Tensor, policy_net
    ) -> torch.Tensor:
        epsilon = self.strategy.get_epsilon(self.current_step)
        self.current_step += 1

        if random.random() < epsilon:
            # explore
            return random.choice(mask.nonzero()).to(self.device)
        else:
            # exploit
            with torch.no_grad():
                return (
                    policy_net(state)
                    .where(mask, float("-inf"))  # DOUBT: 会不会影响梯度计算？
                    .unsqueeze(dim=0)
                    .argmax(dim=1)
                    .to(self.device)
                )


class RandomAgent(AgentBase):
    def __init__(self, action_dim: int, device: torch.Tensor) -> None:
        super().__init__(action_dim, device)

    def select_action(self, mask: torch.Tensor, *args) -> torch.Tensor:
        return random.choice(mask.nonzero()).to(self.device)


class RoundRobinAgent(AgentBase):
    def __init__(self, action_dim: int, device: torch.Tensor) -> None:
        super().__init__(action_dim, device)

        self.step_count = -1

    def select_action(self, mask: torch.Tensor, *args) -> torch.Tensor:
        self.step_count += 1
        return torch.tensor([self.step_count % self.action_dim], device=self.device)


class EarliestAgent(AgentBase):
    def __init__(self, action_dim: int, device: torch.Tensor) -> None:
        super().__init__(action_dim, device)

    def select_action(
        self, mask: torch.Tensor, state: torch.Tensor, *args
    ) -> torch.Tensor:
        wait_time = state[4::2].where(mask, float("inf"))
        return wait_time.unsqueeze(dim=0).argmin(dim=1).to(self.device)
