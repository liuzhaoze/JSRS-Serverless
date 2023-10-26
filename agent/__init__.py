import random

import torch


class AgentBase:
    def __init__(self, action_dim: int, device: torch.Tensor) -> None:
        self.action_dim = action_dim
        self.device = device

    def select_action(self, mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DRLAgent(AgentBase):
    def __init__(self, strategy, action_dim: int, device: torch.Tensor) -> None:
        super().__init__(action_dim, device)

        self.strategy = strategy
        self.current_step = 0

    def select_action(
        self, mask: torch.Tensor, state: torch.Tensor, policy_net
    ) -> torch.Tensor:
        # TODO: 完成带 mask 版本的
        epsilon = self.strategy.get_epsilon(self.current_step)
        self.current_step += 1

        if random.random() < epsilon:
            # explore
            action = random.randrange(self.action_dim)
            return torch.tensor([action], device=self.device)
        else:
            # exploit
            with torch.no_grad():
                return policy_net(state).unsqueeze(dim=0).argmax(dim=1).to(self.device)
