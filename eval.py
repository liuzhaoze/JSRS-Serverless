import os
import sys
from itertools import count

import torch
from torch.utils.tensorboard import SummaryWriter

from agent import DRLAgent
from drl import DQN, EpsilonGreedyStrategy
from environment import Environment
from utils import load_hyperparameters

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python eval.py <model_path>")
        sys.exit(1)

    hyperparameters = load_hyperparameters()
    use_mask = hyperparameters["use_mask"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    env = Environment(use_mask, device)
    env.reset()

    # 加载模型参数
    model_path = os.path.abspath(sys.argv[1])
    policy_net = DQN(env.state_dim(), env.action_dim()).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    epsilon_greedy = EpsilonGreedyStrategy(0, 0, 0)
    drl_agent = DRLAgent(epsilon_greedy, env.action_dim(), device)

    # 开始评估
    env.reset()
    state, mask = env.get_state()

    for step in count():
        action = drl_agent.select_action(mask, state, policy_net)
        reward = env.take_action(action.item())
        next_state, next_mask = env.get_state()
        state, mask = next_state, next_mask

        if env.done():
            break

    print(f"cost: {env.get_total_cost()}")
    print(f"success rate: {env.get_success_rate()}")
    print(
        f"average response time: {sum(jobs_resp := env.get_jobs_response_time()) / len(jobs_resp)}"
    )
