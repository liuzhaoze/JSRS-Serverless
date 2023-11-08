import copy
import os
import sys
from itertools import count

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from agent import DRLAgent, EarliestAgent, RandomAgent, RoundRobinAgent
from drl import DQN, EpsilonGreedyStrategy
from environment import Environment
from utils import load_hyperparameters, set_seed

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("python eval.py <model_path>")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()

    hyperparameters = load_hyperparameters()
    reproducibility = hyperparameters["reproducibility_eval"]
    seed = hyperparameters["seed_eval"]
    writer.add_text("EvalSeeds", str(set_seed(reproducibility, seed)))
    use_mask = hyperparameters["use_mask"]

    env = Environment(use_mask, device)
    env.reset()

    # 加载模型参数
    model_path = os.path.abspath(sys.argv[1])
    policy_net = DQN(env.state_dim(), env.action_dim()).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()

    epsilon_greedy = EpsilonGreedyStrategy(0, 0, 0)
    drl_agent = DRLAgent(epsilon_greedy, env.action_dim(), device)
    random_agent = RandomAgent(env.action_dim(), device)
    rr_agent = RoundRobinAgent(env.action_dim(), device)
    earliest_agent = EarliestAgent(env.action_dim(), device)

    agent_names = ["drl", "random", "round_robin", "earliest"]
    agents = [drl_agent, random_agent, rr_agent, earliest_agent]
    cost = []
    average_response_time = []

    for name, agent in zip(agent_names, agents):
        # 开始评估
        env_eval = copy.deepcopy(env)  # 所有评估使用同一个环境
        state, mask = env_eval.get_state()

        for step in count():
            action = agent.select_action(mask, state, policy_net)
            reward = env_eval.take_action(action.item())
            next_state, next_mask = env_eval.get_state()
            state, mask = next_state, next_mask

            if env_eval.done():
                break

        print(f"agent: {name}")

        cost.append(c := env_eval.get_total_cost())
        print(f"cost: {c}")
        writer.add_text("EvalResults", f"{name} | cost: {c}")

        jobs_resp = env_eval.get_jobs_response_time()
        average_response_time.append(art := sum(jobs_resp) / len(jobs_resp))
        print(f"average response time: {art}")
        writer.add_text("EvalResults", f"{name} | average response time: {art}")

    plt.figure(figsize=(10, 5))
    plt.clf()
    plt.subplot(121)
    plt.title("Cost")
    plt.bar(agent_names, cost)
    plt.subplot(122)
    plt.title("Average Response Time")
    plt.bar(agent_names, average_response_time)
    plt.tight_layout()
    writer.add_figure("Evaluation", plt.gcf())

    writer.flush()
    writer.close()
