import copy
import os
import sys
from itertools import count

import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import DRLAgent, EarliestAgent, RandomAgent, RoundRobinAgent
from drl import DQN, EpsilonGreedyStrategy
from environment import Environment
from utils import load_hyperparameters, send_system_message, set_seed

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
    num_episodes = hyperparameters["eval_episodes"]

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
    cost = {name: [] for name in agent_names}
    average_response_time = {name: [] for name in agent_names}

    for episode in tqdm(range(num_episodes)):
        env.reset()

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

            cost[name].append(env_eval.get_total_cost())
            jobs_resp = env_eval.get_jobs_response_time()
            average_response_time[name].append(sum(jobs_resp) / len(jobs_resp))

    writer.add_text("EvalResults/Cost", str(cost))
    writer.add_text("EvalResults/AverageResponseTime", str(average_response_time))

    plt.figure(figsize=(20, 5))
    plt.clf()
    plt.subplot(121)
    plt.title("Cost")
    for name, c in cost.items():
        plt.plot(c, label=name)
    plt.legend()
    plt.subplot(122)
    plt.title("Average Response Time")
    for name, art in average_response_time.items():
        plt.plot(art, label=name)
    plt.legend()
    plt.tight_layout()
    writer.add_figure("Evaluation", plt.gcf())

    writer.flush()
    writer.close()

    send_system_message("Evaluation finished!")
