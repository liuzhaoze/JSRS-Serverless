import datetime
import os
import sys
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import DRLAgent
from drl import (
    DQN,
    EpsilonGreedyStrategy,
    Experience,
    QValues,
    ReplayMemory,
    extract_tensors,
)
from environment import Environment
from utils import load_hyperparameters

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter()

    hyperparameters = load_hyperparameters()
    use_mask = hyperparameters["use_mask"]
    batch_size = hyperparameters["batch_size"]
    gamma = hyperparameters["gamma"]
    epsilon_start = hyperparameters["epsilon_start"]
    epsilon_end = hyperparameters["epsilon_end"]
    epsilon_decay = hyperparameters["epsilon_decay"]
    target_update = hyperparameters["target_update"]
    replay_memory_size = hyperparameters["replay_memory_size"]
    dqn_update_threshold = hyperparameters["dqn_update_threshold"]
    learning_rate = hyperparameters["learning_rate"]
    num_episodes = hyperparameters["num_episodes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Environment(use_mask, device)
    env.reset()  # 必须 reset 之后才能加载实例配置，env.action_dim() 才能返回正确的值
    epsilon_greedy = EpsilonGreedyStrategy(epsilon_start, epsilon_end, epsilon_decay)
    drl_agent = DRLAgent(epsilon_greedy, env.action_dim(), device)
    memory = ReplayMemory(replay_memory_size)

    writer.add_graph(
        DQN(env.state_dim(), env.action_dim()).to(device), env.get_state()[0]
    )
    policy_net = DQN(env.state_dim(), env.action_dim()).to(device)
    target_net = DQN(env.state_dim(), env.action_dim()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)

    global_step = 0

    for episode in tqdm(range(num_episodes)):
        env.reset()
        state, mask = env.get_state()

        return_per_episode = 0.0

        for step in count():
            action = drl_agent.select_action(mask, state, policy_net)
            reward = env.take_action(action.item())
            next_state, next_mask = env.get_state()
            memory.push(Experience(state, action, reward, next_state, next_mask))
            state, mask = next_state, next_mask

            writer.add_scalar("Step Track/reward", reward, global_step)
            return_per_episode += reward.item()

            if memory.can_provide_sample(batch_size, dqn_update_threshold):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states, next_masks = extract_tensors(
                    experiences
                )

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states, next_masks)
                target_q_values = rewards + gamma * next_q_values

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Step Track/loss", loss, global_step)

            global_step += 1

            if env.done():
                break

        writer.add_scalar("Episode Track/return", return_per_episode, episode)
        writer.add_scalar("Episode Track/cost", env.get_total_cost(), episode)
        writer.add_scalar("Episode Track/success rate", env.get_success_rate(), episode)
        writer.add_scalar(
            "Episode Track/average response time",
            sum(jobs_resp := env.get_jobs_response_time()) / len(jobs_resp),
            episode,
        )

        if episode % target_update == 0:
            # 更新 target_net 参数
            target_net.load_state_dict(policy_net.state_dict())

    writer.flush()
    writer.close()

    # 保存模型
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    torch.save(
        policy_net.state_dict(),
        os.path.join(models_dir, f"{sys.argv[1] if len(sys.argv) == 2 else now}.pth"),
    )
    print(f"Model saved at: {models_dir}")
