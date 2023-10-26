from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
    hyperparameters = load_hyperparameters()
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
    writer = SummaryWriter()
    env = Environment(device)
    env.reset()  # 必须 reset 之后才能加载实例配置，env.action_dim() 才能返回正确的值
    epsilon_greedy = EpsilonGreedyStrategy(epsilon_start, epsilon_end, epsilon_decay)
    drl_agent = DRLAgent(epsilon_greedy, env.action_dim(), device)
    memory = ReplayMemory(replay_memory_size)

    policy_net = DQN(env.state_dim(), env.action_dim()).to(device)
    target_net = DQN(env.state_dim(), env.action_dim()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)

    global_step = 0

    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()

        return_per_episode = 0.0

        for step in count():
            action = drl_agent.select_action(None, state, policy_net)
            reward = env.take_action(action.item())
            next_state = env.get_state()
            memory.push(Experience(state, action, reward, next_state))
            state = next_state

            writer.add_scalar("Step Track/reward", reward, global_step)
            return_per_episode += reward.item()

            if memory.can_provide_sample(batch_size, dqn_update_threshold):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
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

        if episode % target_update == 0:
            # 更新 target_net 参数
            target_net.load_state_dict(policy_net.state_dict())
