import argparse
import logging
import os
import pickle
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FlattenObservation
from tianshou.data import Collector, CollectStats, InfoStats, PrioritizedVectorReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.policy import BasePolicy, DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from torch.utils.tensorboard import SummaryWriter

from argument.run import RunArgument
from baseline import EarliestPolicy, RandomPolicy, RoundRobinPolicy

logger = logging.getLogger(__name__)


def get_args() -> RunArgument:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=f"./logs/{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--render-mode", type=str, default=None, help="Options: console, file.")

    # Parameters for workload
    parser.add_argument("--workload-config-path", type=str, default="./config/workload.yaml")

    # Parameters for cluster
    parser.add_argument("--cluster-config-path", type=str, default="./config/cluster.yaml")

    # Parameters for DRL
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256, 128, 64], help="Hidden sizes of DQN")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--td-step", type=int, default=9, help="Number of steps for multi-step TD learning")
    parser.add_argument("--target-update-freq", type=int, default=320, help="Frequency of updating the target network")
    parser.add_argument("--buffer-size", type=int, default=1e4, help="Size of the replay buffer")
    parser.add_argument("--prioritized-replay", action="store_true", help="Use prioritized replay buffer")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Alpha for prioritized replay buffer, indicating the degree of prioritization; 0 means uniform sampling",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.4,
        help="Beta for prioritized replay buffer, growing to 1 in the training process",
    )
    parser.add_argument("--beta-final", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training the model")
    parser.add_argument("--eps-begin", type=float, default=0.5, help="Exploration rate at the beginning of training")
    parser.add_argument("--eps-end", type=float, default=0.1, help="Exploration rate at the end of training")
    parser.add_argument("--eps-test", type=float, default=0.05, help="Exploration rate during testing")
    parser.add_argument("--anneal-start", type=int, default=1e5, help="Step to start annealing beta and epsilon")
    parser.add_argument("--anneal-end", type=int, default=1e6, help="Step to end annealing beta and epsilon")

    # Parameters for training
    parser.add_argument("--reward-lambda", type=float, default=2.0, help="Hyperparameter for reward function")
    parser.add_argument("--num-train-env", type=int, default=8, help="Number of training environments")
    parser.add_argument("--num-test-env", type=int, default=8, help="Number of testing environments")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs for training the model")
    parser.add_argument("--step-per-epoch", type=int, default=1000, help="Number of steps in each epoch")
    parser.add_argument(
        "--step-per-collect",
        type=int,
        default=8,
        help="Number of trajectories to collect in each collect operation (preferably an integer multiple of num-train-env)",
    )
    parser.add_argument(
        "--update-per-step",
        type=float,
        default=0.125,
        help="How many gradient steps to perform per environment step (generally, update-per-step * step-per-collect = 1; if update-per-step * step-per-collect > 1, it means that each collect operation will perform multiple gradient steps: `update-per-step * step-per-collect` steps specifically)",
    )
    parser.add_argument(
        "--episode-per-test",
        type=int,
        default=8,
        help="Number of episodes to run in each policy evaluation (preferably an integer multiple of num-test-env)",
    )

    # Parameters for evaluation
    parser.add_argument("--evaluation", action="store_true", help="Evaluate the model")
    parser.add_argument("--model-path", type=str, nargs="?", help="Path to the model to be evaluated")
    parser.add_argument("--num-eval-env", type=int, default=8, help="Number of evaluation environments")
    parser.add_argument("--eval-episode", type=int, default=16, help="Number of episodes for evaluation")
    parser.add_argument("--baseline", type=str, nargs="?", help="Options: random, roundrobin, earliest")

    known_args, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:
        print("Unknown arguments:", unknown_args)
        print("Please check the command line arguments.")

    return RunArgument(**vars(known_args))


def get_env(args: RunArgument) -> gym.Env:
    raw_env = gym.make("cloudcomputing_env/CloudComputing-v0", args=args, render_mode=args.render_mode)
    return FlattenObservation(raw_env)


def get_policy(
    args: RunArgument, policy: BasePolicy | None = None, optimizer: torch.optim.Optimizer | None = None
) -> tuple[BasePolicy, torch.optim.Optimizer]:
    env = get_env(args)

    observation_space = env.observation_space
    action_space = env.action_space

    obs_shape = observation_space.shape or int(observation_space.n)
    act_shape = action_space.shape or int(action_space.n)

    if policy is None:
        net = Net(
            state_shape=obs_shape,
            action_shape=act_shape,
            hidden_sizes=args.hidden_sizes,
            device=args.device,
        ).to(args.device)

        if optimizer is None:
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        policy: DQNPolicy = DQNPolicy(
            model=net,
            optim=optimizer,
            action_space=action_space,
            discount_factor=args.gamma,
            estimation_step=args.td_step,
            target_update_freq=args.target_update_freq,
        ).to(args.device)

        if args.evaluation:
            print(f"Loading model from {args.model_path}")
            parameters = torch.load(args.model_path, map_location=args.device)
            policy.load_state_dict(parameters)

    return policy, optimizer


def train_agent(
    args: RunArgument, policy: BasePolicy | None = None, optimizer: torch.optim.Optimizer | None = None
) -> tuple[InfoStats, BasePolicy]:
    # environment
    train_envs = SubprocVectorEnv([lambda: get_env(args) for _ in range(args.num_train_env)])
    test_envs = SubprocVectorEnv([lambda: get_env(args) for _ in range(args.num_test_env)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # policy
    policy, optimizer = get_policy(args, policy, optimizer)

    # buffer
    if args.prioritized_replay:
        buf = PrioritizedVectorReplayBuffer(
            total_size=args.buffer_size, buffer_num=len(train_envs), alpha=args.alpha, beta=args.beta
        )
    else:
        buf = VectorReplayBuffer(total_size=args.buffer_size, buffer_num=len(train_envs))

    # collector
    train_collector = Collector(policy, train_envs, buf, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    train_collector.reset()
    train_collector.collect(n_step=args.batch_size * args.num_train_env)  # pre-fill the buffer to train the model

    # log
    writer = SummaryWriter(args.log_dir)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # train
    def train_fn(num_epoch: int, step_idx: int) -> None:
        # epsilon decay
        if step_idx <= args.anneal_start:
            eps = args.eps_begin
        elif step_idx <= args.anneal_end:
            eps = args.eps_begin + (args.eps_end - args.eps_begin) * (step_idx - args.anneal_start) / (
                args.anneal_end - args.anneal_start
            )
        else:
            eps = args.eps_end
        policy.set_eps(eps)

        # beta decay
        if args.prioritized_replay:
            if step_idx <= args.anneal_start:
                beta = args.beta
            elif step_idx <= args.anneal_end:
                beta = args.beta + (args.beta_final - args.beta) * (step_idx - args.anneal_start) / (
                    args.anneal_end - args.anneal_start
                )
            else:
                beta = args.beta_final
            buf.set_beta(beta)

    def test_fn(num_epoch: int, step_idx: int) -> None:
        policy.set_eps(args.eps_test)

    def stop_fn(mean_rewards: float) -> bool:
        return False

    def save_best_fn(policy: BasePolicy) -> None:
        path = os.path.join(args.log_dir, "best.pth")
        torch.save(policy.state_dict(), path)
        print(f"Best model saved to {path}")

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        batch_size=args.batch_size,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        update_per_step=args.update_per_step,
        episode_per_test=args.episode_per_test,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    return result, policy


def evaluate_agent(
    args: RunArgument, policy: BasePolicy | None = None, optimizer: torch.optim.Optimizer | None = None
) -> CollectStats:
    # environment
    eval_envs = SubprocVectorEnv([lambda: get_env(args) for _ in range(args.num_eval_env)])

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    eval_envs.seed(args.seed)

    # policy
    policy, optimizer = get_policy(args, policy, optimizer)
    if isinstance(policy, DQNPolicy):
        policy.eval()

    # collector
    collector = Collector(policy, eval_envs, exploration_noise=False)
    result = collector.collect(n_episode=args.eval_episode, reset_before_collect=True)

    # log
    writer = SummaryWriter(args.log_dir)
    writer.add_text("args", str(args))

    return result


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, filename=os.path.join(args.log_dir, "log.log"), encoding="utf-8")
    logger.info(" ".join(sys.argv))

    print(f"Using device: {args.device}")

    for i in range(10, 0, -1):
        print("\r" + ("Training" if not args.evaluation else "Evaluation") + f" will start in {i} seconds ", end="", flush=True)  # fmt: skip
        time.sleep(1)
    print()

    if not args.evaluation:
        result, policy = train_agent(args)
    else:
        match args.baseline:
            case None:
                result = evaluate_agent(args)
            case "random":
                env = get_env(args)
                policy = RandomPolicy(action_space=env.action_space)
                result = evaluate_agent(args, policy)
            case "roundrobin":
                env = get_env(args)
                policy = RoundRobinPolicy(action_space=env.action_space)
                result = evaluate_agent(args, policy)
            case "earliest":
                env = get_env(args)
                policy = EarliestPolicy(action_space=env.action_space)
                result = evaluate_agent(args, policy)

    with open(os.path.join(args.log_dir, f"{'train' if not args.evaluation else 'eval'}_result.pkl"), "wb") as f:
        pickle.dump(result, f)
    result.pprint_asdict()
