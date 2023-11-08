from itertools import count

import matplotlib.pyplot as plt
import torch

from environment import (
    BillingType,
    Environment,
    Instance,
    InstanceType,
    Job,
    JobType,
    Zone,
)
from environment.speedup_model import SpeedupModel
from utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################################

# print(Instance.suspend_speed)
# print(Instance.load_speed)
# print(Instance.rental_time)

# instance = Instance(InstanceType.micro, Zone.us_west_1, BillingType.spot)
# print(instance)
# print(instance.rent(0))
# print(instance.rent(0))
# print(instance)
# instance.reset()
# print(instance)

# j = Job(0, 1, 100, JobType.rigid, 0, 1)
# print(j)
# for i in range(10):
#     j.add_history(i, i + 1, i + 2, i + 3, True)
# print(j)
# print(j.get_history())

###########################################################################

# sigma_list = [0, 0.5, 1, 2, 100]
# for sigma in sigma_list:
#     SpeedupModel.sigma = sigma
#     print(SpeedupModel.A, SpeedupModel.sigma)
#     x = list(range(1, 160))
#     y = [SpeedupModel.SU(i) for i in x]
#     plt.plot(x, y)
# plt.show()

###########################################################################

# 任务分配测试
# i = Instance(InstanceType.large, Zone.us_west_1, BillingType.spot)
# j = Job(0, 4, 4096.0, JobType.moldable, 1, 0.5)
# print(i, j)

# # 任务在实例到期之后提交，且任务部分完成
# ir, jr, r = Environment.assign(i, j)
# print(ir, jr, r)
# print(jr.finished())

# # 任务在空闲时刻之前提交，且任务部分完成
# ir, jr, r = Environment.assign(ir, jr)
# print(ir, jr, r)
# print(jr.finished())

# j = Job(0, 1, 4096.0, JobType.rigid, 2.5, 1)
# j.last_zone = Zone.us_west_2

# # 任务在空闲时刻之后，到期之前提交，且任务部分完成，跨区域调度
# ir, jr, r = Environment.assign(ir, j)
# print(ir, jr, r)
# print(jr.finished())

# # 任务在空闲时刻之后，到期之前提交，且任务全部完成
# ir.rent(0)
# ir, jr, r = Environment.assign(ir, jr)
# print(ir, jr, r)
# print(jr.finished())

###########################################################################

# 环境测试（使用一个 xxlarge 实例，两个任务，任务最长0.1，保证能够两次调度完成）
# env = Environment(device)
# env.reset()
# print(env.instances_info(), env.jobs_info(), env.queue_info())

# for step in count():
#     print(state := env.get_state())
#     action = 0
#     reward = env.take_action(action)
#     print(next_state := env.get_state())

#     if env.done():
#         break

# print(env.jobs[0].get_history())

###########################################################################

# 环境测试（使用一个 xxlarge 实例，以个任务，任务最短3，测试多次调度）
# env = Environment(device)
# env.reset()
# print(env.instances_info(), env.jobs_info(), env.queue_info())

# for step in count():
#     print(state := env.get_state())
#     action = 0
#     reward = env.take_action(action)
#     print(next_state := env.get_state())
#     print(env.queue_info())
#     if env.done():
#         break

# print(env.jobs[0].get_history())
# print(env.jobs[0].total_response_time())
# print(env.jobs[0].total_execution_time())
# print(len(env.get_jobs_response_time()))

###########################################################################

# from drl import EpsilonGreedyStrategy

# strategy = EpsilonGreedyStrategy(1, 0.01, 0.001)
# value = []
# for step in range(5000):
#     value.append(strategy.get_epsilon(step))
# plt.plot(value)
# plt.show()

###########################################################################

# j = Job(0, 2, 16384.0, JobType.rigid, 2.5, 1)
# i = Instance(InstanceType.large, Zone.us_west_1, BillingType.spot)
# print(Environment.invalid(i, j))

###########################################################################

# import random

# env = Environment(True, device)
# env.reset()
# s, m = env.get_state()
# print(s)
# print(m)
# t = torch.tensor([1, 2, 3], device=device)
# print(t.where(m, float("-inf")))

# wait_time = s[4:].where(m, float("inf"))
# print(wait_time)
# print(wait_time.unsqueeze(dim=0).argmin(dim=1))

# t = torch.tensor([5, 6, 6], device=device)
# print(t.unsqueeze(dim=0).argmin(dim=1))

###########################################################################

# import datetime

# # 获取当前日期和时间
# now = datetime.datetime.now()

# # 将日期和时间格式化为Oct27_15-50-42的形式
# formatted = now.strftime("%b%d_%H-%M-%S")

# print("Formatted date and time:", formatted)

###########################################################################

# baseline Agent 测试

# from agent import EarliestAgent, RandomAgent, RoundRobinAgent

# random = RandomAgent(3, device)
# round_robin = RoundRobinAgent(3, device)
# earliest = EarliestAgent(3, device)

# print(random.select_action(torch.tensor([True, True, True], device=device)))
# for _ in range(10):
#     print(round_robin.select_action())
# print(
#     earliest.select_action(
#         torch.tensor([False, True, False], device=device),
#         torch.tensor([1, 2, 3, 4, 8, 10, 7], device=device),
#     )
# )

###########################################################################

# 深拷贝环境测试

# import copy

# set_seed(True, 0)
# env = Environment(True, device)
# env.reset()
# print(env.jobs_info())
# print(env.instances_info())
# print(env.queue_info())

# another_env = copy.deepcopy(env)
# print(another_env.jobs_info())
# print(another_env.instances_info())
# print(another_env.queue_info())

# env.take_action(0)
# print(env.jobs_info())
# print(env.instances_info())
# print(env.queue_info())
# print(env.get_success_rate())

# another_env.take_action(3)
# print(another_env.jobs_info())
# print(another_env.instances_info())
# print(another_env.queue_info())
# print(another_env.get_success_rate())
