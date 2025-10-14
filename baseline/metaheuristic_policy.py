from pathlib import Path

current_dir = Path(__file__).parent

import sys

sys.path.append(str(current_dir.parent))

import argparse
import io
import json
import os
import pickle
import secrets
import time

import numpy as np
import yaml
from tianshou.data import CollectStats

from cloudcomputing_env.envs.config import ClusterConfig, WorkloadConfig
from cloudcomputing_env.envs.instance import Instance
from cloudcomputing_env.envs.job import Job
from cloudcomputing_env.envs.speedup_model import SU


def get_workload() -> list[Job]:
    with open(current_dir.parent / "config" / "workload.yaml") as f:
        workload_config = WorkloadConfig(**yaml.safe_load(f))
    # fmt: off
    arrival_intervals = np.random.exponential(1.0 / workload_config.arrival_rate, workload_config.number)
    arrival_times = np.cumsum(arrival_intervals)
    lengths = np.random.normal(workload_config.average_length, 0.1 * workload_config.average_length, workload_config.number)
    parallelisms = np.random.choice([1, 2, 4, 8], workload_config.number, p=[0.6, 0.25, 0.1, 0.05])
    regions = np.random.randint(0, workload_config.region_number, workload_config.number)
    data_sizes = np.random.normal(workload_config.average_data_size, 0.1 * workload_config.average_data_size, workload_config.number)
    # fmt: on
    workload = [
        Job(i, arrival_times[i], lengths[i], parallelisms[i], regions[i], int(data_sizes[i]))
        for i in range(workload_config.number)
    ]
    workload.sort(key=lambda job: job.arrival_time)
    return workload


def get_cluster() -> tuple[list[Instance], np.ndarray, np.ndarray]:
    with open(current_dir.parent / "config" / "cluster.yaml") as f:
        cluster_config = ClusterConfig(**yaml.safe_load(f))
    data_transfer_speed = np.array(cluster_config.data_transfer_speed)
    data_transfer_cost = np.array(cluster_config.data_transfer_cost)
    instances: list[Instance] = []
    for region_cfg in cluster_config.regions:
        for instance_cfg in region_cfg.instances:
            for _ in range(instance_cfg.count):
                instances.append(Instance(instance_cfg, region_cfg.region_code))
    return instances, data_transfer_speed, data_transfer_cost


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument("--log-dir", type=str, default=f"{current_dir.parent}/logs/{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--eval-episode", type=int, default=16, help="Number of episodes for evaluation")

    known_args, unknown_args = parser.parse_known_args()

    if len(unknown_args) > 0:
        print("Unknown arguments:", unknown_args)
        print("Please check the command line arguments.")

    return known_args


if __name__ == "__main__":
    args = get_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if len(os.listdir(args.log_dir)) != 0:
        raise ValueError(f"{args.log_dir} should be an empty directory.")

    for i in range(args.eval_episode):
        file_buffer = io.StringIO()
        jobs = get_workload()
        instances, dts, dtc = get_cluster()

        # ---
        import math
        import random
        from collections import defaultdict

        SLOT = 60.0

        slot_buckets = defaultdict(list)
        for j in jobs:
            slot_idx = int(j.arrival_time // SLOT)
            slot_buckets[slot_idx].append(j)

        def evaluate_assignment(slot_time, jobs_in_slot, assign_vec):
            sim_idle = np.array([max(it.idle_time, slot_time) for it in instances], dtype=float)
            sim_exp = np.array([it.expired_time for it in instances], dtype=float)
            total_cost = 0.0
            all_success = True

            order = sorted(range(len(jobs_in_slot)), key=lambda k: jobs_in_slot[k].arrival_time)

            for k in order:
                job = jobs_in_slot[k]
                inst_id = assign_vec[k]

                arrival_time = job.arrival_time
                scheduling_time = max(arrival_time, slot_time)

                if scheduling_time >= sim_exp[inst_id]:
                    total_cost += instances[inst_id].price
                    sim_exp[inst_id] = scheduling_time + 3600.0

                start_time = max(scheduling_time, sim_idle[inst_id])

                data_transfer_time = job.data_size / 1_000_000.0 / dts[job.region, instances[inst_id].region]
                data_transfer_cost = job.data_size / 1_000_000_000.0 * dtc[job.region, instances[inst_id].region]
                total_cost += data_transfer_cost

                parallelism = job.parallelism
                length = job.length
                execution_time = length * parallelism / SU(parallelism, 0.0, instances[inst_id].cpu)

                finish_time = start_time + data_transfer_time + execution_time

                if finish_time <= sim_exp[inst_id]:
                    sim_idle[inst_id] = finish_time
                    success = True
                else:
                    sim_idle[inst_id] = sim_exp[inst_id]
                    success = False
                    all_success = False

                total_cost += 1e-6 * (start_time - scheduling_time) + 0.0

            return total_cost, all_success

        def random_solution(n_jobs, n_insts):
            return [random.randrange(n_insts) for _ in range(n_jobs)]

        def solution_distance(a, b):
            return sum(1 for x, y in zip(a, b) if x != y)

        def move_towards(sol_i, sol_j, beta):
            new_sol = sol_i[:]
            for idx in range(len(sol_i)):
                if sol_i[idx] != sol_j[idx]:
                    if random.random() < beta:
                        new_sol[idx] = sol_j[idx]
            return new_sol

        def random_walk(sol, n_insts, alpha):
            new_sol = sol[:]
            for idx in range(len(sol)):
                if random.random() < alpha:
                    new_sol[idx] = random.randrange(n_insts)
            return new_sol

        # mFA 超参
        POP = 18  # 种群大小
        MAX_ITERS = 36  # 迭代次数
        BETA0 = 1.0  # 基础吸引力
        GAMMA = 0.3  # 光强衰减系数
        ALPHA = 0.25  # 初始随机游走强度
        ALPHA_DAMP = 0.90  # 衰减
        BIG_PENALTY = 1e12  # 失败重罚，逼迫可行

        for slot_idx in sorted(slot_buckets.keys()):
            jobs_in_slot = slot_buckets[slot_idx]
            if not jobs_in_slot:
                continue

            slot_time = slot_idx * SLOT
            n_jobs = len(jobs_in_slot)
            n_insts = len(instances)

            population = [random_solution(n_jobs, n_insts) for _ in range(POP)]
            fitness = []
            feasible = []
            for sol in population:
                cost, ok = evaluate_assignment(slot_time, jobs_in_slot, sol)
                fitness.append(cost + (0.0 if ok else BIG_PENALTY))
                feasible.append(ok)

            alpha = ALPHA
            for _it in range(MAX_ITERS):
                order = sorted(range(POP), key=lambda i: fitness[i])
                improved = False
                for i_idx in range(POP):
                    i = order[i_idx]
                    for j_idx in range(i_idx):
                        j = order[j_idx]
                        if fitness[j] >= fitness[i]:
                            continue
                        dist = solution_distance(population[i], population[j])
                        beta = BETA0 * math.exp(-GAMMA * (dist**2))
                        cand = move_towards(population[i], population[j], beta)
                        cand = random_walk(cand, n_insts, alpha)
                        cost, ok = evaluate_assignment(slot_time, jobs_in_slot, cand)
                        cand_fit = cost + (0.0 if ok else BIG_PENALTY)
                        if cand_fit < fitness[i]:
                            population[i] = cand
                            fitness[i] = cand_fit
                            feasible[i] = ok
                            improved = True
                alpha *= ALPHA_DAMP
                if not improved:
                    break

            best_idx = min(range(POP), key=lambda i: fitness[i])
            best_sol = population[best_idx]

            exec_order = sorted(range(n_jobs), key=lambda k: jobs_in_slot[k].arrival_time)
            for k in exec_order:
                job = jobs_in_slot[k]
                job_id = job.job_id
                instance_id = best_sol[k]

                arrival_time = jobs[job_id].arrival_time
                scheduling_time = max(arrival_time, slot_time)

                if scheduling_time >= instances[instance_id].expired_time:
                    rental_cost = instances[instance_id].rent(scheduling_time)
                else:
                    rental_cost = 0.0

                start_time = max(scheduling_time, instances[instance_id].idle_time)
                data_transfer_time = (
                    jobs[job_id].data_size
                    / 1_000_000  # Convert bytes to MB
                    / dts[jobs[job_id].region, instances[instance_id].region]
                )
                data_transfer_cost = (
                    jobs[job_id].data_size
                    / 1_000_000_000  # Convert bytes to GB
                    * dtc[jobs[job_id].region, instances[instance_id].region]
                )

                parallelism = jobs[job_id].parallelism
                length = jobs[job_id].length
                execution_time = length * parallelism / SU(parallelism, 0.0, instances[instance_id].cpu)

                finish_time = start_time + data_transfer_time + execution_time

                if finish_time <= instances[instance_id].expired_time:
                    success = True
                    instances[instance_id].idle_time = finish_time
                else:
                    success = False
                    instances[instance_id].idle_time = instances[instance_id].expired_time

                file_content = {
                    "arrival_time": round(arrival_time, 6),
                    "job_id": job_id,
                    "success": success,
                    "start_time": round(start_time, 6),
                    "finish_time": round(finish_time, 6),
                    "data_transfer_time": round(data_transfer_time, 6),
                    "execution_time": round(execution_time, 6),
                    "data_transfer_cost": round(data_transfer_cost, 6),
                    "rental_cost": round(rental_cost, 6),
                }
                render_file_content = json.dumps(file_content) + ",\n"
                file_buffer.write(render_file_content)
        # ---

        with open(f"{args.log_dir}/env_log_{secrets.token_hex(3)}.json", "w") as f:
            f.write("[\n" + file_buffer.getvalue()[:-2] + "\n]\n")

    with open(f"{args.log_dir}/eval_result.pkl", "wb") as f:
        pickle.dump(CollectStats(returns=[], returns_stat={}, lens=[], lens_stat={}), f)
