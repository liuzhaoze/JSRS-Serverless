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
        SLOT = 60.0

        W_RESP = 1.0
        W_COST = 5.0
        W_RISK = 1e6

        pending = set(range(len(jobs)))
        current_time = 0.0

        def estimate_cost(job: Job, inst: Instance, now: float) -> float:
            est_start = max(job.arrival_time, now, inst.idle_time)

            dtr_time = job.data_size / 1_000_000.0 / dts[job.region, inst.region]
            dtr_cost = job.data_size / 1_000_000_000.0 * dtc[job.region, inst.region]

            exec_time = job.length * job.parallelism / SU(job.parallelism, 0.0, inst.cpu)
            est_finish = est_start + dtr_time + exec_time

            if now >= inst.expired_time:
                est_expire = now + 3600.0
            else:
                est_expire = inst.expired_time

            resp = (est_start - job.arrival_time) + dtr_time + exec_time

            overdue = max(0.0, est_finish - est_expire)
            risk_penalty = overdue * W_RISK

            return W_RESP * resp + W_COST * dtr_cost + risk_penalty

        while pending:
            ready_jobs = [jid for jid in pending if jobs[jid].arrival_time <= current_time]

            if not ready_jobs:
                next_arrival = min(jobs[jid].arrival_time for jid in pending)
                current_time = max(current_time, next_arrival)
                ready_jobs = [jid for jid in pending if jobs[jid].arrival_time <= current_time]

            if not ready_jobs:
                break

            triples = []
            for jid in ready_jobs:
                for iid, inst in enumerate(instances):
                    c = estimate_cost(jobs[jid], inst, current_time)
                    triples.append((c, jid, iid))

            triples.sort(key=lambda x: x[0])
            chosen = []
            used_jobs = set()
            used_insts = set()
            target = min(len(ready_jobs), len(instances))
            for c, jid, iid in triples:
                if jid in used_jobs or iid in used_insts:
                    continue
                chosen.append((jid, iid))
                used_jobs.add(jid)
                used_insts.add(iid)
                if len(chosen) >= target:
                    break

            if not chosen:
                candidate_times = []
                for inst in instances:
                    if inst.idle_time > current_time:
                        candidate_times.append(inst.idle_time)
                for jid in pending:
                    if jobs[jid].arrival_time > current_time:
                        candidate_times.append(jobs[jid].arrival_time)
                if candidate_times:
                    current_time = min(candidate_times)
                else:
                    current_time += SLOT
                continue

            completed_this_round = []
            for job_id, instance_id in chosen:
                arrival_time = jobs[job_id].arrival_time
                scheduling_time = max(arrival_time, current_time)

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

                if success:
                    completed_this_round.append(job_id)

            for jid in completed_this_round:
                if jid in pending:
                    pending.remove(jid)

            next_candidates = []
            if pending:
                next_arrivals = [jobs[jid].arrival_time for jid in pending if jobs[jid].arrival_time > current_time]
                if next_arrivals:
                    next_candidates.append(min(next_arrivals))
            next_idles = [inst.idle_time for inst in instances if inst.idle_time > current_time]
            if next_idles:
                next_candidates.append(min(next_idles))
            if next_candidates:
                current_time = min(next_candidates)
            else:
                current_time += SLOT
        # ---

        with open(f"{args.log_dir}/env_log_{secrets.token_hex(3)}.json", "w") as f:
            f.write("[\n" + file_buffer.getvalue()[:-2] + "\n]\n")

    with open(f"{args.log_dir}/eval_result.pkl", "wb") as f:
        pickle.dump(CollectStats(returns=[], returns_stat={}, lens=[], lens_stat={}), f)
