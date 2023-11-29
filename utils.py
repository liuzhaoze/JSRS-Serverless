import os
import random

import numpy as np
import torch
import ujson
import yaml


def load_data():
    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data")

    with open(os.path.join(data_path, "on-demand.json"), "r", encoding="utf8") as f:
        on_demand = ujson.load(f)
    with open(os.path.join(data_path, "spot.json"), "r", encoding="utf8") as f:
        spot = ujson.load(f)

    return on_demand, spot


def load_hyperparameters():
    cwd = os.getcwd()
    config_path = os.path.join(cwd, "config")

    with open(
        os.path.join(config_path, "hyperparameters.yml"), "r", encoding="utf8"
    ) as f:
        hyperparameters = yaml.load(f, Loader=yaml.FullLoader)

    return hyperparameters


def set_seed(reproducibility: bool, seed: int):
    if reproducibility:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        return seed

    return None


def draw_gantt_chart(chart_name: str, workload: list) -> None:
    import plotly.graph_objs as go

    traces = []

    for job_id, job in enumerate(workload):
        time_stamp = []
        instance = []
        for index, record in job.exec_history.iterrows():
            time_stamp.append(record["submit_time"])
            instance.append(record["instance_id"])
            time_stamp.append(None)
            instance.append(None)
            time_stamp.append(record["start_time"])
            instance.append(record["instance_id"])
            time_stamp.append(record["end_time"])
            instance.append(record["instance_id"])
            time_stamp.append(None)
            instance.append(None)

        trace = go.Scatter(
            x=time_stamp, y=instance, mode="lines+markers", name=f"Job {job_id}"
        )
        traces.append(trace)

    fig = go.Figure(data=traces)
    fig.update_layout(title=chart_name, xaxis_title="Time", yaxis_title="Instance ID")
    fig.show()


def send_system_message(message: str):
    from windows_toasts import Toast, WindowsToaster

    toaster = WindowsToaster("Python")
    newToast = Toast()
    newToast.text_fields = [message, message]
    toaster.show_toast(newToast)


if __name__ == "__main__":
    print(load_hyperparameters())
