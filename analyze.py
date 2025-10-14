import argparse
import os
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html
from tianshou.data.collector import CollectStats

from argument.analyze import AnalyzeArgument
from tools import filter_empty_files, load_logfile_from_dir, load_stats_from_dir


def get_args() -> AnalyzeArgument:
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--select", type=str, nargs="*", help="Select directories to analyze")

    return AnalyzeArgument(**vars(parser.parse_known_args()[0]))


def get_metrics(data: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    metrics = []
    for method, episodes in data.items():
        for i, ep in enumerate(episodes):
            total_jobs = ep.copy()
            total_jobs["response_time"] = total_jobs["finish_time"] - total_jobs["arrival_time"]
            total_jobs["waiting_time"] = total_jobs["start_time"] - total_jobs["arrival_time"]
            success_jobs = total_jobs.loc[total_jobs["success"], :]

            success_rate = len(success_jobs) / len(total_jobs) if not total_jobs.empty else 0.0

            if not success_jobs.empty:
                average_response_time = success_jobs["response_time"].mean()
                average_waiting_time = success_jobs["waiting_time"].mean()
                average_data_transfer_time = success_jobs["data_transfer_time"].mean()
                total_rental_cost = success_jobs["rental_cost"].sum()
                total_data_transfer_cost = success_jobs["data_transfer_cost"].sum()
            else:
                average_response_time = np.nan
                average_waiting_time = np.nan
                average_data_transfer_time = np.nan
                total_rental_cost = 0.0
                total_data_transfer_cost = 0.0

            metrics.append(
                {
                    "method": method,
                    "episode": i,
                    "success_rate": success_rate,
                    "average_response_time": average_response_time,
                    "average_waiting_time": average_waiting_time,
                    "average_data_transfer_time": average_data_transfer_time,
                    "total_rental_cost": total_rental_cost,
                    "total_data_transfer_cost": total_data_transfer_cost,
                }
            )

    df = pd.DataFrame(metrics)
    df["total_cost"] = df["total_rental_cost"] + df["total_data_transfer_cost"]
    return df


def stats_to_dataframe(stats: dict[str, CollectStats]) -> pd.DataFrame:
    data = []
    for method, value in stats.items():
        for index, ret in np.ndenumerate(value.returns):
            data.append({"return": ret, "episode": index[0], "method": method})
    return pd.DataFrame(data)


args = get_args()
if args.select is None:
    args.select = [
        Path(os.path.join(args.log_dir, d))
        for d in os.listdir(args.log_dir)
        if os.path.isdir(os.path.join(args.log_dir, d))
    ]

available_log_dirs = [d for d in args.select if filter_empty_files(d) != 0]
data = {d.name: load_logfile_from_dir(d) for d in available_log_dirs}
metrics = get_metrics(data)
stats = {d.name: load_stats_from_dir(d, "eval_result.pkl") for d in available_log_dirs}
stats_df = stats_to_dataframe(stats)

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("Environment Log Analysis Dashboard"),
        html.H2("Return"),
        dcc.Graph(figure=px.box(stats_df, x="method", y="return", points="all")),
        html.H2("Success Rate"),
        dcc.Graph(figure=px.box(metrics, x="method", y="success_rate", points="all")),
        html.H2("Average Response Time"),
        dcc.Graph(figure=px.box(metrics, x="method", y="average_response_time", points="all")),
        html.H2("Average Waiting Time"),
        dcc.Graph(figure=px.box(metrics, x="method", y="average_waiting_time", points="all")),
        html.H2("Average Data Transfer Time"),
        dcc.Graph(figure=px.box(metrics, x="method", y="average_data_transfer_time", points="all")),
        html.H2("Total Cost"),
        dcc.Graph(figure=px.box(metrics, x="method", y="total_cost", points="all")),
        html.H2("Total Rental Cost"),
        dcc.Graph(figure=px.box(metrics, x="method", y="total_rental_cost", points="all")),
        html.H2("Total Data Transfer Cost"),
        dcc.Graph(figure=px.box(metrics, x="method", y="total_data_transfer_cost", points="all")),
    ]
)


if __name__ == "__main__":
    app.run()
