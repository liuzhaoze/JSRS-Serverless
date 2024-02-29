import os
from itertools import cycle

import plotly.graph_objs as go
import ujson
from plotly.subplots import make_subplots

# Settings
SAVE = False
IMG_DIR = r"D:\Rocco\Documents\研究生\论文\ADRL-basedTaskSchedulingMethodinServerlessComputing\asset\image\result"
DATA_PATH = os.path.join(os.path.dirname(__file__), "result.json")

BAR_COLORS = cycle(
    [
        "rgb(55, 83, 109)",
        "rgb(26, 118, 255)",
        "rgb(50, 171, 96)",
        "rgb(255, 153, 51)",
    ]
)
DISPLAY = {
    "arrival_rate": "Arrival Rate",
    "average_job_length": "Average Job Length",
    "moldable_job_proportion": "Moldable Job Proportion",
    "average_response_time": "Average Response Time",
    "overall_cost": "Overall Cost",
}

if not os.path.exists(IMG_DIR):
    raise FileNotFoundError(f"Directory {IMG_DIR} not found")

with open(DATA_PATH, "r") as f:
    data = ujson.load(f)

for key, value in data.items():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)
    for method, result in value["average_response_time"].items():
        fig.add_trace(
            go.Bar(
                x=value["ticks"],
                y=result,
                name=method,
                marker=dict(color=next(BAR_COLORS)),
            ),
            row=1,
            col=1,
        )
    for method, result in value["overall_cost"].items():
        fig.add_trace(
            go.Bar(
                x=value["ticks"],
                y=result,
                name=method,
                marker=dict(color=next(BAR_COLORS)),
                showlegend=False,  # 第二个子图不显示图例
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text=DISPLAY[key], row=1, col=1)
    fig.update_xaxes(title_text=DISPLAY[key], row=1, col=2)
    fig.update_yaxes(
        title_text=DISPLAY["average_response_time"],
        title_standoff=15,
        ticks="outside",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text=DISPLAY["overall_cost"],
        title_standoff=0,
        ticks="outside",
        row=1,
        col=2,
    )
    fig.update_layout(
        font=dict(color="black", size=20),
        legend=dict(
            orientation="h",
            xanchor="center",
            yanchor="top",
            x=0.45,
            y=1.2,
            bgcolor="#EDEDED",
            bordercolor="Silver",
            borderwidth=3,
        ),
        plot_bgcolor="#EDEDED",
        autosize=False,
        width=1000,
        height=500,
        margin=dict(l=10, r=10, b=10, t=10, pad=0),
    )
    fig.show()
    if SAVE:
        fig.write_image(os.path.join(IMG_DIR, key + ".pdf"))
