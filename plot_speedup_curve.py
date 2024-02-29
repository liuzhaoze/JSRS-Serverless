import os

import numpy as np
import plotly.graph_objects as go

from environment.speedup_model import SpeedupModel

SHOW = False
EXPORT = True
img_dir = r"D:\Rocco\Documents\研究生\论文\ADRL-basedTaskSchedulingMethodinServerlessComputing\asset\image"

if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Directory {img_dir} not found")

fig = go.Figure()
x = np.linspace(1, 160, 160)
sigma_list = [
    0,
    0.5,
    1,
    2,
    2**31 - 1,
]

for sigma in sigma_list:
    SpeedupModel.A = 64
    SpeedupModel.sigma = sigma
    y = [SpeedupModel.SU(i) for i in x]
    if sigma == 2**31 - 1:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"sigma=inf"))
    else:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"sigma={sigma}"))

fig.update_layout(
    xaxis=dict(
        title="n (Number of Processors)",
        titlefont_size=20,
        tickfont_size=16,
        linecolor="black",  # 设置x轴线条颜色为黑色
        mirror=True,  # 设置x轴线条在图表两侧都显示
        range=[0, 160],
        tickvals=[0, 32, 64, 96, 128, 160],
        showticklabels=True,
        ticks="outside",
    ),
    yaxis=dict(
        title="SU(n)",
        titlefont_size=20,
        tickfont_size=16,
        linecolor="black",  # 设置y轴线条颜色为黑色
        mirror=True,  # 设置y轴线条在图表两侧都显示
        gridcolor="grey",  # 设置网格线颜色为灰色
        griddash="dash",  # 设置网格线为虚线
        range=[0, 80],
        tickvals=[0, 16, 32, 48, 64, 80],
    ),
    legend=dict(
        x=1,
        y=0,
        xanchor="right",
        yanchor="bottom",
        font=dict(size=14),  # 设置图例文字大小
    ),
    plot_bgcolor="white",  # 设置图表背景颜色为白色
    font=dict(color="black"),  # 设置所有文字颜色为黑色
    autosize=False,
    width=600,
    height=450,
    margin=dict(l=30, r=30, b=30, t=30),
)

if SHOW:
    fig.show()
if EXPORT:
    fig.write_image(os.path.join(img_dir, "speedup_curve.pdf"))
