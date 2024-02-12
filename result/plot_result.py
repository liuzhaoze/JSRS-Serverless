import os

import plotly.graph_objects as go

BAR_COLORS = [
    "rgb(55, 83, 109)",
    "rgb(26, 118, 255)",
    "rgb(50, 171, 96)",
    "rgb(255, 153, 51)",
]


def get_varying_variables_figure(
    variable_name: str,
    variable_values: list,
    results: dict,
    y_axis_label: str,
) -> go.Figure:
    fig = go.Figure()
    for (method_name, method_result), color in zip(results.items(), BAR_COLORS):
        fig.add_trace(
            go.Bar(
                x=variable_values,
                y=method_result,
                name=method_name,
                marker_color=color,
            )
        )
    fig.update_layout(
        xaxis=dict(
            title=variable_name,
            titlefont_size=16,
            tickfont_size=14,
            linecolor="black",  # 设置x轴线条颜色为黑色
            mirror=True,  # 设置x轴线条在图表两侧都显示
        ),
        yaxis=dict(
            title=y_axis_label,
            titlefont_size=16,
            tickfont_size=14,
            linecolor="black",  # 设置y轴线条颜色为黑色
            mirror=True,  # 设置y轴线条在图表两侧都显示
            gridcolor="grey",  # 设置网格线颜色为灰色
            griddash="dash",  # 设置网格线为虚线
        ),
        legend=dict(
            x=0.5,
            y=0.99,
            xanchor="center",
            yanchor="top",
            orientation="h",  # 设置图例为水平排列
            bgcolor="rgba(255, 255, 255, 255)",
        ),
        barmode="group",
        bargap=0.15,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.
        plot_bgcolor="white",  # 设置图表背景颜色为白色
        font=dict(color="black"),  # 设置所有文字颜色为黑色
        autosize=False,
        width=800,
        height=500,
        margin=dict(l=10, r=10, b=10, t=10),
    )
    return fig


if __name__ == "__main__":
    img_dir = r"D:\Rocco\Documents\研究生\论文\ADRL-basedTaskSchedulingMethodinServerlessComputing\asset\image\result"
    SHOW = False
    EXPORT = True

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Directory {img_dir} not found")

    fig = get_varying_variables_figure(
        "Arrival Rate",
        [10, 15, 20, 25, 30],
        {
            "Random": [
                0.25970610649605674,
                0.28324567491926506,
                0.31289668430483386,
                0.35707228367946253,
                0.40525407830042437,
            ],
            "Round-Robin": [
                0.22278230443713126,
                0.2365622606844542,
                0.25057547753487486,
                0.29212053615079603,
                0.35390688935286124,
            ],
            "Earliest": [
                0.30415678331879537,
                0.28921966420209216,
                0.2773791858507039,
                0.26702259293682257,
                0.2591526094768028,
            ],
            "DQN": [
                0.18728502646243636,
                0.18472498001459214,
                0.18488470474406657,
                0.2063907451660665,
                0.2037857366410525,
            ],
        },
        "Average Response Time",
    )
    if SHOW:
        fig.show()
    if EXPORT:
        fig.write_image(os.path.join(img_dir, "arrival_rate_response_time.pdf"))

    fig = get_varying_variables_figure(
        "Arrival Rate",
        [10, 15, 20, 25, 30],
        {
            "Random": [
                21.077708000000012,
                17.55953466666666,
                14.803667999999998,
                13.186802666666667,
                11.739380000000002,
            ],
            "Round-Robin": [
                23.984537333333346,
                18.948769333333324,
                15.712421333333332,
                13.588942666666666,
                12.011686666666666,
            ],
            "Earliest": [
                16.357230666666663,
                15.257297333333328,
                14.208547999999999,
                13.049668,
                12.017688,
            ],
            "DQN": [
                16.201234666666668,
                13.709169333333339,
                12.51984,
                11.662792,
                10.207145333333331,
            ],
        },
        "Overall Cost",
    )
    if SHOW:
        fig.show()
    if EXPORT:
        fig.write_image(os.path.join(img_dir, "arrival_rate_overall_cost.pdf"))

    fig = get_varying_variables_figure(
        "Average Job Length",
        [0.1, 0.15, 0.2, 0.25, 0.3],
        {
            "Random": [
                0.1331125857541081,
                0.21084784424558264,
                0.31289668430483386,
                0.44224714565404705,
                0.6015563250401695,
            ],
            "Round-Robin": [
                0.11360275188376387,
                0.17683243113671168,
                0.25057547753487486,
                0.3609756648431569,
                0.552122952661575,
            ],
            "Earliest": [
                0.15275186752003106,
                0.21716027995623913,
                0.2773791858507039,
                0.33334640698793405,
                0.38940484945587567,
            ],
            "DQN": [
                0.09073132364007455,
                0.14133031093943035,
                0.18488470474406657,
                0.2814704051067922,
                0.32174975856295623,
            ],
        },
        "Average Response Time",
    )
    if SHOW:
        fig.show()
    if EXPORT:
        fig.write_image(os.path.join(img_dir, "average_job_length_response_time.pdf"))

    fig = get_varying_variables_figure(
        "Average Job Length",
        [0.1, 0.15, 0.2, 0.25, 0.3],
        {
            "Random": [
                14.141828,
                14.343634666666668,
                14.803667999999998,
                15.778263999999998,
                16.47078933333333,
            ],
            "Round-Robin": [
                15.023621333333331,
                15.280944,
                15.712421333333332,
                16.253830666666666,
                17.094682666666664,
            ],
            "Earliest": [
                10.527963999999997,
                12.59979333333333,
                14.208547999999999,
                15.718770666666666,
                16.868056,
            ],
            "DQN": [
                8.108601333333333,
                10.916973333333333,
                12.51984,
                13.540614666666668,
                15.297020000000002,
            ],
        },
        "Overall Cost",
    )
    if SHOW:
        fig.show()
    if EXPORT:
        fig.write_image(os.path.join(img_dir, "average_job_length_overall_cost.pdf"))

    fig = get_varying_variables_figure(
        "Proportion of Moldable Jobs",
        [0.1, 0.3, 0.5, 0.7, 0.9],
        {
            "Random": [
                0.32982782283489437,
                0.30694970009516004,
                0.31289668430483386,
                0.3315797958907032,
                0.38745515466929364,
            ],
            "Round-Robin": [
                0.25737366011124707,
                0.2492552567955059,
                0.25057547753487486,
                0.2715504141830538,
                0.2838878529750248,
            ],
            "Earliest": [
                0.22584384720680564,
                0.25999758762309916,
                0.2773791858507039,
                0.28676149456758,
                0.29383998599991024,
            ],
            "DQN": [
                0.22788508082330788,
                0.20249209331556284,
                0.18488470474406657,
                0.17628642597681757,
                0.24470080545346876,
            ],
        },
        "Average Response Time",
    )
    if SHOW:
        fig.show()
    if EXPORT:
        fig.write_image(
            os.path.join(img_dir, "proportion_of_moldable_jobs_response_time.pdf")
        )

    fig = get_varying_variables_figure(
        "Proportion of Moldable Jobs",
        [0.1, 0.3, 0.5, 0.7, 0.9],
        {
            "Random": [
                15.075345333333335,
                14.867366666666666,
                14.803667999999998,
                14.68812,
                14.67484,
            ],
            "Round-Robin": [
                14.561164,
                15.009564000000001,
                15.712421333333332,
                16.375639999999997,
                16.628841333333334,
            ],
            "Earliest": [
                14.827058666666668,
                14.637795999999996,
                14.208547999999999,
                13.376022666666666,
                12.617365333333332,
            ],
            "DQN": [
                13.616592,
                13.099378666666665,
                12.51984,
                11.243950666666667,
                11.377046666666667,
            ],
        },
        "Overall Cost",
    )
    if SHOW:
        fig.show()
    if EXPORT:
        fig.write_image(
            os.path.join(img_dir, "proportion_of_moldable_jobs_overall_cost.pdf")
        )
