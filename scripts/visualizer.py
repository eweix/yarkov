import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("sims.csv")


def create_dashboard(df, output_file="lineage_sims.html"):
    moments = ["mean", "variance", "skew", "kurtosis"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    colorscales = ["RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r"]

    n_samples = df["sample_id"].nunique()
    n_gens = df["gen"].max() + 1

    unique_samples = df.groupby("sample_id").first().reset_index()
    k_syn_vals = unique_samples["k_syn"].values
    M_crit_vals = unique_samples["M_crit"].values
    a_vals = unique_samples["a"].values

    fig = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=(
            "Mean: Time Series",
            "Mean: M_crit vs. k_syn",
            "Mean: M_crit vs. a",
            "Mean: k_syn vs. a",
            "Variance: Time Series",
            "Variance: M_crit vs. k_syn",
            "Variance: M_crit vs. a",
            "Variance: k_syn vs. a",
            "Skewness: Time Series",
            "Skewness: M_crit vs. k_syn",
            "Skewness: M_crit vs. a",
            "Skewness: k_syn vs. a",
            "Kurtosis: Time Series",
            "Kurtosis: M_crit vs. k_syn",
            "Kurtosis: M_crit vs. a",
            "Kurtosis: k_syn vs. a",
        ),
        specs=[
            [
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
            [
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
            [
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
            [
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
                {"type": "scatter"},
            ],
        ],
        horizontal_spacing=0.05,
        vertical_spacing=0.06,
    )

    ts_per_moment = n_samples
    param_per_gen = len(moments) * 3  # 3 param spaces per moment
    param_start_idx = len(moments) * ts_per_moment

    for mi, moment in enumerate(moments):
        for sid in range(n_samples):
            sample_df = df[df["sample_id"] == sid].sort_values("gen")
            fig.add_trace(
                go.Scatter(
                    x=sample_df["gen"].values,
                    y=sample_df[moment].values,
                    mode="lines",
                    line=dict(color=colors[mi], width=1),
                    opacity=0.35,
                    hoverinfo="text",
                    text=[
                        f"sid={sid}<br>k_syn={k_syn_vals[sid]:.3f}<br>M_crit={M_crit_vals[sid]:.1f}<br>a={a_vals[sid]:.2f}<br>{moment}={v:.2f}"
                        for v in sample_df[moment].values
                    ],
                    name=f"{moment}_ts_{sid}",
                    showlegend=False,
                    legendgroup=moment,
                ),
                row=mi + 1,
                col=1,
            )

    # Colorbar positions for 3 param spaces per row
    colorbar_x = 1.02
    colorbar_len = 0.18

    for g in range(n_gens):
        gen_df = df[df["gen"] == g].sort_values("sample_id")
        for mi, moment in enumerate(moments):
            # Col 2: M_crit vs. k_syn
            param_idx = param_start_idx + g * param_per_gen + mi * 3
            fig.add_trace(
                go.Scatter(
                    x=gen_df["k_syn"].values,
                    y=gen_df["M_crit"].values,
                    mode="markers",
                    marker=dict(
                        color=gen_df[moment].values,
                        colorscale=colorscales[mi],
                        opacity=0.8,
                        colorbar=dict(
                            title=moment,
                            x=colorbar_x,
                            len=colorbar_len,
                            y=0.9 - mi * 0.22,
                        ),
                    ),
                    customdata=np.column_stack(
                        [
                            gen_df["sample_id"].values,
                            gen_df["k_syn"].values,
                            gen_df["M_crit"].values,
                            gen_df["a"].values,
                        ]
                    ),
                    hovertemplate="<b>sample_id: %{customdata[0]}</b><br>"
                    + "k_syn: %{customdata[1]:.3f}<br>"
                    + "M_crit: %{customdata[2]:.1f}<br>"
                    + "a: %{customdata[3]:.2f}<br>"
                    + f"{moment}: %{{marker.color:.2f}}<extra></extra>",
                    name=f"param_g{g}_{moment}_ksyn_mcrit",
                    showlegend=False,
                    legendgroup=moment,
                    visible=(g == 0),
                ),
                row=mi + 1,
                col=2,
            )

            # Col 3: M_crit vs. a
            param_idx = param_start_idx + g * param_per_gen + mi * 3 + 1
            fig.add_trace(
                go.Scatter(
                    x=gen_df["a"].values,
                    y=gen_df["M_crit"].values,
                    mode="markers",
                    marker=dict(
                        color=gen_df[moment].values,
                        colorscale=colorscales[mi],
                        opacity=0.8,
                    ),
                    customdata=np.column_stack(
                        [
                            gen_df["sample_id"].values,
                            gen_df["k_syn"].values,
                            gen_df["M_crit"].values,
                            gen_df["a"].values,
                        ]
                    ),
                    hovertemplate="<b>sample_id: %{customdata[0]}</b><br>"
                    + "k_syn: %{customdata[1]:.3f}<br>"
                    + "M_crit: %{customdata[2]:.1f}<br>"
                    + "a: %{customdata[3]:.2f}<br>"
                    + f"{moment}: %{{marker.color:.2f}}<extra></extra>",
                    name=f"param_g{g}_{moment}_a_mcrit",
                    showlegend=False,
                    legendgroup=moment,
                    visible=(g == 0),
                ),
                row=mi + 1,
                col=3,
            )

            # Col 4: k_syn vs. a
            param_idx = param_start_idx + g * param_per_gen + mi * 3 + 2
            fig.add_trace(
                go.Scatter(
                    x=gen_df["a"].values,
                    y=gen_df["k_syn"].values,
                    mode="markers",
                    marker=dict(
                        color=gen_df[moment].values,
                        colorscale=colorscales[mi],
                        opacity=0.8,
                    ),
                    customdata=np.column_stack(
                        [
                            gen_df["sample_id"].values,
                            gen_df["k_syn"].values,
                            gen_df["M_crit"].values,
                            gen_df["a"].values,
                        ]
                    ),
                    hovertemplate="<b>sample_id: %{customdata[0]}</b><br>"
                    + "k_syn: %{customdata[1]:.3f}<br>"
                    + "M_crit: %{customdata[2]:.1f}<br>"
                    + "a: %{customdata[3]:.2f}<br>"
                    + f"{moment}: %{{marker.color:.2f}}<extra></extra>",
                    name=f"param_g{g}_{moment}_a_ksyn",
                    showlegend=False,
                    legendgroup=moment,
                    visible=(g == 0),
                ),
                row=mi + 1,
                col=4,
            )

    def build_visibility(gen):
        vis = []
        # Time series traces: 4 moments * n_samples each
        for mi in range(len(moments)):
            vis.extend([True] * ts_per_moment)
        # Param traces: n_gens * 3 param spaces per moment
        for g in range(n_gens):
            for mi in range(len(moments)):
                for p in range(3):  # 3 param spaces per moment
                    vis.append(g == gen)
        return vis

    slider_steps = []
    for g in range(n_gens):
        step_dict = {"args": [{"visible": build_visibility(g)}], "label": str(g)}
        slider_steps.append(step_dict)

    fig.update_layout(
        title=dict(text="Inheritance Simulation Results", font=dict(size=16)),
        height=1400,
        width=1800,
        template="plotly_white",
        xaxis=dict(title="Generation"),
        yaxis=dict(title="Mean"),
        xaxis2=dict(title="k_syn"),
        yaxis2=dict(title="M_crit"),
        xaxis3=dict(title="a"),
        yaxis3=dict(title="M_crit"),
        xaxis4=dict(title="a"),
        yaxis4=dict(title="k_syn"),
        xaxis5=dict(title="Generation"),
        yaxis5=dict(title="Variance"),
        xaxis6=dict(title="k_syn"),
        yaxis6=dict(title="M_crit"),
        xaxis7=dict(title="a"),
        yaxis7=dict(title="M_crit"),
        xaxis8=dict(title="a"),
        yaxis8=dict(title="k_syn"),
        xaxis9=dict(title="Generation"),
        yaxis9=dict(title="Skewness"),
        xaxis10=dict(title="k_syn"),
        yaxis10=dict(title="M_crit"),
        xaxis11=dict(title="a"),
        yaxis11=dict(title="M_crit"),
        xaxis12=dict(title="a"),
        yaxis12=dict(title="k_syn"),
        xaxis13=dict(title="Generation"),
        yaxis13=dict(title="Kurtosis"),
        xaxis14=dict(title="k_syn"),
        yaxis14=dict(title="M_crit"),
        xaxis15=dict(title="a"),
        yaxis15=dict(title="M_crit"),
        xaxis16=dict(title="a"),
        yaxis16=dict(title="k_syn"),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Generation: "),
                pad=dict(t=45),
                steps=slider_steps,
                x=0.1,
                len=0.8,
                xanchor="left",
                y=-0.02,
                yanchor="top",
            )
        ],
    )

    fig.write_html(output_file, include_plotlyjs="cdn", full_html=True)
    print(f"Dashboard saved to {output_file}")


if __name__ == "__main__":
    create_dashboard(df)
