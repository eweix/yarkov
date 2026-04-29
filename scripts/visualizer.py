import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

df = pd.read_csv("sims.csv")

moments = ["mean", "variance", "skew", "kurtosis"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
colorscales = ["RdBu_r", "RdBu_r", "RdBu_r", "RdBu_r"]

unique_samples = df.groupby("sample_id").first().reset_index()
k_syn_vals = unique_samples["k_syn"].values
M_crit_vals = unique_samples["M_crit"].values
a_vals = unique_samples["a"].values



def plot_cell_traces(df: pd.DataFrame, moment: str, mi: int, fig, row=0, col=0):
    for sid in range(df["sample_id"].nunique()):
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
            row=row,
            col=col,
        )
    return None


def plot_moment_params(df: pd.DataFrame, moment: str, fig, param1: str, param2: str, mi: int, param_idx: int, g: int, row=0, col=0, colorbar=False):
    # Colorbar positions for 3 param spaces per row
    colorbar_x = 1.02
    colorbar_len = 0.12

    if colorbar:
        cbar = dict(
            title=moment,
            title_font=dict(size=11),
            x=colorbar_x,
            y=0.95 - mi * 0.20,
            len=colorbar_len,
            thickness=15,
            tickfont=dict(size=10),
        )
    else:
        cbar = None
    
    fig.add_trace(
        go.Scatter(
            x=df[param1].values,
            y=df[param2].values,
            mode="markers",
            marker = dict(
                color=df[moment].values,
                colorscale=colorscales[mi],
                opacity=0.8,
                colorbar=cbar,
            ),
            customdata=np.column_stack(
                [
                    df["sample_id"].values,
                    df["k_syn"].values,
                    df["M_crit"].values,
                    df["a"].values,
                ]
            ),
            hovertemplate="<b>sample_id: %{customdata[0]}</b><br>"
            + "k_syn: %{customdata[1]:.3f}<br>"
            + "M_crit: %{customdata[2]:.1f}<br>"
            + "a: %{customdata[3]:.2f}<br>"
            + f"{moment}: %{{marker.color:.2f}}<extra></extra>",
            name=f"param_g{g}_{moment}_{param1}_{param2}",
            showlegend=False,
            legendgroup=moment,
            visible=(g == 0),
        ),
        row=row,
        col=col,
    )
    return None

def create_dashboard(df: pd.DataFrame, output_file="lineage_sims.html"):
    # Add new visualization for time-evolution of probability distribution
    
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
        vertical_spacing=0.10,
    )

    ts_per_moment = n_samples
    param_per_gen = len(moments) * 3  # 3 param spaces per moment
    param_start_idx = len(moments) * ts_per_moment

    for mi, moment in enumerate(moments):
        plot_cell_traces(df, moment, mi, fig, row=mi+1, col=1)

    for g in range(n_gens):
        gen_df = df[df["gen"] == g].sort_values("sample_id")
        for mi, moment in enumerate(moments):
            # Col 2: M_crit vs. k_syn
            param_idx = param_start_idx + g * param_per_gen + mi * 3
            plot_moment_params(gen_df, moment, fig, "k_syn", "M_crit", mi, param_idx, g, row=mi+1, col=2, colorbar=True)

            # Col 3: M_crit vs. a
            param_idx = param_start_idx + g * param_per_gen + mi * 3 + 1
            plot_moment_params(gen_df, moment, fig, "M_crit", "a", mi, param_idx, g, row=mi+1, col=3)

            # Col 4: k_syn vs. a
            param_idx = param_start_idx + g * param_per_gen + mi * 3 + 2
            plot_moment_params(gen_df, moment, fig, "k_syn", "a", mi, param_idx, g, row=mi+1, col=4)

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

    gen_title = dict(title="Generation")
    k_syn_title = dict(title="k_syn")
    a_title = dict(title="k_a")
    crit_title = dict(title="M_crit")
    fig.update_layout(
        title=dict(text="Inheritance Simulation Results", font=dict(size=16)),
        height=1400,
        width=1800,
        template="plotly_white",
        xaxis=gen_title,
        yaxis=dict(title="Mean"),
        xaxis2=k_syn_title,
        yaxis2=crit_title,
        xaxis3=a_title,
        yaxis3=crit_title,
        xaxis4=a_title,
        yaxis4=k_syn_title,
        xaxis5=gen_title,
        yaxis5=dict(title="Variance"),
        xaxis6=k_syn_title,
        yaxis6=crit_title,
        xaxis7=a_title,
        yaxis7=crit_title,
        xaxis8=a_title,
        yaxis8=k_syn_title,
        xaxis9=gen_title,
        yaxis9=dict(title="Skewness"),
        xaxis10=k_syn_title,
        yaxis10=crit_title,
        xaxis11=a_title,
        yaxis11=crit_title,
        xaxis12=a_title,
        yaxis12=k_syn_title,
        xaxis13=gen_title,
        yaxis13=dict(title="Kurtosis"),
        xaxis14=k_syn_title,
        yaxis14=crit_title,
        xaxis15=a_title,
        yaxis15=crit_title,
        xaxis16=a_title,
        yaxis16=k_syn_title,
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
