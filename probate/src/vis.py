# %% import libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, to_hex
from rich.progress import track


# %% define visualization functions
def generate_color_dict(paths: list[str], cmap="colorblind") -> dict[str, str]:
    if not paths:
        return {}
    n = len(paths)
    palette = sns.color_palette(cmap, n)
    colors = [to_hex(c) for c in palette]
    return dict(zip(paths, colors))


def plot_tree(df, ax, fig, M_crit=150):
    coords = {}
    df = df.sort_values("gen")
    parent_state_map = df.set_index("id")["state"].to_dict()

    def get_x_pos(cell_id):
        if cell_id == "0":
            return 0.5
        pos, width = 0.5, 0.25
        for char in cell_id[1:]:
            pos = pos - width if char == "m" else pos + width
            width /= 2
        return pos

    # Color normalization based on M_crit as the center
    vmax = max(M_crit, df["mass_protein1"].max()) * 1.1
    norm = TwoSlopeNorm(vmin=0, vcenter=M_crit, vmax=vmax)
    cmap = plt.get_cmap("RdYlBu_r")

    # Lists to collect coordinates for the final scatter plot
    x_vals, y_vals, masses = [], [], []

    for _, row in df.iterrows():
        x, y = get_x_pos(row["id"]), row["gen"]
        coords[row["id"]] = (x, y)
        x_vals.append(x)
        y_vals.append(y)
        masses.append(row["mass_protein1"])

        # Plot edges
        if (row["parent"] is not None) and (row["parent"] is not np.nan):
            px, py = coords[row["parent"]]
            p_state = parent_state_map.get(row["parent"])

            if p_state == "Polarized":
                edge_col, alpha = (
                    ("red", 0.9) if row["mass_protein1"] >= M_crit else ("#DDA0DD", 0.6)
                )
            else:
                edge_col, alpha = ("gray", 0.3)
            ax.plot([px, x], [py, y], color=edge_col, alpha=alpha, lw=2, zorder=1)

    # Plot all cells at once to create a mappable for the colorbar
    scatter = ax.scatter(
        x_vals,
        y_vals,
        s=80,
        c=masses,
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Cell Mass", fontsize=15)

    # Add a horizontal line on the colorbar at M_crit for clarity
    cbar.ax.axhline(M_crit, color="black", linestyle="--", linewidth=1)

    # label axes
    ax.set_title(f"Lineage Trace (M_crit = {M_crit})", fontsize=18)
    # ax.set_xlabel("Relative Lineage Position", fontsize=12)
    ax.set_ylabel("Generation", fontsize=12)

    # style axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.invert_yaxis()

    plt.tight_layout()


def plot_mass_distributions(df, ax, M_crit=150):

    sns.boxplot(
        data=df, x="gen", y="mass_protein1", whis=np.inf, color="lightgray", ax=ax
    )
    sns.stripplot(
        data=df,
        x="gen",
        y="mass_protein1",
        hue="state",
        palette={"Diffuse": "blue", "Polarized": "red"},
        alpha=0.5,
        ax=ax,
    )

    ax.axhline(M_crit, color="red", linestyle="--", label="Threshold $M_{crit}$")
    # ax.set_title("Mass Distribution per Generation (Log Scale)")
    ax.legend()
