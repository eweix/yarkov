# %% import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from probate import plot_mass_distributions, plot_tree, simulate_sample_lineage

# %% this scenario will produce a long-tailed distribution of protein-rich cells
df = simulate_sampled_lineage(generations=20, k_syn=100, M_crit=150, seed=42)


# %% plot out lineage tree
fig1, ax1 = plt.subplots(figsize=(14, 10))
plot_tree(df[df.gen < 10], ax1, fig1, 150)
plt.savefig("../fig/sim_tailed-lineage-tree.svg", transparent=True, bbox_inches="tight")
plt.show()

# %% plot
fig2, ax2 = plt.subplots(figsize=(7, 3))
plot_mass_distributions(df[df.gen < 10], ax2, 150)
sns.set_style("ticks")
sns.despine()
ax2.set_xlabel("Simulation Generation", size=12)
ax2.set_ylabel("Protein Mass", size=12)
plt.savefig("../fig/sim_tailed-mass-dist.svg", transparent=True, bbox_inches="tight")
plt.show()

# %%
cdict = generate_color_dict(list(df["gen"].unique()), cmap="magma")
g = sns.jointplot(
    data=df,
    x="mass_protein1",
    y="mass_protein2",
    hue="gen",
    palette=cdict,
    alpha=0.3,
)
g.ax_joint.legend_.remove()
g.set_axis_labels("MinD Mass (AU)", "MinE Mass (AU)")
plt.savefig("../fig/sim_phase-portrait.svg", transparent=True, bbox_inches="tight")
plt.show()
