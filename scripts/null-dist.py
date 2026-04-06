# %% import libraries
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from probate import Lineage, plot_mass_distributions, plot_tree

# %% this scenario will produce a long-tailed distribution of protein-rich cells
NullSim = Lineage(
    k_syn=100,
    M_crit=150,
    a=0.0,
    random_seed=42,
    data_directory="./null_sim",
    subsample_method=None,
)
NullSim.simulate_lineage(20)
df = NullSim.data

# # %% plot out lineage tree
cells = df.filter(pl.col("gen") < 10)
fig1, ax1 = plt.subplots(figsize=(14, 10))
plot_tree(cells, ax1, fig1, 150)
plt.savefig("fig/null_lineage-tree.svg", transparent=True, bbox_inches="tight")

# # %% plot
fig2, ax2 = plt.subplots(figsize=(7, 3))
plot_mass_distributions(cells, ax2, 150)
sns.set_style("ticks")
sns.despine()
ax2.set_xlabel("Simulation Generation", size=12)
ax2.set_ylabel("Protein Mass", size=12)
plt.savefig("fig/null_mass-dist.svg", transparent=True, bbox_inches="tight")

# %% trace out lineage sim
stats = NullSim.trace_lineage("mass_protein", 10)

fig3, ax3 = plt.subplots(figsize=(10, 6))
for row in stats.iter_rows():
    plt.plot(range(len(row)), row[::-1], color="b", alpha=0.1)
ax3.set_ylabel("Mass Protein", size=18)
ax3.set_xlabel("Generation", size=18)
plt.savefig("fig/null_trace.svg", transparent=True, bbox_inches="tight")
