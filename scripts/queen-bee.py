# %% import libraries
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from probate import Lineage, plot_mass_distributions, plot_tree

# %% this scenario will produce a long-tailed distribution of protein-rich cells
TailedSim = Lineage(
    k_syn=100,
    M_crit=150,
    random_seed=42,
    data_directory="./qb_sim",
    subsample_method=None,
)
TailedSim.simulate_lineage(20)
df = TailedSim.data

# # %% plot out lineage tree
cells = df.filter(pl.col("gen") < 10)
fig1, ax1 = plt.subplots(figsize=(14, 10))
plot_tree(cells, ax1, fig1, 150)
# plt.savefig("../fig/sim_tailed-lineage-tree.svg", transparent=True, bbox_inches="tight")
plt.show()

# # %% plot
fig2, ax2 = plt.subplots(figsize=(7, 3))
plot_mass_distributions(cells, ax2, 150)
sns.set_style("ticks")
sns.despine()
ax2.set_xlabel("Simulation Generation", size=12)
ax2.set_ylabel("Protein Mass", size=12)
# plt.savefig("../fig/sim_tailed-mass-dist.svg", transparent=True, bbox_inches="tight")
plt.show()
