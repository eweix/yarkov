# %% import libraries
import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from yarkov import Lineage, plot_mass_distributions, plot_tree

# %%
NoisySim = Lineage(
    k_syn=100,
    M_crit=150,
    a=0.95,
    random_seed=42,
    data_directory="./qb_sim",
    subsample_method=None,
)
NoisySim.simulate_lineage(20)

# %% 
NullSim = Lineage(
    k_syn=100,
    M_crit=150,
    a=0.0,
    random_seed=42,
    data_directory="./null_sim",
    subsample_method=None,
)
NullSim.simulate_lineage(20)

# %%
TailedSim = Lineage(
    k_syn=40,
    M_crit=150,
    a=0.95,
    random_seed=42,
    data_directory="./qb_sim",
    subsample_method=None,
)
TailedSim.simulate_lineage(20)

# %% trace out lineage sim
fig, ax = plt.subplots(figsize=(10, 6))

sims = [NullSim, NoisySim, TailedSim]
colors = ["k", "b", "g"]
labels = ["Null", "Stratified", "QB"]

for Sim, color, label in zip(sims, colors, labels):
    stats = Sim.trace_lineage("mass_protein", 10)
    for row in stats.iter_rows():
        plt.plot(range(len(row)), row[::-1], color=color, alpha=0.01, label=label)
ax.set_xlim((0,10))
ax.set_ylim((0,500))
ax.set_ylabel("Mass Protein", size=18)
ax.set_xlabel("Generation", size=18)
# plt.legend()
plt.savefig("fig/comparison_traces.svg", transparent=True, bbox_inches="tight")
