import matplotlib.pyplot as plt
import numpy as np

from yarkov import Lineage


def mom_est(data: np.ndarray) -> tuple[float, float]:
    mu = np.mean(data)
    sigma2 = np.mean(data**2) - mu**2
    return mu, sigma2


def mle_est(data: np.ndarray) -> tuple[float, float]:
    mu = np.mean(data)
    sigma2 = np.var(data)
    return mu, sigma2


noisy = dict(k_syn=100, M_crit=150, a=0.95, random_seed=42, subsample_method=None)
qb = dict(k_syn=40, M_crit=150, a=0.95, random_seed=42, subsample_method=None)
null = dict(k_syn=100, M_crit=150, a=0.0, random_seed=42, subsample_method=None)

for sim_name, sim_params in zip(["noisy", "qb", "null"], [noisy, qb, null]):
    Sim = Lineage(**sim_params)
    Sim.simulate_lineage(20)
    df = Sim.data

    generations = df.filter(df["gen"] != 0)["gen"].unique()

    means = []
    variances = []

    for g in generations:
        arr = df.filter(df["gen"] == g)["mass_protein"].to_numpy()
        mu_mom, sigma2_mom = mom_est(arr)
        mu_mle, sigma2_mle = mle_est(arr)
        means.append(mu_mle)
        variances.append(sigma2_mle)

    fig = plt.figure(constrained_layout=True)
    G = fig.add_gridspec(nrows=1, ncols=1)
    m_ax = plt.subplot(G[0, 0])
    var_ax = m_ax.twinx()

    m_ax.set_xlabel("Time (generation)")
    m_ax.set_ylabel("Mean Protein Mass")
    var_ax.set_ylabel("Variance in Protein Mass")

    m_ax.set_ylim((0, 150))
    var_ax.set_ylim((0, 18000))

    m_ax.plot(generations, means, c="r", label="mu")
    var_ax.plot(generations, variances, c="b", label="sigma2")
    plt.savefig(f"{sim_name}_moments.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)
