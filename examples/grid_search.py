from itertools import product
from multiprocessing import Pool
from os.path import join

import matplotlib.pyplot as plt

from yarkov import Lineage

# generate list of conditions to iterate over
k_syn_range = [50, 100, 150, 200]
init = [0, 50, 100, 150, 200]
M_crit_range = [50, 100, 150, 200, 400]
a_range = [0, 1, 0.99, 0.95, 0.90, 0.50, 0.1]
states = product(k_syn_range, init, M_crit_range, a_range, repeat=1)


def fetch_lineage_information(state: tuple[int, int, int, float]):
    sym = Lineage(
        k_syn=state[0],
        M_crit=state[2],
        a=state[3],
        subsample_method=None,
        random_seed=42,
    )

    initial_conditions = [
        {
            "id": "0",
            "parent": None,
            "mass_protein": state[1],
            "gen": 0,
            "state": "Diffuse",
        }
    ]
    sym.simulate_lineage(initial_conditions=initial_conditions, num_generations=20)

    pid = join("grid", f"k{state[0]}_i{state[1]}_M{state[2]}_a{state[3]}")
    sym.data.write_csv(f"{pid}_cells.csv")

    stats = sym.trace_lineage("mass_protein", 10)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for row in stats.iter_rows():
        plt.plot(range(len(row)), row[::-1], color="blue", alpha=0.01)
    ax1.set_xlim((0, 10))
    ax1.set_ylim((0, 500))
    ax1.set_ylabel("Mass Protein", size=18)
    ax1.set_xlabel("Generation", size=18)
    plt.savefig(f"{pid}_lineage.svg", transparent=True, bbox_inches="tight")
    plt.close(fig1)

    # clear up memory
    del sym, stats, fig1, ax1, initial_conditions


if __name__ == "__main__":
    with Pool() as p:
        p.map(fetch_lineage_information, states)
