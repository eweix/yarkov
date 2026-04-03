import os

import numpy as np
import pandas as pd
from rich.progress import track


def subsample_lineage(
    df: pd.DataFrame,
    sample_method: str | None = "random",
    max_count: int = 50,
) -> pd.DataFrame:
    if sample_method == "random":
        n_sample = min(len(df), max_count)
        sampled_indices = np.random.choice(len(df), size=n_sample, replace=False)
    else:
        n_sample = min(len(df), max_count)
        sampled_indices = np.random.choice(len(df), size=n_sample, replace=False)
    return df.iloc[sampled_indices].reset_index(drop=True)


def simulate_sampled_lineage(
    generations: int,
    k_syn: float,
    M_crit: int | float = 150,
    p_noise: float = 0.05,
    max_cells: int = 50,
    output_dir: str = "./sim_data",
    batch_save: bool = True,
    seed: int | None = None,
    sample_method: str | None = "random",
) -> pd.DataFrame:
    """
    Args:
        k_syn: Protein units added per cell cycle.
        M_crit: Mass required to trigger wave activity and polarized, asymmetric outcomes.
        p_noise: Variability in 50/50 split when in diffuse state.
        max_cells: Maximum number of cells to sample per generation.
        output_dir: Directory to save CSV files.
        batch_save: Whether to save each generation to CSV.
        seed: Random seed for reproducibility.
        sample_method: Method for sampling cells: "random" or None.

    Returns:
        DataFrame with columns: id, parent, mass_protein1, mass_protein2, gen, state

    Examples:
        >>> df = simulate_sampled_lineage(generations=8, k_syn=40, M_crit=150)
        >>> df.head()
    """
    if seed is not None:
        np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Start with one cell just below the threshold
    lineage = [
        {
            "id": "0",
            "parent": None,
            "mass_protein1": M_crit * 0.8,
            "mass_protein2": M_crit * 0.8,
            "gen": 0,
            "state": "Diffuse",
        }
    ]
    active_pool = pd.DataFrame(lineage)

    for g in track(range(generations)):
        # Growth phase with random synthesis noise
        total_masses_protein1 = active_pool["mass_protein1"] + k_syn * (
            1 + np.random.normal(0, p_noise, len(active_pool))
        )
        total_masses_protein2 = active_pool["mass_protein2"] + k_syn * (
            1 + np.random.normal(0, p_noise, len(active_pool))
        )

        # Create mask for cells that meet or exceed M_crit
        polarized_mask = (
            total_masses_protein1 >= M_crit
        )  # decision logic based on protein 1

        # Initialize shares arrays for protein 1
        shares_m_protein1 = np.full(len(active_pool), 0.5)  # Default for diffuse
        shares_d_protein1 = np.full(len(active_pool), 0.5)  # Default for diffuse

        # For polarized cells, set shares to 0.95 and 0.05 for protein 1
        shares_m_protein1[polarized_mask] = 0.95
        shares_d_protein1[polarized_mask] = 0.05

        # For diffuse cells, calculate random shares with noise for protein 1
        noise_protein1 = np.random.normal(0.5, p_noise, len(active_pool))
        shares_m_protein1[~polarized_mask] = noise_protein1[~polarized_mask]
        shares_d_protein1[~polarized_mask] = 1 - noise_protein1[~polarized_mask]

        # Ensure shares are within [0, 1] bounds for protein 1
        shares_m_protein1 = np.clip(shares_m_protein1, 0, 1)
        shares_d_protein1 = np.clip(shares_d_protein1, 0, 1)

        # For protein 2, always distribute evenly (50/50 split) regardless of state
        shares_m_protein2 = np.full(len(active_pool), 0.5)
        shares_d_protein2 = np.full(len(active_pool), 0.5)

        # 3. Create Progeny
        m_masses_protein1 = total_masses_protein1 * shares_m_protein1
        d_masses_protein1 = total_masses_protein1 * shares_d_protein1
        m_masses_protein2 = total_masses_protein2 * shares_m_protein2
        d_masses_protein2 = total_masses_protein2 * shares_d_protein2

        # Create new cells dataframe
        m_ids = [f"{cell_id}m" for cell_id in active_pool["id"]]
        d_ids = [f"{cell_id}d" for cell_id in active_pool["id"]]

        new_cells_m = pd.DataFrame(
            {
                "id": m_ids,
                "parent": active_pool["id"],
                "mass_protein1": m_masses_protein1,
                "mass_protein2": m_masses_protein2,
                "gen": g + 1,
                "state": np.where(polarized_mask, "Polarized", "Diffuse"),
            }
        )

        new_cells_d = pd.DataFrame(
            {
                "id": d_ids,
                "parent": active_pool["id"],
                "mass_protein1": d_masses_protein1,
                "mass_protein2": d_masses_protein2,
                "gen": g + 1,
                "state": np.where(polarized_mask, "Polarized", "Diffuse"),
            }
        )

        new_cells = pd.concat([new_cells_m, new_cells_d], ignore_index=True)

        if batch_save:
            new_cells.to_csv(f"{output_dir}/gen_{g:03d}.csv", index=False)

        active_pool = subsample_lineage(
            new_cells,
            sample_method="random",
            max_count=50,
        )
        if len(active_pool) < 2:
            lineage_df = pd.concat(
                [pd.DataFrame(lineage), new_cells], ignore_index=True
            )
            return lineage_df

        lineage_df = pd.concat([pd.DataFrame(lineage), new_cells], ignore_index=True)
        lineage = lineage_df.to_dict("records")

    return pd.DataFrame(lineage)


def reconstruct_lineage(
    cell: pd.Series,
    sim_data: pd.DataFrame,
    attribute: str,
    lineage: np.ndarray | None,
) -> np.ndarray:
    """Takes a dataframe of all cells and a specific row from another dataframe, then searches through the cells to reconstruct a dataframe thingy with all of the lineages. Quite simple really."""

    # make lineage array at beginning
    if lineage is None:
        lineage = np.ndarray(cell["gen"])

    # update lineage array
    lineage[cell["gen"]] = cell[attribute]

    # recursively construct lineage information
    if cell["gen"] == 0:
        return lineage
    elif cell["gen"] < 0:
        raise Exception(
            f"Expected a nonnegative integer for generation, got {cell['gen']}."
        )
    else:
        parent_cell = sim_data[sim_data["id"] == cell["parent"]]
        reconstruct_lineage(parent_cell, sim_data, attribute, lineage)
