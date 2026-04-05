import numpy as np
import polars as pl


class Lineage:
    def __init__(
        self,
        k_syn: float,
        M_crit: int | float = 150,
        syn_noise: float = 0.05,
        div_noise: float = 0.05,
        subsample_method: str | None = "random",
        random_seed: int | None = None,
        subsample_size: int = 50,
        data_directory: str = "./sim_data",
    ):
        # set parameters
        self.k_syn = k_syn
        self.M_crit = M_crit
        self.syn_noise = syn_noise
        self.div_noise = div_noise
        self.subsample_method = subsample_method
        self.random_seed = random_seed
        self.subsample_size = subsample_size
        self.data_directory = data_directory

    def _prune(self, generation: int) -> pl.DataFrame:
        current_generation = self.data.filter(self.data["gen"] == generation)
        gen_size = current_generation.shape[0]
        if self.subsample_method == "random":
            n_sample = min(len(current_generation), self.subsample_size)
            sampled_indices = np.random.choice(gen_size, size=n_sample, replace=False)
            return current_generation[sampled_indices]
        else:
            return current_generation

    def _gen_step(self, input_generation: int):
        active_pool = self._prune(input_generation)
        active_size = active_pool.shape[0]

        # Growth phase with random synthesis noise
        total_masses_protein = active_pool["mass_protein"] + self.k_syn * (
            1 + np.random.normal(0, self.syn_noise, active_size)
        )

        # get cells
        polarized_mask = total_masses_protein >= self.M_crit
        shares_m_protein = np.full(active_size, 0.5)
        shares_d_protein = np.full(active_size, 0.5)
        shares_m_protein[polarized_mask] = 0.95
        shares_d_protein[polarized_mask] = 0.05

        # For diffuse cells, calculate random shares with noise for protein 1
        noise_protein = np.random.normal(0.5, self.syn_noise, active_size)
        shares_m_protein[~polarized_mask] = noise_protein[~polarized_mask]
        shares_d_protein[~polarized_mask] = 1 - noise_protein[~polarized_mask]

        # Ensure shares are within [0, 1] bounds
        shares_m_protein = np.clip(shares_m_protein, 0, 1)
        shares_d_protein = np.clip(shares_d_protein, 0, 1)

        # Create Progeny
        m_masses_protein = total_masses_protein * shares_m_protein
        d_masses_protein = total_masses_protein * shares_d_protein

        # Create new cells dataframe
        m_ids = [f"{cell_id}m" for cell_id in active_pool["id"]]
        d_ids = [f"{cell_id}d" for cell_id in active_pool["id"]]

        new_cells_m = pl.DataFrame(
            {
                "id": m_ids,
                "parent": active_pool["id"].to_list(),
                "mass_protein": m_masses_protein,
                "gen": input_generation + 1,
                "state": np.where(polarized_mask, "Polarized", "Diffuse"),
            }
        )

        new_cells_d = pl.DataFrame(
            {
                "id": d_ids,
                "parent": active_pool["id"].to_list(),
                "mass_protein": d_masses_protein,
                "gen": input_generation + 1,
                "state": np.where(polarized_mask, "Polarized", "Diffuse"),
            }
        )

        self.data = pl.concat(
            [self.data, new_cells_m, new_cells_d],
            how="vertical_relaxed",
        )

    def _checkpoint(self, generation: int):
        output_path = f"{self.data_directory}/gen_{generation:03d}.csv"
        try:
            self.data[self.data["gen"] == generation].write_csv(output_path)
        except TypeError:
            raise TypeError

    def simulate_lineage(
        self,
        num_generations: int,
        initial_conditions: list[dict] | None = None,
        checkpoint=False,
    ):
        if initial_conditions is None:
            initial_conditions = [
                {
                    "id": "0",
                    "parent": None,
                    "mass_protein": self.M_crit * 0.8,
                    "gen": 0,
                    "state": "Diffuse",
                }
            ]
        self.data = pl.DataFrame(initial_conditions)

        for g in range(num_generations):
            self._gen_step(g)
            if checkpoint:
                self._checkpoint(g)

    def trace_lineage(
        self, attribute: str, from_generation: int | None = None
    ) -> pl.DataFrame:
        def check_generation(g: int | None) -> int | None:
            if g is None:
                return self.data["gen"].max()
            elif g <= 0:
                e = f"Expected a positive integer for generation, not {g}"
                raise Exception(e)
            else:
                return g

        def build_lineage_trace(from_generation: int | None = None) -> pl.DataFrame:
            trace = self.data.filter(self.data["gen"] == from_generation)[["id", "gen"]]
            i = 1
            while (
                trace.height - 1
                > trace.select(pl.col(f"gen_{i}_id").null_count()).item()
            ):
                trace = trace.join(
                    self.data.rename({"parent": f"gen_{i + 1}"}),
                    left_on=f"gen_{i}",
                    right_on="id",
                    how="left",
                )
                i += 1
            return trace

        def collect_lineage_stats(trace: pl.DataFrame, attribute: str) -> pl.DataFrame:
            lookup = dict(self.data[["id", attribute]].iter_rows())
            lineage_stats = trace
            lineage_stats.with_columns(pl.all().replace(lookup, default=None))
            return lineage_stats

        g = check_generation(from_generation)
        trace = build_lineage_trace(g)
        stats = collect_lineage_stats(trace, attribute)
        return stats
