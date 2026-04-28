"""
Parameter search module for Yarkov simulations.

Uses Latin Hypercube Sampling (LHS) to efficiently explore the parameter space
of k_syn, M_crit, and initial_mass, computing statistical moments of the
resulting mass_protein distribution.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import polars as pl
import scipy.stats as stats

from yarkov.sim import Ensemble


def latin_hypercube_samples(
    n_samples: int,
    k_syn_bounds: tuple[float, float] = (0, 200),
    M_crit_bounds: tuple[float, float] = (100, 200),
    a_bounds: tuple[float, float] = (0, 1),
    initial_mass_bounds: tuple[float, float] = (50, 250),
    seed: int | None = None,
) -> list[dict]:
    """
    Generate parameter combinations using Latin Hypercube Sampling.

    Parameters
    ----------
    n_samples : int
        Number of parameter combinations to generate.
    k_syn_bounds : tuple[float, float]
        (min, max) bounds for k_syn parameter.
    M_crit_bounds : tuple[float, float]
        (min, max) bounds for M_crit parameter.
    a_bounds : tuple[float, float]
        (min, max) bounds for a parameter.
    initial_mass_bounds : tuple[float, float]
        (min, max) bounds for initial_mass.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    list[dict]
        Parameter dictionaries with keys: k_syn, M_crit, initial_mass.
    """
    sampler = stats.qmc.LatinHypercube(d=4, seed=seed)
    unit_samples = sampler.random(n=n_samples)
    bounds = np.array([k_syn_bounds, M_crit_bounds, a_bounds, initial_mass_bounds])
    scaled = stats.qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])

    return [
        {
            "k_syn": float(k_syn),
            "M_crit": float(M_crit),
            "a": float(a),
            "initial_mass": float(init_mass),
        }
        for k_syn, M_crit, a, init_mass in scaled
    ]


def compute_moments(masses: np.ndarray) -> dict:
    """
    Extract mean, variance, skew, and kurtosis from mass distribution.

    Parameters
    ----------
    masses : np.ndarray
        Array of protein masses.

    Returns
    -------
    dict
        Dictionary with keys: mean, variance, skew, kurtosis.
    """
    return {
        "mean": float(np.mean(masses)),
        "variance": float(np.var(masses)),
        "skew": float(stats.skew(masses)),
        "kurtosis": float(stats.kurtosis(masses)),
    }


def compute_trajectory_moments(ensemble: Ensemble) -> list[dict]:
    """
    Extract mean, variance, skew, and kurtosis of mass_protein at each generation.

    Parameters
    ----------
    ensemble : Ensemble
        A simulated Ensemble object.

    Returns
    -------
    list[dict]
        List of dictionaries with keys: gen, mean, variance, skew, kurtosis.
    """
    results = []
    for gen in range(ensemble.n_gen + 1):
        gen_data = ensemble.data.filter(pl.col("gen") == gen)
        masses = gen_data["mass_protein"].to_numpy()
        moment = compute_moments(masses)
        moment["gen"] = gen
        results.append(moment)
    return results


def run_simulation(
    params: dict,
    n_cells: int = 100,
    n_gen: int = 10,
    seed: int | None = None,
    sample_id: int | None = None,
) -> list[dict]:
    """
    Run a single ensemble simulation with given parameters.

    Parameters
    ----------
    params : dict
        Parameter dictionary with k_syn, M_crit, initial_mass.
    n_cells : int
        Number of cells per generation.
    n_gen : int
        Number of generations to simulate.
    seed : int | None
        Random seed for reproducibility.
    sample_id : int | None
        Unique sample identifier.

    Returns
    -------
    list[dict]
        List of parameter + moment dictionaries for each generation.
    """
    ensemble = Ensemble(
        k_syn=params["k_syn"],
        M_crit=params["M_crit"],
        a=params["a"],
        n_cells=n_cells,
        n_gen=n_gen,
        random_seed=seed,
        initial_mass=params["initial_mass"],
    )

    ensemble.simulate()

    trajectory = compute_trajectory_moments(ensemble)
    return [
        {
            "sample_id": sample_id,
            "k_syn": params["k_syn"],
            "M_crit": params["M_crit"],
            "a": params["a"],
            "initial_mass": params["initial_mass"],
            "seed": seed,
            **moment,
        }
        for moment in trajectory
    ]


def run_parameter_search(
    n_samples: int = 50,
    n_cells: int = 100,
    n_gen: int = 10,
    n_seeds: int = 1,
    k_syn_bounds: tuple[float, float] = (0.5, 5.0),
    M_crit_bounds: tuple[float, float] = (100, 200),
    a_bounds: tuple[float, float] = (0, 1),
    initial_mass_bounds: tuple[float, float] = (50, 250),
    max_workers: int = 4,
    seed: int | None = None,
) -> pl.DataFrame:
    """
    Run ensemble simulations across parameter space using LHS.

    Parameters
    ----------
    n_samples : int
        Number of LHS parameter combinations.
    n_cells : int
        Number of cells per generation in each simulation.
    n_gen : int
        Number of generations per simulation.
    n_seeds : int
        Number of random seeds per parameter combination (for variance estimation).
    k_syn_bounds : tuple[float, float]
        (min, max) bounds for k_syn.
    M_crit_bounds : tuple[float, float]
        (min, max) bounds for M_crit.
    a_bounds : tuple[float, float]
        (min, amx) bounds for asymmetry parameter a.
    initial_mass_bounds : tuple[float, float]
        (min, max) bounds for initial_mass.
    max_workers : int
        Maximum parallel workers.
    seed : int | None
        Random seed for LHS and simulations.

    Returns
    -------
    pl.DataFrame
        Results with columns: sample_id, k_syn, M_crit, initial_mass, seed, gen, mean, variance, skew, kurtosis.
    """
    # Generate LHS samples
    param_list = latin_hypercube_samples(
        n_samples=n_samples,
        k_syn_bounds=k_syn_bounds,
        M_crit_bounds=M_crit_bounds,
        initial_mass_bounds=initial_mass_bounds,
        seed=seed,
    )

    results = []
    failed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {}
        sample_id = 1
        for param_idx, params in enumerate(param_list):
            for s in range(n_seeds):
                sim_seed = (
                    (seed + param_idx * n_seeds + s + 1) if seed is not None else None
                )
                fut = executor.submit(
                    run_simulation, params, n_cells, n_gen, sim_seed, sample_id
                )
                future_to_params[fut] = params
                sample_id += 1

        for future in as_completed(future_to_params):
            params = future_to_params[future]
            try:
                results.extend(future.result())
            except Exception as e:
                cause = e
                while hasattr(cause, "__cause__") and cause.__cause__ is not None:
                    cause = cause.__cause__
                param_str = f"k_syn={params['k_syn']:.3f}, M_crit={params['M_crit']:.1f}, initial_mass={params['initial_mass']:.1f}, a={params['a']:.3f}"
                warnings.warn(
                    f"Simulation failed ({param_str}): {type(cause).__name__}: {cause}"
                )
                failed_count += 1

    if failed_count:
        warnings.warn(
            f"Parameter search completed with {failed_count} failed simulation(s)."
        )

    return pl.DataFrame(results)


def aggregate_results(results: pl.DataFrame) -> pl.DataFrame:
    """
    Aggregate results across seeds, computing mean and std of moments per generation.

    Parameters
    ----------
    results : pl.DataFrame
        Raw results from run_parameter_search.

    Returns
    -------
    pl.DataFrame
        Aggregated results with mean and std of each moment per parameter combination and generation.
    """
    aggregated = (
        results.group_by(["k_syn", "M_crit", "initial_mass", "gen"])
        .agg(
            pl.col("mean").mean().alias("mean_mean"),
            pl.col("mean").std(ddof=0).fill_nan(0.0).alias("std_mean"),
            pl.col("variance").mean().alias("mean_variance"),
            pl.col("variance").std(ddof=0).fill_nan(0.0).alias("std_variance"),
            pl.col("skew").mean().alias("mean_skew"),
            pl.col("skew").std(ddof=0).fill_nan(0.0).alias("std_skew"),
            pl.col("kurtosis").mean().alias("mean_kurtosis"),
            pl.col("kurtosis").std(ddof=0).fill_nan(0.0).alias("std_kurtosis"),
        )
        .sort(["k_syn", "M_crit", "initial_mass", "gen"])
    )

    return aggregated


def save_results(
    results: pl.DataFrame, filepath: str = "sims.csv"
) -> None:
    """
    Save results to CSV file.

    Parameters
    ----------
    results : pl.DataFrame
        Results to save.
    filepath : str
        Output filepath.
    """
    results.write_csv(filepath)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Running parameter search with LHS...")

    # Run search with reasonable defaults
    results = run_parameter_search(
        n_samples=300,
        n_cells=100,
        n_gen=10,
        n_seeds=1,
        k_syn_bounds=(0, 200),
        M_crit_bounds=(100, 200),
        a_bounds=(0, 1),
        initial_mass_bounds=(50, 250),
        max_workers=4,
        seed=42,
    )

    print(f"Completed {len(results)} simulation timesteps")
    print(results.head())

    # Save results
    save_results(results, "sims.csv")
