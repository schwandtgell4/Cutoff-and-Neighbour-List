import importlib.util
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.constants import Avogadro, R

from LJ_gas import (
    ParticleSystem,
    SimulationParameters,
    calculate_force,
    initialize_positions,
    initialize_velocities,
    instantaneous_temperature,
    neighbour_list_error_metrics,
    potential_energy,
    simulate_NVE_step,
    simulate_NVT_step,
    update_neighbour_list,
)


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
n_particles = 1000
mass_argon = 39.95                  # mass in u = 1e-3 kg/mol
sigma_argon = 0.34                  # sigma in nm     Argon: 0.34
epsilon_argon = 120.0 * R * 1e-3    # epsilon in kJ/mol Argon: 120

dt = 0.1                            # ps
temperature = 300.0                 # K
box_length_start = 100.0            # nm
tau_thermostat = 1.0                # thermostat coupling constant in 1/ps
rij_min = 1e-2                      # nm

production_ensembles = ["NVE"] # NVE, NVT or NVE and NVT

n_equil_steps = 500 # number of steps to eq each initial system
equilibration_sample_interval = 10 # store temp and ekin every 10 eq steps
equilibration_n_update = 1 # update of neighbour list during eq

n_timing_steps = 100 # number of MD steps per runtime measurement after equilibration
timing_repeats = 5 # repeat 5 times and use median runtime

n_list_validation_cycles = 20 # neighbour-list cycles checked for each candidate
n_list_validation_candidates = 10 # number of fastest accuracy-approved (r_cut, n_update) candidates

n_update_values = list(range(1, 11))
cutoff_factors = [1, 2, 2.5, 3, 5, 7.5, 10, 15, 20]
density_factors = [0.5, 1.0, 1.5]
seeds = [1, 2, 3, 4, 5]

MAX_CUTOFF_FORCE_ERROR = 1e-2   # 1 %
MAX_LIST_FORCE_ERROR = 1e-3 # 0.1 %
MAX_MISSING_PAIRS = 0   # 0 missing pairs

equilibration_cutoff_factor = max(cutoff_factors)

base_directory = Path(__file__).resolve().parent
original_core_path = base_directory / "Original" / "LJ_gas_original.py"
output_directory = base_directory / "doe_ergebnisse"


def load_original_core(module_path):
    """Load the unchanged core under a unique name without changing sys.path."""

    module_path = Path(module_path).resolve()

    if not module_path.is_file():
        raise FileNotFoundError(
            "Original core not found at "
            f"{module_path}. Expected Original/LJ_gas_original.py next to this file."
        )

    module_name = "_lj_gas_original_reference"

    if module_name in sys.modules:
        return sys.modules[module_name]

    specification = importlib.util.spec_from_file_location(
        module_name,
        module_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load original core from {module_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def validate_ensembles():
    """Reject misspelled ensemble names before the expensive equilibration."""

    normalized = [str(value).upper() for value in production_ensembles]
    invalid = sorted(set(normalized) - {"NVE", "NVT"})

    if invalid:
        raise ValueError(
            "production_ensembles may contain only 'NVE' and 'NVT'; "
            f"received {invalid}"
        )

    return normalized


def make_optimized_simulation(box_length, r_cut, n_steps):
    return SimulationParameters(
        dt=dt,
        n_steps=n_steps,
        temperature=temperature,
        box_length=box_length,
        tau_thermostat=tau_thermostat,
        rij_min=rij_min,
        r_cut=r_cut,
    )


def make_optimized_system(state, sim, n_update):
    """Create an independent optimized system from a stored common state."""

    ps = ParticleSystem(n_particles)
    ps.mass[:] = mass_argon
    ps.sigma[:] = sigma_argon
    ps.epsilon[:] = epsilon_argon
    ps.position[:] = state["position"]
    ps.velocity[:] = state["velocity"]

    # Setup is deliberately outside every performance timer.
    update_neighbour_list(ps, sim, step=0, n_update=n_update)
    calculate_force(ps, sim)
    return ps


def optimized_step_function(ensemble):
    if ensemble == "NVE":
        return simulate_NVE_step
    if ensemble == "NVT":
        return simulate_NVT_step
    raise ValueError("ensemble must be 'NVE' or 'NVT'")


def original_step_function(original_core, ensemble):
    if ensemble == "NVE":
        return original_core.simulate_NVE_step
    if ensemble == "NVT":
        return original_core.simulate_NVT_step
    raise ValueError("ensemble must be 'NVE' or 'NVT'")


def time_optimized_once(state, sim, ensemble, n_update, production_seed):
    """Time only the optimized integrator loop from a fresh state copy."""

    ps = make_optimized_system(state, sim, n_update)
    step_function = optimized_step_function(ensemble)

    # NVT then receives the same Langevin random sequence in every repeat.
    np.random.seed(production_seed)
    start = time.perf_counter()

    for step in range(1, n_timing_steps + 1):
        step_function(ps, sim, step, n_update)

    runtime = time.perf_counter() - start
    timed_rebuilds = ps.neighbour_rebuilds - 1
    return runtime, timed_rebuilds


def time_original_once(
    original_core,
    state,
    box_length,
    ensemble,
    production_seed,
):
    """Time only the unchanged all-pairs integrator loop."""

    sim_original = original_core.SimulationParameters(
        dt=dt,
        n_steps=n_timing_steps,
        temperature=temperature,
        box_length=box_length,
        tau_thermostat=tau_thermostat,
        rij_min=rij_min,
    )

    ps_original = original_core.ParticleSystem(n_particles)
    ps_original.mass[:] = mass_argon
    ps_original.sigma[:] = sigma_argon
    ps_original.epsilon[:] = epsilon_argon
    ps_original.position[:] = state["position"]
    ps_original.velocity[:] = state["velocity"]

    # The initial force is setup and is excluded from the timer in both codes.
    original_core.calculate_force(ps_original, sim_original)
    step_function = original_step_function(original_core, ensemble)

    np.random.seed(production_seed)
    start = time.perf_counter()

    for _ in range(n_timing_steps):
        step_function(ps_original, sim_original)

    return time.perf_counter() - start


def median_original_runtime(
    original_core,
    state,
    box_length,
    ensemble,
    production_seed,
):
    samples = [
        time_original_once(
            original_core,
            state,
            box_length,
            ensemble,
            production_seed,
        )
        for _ in range(timing_repeats)
    ]
    return float(np.median(samples))


def median_optimized_runtime(state, sim, ensemble, n_update, production_seed):
    samples = []
    rebuild_counts = []

    for _ in range(timing_repeats):
        runtime, rebuilds = time_optimized_once(
            state,
            sim,
            ensemble,
            n_update,
            production_seed,
        )
        samples.append(runtime)
        rebuild_counts.append(rebuilds)

    return float(np.median(samples)), int(np.median(rebuild_counts))


def measure_cutoff_error(state, sim):
    """Measure pure cutoff error on the common, fresh step-zero snapshot."""

    ps = make_optimized_system(state, sim, n_update=1)
    metrics = neighbour_list_error_metrics(ps, sim)

    return {
        "cutoff_force_relative_l2": metrics["cutoff_force_relative_l2"],
        "cutoff_energy_absolute_error_kJ_mol": metrics[
            "cutoff_energy_absolute_error_kJ_mol"
        ],
        "cutoff_n_fresh_pairs": metrics["n_fresh_cutoff_pairs"],
        "cutoff_has_interactions": metrics["has_cutoff_interactions"],
    }


def measure_list_error(
    state,
    sim,
    ensemble,
    n_update,
    production_seed,
):
    """Measure stale-list error once at the largest possible list age."""

    ps = make_optimized_system(state, sim, n_update)
    step_function = optimized_step_function(ensemble)
    np.random.seed(production_seed)

    # For n_update=1, step zero is already a fresh-list control sample.
    sample_step = max(0, n_update - 1)

    for step in range(1, sample_step + 1):
        step_function(ps, sim, step, n_update)

    list_age = sample_step - ps.neighbour_list_step
    metrics = neighbour_list_error_metrics(ps, sim)

    return {
        "list_sample_step": sample_step,
        "list_age_steps": list_age,
        "list_force_relative_l2": metrics["list_force_relative_l2"],
        "list_energy_absolute_error_kJ_mol": metrics[
            "list_energy_absolute_error_kJ_mol"
        ],
        "total_force_relative_l2_at_list_sample": metrics[
            "total_force_relative_l2"
        ],
        "list_n_fresh_cutoff_pairs": metrics["n_fresh_cutoff_pairs"],
        "list_n_stale_cutoff_pairs": metrics["n_stale_cutoff_pairs"],
        "n_missing_pairs": metrics["n_missing_pairs"],
        "missed_pair_fraction": metrics["missed_pair_fraction"],
        "list_sample_has_interactions": metrics["has_cutoff_interactions"],
    }


def measure_list_error_over_cycles(
    state,
    sim,
    ensemble,
    n_update,
    production_seed,
    n_cycles,
):
    """Measure stale-list errors before the rebuild in several list cycles.

    For n_update > 1, measurements are taken at steps
    n_update - 1, 2*n_update - 1, ... . At those steps the stored list has
    its maximum possible age. For n_update == 1, every step is a fresh-list
    control measurement with list age zero.
    """

    if n_cycles < 1:
        raise ValueError("n_cycles must be at least 1")

    ps = make_optimized_system(state, sim, n_update)
    step_function = optimized_step_function(ensemble)
    np.random.seed(production_seed)

    samples = []
    final_step = n_cycles * n_update

    for step in range(1, final_step + 1):
        step_function(ps, sim, step, n_update)

        # The next MD step rebuilds the list. Therefore the present step
        # represents the largest list age within the current cycle.
        is_oldest_list = step % n_update == n_update - 1

        if not is_oldest_list:
            continue

        metrics = neighbour_list_error_metrics(ps, sim)

        samples.append(
            {
                "validation_cycle": len(samples) + 1,
                "list_sample_step": step,
                "list_age_steps": step - ps.neighbour_list_step,
                "list_force_relative_l2": metrics[
                    "list_force_relative_l2"
                ],
                "list_energy_absolute_error_kJ_mol": metrics[
                    "list_energy_absolute_error_kJ_mol"
                ],
                "total_force_relative_l2": metrics[
                    "total_force_relative_l2"
                ],
                "n_fresh_cutoff_pairs": metrics[
                    "n_fresh_cutoff_pairs"
                ],
                "n_stale_cutoff_pairs": metrics[
                    "n_stale_cutoff_pairs"
                ],
                "n_missing_pairs": metrics["n_missing_pairs"],
                "missed_pair_fraction": metrics[
                    "missed_pair_fraction"
                ],
                "has_cutoff_interactions": metrics[
                    "has_cutoff_interactions"
                ],
            }
        )

    if len(samples) != n_cycles:
        raise RuntimeError(
            "Unexpected number of list-error samples: "
            f"expected {n_cycles}, received {len(samples)}"
        )

    return samples


def create_equilibrium_states():
    """Create one reproducible common NVT state for each density and seed."""

    states = {}
    history = []

    print("Creating common equilibrated states...", flush=True)

    for density_factor, seed in product(density_factors, seeds):
        np.random.seed(seed)
        box_length = box_length_start / density_factor ** (1.0 / 3.0)
        r_cut = equilibration_cutoff_factor * sigma_argon

        sim = make_optimized_simulation(
            box_length,
            r_cut,
            n_equil_steps,
        )

        ps = ParticleSystem(n_particles)
        ps.mass[:] = mass_argon
        ps.sigma[:] = sigma_argon
        ps.epsilon[:] = epsilon_argon
        initialize_positions(ps, sim.box_length)
        initialize_velocities(ps, sim.temperature)
        update_neighbour_list(
            ps,
            sim,
            step=0,
            n_update=equilibration_n_update,
        )
        calculate_force(ps, sim)

        for step in range(1, n_equil_steps + 1):
            simulate_NVT_step(
                ps,
                sim,
                step,
                equilibration_n_update,
            )

            if step % equilibration_sample_interval == 0:
                history.append(
                    {
                        "density_factor": density_factor,
                        "seed": seed,
                        "step": step,
                        "temperature_K": instantaneous_temperature(ps),
                        "potential_energy_kJ_mol": potential_energy(ps, sim),
                    }
                )

        states[(density_factor, seed)] = {
            "position": ps.position.copy(),
            "velocity": ps.velocity.copy(),
        }

        print(
            "Equilibrated state finished:",
            f"density factor={density_factor}",
            f"seed={seed}",
            flush=True,
        )

    return states, pd.DataFrame(history)


def save_heatmap(table, x_name, y_name, z_name, ensemble=None):
    """Save one mean response surface; raw and summary CSVs remain authoritative."""

    subset = table
    prefix = ""

    if ensemble is not None:
        subset = table[table["ensemble"] == ensemble]
        prefix = f"{ensemble}__"

    data = subset.pivot_table(
        index=y_name,
        columns=x_name,
        values=z_name,
        aggfunc="mean",
    )

    if data.empty:
        return

    plt.figure(figsize=(10, 7))
    image = plt.imshow(data.values, aspect="auto", origin="lower")
    plt.colorbar(image, label=z_name)
    plt.xticks(range(len(data.columns)), [str(x) for x in data.columns], rotation=45)
    plt.yticks(range(len(data.index)), [str(y) for y in data.index])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(f"{z_name}: {x_name} vs. {y_name}")

    for y_index in range(len(data.index)):
        for x_index in range(len(data.columns)):
            value = data.iloc[y_index, x_index]
            label = format(value, ".3g") if np.isfinite(value) else "NaN"
            plt.text(x_index, y_index, label, ha="center", va="center")

    plt.tight_layout()
    filename = f"{prefix}{z_name}__{x_name}__{y_name}.png"
    plt.savefig(output_directory / filename, dpi=300)
    plt.close()


def create_summaries(results, cutoff_rows):
    """Aggregate physical replicates without averaging identifiers such as seed."""

    grouped = results.groupby(
        ["ensemble", "n_update", "cutoff_factor", "density_factor"],
        as_index=False,
    )

    summary = grouped.agg(
        n_seeds=("seed", "nunique"),
        speedup_median=("speedup_vs_original", "median"),
        speedup_mean=("speedup_vs_original", "mean"),
        optimized_runtime_median_s=("optimized_runtime_s", "median"),
        list_error_mean=("list_force_relative_l2", "mean"),
        list_error_median=("list_force_relative_l2", "median"),
        list_error_max=("list_force_relative_l2", "max"),
        list_error_valid=("list_force_relative_l2", "count"),
        missing_pairs_max=("n_missing_pairs", "max"),
        missed_pair_fraction_max=("missed_pair_fraction", "max"),
        cutoff_error_mean=("cutoff_force_relative_l2", "mean"),
        cutoff_error_max=("cutoff_force_relative_l2", "max"),
        cutoff_error_valid=("cutoff_force_relative_l2", "count"),
    )

    cutoff_table = pd.DataFrame(cutoff_rows)
    cutoff_summary = cutoff_table.groupby(
        ["cutoff_factor", "density_factor"],
        as_index=False,
    ).agg(
        n_seeds=("seed", "nunique"),
        cutoff_error_mean=("cutoff_force_relative_l2", "mean"),
        cutoff_error_median=("cutoff_force_relative_l2", "median"),
        cutoff_error_max=("cutoff_force_relative_l2", "max"),
        cutoff_error_valid=("cutoff_force_relative_l2", "count"),
        cutoff_energy_error_mean_kJ_mol=(
            "cutoff_energy_absolute_error_kJ_mol",
            "mean",
        ),
    )

    return summary, cutoff_table, cutoff_summary

def select_robust_optimum(summary):
    """
    Select one robust cutoff/n_update combination.

    A configuration is accepted only if it satisfies the
    accuracy limits for every tested density.
    """

    evaluated = summary.copy()
    expected_seeds = len(seeds)

    # Check whether every seed produced valid measurements.
    evaluated["accuracy_ok"] = (
        (evaluated["n_seeds"] == expected_seeds)
        & (evaluated["list_error_valid"] == expected_seeds)
        & (evaluated["cutoff_error_valid"] == expected_seeds)
        & np.isfinite(evaluated["speedup_median"])
        & np.isfinite(evaluated["list_error_max"])
        & np.isfinite(evaluated["cutoff_error_max"])
        & (
            evaluated["cutoff_error_max"]
            <= MAX_CUTOFF_FORCE_ERROR
        )
        & (
            evaluated["list_error_max"]
            <= MAX_LIST_FORCE_ERROR
        )
        & (
            evaluated["missing_pairs_max"]
            <= MAX_MISSING_PAIRS
        )
    )

    # Combine the results from all tested densities.
    robust_candidates = evaluated.groupby(
        ["ensemble", "n_update", "cutoff_factor"],
        as_index=False,
    ).agg(
        n_densities=("density_factor", "nunique"),

        # Conservative performance:
        # use the worst speedup among all densities.
        speedup_worst_density=("speedup_median", "min"),

        speedup_median_all_densities=(
            "speedup_median",
            "median",
        ),

        cutoff_error_worst=(
            "cutoff_error_max",
            "max",
        ),

        list_error_worst=(
            "list_error_max",
            "max",
        ),

        missing_pairs_worst=(
            "missing_pairs_max",
            "max",
        ),

        accurate_at_all_densities=(
            "accuracy_ok",
            "all",
        ),
    )

    # A robust candidate must have been tested and accepted
    # at every density.
    robust_candidates["robust_ok"] = (
        (
            robust_candidates["n_densities"]
            == len(density_factors)
        )
        & robust_candidates["accurate_at_all_densities"]
    )

    allowed = robust_candidates[robust_candidates["robust_ok"]].copy()

    if allowed.empty:
        robust_optimum = allowed.copy()

    else:
        # One optimum per ensemble. With only NVE this produces
        # exactly one result row.
        best_indices = allowed.groupby(
            "ensemble"
        )["speedup_worst_density"].idxmax()

        robust_optimum = (allowed.loc[best_indices].reset_index(drop=True))

    return evaluated, robust_candidates, robust_optimum


def validate_robust_candidates(
    robust_candidates,
    equilibrium_states,
):
    """Validate the fastest screening candidates over many list cycles."""

    candidate_rows = (
        robust_candidates[robust_candidates["robust_ok"]]
        .sort_values(
            ["ensemble", "speedup_worst_density"],
            ascending=[True, False],
        )
        .groupby("ensemble", as_index=False, group_keys=False)
        .head(n_list_validation_candidates)
    )

    sample_rows = []

    for candidate in candidate_rows.itertuples(index=False):
        ensemble = candidate.ensemble
        n_update = int(candidate.n_update)
        cutoff_factor = float(candidate.cutoff_factor)
        r_cut = cutoff_factor * sigma_argon

        for density_factor, seed in product(density_factors, seeds):
            state = equilibrium_states[(density_factor, seed)]
            box_length = (
                box_length_start
                / density_factor ** (1.0 / 3.0)
            )
            validation_steps = n_list_validation_cycles * n_update
            sim = make_optimized_simulation(
                box_length,
                r_cut,
                validation_steps,
            )
            production_seed = 100000 + seed

            cycle_samples = measure_list_error_over_cycles(
                state,
                sim,
                ensemble,
                n_update,
                production_seed,
                n_list_validation_cycles,
            )

            for sample in cycle_samples:
                sample_rows.append(
                    {
                        "ensemble": ensemble,
                        "n_update": n_update,
                        "cutoff_factor": cutoff_factor,
                        "cutoff_radius_nm": r_cut,
                        "density_factor": density_factor,
                        "seed": seed,
                        "state_id": f"{density_factor}:{seed}",
                        **sample,
                    }
                )

    samples = pd.DataFrame(sample_rows)

    if samples.empty:
        return samples, pd.DataFrame(), pd.DataFrame()

    candidate_keys = ["ensemble", "n_update", "cutoff_factor"]

    # Every density/seed state must contain at least one finite relative error.
    state_validity = (
        samples.groupby(candidate_keys + ["state_id"], as_index=False)
        .agg(
            valid_error_samples=(
                "list_force_relative_l2",
                "count",
            )
        )
    )
    state_validity["state_has_valid_error"] = (
        state_validity["valid_error_samples"] > 0
    )
    state_counts = (
        state_validity.groupby(candidate_keys, as_index=False)
        .agg(
            n_states=("state_id", "nunique"),
            n_valid_states=("state_has_valid_error", "sum"),
        )
    )

    validation_summary = (
        samples.groupby(candidate_keys, as_index=False)
        .agg(
            n_densities=("density_factor", "nunique"),
            n_samples=("validation_cycle", "count"),
            list_error_valid=("list_force_relative_l2", "count"),
            list_error_mean=("list_force_relative_l2", "mean"),
            list_error_p95=(
                "list_force_relative_l2",
                lambda values: values.quantile(0.95),
            ),
            list_error_max=("list_force_relative_l2", "max"),
            list_energy_error_max_kJ_mol=(
                "list_energy_absolute_error_kJ_mol",
                "max",
            ),
            missing_pairs_max=("n_missing_pairs", "max"),
            missed_pair_fraction_max=("missed_pair_fraction", "max"),
        )
        .merge(state_counts, on=candidate_keys, how="left")
    )

    performance_columns = candidate_rows[
        candidate_keys
        + [
            "speedup_worst_density",
            "speedup_median_all_densities",
            "cutoff_error_worst",
        ]
    ]
    validation_summary = validation_summary.merge(
        performance_columns,
        on=candidate_keys,
        how="left",
    )

    expected_states = len(density_factors) * len(seeds)
    validation_summary["validation_ok"] = (
        (validation_summary["n_densities"] == len(density_factors))
        & (validation_summary["n_states"] == expected_states)
        & (validation_summary["n_valid_states"] == expected_states)
        & np.isfinite(validation_summary["list_error_p95"])
        & np.isfinite(validation_summary["list_error_max"])
        & (
            validation_summary["list_error_p95"]
            <= MAX_LIST_FORCE_ERROR
        )
        & (
            validation_summary["list_error_max"]
            <= MAX_LIST_FORCE_ERROR
        )
        & (
            validation_summary["missing_pairs_max"]
            <= MAX_MISSING_PAIRS
        )
    )

    allowed = validation_summary[validation_summary["validation_ok"]].copy()

    if allowed.empty:
        validated_optimum = allowed.copy()
    else:
        best_indices = allowed.groupby("ensemble")["speedup_worst_density"].idxmax()
        validated_optimum = (allowed.loc[best_indices].reset_index(drop=True))

    return samples, validation_summary, validated_optimum


def main():
    ensembles = validate_ensembles()
    output_directory.mkdir(parents=True, exist_ok=True)
    original_core = load_original_core(original_core_path)

    equilibrium_states, equilibration_history = create_equilibrium_states()
    equilibration_history.to_csv(
        output_directory / "equilibration_history.csv",
        index=False,
    )

    combinations = list(
        product(
            ensembles,
            n_update_values,
            cutoff_factors,
            density_factors,
            seeds,
        )
    )
    random_order = np.random.default_rng(12345)
    random_order.shuffle(combinations)

    # The original has no cutoff or update interval, so one baseline is reused
    # for all matching optimized configurations.
    original_runtime_cache = {}

    # The pure cutoff error is measured on the same step-zero state and therefore
    # only once for each cutoff, density and seed (not once per n_update).
    cutoff_error_cache = {}
    cutoff_rows = []
    rows = []

    for run_number, combination in enumerate(combinations, start=1):
        ensemble, n_update, cutoff_factor, density_factor, seed = combination
        production_seed = 100000 + seed
        box_length = box_length_start / density_factor ** (1.0 / 3.0)
        r_cut = cutoff_factor * sigma_argon
        state = equilibrium_states[(density_factor, seed)]

        sim = make_optimized_simulation(
            box_length,
            r_cut,
            n_timing_steps,
        )

        original_key = (ensemble, density_factor, seed)

        if original_key not in original_runtime_cache:
            original_runtime_cache[original_key] = median_original_runtime(
                original_core,
                state,
                box_length,
                ensemble,
                production_seed,
            )

        original_runtime = original_runtime_cache[original_key]
        optimized_runtime, timed_rebuilds = median_optimized_runtime(
            state,
            sim,
            ensemble,
            n_update,
            production_seed,
        )

        cutoff_key = (cutoff_factor, density_factor, seed)

        if cutoff_key not in cutoff_error_cache:
            cutoff_metrics = measure_cutoff_error(state, sim)
            cutoff_error_cache[cutoff_key] = cutoff_metrics
            cutoff_rows.append(
                {
                    "cutoff_factor": cutoff_factor,
                    "cutoff_radius_nm": r_cut,
                    "density_factor": density_factor,
                    "seed": seed,
                    **cutoff_metrics,
                }
            )

        cutoff_metrics = cutoff_error_cache[cutoff_key]
        list_metrics = measure_list_error(
            state,
            sim,
            ensemble,
            n_update,
            production_seed,
        )

        speedup = original_runtime / optimized_runtime

        row = {
            "ensemble": ensemble,
            "n_update": n_update,
            "cutoff_factor": cutoff_factor,
            "cutoff_radius_nm": r_cut,
            "density_factor": density_factor,
            "density_g_cm3": (
                n_particles
                * mass_argon
                / Avogadro
                / (box_length**3 * 1e-21)
            ),
            "seed": seed,
            "production_seed": production_seed,
            "timing_steps": n_timing_steps,
            "timing_repeats": timing_repeats,
            "original_runtime_s": original_runtime,
            "optimized_runtime_s": optimized_runtime,
            "original_steps_per_s": n_timing_steps / original_runtime,
            "optimized_steps_per_s": n_timing_steps / optimized_runtime,
            "speedup_vs_original": speedup,
            "timed_neighbour_rebuilds": timed_rebuilds,
            **cutoff_metrics,
            **list_metrics,
        }

        rows.append(row)
        results = pd.DataFrame(rows)
        results.to_csv(output_directory / "doe_ergebnisse.csv", index=False)

        print(
            f"{run_number}/{len(combinations)}, "
            f"ensemble={ensemble}, n_update={n_update}, "
            f"cutoff={r_cut:.3f} nm, density factor={density_factor}, "
            f"speedup={speedup:.3f}, "
            f"list error={list_metrics['list_force_relative_l2']:.3e}, "
            f"cutoff error={cutoff_metrics['cutoff_force_relative_l2']:.3e}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    summary, cutoff_table, cutoff_summary = create_summaries(results, cutoff_rows)
    summary, robust_candidates, robust_optimum = (select_robust_optimum(summary))

    (
        list_validation_samples,
        list_validation_summary,
        validated_optimum,
    ) = validate_robust_candidates(
        robust_candidates,
        equilibrium_states,
    )

    robust_candidates.to_csv(output_directory / "robuste_kandidaten.csv", index=False)
    robust_optimum.to_csv(output_directory / "robustes_optimum_screening.csv", index=False)
    list_validation_samples.to_csv(output_directory / "listenfehler_validierung_rohdaten.csv", index=False)
    list_validation_summary.to_csv(output_directory / "listenfehler_validierung_mittelwerte.csv", index=False)
    validated_optimum.to_csv(output_directory / "robustes_optimum.csv", index=False)

    if validated_optimum.empty:
        print(
            "No screening candidate passed the multi-cycle validation.",
            flush=True,
        )
    else:
        print("\nValidated robust optimum:", flush=True)
        print(validated_optimum.to_string(index=False), flush=True)

    summary.to_csv(output_directory / "doe_mittelwerte.csv", index=False)
    cutoff_table.to_csv(output_directory / "cutoff_fehler.csv", index=False)
    cutoff_summary.to_csv(output_directory / "cutoff_fehler_mittelwerte.csv", index=False)

    for ensemble in ensembles:
        save_heatmap(
            results,
            "n_update",
            "cutoff_factor",
            "speedup_vs_original",
            ensemble,
        )
        save_heatmap(
            results,
            "n_update",
            "cutoff_factor",
            "list_force_relative_l2",
            ensemble,
        )
        save_heatmap(
            results,
            "n_update",
            "cutoff_factor",
            "missed_pair_fraction",
            ensemble,
        )

    save_heatmap(
        cutoff_table,
        "cutoff_factor",
        "density_factor",
        "cutoff_force_relative_l2",
    )


if __name__ == "__main__":
    main()
