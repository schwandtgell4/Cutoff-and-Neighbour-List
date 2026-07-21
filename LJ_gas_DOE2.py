import importlib.util
import sys
import time
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pandas as pd
from scipy.constants import R

from LJ_gas import (
    ParticleSystem,
    SimulationParameters,
    calculate_all_pairs_reference,
    calculate_force,
    initialize_velocities,
    instantaneous_temperature,
    simulate_NVT_step,
    update_neighbour_list,
)


# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
n_particles = 2000
mass_argon = 39.95                 # u = 1e-3 kg/mol
sigma_argon = 0.34                 # nm
epsilon_argon = 120.0 * R * 1e-3  # kJ/mol

dt = 0.01                          # ps
temperature = 300.0                # K
box_length = 20.0                  # nm
tau_thermostat = 1.0               # ps
rij_min = 1e-2                     # nm
minimum_initial_distance = 1.0 * sigma_argon
maximum_position_attempts = 10_000

n_equil_steps = 500
n_timing_steps = 100
timing_repeats = 3
error_cycles = 3

n_update_values = list(range(1, 50))
cutoff_factors = [1, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]
seeds = [1, 2, 3]

MAX_TOTAL_FORCE_ERROR = 1e-2

base_directory = Path(__file__).resolve().parent
original_core_path = base_directory / "Original" / "LJ_gas_original.py"
output_directory = base_directory / "doe2_ergebnisse"


def save_csv(table, filename):
    """Save one Excel-friendly CSV using German number formatting."""

    table.to_csv(
        output_directory / filename,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )


def load_original_core():
    """Load the unchanged reference code from the Original directory."""

    if not original_core_path.is_file():
        raise FileNotFoundError(
            f"Original code not found: {original_core_path}"
        )

    name = "_lj_gas_original_for_light_doe"
    specification = importlib.util.spec_from_file_location(
        name,
        original_core_path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load {original_core_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def make_simulation(r_cut, n_steps):
    return SimulationParameters(
        dt=dt,
        n_steps=n_steps,
        temperature=temperature,
        box_length=box_length,
        tau_thermostat=tau_thermostat,
        rij_min=rij_min,
        r_cut=r_cut,
    )


def initialize_safe_positions(ps, box_length):
    """Place particles randomly without unphysical LJ overlaps."""

    positions = np.empty((ps.n, 3), dtype=float)
    minimum_distance_squared = minimum_initial_distance**2

    for particle_index in range(ps.n):
        for _ in range(maximum_position_attempts):
            candidate = np.random.uniform(0.0, box_length, size=3)

            if particle_index == 0:
                positions[particle_index] = candidate
                break

            displacement = positions[:particle_index] - candidate
            displacement -= box_length * np.rint(
                displacement / box_length
            )
            distance_squared = np.einsum(
                "ij,ij->i",
                displacement,
                displacement,
            )

            if np.all(distance_squared >= minimum_distance_squared):
                positions[particle_index] = candidate
                break
        else:
            raise RuntimeError(
                "Could not generate non-overlapping initial positions. "
                "Increase the box length or reduce the minimum distance."
            )

    ps.position[:] = positions


def validate_state(ps, context, check_temperature=False):
    """Stop immediately if a simulation state becomes unstable."""

    for name in ("position", "velocity", "force"):
        if not np.all(np.isfinite(getattr(ps, name))):
            raise FloatingPointError(
                f"Non-finite {name} values during {context}"
            )

    if check_temperature:
        current_temperature = instantaneous_temperature(ps)

        if (
            not np.isfinite(current_temperature)
            or current_temperature > 10.0 * temperature
        ):
            raise FloatingPointError(
                f"Unstable temperature during {context}: "
                f"{current_temperature:.3e} K"
            )


def make_system(state, sim, n_update):
    """Create a fresh optimized system from a stored equilibrium state."""

    ps = ParticleSystem(n_particles)
    ps.mass[:] = mass_argon
    ps.sigma[:] = sigma_argon
    ps.epsilon[:] = epsilon_argon
    ps.position[:] = state["position"]
    ps.velocity[:] = state["velocity"]
    update_neighbour_list(ps, sim, step=0, n_update=n_update)
    calculate_force(ps, sim)
    return ps


def create_equilibrium_states():
    """Create one common NVT starting state for each seed."""

    states = {}
    r_cut = max(cutoff_factors) * sigma_argon
    sim = make_simulation(r_cut, n_equil_steps)

    print("Creating equilibrium states...", flush=True)

    for seed in seeds:
        np.random.seed(seed)
        ps = ParticleSystem(n_particles)
        ps.mass[:] = mass_argon
        ps.sigma[:] = sigma_argon
        ps.epsilon[:] = epsilon_argon
        initialize_safe_positions(ps, sim.box_length)
        initialize_velocities(ps, sim.temperature)
        update_neighbour_list(ps, sim, step=0, n_update=1)
        calculate_force(ps, sim)
        validate_state(
            ps,
            f"initialization for seed {seed}",
            check_temperature=True,
        )

        for step in range(1, n_equil_steps + 1):
            simulate_NVT_step(ps, sim, step, n_update=1)
            validate_state(
                ps,
                f"equilibration for seed {seed}, step {step}",
                check_temperature=True,
            )

        states[seed] = {
            "position": ps.position.copy(),
            "velocity": ps.velocity.copy(),
        }
        print(f"  seed {seed} finished", flush=True)

    return states


def median_original_runtime(original_core, state, production_seed):
    """Time only the MD loop of the unchanged all-pairs code."""

    runtimes = []

    for _ in range(timing_repeats):
        sim = original_core.SimulationParameters(
            dt=dt,
            n_steps=n_timing_steps,
            temperature=temperature,
            box_length=box_length,
            tau_thermostat=tau_thermostat,
            rij_min=rij_min,
        )
        ps = original_core.ParticleSystem(n_particles)
        ps.mass[:] = mass_argon
        ps.sigma[:] = sigma_argon
        ps.epsilon[:] = epsilon_argon
        ps.position[:] = state["position"]
        ps.velocity[:] = state["velocity"]
        original_core.calculate_force(ps, sim)

        np.random.seed(production_seed)
        start = time.perf_counter()

        for _ in range(n_timing_steps):
            original_core.simulate_NVT_step(ps, sim)

        runtime = time.perf_counter() - start
        validate_state(ps, "original-code timing")
        runtimes.append(runtime)

    return float(np.median(runtimes))


def median_optimized_runtime(state, r_cut, n_update, production_seed):
    """Time only the MD loop of the cutoff/neighbour-list code."""

    runtimes = []
    sim = make_simulation(r_cut, n_timing_steps)

    for _ in range(timing_repeats):
        ps = make_system(state, sim, n_update)
        np.random.seed(production_seed)
        start = time.perf_counter()

        for step in range(1, n_timing_steps + 1):
            simulate_NVT_step(ps, sim, step, n_update)

        runtime = time.perf_counter() - start
        validate_state(ps, "optimized-code timing")
        runtimes.append(runtime)

    return float(np.median(runtimes))


def current_force_error_metrics(ps, sim):
    """Return relative and absolute errors at the current positions."""

    stale_force = ps.force.copy()
    stale_energy = float(ps.current_potential_energy)

    fresh_force, fresh_energy, fresh_pairs = (
        calculate_all_pairs_reference(
            ps,
            sim,
            apply_cutoff=True,
        )
    )
    full_force, full_energy, _ = calculate_all_pairs_reference(
        ps,
        sim,
        apply_cutoff=False,
    )

    def force_errors(value, reference):
        absolute = float(np.linalg.norm(value - reference))
        reference_norm = float(np.linalg.norm(reference))
        relative = (
            absolute / reference_norm
            if reference_norm > 0.0
            else np.nan
        )
        return float(relative), absolute

    list_relative, list_absolute = force_errors(
        stale_force,
        fresh_force,
    )
    cutoff_relative, cutoff_absolute = force_errors(
        fresh_force,
        full_force,
    )
    total_relative, total_absolute = force_errors(
        stale_force,
        full_force,
    )

    stale_pairs = ps.neighbour_pairs

    if stale_pairs.shape[0] > 0:
        stale_i = stale_pairs[:, 0]
        stale_j = stale_pairs[:, 1]
        stale_rij = ps.position[stale_i] - ps.position[stale_j]
        stale_rij -= sim.box_length * np.rint(
            stale_rij / sim.box_length
        )
        stale_r_squared = np.einsum(
            "ij,ij->i",
            stale_rij,
            stale_rij,
        )
        stale_cutoff_pairs = stale_pairs[
            stale_r_squared <= sim.r_cut * sim.r_cut
        ]
    else:
        stale_cutoff_pairs = np.empty((0, 2), dtype=np.int64)

    fresh_pair_ids = fresh_pairs[:, 0] * ps.n + fresh_pairs[:, 1]
    stale_pair_ids = (
        stale_cutoff_pairs[:, 0] * ps.n
        + stale_cutoff_pairs[:, 1]
    )
    n_missing_pairs = int(
        np.setdiff1d(
            fresh_pair_ids,
            stale_pair_ids,
            assume_unique=False,
        ).size
    )
    n_fresh_pairs = int(fresh_pairs.shape[0])
    missed_pair_fraction = (
        n_missing_pairs / n_fresh_pairs
        if n_fresh_pairs > 0
        else 0.0
    )

    return {
        "list_force_relative_l2": list_relative,
        "cutoff_force_relative_l2": cutoff_relative,
        "total_force_relative_l2": total_relative,
        "list_force_absolute_l2_kJ_per_mol_nm": list_absolute,
        "cutoff_force_absolute_l2_kJ_per_mol_nm": cutoff_absolute,
        "total_force_absolute_l2_kJ_per_mol_nm": total_absolute,
        "list_energy_absolute_error_kJ_mol": abs(
            stale_energy - fresh_energy
        ),
        "cutoff_energy_absolute_error_kJ_mol": abs(
            fresh_energy - full_energy
        ),
        "total_energy_absolute_error_kJ_mol": abs(
            stale_energy - full_energy
        ),
        "n_missing_pairs": n_missing_pairs,
        "missed_pair_fraction": float(missed_pair_fraction),
    }


def measure_force_errors(state, r_cut, n_update, production_seed):
    """Measure errors at the oldest sampled neighbour-list states."""

    final_step = error_cycles * n_update
    sim = make_simulation(r_cut, final_step)
    ps = make_system(state, sim, n_update)
    np.random.seed(production_seed)
    samples = []

    for step in range(1, final_step + 1):
        simulate_NVT_step(ps, sim, step, n_update)

        # At k*n_update - 1 the stored list has its largest possible age.
        # With n_update == 1 every step is a fresh-list control measurement.
        if step % n_update != n_update - 1:
            continue

        samples.append(current_force_error_metrics(ps, sim))

    def finite_maximum(key):
        values = [sample[key] for sample in samples]
        finite_values = [
            value for value in values if np.isfinite(value)
        ]
        return (
            float(max(finite_values))
            if finite_values
            else np.nan
        )

    return {
        "list_force_relative_l2": finite_maximum(
            "list_force_relative_l2"
        ),
        "cutoff_force_relative_l2": finite_maximum(
            "cutoff_force_relative_l2"
        ),
        "total_force_relative_l2": finite_maximum(
            "total_force_relative_l2"
        ),
        "list_force_absolute_l2_kJ_per_mol_nm": finite_maximum(
            "list_force_absolute_l2_kJ_per_mol_nm"
        ),
        "cutoff_force_absolute_l2_kJ_per_mol_nm": finite_maximum(
            "cutoff_force_absolute_l2_kJ_per_mol_nm"
        ),
        "total_force_absolute_l2_kJ_per_mol_nm": finite_maximum(
            "total_force_absolute_l2_kJ_per_mol_nm"
        ),
        "list_energy_absolute_error_kJ_mol": finite_maximum(
            "list_energy_absolute_error_kJ_mol"
        ),
        "cutoff_energy_absolute_error_kJ_mol": finite_maximum(
            "cutoff_energy_absolute_error_kJ_mol"
        ),
        "total_energy_absolute_error_kJ_mol": finite_maximum(
            "total_energy_absolute_error_kJ_mol"
        ),
        "n_missing_pairs_max": int(
            max(sample["n_missing_pairs"] for sample in samples)
        ),
        "missed_pair_fraction_max": finite_maximum(
            "missed_pair_fraction"
        ),
    }


def create_summary(results):
    """Aggregate seeds and select the fastest sufficiently accurate pair."""

    summary = (
        results.groupby(
            ["n_update", "cutoff_factor"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            speedup_valid=("speedup", "count"),
            speedup_median=("speedup", "median"),
            list_error_max=("list_force_relative_l2", "max"),
            cutoff_error_max=("cutoff_force_relative_l2", "max"),
            total_error_valid=("total_force_relative_l2", "count"),
            total_error_max=("total_force_relative_l2", "max"),
            list_force_absolute_l2_max_kJ_per_mol_nm=(
                "list_force_absolute_l2_kJ_per_mol_nm",
                "max",
            ),
            cutoff_force_absolute_l2_max_kJ_per_mol_nm=(
                "cutoff_force_absolute_l2_kJ_per_mol_nm",
                "max",
            ),
            total_force_absolute_l2_max_kJ_per_mol_nm=(
                "total_force_absolute_l2_kJ_per_mol_nm",
                "max",
            ),
            list_energy_absolute_error_max_kJ_mol=(
                "list_energy_absolute_error_kJ_mol",
                "max",
            ),
            cutoff_energy_absolute_error_max_kJ_mol=(
                "cutoff_energy_absolute_error_kJ_mol",
                "max",
            ),
            total_energy_absolute_error_max_kJ_mol=(
                "total_energy_absolute_error_kJ_mol",
                "max",
            ),
            n_missing_pairs_max=("n_missing_pairs_max", "max"),
            missed_pair_fraction_max=(
                "missed_pair_fraction_max",
                "max",
            ),
        )
    )

    complete = (
        (summary["n_seeds"] == len(seeds))
        & (summary["speedup_valid"] == len(seeds))
        & (summary["total_error_valid"] == len(seeds))
        & np.isfinite(summary["speedup_median"])
        & np.isfinite(summary["total_error_max"])
    )
    summary["acceptable"] = (
        complete
        & (summary["total_error_max"] <= MAX_TOTAL_FORCE_ERROR)
    )
    acceptable = summary[summary["acceptable"]]

    if acceptable.empty:
        optimum = acceptable.copy()
    else:
        optimum = acceptable.nlargest(1, "speedup_median").copy()

    return summary, optimum


def pivot_summary(summary, column):
    return (
        summary.pivot(
            index="cutoff_factor",
            columns="n_update",
            values=column,
        )
        .reindex(index=cutoff_factors, columns=n_update_values)
    )


def save_heatmap(
    summary,
    column,
    title,
    colorbar_label,
    filename,
    logarithmic=False,
    value_scale=1.0,
    logarithmic_limits=None,
    caption="",
):
    """Save one readable cutoff-by-update heatmap."""

    matrix = pivot_summary(summary, column)
    values = matrix.to_numpy(dtype=float) * value_scale
    fig, axis = plt.subplots(figsize=(10, 7.5))
    image = None
    colour_map = plt.get_cmap(
        "magma" if logarithmic else "viridis"
    ).copy()
    colour_map.set_bad("#d9d9d9")

    if logarithmic:
        finite = np.isfinite(values)
        floor = MAX_TOTAL_FORCE_ERROR * value_scale * 1e-4

        if finite.any():
            plotted = np.where(
                finite,
                np.maximum(values, floor),
                np.nan,
            )

            if logarithmic_limits is None:
                minimum = floor
                maximum = max(
                    float(np.nanmax(plotted)),
                    floor * 10.0,
                )
            else:
                minimum, maximum = logarithmic_limits

            image = axis.imshow(
                np.ma.masked_invalid(plotted),
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=colour_map,
                norm=LogNorm(vmin=minimum, vmax=maximum),
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No finite relative errors",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    else:
        image = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=colour_map,
        )

    if image is not None:
        colour_bar = fig.colorbar(
            image,
            ax=axis,
            pad=0.025,
        )
        colour_bar.set_label(colorbar_label)

    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns)
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels([f"{value:g}" for value in matrix.index])

    # Thin neutral cell borders make values easier to associate with
    # both parameter axes without adding acceptance contours or markers.
    axis.set_xticks(
        np.arange(-0.5, len(matrix.columns), 1.0),
        minor=True,
    )
    axis.set_yticks(
        np.arange(-0.5, len(matrix.index), 1.0),
        minor=True,
    )
    axis.grid(
        which="minor",
        color="white",
        linewidth=0.6,
    )
    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )
    axis.set_xlabel(
        r"Neighbour-list update interval $n_\mathrm{update}$"
    )
    axis.set_ylabel(r"Cutoff factor $r_\mathrm{cut}/\sigma$")
    axis.set_title(title, pad=12)

    if caption:
        fig.text(
            0.5,
            0.015,
            caption,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

    fig.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    fig.savefig(output_directory / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def shared_error_plot_limits(summary):
    """Return one logarithmic percentage scale for all error plots."""

    values = (
        summary[
            [
                "list_error_max",
                "cutoff_error_max",
                "total_error_max",
            ]
        ]
        .to_numpy(dtype=float)
        * 100.0
    )
    finite_positive = values[
        np.isfinite(values) & (values > 0.0)
    ]
    minimum = MAX_TOTAL_FORCE_ERROR * 100.0 * 1e-4

    if finite_positive.size == 0:
        maximum = minimum * 10.0
    else:
        maximum = max(
            float(np.max(finite_positive)),
            MAX_TOTAL_FORCE_ERROR * 100.0 * 10.0,
        )

    return minimum, maximum


def save_error_comparison(
    summary,
    logarithmic_limits,
    system_caption,
):
    """Compare list, cutoff and total errors on one shared scale."""

    panels = [
        (
            "list_error_max",
            "Neighbour-list reuse",
        ),
        (
            "cutoff_error_max",
            "Cutoff truncation",
        ),
        (
            "total_error_max",
            "Total versus original",
        ),
    ]
    minimum, maximum = logarithmic_limits
    colour_map = plt.get_cmap("magma").copy()
    colour_map.set_bad("#d9d9d9")
    norm = LogNorm(vmin=minimum, vmax=maximum)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(18, 7),
        sharex=True,
        sharey=True,
    )
    image = None

    for axis, (column, panel_title) in zip(axes, panels):
        matrix = pivot_summary(summary, column)
        values = matrix.to_numpy(dtype=float) * 100.0
        plotted = np.where(
            np.isfinite(values),
            np.maximum(values, minimum),
            np.nan,
        )
        image = axis.imshow(
            np.ma.masked_invalid(plotted),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            cmap=colour_map,
            norm=norm,
        )

        axis.set_xticks(range(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns)
        axis.set_yticks(range(len(matrix.index)))
        axis.set_yticklabels(
            [f"{value:g}" for value in matrix.index]
        )
        axis.set_xticks(
            np.arange(-0.5, len(matrix.columns), 1.0),
            minor=True,
        )
        axis.set_yticks(
            np.arange(-0.5, len(matrix.index), 1.0),
            minor=True,
        )
        axis.grid(
            which="minor",
            color="white",
            linewidth=0.5,
        )
        axis.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )
        axis.set_xlabel(r"Update interval $n_\mathrm{update}$")
        axis.set_title(panel_title, pad=10)

    axes[0].set_ylabel(r"Cutoff factor $r_\mathrm{cut}/\sigma$")

    fig.suptitle(
        "Comparison of maximum relative force errors",
        y=0.965,
    )
    fig.text(
        0.5,
        0.035,
        (
            system_caption
            + " Maximum across seeds and sampled list ages."
            + " Grey cells are undefined; zero errors use the lower colour limit."
            + " Total vector error is not the arithmetic sum of the components."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.88,
        bottom=0.15,
        top=0.88,
        wspace=0.10,
    )

    if image is not None:
        colour_bar_axis = fig.add_axes(
            [0.90, 0.15, 0.014, 0.73]
        )
        colour_bar = fig.colorbar(
            image,
            cax=colour_bar_axis,
        )
        colour_bar.set_label("Maximum relative force error [%]")

    fig.savefig(
        output_directory / "force_error_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_performance_landscape(
    summary,
    logarithmic_limits,
    system_caption,
):
    """Save a 3D response surface for speedup and total force error."""

    speedup_matrix = (
        pivot_summary(summary, "speedup_median")
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
    )

    if speedup_matrix.empty:
        raise ValueError("no finite DOE values for the 3D landscape")

    error_matrix = pivot_summary(
        summary,
        "total_error_max",
    ).reindex(
        index=speedup_matrix.index,
        columns=speedup_matrix.columns,
    )
    speedup = speedup_matrix.to_numpy(dtype=float)
    total_error_percent = (
        error_matrix.to_numpy(dtype=float) * 100.0
    )
    cutoff_grid, update_grid = np.meshgrid(
        speedup_matrix.index.to_numpy(dtype=float),
        speedup_matrix.columns.to_numpy(dtype=float),
        indexing="ij",
    )
    minimum_error, maximum_error = logarithmic_limits
    error_norm = LogNorm(
        vmin=minimum_error,
        vmax=maximum_error,
    )
    colour_map = plt.get_cmap("magma").copy()
    colour_map.set_bad("#d9d9d9")
    plotted_error = np.where(
        np.isfinite(total_error_percent),
        np.maximum(total_error_percent, minimum_error),
        np.nan,
    )
    surface_colours = colour_map(error_norm(plotted_error))
    valid = np.isfinite(speedup) & np.isfinite(total_error_percent)

    fig = plt.figure(figsize=(12, 8))
    axis = fig.add_subplot(111, projection="3d")

    axis.plot_surface(
        cutoff_grid,
        update_grid,
        np.ma.masked_invalid(speedup),
        facecolors=surface_colours,
        edgecolor="#666666",
        linewidth=0.35,
        antialiased=True,
        shade=False,
        alpha=0.92,
    )
    axis.scatter(
        cutoff_grid[valid],
        update_grid[valid],
        speedup[valid],
        c=plotted_error[valid],
        cmap=colour_map,
        norm=error_norm,
        s=13,
        edgecolors="black",
        linewidths=0.25,
        depthshade=False,
    )

    x_end = float(np.max(cutoff_grid)) * 1.03
    y_end = float(np.max(update_grid)) * 1.03
    z_end = 1.1
    if valid.any():
        finite_speedup = speedup[valid]
        upper = max(1.0, float(np.max(finite_speedup)))
        z_end = upper * 1.06

    cutoff_plane, update_plane = np.meshgrid(
        [0.0, x_end],
        [0.0, y_end],
        indexing="ij",
    )
    axis.plot_surface(
        cutoff_plane,
        update_plane,
        np.ones_like(cutoff_plane),
        color="#777777",
        alpha=0.20,
        linewidth=0.0,
        shade=False,
    )

    # Draw three explicit axes from one visible coordinate origin.
    axis.plot(
        [0.0, x_end],
        [0.0, 0.0],
        [0.0, 0.0],
        color="#333333",
        linewidth=1.3,
    )
    axis.plot(
        [0.0, 0.0],
        [0.0, y_end],
        [0.0, 0.0],
        color="#333333",
        linewidth=1.3,
    )
    axis.plot(
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, z_end],
        color="#333333",
        linewidth=1.3,
    )
    axis.scatter(
        [0.0],
        [0.0],
        [0.0],
        color="#333333",
        s=18,
        depthshade=False,
    )

    def sparse_ticks(values, maximum_count=7):
        values = np.asarray(values, dtype=float)

        if values.size <= maximum_count:
            return values

        indices = np.unique(
            np.linspace(
                0,
                values.size - 1,
                maximum_count,
            ).astype(int)
        )
        return values[indices]

    axis.set_xlabel(r"Cutoff factor $r_\mathrm{cut}/\sigma$", labelpad=10)
    axis.set_ylabel(
        r"Update interval $n_\mathrm{update}$",
        labelpad=10,
    )
    z_axis_label = (
        r"Median speedup $S=t_\mathrm{orig}/t_\mathrm{opt}$ [$\times$]"
    )
    axis.set_xlim(0.0, x_end)
    axis.set_ylim(0.0, y_end)
    axis.set_zlim(0.0, z_end)
    axis.set_xticks(
        np.concatenate(
            ([0.0], sparse_ticks(speedup_matrix.index))
        )
    )
    axis.set_yticks(
        np.concatenate(
            ([0.0], sparse_ticks(speedup_matrix.columns))
        )
    )
    axis.zaxis.set_major_locator(MaxNLocator(nbins=6))
    # Exact isometric projection: the visible coordinate axes have equal
    # lengths, and the common origin with the vertical z-axis is on the left.
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_proj_type("ortho")
    axis.view_init(
        elev=np.degrees(np.arctan(1.0 / np.sqrt(2.0))),
        azim=-45,
    )

    # Hide Matplotlib's native z-axis. Its automatic position is a rear edge
    # that does not pass through the common origin used by the explicit axes.
    axis.zaxis._axinfo["juggled"] = (0, 1, 2)
    axis.zaxis.line.set_visible(False)
    axis.zaxis.label.set_visible(False)
    axis.tick_params(
        axis="z",
        which="both",
        color=(0.0, 0.0, 0.0, 0.0),
        labelcolor=(0.0, 0.0, 0.0, 0.0),
    )
    axis.set_title(
        "Performance landscape: height = speedup, colour = total force error",
        pad=20,
    )

    colour_mappable = plt.cm.ScalarMappable(
        norm=error_norm,
        cmap=colour_map,
    )
    colour_mappable.set_array([])
    colour_bar = fig.colorbar(
        colour_mappable,
        ax=axis,
        pad=0.10,
        shrink=0.68,
    )
    colour_bar.set_label("Maximum total relative force error [%]")

    fig.text(
        0.5,
        0.025,
        (
            system_caption
            + " Black-edged dots are measured DOE combinations;"
            + " the surface only connects these grid points."
            + " The translucent plane is the unchanged original code"
            + " at speedup = 1."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.90,
        bottom=0.11,
        top=0.91,
    )

    # Project custom z-ticks onto the visible line (0, 0, z). This makes the
    # three labelled axes meet at exactly the same point without changing the
    # selected viewing angle.
    fig.canvas.draw()

    def project_to_axis_coordinates(x_value, y_value, z_value):
        x_projected, y_projected, _ = proj3d.proj_transform(
            x_value,
            y_value,
            z_value,
            axis.get_proj(),
        )
        display_coordinates = axis.transData.transform(
            (x_projected, y_projected)
        )
        return axis.transAxes.inverted().transform(display_coordinates)

    projected_origin = project_to_axis_coordinates(0.0, 0.0, 0.0)
    projected_top = project_to_axis_coordinates(0.0, 0.0, z_end)
    projected_direction = projected_top - projected_origin
    direction_norm = np.linalg.norm(projected_direction)
    outward_normal = np.array(
        [-projected_direction[1], projected_direction[0]]
    ) / direction_norm

    for z_tick in axis.get_zticks():
        if z_tick < 0.0 or z_tick > z_end:
            continue

        projected_tick = project_to_axis_coordinates(
            0.0,
            0.0,
            float(z_tick),
        )
        tick_start = projected_tick - 0.007 * outward_normal
        tick_end = projected_tick + 0.007 * outward_normal
        axis.add_line(
            Line2D(
                [tick_start[0], tick_end[0]],
                [tick_start[1], tick_end[1]],
                transform=axis.transAxes,
                color="#333333",
                linewidth=0.9,
                clip_on=False,
            )
        )
        label_position = projected_tick + 0.018 * outward_normal
        axis.text2D(
            label_position[0],
            label_position[1],
            f"{z_tick:g}",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=9,
            clip_on=False,
        )

    label_position = (
        0.5 * (projected_origin + projected_top)
        + 0.075 * outward_normal
    )
    label_rotation = np.degrees(
        np.arctan2(projected_direction[1], projected_direction[0])
    )
    axis.text2D(
        label_position[0],
        label_position[1],
        z_axis_label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        rotation=label_rotation,
        rotation_mode="anchor",
        fontsize=10,
        clip_on=False,
    )
    fig.savefig(
        output_directory / "performance_landscape_3d.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    """Run the DOE and save data, optimum and result plots."""

    if min(n_update_values) < 1:
        raise ValueError("n_update values must be at least 1")

    if max(cutoff_factors) * sigma_argon > box_length / 2.0:
        raise ValueError("largest r_cut must not exceed half the box length")

    output_directory.mkdir(parents=True, exist_ok=True)
    original_core = load_original_core()
    states = create_equilibrium_states()

    print("Measuring original-code baseline...", flush=True)
    original_runtimes = {}

    for seed in seeds:
        production_seed = 100000 + seed
        original_runtimes[seed] = median_original_runtime(
            original_core,
            states[seed],
            production_seed,
        )
        print(
            f"  seed {seed}: {original_runtimes[seed]:.3f} s",
            flush=True,
        )

    combinations = list(product(n_update_values, cutoff_factors, seeds))
    np.random.default_rng(12345).shuffle(combinations)
    rows = []

    for run_number, (n_update, cutoff_factor, seed) in enumerate(
        combinations,
        start=1,
    ):
        r_cut = cutoff_factor * sigma_argon
        production_seed = 100000 + seed
        optimized_runtime = median_optimized_runtime(
            states[seed],
            r_cut,
            n_update,
            production_seed,
        )
        error_metrics = measure_force_errors(
            states[seed],
            r_cut,
            n_update,
            production_seed,
        )
        speedup = original_runtimes[seed] / optimized_runtime

        rows.append(
            {
                "n_update": n_update,
                "cutoff_factor": cutoff_factor,
                "cutoff_radius_nm": r_cut,
                "seed": seed,
                "original_runtime_s": original_runtimes[seed],
                "optimized_runtime_s": optimized_runtime,
                "speedup": speedup,
                **error_metrics,
            }
        )

        results = pd.DataFrame(rows)
        save_csv(results, "doe_rohdaten.csv")
        print(
            f"{run_number}/{len(combinations)}, "
            f"n_update={n_update}, cutoff={cutoff_factor:g}, "
            f"seed={seed}, speedup={speedup:.2f}, "
            f"total error="
            f"{error_metrics['total_force_relative_l2']:.3e}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    summary, optimum = create_summary(results)
    save_csv(summary, "doe_mittelwerte.csv")
    save_csv(optimum, "optimum.csv")

    seed_text = ", ".join(str(seed) for seed in seeds)
    system_caption = (
        f"NVT; N = {n_particles}; L = {box_length:g} nm; "
        f"T = {temperature:g} K; seeds {seed_text}."
    )
    error_limits = shared_error_plot_limits(summary)

    save_heatmap(
        summary,
        "speedup_median",
        "Median speedup versus unchanged original code",
        r"Median speedup $t_\mathrm{original}/t_\mathrm{optimized}$",
        "speedup_vs_cutoff_and_n_update.png",
        caption=system_caption + " Median across seeds.",
    )
    save_heatmap(
        summary,
        "total_error_max",
        "Maximum total force error (selection limit: 1 %)",
        "Maximum relative force error [%]",
        "total_error_vs_cutoff_and_n_update.png",
        logarithmic=True,
        value_scale=100.0,
        logarithmic_limits=error_limits,
        caption=system_caption + " Maximum across seeds and sampled list ages.",
    )
    save_heatmap(
        summary,
        "list_error_max",
        "Force error caused only by neighbour-list reuse",
        "Maximum relative stale-list force error [%]",
        "list_error_vs_cutoff_and_n_update.png",
        logarithmic=True,
        value_scale=100.0,
        logarithmic_limits=error_limits,
        caption=(
            system_caption
            + " Old list compared with a fresh list at identical positions."
            + f" Maximum across seeds and {error_cycles} oldest-list"
            + " samples per run."
            + " Grey cells indicate an undefined relative error."
        ),
    )
    save_error_comparison(
        summary,
        error_limits,
        system_caption,
    )
    save_performance_landscape(
        summary,
        error_limits,
        system_caption,
    )

    if optimum.empty:
        print("No tested combination satisfies the 1 % force-error limit.")
    else:
        selected = optimum.iloc[0]
        print("\nSelected optimum:")
        print(
            f"  n_update = {int(selected['n_update'])}\n"
            f"  cutoff factor = {selected['cutoff_factor']:g}\n"
            f"  median speedup = {selected['speedup_median']:.2f}\n"
            f"  maximum total force error = "
            f"{selected['total_error_max']:.3e}"
        )


if __name__ == "__main__":
    main()
