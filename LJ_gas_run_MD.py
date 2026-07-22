#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one presentation case of the Lennard-Jones MD simulation.

Change only ``scenario_name`` and run the file once for each of the three
cases.  The script saves the normal energy/temperature histories plus the
total force and potential-energy errors against a full all-pairs reference.
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import R

from LJ_gas import (
    ParticleSystem,
    SimulationParameters,
    calculate_all_pairs_reference,
    calculate_force,
    initialize_velocities,
    instantaneous_temperature,
    kinetic_energy,
    potential_energy,
    simulate_NVT_step,
    update_neighbour_list,
)


# ---------------------------------------------------------------------------
# Select one case and run the file three times
# ---------------------------------------------------------------------------
scenario_name = "accurate"

scenarios = {
    "accurate": {
        "label": "Accurate control",
        "r_cut_factor": 3.0,
        "n_update": 1,
    },
    "optimal": {
        "label": "Selected DoE candidate",
        "r_cut_factor": 2.4,
        "n_update": 35,
    },
    "fast_inaccurate": {
        "label": "Fast, inaccurate control",
        "r_cut_factor": 1.0,
        "n_update": 49,
    },
}

if scenario_name not in scenarios:
    raise ValueError("Choose accurate, optimal or fast_inaccurate")

scenario = scenarios[scenario_name]


# ---------------------------------------------------------------------------
# Common parameters: identical to the final DOE2 experiment
# ---------------------------------------------------------------------------
n_particles = 2000
mass_argon = 39.95                 # u = g/mol
sigma_argon = 0.34                 # nm
epsilon_argon = 120.0 * R * 1e-3  # kJ/mol

dt = 0.01                          # ps
n_steps = 10000                     # 100 ps
temperature = 300.0                # K
box_length = 20.0                  # nm
tau_thermostat = 1.0               # ps
rij_min = 1e-2                     # nm

initial_seed = 1
production_seed = 100001
n_equil_steps = 500
diagnostic_interval = 20           # full reference every 0.2 ps

r_cut_factor = scenario["r_cut_factor"]
n_update = scenario["n_update"]
r_cut = r_cut_factor * sigma_argon

output_directory = Path(__file__).resolve().parent / "md_vergleich_ergebnisse"
output_directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Small helper functions
# ---------------------------------------------------------------------------
def set_argon_parameters(ps):
    ps.mass[:] = mass_argon
    ps.sigma[:] = sigma_argon
    ps.epsilon[:] = epsilon_argon


def initialize_safe_positions(ps):
    """Random positions with a minimum initial distance of one sigma."""

    positions = np.empty((ps.n, 3))
    minimum_distance_squared = sigma_argon**2

    for particle_index in range(ps.n):
        for _ in range(100000):
            candidate = np.random.uniform(0.0, box_length, size=3)

            if particle_index == 0:
                positions[particle_index] = candidate
                break

            displacement = positions[:particle_index] - candidate
            displacement -= box_length * np.rint(displacement / box_length)
            distance_squared = np.einsum(
                "ij,ij->i",
                displacement,
                displacement,
            )

            if np.all(distance_squared >= minimum_distance_squared):
                positions[particle_index] = candidate
                break
        else:
            raise RuntimeError("Could not create safe initial positions")

    ps.position[:] = positions


def make_simulation(cutoff_factor, steps):
    return SimulationParameters(
        dt=dt,
        n_steps=steps,
        temperature=temperature,
        box_length=box_length,
        tau_thermostat=tau_thermostat,
        rij_min=rij_min,
        r_cut=cutoff_factor * sigma_argon,
    )


def full_reference_error(ps, sim):
    """Errors of the force/energy actually used versus all particle pairs."""

    full_force, full_energy, _ = calculate_all_pairs_reference(
        ps,
        sim,
        apply_cutoff=False,
    )
    reference_norm = np.linalg.norm(full_force)
    force_error_percent = (
        100.0 * np.linalg.norm(ps.force - full_force) / reference_norm
        if reference_norm > 0.0
        else np.nan
    )
    energy_error = abs(potential_energy(ps, sim) - full_energy)
    return float(force_error_percent), float(energy_error)


def german_number(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not np.isfinite(value):
        return ""
    return format(float(value), ".15g").replace(".", ",")


def write_german_csv(filename, header, rows):
    with open(filename, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow([german_number(value) for value in row])


def check_finite(ps, context):
    if not all(
        np.all(np.isfinite(values))
        for values in (ps.position, ps.velocity, ps.force)
    ):
        raise FloatingPointError(f"Non-finite state during {context}")


# ---------------------------------------------------------------------------
# Create the same equilibrated start state for every separate run
# ---------------------------------------------------------------------------
print("Creating common equilibrated state...", flush=True)
np.random.seed(initial_seed)

sim_equilibration = make_simulation(3.0, n_equil_steps)
ps_equilibration = ParticleSystem(n_particles)
set_argon_parameters(ps_equilibration)
initialize_safe_positions(ps_equilibration)
initialize_velocities(ps_equilibration, temperature)
update_neighbour_list(
    ps_equilibration,
    sim_equilibration,
    step=0,
    n_update=1,
)
calculate_force(ps_equilibration, sim_equilibration)

for step in range(1, n_equil_steps + 1):
    simulate_NVT_step(ps_equilibration, sim_equilibration, step, 1)

    if step % 100 == 0:
        check_finite(ps_equilibration, f"equilibration step {step}")
        current_temperature = instantaneous_temperature(ps_equilibration)
        if current_temperature > 10.0 * temperature:
            raise FloatingPointError(
                "Unstable temperature during equilibration: "
                f"{current_temperature:.3e} K"
            )

check_finite(ps_equilibration, "equilibration")


# ---------------------------------------------------------------------------
# Initialize the selected production case from that state
# ---------------------------------------------------------------------------
sim = make_simulation(r_cut_factor, n_steps)
ps = ParticleSystem(n_particles)
set_argon_parameters(ps)
ps.position[:] = ps_equilibration.position
ps.velocity[:] = ps_equilibration.velocity
update_neighbour_list(ps, sim, step=0, n_update=n_update)
calculate_force(ps, sim)
np.random.seed(production_seed)

print(
    f"Running {scenario['label']}: "
    f"r_cut={r_cut_factor:g} sigma, n_update={n_update}",
    flush=True,
)


# ---------------------------------------------------------------------------
# Production trajectory and periodically sampled reference errors
# ---------------------------------------------------------------------------
trajectory_rows = []
error_rows = []
pure_md_runtime = 0.0


def store_trajectory(step):
    e_pot = potential_energy(ps, sim)
    e_kin = kinetic_energy(ps)
    current_temperature = 2.0 * e_kin * 1e3 / (3.0 * ps.n * R)
    trajectory_rows.append(
        (
            step,
            step * dt,
            e_pot,
            e_kin,
            e_pot + e_kin,
            current_temperature,
        )
    )


def store_error(step):
    force_error, energy_error = full_reference_error(ps, sim)
    error_rows.append(
        (step, step * dt, force_error, energy_error)
    )


store_trajectory(0)
store_error(0)

for step in range(1, n_steps + 1):
    # Only this call contributes to the reported MD runtime.
    start = time.perf_counter()
    simulate_NVT_step(ps, sim, step, n_update)
    pure_md_runtime += time.perf_counter() - start

    store_trajectory(step)

    # Also sample immediately before a rebuild, when the list is oldest.
    oldest_list_step = (
        n_update > 1
        and step % n_update == n_update - 1
    )

    # The expensive all-pairs reference is outside the runtime measurement.
    if (
        step % diagnostic_interval == 0
        or oldest_list_step
        or step == n_steps
    ):
        store_error(step)

    if step % max(1, n_steps // 10) == 0:
        print(f"  step {step}/{n_steps}", flush=True)

check_finite(ps, "production")


# ---------------------------------------------------------------------------
# Save German Excel-compatible data
# ---------------------------------------------------------------------------
trajectory_file = output_directory / f"{scenario_name}_trajectory.csv"
error_file = output_directory / f"{scenario_name}_errors.csv"

write_german_csv(
    trajectory_file,
    [
        "step",
        "time_ps",
        "potential_energy_kJ_mol",
        "kinetic_energy_kJ_mol",
        "total_energy_kJ_mol",
        "temperature_K",
    ],
    trajectory_rows,
)
write_german_csv(
    error_file,
    [
        "step",
        "time_ps",
        "total_force_error_percent",
        "total_potential_energy_error_kJ_mol",
    ],
    error_rows,
)


# ---------------------------------------------------------------------------
# Separate presentation figures for this case
# ---------------------------------------------------------------------------
trajectory = np.asarray(trajectory_rows, dtype=float)
errors = np.asarray(error_rows, dtype=float)

case_title = (
    f"{scenario['label']}: "
    rf"$r_\mathrm{{cut}}={r_cut_factor:g}\sigma$, "
    rf"$n_\mathrm{{update}}={n_update}$"
)
time_label = (
    "Simulation time [ps]\n"
    f"Simulated duration: {n_steps * dt:g} ps | "
    f"Pure MD computation time: {pure_md_runtime:.3f} s "
    "(reference diagnostics excluded)"
)


def save_single_plot(filename_suffix, title, ylabel, x_values, y_values):
    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.plot(x_values, y_values)
    axis.set_title(f"{title}\n{case_title}")
    axis.set_xlabel(time_label)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    figure.savefig(
        output_directory / f"{scenario_name}_{filename_suffix}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


save_single_plot(
    "potential_energy",
    "Potential energy",
    "Potential energy [kJ/mol]",
    trajectory[:, 1],
    trajectory[:, 2],
)
save_single_plot(
    "kinetic_energy",
    "Kinetic energy",
    "Kinetic energy [kJ/mol]",
    trajectory[:, 1],
    trajectory[:, 3],
)

figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
axis.plot(trajectory[:, 1], trajectory[:, 5])
axis.axhline(300.0, color="0.35", linestyle="--", label="Target: 300 K")
axis.set_title(f"Temperature\n{case_title}")
axis.set_xlabel(time_label)
axis.set_ylabel("Temperature [K]")
axis.grid(True, alpha=0.25)
axis.legend()
figure.savefig(
    output_directory / f"{scenario_name}_temperature.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)

figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
axis.scatter(errors[:, 1], errors[:, 2], s=14)
axis.axhline(1.0, color="0.35", linestyle="--", label="DOE limit: 1 %")
axis.set_title(f"Total relative force error\n{case_title}")
axis.set_xlabel(time_label)
axis.set_ylabel("Total relative force error [%]")
axis.grid(True, alpha=0.25)
axis.legend()
figure.savefig(
    output_directory / f"{scenario_name}_force_error.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)

figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
axis.scatter(errors[:, 1], errors[:, 3], s=14)
axis.set_title(f"Absolute potential-energy error\n{case_title}")
axis.set_xlabel(time_label)
axis.set_ylabel("Absolute potential-energy error [kJ/mol]")
axis.grid(True, alpha=0.25)
figure.savefig(
    output_directory / f"{scenario_name}_potential_energy_error.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(figure)


# ---------------------------------------------------------------------------
# Compact numerical summary
# ---------------------------------------------------------------------------
maximum_force_error = float(np.nanmax(errors[:, 2]))
maximum_energy_error = float(np.nanmax(errors[:, 3]))
summary_file = output_directory / f"{scenario_name}_summary.txt"

summary_lines = [
    f"Scenario: {scenario['label']}",
    f"r_cut = {r_cut_factor:g} sigma = {r_cut:.6f} nm",
    f"n_update = {n_update}",
    f"N = {n_particles}",
    f"L = {box_length:g} nm",
    f"dt = {dt:g} ps",
    f"production time = {n_steps * dt:g} ps",
    f"pure MD runtime = {pure_md_runtime:.9f} s",
    f"maximum total force error = {maximum_force_error:.9g} %",
    f"maximum potential-energy error = {maximum_energy_error:.9g} kJ/mol",
    "Reference calculations and file output are excluded from the runtime.",
]
summary_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print("\n" + "\n".join(summary_lines), flush=True)
print(f"Results written to: {output_directory}", flush=True)
