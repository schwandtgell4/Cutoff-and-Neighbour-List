#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas_run_MD_torch.py

Run script for the PyTorch/GPU Lennard-Jones gas implementation.
"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R
import matplotlib.pyplot as plt

import time
from datetime import datetime

from LJ_gas_torch import (
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    simulate_NVT_step,
    initialize_positions,
    initialize_velocities,
    calculate_force_torch,
    density,
    write_xyz_trajectory,
    potential_energy,
    kinetic_energy,
    instantaneous_temperature,
    ideal_gas_pressure,
    cutoff_pair_statistics,
    move_particle_system_to_torch,
    update_neighbour_list,
    synchronize_device,
    tensor_to_numpy,
)


#----------------------------------------------------------------
#   F U N C T I O N S
#----------------------------------------------------------------
# Define tic and toc functions
def tic():
    """Start a timer."""
    global _tic_time
    _tic_time = time.time()


def toc():
    """Stop the timer and return the elapsed time in seconds."""
    
    elapsed_time = None

    if "_tic_time" in globals():
        elapsed_time = time.time() - _tic_time
    else:
        print("Error: tic() was not called before toc()")

    return elapsed_time

def kinetic_energy_tensor(ps):
    """
    Calculate kinetic energy as a scalar tensor
    without CPU transfer.
    """
    v_squared = (
        ps.velocity
        * ps.velocity
    ).sum(dim=1)

    return 0.5 * (
        ps.mass
        * v_squared
    ).sum()


def save_plot(x, y, filename, xlabel, ylabel, y_margin):
    """Save one trajectory plot without blocking the program with plt.show()."""
    y_mean = np.mean(y)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.ylim(y_mean - y_margin, y_mean + y_margin)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 2000
mass_argon =  39.95             # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm     Argon: 0.34
epsilon_argon = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# simulation
dt = 0.1             # ps
n_steps = 1000 
temperature = 300     # K
box_length = 100      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
NVT = True          # switch to decide between NVT and NVE
n_update = 10       # define how often list is updated
use_cutoff = True
r_cut_factor = 20.5
r_cut = r_cut_factor * sigma_argon   # cutoff radius in nm; reference: (3.8) at page 15 of lecture script
                            # sigma is LJ length scale and with 2.5 as factor

# output
file_name_base = "my_simulation_torch"

# Save every MD step with 1.
# Use 10 to save only every tenth step.
sample_every = 1

# Set False to skip the large XYZ position trajectory.
save_position_trajectory = True


#----------------------------------------------------------------
#   P R O G R A M
#----------------------------------------------------------------
total_start_time = time.perf_counter()

sim = SimulationParameters(dt = dt, 
                           n_steps = n_steps, 
                           temperature = temperature, 
                           box_length = box_length, 
                           tau_thermostat = tau_thermostat,
                           rij_min=rij_min,
                           r_cut=r_cut,
                           use_cutoff=use_cutoff
                           )

# initialize ParticleSystem
ps = ParticleSystem(n_particles)

# Vectorized initialization.
ps.mass[:] = mass_argon
ps.sigma[:] = sigma_argon
ps.epsilon[:] = epsilon_argon
# set initial positions
initialize_positions(ps, sim.box_length)
initialize_velocities(ps, sim.temperature)

# move all particle data to CUDA, MPS, or CPU
# Keep dtype=torch.float32 in LJ_gas_torch for better GPU compatibility and speed.
device = move_particle_system_to_torch(ps)
print("Using device:", device)
# first create of neighbour list
update_neighbour_list(ps, sim, step=0, n_update=n_update)
# calculate force according to initial positions on the selected device
calculate_force_torch(ps, sim)

# calculate box density
rho = density(ps, sim)

# calculate cutoff pair statistics for the initial configuration
n_total_init, n_cut_init, percent_cut_init = cutoff_pair_statistics(ps, sim)
print(f"r_cut = {sim.r_cut:.3f} nm")
print(f"Initial pairs inside cutoff: {n_cut_init}/{n_total_init} ({percent_cut_init:.4f}%)")

# Initial values are calculated once outside
# the timed MD loop.
E_pot_init = potential_energy(
    ps,
    sim
)

E_kin_init = kinetic_energy(
    ps
)

T_init = (
    2.0
    * E_kin_init
    * 1e3
    / (
        3.0
        * ps.n
        * R
    )
)

P_init = (
    (
        ps.n
        / 6.02214076e23
    )
    * R
    * T_init
    / (
        sim.box_length**3
        * 1e-27
    )
)

if sample_every < 1:
    raise ValueError(
        "sample_every must be at least 1"
    )

saved_steps = list(
    range(
        0,
        sim.n_steps + 1,
        sample_every
    )
)

if saved_steps[-1] != sim.n_steps:
    saved_steps.append(
        sim.n_steps
    )

step_to_frame = {
    step: frame
    for frame, step
    in enumerate(saved_steps)
}

n_frames = len(
    saved_steps
)

# Keep the trajectory on the selected device
# during the MD loop.
if save_position_trajectory:
    position_trajectory_device = (
        ps.position.new_empty(
            (
                n_frames,
                ps.n,
                3
            )
        )
    )

    position_trajectory_device[0].copy_(
        ps.position
    )
else:
    position_trajectory_device = None

energy_trajectory_device = (
    ps.position.new_empty(
        (
            n_frames,
            4
        )
    )
)

energy_trajectory_device[
    0,
    0
] = ps.current_potential_energy

energy_trajectory_device[
    0,
    1
] = E_kin_init

energy_trajectory_device[
    0,
    2
] = T_init

energy_trajectory_device[
    0,
    3
] = P_init


#--------------------------------------------------
#  The actual MD simulation
#--------------------------------------------------
print(
    "Starting MD simulation...",
    flush=True
)

# CUDA and MPS operations are asynchronous.
synchronize_device(device)

md_start_time = (
    time.perf_counter()
)

for step in range(
    1,
    sim.n_steps + 1
):
    if NVT:
        simulate_NVT_step(
            ps,
            sim,
            step,
            n_update
        )
    else:
        simulate_NVE_step(
            ps,
            sim,
            step,
            n_update
        )

    frame = step_to_frame.get(
        step
    )

    if frame is not None:
        if save_position_trajectory:
            position_trajectory_device[
                frame
            ].copy_(
                ps.position
            )

        # Calculate all values on the device.
        # No .item() or NumPy conversion here.
        e_kin_device = (
            kinetic_energy_tensor(
                ps
            )
        )

        temperature_device = (
            2.0
            * e_kin_device
            * 1e3
            / (
                3.0
                * ps.n
                * R
            )
        )

        pressure_device = (
            (
                ps.n
                / 6.02214076e23
            )
            * R
            * temperature_device
            / (
                sim.box_length**3
                * 1e-27
            )
        )

        energy_trajectory_device[
            frame,
            0
        ] = ps.current_potential_energy

        energy_trajectory_device[
            frame,
            1
        ] = e_kin_device

        energy_trajectory_device[
            frame,
            2
        ] = temperature_device

        energy_trajectory_device[
            frame,
            3
        ] = pressure_device

    if (
        step
        % max(
            1,
            sim.n_steps // 10
        )
        == 0
    ):
        percent = (
            100
            * step
            / sim.n_steps
        )

        print(
            f"Step {step}/{sim.n_steps} "
            f"finished ({percent:.0f}%)",
            flush=True
        )

# Wait for all queued GPU kernels.
synchronize_device(device)

md_elapsed_time = (
    time.perf_counter()
    - md_start_time
)

print(
    "MD simulation finished. "
    "Transferring and writing output...",
    flush=True
)

# Transfer once after the simulation instead
# of once per MD step.
if save_position_trajectory:
    position_trajectory = (
        tensor_to_numpy(
            position_trajectory_device
        )
    )
else:
    position_trajectory = None

energy_trajectory = tensor_to_numpy(
    energy_trajectory_device
)


#--------------------------------------
# W R I T E    T R A J E C T O R I E S
#--------------------------------------
if save_position_trajectory:
    write_xyz_trajectory(
        file_name_base + "_pos.xyz",
        position_trajectory,
        atom_symbol="Ar"
    )

np.save(
    file_name_base + "_ene.npy",
    energy_trajectory
)

np.savetxt(
    file_name_base + "_ene.dat",
    energy_trajectory,
    fmt="%.6e",
    header="#E_pot  E_kin  T  P",
    comments=""
)


#----------------------------------------------------
# P L O T   E N E R G Y   T R A J E C T O R I E S
#----------------------------------------------------
time_ps = (
    np.asarray(saved_steps)
    * sim.dt
)

save_plot(time_ps, energy_trajectory[:, 0], file_name_base + "_Epot.png", "time [ps]", "E_pot [kJ/mol]", 1)
save_plot(time_ps, energy_trajectory[:, 1], file_name_base + "_Ekin.png", "time [ps]", "E_kin [kJ/mol]", 100)
save_plot(time_ps, energy_trajectory[:, 2], file_name_base + "_T.png", "time [ps]", "T [K]", 100)
save_plot(time_ps, energy_trajectory[:, 3], file_name_base + "_P.png", "time [ps]", "P [Pa]", 200)


#--------------------------------------
# O U T P U T
#--------------------------------------


# Calculate final pair statistics.
n_total_final, n_cut_final, percent_cut_final = (
    cutoff_pair_statistics(
        ps,
        sim
    )
)

total_elapsed_time = (
    time.perf_counter()
    - total_start_time
)

output_lines = []

output_lines.append("")
output_lines.append("----------------------------------------------------------")
output_lines.append("Simulation parameters")
output_lines.append("----------------------------------------------------------")
output_lines.append(f"{'Number of particles:':<30}{ps.n:>10.0f} ")
output_lines.append(f"{'Device:':<30}{str(device):>10}")
output_lines.append(f"{'Box length:':<30}{sim.box_length:>10.3e} nm")
output_lines.append(f"{'Box volume:':<30}{sim.box_length**3:>10.3e} nm^3")
output_lines.append(f"{'Density:':<30}{rho:>10.3e} g/cm^3")
output_lines.append("")
output_lines.append(f"{'Time step:':<30}{sim.dt:>10.3f} ps")
output_lines.append(f"{'Number of time steps:':<30}{sim.n_steps:>10.0f}")
output_lines.append(
    f"{'Simulation time:':<30}"
    f"{sim.n_steps * sim.dt:>10.3e} ps"
)

output_lines.append(
    f"{'Neighbour update interval:':<30}"
    f"{n_update:>10}"
)

output_lines.append(
    f"{'Neighbour-list rebuilds:':<30}"
    f"{ps.neighbour_rebuilds:>10}"
)

output_lines.append(
    f"{'Output sample interval:':<30}"
    f"{sample_every:>10}"
)

output_lines.append("")
if NVT:
    output_lines.append(f"{'Ensemble:':<30}{'NVT':>10}")
    output_lines.append(f"{'Thermostat temperature:':<30}{sim.temperature:>10.0f} K")
    output_lines.append(f"{'Thermostat coupling:':<30}{sim.tau_thermostat:>10.3e} ps")
else:
    output_lines.append(f"{'Ensemble:':<30}{'NVE':>10}")
    output_lines.append(f"{'Initial velocities:':<30}{sim.temperature:>10.0f} K")

output_lines.append("")
output_lines.append(f"{'Lower cutoff radius:':<30}{sim.rij_min:>10.3f} nm")
output_lines.append(f"{'Use upper cutoff:':<30}{str(sim.use_cutoff):>10}")
output_lines.append(f"{'Upper cutoff factor:':<30}{r_cut_factor:>10.3f} sigma")
output_lines.append(f"{'Upper cutoff radius:':<30}{sim.r_cut:>10.3f} nm")
output_lines.append(f"{'Initial LJ pairs:':<30}{n_total_init:>10}")
output_lines.append(f"{'Initial pairs in cutoff:':<30}{n_cut_init:>10}")
output_lines.append(f"{'Initial pairs in cutoff [%]:':<30}{percent_cut_init:>10.4f}")
output_lines.append(f"{'Final LJ pairs:':<30}{n_total_final:>10}")
output_lines.append(f"{'Final pairs in cutoff:':<30}{n_cut_final:>10}")
output_lines.append(f"{'Final pairs in cutoff [%]:':<30}{percent_cut_final:>10.4f}")

output_lines.append(
    "----------------------------------------------------------"
)

total_time_per_step = (
    total_elapsed_time
    / sim.n_steps
)

md_time_per_step = (
    md_elapsed_time
    / sim.n_steps
)

now = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

output_lines.append(
    f"{'Total elapsed time:':<30}"
    f"{total_elapsed_time:>10.3f} s"
)

output_lines.append(
    f"{'Total time per step:':<30}"
    f"{total_time_per_step:>10.6f} s"
)

output_lines.append(
    f"{'MD elapsed time:':<30}"
    f"{md_elapsed_time:>10.3f} s"
)

output_lines.append(
    f"{'MD time per step:':<30}"
    f"{md_time_per_step:>10.6f} s"
)

output_lines.append(
    f"{'Time stamp:':<30}"
    f"{now}"
)

output_lines.append(
    "----------------------------------------------------------"
)

output_lines.append("END")

output_lines.append(
    "----------------------------------------------------------"
)

for line in output_lines:
    print(line)

with open(file_name_base + ".out", "w") as f:
    for line in output_lines:
        f.write(line + "\n")
