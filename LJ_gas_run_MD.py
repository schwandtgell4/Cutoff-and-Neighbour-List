#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas_run_MD.py

Main program for running molecular dynamics simulations using Lennard-Jones particles.
Initializes the system, runs the integrator loop, records energy and trajectory data, 
and visualizes results.

Author: Bettina Keller
Created: May 28, 2025

This script imports all classes and functions from md_simulation.py and controls
the simulation workflow.

"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R
import matplotlib.pyplot as plt

import time
from datetime import datetime

from LJ_gas import(
    ParticleSystem,
    SimulationParameters,
    simulate_NVE_step,
    simulate_NVT_step,
    initialize_positions,
    initialize_velocities,
    calculate_force,
    density,
    write_xyz_trajectory,
    potential_energy,
    kinetic_energy,
    instantaneous_temperature,
    ideal_gas_pressure
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
    
    if '_tic_time' in globals():
        elapsed_time = time.time() - _tic_time
    
    else:
        print("Error: tic() was not called before toc()")
    
    return elapsed_time


#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 200
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

# output
file_name_base = "my_simulation"  # file name for all output files

#----------------------------------------------------------------
#   P R O G R A M
#----------------------------------------------------------------
# start the timer
tic()

#
# initialize simulation parameters
#
sim = SimulationParameters(dt = dt, 
                           n_steps = n_steps, 
                           temperature = temperature, 
                           box_length = box_length, 
                           tau_thermostat = tau_thermostat,
                           rij_min=rij_min
                           )

#
# initialize ParticleSystem 
#
ps = ParticleSystem(n_particles)

# fill in the parameters for argon
for i in range(n_particles): 
    ps.set_parameters(i, mass=mass_argon, sigma=sigma_argon, epsilon=epsilon_argon)

# set initial positions     
initialize_positions(ps, sim.box_length)

# set initial velocities     
initialize_velocities(ps, sim.temperature)

# calculate force according to initial positions
calculate_force(ps, sim)

# calculate box density
rho = density(ps, sim)

# calculate initial values of variable properties
E_pot_init = potential_energy(ps, sim)
E_kin_init = kinetic_energy(ps)
T_init = instantaneous_temperature(ps)
P_init = ideal_gas_pressure(ps, sim)


# initialize position trajectory
position_trajectory = np.zeros((sim.n_steps+1, n_particles, 3))
position_trajectory[0,:,:] = ps.position # initial position

# initialize energy trajectory
energy_trajectory = np.zeros((sim.n_steps+1, 4))
energy_trajectory[0,0] = potential_energy( ps, sim)       # potential energy
energy_trajectory[0,1] = kinetic_energy(ps)               # kinetic energy
energy_trajectory[0,2] = instantaneous_temperature(ps)    # instantaneous pressure
energy_trajectory[0,3] = ideal_gas_pressure(ps, sim)      # ideal gas pressure


#--------------------------------------------------
#  The acutal MD simulation
#--------------------------------------------------
for i in range(sim.n_steps):
    if NVT==True:
        simulate_NVT_step(ps, sim)
    else: 
        simulate_NVE_step(ps, sim)
        
    # store updated positions
    position_trajectory[i+1,:,:] = ps.position # store updated positions

    # store updated energies, temperature and pressure
    energy_trajectory[i+1,0] = potential_energy( ps, sim)     # potential energy
    energy_trajectory[i+1,1] = kinetic_energy(ps)             # kinetic energy
    energy_trajectory[i+1,2] = instantaneous_temperature(ps)  # instantaneous pressure
    energy_trajectory[i+1,3] = ideal_gas_pressure(ps, sim)    # ideal gas pressure


#--------------------------------------
# W R I T E    T R A J E C T O R I E S 
#--------------------------------------
# write position trajectory to file
write_xyz_trajectory(file_name_base + "_pos.xyz", position_trajectory, atom_symbol="Ar")
# write energy trajectory to file (binary and text)
np.save(file_name_base + "_ene.npy", energy_trajectory)
np.savetxt(file_name_base + "_ene.dat", energy_trajectory, fmt="%.6e", header="#E_pot  E_kin  T  P", comments='')


#----------------------------------------------------
# P L O T   E N E R G Y   T R A J E C T O R I E S
#----------------------------------------------------
# set time axis
time_ps = np.arange(sim.n_steps + 1) * sim.dt

#
# potential energy
# 
E_pot_min = np.mean(energy_trajectory[:,0]) - 1   # lower limit of E_pot axis
E_pot_max = np.mean(energy_trajectory[:,0]) + 1   # upper limit of E_pot axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,0]) 
plt.ylim(E_pot_min, E_pot_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("E_pot [kJ/mol]", fontsize=14)

plt.savefig(file_name_base + "_Epot.png", dpi=300, bbox_inches='tight')
plt.show()

#
# kinetic energy
# 
E_kin_min = np.mean(energy_trajectory[:,1]) - 100   # lower limit of E_kin axis
E_kin_max = np.mean(energy_trajectory[:,1]) + 100   # upper limit of E_kin axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,1]) 
plt.ylim(E_kin_min, E_kin_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("E_kin [kJ/mol]", fontsize=14)

plt.savefig(file_name_base + "_Ekin.png", dpi=300, bbox_inches='tight')
plt.show()

#
# temperature
# 
T_min = np.mean(energy_trajectory[:,2]) - 100   # lower limit of T axis
T_max = np.mean(energy_trajectory[:,2]) + 100   # upper limit of T axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,2]) 
plt.ylim(T_min, T_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("T [K]", fontsize=14)

plt.savefig(file_name_base + "_T.png", dpi=300, bbox_inches='tight')
plt.show()

#
# pressure
# 
P_min = np.mean(energy_trajectory[:,3]) - 200   # lower limit of P axis
P_max = np.mean(energy_trajectory[:,3]) + 200   # upper limit of P axis 

plt.figure(figsize=(8, 6))
plt.plot(time_ps, energy_trajectory[:,3]) 
plt.ylim(P_min, P_max)
plt.xlabel("time [ps]", fontsize=14)
plt.ylabel("P [Pa]", fontsize=14)

plt.savefig(file_name_base + "_P.png", dpi=300, bbox_inches='tight')
plt.show()


#--------------------------------------
# O U T P U T 
#--------------------------------------
elapsed_time = toc()   # stop the timer
output_lines = []

output_lines.append("")
output_lines.append("----------------------------------------------------------")
output_lines.append("Simulation parameters ")    
output_lines.append("----------------------------------------------------------")
output_lines.append(f"{'Number of particles:':<30}{ps.n:>10.0f} ")
output_lines.append(f"{'Box length:':<30}{sim.box_length:>10.3e} nm")
output_lines.append(f"{'Box volume:':<30}{sim.box_length**3:>10.3e} nm^3")
output_lines.append(f"{'Density:':<30}{rho:>10.3e} g/cm^3")
output_lines.append("")   
output_lines.append(f"{'Time step:':<30}{sim.dt:>10.3f} ps")
output_lines.append(f"{'Number of time steps:':<30}{sim.n_steps:>10.0f}")
output_lines.append(f"{'Simulation time:':<30}{sim.n_steps * sim.dt :>10.3e} ps")
output_lines.append("")   
if NVT==True: 
    output_lines.append(f"{'Ensemble:':<30}{'NVT':>10}")
    output_lines.append(f"{'Thermostat temperature:':<30}{sim.temperature:>10.0f} K")
    output_lines.append(f"{'Thermostat coupling:':<30}{sim.tau_thermostat:>10.3e} ps")
else: 
    output_lines.append(f"{'Ensemble:':<30}{'NVE':>10}")
    output_lines.append(f"{'Initial velocities:':<30}{sim.temperature:>10.0f} K")

output_lines.append("")     
output_lines.append(f"{'Lower cutoff radius:':<30}{sim.rij_min:>10.3f} nm")
output_lines.append("----------------------------------------------------------")
if elapsed_time: 
    time_per_time_step = elapsed_time/sim.n_steps
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_lines.append(f"{'Elapsed time:':<30}{elapsed_time:>10.3f} s")   
    output_lines.append(f"{'Elapsed time per time step:':<30}{time_per_time_step:>10.3f} s")
    output_lines.append(f"{'Time stamp:':<30}{now} s")
output_lines.append("----------------------------------------------------------")
output_lines.append("END")  
output_lines.append("----------------------------------------------------------")

# Print to screen
for line in output_lines:
    print(line)
  
# Write to file
with open(file_name_base + ".out", "w") as f:
    for line in output_lines:
        f.write(line + "\n")    