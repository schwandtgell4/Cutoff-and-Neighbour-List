# LennardJonesMD: A Minimal Molecular Dynamics Simulation in Python

This project implements a velocity-Verlet and Langevin dynamics integrator for simulating Lennard-Jones particles in 3D. It supports both NVE and NVT ensembles and includes periodic boundary conditions, trajectory output, and energy diagnostics.

## Features

- Velocity Verlet (NVE) and BAOAB Langevin (NVT) integration
- Lennard-Jones pairwise interaction
- Uses periodic boundary conditions (PBC)in all directions (always on)
- Temperature control via Langevin thermostat
- Trajectory output in `.xyz` format
- Energy diagnostics and plotting
- Unit handling in SI-compatible conventions

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy


## Units

| Quantity    | Unit          |
| ----------- | ------------- |
| Distance    | nm            |
| Time        | ps            |
| Temperature | K             |
| Energy      | kJ/mol        |
| Pressure    | kJ mol⁻¹ nm⁻³ |

## How to Use

### 1. **Configure the Simulation**

Edit the parameter section of the script to configure the simulation. 

```python
#----------------------------------------------------------------
#   P A R A M E T E R S
#----------------------------------------------------------------
# system
n_particles = 200               # number of particles
mass_argon =  39.95             # mass in u = 1e-3 kg/mol
sigma_argon = 0.34              # sigma in nm     Argon: 0.34
epsilon_argon = 120*R*1e-3      # epsilon in kJ/mol Argon: 120

# simulation
dt = 0.1             # ps
n_steps = 1000        # number of timesteps 
temperature = 300     # K
box_length = 100      # nm
tau_thermostat = 1  # thermostat coupling constant in 1/ps
rij_min = 1e-2      # nm
NVT = True          # switch to decide between NVT and NVE

# output
file_name_base = "my_simulation"  # file name for all output files
```
(rij_min is a parameter to buffer extremely high forces, when two particles collide.)

The simulation parameters are then automatically set in the SimulationParameters class:

```python
#
# initialize simulation parameters
#
sim = SimulationParameters(dt = dt, 
                           n_steps = n_steps,         # number of timesteps
                           temperature = temperature, # K
                           box_length = box_length,   # nm  
                           tau_thermostat = tau_thermostat, # ps
                           rij_min=rij_min            # nm
                           )
```

The system parameters are initialized using the ParticleSystem class

```python
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
```
---

### 2. **Run the Simulation**

From a terminal, execute:

```bash
python LJ_gas_run_MD.py
```

This will:

* Initialize particle positions (on a cubic grid) and Maxwell-Boltzmann velocities
* Run the MD loop (NVE or NVT with Langevin thermostat)
* Write particle positions to a `.xyz` trajectory file
* Plot energy and temperature over time

---

### 3. **Output Files**

* `trajectory.xyz` – Atom positions (open with VMD or Ovito)
* `energy_plot.png` – Time evolution of potential, kinetic, and total energy
* `temperature_plot.png` – Instantaneous temperature vs. time

---
### 4. **Visualizing the Trajectory in VMD**

To inspect particle motion, you can open the trajectory in **VMD (Visual Molecular Dynamics)**:

1. Launch VMD.
2. Go to `File` → `New Molecule`.
3. Under *Filename*, select `trajectory.xyz`.
4. Click **Load**.

💡 **Tips**:

* VMD assumes XYZ files contain atomic symbols. If the file only contains numerical data, you may need to:

  * Edit the `.xyz` file to assign a dummy element label (e.g., "Ar" or "X") to each atom line.
  * Alternatively, modify `write_xyz_trajectory()` in the code to output proper element tags.

* In the **Graphics → Representations** menu, adjust drawing methods (e.g., "VDW" or "Points") and box size display to better view the particle system in a periodic box.

* Use the **Animation** controls to play, pause, or step through the simulation frames.

---

## Code Components

### Main Simulation Workflow

* `initialize_positions` – Places particles on a grid
* `initialize_velocities` – Draws initial velocities from the Maxwell-Boltzmann distribution
* `simulate_NVE_step` – Velocity Verlet integration (energy-conserving)
* `simulate_NVT_step` – Langevin dynamics integration (temperature-controlled)
* `write_xyz_trajectory` – Saves snapshots of particle coordinates
* `potential_energy`, `kinetic_energy`, `instantaneous_temperature` – Diagnostics

---

## Notes for Students

* The MD code is designed for atomic gases. This allows you to use much larger **time steps** than in usual MD simulation (e.g. dt = 0.1 ps = 100 fs in the example above). The maximum timestep depends on the ensemble (NVE vs NVT) and system. Monitor the energy and the temperature to make sure that your simulation is stable.
* You can modify the number of particles, temperature, time step, or box size for experimentation.
* The **NVE ensemble** should conserve total energy (check energy plot).
* The **NVT ensemble** uses Langevin dynamics and should regulate the temperature.
* Periodic boundary conditions mean particles that leave one side of the box re-enter on the opposite side and switched in per default
* Read the file LJ_gas.py to understand how the code works.

