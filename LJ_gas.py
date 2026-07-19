#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LJ_gas.py

Core module for molecular dynamics simulations of Lennard-Jones gases in the 
NVE and NVT ensembles. Defines data structures (ParticleSystem, SimulationParameters), 
integration schemes (Velocity Verlet, Langevin BAOAB), and energy/force calculations 
based on Lennard-Jones interactions.

Author: Bettina Keller
Created: May 28, 2025

"""

#----------------------------------------------------------------
#   I M P O R T S
#----------------------------------------------------------------
import numpy as np
from scipy.constants import R, Avogadro

#----------------------------------------------------------------
#   C L A S S E S
#----------------------------------------------------------------
class ParticleSystem:
    def __init__(self, n_particles):
        self.n = n_particles
        
        # Properties for each particle
        self.mass = np.zeros(n_particles)
        self.sigma = np.zeros(n_particles)
        self.epsilon = np.zeros(n_particles)
        
        # 3D positions, velocities, forces, and random numbers (shape: n_particles x 3)
        self.position = np.zeros((n_particles, 3))
        self.velocity = np.zeros((n_particles, 3))
        self.force = np.zeros((n_particles, 3))
        self.random_number = np.zeros((n_particles, 3))
    
    #---------------------
    # With these functions the parameters and states of individual atoms can be changed.
    # In vectorized programming, they will not be used very often
    #
    def set_parameters(self, i, mass, sigma, epsilon):
        """Set the paramters of the i-th particle
            mass in units of u 
            sigma in units of nm 
            epsilon in units of kJ/mol 
        """
        self.mass[i] = mass
        self.sigma[i] = sigma
        self.epsilon[i] = epsilon

    def set_position(self, i, position):
        """Set the paramters of the i-th particle"""
        self.position[i] = position
        
    def set_velocity(self, i, velocity):
        """Set the paramters of the i-th particle"""    
        self.velocity[i] = velocity            

    def set_force(self, i, force):
        """Set the paramters of the i-th particle"""    
        self.force[i] = force            

    def set_random_number(self, i, random_number):
        """Set the paramters of the i-th particle"""    
        self.random_number[i] = random_number            

    def __repr__(self):
        return f"<ParticleSystem with {self.n} particles>"


class SimulationParameters:
    def __init__(self, dt, n_steps, temperature, box_length, tau_thermostat=None, rij_min=0.0, r_cut=None):
        """ 
        Parameters:
            dt (float): Time step in ps.
            n_steps (int): Number of time steps.
            temperature (float): Temperature in K.
            box_length (float): Length of the (cubic) simulation box in nm.

        Parameters with default values: 
            tau_thermostat (float or None) = None: Thermostat coupling constant in ps
                                                   If None, not thermostat is applied 
            rij_min (float) = 0.0: Lower cutoff for interparticle distances (in nm).
            r_cut: cutoff-radius: float in nm. Max distance which is considered.
        """
        self.dt = dt
        self.n_steps = n_steps
        self.temperature = temperature
        self.box_length = box_length  # in nm
        self.tau_thermostat = tau_thermostat  # thermostat coupling time in ps
        self.rij_min = rij_min        # minimum allowed pairwise distance

        self.r_cut = r_cut            # cutoff radius for LJ interactions

        if self.r_cut is None:
            raise ValueError("r_cut needs to be set!")

        # For periodic boundary conditions, r_cut must be <= L/2.
        # Otherwise, a particle could interact with more than one periodic image
        # of the very same particle, making the cutoff ambiguous. But only the NEAREST
        # periodic image should be considered!
        
        if self.r_cut > 0.5 * self.box_length:
            raise ValueError("r_cut is set larger than half the box length.")

        # Optional: friction coefficient for Langevin or stochastic thermostats
        self.xi = None
        if self.tau_thermostat and self.tau_thermostat > 0.0: 
            self.xi = 1/self.tau_thermostat


#----------------------------------------------------------------
#   F U N C T I O N S
#----------------------------------------------------------------

#--------------------------------------
# Initialization
#--------------------------------------
def initialize_positions(ps: ParticleSystem, box_length_in_nm: float):
    """Initialize particle positions uniformly in a cubic box."""
    ps.position[:] = np.random.uniform(0, box_length_in_nm, size=(ps.n, 3))

def initialize_velocities(ps: ParticleSystem, temperature: float):
    """
    Initializes velocities of a ParticleSystem according to the Maxwell-Boltzmann
    distribution at a given temperature T (in Kelvin), using vectorized NumPy operations.

    Each velocity component is sampled from a Gaussian with:
        variance = sigma^2 = R*T / M
    
    Velocities are returned in units of nm/ps.
    """
    # molar masses in kg/mol (convert from u)
    M = ps.mass * 1e-3  # shape: (n,)
    
    # Compute standard deviations σ = sqrt(RT/M) in m/s
    stddev = np.sqrt(R * temperature / M)  # shape: (n,) 
    
    # Sample velocities: each component independently, shape (n, 3)
    velocities_m_s = np.random.normal(0.0, stddev[:, np.newaxis], size=(ps.n, 3))  # m/s

    # Convert to nm/ps
    velocities_nm_ps = velocities_m_s * 1e-3

    # Set velocities
    ps.velocity[:] = velocities_nm_ps

    # Remove center-of-mass velocity
    v_cm = np.average(ps.velocity, axis=0, weights=ps.mass)
    ps.velocity -= v_cm
    

#--------------------------------------
# nachbarliste def
#--------------------------------------

def update_neighbour_list(
    ps: ParticleSystem,
    sim: SimulationParameters,
    step: int,
    n_update: int = 10
):
    """Update the neighbour list every n_update MD steps."""

    if n_update < 1:
        raise ValueError("n_update must be at least 1")

    first_build = not hasattr(ps, "neighbour_pairs")

    if not first_build and step % n_update != 0:
        return ps.neighbour_pairs

    if first_build:
        ps.neighbour_skin = 0.3 * ps.sigma[0]
        ps.neighbour_pairs = np.empty(
            (0, 2),
            dtype=np.int64
        )
        ps.neighbour_list_step = -1
        ps.neighbour_rebuilds = 0

    box_length = sim.box_length

    list_cutoff = sim.r_cut + ps.neighbour_skin
    list_cutoff_squared = list_cutoff * list_cutoff


    n_cells = max(
        1,
        int(box_length / list_cutoff)
    )

    cell_length = box_length / n_cells

    
    cell_xyz = np.floor(
        ps.position / cell_length
    ).astype(np.int64)

    cell_xyz %= n_cells

    
    cell_id = (
        cell_xyz[:, 0]
        + n_cells
        * (
            cell_xyz[:, 1]
            + n_cells * cell_xyz[:, 2]
        )
    )

    # Store only occupied cells.
    occupied_cells = {}

    for particle_index, current_cell_id in enumerate(cell_id):
        occupied_cells.setdefault(
            int(current_cell_id),
            []
        ).append(particle_index)

    
    cell_offsets = tuple(
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    )

    pair_blocks = []

    for i in range(ps.n):
        cx, cy, cz = cell_xyz[i]

        
        neighbour_cell_ids = set()

        for dx, dy, dz in cell_offsets:
            nx = (cx + dx) % n_cells
            ny = (cy + dy) % n_cells
            nz = (cz + dz) % n_cells

            current_cell_id = (
                nx
                + n_cells
                * (
                    ny
                    + n_cells * nz
                )
            )

            neighbour_cell_ids.add(
                int(current_cell_id)
            )

        candidates = []

        for current_cell_id in neighbour_cell_ids:
            particles = occupied_cells.get(
                current_cell_id
            )

            if particles is not None:
                candidates.extend(particles)

        if not candidates:
            continue

        candidates = np.asarray(
            candidates,
            dtype=np.int64
        )

        
        candidates = candidates[candidates > i]

        if candidates.size == 0:
            continue

        rij = (
            ps.position[i]
            - ps.position[candidates]
        )

        rij -= box_length * np.rint(
            rij / box_length
        )

        r_squared = np.einsum(
            "ij,ij->i",
            rij,
            rij
        )

        neighbours = candidates[
            r_squared <= list_cutoff_squared
        ]

        if neighbours.size > 0:
            pair_blocks.append(
                np.column_stack(
                    (
                        np.full(
                            neighbours.size,
                            i,
                            dtype=np.int64
                        ),
                        neighbours
                    )
                )
            )

    if pair_blocks:
        ps.neighbour_pairs = np.concatenate(
            pair_blocks,
            axis=0
        )
    else:
        ps.neighbour_pairs = np.empty(
            (0, 2),
            dtype=np.int64
        )

    ps.neighbour_list_step = step
    ps.neighbour_rebuilds += 1

    return ps.neighbour_pairs



#--------------------------------------
# Energies
#--------------------------------------

def potential_energy(
    ps: ParticleSystem,
    sim: SimulationParameters
) -> float:
    """Return the energy calculated together with the forces."""

    if not hasattr(ps, "current_potential_energy"):
        raise ValueError(
            "forces must be calculated before potential_energy"
        )

    return float(ps.current_potential_energy)

def kinetic_energy(ps: ParticleSystem) -> float:
    """
   Computes the total kinetic energy of the system in units of kJ/mol.

    Assumes:
    - Mass is in u = 1e-3 g/mol
    - Velocity is in nm/ps = 1e3 m/s

    Returns:
        Kinetic energy in kJ/mol.

    """
    # unit: (1e3 ms/s)^2  = 1e6 m^2/s^2        
    v_squared = np.sum(ps.velocity**2, axis=1)   # shape (N,)    
    # unit: 1e-3 kg/mol * 1e6 m^2/s^2 = 1e3 J/mol = 1 kJ/mol
    return 0.5 * np.sum(ps.mass * v_squared)      

def instantaneous_temperature(ps: ParticleSystem) -> float:
    """
    Computes the instantaneous temperature of the particle system 
    from the total kinetic energy using the equipartition theorem.

    Formula:
        T = (2 * E_kin) / (dof * R)

    Where:
        - E_kin is the total kinetic energy in kJ/mol
        - dof is the number of degrees of freedom
        - R is the gas constant in J/(mol·K)

    Returns:
        Temperature in Kelvin (K).
    """
    # kinetic energy is returned in kJ/mol, convert to J/mol
    E_kin = kinetic_energy(ps)*1e3
    # degrees of freedom: 3 per particle
    dof = ps.n*3
        
    return (2* E_kin) / (dof *R)


def density(ps: ParticleSystem, sim: SimulationParameters) -> float: 
    """
    Computes the density of the system in g/cm^3.

    Assumes:
        - box_length is in nm
        - mass is in atomic mass units (g/mol)

    Returns:
        - Density in g/cm^3
    """
    L_in_nm = sim.box_length
    # nm^3 = 10^{-27} m^3 = 10^{-27} m^3* 1000 L/m^3 = 10^{-24} L
    V_in_cm3 = L_in_nm**3 * 1e-21 
    # Mass is stored in u = g/mol
    # Total mass in g (sum of all molar masses divided by Avogadro)
    m_in_g = np.sum(ps.mass) / Avogadro 

    return m_in_g/V_in_cm3

def ideal_gas_pressure(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """
    Computes the instantaneous ideal gas pressure of the system in Pascals (Pa),
    using the ideal gas law: P = nRT/V.

    Assumes:
    - Positions are in nanometers (nm), volume is converted to m³.
    - Temperature is in Kelvin.
    - Returns pressure in SI units (Pa = J/m^3 = N/m^2).
    """
    L_in_nm = sim.box_length
    V_in_m3 = L_in_nm**3 * 1e-27  # Convert volume to m³
    n_mol = ps.n / Avogadro  # Amount of substance in mol
    T = instantaneous_temperature(ps)  # in Kelvin

    return n_mol * R * T / V_in_m3  # Pressure in Pascals (Pa)
    
#--------------------------------------
# MD integrators
#--------------------------------------

def calculate_force(
    ps: ParticleSystem,
    sim: SimulationParameters
):
    """Calculate LJ forces using the current neighbour list."""

    if not hasattr(ps, "neighbour_pairs"):
        raise ValueError(
            "neighbour list was not created"
        )

    if ps.neighbour_pairs.shape[0] == 0:
        ps.force.fill(0.0)
        ps.current_potential_energy = 0.0
        return None

    i_pairs = ps.neighbour_pairs[:, 0]
    j_pairs = ps.neighbour_pairs[:, 1]

    # Distance vectors for neighbour-list pairs.
    rij = (
        ps.position[i_pairs]
        - ps.position[j_pairs]
    )

    rij -= sim.box_length * np.rint(
        rij / sim.box_length
    )

    # Squared distances avoid unnecessary square roots.
    r_squared = np.einsum(
        "ij,ij->i",
        rij,
        rij
    )

    # Apply the exact cutoff.
    inside_cutoff = (
        r_squared
        <= sim.r_cut * sim.r_cut
    )

    if not np.any(inside_cutoff):
        ps.force.fill(0.0)
        ps.current_potential_energy = 0.0
        return None

    i_pairs = i_pairs[inside_cutoff]
    j_pairs = j_pairs[inside_cutoff]
    rij = rij[inside_cutoff]
    r_squared = r_squared[inside_cutoff]

    minimum_r_squared = max(
        sim.rij_min * sim.rij_min,
        1e-24
    )

    r_squared = np.maximum(
        r_squared,
        minimum_r_squared
    )

    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]

    inverse_r_squared = 1.0 / r_squared

    sigma_over_r_squared = (
        sigma
        * sigma
        * inverse_r_squared
    )

    sr6 = sigma_over_r_squared**3
    sr12 = sr6 * sr6

    # Force on particle i:
    # F_i = 24 epsilon / r^2
    #       * (2 (sigma/r)^12 - (sigma/r)^6)
    #       * r_ij
    force_factor = (
        24.0
        * epsilon
        * inverse_r_squared
        * (2.0 * sr12 - sr6)
    )

    pair_force = (
        force_factor[:, np.newaxis]
        * rij
    )

    # Accumulate the pair forces without a Python loop
    # over all interacting pairs.
    for dimension in range(3):
        force_on_i = np.bincount(
            i_pairs,
            weights=pair_force[:, dimension],
            minlength=ps.n
        )

        force_on_j = np.bincount(
            j_pairs,
            weights=pair_force[:, dimension],
            minlength=ps.n
        )

        ps.force[:, dimension] = (
            force_on_i - force_on_j
        )

    # Calculate and store the potential energy from the
    # already available distance values.
    ps.current_potential_energy = float(
        np.sum(
            4.0
            * epsilon
            * (sr12 - sr6)
        )
    )

    return None

def A_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
    half_step=False
):
    """Update particle positions."""

    dt = (
        0.5 * sim.dt
        if half_step
        else sim.dt
    )

    ps.position += ps.velocity * dt

    return None

def B_step(
    ps: ParticleSystem,
    sim: SimulationParameters,
    half_step=False
):
    """Update particle velocities."""

    dt = (
        0.5 * sim.dt
        if half_step
        else sim.dt
    )

    # Masses do not change during the simulation.
    # Calculate their inverse only once.
    if not hasattr(ps, "inverse_mass"):
        ps.inverse_mass = 1.0 / ps.mass

    ps.velocity += (
        ps.inverse_mass[:, np.newaxis]
        * dt
        * ps.force
    )

    return None

def O_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """
    Performs the O-step (velocity update) in Langevin dynamics.

    The update integrates the effect of the stochastic (random) and friction forces:
        v ← exp(-ξ Δt) * v + sqrt(RT/m * (1 - exp(-2ξΔt))) * η

    Parameters:
        ps (ParticleSystem): Contains velocities, masses, and random number storage.
        sim (SimulationParameters): Contains xi, temperature, dt, and constants.
        half_step (bool): If True, use half the time step Δt / 2.

    Returns:
        None. Updates ps.velocity in-place.
    """

    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * sim.dt
    else:
        dt = sim.dt

    # Draw random numbers from Gaussian normal distribution for stochastic term
    ps.random_number = np.random.normal(size=(ps.n,3))
    
    # dissipation term
    d = np.exp(- sim.xi * dt)

    # fluctuation term
    scalar = sim.temperature * R * (1.0 - np.exp(-2 * sim.xi * dt))
    # mass is stored in units of u ~ g/mol, but needs to be converted to kg/mol
    mass = ps.mass *1e3
    f = np.sqrt(scalar / mass)[:, np.newaxis]  # now shape (N, 1)
    f = np.broadcast_to(f, ps.random_number.shape)  # ensures (N, 3)
 
    ps.velocity = d * ps.velocity + f * ps.random_number 
    
    return None    

def simulate_NVE_step(ps: ParticleSystem, sim: SimulationParameters, step: int, n_update: int):
    B_step(ps, sim, half_step=True)
    A_step(ps, sim, half_step=False)

    apply_periodic_boundary(ps, sim)

    update_neighbour_list(
        ps,
        sim,
        step,
        n_update
    )

    calculate_force(
        ps,
        sim
    )

    B_step(ps, sim, half_step=True)

    return None


def simulate_NVT_step(ps: ParticleSystem, sim: SimulationParameters, step: int, n_update: int):
    if sim.tau_thermostat is None:
        raise ValueError("Thermostat coupling time is not set")

    B_step(ps, sim, half_step=True)
    A_step(ps, sim, half_step=True)
    O_step(ps, sim, half_step=False)
    A_step(ps, sim, half_step=True)

    apply_periodic_boundary(ps, sim)

    update_neighbour_list(
        ps,
        sim,
        step,
        n_update
    )

    calculate_force(
        ps,
        sim
    )

    B_step(ps, sim, half_step=True)

    return None

def apply_periodic_boundary(ps: ParticleSystem, sim: SimulationParameters): 
    """
    Applies periodic boundary conditions to all particle positions.
    Wraps positions into the interval (-L/2, L/2] using centered PBC.
    """
    L = sim.box_length
    # modulus
    # x < L: x/L = -1*L + remainder => return remainder => shifts x by L to the right
    # x in[ 0, L[ : x/L = 0*L + remainder => return remainder => leaves x where it is
    # x >= L : x/L = 1*L + remainder => return remainder => shifts x by L to the left
    ps.position = np.mod(ps.position, L)
    

#--------------------------------------
# Output
#--------------------------------------
def write_xyz_trajectory(filename, trajectory, atom_symbol="Ar"):
    """
    Writes a trajectory to an .xyz file.

    Parameters:
        filename (str): Name of the output .xyz file.
        trajectory (np.ndarray): Array of shape (n_frames, n_particles, 3)
                                 containing atomic positions.
        atom_symbol (str): Element symbol to use for all atoms (default: "Ar").

    Returns:
        None. Writes file to disk.
    """
    
    trajectory = 10.0 * trajectory  # convert nm to Å
    n_frames, n_atoms, _ = trajectory.shape

    with open(filename, "w") as f:
        for frame in trajectory:
            f.write(f"{n_atoms}\n")
            f.write("Generated by write_xyz_trajectory\n")
            for pos in frame:
                f.write(f"{atom_symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")


#--------------------------------------
# Statistics
#--------------------------------------
def cutoff_pair_statistics(
    ps: ParticleSystem,
    sim: SimulationParameters
):
    """Count total pairs and current pairs inside the cutoff."""

    if sim.r_cut is None:
        raise ValueError(
            "r_cut must be set to calculate cutoff pair statistics."
        )

    if not hasattr(ps, "neighbour_pairs"):
        raise ValueError(
            "neighbour list was not created"
        )

    n_pairs_total = (
        ps.n * (ps.n - 1) // 2
    )

    if ps.neighbour_pairs.shape[0] == 0:
        return n_pairs_total, 0, 0.0

    i_pairs = ps.neighbour_pairs[:, 0]
    j_pairs = ps.neighbour_pairs[:, 1]

    rij = (
        ps.position[i_pairs]
        - ps.position[j_pairs]
    )

    rij -= sim.box_length * np.rint(
        rij / sim.box_length
    )

    r_squared = np.einsum(
        "ij,ij->i",
        rij,
        rij
    )

    n_pairs_cutoff = int(
        np.count_nonzero(
            r_squared
            <= sim.r_cut * sim.r_cut
        )
    )

    if n_pairs_total > 0:
        percent_pairs_cutoff = (
            100.0
            * n_pairs_cutoff
            / n_pairs_total
        )
    else:
        percent_pairs_cutoff = 0.0

    return (
        n_pairs_total,
        n_pairs_cutoff,
        percent_pairs_cutoff
    )