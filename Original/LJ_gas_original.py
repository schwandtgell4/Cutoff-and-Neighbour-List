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
    def __init__(self, dt, n_steps, temperature, box_length, tau_thermostat = None, rij_min=0.0, ):
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
        """
        self.dt = dt
        self.n_steps = n_steps
        self.temperature = temperature
        self.box_length = box_length  # in nm
        self.tau_thermostat = tau_thermostat  # thermostat coupling time in ps
        self.rij_min = rij_min        # minimum allowed pairwise distance

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
# Energies
#--------------------------------------

def potential_energy(ps: ParticleSystem, sim: SimulationParameters) -> float:
    """
    Computes the total Lennard-Jones potential energy of the system.
    
    Assumes uniform Lennard-Jones parameters:
        epsilon and sigma (taken from particle 0)
    
    Units:
        Energy is in the same units as epsilon (kJ/mol).
        Positions must be in the same units as sigma (nm).
    """
    n_particles = ps.n
    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]
    L = sim.box_length
        
    # vectorized code to calculate the pairwise distances
    # positions[:, np.newaxis, :] has shape (N, 1, 3)
    # positions[np.newaxis, :, :] has shape (1, N, 3)
    # The difference broadcasted has shape (N, N, 3)
    rij_matrix = ps.position[:, np.newaxis, :] - ps.position[np.newaxis, :, :]
    
    # apply periodic boundary conditions
    rij_matrix -= L * np.rint(rij_matrix / L)
    
    # Pairwise distances (shape: N, N)
    r_matrix = np.linalg.norm(rij_matrix, axis=-1)  

    # Extract upper triangle indices (i < j), i.e. the list of unique pairs
    i_upper = np.triu_indices(n_particles, k=1)
    
    # Get list of unique distance vectors and unique distances
    r = r_matrix[i_upper]                           # shape (N_pairs,)

    # reset the very small distance to 0.00001 nm to make sure
    # that the sr6**2 term is numerically stable
    r = np.clip(r, sim.rij_min, None)
    
    # Compute Lennard-Jones potential for each unique pair
    sr6 = (sigma / r)**6
    lj_pairwise = 4 * epsilon * (sr6**2 - sr6)

    # Total potential energy
    E_pot = np.sum(lj_pairwise)
    
    return E_pot

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

def calculate_force(ps: ParticleSystem, sim: SimulationParameters):
    """
    Computes and assigns Lennard-Jones forces between all unique particle pairs.

    Assumes:
        - Pairwise interactions use identical sigma and epsilon values.
        - Positions are in units compatible with sigma (e.g. nm).
        - Returns no value; updates ps.force in-place (shape: (N, 3)).
    """
    
    n_particles = ps.n
    sigma = ps.sigma[0]
    epsilon = ps.epsilon[0]
    L = sim.box_length


    # vectorized code to calculate the pairwise distances
    # positions[:, np.newaxis, :] has shape (N, 1, 3)
    # positions[np.newaxis, :, :] has shape (1, N, 3)
    # The difference broadcasted has shape (N, N, 3)
    rij_matrix = ps.position[:, np.newaxis, :] - ps.position[np.newaxis, :, :]
    
    # apply periodic boundary conditions
    rij_matrix -= L * np.rint(rij_matrix / L)
    
    # Pairwise distances (shape: N, N)
    r_matrix = np.linalg.norm(rij_matrix, axis=-1)  

    # Extract upper triangle indices (i < j), i.e. the list of unique pairs
    i_upper = np.triu_indices(n_particles, k=1)
    
    # Get list of unique distance vectors and unique distances
    rij = rij_matrix[i_upper]                       # shape (N_pairs, 3)    
    r = r_matrix[i_upper]                           # shape (N_pairs,)
    
    # reset distances < rij_min  to rij_min to make sure
    # that the sr6**2 term is numerically stable
    r = np.clip(r, sim.rij_min, None)
    
    # Normalize rij to unit vectors and rescale to match clipped r
    rij = rij / np.linalg.norm(rij, axis=1)[:, np.newaxis]  # normalize each rij
    rij *= r[:, np.newaxis]                                 # rescale to clipped r

    # Lennard-Jones force magnitude: dV/dr
    sr6 = (sigma / r)**6                            # shape (N_pairs,)
    dV_dr = 24 * epsilon / r * (-2 * sr6**2 + sr6)  # shape (N_pairs,)

    # Force vectors: shape (N_pairs, 3)
    # dV_dr[:, np.newaxis] shapes it to (N_pairs, 1), i.e. 2D column vector
    # broadcasting to rij with shape (N_pairs, 3) is then possible
    f_ij = (dV_dr[:, np.newaxis] / r[:, np.newaxis]) * rij

    # Initialize total force array
    force = np.zeros_like(ps.position)  # shape (N, 3)

    # Distribute pairwise forces to particle i and j
    for idx, (i, j) in enumerate(zip(i_upper[0], i_upper[1])):
        force[i] -= f_ij[idx]
        force[j] += f_ij[idx]

    # update the force vector in the ParticleSystem class
    ps.force = force

def A_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """
    Performs the A-step (position update) of an MD integration scheme.

    This step updates particle positions using the current velocities:
    r(t + Δt) = r(t) + v(t) * Δt

    Parameters:
        - ps (ParticleSystem): The particle system containing positions and velocities.
        - sim (SimulationParameters): Simulation settings, including the time step.
        - half_step (bool): If True, performs a half step (Δt / 2) instead of a full step.

    Returns:
        None. Updates ps.position in-place.
    """
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * sim.dt
    else:
        dt = sim.dt
        
    ps.position = ps.position + ps.velocity * dt
    
    return None    

def B_step(ps: ParticleSystem, sim: SimulationParameters, half_step=False):
    """
    Performs the B-step (velocity update) of an MD integration scheme.

    This step updates particle velocities using the current forces:
    v(t + Δt) = v(t) + 1/m * Δt * F(t) 
 
    Parameters:
        - ps (ParticleSystem): The particle system containing positions and velocities.
        - sim (SimulationParameters): Simulation settings, including the time step.
        - half_step (bool): If True, performs a half step (Δt / 2) instead of a full step.

    Returns:
        None. Updates ps.velocity in-place.
    """
    # set time step, depending on whether a half- or full step is performed
    if half_step == True:
        dt = 0.5 * sim.dt
    else:
        dt = sim.dt
        
    # (1/ps.mass)[:, np.newaxis] = explicit reshaping to avoid
    # broadcasting issues when multiplying (N,) with (N,3) elementwise
    # now it is explicit: (N,1) * (N,3)
    ps.velocity = ps.velocity + (1/ps.mass)[:, np.newaxis]* dt * ps.force 
    
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

def simulate_NVE_step(ps: ParticleSystem, sim: SimulationParameters):
    """
    Performs a single time step of molecular dynamics in the NVE ensemble
    using the velocity Verlet integrator in BAB form (half-step B, full-step A, half-step B).

    The steps are:
    1. Half-step velocity update (B-step)
    2. Full-step position update (A-step)
    3. Force recalculation based on new positions
    4. Second half-step velocity update (B-step)
    5. Apply periodic boundary conditions

    This corresponds to a time-symmetric, second-order accurate integrator for Newtonian dynamics.

    Parameters:
        - ps (ParticleSystem): The particle system containing positions, velocities, and forces.
        - sim (SimulationParameters): Simulation parameters including time step.

    Returns:
        None. Updates ps.position, ps.velocity, and ps.force in-place.
    """
    B_step(ps, sim, half_step=True)   # update velocity by a half-step
    A_step(ps, sim, half_step=False)  # update position by a full time step
    calculate_force(ps, sim)          # udpate force  
    B_step(ps, sim, half_step=True)   # update velocity by a second half-step

    apply_periodic_boundary(ps, sim)
        
    return None      

def simulate_NVT_step(ps: ParticleSystem, sim: SimulationParameters):
    """
    Performs a single time step of molecular dynamics in the NVT ensemble
    using the BAOAB Langevin integrator.

    The steps are:
    1. Half-step velocity update from force (B)
    2. Half-step position update (A)
    3. Full-step velocity update via Langevin thermostat (O)
    4. Second half-step position update (A)
    5. Force recalculation
    6. Second half-step velocity update from force (B)
    7. Apply periodic boundary conditions

    Parameters:
        ps (ParticleSystem): Particle data including velocity, position, mass, etc.
        sim (SimulationParameters): Simulation control parameters.

    Returns:
        None
    """
    
    if sim.tau_thermostat is None:
        raise ValueError("Thermostat coupling time (tau_thermostat) is not set. Cannot run NVT simulation.")
    
    B_step(ps, sim, half_step=True)   # update velocity by a half-step
    A_step(ps, sim, half_step=True)  # update position by a half-step
    # thermostat
    O_step(ps, sim, half_step=False)  # Full-step velocity update using the Langevin thermostat (friction + noise)
    A_step(ps, sim, half_step=True)  # update position by a half-step
    calculate_force(ps, sim)          # udpate force  
    B_step(ps, sim, half_step=True)   # update velocity by a second half-step

    apply_periodic_boundary(ps, sim)
        
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
 